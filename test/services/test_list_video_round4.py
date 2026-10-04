import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript
from app.services import gemini_media, list_video, llm, web_images
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_host as host
from app.services import list_video_scenes as scenes
from app.utils import utils

RESOURCES = Path(__file__).parent.parent / "resources"
FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NAMES = ["explicando", "neutral", "feliz", "pensando", "saludando", "senalando", "sorprendido", "preocupado"]


def _icon(path, color=(220, 60, 60, 255), size=120):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, size - 10, size - 10), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


def _segments(text, names=("Ojos",)):
    return list_video.build_segments(
        ListVideoScript(
            title="T", intro="Hola nutrias.", items=[ListVideoItem(name=n, text=text) for n in names], outro="Chao."
        )
    )


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class TestChapterChip(unittest.TestCase):
    CHIP = fx.Overlay("chip.png", x="20-(300+44)*(1-(1-pow(1-clip((t-0.120)/0.450,0,1),3)))", y="20")

    def test_chip_hides_while_scenes_fill_the_screen(self):
        spans = editor.chip_spans(self.CHIP, [(4.0, 9.0), (14.0, 18.0)], 25.0, 344)
        self.assertEqual([(round(o.start, 2), round(o.end, 2)) for o in spans], [(0.0, 3.95), (9.05, 13.95), (18.05, 25.0)])
        first, middle, last = spans
        self.assertTrue(first.x.startswith(self.CHIP.x))  # the first entrance keeps its own slide
        self.assertIn("clip((t-3.650)/0.3,0,1)", first.x)  # and it slides out before the scene
        self.assertTrue(middle.x.startswith("20-344*(1-"))  # slides back in after the scene
        self.assertTrue(last.x.endswith("-0"))  # stays until the cut
        self.assertTrue(all(o.y == "20" and o.source == "chip.png" for o in spans))

    def test_short_gaps_and_scenes_at_the_edges(self):
        spans = editor.chip_spans(self.CHIP, [(0.0, 5.0), (5.8, 9.0)], 12.0, 344)
        self.assertEqual([(round(o.start, 2), round(o.end, 2)) for o in spans], [(9.05, 12.0)])
        self.assertEqual(editor.chip_spans(self.CHIP, [(0.0, 12.0)], 12.0, 344), [])
        alone = editor.chip_spans(self.CHIP, [], 12.0, 344)
        self.assertEqual(len(alone), 1)
        self.assertEqual((alone[0].start, alone[0].end), (0.0, 12.0))


class TestHostPresence(unittest.TestCase):
    def _share(self, presence):
        infos = [host.SegmentInfo(kind="intro", duration=6.0, expression="feliz")]
        infos += [host.SegmentInfo(kind="item", duration=24.0, expression="explicando") for _ in range(6)]
        infos += [host.SegmentInfo(kind="outro", duration=6.0, expression="feliz")]
        plan = host.plan_host(infos, NAMES, presence=presence)
        items = plan[1:-1]
        return sum(w.end - w.start for p in items for w in p.windows) / (24.0 * 6)

    def test_less_host_by_default(self):
        low, normal, high = self._share("low"), self._share("normal"), self._share("high")
        self.assertLess(low, normal)
        self.assertLess(normal, high)
        self.assertLess(low, 0.3)
        self.assertEqual(host.HOST_PRESENCES, ("low", "normal", "high"))
        # Intro and outro always keep the host, and unknown values behave like "normal".
        self.assertEqual(host.segment_mode("intro", 0, 0, "low"), "full")
        self.assertEqual(host.segment_mode("item", 2, 0, "nope"), host.ITEM_PATTERN[2])

    def test_lead_is_shorter_with_low_presence(self):
        info = host.SegmentInfo(kind="item", duration=20.0, expression="explicando", mode="lead")
        low = host.plan_host([info], NAMES, presence="low")[0].windows[0]
        normal = host.plan_host([info], NAMES, presence="normal")[0].windows[0]
        self.assertLess(low.end, normal.end)


class TestSoundVolume(_TempDirCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def _edit(self, **options):
        assets = os.path.join(self.temp_dir, "assets")
        os.makedirs(os.path.join(assets, "sfx"), exist_ok=True)
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        options.setdefault("host", "none")
        with patch.object(editor.gemini_media, "enabled", return_value=False):
            ed = editor.Editor(editor.EditOptions(**options), theme, self.temp_dir, _segments(self.TEXT), narrations)
        ed.sfx = {"whoosh": "w.wav", "pop": "p.wav", "click": "c.wav"}
        ed.plan = [
            {"index": 0, "expression": "", "scenes": [], "beats": []},
            {"index": 1, "expression": "", "scenes": [], "beats": []},
            {"index": 2, "expression": "", "scenes": [], "beats": []},
        ]
        return ed.segment_edit(1, 1.5, show_titles=True)

    def test_effects_are_quieter_by_default_and_can_be_muted(self):
        default = self._edit()
        louder = self._edit(sfx_volume=1.0)
        muted = self._edit(sfx_volume=0.0)
        self.assertTrue(default.sounds)
        for (_, _, quiet), (_, _, loud) in zip(default.sounds, louder.sounds):
            self.assertAlmostEqual(quiet, loud * 0.65)
        self.assertEqual(muted.sounds, [])
        chips = [o for o in default.overlays if os.path.basename(o.source).startswith("chip-")]
        self.assertEqual(len(chips), 1)


class TestOpenerTiming(unittest.TestCase):
    def test_opener_scene(self):
        spec = {"query": "voltage current diagram", "query_local": "", "look": "diagram", "icon": "⚡", "draw": "a battery"}
        opener = scenes.opener_scene("Voltaje y corriente", 2, spec, [3.4, 8.0], 20.0)
        self.assertEqual((opener.type, opener.start, opener.number, opener.enter), ("opener", 0.0, 2, False))
        self.assertAlmostEqual(opener.end, 3.65)  # snapped to the breath
        self.assertEqual((opener.query, opener.center.icon, opener.center.draw), ("voltage current diagram", "⚡", "a battery"))
        # Without a plan the title itself is searched.
        plain = scenes.opener_scene("La carga", 1, None, [], 20.0)
        self.assertEqual((plain.query, plain.query_local, plain.center.query), ("La carga", "La carga", "La carga"))
        self.assertIsNone(scenes.opener_scene("Corto", 1, None, [], 6.0))
        self.assertIsNone(scenes.opener_scene("", 1, None, [], 20.0))

    def test_scenes_said_under_the_opener_wait_for_it(self):
        times = {"uno": 1.0, "dos": 2.0, "tres": 6.0}

        def locate(anchor):
            return times.get(anchor)

        specs = [{"type": "sequence", "items": [{"at": "uno", "label": "A"}, {"at": "dos", "label": "B"}, {"at": "tres", "label": "C"}]}]
        timed = scenes.time_scenes(specs, locate, [], 20.0, blocked=[(0.0, 3.6)])
        self.assertEqual(len(timed), 1)
        self.assertAlmostEqual(timed[0].start, 3.75)
        item_times = [i.time for i in timed[0].items]
        self.assertEqual(item_times, sorted(item_times))
        self.assertGreaterEqual(item_times[0], 3.75 + scenes.SLIDE_SECONDS)
        self.assertGreater(item_times[1] - item_times[0], 0.4)  # still one after the other
        # Too long a wait drops it.
        self.assertEqual(scenes.time_scenes(specs, locate, [], 20.0, blocked=[(0.0, 9.5)]), [])

    def test_delayed_scenes_keep_their_length_and_can_follow_each_other(self):
        times = {"voltaje": 1.4, "corriente": 7.4}
        specs = [
            {"type": "definition", "at": "voltaje", "term": "voltaje", "text": "el empuje"},
            {"type": "definition", "at": "corriente", "term": "corriente", "text": "el flujo"},
        ]
        for pauses in ([], [4.0, 7.2, 12.0, 17.5]):
            first, second = scenes.time_scenes(specs, times.get, pauses, 25.0, blocked=[(0.0, 3.75)])
            self.assertAlmostEqual(first.start, 3.9)
            self.assertGreaterEqual(first.end - first.start, scenes.DELAYED_SECONDS - 1e-6)
            self.assertAlmostEqual(second.start, first.end + 0.6)
        # A scene that would wait too long for the previous one is dropped.
        times["corriente"] = 2.0
        self.assertEqual(len(scenes.time_scenes(specs, times.get, [], 25.0)), 1)

    def test_opener_drawing(self):
        with tempfile.TemporaryDirectory() as folder:
            theme = fx.Theme(640, 360, FONT, (255, 79, 94))
            icon = scenes.prepare_picture(_icon(os.path.join(folder, "i.png")))
            renderer = scenes.SceneRenderer(theme, folder, (250, 230, 230), lambda item: icon if item.icon else None)
            scene = scenes.opener_scene("Voltaje", 2, {"query": "q", "icon": "⚡"}, [], 20.0)
            overlays, sounds = renderer.build(scene, "o")
            names = [os.path.basename(o.source) for o in overlays]
            self.assertTrue(any(n.startswith("number_") for n in names))
            self.assertTrue(any(n.startswith("picture_") for n in names))
            self.assertTrue(overlays[0].x.startswith("0-W*pow(clip("))  # no slide in, only out
            self.assertNotIn("pow(1-clip", overlays[0].x)
            self.assertNotIn(-0.1, [t for t, _, _ in sounds])
            bare = scenes.opener_scene("Sin imagen", 0, None, [], 20.0)
            bare.center = scenes.SceneItem()
            names = [os.path.basename(o.source) for o in renderer.build(bare, "b")[0]]
            self.assertFalse(any(n.startswith(("number_", "picture_")) for n in names))


class TestOpenerPictures(_TempDirCase):
    TEXT = "El voltaje empuja. " + " ".join(f"palabra{i}" for i in range(60))

    def _editor(self, gemini=True, **options):
        patcher = patch.object(editor.gemini_media, "enabled", return_value=gemini)
        patcher.start()
        self.addCleanup(patcher.stop)
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        ed = editor.Editor(editor.EditOptions(**options), theme, self.temp_dir, _segments(self.TEXT, ("Voltaje",)), narrations)
        ed.plan = [
            {"index": 0, "expression": "", "scenes": [], "beats": []},
            {"index": 1, "expression": "", "scenes": [], "beats": [
                {"type": "image", "at": "El voltaje empuja", "query": "early picture"},
                {"type": "text", "at": "palabra1", "text": "too early"},
            ], "opener": {"query": "voltage diagram", "query_local": "diagrama voltaje", "look": "diagram", "icon": "⚡", "draw": "a battery"}},
            {"index": 2, "expression": "", "scenes": [], "beats": []},
        ]
        return ed

    def _found(self, n):
        return web_images.WebImage(str(RESOURCES / f"{n}.png"), "pexels", f"P{n}", "A", "L", "https://p", url=f"https://u/{n}")

    def test_checked_web_picture_strictly_about_the_title(self):
        ed = self._editor()
        queries = []

        def candidates(query, save_dir, kind="diagram", exclude_urls=None, limit=4):
            queries.append(query)
            return [self._found(1), self._found(2)]

        with patch.object(editor.web_images, "find_candidates", side_effect=candidates), patch.object(
            editor.gemini_media, "choose_picture", return_value=1
        ) as choose, patch.object(editor.gemini_media, "illustrate") as draw:
            edit = ed.segment_edit(1, 1.5, show_titles=True)
        opener = ed._scenes[1][0]
        self.assertEqual(opener.type, "opener")
        self.assertIn("diagrama voltaje", queries)
        self.assertIn("voltage diagram", queries)
        self.assertNotIn("Voltaje", queries)
        call = next(c for c in choose.call_args_list if c.kwargs.get("purpose") == "opener")
        self.assertTrue(call.args[1].startswith("Voltaje. El voltaje empuja."))
        draw.assert_not_called()
        self.assertIsNotNone(ed._scene_picture(opener.center))
        self.assertIn("https://u/2", ed._used_urls)
        # The beat said under the opener waits for it; the one that would be too short is dropped.
        beats = [b for b, _ in ed._beats[1]]
        self.assertTrue(all(b.start >= opener.end for b in beats))
        # The chapter label comes in after the opener.
        chips = [o for o in edit.overlays if os.path.basename(o.source).startswith("chip-")]
        self.assertGreaterEqual(min(o.start for o in chips), opener.end)
        self.assertTrue(any("work" in o.source or "scenes" in o.source for o in edit.overlays))

    def test_falls_back_to_a_drawing_then_the_icon(self):
        ed = self._editor()
        drawing = _icon(self.path("drawing.png"), color=(30, 160, 90, 255))
        with patch.object(editor.web_images, "find_candidates", return_value=[self._found(1)]), patch.object(
            editor.gemini_media, "choose_picture", return_value=-1
        ), patch.object(editor.gemini_media, "illustrate", return_value=drawing) as draw:
            ed._prepare()
        draw.assert_called_once_with("a battery")
        self.assertIsNotNone(ed._scene_picture(ed._scenes[1][0].center))

        offline = self._editor(gemini=False)
        icon = _icon(self.path("icon.png"))
        with patch.object(editor.web_images, "find_candidates") as find, patch.object(editor.icons, "fetch", return_value=icon):
            offline._prepare()
        self.assertNotIn("voltage diagram", [c.args[0] for c in find.call_args_list])
        self.assertIsNotNone(offline._scene_picture(offline._scenes[1][0].center))

    def test_openers_can_be_turned_off(self):
        ed = self._editor(gemini=False, openers=False)
        with patch.object(editor.icons, "fetch", return_value=""):
            ed._prepare()
        self.assertFalse(any(s.type == "opener" for s in ed._scenes[1]))


class TestOpenerPlan(unittest.TestCase):
    def test_normalized_opener(self):
        data = {"segments": [
            {"index": 0, "opener": {"query": "x"}},
            {"index": 1, "opener": {"query": "voltage diagram", "query_local": "diagrama", "look": "photo", "icon": "⚡", "draw": "a battery"}},
            {"index": 2, "opener": {"icon": "⚡"}},
        ]}
        plan = llm.normalize_edit_plan(data, 3, [])
        self.assertEqual(plan[1]["opener"], {"query": "voltage diagram", "query_local": "diagrama", "look": "photo", "icon": "⚡", "draw": "a battery"})
        self.assertEqual(plan[0]["opener"]["look"], "diagram")
        self.assertNotIn("opener", plan[2])

    def test_choice_prompt_for_openers(self):
        prompt = gemini_media.build_choice_prompt("Voltaje y corriente. El voltaje empuja.", "voltage diagram", 3, "es", "opener")
        self.assertIn("opens a new section", prompt)
        self.assertIn("strictly about the title", prompt)
        self.assertNotIn("The narrator says", prompt)
        self.assertIn("opener", gemini_media.PURPOSES)


class TestScientificScenes(_TempDirCase):
    SPECS = [
        {"type": "definition", "at": "uno", "term": "voltaje", "text": "el empuje", "symbol": "V", "unit": "se mide en voltios (V)", "icon": "🔋"},
        {"type": "equation", "at": "cinco", "name": "Ley de Ohm", "formula": "V = I × R", "example": "12 V = 2 A × 6 Ω", "terms": [
            {"symbol": "V", "label": "voltaje", "unit": "voltios"}, {"symbol": "I", "label": "corriente", "unit": "amperios", "at": "corriente"},
            {"symbol": "R", "label": "resistencia", "unit": "ohmios"},
        ]},
        {"type": "annotate", "at": "corazon", "query": "heart conduction", "labels": [{"label": "nodo sinusal"}, {"label": "nodo AV", "at": "nodo"}]},
        {"type": "chain", "items": [{"at": "turbina", "label": "turbina", "icon": "🌀", "link": "gira"}, {"at": "casa", "label": "casa", "icon": "🏠"}]},
        {"type": "branch", "center": {"label": "corriente", "icon": "⚡"}, "items": [{"at": "luz", "label": "luz", "icon": "💡"}, {"at": "calor", "label": "calor", "icon": "🔥"}]},
    ]
    TIMES = {"uno": 1.0, "cinco": 9.0, "corriente": 12.5, "corazon": 20.0, "nodo": 22.6, "turbina": 30.0, "casa": 31.5, "luz": 40.0, "calor": 41.0}

    def test_timing_of_parts(self):
        timed = scenes.time_scenes(self.SPECS, self.TIMES.get, [], 60.0)
        self.assertEqual([s.type for s in timed], ["definition", "equation", "annotate", "chain", "branch"])
        definition, equation, annotate, chain, branch = timed
        self.assertEqual((definition.center.label, definition.symbol, definition.unit), ("voltaje", "V", "se mide en voltios (V)"))
        self.assertGreaterEqual(definition.end - definition.start, 5.0)  # time to read it
        self.assertEqual((equation.text, equation.center.label, equation.example), ("V = I × R", "Ley de Ohm", "12 V = 2 A × 6 Ω"))
        times = [t.time for t in equation.items]
        self.assertAlmostEqual(times[0], 9.0 + scenes.PART_FIRST["equation"])
        self.assertAlmostEqual(times[1], 12.5)  # explained at its own words
        self.assertAlmostEqual(times[2], 12.5 + scenes.PART_STEP)
        self.assertGreater(equation.end, times[2] + 3.0)  # the example has time to show
        self.assertEqual([i.label for i in annotate.items], ["nodo sinusal", "nodo AV"])
        self.assertAlmostEqual(annotate.items[1].time, 22.6)
        self.assertEqual(annotate.query, "heart conduction")
        self.assertEqual(chain.items[0].link, "gira")
        self.assertEqual(branch.center.label, "corriente")

    def test_every_new_type_draws(self):
        for portrait in (False, True):
            theme = fx.Theme(360, 640, FONT, (255, 79, 94)) if portrait else fx.Theme(640, 360, FONT, (255, 79, 94))
            icon = scenes.prepare_picture(_icon(self.path("icon.png")))
            photo = Image.new("RGB", (300, 200), (200, 80, 80))
            photo.save(self.path("photo.jpg"))
            framed = scenes.prepare_picture(self.path("photo.jpg"), allow_cutout=False)

            def picture(item):
                return framed if item.query == "heart conduction" else icon

            renderer = scenes.SceneRenderer(theme, self.path(f"r{portrait}"), (250, 230, 230), picture, font_path=scenes.hand_font_path())
            timed = scenes.time_scenes(self.SPECS, self.TIMES.get, [], 60.0)
            timed[2].items[0].point = (0.2, 0.3)
            for number, scene in enumerate(timed):
                overlays, sounds = renderer.build(scene, f"s{number}")
                names = " ".join(os.path.basename(o.source) for o in overlays)
                self.assertTrue(overlays, scene.type)
                if scene.type == "equation":
                    self.assertEqual(names.count("token"), 5)  # V = I × R
                    self.assertEqual(names.count("pointer"), 3)
                    self.assertIn("example", names)
                if scene.type == "annotate":
                    self.assertIn("dot0", names)
                    self.assertNotIn("dot1", names)  # Gemini did not find it: label only
                if scene.type == "chain":
                    self.assertIn("verb0", names)
                if scene.type == "definition":
                    self.assertIn("unit", names)
            missing = scenes.Scene("annotate", 0, 4, center=scenes.SceneItem(query="nothing"))
            renderer.picture = lambda item: None
            with self.assertRaises(ValueError):
                renderer.build(missing, "missing")


class TestScientificPlan(unittest.TestCase):
    def test_new_scene_types_are_normalized(self):
        data = {"segments": [{"index": 0, "scenes": [
            {"type": "definition", "at": "a", "term": "Voltaje", "text": "el empuje de los electrones", "symbol": "V", "unit": "se mide en voltios (V)", "icon": "🔋"},
            {"type": "equation", "at": "b", "name": "Ley de Ohm", "formula": "V = I × R", "terms": [{"symbol": "V", "label": "voltaje"}, {"symbol": "", "label": "x"}], "example": "12 V = 2 A × 6 Ω"},
            {"type": "annotate", "at": "c", "query": "heart conduction system", "labels": [{"label": "nodo sinusal"}, {"label": "nodo AV", "at": "nodo"}]},
            {"type": "chain", "items": [{"at": "x", "label": "A", "link": "gira"}, {"at": "y", "label": "B"}]},
        ]}, {"index": 1, "scenes": [
            {"type": "branch", "center": {"label": "C", "icon": "⚡"}, "items": [{"at": "x", "label": "A"}, {"at": "y", "label": "B"}]},
            {"type": "definition", "at": "a", "term": "x"},
            {"type": "equation", "at": "b"},
            {"type": "annotate", "at": "c", "query": "q", "labels": [{"label": "solo"}]},
        ]}]}
        plan = llm.normalize_edit_plan(data, 2, [])
        definition, equation, annotate, chain = plan[0]["scenes"]
        self.assertEqual((definition["term"], definition["symbol"], definition["unit"]), ("Voltaje", "V", "se mide en voltios (V)"))
        self.assertEqual(equation["terms"], [{"symbol": "V", "label": "voltaje", "unit": "", "at": ""}])
        self.assertEqual(equation["example"], "12 V = 2 A × 6 Ω")
        self.assertEqual([lab["label"] for lab in annotate["labels"]], ["nodo sinusal", "nodo AV"])
        self.assertEqual(chain["items"][0]["link"], "gira")
        self.assertEqual([s["type"] for s in plan[1]["scenes"]], ["branch"])  # the rest is incomplete
        self.assertEqual(plan[1]["scenes"][0]["center"]["label"], "C")

    def test_prompts_ask_for_science(self):
        prompt = llm.build_edit_plan_prompt([{"index": 0, "kind": "item", "title": "1. Voltaje", "text": "t"}], ["feliz"], "es-CO")
        for word in ('"definition"', '"equation"', '"annotate"', '"chain"', '"branch"', '"opener"', "Footage alone is the last resort", "heart electrical conduction system"):
            self.assertIn(word, prompt)
        script = llm.build_list_script_prompt("Electricidad", 6, "es-CO", 120)
        for word in ("WHY", "measured in volts", "Ohm's law", "worked example", "sinoatrial node"):
            self.assertIn(word, script)
        beat = gemini_media.build_choice_prompt("los electrones fluyen", "electrons flowing", 3, "es", "beat")
        self.assertIn("explanatory diagram", beat)
        self.assertNotIn("explanatory diagram", gemini_media.build_choice_prompt("x", "y", 3, "es", "scene"))
        self.assertIn("parts we will point at", gemini_media.build_choice_prompt("x", "y", 3, "es", "annotate"))


class TestGeminiPoints(unittest.TestCase):
    def test_locate_parts(self):
        answer = json.dumps({"points": [{"label": "Nodo sinusal", "point": [300, 200]}, {"label": "nodo AV", "point": None}, {"label": "x", "point": [1500, -5]}]})
        with patch.object(gemini_media, "_ask", return_value=answer) as ask:
            points = gemini_media.locate_parts("p.png", ["nodo sinusal", "nodo AV", "haz"])
        self.assertEqual(points, [(0.2, 0.3), None, (0.0, 1.0)])
        self.assertIn("0-1000", ask.call_args.args[1])
        with patch.object(gemini_media, "_ask", side_effect=RuntimeError("boom")):
            self.assertEqual(gemini_media.locate_parts("p.png", ["a", "b"]), [None, None])
        self.assertEqual(gemini_media.locate_parts("", ["a"]), [None])


class TestAnnotatePictures(_TempDirCase):
    TEXT = "El corazon late. " + " ".join(f"palabra{i}" for i in range(60))

    def _editor(self, gemini=True):
        patcher = patch.object(editor.gemini_media, "enabled", return_value=gemini)
        patcher.start()
        self.addCleanup(patcher.stop)
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        ed = editor.Editor(editor.EditOptions(subscribe="none", openers=False), theme, self.temp_dir, _segments(self.TEXT), narrations)
        ed.plan = [
            {"index": 0, "expression": "", "scenes": [], "beats": []},
            {"index": 1, "expression": "", "scenes": [
                {"type": "annotate", "at": "palabra10", "query": "heart conduction system", "query_local": "sistema de conduccion", "labels": [{"label": "nodo sinusal"}, {"label": "nodo AV"}]},
            ], "beats": []},
            {"index": 2, "expression": "", "scenes": [], "beats": []},
        ]
        return ed

    def test_checked_picture_and_points(self):
        ed = self._editor()
        found = web_images.WebImage(str(RESOURCES / "1.png"), "pexels", "P", "A", "L", "https://p", url="https://u/1")
        with patch.object(editor.web_images, "find_candidates", return_value=[found]) as find, patch.object(
            editor.gemini_media, "choose_picture", return_value=0
        ) as choose, patch.object(editor.gemini_media, "locate_parts", return_value=[(0.5, 0.5), None]):
            ed.segment_edit(1, 1.5, show_titles=False)
        self.assertEqual([c.args[0] for c in find.call_args_list], ["sistema de conduccion", "heart conduction system"])
        self.assertEqual(choose.call_args.kwargs["purpose"], "annotate")
        self.assertIn("nodo sinusal, nodo AV", choose.call_args.args[2])
        scene = ed._scenes[1][0]
        self.assertEqual([i.point for i in scene.items], [(0.5, 0.5), None])
        self.assertTrue(ed._scene_picture(scene.center).info.get("framed"))

    def test_needs_gemini_and_a_passing_picture(self):
        ed = self._editor(gemini=False)
        with patch.object(editor.web_images, "find_candidates") as find:
            ed._prepare()
        find.assert_not_called()
        self.assertEqual(ed._scenes[1], [])
        rejected = self._editor()
        found = web_images.WebImage(str(RESOURCES / "1.png"), "pexels", "P", "A", "L", "https://p", url="https://u/1")
        with patch.object(editor.web_images, "find_candidates", return_value=[found]), patch.object(
            editor.gemini_media, "choose_picture", return_value=-1
        ), patch.object(editor.gemini_media, "locate_parts") as locate:
            rejected._prepare()
        locate.assert_not_called()
        self.assertEqual(rejected._scenes[1], [])


class TestGapFilling(_TempDirCase):
    TEXT = (
        "La corriente es el flujo de electrones. Los electrones salen del polo negativo de la pila y viajan por el cable. "
        "En el camino se encuentran con la resistencia del filamento del bombillo. Por eso el filamento se calienta y brilla con fuerza. "
        "Corto."
    )

    def test_uncovered_sentences(self):
        entry = {"beats": [{"type": "image", "at": "viajan por el cable", "query": "q"}], "scenes": [
            {"type": "chain", "items": [{"at": "la resistencia del filamento"}]},
        ]}
        self.assertEqual(editor.uncovered_sentences(self.TEXT, entry, "item"), [])
        bare = {"beats": [], "scenes": []}
        gaps = editor.uncovered_sentences(self.TEXT, bare, "item")
        self.assertEqual(len(gaps), 3)  # the first is the opener's and "Corto." is too short
        self.assertTrue(gaps[0].startswith("Los electrones"))
        self.assertEqual(len(editor.uncovered_sentences(self.TEXT, bare, "intro")), 4)

    def test_second_pass_adds_pictures(self):
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        ed = editor.Editor(editor.EditOptions(subscribe="none"), theme, self.temp_dir, _segments(self.TEXT), narrations)
        plan = [{"index": i, "expression": "", "backgrounds": [], "scenes": [], "beats": []} for i in range(3)]
        reply = json.dumps({"beats": [
            {"index": 1, "at": "viajan por el cable", "query": "electrons flowing through a wire diagram", "look": "diagram", "icon": "⚡"},
            {"index": 7, "at": "x", "query": "nope"},
            {"index": 1, "at": "", "query": "no anchor"},
        ]})
        with patch.object(editor.llm, "generate_edit_plan", return_value=plan), patch.object(llm, "_generate_response", return_value=reply) as ask:
            result = ed.make_plan()
        self.assertIn("electrons flowing", ask.call_args.args[0] if ask.call_args.args else "")
        self.assertEqual(result[1]["beats"], [{"type": "image", "at": "viajan por el cable", "query": "electrons flowing through a wire diagram", "look": "diagram", "icon": "⚡"}])
        prompt = llm.build_gap_beats_prompt([{"index": 1, "sentence": "Los electrones viajan."}], "es-CO")
        self.assertIn("Los electrones viajan.", prompt)

        off = editor.Editor(editor.EditOptions(subscribe="none", fill_gaps=False), theme, self.path("off"), _segments(self.TEXT), narrations)
        with patch.object(editor.llm, "generate_edit_plan", return_value=[dict(e, beats=[]) for e in plan]), patch.object(llm, "generate_gap_beats") as gap:
            off.make_plan()
        gap.assert_not_called()
        self.assertEqual(llm.generate_gap_beats([]), {})
        with patch.object(llm, "_generate_response", return_value="Error: nope"):
            self.assertEqual(llm.generate_gap_beats([{"index": 0, "sentence": "s"}]), {})


class TestRound4Cli(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.script = os.path.join(self.temp_dir, "s.json")
        Path(self.script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")

    def _edit(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, redirect_stdout(stdout), redirect_stderr(stderr):
            list_video_cli.run(["--script", self.script, *args])
        return generate.call_args.kwargs["edit"]

    def test_presence_and_volume_flags(self):
        default = self._edit()
        self.assertEqual((default.host_presence, default.sfx_volume), ("low", 0.65))
        custom = self._edit("--host-presence", "high", "--sfx-volume", "5")
        self.assertEqual((custom.host_presence, custom.sfx_volume), ("high", 2.0))
        self.assertTrue(default.openers)
        self.assertFalse(self._edit("--no-openers").openers)


if __name__ == "__main__":
    unittest.main()
