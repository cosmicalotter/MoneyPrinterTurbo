import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript, VideoParams
from app.services import gemini_media, icons, list_video, llm, web_images
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_scenes as scenes
from app.utils import utils

RESOURCES = Path(__file__).parent.parent / "resources"
FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))
INDEX = [
    {"emoji": "🕯️", "hexcode": "1F56F", "group": "objects", "annotation": "candle", "tags": "light", "openmoji_tags": "fire"},
    {"emoji": "💪", "hexcode": "1F4AA", "group": "people-body", "annotation": "flexed biceps", "tags": "muscle, strong", "openmoji_tags": ""},
    {"emoji": "", "hexcode": "E313", "group": "extras-openmoji", "annotation": "stomach", "tags": "", "openmoji_tags": "organ"},
    {"emoji": "🇨🇴", "hexcode": "1F1E8-1F1F4", "group": "flags", "annotation": "flag: Colombia", "tags": "flag", "openmoji_tags": ""},
]


def _icon(path, color=(220, 60, 60, 255)):
    image = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, 110, 110), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class TestIcons(_TempDirCase):
    def test_emoji_and_keyword_lookup(self):
        self.assertEqual(icons.find("🕯️", INDEX)["hexcode"], "1F56F")
        self.assertEqual(icons.find("🕯", INDEX)["hexcode"], "1F56F")  # without the variation selector
        self.assertEqual(icons.find("candle", INDEX)["hexcode"], "1F56F")
        self.assertEqual(icons.find("muscle", INDEX)["hexcode"], "1F4AA")
        self.assertEqual(icons.find("stomach", INDEX)["hexcode"], "E313")
        self.assertIsNone(icons.find("spaceship", INDEX))
        self.assertIsNone(icons.find("", INDEX))
        self.assertIsNone(icons.find("🚀", INDEX))

    def test_catalogue_and_pictures_are_cached(self):
        catalogue = MagicMock()
        catalogue.json.return_value = INDEX
        picture = MagicMock(content=b"png-bytes")
        with patch.object(icons.utils, "storage_dir", return_value=self.temp_dir), patch.object(
            icons, "_index", None
        ), patch.object(icons, "_get", side_effect=[catalogue, picture]) as get:
            index = icons.load_index()
            self.assertEqual([e["hexcode"] for e in index], ["1F56F", "1F4AA", "E313"])  # no flags
            path = icons.fetch("candle")
            self.assertEqual(Path(path).read_bytes(), b"png-bytes")
            self.assertEqual(icons.fetch("🕯️"), path)  # cached file, no download
            self.assertEqual(get.call_count, 2)
            self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, f"openmoji-{icons.OPENMOJI_VERSION}.json")))

    def test_offline_catalogue_is_empty(self):
        with patch.object(icons.utils, "storage_dir", return_value=self.temp_dir), patch.object(
            icons, "_index", None
        ), patch.object(icons, "_get", side_effect=OSError("offline")):
            self.assertEqual(icons.load_index(), [])
            self.assertEqual(icons.fetch("candle"), "")


class TestGeminiMedia(_TempDirCase):
    def _client(self, **responses):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = responses.get("content")
        client.models.generate_images.return_value = responses.get("images")
        return client

    def test_choice_parsing_and_prompt(self):
        self.assertEqual(gemini_media._parse_choice('{"choice": 2, "reason": "x"}', 3), 1)
        self.assertEqual(gemini_media._parse_choice('```json\n{"choice": 0}\n```', 3), -1)
        self.assertIsNone(gemini_media._parse_choice('{"choice": 9}', 3))
        self.assertIsNone(gemini_media._parse_choice("no json", 3))
        prompt = gemini_media.build_choice_prompt("Dormir poco afecta", "tired man", 3, "es-CO")
        self.assertIn("Spanish", prompt)
        self.assertIn("another language or alphabet", prompt)
        self.assertIn("answer 0", prompt)

    def test_enabled_follows_credentials(self):
        self.assertTrue(gemini_media.enabled({"gemini_api_key": "k"}))
        self.assertFalse(gemini_media.enabled({"gemini_api_key": ""}))
        self.assertTrue(gemini_media.enabled({"gemini_use_vertexai": True, "gemini_vertex_project": "p"}))

    def test_choose_picture_sends_the_pictures(self):
        pictures = [str(RESOURCES / "1.png"), str(RESOURCES / "2.png")]
        client = self._client(content=SimpleNamespace(text='{"choice": 2}'))
        with patch("google.genai.Client", return_value=client):
            choice = gemini_media.choose_picture(pictures, "line", "query", "es", {"gemini_api_key": "k"})
        self.assertEqual(choice, 1)
        contents = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(len(contents), 1 + 2 * len(pictures))
        self.assertEqual(client.models.generate_content.call_args.kwargs["model"], gemini_media.VISION_DEFAULT_MODEL)
        with patch("google.genai.Client", side_effect=RuntimeError("quota")):
            self.assertIsNone(gemini_media.choose_picture(pictures, "l", "q", "", {"gemini_api_key": "k"}))
        self.assertEqual(gemini_media.choose_picture([], "l", "q"), -1)

    def test_illustrations_are_drawn_once(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
        part = SimpleNamespace(inline_data=SimpleNamespace(data=buffer.getvalue()))
        client = self._client(content=SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]))
        settings = {"gemini_api_key": "k"}
        with patch.object(gemini_media.utils, "storage_dir", return_value=self.temp_dir), patch(
            "google.genai.Client", return_value=client
        ), patch.dict(gemini_media._state, {"gone": {}, "no_size": set(), "imagen_failed": "", "last_error": ""}):
            path = gemini_media.illustrate("a lit candle", settings)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(gemini_media.illustrate("a lit candle", settings), path)
            self.assertEqual(client.models.generate_content.call_count, 1)  # the Gemini image model, not Imagen
            self.assertEqual(client.models.generate_content.call_args.kwargs["model"], gemini_media.FLASH_IMAGE_MODELS[0])
            prompt = client.models.generate_content.call_args.kwargs["contents"]
            self.assertIn("a lit candle", prompt)
            self.assertIn("No text", prompt)
            self.assertEqual(gemini_media.illustrate("", settings), "")
        with patch.object(gemini_media.utils, "storage_dir", return_value=self.temp_dir), patch(
            "google.genai.Client", side_effect=RuntimeError("denied")
        ):
            self.assertEqual(gemini_media.illustrate("a whale", settings), "")

    def test_gemini_image_models_return_inline_data(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
        part = SimpleNamespace(inline_data=SimpleNamespace(data=buffer.getvalue()))
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        client = self._client(content=response)
        settings = {"gemini_api_key": "k", "gemini_image_model": "gemini-2.5-flash-image"}
        with patch.object(gemini_media.utils, "storage_dir", return_value=self.temp_dir), patch(
            "google.genai.Client", return_value=client
        ):
            self.assertTrue(gemini_media.illustrate("a fox", settings).endswith(".png"))


class TestSceneTiming(unittest.TestCase):
    TIMES = {"uno": 2.0, "dos": 3.0, "tres": 3.1, "setenta": 10.0, "fin": 16.5}

    def locate(self, anchor):
        return self.TIMES.get(anchor)

    def test_scenes_are_timed_spaced_and_kept_apart(self):
        specs = [
            {"type": "sequence", "items": [
                {"at": "uno", "label": "A", "icon": "x"}, {"at": "dos", "label": "B", "icon": "y", "mark": "cross"},
                {"at": "tres", "label": "C"}, {"at": "missing", "label": "D"},
            ]},
            {"type": "stat", "at": "dos", "value": 70, "unit": "%", "label": "overlaps", "chart": "pie"},
            {"type": "stat", "at": "setenta", "value": 70, "unit": "%", "label": "ojos", "chart": "pie", "icon": "👁️"},
            {"type": "statement", "at": "missing", "text": "x"},
            {"type": "bogus", "at": "uno"},
        ]
        timed = scenes.time_scenes(specs, self.locate, [5.6, 13.0], 20.0)
        self.assertEqual([s.type for s in timed], ["sequence", "stat"])
        sequence, stat = timed
        self.assertAlmostEqual(sequence.start, 1.55)
        self.assertEqual([i.label for i in sequence.items], ["A", "B", "C"])
        self.assertAlmostEqual(sequence.items[2].time, 3.5)  # pushed after "B"
        self.assertEqual(sequence.items[1].mark, "cross")
        self.assertAlmostEqual(sequence.end, 5.85)  # snapped to the pause after the last item
        self.assertEqual((stat.chart, stat.center.icon, stat.center.label), ("pie", "👁️", "ojos"))
        self.assertTrue(stat.vertical)  # scenes alternate their entrance
        self.assertFalse(sequence.vertical)

    def test_end_of_segment_and_blocked_moments(self):
        specs = [{"type": "statement", "at": "fin", "text": "Chao"}]
        last = scenes.time_scenes(specs, self.locate, [], 20.0)[0]
        self.assertEqual((last.end, last.exit), (20.0, False))
        specs = [{"type": "stat", "at": "setenta", "value": 3, "label": "x"}]
        shortened = scenes.time_scenes(specs, self.locate, [], 20.0, blocked=[(13.0, 16.0)])[0]
        self.assertAlmostEqual(shortened.end, 12.8)
        self.assertEqual(scenes.time_scenes(specs, self.locate, [], 20.0, blocked=[(10.0, 16.0)]), [])


class TestSceneDrawing(_TempDirCase):
    def test_primitives(self):
        mark = scenes.draw_mark("cross", 100)
        self.assertEqual(mark.mode, "RGBA")
        self.assertGreater(mark.getchannel("A").getbbox()[2], 80)
        frames, origin = scenes.arrow_frames((10, 10), (200, 120), 6, frames=5)
        self.assertEqual(len(frames), 5)
        self.assertLess(frames[0].getchannel("A").histogram()[0], frames[0].width * frames[0].height)
        drawn = [sum(f.getchannel("A").histogram()[1:]) for f in frames]
        self.assertEqual(drawn, sorted(drawn))  # the stroke only grows
        pie = scenes.pie_frames(70, 60, 5, frames=4)
        self.assertEqual(len({f.size for f in pie}), 1)
        pop = scenes.pop_frames(Image.new("RGBA", (50, 40), (255, 0, 0, 255)))
        self.assertEqual(len({f.size for f in pop}), 1)
        stamp = scenes.stamp_frames(mark)
        self.assertEqual(len({f.size for f in stamp}), 1)
        page = scenes.paper((64, 36), (255, 220, 225))
        self.assertEqual(page.size, (64, 36))
        self.assertTrue(scenes.hand_font_path(["Niebla del amanecer"]).endswith(scenes.HAND_FONT))
        text = scenes._Text(scenes.hand_font_path()).render("una frase bastante larga para dos lineas", 80, 400)
        self.assertLessEqual(text.width, 410)
        cut = scenes.prepare_picture(_icon(self.path("i.png")))
        self.assertLess(cut.width, 120)

    def test_every_scene_type_renders_overlays(self):
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        icon = scenes.prepare_picture(_icon(self.path("icon.png")))
        host_still = MagicMock(side_effect=lambda expression, height: Image.new("RGBA", (height // 2, height), (0, 128, 0, 255)))
        sfx = {name: f"/sfx/{name}.wav" for name in ("pop", "stamp", "scribble", "whoosh", "tick")}
        renderer = scenes.SceneRenderer(
            theme, self.temp_dir, (255, 230, 230), lambda item: icon if item.icon else None, host_still, sfx,
            font_path=scenes.hand_font_path(),
        )
        items = [scenes.SceneItem(label="uno", icon="x", time=1.0, mark="cross"), scenes.SceneItem(label="dos", time=2.0)]
        cases = [
            scenes.Scene("statement", 0.5, 4.0, text="Tu cerebro se lava", expression="feliz"),
            scenes.Scene("stat", 0.5, 4.0, value=70, unit="%", chart="pie", center=scenes.SceneItem(label="ojos")),
            scenes.Scene("stat", 0.5, 4.0, value=2.5, unit=" h", center=scenes.SceneItem(label="horas", icon="x")),
            scenes.Scene("sequence", 0.5, 4.0, items=list(items)),
            scenes.Scene("compare", 0.5, 4.0, items=list(items), vertical=True),
            scenes.Scene("diagram", 0.5, 4.0, items=list(items) + [scenes.SceneItem(label="tres", icon="x", time=2.5)],
                         center=scenes.SceneItem(icon="x")),
        ]
        for number, scene in enumerate(cases):
            with self.subTest(scene=scene.type):
                overlays, sounds = renderer.build(scene, f"s{number}")
                self.assertTrue(overlays[0].source.endswith(".png"))  # the paper
                self.assertGreater(len(overlays), 1)
                self.assertTrue(all(o.hold for o in overlays[1:]))
                self.assertTrue(all(o.end == scene.end for o in overlays))
                names = [os.path.basename(path) for _, path, _ in sounds]
                self.assertIn("whoosh.wav", names)
                axis = overlays[0].y if scene.vertical else overlays[0].x
                self.assertIn("pow(", axis)
        host_still.assert_called_once()
        statement_paper = renderer._backgrounds[theme.accent]
        self.assertTrue(os.path.isfile(statement_paper))  # statements use the brand colour

    def test_scene_overlays_render_exact_frames(self):
        theme = fx.Theme(640, 360, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        icon = scenes.prepare_picture(_icon(self.path("icon.png")))
        renderer = scenes.SceneRenderer(theme, self.temp_dir, (250, 240, 240), lambda item: icon, font_path=scenes.hand_font_path())
        scene = scenes.Scene("diagram", 0.2, 1.4, items=[
            scenes.SceneItem(label="a", icon="x", time=0.5), scenes.SceneItem(label="b", icon="x", time=0.8)],
            center=scenes.SceneItem(icon="x"))
        overlays, _ = renderer.build(scene, "render")
        params = VideoParams(video_subject="t", video_aspect="16:9")
        output = list_video.render_segment_video(list_video._Visual("none"), 45, params, self.path("seg.mp4"), overlays=overlays)
        self.assertEqual(list_video.count_video_frames(output), 45)


class TestScenePlan(unittest.TestCase):
    def test_normalize_scenes(self):
        data = {"segments": [{"index": 0, "scenes": [
            {"type": "statement", "at": "a b", "text": "t" * 80, "expression": "FELIZ"},
            {"type": "stat", "at": "c", "value": "70", "unit": "%", "chart": "pie", "label": "ojos", "icon": "👁️"},
            {"type": "stat", "at": "c", "value": 300, "unit": "%", "chart": "pie"},
            {"type": "sequence", "items": [{"at": "x", "label": "A", "mark": "cross"}, {"at": "y", "icon": "🧬", "mark": "maybe"}]},
            {"type": "compare", "items": [{"at": "x", "label": "A"}]},
            {"type": "diagram", "items": [{"at": "x", "label": "A"}, {"at": "y", "label": "B"}]},
            {"type": "stat", "at": "c", "value": "lots"},
            {"type": "chart"},
        ], "beats": [{"type": "image", "at": "q", "query": "tired man", "icon": "🥱"}]}]}
        plan = llm.normalize_edit_plan(data, 1, ["feliz"])[0]
        types = [s["type"] for s in plan["scenes"]]
        self.assertEqual(types, ["statement", "stat", "stat", "sequence"])  # at most four per segment
        statement, pie, number, _ = plan["scenes"]
        self.assertEqual(len(statement["text"]), llm.MAX_STATEMENT_LENGTH)
        self.assertEqual(statement["expression"], "feliz")
        self.assertEqual((pie["value"], pie["chart"]), (70, "pie"))
        self.assertEqual(number["chart"], "number")  # 300 % is not a share of a whole
        self.assertEqual(plan["beats"][0]["icon"], "🥱")

        data["segments"][0]["scenes"] = data["segments"][0]["scenes"][3:]
        plan = llm.normalize_edit_plan(data, 1, [])[0]
        sequence, diagram = plan["scenes"]
        self.assertEqual(sequence["items"][0]["mark"], "cross")
        self.assertNotIn("mark", sequence["items"][1])
        self.assertEqual(diagram["center"], {"at": "", "label": "", "icon": "", "draw": ""})
        self.assertEqual(llm.normalize_edit_plan({"segments": []}, 1, [])[0]["scenes"], [])

    def test_prompt_lists_scene_types_and_reference(self):
        segments = [{"index": 0, "kind": "item", "title": "1. A", "text": "hola mundo"}]
        prompt = llm.build_edit_plan_prompt(segments, ["feliz"], "es-CO")
        for word in ('"scenes"', '"statement"', '"stat"', '"sequence"', '"compare"', '"diagram"', "SIMPLEST picture"):
            self.assertIn(word, prompt)
        self.assertNotIn("Reference plan", prompt)
        reference = [{"index": 0, "scenes": [{"type": "stat", "at": "about seventy", "value": 70}]}]
        prompt = llm.build_edit_plan_prompt(segments, ["feliz"], "pt-BR", reference)
        self.assertIn("Reference plan", prompt)
        self.assertIn("about seventy", prompt)
        reply = json.dumps({"segments": [{"index": 0, "expression": "feliz", "beats": []}]})
        with patch.object(llm, "_generate_response", return_value=reply) as generate:
            self.assertEqual(llm.generate_edit_plan(segments, ["feliz"], "pt-BR", reference=reference)[0]["scenes"], [])
        self.assertIn("Reference plan", generate.call_args.args[0])


def _segments():
    script = ListVideoScript(
        title="T",
        intro="Hola nutrias.",
        items=[ListVideoItem(name="Ojos", text=" ".join(f"palabra{i}" for i in range(60)))],
        outro="Chao.",
    )
    return list_video.build_segments(script)


class TestEditorScenesAndPictures(_TempDirCase):
    def setUp(self):
        super().setUp()
        for target, value in ((editor.gemini_media, "enabled"),):
            patcher = patch.object(target, value, return_value=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _editor(self, **options):
        narrations = [
            editor.Narration(pcm=b"", speech_seconds=1.0, frames=45),
            editor.Narration(pcm=b"", speech_seconds=24.0, frames=730),
            editor.Narration(pcm=b"", speech_seconds=1.0, frames=45),
        ]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        options.setdefault("progress_bar", False)
        options.setdefault("openers", False)  # covered by test_list_video_round4
        return editor.Editor(editor.EditOptions(**options), theme, self.temp_dir, _segments(), narrations)

    def test_picture_check_picks_rejects_and_falls_back_to_icons(self):
        ed = self._editor()
        candidates = [
            web_images.WebImage(str(RESOURCES / f"{n}.png"), "pexels", f"P{n}", "A", "L", "https://p", url=f"https://u/{n}")
            for n in (1, 2)
        ]
        beat = editor._Beat("image", 1.0, 3.0, query="tired man", icon="🥱", at="palabra3")
        with patch.object(editor.web_images, "find_candidates", return_value=candidates) as find, patch.object(
            editor.gemini_media, "choose_picture", return_value=1
        ) as choose:
            self.assertEqual(ed._beat_picture(beat, "una frase"), str(RESOURCES / "2.png"))
        self.assertEqual(find.call_args.kwargs["limit"], 4)
        self.assertEqual(choose.call_args.args[1:3], ("una frase", "tired man"))
        self.assertIn("https://u/2", ed._used_urls)
        with patch.object(editor.web_images, "find_candidates", return_value=candidates), patch.object(
            editor.gemini_media, "choose_picture", return_value=None
        ):
            self.assertEqual(ed._beat_picture(beat), str(RESOURCES / "1.png"))  # check failed: first one
        icon = _icon(self.path("icon.png"))
        with patch.object(editor.web_images, "find_candidates", return_value=candidates), patch.object(
            editor.gemini_media, "choose_picture", return_value=-1
        ), patch.object(editor.icons, "fetch", return_value=icon):
            self.assertEqual(ed._beat_picture(beat), icon)
        self.assertIn(icons.CREDIT, ed.credits)
        with patch.object(editor.web_images, "find_candidates", return_value=[]), patch.object(
            editor.icons, "fetch", return_value=""
        ):
            self.assertEqual(ed._beat_picture(beat), "")
        self.assertTrue(any("no picture found" in w for w in ed.warnings))

        unchecked = self._editor(picture_check=False)
        with patch.object(editor.web_images, "find_candidates", return_value=candidates[:1]) as find, patch.object(
            editor.gemini_media, "choose_picture"
        ) as choose:
            unchecked._beat_picture(beat)
        choose.assert_not_called()
        self.assertEqual(find.call_args.kwargs["limit"], 1)

        drawn = self._editor(beats="ai")
        with patch.object(editor.gemini_media, "illustrate", return_value=icon):
            self.assertEqual(drawn._beat_picture(beat), icon)

    def test_scenes_hide_beats_and_host_entrances(self):
        ed = self._editor(assets_dir=NUTRIA, host="always")
        ed.plan = [
            {"index": 0, "expression": "feliz", "scenes": [], "beats": []},
            {"index": 1, "expression": "explicando", "backgrounds": [], "scenes": [
                {"type": "sequence", "items": [
                    {"at": "palabra10", "label": "uno", "icon": "🕯️"},
                    {"at": "palabra14", "label": "dos", "icon": "🧬", "mark": "cross"},
                ]},
            ], "beats": [
                {"type": "image", "at": "palabra12", "query": "hidden under the scene"},
                {"type": "image", "at": "palabra40", "query": "after the scene"},
                {"type": "react", "at": "palabra13", "expression": "sorprendido"},
            ]},
            {"index": 2, "expression": "feliz", "scenes": [], "beats": []},
        ]
        icon = _icon(self.path("icon.png"))
        picture = web_images.WebImage(str(RESOURCES / "1.png"), "pexels", "P", "A", "L", "https://p", url="https://u")
        with patch.object(editor.web_images, "find_candidates", return_value=[picture]) as find, patch.object(
            editor.gemini_media, "choose_picture", return_value=0
        ), patch.object(editor.icons, "fetch", return_value=icon):
            edit = ed.segment_edit(1, 1.5, show_titles=True)
        self.assertEqual([c.args[0] for c in find.call_args_list], ["after the scene"])
        timed = ed._scenes[1]
        self.assertEqual([s.type for s in timed], ["sequence"])
        scene = timed[0]
        modes = [os.path.basename(o.source) for o in edit.overlays]
        host_at = next(i for i, o in enumerate(edit.overlays) if o.mode == "concat")
        paper_at = next(i for i, name in enumerate(modes) if name.startswith("paper-"))
        chip_at = next(i for i, name in enumerate(modes) if name.startswith("chip-"))
        self.assertLess(host_at, paper_at)  # the scene covers the host ...
        self.assertLess(paper_at, chip_at)  # ... but not the chapter label
        sounds = [os.path.basename(p) for _, p, _ in edit.sounds]
        self.assertIn("sfx-stamp.wav", sounds)
        # The reaction during the scene is dropped, so is any entrance under it.
        planned = ed.host_plan[1]
        self.assertFalse(any(scene.start - 0.4 <= w.start < scene.end and w.enter for w in planned.windows))
        self.assertIn(icons.CREDIT, ed.credits)

    def test_scene_pictures_prefer_ai_when_asked(self):
        ed = self._editor(illustrations="ai")
        drawn = _icon(self.path("drawn.png"), color=(0, 0, 255, 255))
        with patch.object(editor.gemini_media, "illustrate", return_value=drawn) as illustrate, patch.object(
            editor.icons, "fetch"
        ) as fetch:
            item = scenes.SceneItem(label="vela", icon="🕯️", draw="a lit candle")
            ed._item_picture(item)
            self.assertIsNotNone(ed._scene_picture(item))
            self.assertIs(ed._scene_picture(item), ed._scene_picture(item))  # prepared once
        illustrate.assert_called_once_with("a lit candle")
        fetch.assert_not_called()
        icons_only = self._editor(picture_check=False)
        with patch.object(editor.icons, "fetch", side_effect=["", ""]) as fetch:
            self.assertIsNone(icons_only._scene_picture(scenes.SceneItem(icon="🛸", draw="ufo")))
        self.assertEqual([c.args[0] for c in fetch.call_args_list], ["🛸", "ufo"])

    def test_scene_colour_and_reference_plan(self):
        accent = (255, 79, 94)
        self.assertEqual(editor.scene_canvas_color("accent", accent), accent)
        self.assertEqual(editor.scene_canvas_color("#102030", accent), (16, 32, 48))
        soft = editor.scene_canvas_color("", accent)
        self.assertTrue(all(s > a for s, a in zip(soft[1:], accent[1:])))
        self.assertEqual(editor.sentence_at("Uno dos. Tres cuatro cinco! Seis.", "cuatro cinco"), "Tres cuatro cinco!")

        reference = self.path("ref.json")
        Path(reference).write_text(json.dumps({"segments": [{"index": 0, "host": "off", "scenes": []}]}), "utf-8")
        ed = self._editor(reference_plan=reference)
        with patch.object(editor.llm, "generate_edit_plan", return_value=None) as generate:
            ed.make_plan()
        self.assertEqual(generate.call_args.kwargs["reference"], [{"index": 0, "scenes": []}])
        no_scenes = self._editor(scenes=False, plan_file=reference)
        self.assertTrue(all(entry["scenes"] == [] for entry in no_scenes.make_plan()))


class TestSceneCli(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.script = os.path.join(self.temp_dir, "s.json")
        Path(self.script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")

    def _run(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = list_video_cli.run(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_scene_options(self):
        with patch.object(list_video, "generate_list_video", return_value={}) as generate:
            self._run(["--script", self.script])
            options = generate.call_args.kwargs["edit"]
            self.assertEqual(
                (options.scenes, options.illustrations, options.scene_color, options.picture_check, options.lip_sync),
                (True, "icons", "", True, False),
            )
            self._run([
                "--script", self.script, "--no-scenes", "--illustrations", "ai", "--scene-color", "white",
                "--no-picture-check", "--lip-sync",
            ])
            options = generate.call_args.kwargs["edit"]
            self.assertEqual(
                (options.scenes, options.illustrations, options.scene_color, options.picture_check, options.lip_sync),
                (False, "ai", "white", False, True),
            )
        code, _, stderr = self._run(["--script", self.script, "--scene-color", "blue"])
        self.assertEqual(code, 2)
        self.assertIn("--scene-color", stderr)


if __name__ == "__main__":
    unittest.main()
