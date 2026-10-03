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

import numpy as np
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
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))
NAMES = ["explicando", "neutral", "feliz", "pensando", "saludando", "senalando", "sorprendido", "preocupado"]


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


def _icon(path, color=(220, 60, 60, 255), size=120):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, size - 10, size - 10), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


class TestHostFaces(unittest.TestCase):
    def test_face_changes_about_every_three_seconds_on_breaths(self):
        info = host.SegmentInfo(
            kind="item", duration=30.0, expression="explicando", pauses=[4.0, 7.1, 10.4, 14.2, 20.0], mode="full"
        )
        cues = host.plan_host([info], NAMES)[0].cues
        times = [c.time for c in cues]
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertGreaterEqual(len(cues), 7)
        self.assertTrue(all(2.3 <= gap <= 4.1 for gap in gaps), gaps)
        self.assertIn(7.1 - 1 / 30 * 0, [round(t, 1) for t in times])  # snapped to a breath
        faces = [c.expression for c in cues]
        self.assertTrue(all(a != b for a, b in zip(faces, faces[1:])))
        self.assertTrue(set(faces) <= {"explicando", "neutral", "feliz", "pensando"})
        self.assertTrue(all(c.bounce for c in cues[1:]))

    def test_palettes_follow_the_mood_and_skip_special_poses(self):
        self.assertEqual(host.idle_palette("preocupado", NAMES), ["preocupado", "pensando", "explicando"])
        self.assertEqual(host.idle_palette("custom", ["custom", "feliz"]), ["custom", "feliz"])
        self.assertNotIn("senalando", host.idle_palette("explicando", NAMES, ["senalando"]))
        # A single pose cannot vary.
        info = host.SegmentInfo(kind="item", duration=20.0, expression="feliz", mode="full")
        self.assertEqual(len(host.plan_host([info], ["feliz"])[0].cues), 1)

    def test_reactions_keep_their_face_and_blocked_moments_send_the_host_away(self):
        info = host.SegmentInfo(
            kind="item", duration=24.0, expression="explicando", mode="full",
            reactions=[(6.0, "sorprendido")], blocked=[(12.0, 17.0)],
        )
        planned = host.plan_host([info], NAMES)[0]
        surprised = next(c for c in planned.cues if c.expression == "sorprendido")
        following = planned.cues[planned.cues.index(surprised) + 1]
        self.assertGreaterEqual(following.time - surprised.time, host.REACTION_HOLD - 0.05)
        for window in planned.windows:
            self.assertFalse(window.start < 17.2 and window.end > 12.0 - host.EXIT_SECONDS - 0.1, window)
        self.assertEqual(len(planned.windows), 2)
        self.assertTrue(planned.windows[0].exit and planned.windows[1].enter)


class TestSafeCutouts(unittest.TestCase):
    def test_only_clean_light_backgrounds_are_cut(self):
        rng = np.random.default_rng(1)
        dark = Image.fromarray(np.clip(rng.normal(18, 6, (300, 260, 3)), 0, 255).astype(np.uint8))
        ImageDraw.Draw(dark).ellipse((60, 80, 200, 240), fill=(40, 30, 20))
        self.assertIsNone(fx.cutout_or_none(dark))
        white = Image.new("RGB", (300, 260), "white")
        ImageDraw.Draw(white).ellipse((40, 40, 260, 220), fill=(240, 180, 40), outline=(0, 0, 0), width=6)
        self.assertIsNotNone(fx.cutout_or_none(white))
        self.assertIsNone(fx.cutout_or_none(white, allow_cutout=False))
        scattered = Image.new("RGB", (300, 260), "white")
        draw = ImageDraw.Draw(scattered)
        for k in range(10):
            draw.rectangle((10 + k * 28, 30 + (k % 3) * 70, 24 + k * 28, 50 + (k % 3) * 70), fill=(0, 0, 0))
        self.assertIsNone(fx.cutout_or_none(scattered))
        flat_blue = Image.new("RGB", (300, 260), (40, 90, 200))
        ImageDraw.Draw(flat_blue).ellipse((60, 60, 240, 200), fill=(250, 220, 0))
        self.assertIsNotNone(fx.cutout_or_none(flat_blue))  # perfectly flat digital colour

    def test_shape_share_and_fragmented_transparency(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[2:20, 2:20] = True
        mask[30:32, 30:32] = True
        self.assertAlmostEqual(fx.main_shape_share(mask), 324 / 328, places=3)
        self.assertEqual(fx.main_shape_share(np.zeros((5, 5), dtype=bool)), 0.0)
        confetti = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(confetti)
        for k in range(12):
            draw.rectangle((5 + (k % 4) * 50, 5 + (k // 4) * 60, 20 + (k % 4) * 50, 20 + (k // 4) * 60), fill=(255, 0, 0, 255))
        self.assertIsNone(fx.cutout_or_none(confetti))

    def test_stickers_and_scene_pictures_fall_back_to_frames(self):
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        rng = np.random.default_rng(2)
        photo = os.path.join(folder, "photo.png")
        Image.fromarray(np.clip(rng.normal(20, 8, (300, 400, 3)), 0, 255).astype(np.uint8)).save(photo)
        card = fx.make_sticker(theme, photo, 400, 300)
        self.assertGreater(np.asarray(card.getchannel("A")).mean(), 150)  # a solid card, not shreds
        prepared = scenes.prepare_picture(photo)
        self.assertTrue(prepared.info.get("framed"))
        renderer = scenes.SceneRenderer(theme, folder, (250, 240, 240), lambda item: prepared)
        framed = renderer._picture(scenes.SceneItem(label="x"), 200)
        self.assertLessEqual(max(framed.size), 200 + 60)
        icon = scenes.prepare_picture(_icon(os.path.join(folder, "icon.png")))
        self.assertFalse(icon.info.get("framed"))


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


class TestPictureBeats(unittest.TestCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def test_groups_are_scheduled_with_their_members(self):
        beats = [
            {"type": "images", "items": [
                {"at": "palabra10", "query": "a", "icon": "🅰"},
                {"at": "palabra10", "query": "b"},
                {"at": "palabra14", "query": "c", "look": "photo"},
                {"at": "missing", "query": "d"},
            ]},
            {"type": "image", "at": "palabra12", "query": "hidden behind the group"},
            {"type": "image", "at": "palabra40", "query": "later"},
            {"type": "images", "items": [{"at": "palabra50", "query": "alone"}]},
            {"type": "text", "at": "palabra11", "text": "dato"},
        ]
        scheduled = editor.schedule_beats(beats, self.TEXT, _narration(24.0, 760), 25.3)
        kinds = [(b.kind, b.query or b.text or len(b.members)) for b in scheduled]
        self.assertEqual(kinds, [("images", 3), ("text", "dato"), ("image", "later"), ("image", "alone")])
        group = scheduled[0]
        self.assertEqual([m.query for m in group.members], ["a", "b", "c"])
        self.assertAlmostEqual(group.members[1].start - group.members[0].start, 0.5)
        self.assertEqual(group.members[2].look, "photo")
        self.assertLessEqual(group.end - group.members[-1].start, editor.GROUP_TAIL + 1e-6)
        self.assertTrue(all(m.end == group.end for m in group.members))

    def test_picture_slots_are_centred_and_symmetric(self):
        theme = fx.Theme(1920, 1080, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        segments = list_video.build_segments(ListVideoScript(title="T", items=[ListVideoItem(name="A", text="B")]))
        ed = editor.Editor(editor.EditOptions(subscribe="none"), theme, folder, segments, [_narration(1, 30)])
        (one,) = ed._picture_slots(1)
        self.assertEqual(one[0], 960)
        two = ed._picture_slots(2)
        self.assertAlmostEqual(two[0][0] + two[1][0], 1920)
        three = ed._picture_slots(3)
        self.assertEqual(three[1][0], 960)
        self.assertAlmostEqual(three[0][0] + three[2][0], 1920)
        self.assertTrue(all(box_w <= 1920 * 0.34 for _, _, box_w, _ in ed._picture_slots(4)))
        portrait = editor.Editor(
            editor.EditOptions(subscribe="none"), fx.Theme(1080, 1920, FONT, (255, 0, 0)), folder, segments, [_narration(1, 30)]
        )
        self.assertTrue(all(x == 540 for x, _, _, _ in portrait._picture_slots(3)))


class TestRound3Plan(unittest.TestCase):
    LOOKUP = ["pensando", "sorprendido"]

    def _scene(self, data):
        plan = llm.normalize_edit_plan({"segments": [{"index": 0, "scenes": [data]}]}, 1, self.LOOKUP)[0]
        return plan["scenes"][0] if plan["scenes"] else None

    def test_every_new_scene_type(self):
        self.assertEqual(self._scene({"type": "question", "at": "a", "text": "¿Por qué?", "expression": "PENSANDO"})["expression"], "pensando")
        figure = self._scene({"type": "figure", "at": "a", "query": "voltage diagram", "query_local": "diagrama", "seconds": 30, "look": "photo"})
        self.assertEqual((figure["seconds"], figure["look"]), (9, "photo"))
        self.assertIsNone(self._scene({"type": "figure", "at": "a"}))
        zoom = self._scene({"type": "zoom", "at": "a", "label": "x", "icon": "🔋", "direction": "out"})
        self.assertEqual((zoom["type"], zoom["direction"]), ("zoom", "out"))
        grid = self._scene({"type": "grid", "at": "a", "value": 7, "total": 10, "label": "7 de 10"})
        self.assertEqual((grid["value"], grid["total"]), (7, 10))
        self.assertIsNone(self._scene({"type": "grid", "at": "a", "value": 12, "total": 10}))
        self.assertEqual(self._scene({"type": "gauge", "at": "a", "value": 130})["value"], 100)
        bars = self._scene({"type": "bars", "unit": " V", "items": [
            {"at": "a", "label": "pila", "value": "1.5"}, {"at": "b", "label": "rayo", "value": 300}, {"at": "c", "label": "x"}]})
        self.assertEqual([i["value"] for i in bars["items"]], [1.5, 300])
        steps = self._scene({"type": "steps", "cycle": True, "items": [{"at": "a", "label": "x"}, {"at": "b", "label": "y"}]})
        self.assertTrue(steps["cycle"])
        timeline = self._scene({"type": "timeline", "items": [{"at": "a", "label": "x", "date": "1800"}, {"at": "b", "label": "y", "date": "1900"}]})
        self.assertEqual(timeline["items"][1]["date"], "1900")
        formula = self._scene({"type": "formula", "operator": "*", "items": [{"at": "a", "label": "x"}, {"at": "b", "label": "y"}], "result": {"at": "c", "label": "z"}})
        self.assertEqual((formula["operator"], formula["result"]["label"]), ("+", "z"))
        self.assertIsNone(self._scene({"type": "formula", "items": [{"at": "a", "label": "x"}, {"at": "b", "label": "y"}]}))
        story = self._scene({"type": "story", "items": [
            {"at": "a", "label": "x", "expression": "sorprendido", "query": "bulb"}, {"at": "b", "label": "y", "expression": "nope"}]})
        self.assertEqual(story["items"][0]["expression"], "sorprendido")
        self.assertEqual(story["items"][0]["query"], "bulb")
        self.assertNotIn("expression", story["items"][1])

    def test_image_groups_and_prompt(self):
        beats = llm.normalize_edit_plan({"segments": [{"index": 0, "beats": [
            {"type": "images", "items": [{"at": "a", "query": "x", "icon": "⚡"}, {"at": "b", "query": "y"}, {"at": "", "query": "z"}]},
            {"type": "images", "items": [{"at": "c", "query": "only"}]},
        ]}]}, 1, [])[0]["beats"]
        self.assertEqual(beats[0]["type"], "images")
        self.assertEqual([i["query"] for i in beats[0]["items"]], ["x", "y"])
        self.assertEqual(beats[1], {"type": "image", "at": "c", "query": "only", "look": "diagram"})
        prompt = llm.build_edit_plan_prompt([{"index": 0, "kind": "item", "title": "1", "text": "t"}], ["pensando"], "es-CO")
        for word in ('"images"', '"figure"', '"zoom"', '"story"', '"steps"', '"bars"', '"grid"', '"formula"', '"timeline"', '"gauge"', '"question"', "MANY different types", "in the centre"):
            self.assertIn(word, prompt)


class TestGeminiPurposes(_TempDirCase):
    def test_prompts_by_purpose(self):
        scene = gemini_media.build_choice_prompt("x", "bulb", 3, "es", "scene")
        self.assertIn("very closely related", scene)
        figure = gemini_media.build_choice_prompt("x", "voltage diagram", 3, "pt-BR", "figure")
        self.assertIn('"seconds"', figure)
        self.assertIn("Portuguese", figure)
        self.assertNotIn("very closely related", gemini_media.build_choice_prompt("x", "y", 2))

    def test_choose_figure_returns_seconds(self):
        pictures = [str(RESOURCES / "1.png"), str(RESOURCES / "2.png")]
        with patch.object(gemini_media, "_ask", return_value='{"choice": 2, "seconds": 7, "reason": "clear"}'):
            self.assertEqual(gemini_media.choose_figure(pictures, "l", "q", "es", {"gemini_api_key": "k"}), (1, 7.0))
        with patch.object(gemini_media, "_ask", return_value='{"choice": 0}'):
            self.assertEqual(gemini_media.choose_figure(pictures, "l", "q", "es", {"gemini_api_key": "k"})[0], -1)
        with patch.object(gemini_media, "_ask", return_value='{"choice": 1, "seconds": 40}'):
            self.assertEqual(gemini_media.choose_figure(pictures, "l", "q", "es", {"gemini_api_key": "k"}), (0, 9.0))
        with patch.object(gemini_media, "_ask", side_effect=RuntimeError("quota")):
            self.assertIsNone(gemini_media.choose_figure(pictures, "l", "q", "es", {"gemini_api_key": "k"}))
        with patch.object(gemini_media, "_ask", return_value="nonsense"):
            self.assertIsNone(gemini_media.choose_figure(pictures, "l", "q", "es", {"gemini_api_key": "k"}))
        self.assertEqual(gemini_media.choose_figure([], "l", "q"), (-1, 0.0))

    def test_story_frames_are_drawn_from_each_other(self):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
        part = SimpleNamespace(inline_data=SimpleNamespace(data=buffer.getvalue()))
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = response
        settings = {"gemini_api_key": "k", "gemini_image_model": "imagen-4.0-fast-generate-001"}
        with patch.object(gemini_media.utils, "storage_dir", return_value=self.temp_dir), patch(
            "google.genai.Client", return_value=client
        ):
            frames = gemini_media.illustrate_sequence(["an otter plugs a lamp", "the lamp lights up"], settings)
            self.assertEqual(len(frames), 2)
            calls = client.models.generate_content.call_args_list
            self.assertEqual(calls[0].kwargs["model"], gemini_media.SEQUENCE_DEFAULT_MODEL)  # Imagen cannot edit
            self.assertEqual(len(calls[1].kwargs["contents"]), 2)  # previous frame + instruction
            self.assertEqual(gemini_media.illustrate_sequence(["an otter plugs a lamp", "the lamp lights up"], settings), frames)
            self.assertEqual(client.models.generate_content.call_count, 2)  # cached
            self.assertEqual(gemini_media.illustrate_sequence(["only one"], settings), [])
        with patch.object(gemini_media.utils, "storage_dir", return_value=self.temp_dir), patch(
            "google.genai.Client", side_effect=RuntimeError("no access")
        ):
            self.assertEqual(gemini_media.illustrate_sequence(["a", "b", "c"], settings), [])


class TestNewSceneTypes(_TempDirCase):
    TIMES = {"a": 2.0, "b": 3.0, "c": 4.0, "fig": 6.0}

    def test_timing_of_new_types(self):
        locate = self.TIMES.get
        specs = [
            {"type": "formula", "operator": "×", "items": [{"at": "a", "label": "x"}, {"at": "b", "label": "y"}], "result": {"at": "c", "label": "z"}},
        ]
        formula = scenes.time_scenes(specs, locate, [], 30.0)[0]
        self.assertEqual([i.role for i in formula.items], ["", "", "result"])
        self.assertEqual(formula.operator, "×")
        figure = scenes.time_scenes([{"type": "figure", "at": "fig", "query": "q", "seconds": 7, "look": "diagram", "label": "cap"}], locate, [], 30.0)[0]
        self.assertAlmostEqual(figure.start, 5.7)
        self.assertAlmostEqual(figure.end, 5.7 + 7 + 0.25)
        self.assertEqual((figure.center.label, figure.center.at, figure.query), ("cap", "fig", "q"))
        grid = scenes.time_scenes([{"type": "grid", "at": "a", "value": 7, "total": 10, "label": "x", "icon": "🧑"}], locate, [], 30.0)[0]
        self.assertEqual((grid.total, grid.value, grid.center.icon), (10, 7.0, "🧑"))
        story = scenes.time_scenes([{"type": "story", "items": [{"at": "a", "label": "x", "expression": "feliz"}, {"at": "c", "label": "y"}]}], locate, [], 30.0)[0]
        self.assertEqual(story.items[0].expression, "feliz")
        late = scenes.time_scenes([{"type": "steps", "items": [{"at": "a", "label": "x"}, {"at": "b", "label": "y"}]}], locate, [], 4.0)
        self.assertEqual(late, [])  # the second step would only flash before the cut

    def test_every_new_type_builds(self):
        theme = fx.Theme(640, 360, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        icon = scenes.prepare_picture(_icon(self.path("icon.png")))
        rng = np.random.default_rng(3)
        photo_path = self.path("photo.png")
        Image.fromarray(np.clip(rng.normal(30, 9, (200, 300, 3)), 0, 255).astype(np.uint8)).save(photo_path)
        photo = scenes.prepare_picture(photo_path, allow_cutout=False)
        moment = photo.copy()
        moment.info["framed"] = True
        moment.info["scene"] = True

        def picture(item):
            return {"photo": photo, "moment": moment}.get(item.query, icon)

        host_still = MagicMock(side_effect=lambda expression, height: Image.new("RGBA", (height // 2, height), (0, 128, 0, 255)))
        renderer = scenes.SceneRenderer(theme, self.temp_dir, (250, 235, 235), picture, host_still, {"pop": "p", "whoosh": "w"}, scenes.hand_font_path())
        Item = scenes.SceneItem
        two = [Item(label="uno", icon="x", time=1.0, value=3, date="1900"), Item(label="dos", icon="x", time=1.5, value=9, date="2000")]
        cases = [
            scenes.Scene("figure", 0.5, 2.0, look="photo", center=Item(query="photo", label="caption")),
            scenes.Scene("figure", 0.5, 2.0, look="diagram", center=Item(query="x")),
            scenes.Scene("zoom", 0.5, 2.0, direction="out", center=Item(label="pila", icon="x")),
            scenes.Scene("story", 0.5, 3.0, items=[Item(label="a", icon="x", time=1.0, expression="feliz"), Item(label="b", icon="x", time=1.8, expression="sorprendido")]),
            scenes.Scene("story", 0.5, 3.0, items=[Item(label="a", query="moment", time=1.0), Item(label="b", query="moment", time=1.8)]),
            scenes.Scene("steps", 0.5, 3.0, items=list(two)),
            scenes.Scene("steps", 0.5, 3.0, cycle=True, items=list(two) + [Item(label="tres", icon="x", time=2.0)]),
            scenes.Scene("bars", 0.5, 3.0, unit=" V", items=list(two)),
            scenes.Scene("grid", 0.5, 3.0, value=3, total=10, center=Item(label="3 de 10", icon="x")),
            scenes.Scene("formula", 0.5, 3.0, operator="×", items=[Item(label="a", icon="x", time=1.0), Item(label="b", icon="x", time=1.4), Item(label="c", icon="x", time=1.8, role="result")]),
            scenes.Scene("timeline", 0.5, 3.0, items=list(two)),
            scenes.Scene("gauge", 0.5, 3.0, value=80, low="bajo", high="alto", center=Item(label="riesgo")),
            scenes.Scene("question", 0.5, 3.0, text="¿Por qué?", expression="pensando"),
        ]
        for number, scene in enumerate(cases):
            with self.subTest(scene=scene.type, number=number):
                overlays, sounds = renderer.build(scene, f"n{number}")
                self.assertTrue(overlays)
                if scene.type in scenes.CLIP_SCENES:
                    self.assertEqual([o.mode for o in overlays], ["media"])
                    self.assertTrue(overlays[0].source.endswith("clip.mp4"))
                    self.assertGreaterEqual(list_video.count_video_frames(overlays[0].source), int((scene.end - scene.start) * 30))
                else:
                    self.assertTrue(overlays[0].source.endswith(".png"))
        empty = scenes.SceneRenderer(theme, self.temp_dir, (250, 235, 235), lambda item: None)
        with self.assertRaises(ValueError):
            empty.build(scenes.Scene("figure", 0.5, 2.0, center=Item()), "empty")

    def test_story_frames_replace_each_other(self):
        theme = fx.Theme(640, 360, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        icon = scenes.prepare_picture(_icon(self.path("icon.png")))
        host_still = MagicMock(side_effect=lambda expression, height: Image.new("RGBA", (height // 2, height), (0, 128, 0, 255)))
        renderer = scenes.SceneRenderer(theme, self.temp_dir, (250, 235, 235), lambda item: icon, host_still, {}, scenes.hand_font_path())
        Item = scenes.SceneItem
        scene = scenes.Scene("story", 0.5, 3.0, items=[Item(label="a", time=1.0, expression="feliz"), Item(label="b", time=1.8, expression="sorprendido")])
        overlays, _ = renderer.build(scene, "story")
        props = [o for o in overlays if "prop" in o.source]
        self.assertEqual([(o.start, o.end) for o in props], [(1.0, 1.8), (1.8, 3.0)])
        bursts = [o for o in overlays if "burst" in o.source]
        self.assertEqual(len(bursts), 1)
        self.assertFalse(bursts[0].hold)
        self.assertEqual([c.args[0] for c in host_still.call_args_list], ["feliz", "sorprendido"])


def _segments(text):
    return list_video.build_segments(
        ListVideoScript(title="T", intro="Hola nutrias.", items=[ListVideoItem(name="Ojos", text=text)], outro="Chao.")
    )


class TestEditorRound3(_TempDirCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def _editor(self, gemini=True, **options):
        patcher = patch.object(editor.gemini_media, "enabled", return_value=gemini)
        patcher.start()
        self.addCleanup(patcher.stop)
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        return editor.Editor(editor.EditOptions(**options), theme, self.temp_dir, _segments(self.TEXT), narrations)

    def _found(self, n):
        return web_images.WebImage(str(RESOURCES / f"{n}.png"), "pexels", f"P{n}", "A", "L", "https://p", url=f"https://u/{n}")

    def test_groups_figures_and_scene_pictures(self):
        ed = self._editor(assets_dir=NUTRIA, host="always")
        ed.plan = [
            {"index": 0, "expression": "feliz", "scenes": [], "beats": []},
            {"index": 1, "expression": "explicando", "backgrounds": [], "scenes": [
                {"type": "figure", "at": "palabra30", "query": "voltage diagram", "query_local": "diagrama voltaje", "look": "diagram", "seconds": 4},
                {"type": "sequence", "items": [
                    {"at": "palabra44", "label": "uno", "icon": "🕯️", "query": "candle"},
                    {"at": "palabra47", "label": "dos", "icon": "🧬"},
                ]},
            ], "beats": [
                {"type": "images", "items": [
                    {"at": "palabra5", "query": "first"}, {"at": "palabra7", "query": "second"}, {"at": "palabra9", "query": "third"},
                ]},
                {"type": "image", "at": "palabra20", "query": "single"},
            ]},
            {"index": 2, "expression": "feliz", "scenes": [], "beats": []},
        ]
        queries = []

        def candidates(query, save_dir, kind="diagram", exclude_urls=None, limit=4):
            queries.append(query)
            return [] if query == "third" else [self._found(1), self._found(2)]

        icon = _icon(self.path("icon.png"))
        with patch.object(editor.web_images, "find_candidates", side_effect=candidates), patch.object(
            editor.gemini_media, "choose_picture", return_value=1
        ) as choose, patch.object(editor.gemini_media, "choose_figure", return_value=(0, 4.0)), patch.object(
            editor.icons, "fetch", return_value=icon
        ):
            edit = ed.segment_edit(1, 1.5, show_titles=False)
        self.assertEqual(set(queries), {"first", "second", "third", "single", "voltage diagram", "diagrama voltaje", "candle"})
        purposes = sorted(c.kwargs.get("purpose", "beat") for c in choose.call_args_list)
        self.assertIn("scene", purposes)
        figure = next(s for s in ed._scenes[1] if s.type == "figure")
        self.assertLessEqual(figure.end - figure.start, 4.0 + 0.3 + 1e-6)  # Gemini said four seconds
        group = next(b for b, _ in ed._beats[1] if b.kind == "images")
        self.assertEqual([m.query for m in group.members], ["first", "second"])  # "third" had no picture
        stickers = [o for o in edit.overlays if os.path.basename(o.source).startswith("beat-01-")]
        self.assertEqual(len(stickers), 3)
        xs = sorted(int(o.x) for o in stickers if int(o.x) > 0)
        self.assertTrue(xs)
        self.assertTrue(any(o.mode == "media" for o in edit.overlays))  # the figure clip
        planned = ed.host_plan[1]
        for window in planned.windows:
            self.assertFalse(window.start < group.end and window.end > group.start, (window, group.start, group.end))

    def test_figures_need_gemini_and_story_frames_use_the_image_model(self):
        ed = self._editor(gemini=False)
        ed.plan = [
            {"index": 0, "expression": "", "scenes": [], "beats": []},
            {"index": 1, "expression": "", "scenes": [
                {"type": "figure", "at": "palabra10", "query": "q", "seconds": 5},
            ], "beats": []},
            {"index": 2, "expression": "", "scenes": [], "beats": []},
        ]
        with patch.object(editor.web_images, "find_candidates") as find:
            ed._prepare()
        find.assert_not_called()
        self.assertEqual(ed._scenes[1], [])

        drawn = self._editor(illustrations="ai")
        drawn.plan = [
            {"index": 0, "expression": "", "scenes": [], "beats": []},
            {"index": 1, "expression": "", "scenes": [
                {"type": "story", "items": [{"at": "palabra10", "label": "a", "draw": "otter plugs a lamp"}, {"at": "palabra16", "label": "b", "draw": "the lamp lights up"}]},
            ], "beats": []},
            {"index": 2, "expression": "", "scenes": [], "beats": []},
        ]
        frames = [_icon(self.path("f1.png")), _icon(self.path("f2.png"))]
        with patch.object(editor.gemini_media, "illustrate_sequence", return_value=frames) as sequence, patch.object(
            editor.gemini_media, "illustrate", return_value=""
        ), patch.object(editor.icons, "fetch", return_value=""):
            drawn._prepare()
        sequence.assert_called_once_with(["otter plugs a lamp", "the lamp lights up"])
        story = drawn._scenes[1][0]
        self.assertTrue(all(drawn._scene_picture(item).info.get("scene") for item in story.items))


class TestRound3Cli(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.script = os.path.join(self.temp_dir, "s.json")
        Path(self.script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")

    def test_progress_bar_is_opt_in(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, redirect_stdout(stdout), redirect_stderr(stderr):
            list_video_cli.run(["--script", self.script])
            self.assertFalse(generate.call_args.kwargs["edit"].progress_bar)
            list_video_cli.run(["--script", self.script, "--progress-bar"])
            self.assertTrue(generate.call_args.kwargs["edit"].progress_bar)


if __name__ == "__main__":
    unittest.main()
