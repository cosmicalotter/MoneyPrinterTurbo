"""Round 7: almost everything drawn by AI as short animations, a smooth camera, longer
compositions that always show pictures, audited drawings and a render report."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import ListVideoItem, ListVideoScript, MaterialInfo
from app.services import gemini_media, list_video, llm, studio
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_scenes as scenes
from app.utils import utils

FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))


def _picture(path, color=(220, 60, 60), size=(320, 180)):
    image = Image.new("RGB", size, color)
    ImageDraw.Draw(image).ellipse((40, 30, 160, 150), fill=(250, 240, 200), outline=(0, 0, 0), width=6)
    image.save(path)
    return path


def _icon(path, color=(220, 60, 60, 255), size=120):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, size - 10, size - 10), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


def _frames(video):
    """Every frame of a video as an array (frames, height, width, 3)."""
    probe = subprocess.run([utils.get_ffmpeg_binary(), "-i", video], capture_output=True, text=True).stderr
    width, height = (int(v) for v in re.search(r", (\d+)x(\d+)", probe).groups())
    raw = subprocess.run([utils.get_ffmpeg_binary(), "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class TestSmoothCamera(_TempDirCase):
    def test_camera_moves_are_eased_and_gentle(self):
        self.assertEqual(scenes.ease_in_out(0), 0.0)
        self.assertAlmostEqual(scenes.ease_in_out(1), 1.0)
        self.assertAlmostEqual(scenes.ease_in_out(0.5), 0.5)
        self.assertLess(scenes.ease_in_out(0.1), 0.1)  # starts slowly
        for name in scenes.CAMERA_MOVES:
            path = scenes.camera_path(name)
            (zoom_start, pan_start), (zoom_end, pan_end) = path(0, 0.0), path(9, 1.0)
            self.assertLessEqual(max(zoom_start, zoom_end), 1.06)  # slow and small, never a big push
            self.assertGreaterEqual(min(zoom_start, zoom_end), 1.0)
            self.assertLessEqual(max(abs(pan_start), abs(pan_end)), 0.6)
        self.assertGreater(scenes.camera_path("right")(0, 1.0)[1], scenes.camera_path("right")(0, 0.0)[1])

    def test_motion_clip_plays_drawings_as_an_animation(self):
        red, blue = _picture(self.path("r.png")), _picture(self.path("b.png"), color=(40, 60, 220))
        output = scenes.motion_clip(
            [(Image.open(red), 0.0), (Image.open(blue), 1.0)], self.path("anim.mp4"), 60, (160, 90),
            path=scenes.camera_path("left"), fade=0.2,
        )
        video = _frames(output)
        self.assertEqual(len(video), 60)
        corner = video[:, 4, 4].astype(int)
        self.assertGreater(corner[10][0], corner[10][2])  # red first
        self.assertGreater(corner[50][2], corner[50][0])  # blue after a second
        middle = corner[33]  # half-way through the dissolve
        self.assertTrue(abs(middle[0] - middle[2]) < abs(corner[10][0] - corner[10][2]))
        with self.assertRaises(ValueError):
            scenes.motion_clip([], self.path("none.mp4"), 10, (160, 90))

    def test_white_pages_are_trimmed_but_scenes_and_objects_are_not(self):
        page = Image.new("RGB", (400, 240), (252, 252, 250))
        page.paste(Image.new("RGB", (300, 170), (30, 40, 70)), (40, 30))
        self.assertEqual(scenes.trim_border(page).size, (300, 170))
        dark = Image.new("RGB", (400, 240), (20, 20, 30))
        dark.paste(Image.new("RGB", (100, 100), (200, 100, 50)), (150, 70))
        self.assertEqual(scenes.trim_border(dark).size, (400, 240))  # a dark scene is never cut
        drawing = Image.new("RGB", (400, 400), "white")
        drawing.paste(Image.new("RGB", (60, 60), "red"), (170, 170))
        self.assertEqual(scenes.trim_border(drawing).size, (400, 400))  # one object on white stays whole


class TestAnimationShots(unittest.TestCase):
    TIMES = {"uno": 1.0, "dos": 2.5, "tres": 3.4, "cuatro": 9.0, "cinco": 9.5, "seis": 14.0}

    def test_normalize(self):
        data = {"segments": [{"index": 0, "shots": [
            {"type": "animation", "at": "uno", "otter": True, "camera": "right", "continue": True, "query": "light switch",
             "frames": [{"draw": "the otter reaches for the switch"}, {"draw": "it flips it", "at": "dos"}, "the bulb glows", {"at": "x"}]},
            {"type": "animation", "at": "dos", "frames": [{"draw": "a lone frame"}]},
            {"type": "animation", "at": "tres", "frames": []},
            {"type": "stat", "at": "cuatro", "value": 120, "unit": "V", "label": "enchufe", "draw": "a wall socket", "query": "wall socket"},
        ]}]}
        shots = llm.normalize_storyboard(data, 1, [])[0]["shots"]
        self.assertEqual([s["type"] for s in shots], ["animation", "illustration", "stat"])
        animation = shots[0]
        self.assertEqual([f["draw"] for f in animation["frames"]], ["the otter reaches for the switch", "it flips it", "the bulb glows"])
        self.assertEqual(animation["frames"][1]["at"], "dos")
        self.assertEqual((animation["otter"], animation["camera"], animation["continue"], animation["query"]),
                         (True, "right", True, "light switch"))
        self.assertEqual(shots[1]["draw"], "a lone frame")  # one frame is just an illustration
        self.assertEqual((shots[2]["draw"], shots[2]["query"]), ("a wall socket", "wall socket"))

    def test_frames_last_about_a_second_and_follow_their_words(self):
        spec = {"type": "animation", "at": "uno", "camera": "in", "otter": True, "query": "switch", "frames": [
            {"draw": "a"}, {"draw": "b", "at": "tres"}, {"draw": "c"}, {"draw": "d"}]}
        shot, = scenes.time_shots([spec], self.TIMES.get, 6.0)
        times = [frame.time for frame in shot.items]
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(times[1], 3.4)  # on its own words
        self.assertTrue(all(b - a >= scenes.ANIMATION_MIN_FRAME - 1e-6 for a, b in zip(times, times[1:])))
        self.assertTrue(all(frame.otter for frame in shot.items))
        self.assertEqual(shot.center.query, "switch")
        short, = scenes.time_shots([dict(spec, frames=[{"draw": str(n)} for n in range(6)])], self.TIMES.get, 4.0)
        self.assertLessEqual(len(short.items), 3)  # 4 s leave room for three drawings, not six
        self.assertTrue(all(f.time <= 4.0 - scenes.ANIMATION_MIN_FRAME + 1e-6 for f in short.items[1:]))
        gaps = [b.time - a.time for a, b in zip(short.items, short.items[1:])]
        self.assertTrue(all(gap >= scenes.ANIMATION_MIN_FRAME - 1e-6 for gap in gaps))

    def test_compositions_stay_long_enough_to_follow(self):
        times = {"uno": 1.0, "dos": 2.5, "tres": 4.0, "cinco": 11.0, "seis": 16.0}
        specs = [
            {"type": "single", "at": "uno", "label": "a", "draw": "x"},
            {"type": "illustration", "at": "tres", "draw": "y"},  # 3 s after the single: waits until it is read
            {"type": "illustration", "at": "cinco", "draw": "z"},
            {"type": "compare", "items": [{"at": "seis", "label": "p"}, {"at": "seis", "label": "q"}]},
        ]
        shots = scenes.time_shots(specs, times.get, 22.0)
        self.assertEqual([s.type for s in shots], ["single", "illustration", "illustration", "compare"])
        self.assertAlmostEqual(shots[1].start, 0.85 + scenes.COMPOSITION_SECONDS)
        far = scenes.time_shots([specs[0], {"type": "illustration", "at": "dos", "draw": "y"},
                                 {"type": "illustration", "at": "cinco", "draw": "w"}], times.get, 22.0)
        self.assertEqual([s.center.draw for s in far], ["x", "w"])  # "y" would have waited too long: left out
        holds = scenes.long_holds(shots, 22.0, 6.0)
        self.assertTrue(all(b - a > 6.0 for a, b in holds))


class TestRendererRules(_TempDirCase):
    def _renderer(self, picture, accent=(253, 199, 76)):
        theme = fx.Theme(320, 180, FONT, accent)
        host = lambda expression, height: Image.new("RGBA", (height // 2, height), (40, 160, 140, 255))  # noqa: E731
        return scenes.SceneRenderer(theme, self.path("r"), editor.DOODLE_COLOR, picture, host, {},
                                    font_path=scenes.hand_font_path(), doodle=True)

    def test_brand_text_stays_readable_on_the_canvas(self):
        self.assertLess(scenes.contrast((253, 199, 76), editor.DOODLE_COLOR), 1.2)
        renderer = self._renderer(lambda item: None)
        self.assertGreaterEqual(scenes.contrast(renderer.accent_text, editor.DOODLE_COLOR), 2.6)
        self.assertEqual(scenes.readable((27, 27, 47), (255, 255, 255)), (27, 27, 47))
        self.assertAlmostEqual(scenes.contrast((0, 0, 0), (255, 255, 255)), 21.0, places=1)

    def test_animation_and_formula_scenes(self):
        frame_a, frame_b = _picture(self.path("a.png")), _picture(self.path("b.png"), color=(30, 30, 200))
        pictures = {"a": scenes.prepare_picture(frame_a, allow_cutout=False), "b": scenes.prepare_picture(frame_b, allow_cutout=False)}
        renderer = self._renderer(lambda item: pictures.get(item.draw))
        seen = []
        real = scenes.motion_clip

        def spy(pictures, output, frames, size, path=None, fade=0.2, trim=False):
            seen.append(([round(at, 2) for _, at in pictures], fade, trim))
            return real(pictures, output, frames, size, path=path, fade=fade, trim=trim)

        Item = scenes.SceneItem
        shot = scenes.Scene("animation", 2.0, 4.0, items=[Item(draw="a", time=2.0), Item(draw="missing", time=2.6), Item(draw="b", time=3.0)],
                            text="05:30")
        with patch.object(scenes, "motion_clip", side_effect=spy):
            overlays, sounds = renderer.build(shot, "anim")
        self.assertEqual(seen[0], ([0.0, 1.0], scenes.ANIMATION_FADE, True))  # the missing frame is skipped
        self.assertEqual(overlays[0].mode, "media")
        self.assertEqual(len(overlays), 2)  # the animation and its caption
        with self.assertRaises(ValueError):
            renderer.build(scenes.Scene("animation", 0, 2, items=[Item(draw="missing")]), "none")
        equation = scenes.Scene("equation", 0, 4, text="V = I × R", center=Item(label="Ley de Ohm"),
                                items=[Item(symbol="V", label="voltaje", time=1.0)])
        overlays, _ = renderer.build(equation, "eq")
        self.assertTrue(any(os.path.basename(o.source).startswith("host") for o in overlays))  # never a formula alone


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


class TestPicturesAlways(_TempDirCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def _editor(self, **options):
        patcher = patch.object(editor.gemini_media, "enabled", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        segments = list_video.build_segments(ListVideoScript(
            title="T", intro="Hola nutrias.", items=[ListVideoItem(name="Voltaje", text=self.TEXT, image_term="battery")], outro="Chao."))
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(640, 360, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        options.setdefault("look", "doodle")
        options.setdefault("openers", False)
        return editor.Editor(editor.EditOptions(assets_dir=NUTRIA, **options), theme, self.temp_dir, segments, narrations)

    def test_animation_frames_follow_each_other(self):
        ed = self._editor()
        asked = []

        def draw(description, scene=False, mascot="", previous="", **kwargs):
            asked.append((description, os.path.basename(previous), bool(mascot)))
            return "" if description == "it fails" else _picture(self.path(f"d{len(asked)}.png"))

        Item = scenes.SceneItem
        first = scenes.Scene("animation", 0, 3, items=[Item(draw="the otter in the dark", otter=True), Item(draw="it fails"),
                                                       Item(draw="the light is on", otter=True)])
        second = scenes.Scene("animation", 3, 5, follows=True, items=[Item(draw="it smiles", otter=True)])
        with patch.object(editor.gemini_media, "draw", side_effect=draw), \
                patch.object(editor.gemini_media, "audit_drawing", return_value=(True, "")):
            ed._illustration_chain([(first, ""), (second, "")])
        self.assertEqual(asked[0], ("the otter in the dark", "", True))
        self.assertEqual(asked[1][1], "d1.png")
        self.assertEqual(asked[2], ("the light is on", "d1.png", False))  # the failed frame is skipped
        self.assertEqual(asked[3], ("it smiles", "d3.png", False))  # the next shot continues the last drawing
        self.assertTrue(ed._has_picture(first))
        self.assertIsNone(ed._item_pictures.get(id(first.items[1])))

    def test_drawings_are_audited_and_redrawn_once(self):
        ed = self._editor()
        calls = []

        def draw(description, **kwargs):
            calls.append(description)
            return _picture(self.path(f"a{len(calls)}.png"))

        with patch.object(editor.gemini_media, "draw", side_effect=draw), \
                patch.object(editor.gemini_media, "audit_drawing", side_effect=[(False, ""), (True, "")]) as check:
            image = ed._drawing("a copper wire", scene=True)
        self.assertIsNotNone(image)
        self.assertEqual(len(calls), 2)
        self.assertIn("drawn again", calls[1])
        self.assertEqual(image.info["source"], self.path("a2.png"))
        self.assertEqual(check.call_count, 1)  # the redraw is used as it is (one extra drawing at most)
        self.assertEqual((ed._stats["rejected"], ed._stats["drawn"], ed._drawings), (1, 1, 2))

    def test_missing_drawings_become_real_photos_or_leave(self):
        ed = self._editor()
        Item = scenes.SceneItem
        scene = scenes.Scene("animation", 0, 3, center=Item(query="light switch"), items=[Item(draw="x"), Item(draw="y")])
        empty = scenes.Scene("illustration", 3, 5, center=Item(draw="a lab at night"))
        ed._scenes = {1: [scene, empty]}
        ed._unpictured = [scene, empty]
        photo = _picture(self.path("photo.png"))
        with patch.object(editor.gemini_media, "draw", return_value=""), \
                patch.object(ed, "_web_picture", side_effect=[photo, ""]) as web:
            ed._rescue({1: self.TEXT})
        self.assertEqual(web.call_args_list[0].args[:2], ("light switch", "photo"))
        self.assertEqual(web.call_args_list[1].args[0], "a lab at night")
        self.assertEqual(scene.type, "illustration")  # shown as a full-screen photo with the camera move
        self.assertTrue(ed._has_picture(scene))
        self.assertFalse(ed._has_picture(empty))
        self.assertEqual(ed._stats["photo"], 1)

    def test_compositions_never_show_text_alone(self):
        ed = self._editor()
        Item = scenes.SceneItem
        single = scenes.Scene("single", 0, 4, text="MEDICIÓN EN VOLTIOS", center=Item(label="voltios"))
        self.assertTrue(ed._complete(single))
        self.assertIsNotNone(ed._item_pictures[id(single.center)])  # the otter stands in
        a, b, c = Item(label="a", time=1.0), Item(label="b", time=2.0), Item(label="c", time=2.5)
        ed._item_pictures[id(b)] = ed._pose_picture("feliz")
        sequence = scenes.Scene("sequence", 0, 4, items=[a, b, c])
        self.assertTrue(ed._complete(sequence))
        self.assertEqual(sequence.items, [b])
        self.assertEqual(b.time, 1.0)  # the first element left still shows from the start
        compare = scenes.Scene("compare", 0, 4, items=[Item(label="x"), Item(label="y")])
        ed._item_pictures[id(compare.items[0])] = ed._pose_picture("feliz")
        self.assertFalse(ed._complete(compare))  # half a comparison is left out
        speech = scenes.Scene("speech", 0, 4, text="hola", items=[Item(label=""), Item(label="robot")])
        self.assertTrue(ed._complete(speech))
        self.assertEqual(len(speech.items), 1)
        self.assertTrue(ed._complete(scenes.Scene("illustration", 0, 2, center=Item())))
        self.assertEqual(ed._stats["items_dropped"], 3)

    def test_a_shot_that_fails_to_render_gets_the_otter(self):
        ed = self._editor()
        ed.plan = [{"expression": "", "shots": []}] * 3
        ed.host_plan = [editor.host.HostSegment("off")] * 3
        Item = scenes.SceneItem
        broken = scenes.Scene("clip", 0, 3, center=Item(label="central"), media="")
        ed._scenes = {0: [], 1: [broken], 2: []}
        ed._beats = {0: [], 1: [], 2: []}
        edit = ed.segment_edit(1, 0.0, show_titles=False)
        self.assertTrue(any("-otter" in o.source for o in edit.overlays))
        self.assertEqual(ed._stats["build_failed"], 1)
        self.assertTrue(any("otter stood in" in w for w in ed.warnings))

    def test_dark_clips_are_skipped(self):
        ed = self._editor()
        dark = self.path("dark.mp4")
        subprocess.run([utils.get_ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x080808:s=160x90:r=30",
                        "-t", "1.5", "-pix_fmt", "yuv420p", dark], check=True)
        found = [MaterialInfo(provider="pexels", url="https://v/dark", duration=8)]
        shot = scenes.Scene("clip", 0, 3, center=scenes.SceneItem(query="lightning", draw=""))
        with patch.object(editor.material, "search_videos_pexels", return_value=found), \
                patch.object(editor.material, "search_videos_pixabay", return_value=[]), \
                patch.object(editor.material, "save_video", return_value=dark), \
                patch.object(editor.gemini_media, "choose_picture") as check:
            ed._clip_media(shot, self.TEXT)
        check.assert_not_called()
        self.assertEqual(shot.media, "")
        self.assertLess(fx.brightness(fx.extract_frame(utils.get_ffmpeg_binary(), dark, self.path("f.jpg"), 0.5)), 20)
        self.assertEqual(fx.brightness(self.path("missing.jpg")), 255.0)

    def test_render_report(self):
        ed = self._editor(max_drawings=10)
        Item = scenes.SceneItem
        ed._scenes = {1: [scenes.Scene("animation", 0, 3, items=[Item(draw="a"), Item(draw="b")]),
                          scenes.Scene("illustration", 3, 6, center=Item())]}
        ed._stats.update(drawn=7, photo=1, otter=2)
        ed._failed_drawings = 3
        lines = ed.report()
        text = "\n".join(lines)
        for words in ("animation 1", "illustration 1", "Animation frames: 2", "7 drawn, 3 failed", "budget of 10",
                      "1 photos instead of drawings", "2 otter poses"):
            self.assertIn(words, text)
        path = list_video.write_render_report(self.temp_dir, lines, ["a warning"])
        content = Path(path).read_text(encoding="utf-8")
        self.assertIn("- a warning", content)
        self.assertIn("No warnings.", Path(list_video.write_render_report(self.temp_dir, [], [])).read_text(encoding="utf-8"))


class TestSturdierDrawing(_TempDirCase):
    def test_drawing_check(self):
        prompt = gemini_media.build_drawing_check_prompt("a copper wire", mascot=True)
        self.assertIn("a copper wire", prompt)
        self.assertIn("otter", prompt)
        self.assertNotIn("otter", gemini_media.build_drawing_check_prompt("a wire"))
        path = _picture(self.path("p.png"))
        with patch.object(gemini_media, "_ask", return_value='{"ok": false, "reason": "text in it"}'):
            self.assertFalse(gemini_media.check_drawing(path, "x", app_config={}))
        with patch.object(gemini_media, "_ask", return_value='{"ok": "yes"}'):
            self.assertTrue(gemini_media.check_drawing(path, "x", app_config={}))
        with patch.object(gemini_media, "_ask", return_value="no json"):
            self.assertIsNone(gemini_media.check_drawing(path, "x", app_config={}))
        with patch.object(gemini_media, "_ask", side_effect=RuntimeError("quota")):
            self.assertIsNone(gemini_media.check_drawing(path, "x", app_config={}))
        self.assertFalse(gemini_media.check_drawing(self.path("missing.png"), "x", app_config={}))

    def test_busy_models_wait_longer_and_fewer_draw_at_once(self):
        self.assertGreaterEqual(gemini_media.RETRY_SECONDS[-1], 30)
        self.assertLessEqual(gemini_media.IMAGE_SLOTS, 3)
        waits = []
        calls = []

        class Busy(Exception):
            code = 429

        class Client:
            def __init__(self, **kwargs):
                self.models = self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def generate_content(self, **kwargs):
                calls.append(1)
                raise Busy("429 RESOURCE_EXHAUSTED")

        with patch("google.genai.Client", Client), patch.object(gemini_media, "_client_kwargs", return_value={}), \
                patch.object(gemini_media.time, "sleep", side_effect=waits.append), \
                patch.dict(gemini_media._state, {"imagen_failed": "", "last_error": ""}):
            data = gemini_media._generate_image("gemini-2.5-flash-image", "p", "16:9", [], {}, "x")
        self.assertEqual(data, b"")
        self.assertEqual(len(calls), len(gemini_media.RETRY_SECONDS) + 1)
        self.assertEqual(len(waits), len(gemini_media.RETRY_SECONDS))
        self.assertGreater(waits[-1], waits[0])


class TestStudioRound7(_TempDirCase):
    def test_old_projects_lose_the_corner_badge(self):
        old = studio.RenderSettings.from_dict({"logo": "nutria", "max_drawings": 160, "aspect": "16:9"})
        self.assertEqual((old.logo, old.max_drawings, old.version), ("", 260, studio.SETTINGS_VERSION))
        kept = studio.RenderSettings.from_dict({"logo": "nutria", "max_drawings": 90, "version": 2})
        self.assertEqual((kept.logo, kept.max_drawings), ("nutria", 90))  # chosen after the change: kept
        self.assertEqual(studio.RenderSettings().logo, "")
        self.assertNotIn("--logo", studio.build_argv(studio.RenderSettings(), script_file="s.json"))

    def test_results_show_the_report(self):
        with patch.object(utils, "storage_dir", return_value=self.temp_dir):
            folder = Path(self.temp_dir, "tasks", "abc")
            folder.mkdir(parents=True)
            (folder / "render-report.txt").write_text("Shots: animation 3\n", encoding="utf-8")
            outputs = studio.task_outputs("abc")
        self.assertEqual(outputs["report"], "Shots: animation 3\n")
        self.assertIn('"animation"', json.dumps(llm.SHOT_TYPES))


if __name__ == "__main__":
    unittest.main()
