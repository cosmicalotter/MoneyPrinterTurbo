import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript, VideoParams
from app.services import list_video, llm
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_host as host
from app.utils import utils

FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))
NAMES = ["explicando", "feliz", "riendo", "saludando", "senalando", "sorprendido"]


def _info(kind="item", duration=20.0, **kwargs):
    kwargs.setdefault("expression", "explicando")
    return host.SegmentInfo(kind=kind, duration=duration, **kwargs)


def _eval_offset(expression: str, t: float) -> float:
    """Evaluate the ffmpeg position expression in Python."""
    code = re.sub(r"\bt\b", repr(t), expression)
    return eval(  # noqa: S307 - expression built by the code under test
        code,
        {
            "between": lambda x, a, b: 1.0 if a <= x <= b else 0.0,
            "clip": lambda x, a, b: min(b, max(a, x)),
            "pow": math.pow,
        },
    )


class TestHostPlan(unittest.TestCase):
    def test_intro_waves_then_items_take_turns(self):
        infos = [_info("intro", 8.0, expression="feliz")]
        infos += [_info(duration=20.0, pauses=[8.0, 14.0]) for _ in range(6)]
        infos += [_info("outro", 8.0, expression="feliz")]
        plan = host.plan_host(infos, NAMES, seed=0)
        self.assertEqual(
            [p.mode for p in plan], ["full", "lead", "react", "lead+react", "off", "react", "full", "full"]
        )
        intro = plan[0]
        self.assertEqual((intro.cues[0].expression, intro.cues[0].bounce), ("saludando", False))
        self.assertEqual((intro.cues[1].expression, intro.cues[1].bounce), ("feliz", True))
        self.assertEqual(intro.windows[0].start, 0.3)
        # The host stays on stage across the cut into the first item.
        self.assertFalse(intro.windows[0].exit)
        first = plan[1]
        self.assertFalse(first.windows[0].enter)
        self.assertTrue(first.cues[0].bounce)  # a hop for the new item
        # ...and leaves on a pause instead of mid-word.
        self.assertAlmostEqual(first.windows[0].end, 8.45)
        self.assertTrue(first.windows[0].exit)
        self.assertEqual(plan[4].windows, [])
        # The outro ends with a goodbye wave and no exit (the video fades out).
        outro = plan[-1]
        self.assertEqual(outro.cues[-1].expression, "saludando")
        self.assertFalse(outro.windows[-1].exit)
        on_screen = sum(w.end - w.start for p in plan for w in p.windows)
        self.assertLess(on_screen, 0.75 * sum(i.duration for i in infos))

    def test_reactions_drop_in_and_pictures_get_pointed_at(self):
        info = _info(duration=24.0, reactions=[(15.0, "sorprendido")], pictures=[2.0], pauses=[7.0, 19.0])
        plan = host.plan_host([info], NAMES, seed=0)[0]
        self.assertEqual(plan.mode, "lead")
        lead, visit = plan.windows
        self.assertEqual(lead.start, 0.3)  # the video's first segment
        self.assertAlmostEqual(visit.start, 14.65)
        self.assertTrue(visit.enter and visit.exit)
        faces = [(round(c.time, 2), c.expression, c.bounce) for c in plan.cues]
        self.assertIn((2.0, "senalando", True), faces)
        self.assertIn((4.0, "explicando", True), faces)
        # The visit opens with the reaction face; entering already moves it.
        self.assertIn((14.65, "sorprendido", False), faces)

        # A reaction inside a visit is a bouncy change and returns to the base face.
        info = _info(duration=12.0, reactions=[(5.0, "riendo")], mode="full")
        cues = host.plan_host([info], NAMES)[0].cues
        self.assertEqual([(c.time, c.expression) for c in cues], [(0.3, "explicando"), (5.0, "riendo"), (7.6, "explicando")])

    def test_cameo_modes_overrides_and_switches(self):
        cameo = host.plan_host([_info(duration=20.0, pictures=[12.0], mode="react")], NAMES)[0]
        self.assertEqual(cameo.cues[0].expression, "senalando")
        self.assertEqual(len(cameo.windows), 1)
        quiet = host.plan_host([_info(duration=20.0, mode="off")], NAMES)[0]
        self.assertEqual(quiet.windows, [])
        short = host.plan_host([_info(duration=4.0, mode="lead")], NAMES)[0]
        self.assertEqual(short.mode, "full")
        always = host.plan_host([_info(duration=20.0, mode="off")], NAMES, host_mode="always")[0]
        self.assertEqual(always.mode, "full")
        self.assertEqual(host.plan_host([_info()], NAMES, host_mode="none")[0].windows, [])
        self.assertEqual(host.plan_host([_info()], [])[0].windows, [])
        # Without wave/point poses the host just keeps its base face.
        plain = host.plan_host([_info("intro", 8.0)], ["explicando"])[0]
        self.assertEqual([c.expression for c in plain.cues], ["explicando"])

    def test_pauses_and_bounce_shape(self):
        pcm = np.zeros(24000 * 3, dtype=np.int16)
        t = np.arange(24000) / 24000
        voice = (8000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        pcm[:24000] = voice
        pcm[48000:] = voice
        self.assertEqual(host.find_pauses(pcm.tobytes()), [1.0])
        self.assertEqual(host.find_pauses(b""), [])

        first = host.bounce_shape(0)
        middle = host.bounce_shape(3)
        last = host.bounce_shape(host.BOUNCE_FRAMES)
        self.assertLess(first[1], 0.95)  # squashed ...
        self.assertGreater(first[0], 1.0)
        self.assertGreater(middle[1], 1.0)  # ... then stretched and lifted
        self.assertGreater(middle[2], 0.0)
        self.assertAlmostEqual(last[1], 1.0, places=2)

    def test_entrance_and_exit_positions(self):
        windows = [host.HostWindow(1.0, 5.0), host.HostWindow(8.0, 12.0, enter=False, exit=False)]
        expression = host.position_offset(windows)
        self.assertAlmostEqual(_eval_offset(expression, 1.0), 1.0)
        self.assertLess(_eval_offset(expression, 1.3), 0.0)  # overshoot above its spot
        self.assertAlmostEqual(_eval_offset(expression, 2.0), 0.0)
        self.assertLess(_eval_offset(expression, 4.65), 0.0)  # anticipation before diving
        self.assertAlmostEqual(_eval_offset(expression, 5.0), 1.0)
        self.assertAlmostEqual(_eval_offset(expression, 9.0), 0.0)
        self.assertEqual(host.position_offset([host.HostWindow(0, 2, False, False)]), "0")


class TestHostRenderer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def _pose(self, name, size, box, color):
        image = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle(box, fill=color)
        path = os.path.join(self.temp_dir, name)
        image.save(path)
        return path

    def test_poses_on_one_canvas_keep_their_place(self):
        poses = {
            "a": fx.CharacterPose(self._pose("a.png", (400, 400), (100, 100, 200, 399), (255, 0, 0, 255))),
            "b": fx.CharacterPose(
                self._pose("b.png", (400, 400), (100, 50, 300, 399), (0, 255, 0, 255)),
                self._pose("b_habla.png", (400, 400), (100, 50, 300, 399), (0, 0, 255, 255)),
            ),
        }
        renderer = host.HostRenderer(poses, 175, self.temp_dir)
        self.assertEqual(renderer.char_size, (100, 175))  # union box 201x350 scaled to 175
        with Image.open(renderer.frame("a", True)) as frame:  # no talking pose: idle
            self.assertEqual(frame.size, renderer.canvas)
            box = frame.getchannel("A").getbbox()
        with Image.open(renderer.frame("b", True)) as frame:
            self.assertEqual(frame.getpixel((frame.width // 2, frame.height - 2))[:3], (0, 0, 255))
            wide = frame.getchannel("A").getbbox()
        self.assertEqual(box[0], wide[0])  # same left edge: the body did not move
        with Image.open(renderer.frame("b", False, bounce=0)) as squashed:
            squashed_box = squashed.getchannel("A").getbbox()
        self.assertLess(squashed_box[3] - squashed_box[1], wide[3] - wide[1])
        self.assertEqual(renderer.frame("b", False, bounce=0), renderer.frame("b", False, bounce=0))
        with Image.open(renderer.hidden()) as hidden:
            self.assertEqual(hidden.getchannel("A").getbbox(), None)

    def test_separate_drawings_line_up_on_the_body(self):
        poses = {
            "a": fx.CharacterPose(self._pose("a.png", (100, 200), (0, 0, 99, 199), (255, 0, 0, 255))),
            "b": fx.CharacterPose(self._pose("b.png", (300, 220), (0, 0, 299, 219), (0, 255, 0, 255))),
        }
        renderer = host.HostRenderer(poses, 100, self.temp_dir)
        self.assertEqual(renderer.char_size[1], 100)
        self.assertEqual(renderer.images[("a", False)].size, renderer.images[("b", False)].size)

    def test_segment_track_shows_cues_bounces_and_lip_flap(self):
        poses = {
            name: fx.CharacterPose(
                self._pose(f"{name}.png", (60, 60), (10, 10, 50, 59), (200, 100, 50, 255)),
                self._pose(f"{name}_habla.png", (60, 60), (10, 10, 50, 59), (200, 100, 60, 255)),
            )
            for name in ("explicando", "riendo")
        }
        renderer = host.HostRenderer(poses, 60, self.temp_dir)
        segment = host.HostSegment(
            "lead",
            [host.HostWindow(0.5, 2.0)],
            [host.HostCue(0.5, "explicando", False), host.HostCue(1.0, "riendo", True)],
        )
        frames = host.segment_frames(segment, renderer, 75, [(True, 20), (False, 20), (True, 10)])
        self.assertEqual(len(frames), 75)
        names = [os.path.basename(f) for f in frames]
        self.assertTrue(all(n == "hidden.png" for n in names[:15] + names[60:]))
        self.assertIn("talk-still", names[15])
        self.assertIn("idle-still", names[25])
        self.assertTrue(names[30].startswith("riendo") and names[30].endswith("idle-b00.png"))
        self.assertTrue(names[40].endswith("talk-still.png"))
        track = host.write_concat(frames, os.path.join(renderer.work_dir, "t.txt"))
        lines = Path(track).read_text().splitlines()
        total = sum(float(line.split()[1]) for line in lines if line.startswith("duration"))
        self.assertAlmostEqual(total, 75 / 30, places=4)
        self.assertEqual(lines[-1], f"file '{names[-1]}'")


class TestBundledOtter(unittest.TestCase):
    def test_otter_has_every_expression_with_a_talking_frame(self):
        poses = fx.load_character(NUTRIA)
        expected = {
            "neutral", "feliz", "explicando", "senalando", "saludando", "sorprendido",
            "sin_palabras", "pensando", "triste", "preocupado", "emocionado", "riendo",
        }
        self.assertEqual(set(poses), expected)
        self.assertTrue(all(pose.talk for pose in poses.values()))
        sizes = set()
        for pose in poses.values():
            for path in (pose.idle, pose.talk):
                with Image.open(path) as image:
                    sizes.add(image.size)
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.getpixel((0, 0))[3], 0)
        self.assertEqual(len(sizes), 1)  # one canvas, so poses line up
        self.assertEqual(host.pick_name(list(poses), host.WAVE_NAMES), "saludando")
        self.assertEqual(host.pick_name(list(poses), host.POINT_NAMES), "senalando")
        sources = os.listdir(os.path.join(NUTRIA, "svg"))
        self.assertEqual(len(sources), 2 * len(expected))

    def test_bloop_sound(self):
        sound = fx.synthesize_sfx("bloop")
        self.assertGreater(sound.size, 1000)
        self.assertLessEqual(float(np.abs(sound).max()), 1.0)


class TestEditPlanExtras(unittest.TestCase):
    def test_reactions_backgrounds_and_host_are_validated(self):
        data = {"segments": [{
            "index": 0, "expression": "feliz", "host": "OFF",
            "backgrounds": [{"at": "a b", "query": "city night"}, {"at": "", "query": "x"}, "bad", {"at": "c", "query": ""}],
            "beats": [
                {"type": "react", "at": "x y", "expression": "RIENDO"},
                {"type": "react", "at": "x y", "expression": "unknown"},
                {"type": "react", "at": "", "expression": "riendo"},
            ],
        }, {"index": 1, "backgrounds": "nope", "host": "sometimes"}]}
        plan = llm.normalize_edit_plan(data, 2, ["feliz", "riendo"])
        self.assertEqual(plan[0]["backgrounds"], [{"at": "a b", "query": "city night"}])
        self.assertEqual(plan[0]["beats"], [{"type": "react", "at": "x y", "expression": "riendo"}])
        self.assertEqual(plan[0]["host"], "off")
        self.assertEqual(plan[1]["backgrounds"], [])
        self.assertNotIn("host", plan[1])

    def test_prompt_mentions_backgrounds_and_reactions(self):
        segments = [{"index": 0, "kind": "item", "title": "1. A", "text": "hola mundo"}]
        prompt = llm.build_edit_plan_prompt(segments, ["feliz", "riendo"], "es-CO")
        self.assertIn('"backgrounds"', prompt)
        self.assertIn('"type": "react"', prompt)
        self.assertIn("never use react beats", llm.build_edit_plan_prompt(segments, [], ""))


def _segments():
    script = ListVideoScript(
        title="T",
        intro="Hola nutrias. Hoy vemos algo increíble.",
        items=[ListVideoItem(name="Agua", text=" ".join(f"palabra{i}" for i in range(60)))],
        outro="Chao.",
    )
    return list_video.build_segments(script)


class TestEditorHostAndScenes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def _editor(self, **options):
        segments = _segments()
        narrations = [
            editor.Narration(pcm=b"", speech_seconds=6.0, frames=200),
            editor.Narration(pcm=b"", speech_seconds=24.0, frames=730),
            editor.Narration(pcm=b"", speech_seconds=2.0, frames=72),
        ]
        theme = fx.Theme(1920, 1080, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        return editor.Editor(editor.EditOptions(**options), theme, self.temp_dir, segments, narrations)

    def test_background_shots_follow_the_anchors(self):
        ed = self._editor(subscribe="none")
        ed.plan = [
            {"index": 0, "expression": "", "backgrounds": [{"at": "no aparece", "query": "first"}], "beats": []},
            {"index": 1, "expression": "", "beats": [], "backgrounds": [
                {"at": "palabra0", "query": "a"},
                {"at": "palabra20", "query": "b"},
                {"at": "palabra22", "query": "too close"},
                {"at": "missing words", "query": "lost"},
                {"at": "palabra59", "query": "too late"},
                {"at": "palabra40", "query": "c"},
            ]},
            {"index": 2, "expression": "", "beats": []},
        ]
        self.assertEqual(ed.background_shots(0), [(0.0, "first")])
        shots = ed.background_shots(1)
        self.assertEqual([q for _, q in shots], ["a", "b", "c"])
        self.assertEqual(shots[0][0], 0.0)
        self.assertTrue(7.0 < shots[1][0] < 9.0)
        self.assertEqual(ed.background_shots(2), [])

    def test_host_overlay_with_the_otter(self):
        ed = self._editor(assets_dir=NUTRIA, subscribe="none", progress_bar=False)
        with patch.object(editor.llm, "generate_edit_plan", return_value=None):
            ed.make_plan()
        intro = ed.segment_edit(0, 0.0, show_titles=True)
        overlay = intro.overlays[-1]
        self.assertEqual(overlay.mode, "concat")
        self.assertIn("pow(", overlay.y)
        names = [os.path.basename(path) for _, path, _ in intro.sounds]
        self.assertIn("sfx-bloop.wav", names)
        canvas_w, canvas_h = ed.renderer().canvas
        self.assertGreater(canvas_h, 1080 * editor.HOST_HEIGHT[False])
        track = Path(overlay.source).read_text()
        self.assertIn("saludando", track)
        saved = json.loads(Path(self.temp_dir, "edit-plan.json").read_text("utf-8"))
        self.assertEqual([s["host"] for s in saved["segments"]], ["full", "lead", "full"])

        hidden = self._editor(assets_dir=NUTRIA, subscribe="none", progress_bar=False, host="none")
        self.assertEqual(hidden.poses, {})


class TestScenes(unittest.TestCase):
    def test_scenes_are_cut_into_even_shots(self):
        clips_a = [("a1", 8.0), ("a2", 8.0), ("a3", 8.0)]
        clips_b = [("b1", 3.0), ("b2", 10.0)]
        pieces = list_video.plan_scenes([(0.0, clips_a), (12.0, clips_b)], 18.0, 5)
        self.assertEqual([p for p, _ in pieces], ["a1", "a2", "a3", "b2"])
        self.assertTrue(all(abs(take - 4.0) < 1e-6 for _, take in pieces[:3]))
        self.assertAlmostEqual(sum(take for _, take in pieces), 18.0)
        # Short clips are chained so the scene still lasts until its end.
        pieces = list_video.plan_scenes([(0.0, [("s", 2.0)])], 5.0, 5)
        self.assertAlmostEqual(sum(take for _, take in pieces), 5.0)
        self.assertEqual(list_video.plan_scenes([(0.0, [])], 5.0, 5), [])

    def test_each_scene_downloads_its_own_footage(self):
        segment = list_video.build_segments(
            ListVideoScript(title="T", items=[ListVideoItem(name="A", text="B", image_term="term")])
        )[0]
        params = VideoParams(video_subject="t", video_source="pexels", video_clip_duration=5)
        calls = []

        def fake_download(task_id, search_terms, audio_duration, **kwargs):
            calls.append((search_terms[0], round(audio_duration, 1)))
            return {"city": ["c1", "c2"], "term": ["t1"]}.get(search_terms[0], [])

        with patch.object(list_video.material, "download_videos", side_effect=fake_download), patch.object(
            list_video.task_artifacts, "patch_script_data"
        ), patch.object(list_video, "_read_material_sources", return_value=[{"id": 1}]), patch.object(
            list_video, "_probe_duration", return_value=6.0
        ):
            sources = []
            visual = list_video._prepare_visual(
                "task", segment, params, 20.0, sources, [], shots=[(0.0, "city"), (9.0, "nothing")]
            )
        self.assertEqual(calls, [("city", 9.0), ("nothing", 11.0)])
        self.assertEqual(visual.kind, "videos")
        self.assertAlmostEqual(sum(take for _, take in visual.pieces), 20.0)
        # The empty scene reuses the previous footage.
        self.assertTrue(all(path in ("c1", "c2") for path, _ in visual.pieces))
        self.assertEqual(len(sources), 2)

        with patch.object(list_video.material, "download_videos", side_effect=fake_download), patch.object(
            list_video.task_artifacts, "patch_script_data"
        ), patch.object(list_video, "_read_material_sources", return_value=[]), patch.object(
            list_video, "_probe_duration", return_value=6.0
        ):
            calls.clear()
            visual = list_video._prepare_visual("task", segment, params, 20.0, [], [], shots=[(0.0, "nothing")])
        # No footage for any scene: the item's own term is used.
        self.assertEqual([c[0] for c in calls], ["nothing", "term"])
        self.assertEqual(visual.paths, ["t1"])


class TestHostCli(unittest.TestCase):
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

    def test_bundled_otter_keyword_and_host_option(self):
        with patch.object(list_video, "generate_list_video", return_value={}) as generate:
            code, _, _ = self._run(["--script", self.script, "--assets", "nutria", "--host", "always"])
        self.assertEqual(code, 0)
        options = generate.call_args.kwargs["edit"]
        self.assertEqual(options.assets_dir, os.path.abspath(NUTRIA))
        self.assertEqual(options.host, "always")
        self.assertEqual(list_video_cli.built_in_assets_dir("../nutria"), "")
        self.assertEqual(list_video_cli.built_in_assets_dir("missing"), "")
        code, _, stderr = self._run(["--script", self.script, "--host", "sometimes"])
        self.assertEqual(code, 2)
        self.assertIn("--host", stderr)


if __name__ == "__main__":
    unittest.main()
