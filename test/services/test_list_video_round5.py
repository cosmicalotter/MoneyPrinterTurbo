import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from PIL import Image, ImageDraw

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript
from app.services import gemini_media, list_video, llm
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_scenes as scenes
from app.services import voice_polish as vp
from app.utils import utils

FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))


def _icon(path, color=(220, 60, 60, 255), size=120):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, size - 10, size - 10), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


def _speech(pattern, rate=24000, level=0.3):
    """Voiced bursts (seconds) and silences (negative seconds), like a voice."""
    pieces = []
    for seconds in pattern:
        n = int(abs(seconds) * rate)
        if seconds > 0:
            t = np.arange(n) / rate
            wave_ = level * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) + 0.05 * np.sin(2 * np.pi * 2400 * t)
            pieces.append(wave_)
        else:
            pieces.append(np.zeros(n))
    return (np.concatenate(pieces) * 32767).astype(np.int16)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class TestPauses(unittest.TestCase):
    def test_silences_and_stretch(self):
        samples = _speech([-0.3, 2.0, -0.3, 1.5, -0.1, 1.0, -0.2], rate=48000)
        gaps = vp.silences(samples, 48000, 0.12)
        self.assertEqual(len(gaps), 3)  # leading, the breath and the trailing one (the 0.1 s one is too short)
        longer, inserted = vp.stretch_pauses(samples, 48000, 0.6)
        self.assertEqual(len(inserted), 1)  # only the inner breath grows
        self.assertAlmostEqual(inserted[0][1], 0.3, delta=0.03)
        self.assertAlmostEqual(len(longer) / 48000 - len(samples) / 48000, 0.3, delta=0.03)
        self.assertAlmostEqual(vp.shift_time(3.0, inserted), 3.3, delta=0.03)
        self.assertEqual(vp.shift_time(1.0, inserted), 1.0)
        same, nothing = vp.stretch_pauses(samples, 48000, 0)
        self.assertIs(same, samples)
        self.assertEqual(nothing, [])
        self.assertEqual(len(vp.half_rate(samples)), len(samples) // 2)

    def test_sentences_are_found_from_the_breaths(self):
        text = "Primera frase corta. Esta segunda frase es bastante mas larga que la otra. Fin."
        # Spoken unevenly: the long sentence is said fast, so proportions would be wrong.
        samples = _speech([-0.2, 2.0, -0.5, 2.2, -0.15, 0.4, -0.6, 1.0, -0.2])
        spans = vp.align_sentences(text, samples, 24000)
        self.assertEqual(len(spans), 3)
        self.assertAlmostEqual(spans[0][0], 0.2, delta=0.05)
        self.assertAlmostEqual(spans[1][0], 2.7, delta=0.05)  # after the first breath
        self.assertAlmostEqual(spans[1][1], 5.45, delta=0.05)  # the short pause inside it is skipped
        self.assertAlmostEqual(spans[2][0], 6.05, delta=0.05)
        self.assertAlmostEqual(spans[2][1], 7.05, delta=0.05)
        # Too few pauses: proportional spans.
        flat = _speech([3.0])
        spans = vp.align_sentences(text, flat, 24000)
        self.assertEqual(len(spans), 3)
        self.assertAlmostEqual(spans[-1][1], 3.0, delta=0.02)
        self.assertEqual(vp.align_sentences("", flat, 24000), [])
        self.assertEqual(vp.align_sentences("Una sola.", flat, 24000), [(0.0, 3.0)])

    def test_anchor_times_follow_the_sentences(self):
        text = "Primera frase corta. Esta segunda frase es bastante mas larga que la otra. Fin."
        spans = [(0.2, 2.2), (2.7, 4.9), (5.45, 6.45)]
        self.assertAlmostEqual(editor.anchor_time(text, "Esta segunda", None, 6.6, spans), 2.7)
        self.assertAlmostEqual(editor.anchor_time(text, "Fin", None, 6.6, spans), 5.45)
        proportional = editor.anchor_time(text, "Fin", None, 6.6)
        self.assertGreater(proportional, 6.0)
        # Spans that do not match the sentences are ignored.
        self.assertAlmostEqual(editor.anchor_time(text, "Fin", None, 6.6, spans[:2]), proportional)


class TestMastering(_TempDirCase):
    def _wav(self, name, samples, rate):
        path = self.path(name)
        fx.write_wav(samples.astype(np.float32) / 32768, path, rate)
        return path

    def test_master_voice_reaches_the_target_without_clipping(self):
        source = self._wav("voice.wav", _speech([-0.2, 2.0, -0.4, 2.0, -0.2], level=0.08), 24000)
        out = vp.master_voice(source, self.path("master.wav"))
        with wave.open(out, "rb") as w:
            self.assertEqual((w.getframerate(), w.getnchannels()), (48000, 1))
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self.assertLess(np.abs(samples).max() / 32768, 0.9)
        measured = vp._measure(out, "")
        self.assertAlmostEqual(float(measured["input_i"]), -14.0, delta=1.5)
        plain = vp.master_voice(source, self.path("plain.wav"), polish=False)
        self.assertTrue(os.path.isfile(plain))

    def test_polish_adds_air_only_to_band_limited_voices(self):
        self.assertIn("aexciter", vp.polish_chain(True))
        self.assertNotIn("aexciter", vp.polish_chain(False))
        narrow = _speech([1.0], rate=48000)
        self.assertLess(vp.high_band_share(narrow, 48000), 0.002)
        noise = (np.random.default_rng(1).normal(0, 0.2, 48000) * 32767).astype(np.int16)
        self.assertGreater(vp.high_band_share(noise, 48000), 0.2)

    def test_soft_limit_leaves_the_voice_alone(self):
        mix = np.array([0.1, -0.5, 0.9, 1.4, -2.0])
        limited = vp.soft_limit(mix)
        np.testing.assert_allclose(limited[:3], mix[:3])
        self.assertTrue(np.all(np.abs(limited) < 1.0))
        self.assertGreater(limited[3], 0.92)


class TestShotTiming(unittest.TestCase):
    TIMES = {"uno": 1.0, "dos": 4.0, "tres": 4.8, "cuatro": 9.0, "cinco": 9.6, "seis": 15.0}

    def test_shots_follow_each_other_without_gaps(self):
        specs = [
            {"type": "single", "at": "uno", "label": "a", "draw": "x"},
            {"type": "sequence", "items": [{"at": "dos", "label": "b"}, {"at": "tres", "label": "c"}]},
            {"type": "single", "at": "cinco", "label": "too close"},  # 0.6 s after the next one
            {"type": "stat", "at": "cuatro", "value": 7, "label": "d"},
            {"type": "speech", "at": "seis", "text": "hola", "items": [{"label": "e"}, {"label": "f", "at": "seis"}]},
            {"type": "bogus", "at": "uno"},
            {"type": "single", "at": "missing", "label": "x"},
        ]
        shots = scenes.time_shots(specs, self.TIMES.get, 21.0, start=0.5)
        self.assertEqual([s.type for s in shots], ["single", "sequence", "stat", "speech"])
        self.assertEqual(shots[0].start, 0.5)
        for shot, following in zip(shots, shots[1:]):
            self.assertAlmostEqual(shot.end, following.start)
            self.assertGreaterEqual(shot.end - shot.start, scenes.COMPOSITION_SECONDS - 0.5)  # long enough to follow
        self.assertEqual(shots[-1].end, 21.0)
        self.assertTrue(all(not s.exit for s in shots))
        self.assertAlmostEqual(shots[1].start, 0.85 + scenes.COMPOSITION_SECONDS)  # waited for the single to be read
        self.assertEqual([i.label for i in shots[1].items], ["b", "c"])
        first, second = shots[1].items
        self.assertGreaterEqual(second.time - first.time, 0.4)  # still one after the other
        speaker, listener = shots[-1].items
        self.assertAlmostEqual(speaker.time, shots[-1].start + 0.12)
        self.assertGreaterEqual(listener.time, speaker.time)
        # A single item is enough for a list shot; a shot at the very end is dropped.
        one = scenes.time_shots([{"type": "sequence", "items": [{"at": "uno", "label": "b"}]}], self.TIMES.get, 5.0)
        self.assertEqual(len(one[0].items), 1)
        self.assertEqual(scenes.time_shots([{"type": "single", "at": "seis", "label": "x"}], self.TIMES.get, 17.0), [])


class TestStoryboardPlan(unittest.TestCase):
    def test_normalize_storyboard(self):
        data = {"segments": [{"index": 0, "opener": {"query": "call center"}, "shots": [
            {"type": "single", "at": "a", "label": "Robot", "draw": "a robot", "otter": 1, "text": "Título"},
            {"type": "single", "at": "b", "pose": "FELIZ"},
            {"type": "speech", "at": "c", "text": "hola", "items": [{"pose": "explicando"}, {"label": "voz", "draw": "a voice", "at": "d"}]},
            {"type": "speech", "at": "c", "items": []},
            {"type": "illustration", "at": "e", "draw": "the otter waking up at dawn", "text": "05:30 de la mañana", "otter": True},
            {"type": "illustration", "at": "e", "text": "sin dibujo"},
            {"type": "sequence", "items": [{"at": "f", "label": "solo"}]},
            {"type": "stat", "at": "g", "value": 17000000, "label": "personas"},
            {"type": "nope"},
        ]}]}
        board = llm.normalize_storyboard(data, 2, ["feliz", "explicando"])
        shots = board[0]["shots"]
        self.assertEqual([s["type"] for s in shots], ["single", "single", "speech", "illustration", "sequence", "stat"])
        self.assertEqual((shots[0]["otter"], shots[0]["text"]), (True, "Título"))
        self.assertEqual(shots[1]["pose"], "feliz")
        self.assertEqual(shots[2]["items"][0]["pose"], "explicando")
        self.assertEqual(shots[3]["text"], "05:30 de la mañana")
        self.assertEqual(board[0]["opener"]["query"], "call center")
        self.assertEqual(board[1]["shots"], [])
        prompt = llm.build_storyboard_prompt([{"index": 0, "kind": "item", "title": "t", "text": "hola"}], ["feliz"], "es-CO")
        for words in ("animated documentary", "EVERY sentence", '"archive"', '"illustration"', '"animation"', "otter", "es-CO"):
            self.assertIn(words, prompt)
        reply = json.dumps(data)
        with patch.object(llm, "_generate_response", return_value=reply):
            self.assertEqual(len(llm.generate_storyboard([{}, {}], ["feliz"])[0]["shots"]), 6)
        with patch.object(llm, "_generate_response", return_value="Error: no"):
            self.assertIsNone(llm.generate_storyboard([{}], []))


class TestDrawing(_TempDirCase):
    def test_draw_uses_the_gemini_image_models_and_caches(self):
        calls = []

        class Models:
            def generate_images(self, **kwargs):
                calls.append(("imagen", kwargs))
                buffer = io.BytesIO()
                Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
                image = type("I", (), {"image_bytes": buffer.getvalue()})
                return type("R", (), {"generated_images": [type("G", (), {"image": image})]})

            def generate_content(self, **kwargs):
                calls.append(("gemini", kwargs))
                buffer = io.BytesIO()
                Image.new("RGB", (64, 36), "white").save(buffer, format="PNG")
                part = type("P", (), {"inline_data": type("D", (), {"data": buffer.getvalue()})})
                content = type("C", (), {"parts": [part]})
                return type("R", (), {"candidates": [type("Ca", (), {"content": content})]})

        class Client:
            def __init__(self, **kwargs):
                self.models = Models()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        mascot = _icon(self.path("otter.png"))
        imagen = {"gemini_image_model": "imagen-4.0-fast-generate-001"}
        with patch("google.genai.Client", Client), patch.object(gemini_media, "_client_kwargs", return_value={}), patch.object(
            gemini_media, "_cache_path", side_effect=lambda model, key: self.path(f"{abs(hash((model, key)))}.png")
        ), patch.dict(gemini_media._state, {"gone": {}, "no_size": set(), "imagen_failed": "", "last_error": ""}):
            first = gemini_media.draw("a robot with a headset", app_config={})
            again = gemini_media.draw("a robot with a headset", app_config={})
            otter = gemini_media.draw("the otter on the phone", mascot=mascot, app_config={})
            scene = gemini_media.draw("a dark bedroom at dawn", scene=True, app_config={})
            self.assertTrue(os.path.isfile(scene))
            self.assertEqual(gemini_media.draw("  ", app_config={}), "")
            gemini_media.draw("a lamp", app_config=imagen)  # Imagen only when config.toml names it
            gemini_media.draw("the otter waves", mascot=mascot, app_config=imagen)
        self.assertEqual(first, again)
        self.assertTrue(os.path.isfile(otter))
        kinds = [kind for kind, _ in calls]
        self.assertEqual(kinds, ["gemini", "gemini", "gemini", "imagen", "gemini"])  # cached once; Imagen cannot follow a picture
        self.assertEqual(calls[0][1]["model"], gemini_media.FLASH_IMAGE_MODELS[0])
        self.assertIn("cel shading", calls[0][1]["contents"])  # the polished cartoon look by default
        self.assertEqual(len(calls[1][1]["contents"]), 2)  # the reference picture and the prompt
        self.assertIn("mascot", calls[1][1]["contents"][1])
        scene_call = calls[2][1]
        self.assertEqual(scene_call["config"].image_config.aspect_ratio, "16:9")
        self.assertEqual(scene_call["config"].image_config.image_size, gemini_media.SHARP_SIZE)  # sharp on a 1080p video
        self.assertIn("16:9", scene_call["contents"])
        self.assertEqual(calls[4][1]["model"], gemini_media.FLASH_IMAGE_MODELS[0])


class TestDoodleRenderer(_TempDirCase):
    def _renderer(self, portrait=False):
        theme = fx.Theme(360, 640, FONT, (255, 79, 94)) if portrait else fx.Theme(640, 360, FONT, (255, 79, 94))
        icon = scenes.prepare_picture(_icon(self.path("icon.png")))
        photo_path = self.path("photo.jpg")
        Image.new("RGB", (320, 180), (90, 120, 160)).save(photo_path)
        photo = scenes.prepare_picture(photo_path, allow_cutout=False)

        def picture(item):
            return photo if item.query == "photo" else icon

        host_still = lambda expression, height: Image.new("RGBA", (height // 2, height), (40, 160, 140, 255))  # noqa: E731
        return scenes.SceneRenderer(theme, self.path(f"r{portrait}"), editor.DOODLE_COLOR, picture, host_still,
                                    {"pop": "p.wav", "whoosh": "w.wav"}, font_path=scenes.hand_font_path(), doodle=True)

    def test_shots_pop_and_boil_on_one_canvas(self):
        Item = scenes.SceneItem
        for portrait in (False, True):
            renderer = self._renderer(portrait)
            for number, scene in enumerate([
                scenes.Scene("single", 0, 4, text="la profesión más común", center=Item(label="call center", icon="x")),
                scenes.Scene("speech", 0, 4, text="hola", items=[Item(label="", time=0.2), Item(label="robot", time=1.0)]),
                scenes.Scene("illustration", 0, 4, text="05:30", center=Item(query="photo")),
                scenes.Scene("sequence", 0, 4, items=[Item(label="a", icon="x", mark="cross", time=0.3)]),
                scenes.Scene("figure", 0, 4, look="photo", center=Item(query="photo")),
            ]):
                overlays, sounds = renderer.build(scene, f"s{number}{portrait}")
                self.assertTrue(overlays)
                self.assertFalse(any(os.path.basename(o.source).startswith("paper-") for o in overlays))  # no own background
                self.assertNotIn("w.wav", [path for _, path, _ in sounds])  # no whooshes between shots
                self.assertTrue(all(o.x.split("+")[-1] in ("0", o.x) for o in overlays))  # nothing slides
                self.assertTrue(all(o.end == 4 and o.fade_out >= 0.12 for o in overlays))
                drawn = [o for o in overlays if o.mode == "sequence"]
                if scene.type != "illustration" and scene.type != "figure":
                    self.assertTrue(drawn, scene.type)
                    listing = Path(drawn[0].source).read_text(encoding="utf-8")
                    self.assertIn("_boil", listing)
                    self.assertTrue(listing.startswith("ffconcat version 1.0"))

    def test_boil_variants_and_helpers(self):
        image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
        ImageDraw.Draw(image).rectangle((20, 20, 80, 60), outline=(0, 0, 0, 255), width=4)
        variants = scenes.boil_variants(image, seed=3)
        self.assertEqual(len(variants), 3)
        self.assertTrue(all(v.size == image.size for v in variants))
        self.assertNotEqual(variants[0].tobytes(), image.tobytes())
        self.assertEqual(len(scenes.boil_variants(Image.new("RGBA", (4, 4)))), 3)
        small = Image.new("RGBA", (50, 40))
        self.assertEqual(scenes.fit_inside(small, 200).size, (125, 100))  # grows at most 2.5 times
        self.assertEqual(scenes.fit_inside(Image.new("RGBA", (400, 200)), 100).size, (100, 50))
        self.assertEqual(scenes.format_number(17000000), "17,000,000")
        self.assertEqual(scenes.format_number(2.5), "2.5")
        self.assertEqual(scenes.format_number(7), "7")
        listing = fx.write_ffconcat(self.path("a.ffconcat"), [(self.path("x.png"), 0.1), (self.path("y.png"), 0.2)])
        lines = Path(listing).read_text().splitlines()
        self.assertEqual(lines[-1], f"file '{self.path('y.png')}'")  # the last picture repeats
        badge = fx.render_badge(Image.new("RGBA", (80, 120), (200, 100, 50, 255)), 60, (255, 79, 94))
        self.assertEqual(badge.size, (60, 60))
        self.assertEqual(badge.getpixel((0, 0))[3], 0)  # round


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


class TestDoodleEditor(_TempDirCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def _editor(self, gemini=True, **options):
        patcher = patch.object(editor.gemini_media, "enabled", return_value=gemini)
        patcher.start()
        self.addCleanup(patcher.stop)
        segments = list_video.build_segments(ListVideoScript(
            title="T", intro="Hola nutrias.", items=[ListVideoItem(name="Voltaje", text=self.TEXT, image_term="battery")], outro="Chao."))
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(640, 360, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        options.setdefault("look", "doodle")
        return editor.Editor(editor.EditOptions(assets_dir=NUTRIA, **options), theme, self.temp_dir, segments, narrations)

    BOARD = {"segments": [
        {"index": 0, "shots": []},
        {"index": 1, "shots": [
            {"type": "single", "at": "palabra3", "label": "a", "draw": "a battery with arms", "otter": True},
            {"type": "illustration", "at": "palabra16", "draw": "a dark lab", "text": "05:30"},
            {"type": "speech", "at": "palabra30", "text": "hola", "items": [{"pose": "feliz"}, {"label": "robot", "draw": "a robot", "icon": "🤖"}]},
            {"type": "sequence", "items": [{"at": "palabra40", "label": "x", "icon": "🔋"}, {"at": "palabra44", "label": "y", "draw": "a plug"}]},
        ]},
        {"index": 2, "shots": []},
    ]}

    def test_storyboard_drawings_and_canvas(self):
        ed = self._editor(openers=False, logo="nutria", max_drawings=3)
        drawing = _icon(self.path("drawing.png"), color=(30, 160, 90, 255))
        icon = _icon(self.path("icon.png"))
        drawn = []

        def draw(description, scene=False, mascot="", app_config=None, **kwargs):
            drawn.append((description, scene, bool(mascot)))
            return "" if scene else drawing

        with patch.object(editor.llm, "generate_storyboard", return_value=llm.normalize_storyboard(self.BOARD, 3, sorted(ed.poses))) as board, \
                patch.object(editor.llm, "generate_storyboard_gaps", return_value={}), \
                patch.object(editor.gemini_media, "draw", side_effect=draw), patch.object(editor.icons, "fetch", return_value=icon), \
                patch.object(editor.gemini_media, "check_drawing", return_value=True), \
                patch.object(editor.web_images, "find_candidates", return_value=[]) as find:
            ed.make_plan()
            edit = ed.segment_edit(1, 1.5, show_titles=True)
        board.assert_called_once()
        # Real pictures are only looked for when a drawing failed (the plug and the dark lab), never footage.
        self.assertEqual(sorted(c.args[0] for c in find.call_args_list), ["a dark lab", "a plug"])
        self.assertFalse(ed.wants_footage)
        self.assertEqual(drawn[0], ("a dark lab", True, False))  # illustrations are drawn first (the budget goes to them)
        self.assertIn(("a battery with arms", False, True), drawn)  # the otter is drawn from its own picture
        self.assertLessEqual(ed._drawings, 3)
        shots = ed._scenes[1]
        self.assertNotIn("illustration", [s.type for s in shots])  # not drawn: dropped, and the shot before stays longer
        self.assertEqual(shots[0].start, 0.0)
        for shot, following in zip(shots, shots[1:]):
            self.assertAlmostEqual(shot.end, following.start)
        speech = next(s for s in shots if s.type == "speech")
        self.assertIsNotNone(ed._scene_picture(speech.items[0]))  # the feliz pose
        self.assertEqual(os.path.basename(edit.overlays[0].source), "canvas.png")
        self.assertEqual(os.path.basename(edit.overlays[-1].source), "logo.png")
        self.assertEqual(ed.host_plan[1].windows, [])  # the otter lives inside the drawings
        self.assertFalse(any(os.path.basename(o.source).startswith("chip-") for o in edit.overlays))
        self.assertEqual(ed.canvas_color(), editor.DOODLE_COLOR)

    def test_fallback_storyboard_and_plan_files(self):
        ed = self._editor(gemini=False, canvas_color="white")
        with patch.object(editor.llm, "generate_storyboard", return_value=None):
            plan = ed.make_plan()
        self.assertEqual(plan[1]["shots"][0]["draw"], "battery")
        self.assertTrue(plan[0]["shots"][0]["pose"])
        self.assertTrue(any("storyboard" in w for w in ed.warnings))
        self.assertEqual(ed.canvas_color(), (250, 247, 242))
        path = self.path("board.json")
        Path(path).write_text(json.dumps(self.BOARD), encoding="utf-8")
        self.assertEqual(len(editor.load_plan_file(path, 3, [])[1]["shots"]), 4)

    def test_no_footage_is_downloaded(self):
        ed = self._editor(gemini=False)
        self.assertFalse(ed.wants_footage)
        footage = self._editor(gemini=False, look="footage")
        self.assertTrue(footage.wants_footage)


class TestDoodleCli(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.script = os.path.join(self.temp_dir, "s.json")
        Path(self.script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")

    def _run(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = list_video_cli.run(["--script", self.script, *args])
            except SystemExit as exc:
                code = exc.code
        return code, generate, stderr.getvalue()

    def test_flags(self):
        _, generate, _ = self._run()
        options = generate.call_args.kwargs
        self.assertEqual((options["edit"].look, options["edit"].boil, options["edit"].logo), ("footage", True, ""))
        self.assertTrue(options["voice_polish_enabled"])
        self.assertNotIn("pause_seconds", options)
        _, generate, _ = self._run("--look", "doodle", "--canvas-color", "#112233", "--no-boil", "--logo", "nutria",
                                   "--max-drawings", "20", "--pause", "5", "--no-voice-polish")
        options = generate.call_args.kwargs
        edit = options["edit"]
        self.assertEqual((edit.look, edit.canvas_color, edit.boil, edit.logo, edit.max_drawings), ("doodle", "#112233", False, "nutria", 20))
        self.assertEqual(options["pause_seconds"], 2.0)
        self.assertFalse(options["voice_polish_enabled"])
        code, _, stderr = self._run("--logo", "/missing.png")
        self.assertEqual(code, 2)
        self.assertIn("--logo picture not found", stderr)


class TestFreshPictures(_TempDirCase):
    def _editor(self):
        narrations = [_narration(1.0, 45)]
        theme = fx.Theme(1280, 720, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        segments = list_video.build_segments(ListVideoScript(title="T", items=[ListVideoItem(name="A", text="hola")]))
        with patch.object(editor.gemini_media, "enabled", return_value=True):
            return editor.Editor(editor.EditOptions(subscribe="none", picture_check=False), theme, self.temp_dir, segments, narrations)

    def test_icons_are_not_repeated(self):
        ed = self._editor()
        fetched = []

        def fetch(query):
            fetched.append(query)
            return _icon(self.path(f"{len(fetched)}.png"))

        with patch.object(editor.icons, "fetch", side_effect=fetch), patch.object(
            editor.icons, "alternatives", return_value=["🪫", "🔌"]
        ) as alternatives:
            first = scenes.SceneItem(icon="🔋", draw="a battery")
            second = scenes.SceneItem(icon="🔋", draw="a battery")
            self.assertIsNotNone(ed._icon_picture(first))
            self.assertIsNotNone(ed._icon_picture(second))
            self.assertIs(ed._icon_picture(first), ed._icon_picture(first))  # cached per element
        self.assertEqual(fetched[0], "🔋")
        self.assertEqual(fetched[1], "a battery")  # the description is tried before searching
        third = scenes.SceneItem(icon="🔋", draw="a battery")
        with patch.object(editor.icons, "fetch", side_effect=fetch), patch.object(
            editor.icons, "alternatives", return_value=["🪫", "🔌"]
        ) as alternatives:
            ed._icon_picture(third)
        alternatives.assert_called_once()
        self.assertEqual(fetched[2], "🪫")

    def test_drawings_are_not_repeated_and_floating_pictures_are_big(self):
        ed = self._editor()
        asked = []
        with patch.object(editor.gemini_media, "draw", side_effect=lambda d, **k: asked.append(d) or ""):
            ed._drawing("a robot")
            ed._drawing("A  robot")
        self.assertEqual(asked[0], "a robot")
        self.assertIn("variation 2", asked[1])
        (cx, cy, w, h), = ed._picture_slots(1)
        self.assertGreaterEqual(w, 1280 * 0.6)
        self.assertGreaterEqual(h, 720 * 0.75)
        boxes = ed._picture_slots(3)
        self.assertGreaterEqual(boxes[0][2], 1280 * 0.3)
        small = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        small.save(self.path("small.png"))
        sticker = fx.make_sticker(ed.theme, self.path("small.png"), 400, 400)
        self.assertGreater(sticker.width, 150)  # small pictures are enlarged


class TestStoryFormat(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def test_story_prompt(self):
        prompt = llm.build_story_script_prompt("Cómo es ser cada rango de la NCIS", 5, "es-CO", 140)
        for words in ("ONE continuous story", "exactly 5 chapters", "INSIDE a concrete", "Never greet",
                      "about 140 words", "Short sentences", "in es-CO", "why things happen"):
            self.assertIn(words, prompt)
        self.assertTrue(prompt.endswith("Cómo es ser cada rango de la NCIS"))
        reply = json.dumps({"title": "T", "intro": "Son las 5:30.", "items": [{"name": "A", "text": "B"}], "outro": "C"})
        with patch.object(llm, "_generate_response", return_value=reply) as ask:
            llm.generate_list_script("NCIS", 3, script_format="story")
        self.assertIn("ONE continuous story", ask.call_args.args[0])
        list_prompt = llm.build_list_script_prompt("X", 3)
        self.assertIn("gripping moment", list_prompt)
        self.assertNotIn("ONE continuous story", list_prompt)
        self.assertIn("flow into each other", llm.build_edit_plan_prompt([], [], "es", openers=False))
        self.assertIn('"opener"', llm.build_storyboard_prompt([], [], "es", openers=True))
        self.assertIn("continuous story", llm.build_storyboard_prompt([], [], "es", openers=False))

    def test_story_cli_hides_sections(self):
        script = os.path.join(self.temp_dir, "s.json")
        Path(script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, redirect_stdout(stdout), redirect_stderr(stderr):
            list_video_cli.run(["--script", script, "--format", "story"])
        kwargs = generate.call_args.kwargs
        # A story keeps its section cards (a photo, the name said aloud) and their labels, and flows on.
        self.assertTrue(kwargs["number_items"])
        self.assertTrue(kwargs["show_item_titles"])
        self.assertTrue(kwargs["edit"].openers)
        self.assertTrue(kwargs["say_names"])
        self.assertTrue(kwargs["edit"].seamless)
        self.assertEqual(kwargs["gap_seconds"], 0.25)
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, redirect_stdout(stdout), redirect_stderr(stderr):
            list_video_cli.run(["--script", script, "--format", "story", "--no-openers", "--no-item-titles", "--no-numbers"])
        kwargs = generate.call_args.kwargs
        self.assertFalse(kwargs["number_items"] or kwargs["show_item_titles"] or kwargs["edit"].openers or kwargs["say_names"])
        with patch.object(llm, "generate_list_script", return_value=None) as write, redirect_stdout(stdout), redirect_stderr(stderr):
            list_video_cli.run(["--subject", "NCIS", "--format", "story", "--script-only"])
        self.assertEqual(write.call_args.kwargs["script_format"], "story")


class TestElevenLabsSettings(unittest.TestCase):
    def test_settings_come_from_config(self):
        from unittest.mock import patch

        from app.config import config
        from app.services import voice, voice_google

        with patch.dict(config.elevenlabs, {"stability": 0.7, "style": "bad", "similarity_boost": 3}):
            settings = voice.elevenlabs_voice_settings(0.9)
        self.assertEqual(settings["stability"], 0.7)
        self.assertEqual(settings["style"], 0.0)
        self.assertEqual(settings["similarity_boost"], 1.0)
        self.assertEqual(settings["speed"], 0.9)
        self.assertNotIn("speed", voice.elevenlabs_voice_settings(1.0))
        self.assertEqual(voice.elevenlabs_voice_settings(3)["speed"], 1.2)
        self.assertIn("pausa", voice_google.VOICE_STYLE_PRESETS["sereno"])


if __name__ == "__main__":
    unittest.main()
