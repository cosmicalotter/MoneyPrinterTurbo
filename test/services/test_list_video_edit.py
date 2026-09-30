import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript, VideoParams
from app.services import list_video, llm, web_images
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.utils import utils

RESOURCES = Path(__file__).parent.parent / "resources"
FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")


def _theme(width=1920, height=1080):
    return fx.Theme(width, height, FONT, fx.parse_color(fx.DEFAULT_ACCENT))


def _speech_pcm(seconds, sample_rate=24000):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    envelope = (np.sin(2 * np.pi * 4 * t) > 0).astype(np.float32)
    return (0.4 * np.sin(2 * np.pi * 180 * t) * envelope * 32767).astype(np.int16).tobytes()


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class TestGraphics(_TempDirCase):
    def test_parse_color(self):
        self.assertEqual(fx.parse_color("#FF4F5E"), (255, 79, 94))
        self.assertEqual(fx.parse_color("0af"), (0, 170, 255))
        with self.assertRaises(ValueError):
            fx.parse_color("red")

    def test_chip_callout_and_bar_sizes(self):
        theme = _theme()
        chip = fx.render_chapter_chip(theme, 3, "Parálisis del sueño")
        plain = fx.render_chapter_chip(theme, None, "Intro")
        self.assertGreater(chip.width, plain.width)
        self.assertEqual(chip.mode, "RGBA")
        callout = fx.render_callout(theme, "W" * 80)
        self.assertLessEqual(callout.width, int(1920 * 0.62) + 2 * 60)
        self.assertEqual(fx.render_progress_bar(theme).width, 1920)
        self.assertEqual(fx.shadow_padding(theme), 36)

    def test_flat_background_is_removed_but_photos_are_kept(self):
        drawing = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(drawing).ellipse((100, 60, 300, 240), fill=(200, 40, 60))
        cutout = fx.remove_flat_background(drawing)
        self.assertEqual(cutout.getpixel((5, 5))[3], 0)
        self.assertEqual(cutout.getpixel((200, 150))[3], 255)

        noise = Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8))
        self.assertIsNone(fx.remove_flat_background(noise))
        self.assertIsNone(fx.remove_flat_background(Image.new("RGB", (100, 100), "white")))

    def test_stickers_and_cards(self):
        theme = _theme()
        drawing = self.path("drawing.png")
        image = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(image).rectangle((120, 80, 280, 220), fill=(20, 90, 200))
        image.save(drawing)
        sticker = fx.make_sticker(theme, drawing, 500, 500)
        self.assertEqual(sticker.getpixel((0, 0))[3], 0)

        photo = self.path("photo.png")
        Image.fromarray(
            np.random.default_rng(2).integers(0, 255, (300, 500, 3), dtype=np.uint8)
        ).save(photo)
        card = fx.make_sticker(theme, photo, 600, 400, seed=4)
        self.assertEqual(card.mode, "RGBA")
        self.assertLessEqual(card.width, 600 + 200)

    def test_subscribe_frames_and_labels(self):
        pattern, click = fx.render_subscribe_frames(_theme(), self.path("sub"), "en-US", 10, 1.0)
        self.assertEqual(len(os.listdir(self.path("sub"))), 10)
        self.assertTrue(pattern.endswith("sub_%04d.png"))
        self.assertGreater(click, 0)
        self.assertEqual(fx.subscribe_labels("en-US")[0], "SUBSCRIBE")
        self.assertEqual(fx.subscribe_labels("pt-BR")[1], "INSCRITO")
        self.assertEqual(fx.subscribe_labels("es-CO")[0], "SUSCRÍBETE")


class TestCharacterAndSounds(_TempDirCase):
    def test_demo_assets_pair_idle_and_talking_frames(self):
        fx.create_demo_assets(self.temp_dir)
        Path(self.path("personaje/notes.txt")).write_text("ignored")
        poses = fx.load_character(self.temp_dir)
        self.assertEqual(sorted(poses), sorted(fx.DEMO_EXPRESSIONS))
        self.assertTrue(poses["feliz"].talk.endswith("feliz_habla.png"))
        self.assertEqual(fx.load_character(""), {})
        self.assertEqual(fx.load_character(self.path("missing")), {})

        prepared = fx.prepare_character_image(poses["feliz"].idle, self.path("c.png"), 200, mirror=True)
        with Image.open(prepared) as image:
            self.assertEqual(image.height, 200)

        # A character drawn on a white background is cut out.
        white = Image.new("RGB", (300, 400), "white")
        ImageDraw.Draw(white).ellipse((50, 80, 250, 360), fill=(40, 160, 90))
        white.save(self.path("white.png"))
        cut = fx.prepare_character_image(self.path("white.png"), self.path("w.png"), 140)
        with Image.open(cut) as image:
            self.assertEqual(image.getpixel((0, 0))[3], 0)
            self.assertEqual(image.height, 140)

    def test_mouth_follows_the_voice(self):
        runs = fx.mouth_schedule(_speech_pcm(2.0), 24000, 30)
        self.assertEqual(sum(count for _, count in runs), 60)
        self.assertIn(True, [state for state, _ in runs])
        self.assertIn(False, [state for state, _ in runs])
        silent = fx.mouth_schedule(b"\x00\x00" * 24000, 24000, 30)
        self.assertEqual(silent, [(False, 30)])
        self.assertEqual(fx.mouth_schedule(b"", 24000, 30), [])

    def test_sounds_are_synthesized_or_overridden(self):
        for name in fx.SFX_NAMES:
            sound = fx.synthesize_sfx(name)
            self.assertGreater(sound.size, 100)
            self.assertLessEqual(float(np.abs(sound).max()), 1.0)
        with self.assertRaises(ValueError):
            fx.synthesize_sfx("boing")

        os.makedirs(self.path("assets/sfx"))
        custom = fx.write_wav(fx.synthesize_sfx("pop"), self.path("assets/sfx/POP.wav"))
        resolved = fx.resolve_sfx(self.path("assets"), self.temp_dir)
        self.assertEqual(resolved["pop"], custom)
        self.assertTrue(resolved["whoosh"].endswith("sfx-whoosh.wav"))

    def test_find_asset_matches_names_and_extensions(self):
        Path(self.path("Suscribete.GIF")).write_bytes(b"x")
        Path(self.path("suscribete.mp3")).write_bytes(b"x")
        self.assertTrue(fx.find_asset(self.temp_dir, fx.SUBSCRIBE_NAMES, fx.SUBSCRIBE_EXTENSIONS).endswith("Suscribete.GIF"))
        self.assertTrue(fx.find_asset(self.temp_dir, fx.SUBSCRIBE_NAMES, fx.AUDIO_EXTENSIONS).endswith(".mp3"))
        self.assertEqual(fx.find_asset("", ["x"], [".png"]), "")


class TestAnchors(unittest.TestCase):
    TEXT = "Mientras duermes, el líquido que rodea tu cerebro circula con más fuerza."

    def test_anchor_matches_ignoring_accents_and_case(self):
        early = editor.anchor_time(self.TEXT, "Mientras duermes", None, 10.0)
        later = editor.anchor_time(self.TEXT, "el LIQUIDO que rodea", None, 10.0)
        self.assertEqual(early, 0.0)
        self.assertGreater(later, 1.0)
        self.assertIsNone(editor.anchor_time(self.TEXT, "algo distinto", None, 10.0))
        self.assertIsNone(editor.anchor_time("", "x", None, 1.0))
        # Partial anchors fall back to their first words.
        self.assertIsNotNone(editor.anchor_time(self.TEXT, "rodea tu hígado entero", None, 10.0))

    def test_word_boundaries_are_used_when_available(self):
        words = editor.normalize_words(self.TEXT)
        cues = [SimpleNamespace(start=timedelta(seconds=i * 0.5)) for i in range(len(words))]
        sub_maker = SimpleNamespace(cues=cues)
        self.assertAlmostEqual(editor.anchor_time(self.TEXT, "circula con", sub_maker, 20.0), 4.0)

    def test_beats_are_scheduled_without_overlaps(self):
        narration = editor.Narration(pcm=b"", speech_seconds=10.0, frames=330)
        text = " ".join(f"palabra{i}" for i in range(40))
        beats = [
            {"type": "image", "at": "palabra2", "query": "a"},
            {"type": "image", "at": "palabra5", "query": "b"},
            {"type": "image", "at": "palabra12", "query": "c"},
            {"type": "text", "at": "palabra20", "text": "dato"},
            {"type": "text", "at": "no existe", "text": "x"},
            {"type": "image", "at": "palabra39", "query": "late"},
        ]
        scheduled = editor.schedule_beats(beats, text, narration, 11.0)
        kinds = [(b.kind, b.query or b.text) for b in scheduled]
        # "b" would cut "a" short and "late" starts too close to the end.
        self.assertEqual(kinds, [("image", "a"), ("image", "c"), ("text", "dato")])
        self.assertLessEqual(scheduled[0].end, scheduled[1].start)


class TestEditPlan(unittest.TestCase):
    def test_normalize_drops_invalid_parts(self):
        data = {
            "segments": [
                {"index": 1, "expression": "FELIZ", "beats": [
                    {"type": "image", "at": "x y", "query": "brain", "look": "weird"},
                    {"type": "image", "at": "x y", "query": ""},
                    {"type": "text", "at": "z", "text": "t" * 80},
                    {"type": "video", "at": "z"},
                    "nonsense",
                ]},
                {"index": 1, "expression": "nope"},
                {"index": 99, "expression": "feliz"},
                "bad",
            ]
        }
        plan = llm.normalize_edit_plan(data, 3, ["feliz", "triste"])
        self.assertEqual([p["index"] for p in plan], [0, 1, 2])
        self.assertEqual(plan[1]["expression"], "feliz")
        self.assertEqual(plan[1]["beats"][0]["look"], "diagram")
        self.assertEqual(len(plan[1]["beats"]), 2)
        self.assertEqual(len(plan[1]["beats"][1]["text"]), llm.MAX_EDIT_TEXT_LENGTH)
        self.assertEqual(plan[0]["expression"], "")
        with self.assertRaises(ValueError):
            llm.normalize_edit_plan({"nothing": 1}, 1, [])

    def test_prompt_and_generation(self):
        segments = [{"index": 0, "kind": "item", "title": "1. A", "text": "hola mundo"}]
        prompt = llm.build_edit_plan_prompt(segments, ["feliz"], "es-CO")
        self.assertIn('["feliz"]', prompt)
        self.assertIn("es-CO", prompt)
        self.assertIn("hola mundo", prompt)
        self.assertIn('always ""', llm.build_edit_plan_prompt(segments, [], ""))

        reply = json.dumps({"segments": [{"index": 0, "expression": "feliz", "beats": []}]})
        with patch.object(llm, "_generate_response", side_effect=["garbage", reply]):
            self.assertEqual(llm.generate_edit_plan(segments, ["feliz"])[0]["expression"], "feliz")
        with patch.object(llm, "_generate_response", return_value="Error: quota"):
            self.assertIsNone(llm.generate_edit_plan(segments, ["feliz"]))
        with patch.object(llm, "_generate_response", return_value="[]"):
            self.assertIsNone(llm.generate_edit_plan(segments, ["feliz"]))


def _segments():
    script = ListVideoScript(
        title="T",
        intro="Hola a todos, esto es increíble de verdad amigos.",
        items=[
            ListVideoItem(name="Cerebro", text="El cerebro limpia desechos mientras duermes toda la noche."),
            ListVideoItem(name="Hormona", text="La hormona repara músculos y tejidos cada noche."),
        ],
        outro="Suscríbete para más datos curiosos del cuerpo humano.",
    )
    return list_video.build_segments(script)


class TestEditor(_TempDirCase):
    def _editor(self, options, narrations=None):
        segments = _segments()
        narrations = narrations or [
            editor.Narration(pcm=_speech_pcm(6.0), speech_seconds=6.0, frames=190) for _ in segments
        ]
        return editor.Editor(options, _theme(), self.temp_dir, segments, narrations)

    def test_segment_edit_builds_every_layer(self):
        assets = self.path("assets")
        fx.create_demo_assets(assets)
        plan = {"segments": [
            {"index": 1, "expression": "pensando", "beats": [
                {"type": "image", "at": "limpia desechos", "query": "brain", "look": "diagram"},
                {"type": "text", "at": "toda la noche", "text": "8 horas"}]},
        ]}
        plan_file = self.path("plan.json")
        Path(plan_file).write_text(json.dumps(plan), encoding="utf-8")
        found = web_images.WebImage(str(RESOURCES / "1.png"), "wikimedia", "Brain", "Ana", "CC BY 4.0", "https://x")
        ed = self._editor(editor.EditOptions(assets_dir=assets, plan_file=plan_file, language="es-CO"))
        with patch.object(editor.web_images, "find_image", return_value=found) as find:
            ed.make_plan()
            edit = ed.segment_edit(1, 6.3, show_titles=True)
        self.assertEqual(find.call_args.args[0], "brain")
        modes = [o.mode for o in edit.overlays]
        self.assertIn("concat", modes)  # talking character
        sources = [os.path.basename(o.source) for o in edit.overlays]
        self.assertTrue(any(s.startswith("chip-01") for s in sources))
        self.assertTrue(any(s.startswith("beat-01") for s in sources))
        self.assertTrue(any(s.startswith("text-01") for s in sources))
        self.assertIn("progress.png", sources)
        self.assertIn("6.300", edit.overlays[-1].x)
        sounds = [os.path.basename(path) for _, path, _ in edit.sounds]
        self.assertEqual(sounds[0], "sfx-whoosh.wav")
        self.assertIn("sfx-pop.wav", sounds)
        self.assertIn("sfx-tick.wav", sounds)
        mouth = Path(ed.work_dir, "mouth-01.txt").read_text()
        self.assertTrue(mouth.startswith("ffconcat version 1.0"))
        self.assertEqual(ed.write_credits(), os.path.join(self.temp_dir, "credits.txt"))
        saved = json.loads(Path(self.temp_dir, "edit-plan.json").read_text("utf-8"))
        self.assertEqual(saved["segments"][0]["expression"], "sorprendido")

    def test_subscribe_placement_and_fallback_plan(self):
        ed = self._editor(editor.EditOptions(subscribe="both", beats="none", sound_effects=False))
        with patch.object(editor.llm, "generate_edit_plan") as generate:
            ed.make_plan()
        generate.assert_not_called()
        intro = ed.segment_edit(0, 0.0, show_titles=True)
        outro = ed.segment_edit(3, 19.0, show_titles=True)
        middle = ed.segment_edit(1, 6.3, show_titles=True)
        self.assertEqual([o.mode for o in intro.overlays], ["frames", "still"])
        self.assertAlmostEqual(intro.overlays[0].start, 190 / 30 - 0.3 - editor.SUBSCRIBE_SECONDS)
        self.assertEqual(outro.overlays[0].start, 0.6)
        self.assertFalse(any(o.mode == "frames" for o in middle.overlays))
        self.assertEqual(intro.sounds, [])

    def test_llm_failure_keeps_base_visuals(self):
        ed = self._editor(editor.EditOptions(subscribe="none"))
        with patch.object(editor.llm, "generate_edit_plan", return_value=None):
            plan = ed.make_plan()
        self.assertTrue(all(entry["beats"] == [] for entry in plan))
        self.assertIn("edit plan", ed.warnings[0])

    def test_custom_subscribe_assets_and_missing_pictures(self):
        assets = self.path("assets")
        os.makedirs(assets)
        Image.new("RGBA", (300, 100), (255, 0, 0, 255)).save(os.path.join(assets, "suscribete.png"))
        fx.write_wav(fx.synthesize_sfx("click"), os.path.join(assets, "suscribete.wav"))
        ed = self._editor(editor.EditOptions(assets_dir=assets, subscribe="outro"))
        ed.plan = [
            {"index": i, "expression": "", "beats": [{"type": "image", "at": "cerebro limpia", "query": "q"}]}
            for i in range(4)
        ]
        with patch.object(editor.web_images, "find_image", return_value=None):
            edit = ed.segment_edit(3, 19.0, show_titles=False)
            ed.segment_edit(1, 6.3, show_titles=False)
        self.assertEqual(edit.overlays[0].mode, "still")
        self.assertTrue(edit.overlays[0].source.endswith("suscribete.png"))
        self.assertTrue(any(path.endswith("suscribete.wav") for _, path, _ in edit.sounds))
        self.assertTrue(any("no picture found" in w for w in ed.warnings))


class _Response:
    def __init__(self, payload=None, content=b"", content_type="image/png", status=200):
        self._payload = payload
        self._content = content
        self.headers = {"content-type": content_type}
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, size):
        for start in range(0, len(self._content), size):
            yield self._content[start : start + size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _png(width=800, height=600):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestWebImages(_TempDirCase):
    def test_wikimedia_results_are_parsed_in_rank_order(self):
        payload = {"query": {"pages": {
            "2": {"index": 2, "title": "File:Second.png", "imageinfo": [{"mime": "image/png", "thumburl": "https://u/2.png"}]},
            "1": {"index": 1, "title": "File:Brain diagram.svg", "imageinfo": [{
                "mime": "image/svg+xml", "thumburl": "https://u/1.png", "descriptionurl": "https://commons/1",
                "extmetadata": {"Artist": {"value": "<a href='x'>Ana &amp; Bo</a>"}, "LicenseShortName": {"value": "CC BY-SA 4.0"}},
            }]},
            "3": {"index": 3, "title": "File:Doc.pdf", "imageinfo": [{"mime": "application/pdf", "url": "https://u/3.pdf"}]},
        }}}
        with patch.object(web_images, "_request", return_value=_Response(payload)):
            results = web_images.search_wikimedia("brain")
        self.assertEqual([r.url for r in results], ["https://u/1.png", "https://u/2.png"])
        self.assertEqual(results[0].author, "Ana & Bo")
        self.assertEqual(results[0].title, "Brain diagram")

    def test_stock_photo_searches_need_keys(self):
        with patch.dict(app_config.app, {"pexels_api_keys": [], "pixabay_api_keys": []}):
            self.assertEqual(web_images.search_pexels_photos("x"), [])
            self.assertEqual(web_images.search_pixabay_images("x"), [])
        pexels = {"photos": [{"src": {"large2x": "https://p/1.jpg"}, "photographer": "Luz", "url": "https://p"}]}
        pixabay = {"hits": [{"largeImageURL": "https://x/1.jpg", "user": "Leo", "tags": "cell"}]}
        with patch.dict(app_config.app, {"pexels_api_keys": ["k1"], "pixabay_api_keys": ["k2"]}), patch.object(
            web_images, "_request", side_effect=[_Response(pexels), _Response(pixabay)]
        ):
            self.assertEqual(web_images.search_pexels_photos("x")[0].author, "Luz")
            self.assertEqual(web_images.search_pixabay_images("x")[0].license, "Pixabay Content License")

    def test_downloads_are_validated(self):
        with patch.object(web_images, "_request", return_value=_Response(content=_png())):
            path = web_images.download_image("https://u/ok.png", self.temp_dir)
        self.assertTrue(path.endswith(".png"))
        cases = [
            _Response(content=b"<html>", content_type="text/html"),
            _Response(content=b"not an image"),
            _Response(content=_png(200, 150)),
            _Response(content=_png(2000, 400)),
        ]
        for response in cases:
            with self.subTest(response=response.headers), patch.object(web_images, "_request", return_value=response):
                with self.assertRaises(ValueError):
                    web_images.download_image("https://u/bad", self.temp_dir)
        with patch.object(web_images, "MAX_DOWNLOAD_BYTES", 10), patch.object(
            web_images, "_request", return_value=_Response(content=_png())
        ):
            with self.assertRaisesRegex(ValueError, "larger"):
                web_images.download_image("https://u/big", self.temp_dir)

    def test_find_image_tries_sources_in_order(self):
        good = web_images._Candidate("https://ok", "pexels", "t", "a", "l", "p")
        used = web_images._Candidate("https://used", "wikimedia", "t", "a", "l", "p")
        searchers = {
            "wikimedia": lambda q: [used],
            "pixabay": lambda q: (_ for _ in ()).throw(RuntimeError("down")),
            "pexels": lambda q: [good],
        }
        with patch.dict(web_images.SEARCHERS, searchers), patch.object(
            web_images, "download_image", return_value=self.path("x.png")
        ):
            found = web_images.find_image("q", self.temp_dir, kind="diagram", exclude_urls={"https://used"})
        self.assertEqual(found.source, "pexels")
        self.assertIn("t — a — l — p", found.credit())
        self.assertEqual(web_images.source_order("photo")[0], "pexels")
        with patch.dict(web_images.SEARCHERS, {k: (lambda q: []) for k in web_images.SEARCHERS}):
            self.assertIsNone(web_images.find_image("q", self.temp_dir))


class TestAudioAndRendering(_TempDirCase):
    def _wav(self, name, seconds, value=0):
        path = self.path(name)
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(np.full(int(seconds * 24000), value, dtype=np.int16).tobytes())
        return path

    def test_sound_effects_land_at_their_time(self):
        narration = self._wav("voice.wav", 2.0)
        effect = fx.write_wav(np.full(2400, 0.5, dtype=np.float32), self.path("fx.wav"))
        mixed = list_video.mix_sound_effects(
            narration, [(1.0, effect, 1.0), (-1.0, effect, 1.0), (5.0, effect, 1.0)], self.path("mix.wav")
        )
        with wave.open(mixed, "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        self.assertEqual(samples[23999], 0)
        self.assertGreater(samples[24100], 10000)
        self.assertGreater(samples[100], 10000)

    def test_overlay_modes_keep_the_exact_frame_count(self):
        params = VideoParams(video_subject="t", video_aspect="1:1")
        still = self.path("still.png")
        Image.new("RGBA", (200, 100), (255, 255, 255, 200)).save(still)
        frames_dir = self.path("seq")
        fx.render_subscribe_frames(_theme(1080, 1080), frames_dir, "es", 30, 0.5)
        concat = self.path("mouth.txt")
        Path(concat).write_text(f"ffconcat version 1.0\nfile '{still}'\nduration 0.2\nfile '{still}'\n")
        overlays = [
            fx.Overlay(still, x="10", y="10+5*sin(t)", start=0.1, end=0.6, fade_in=0.1, fade_out=0.1),
            fx.Overlay(os.path.join(frames_dir, "sub_%04d.png"), x="0", y="0", start=0.2, mode="frames"),
            fx.Overlay(concat, x="0", y="H-h", mode="concat"),
        ]
        output = list_video.render_segment_video(
            list_video._Visual("none"), 24, params, self.path("seg.mp4"),
            overlays=overlays, fade_in=0.2, fade_out=0.2,
        )
        self.assertEqual(list_video.count_video_frames(output), 24)

    def test_final_mux_normalizes_and_ducks_music(self):
        video_file = self.path("v.mp4")
        subprocess.run(
            [utils.get_ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i",
             "color=c=black:s=320x240:r=30", "-frames:v", "60", video_file],
            check=True,
        )
        voice_file = fx.write_wav(
            np.frombuffer(_speech_pcm(2.0), dtype=np.int16).astype(np.float32) / 32768, self.path("voice.wav")
        )
        music = fx.write_wav(0.2 * np.sin(np.linspace(0, 800, 24000)).astype(np.float32), self.path("m.wav"))
        for bgm in ("", music):
            output = list_video.mux_final_video(video_file, voice_file, self.path(f"out{bool(bgm)}.mp4"), 2.0, bgm_file=bgm)
            probe = subprocess.run([utils.get_ffmpeg_binary(), "-hide_banner", "-i", output], capture_output=True, text=True)
            self.assertIn("Audio: aac", probe.stderr)
        normalized = list_video.normalize_loudness(voice_file, self.path("n.wav"))
        self.assertTrue(os.path.isfile(normalized))
        with self.assertRaises(list_video.ListVideoError):
            list_video.normalize_loudness(self.path("missing.wav"), self.path("n2.wav"))

    def test_background_music_selection(self):
        warnings = []
        for bgm_type in ("none", ""):
            params = VideoParams(video_subject="t", bgm_type=bgm_type)
            self.assertEqual(list_video._list_bgm_file(params, warnings), "")
        params = VideoParams(video_subject="t", bgm_type="sonilo")
        self.assertEqual(list_video._list_bgm_file(params, warnings), "")
        self.assertIn("not supported", warnings[0])
        params = VideoParams(video_subject="t", bgm_type="random")
        with patch.object(list_video.video, "get_bgm_file", return_value="/songs/a.mp3"):
            self.assertEqual(list_video._list_bgm_file(params, warnings), "/songs/a.mp3")


class TestEditedGeneration(unittest.TestCase):
    def setUp(self):
        self.task_id = str(uuid4())
        self.addCleanup(shutil.rmtree, os.path.join(utils.storage_dir(), "tasks", self.task_id), True)

    def test_edited_video_mixes_effects_and_muxes_without_moviepy(self):
        items = [ListVideoItem(name=f"Item {n}", text=f"Frase numero {n} aqui.", image_file=str(RESOURCES / f"{n}.png")) for n in (1, 2)]
        script = ListVideoScript(title="T", intro="Hola.", items=items, outro="Chao.")
        params = VideoParams(
            video_subject="t", video_source="local", voice_name="no-voice",
            subtitle_enabled=False, bgm_type="", video_aspect="16:9",
        )
        rendered = []

        def fake_render(visual, frames, params, output_file, **kwargs):
            rendered.append(kwargs)
            Path(output_file).write_bytes(b"x")
            return output_file

        with patch.object(list_video, "render_segment_video", side_effect=fake_render), patch.object(
            list_video, "count_video_frames", return_value=0
        ), patch.object(list_video, "concat_segments", side_effect=lambda files, out: out), patch.object(
            list_video, "mux_final_video"
        ) as mux, patch.object(list_video.video, "generate_video") as generate_video, patch.object(
            editor.llm, "generate_edit_plan", return_value=None
        ):
            result = list_video.generate_list_video(
                self.task_id, script, params, edit=editor.EditOptions(subscribe="none")
            )
        generate_video.assert_not_called()
        audio = mux.call_args.args[1]
        self.assertTrue(audio.endswith("narration-sfx.wav"))
        self.assertEqual(rendered[0]["fade_in"], 0.5)
        self.assertEqual(rendered[-1]["fade_out"], 0.6)
        self.assertEqual(rendered[1]["title"], "")
        self.assertTrue(any("edit plan" in w for w in result["warnings"]))
        self.assertTrue(os.path.isfile(os.path.join(utils.task_dir(self.task_id), "edit-plan.json")))


class TestEditingCli(unittest.TestCase):
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

    def test_defaults_turn_off_subtitles_and_music_and_enable_editing(self):
        with patch.object(list_video, "generate_list_video", return_value={}) as generate:
            code, _, _ = self._run(["--script", self.script, "--accent", "#00AAFF", "--beats", "none"])
        self.assertEqual(code, 0)
        params = generate.call_args.args[2]
        options = generate.call_args.kwargs["edit"]
        self.assertFalse(params.subtitle_enabled)
        self.assertEqual(params.bgm_type, "")
        self.assertEqual((options.accent, options.beats, options.subscribe), ("#00AAFF", "none", "both"))

    def test_explicit_subtitles_music_and_no_edit(self):
        with patch.object(list_video, "generate_list_video", return_value={}) as generate:
            self._run(["--script", self.script, "--subtitle-enabled", "--bgm-type", "random", "--no-edit"])
        params = generate.call_args.args[2]
        self.assertTrue(params.subtitle_enabled)
        self.assertEqual(params.bgm_type, "random")
        self.assertNotIn("edit", generate.call_args.kwargs)

    def test_voice_style_is_applied_to_gemini(self):
        with patch.dict(app_config.app, {}), patch.object(list_video, "generate_list_video", return_value={}):
            self._run(["--script", self.script, "--voice-style", "Con entusiasmo"])
            self.assertEqual(app_config.app["gemini_tts_style"], "Con entusiasmo")

    def test_demo_assets_and_input_errors(self):
        code, stdout, _ = self._run(["--create-demo-assets", os.path.join(self.temp_dir, "a")])
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(stdout)["files"]), 10)
        for argv, message in (
            ([], "one of --subject or --script"),
            (["--script", self.script, "--assets", "/missing/dir"], "--assets folder not found"),
            (["--script", self.script, "--edit-plan", "/missing.json"], "--edit-plan file not found"),
            (["--script", self.script, "--accent", "blue"], "invalid color"),
        ):
            code, _, stderr = self._run(argv)
            self.assertEqual(code, 2, argv)
            self.assertIn(message, stderr)


class TestBilingual(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.script_path = os.path.join(self.temp_dir, "electricidad.json")
        Path(self.script_path).write_text(json.dumps({
            "title": "La electricidad explicada para nutrias",
            "intro": "Hola, nutrias.",
            "intro_image_term": "lightning",
            "items": [
                {"name": "Voltaje", "text": "El voltaje empuja.", "image_term": "battery"},
                {"name": "Corriente", "text": "La corriente fluye.", "image_term": "wire", "image_file": "a.png"},
            ],
        }), encoding="utf-8")
        Image.new("RGB", (600, 600)).save(os.path.join(self.temp_dir, "a.png"))

    def _run(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = list_video_cli.run(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def _translation(self, **overrides):
        data = {
            "title": "Electricity Explained for Otters",
            "intro": "Hi, otters.",
            "intro_image_term": "changed by the model",
            "items": [
                {"name": "Voltage", "text": "Voltage pushes.", "image_term": "x"},
                {"name": "Current", "text": "Current flows.", "image_term": "y"},
            ],
        }
        data.update(overrides)
        return json.dumps(data)

    def test_translation_keeps_pictures_and_structure(self):
        script = list_video_cli.load_script_file(self.script_path)
        with patch.object(llm, "_generate_response", return_value=self._translation()):
            translated = llm.translate_list_script(script, "en-US")
        self.assertEqual(translated.title, "Electricity Explained for Otters")
        self.assertEqual(translated.items[1].name, "Current")
        self.assertEqual(translated.intro_image_term, "lightning")
        self.assertEqual(translated.items[0].image_term, "battery")
        self.assertTrue(translated.items[1].image_file.endswith("a.png"))

        prompt = llm.build_translate_list_script_prompt({"title": "x"}, "en-US")
        self.assertIn("Explained for Otters", prompt)
        self.assertIn("in en-US", prompt)

    def test_translation_failures(self):
        script = list_video_cli.load_script_file(self.script_path)
        wrong_count = self._translation(items=[{"name": "A", "text": "B"}])
        with patch.object(llm, "_generate_response", side_effect=[wrong_count, self._translation()]):
            self.assertIsNotNone(llm.translate_list_script(script, "en-US"))
        with patch.object(llm, "_generate_response", return_value="Error: quota"):
            self.assertIsNone(llm.translate_list_script(script, "en-US"))
        with patch.object(llm, "_generate_response", return_value=wrong_count):
            self.assertIsNone(llm.translate_list_script(script, "en-US"))
        with self.assertRaises(ValueError):
            llm.translate_list_script(script, " ")

    def test_default_voices(self):
        from app.services import voice

        self.assertEqual(voice.default_edge_voice("en-US"), "en-US-AndrewMultilingualNeural-Male")
        self.assertEqual(voice.default_edge_voice("en"), "en-US-AndrewMultilingualNeural-Male")
        self.assertTrue(voice.default_edge_voice("es-CO").endswith("-Male"))
        self.assertEqual(voice.default_edge_voice("xx-YY"), "")
        self.assertEqual(voice.default_edge_voice(""), "")

    def test_script_only_saves_both_versions(self):
        with patch.object(llm, "_generate_response", return_value=self._translation()):
            code, stdout, _ = self._run(["--script", self.script_path, "--also-in", "en-US", "--script-only"])
        self.assertEqual(code, 2)  # --script-only needs --subject

        output = os.path.join(self.temp_dir, "new.json")
        script = list_video_cli.load_script_file(self.script_path)
        with patch.object(llm, "generate_list_script", return_value=script), patch.object(
            llm, "_generate_response", return_value=self._translation()
        ):
            code, stdout, _ = self._run(
                ["--subject", "Electricidad", "--also-in", "en-US", "--script-only", "--output", output]
            )
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["also"]["script_file"], os.path.join(self.temp_dir, "new.en-US.json"))
        saved = list_video_cli.load_script_file(summary["also"]["script_file"])
        self.assertEqual(saved.title, "Electricity Explained for Otters")

    def test_both_versions_are_rendered_with_their_own_voice_and_plan(self):
        plan = os.path.join(self.temp_dir, "plan.json")
        Path(plan).write_text('{"segments": []}', encoding="utf-8")
        calls = []

        def fake_generate(task_id, script, params, **kwargs):
            calls.append((task_id, script.title, params.voice_name, params.video_language,
                          kwargs["edit"], app_config.app.get("gemini_tts_style")))
            return {"videos": [f"{task_id}.mp4"]}

        with patch.dict(app_config.app, {}), patch.object(
            llm, "_generate_response", return_value=self._translation()
        ), patch.object(list_video, "generate_list_video", side_effect=fake_generate):
            code, stdout, _ = self._run([
                "--script", self.script_path, "--video-language", "es-CO",
                "--voice-name", "es-CO-GonzaloNeural-Male", "--voice-style", "Con entusiasmo",
                "--edit-plan", plan, "--also-in", "en-US",
            ])
        self.assertEqual(code, 0)
        (_, title_es, voice_es, lang_es, edit_es, style_es), (_, title_en, voice_en, lang_en, edit_en, style_en) = calls
        self.assertEqual((title_es, voice_es, lang_es, style_es), (
            "La electricidad explicada para nutrias", "es-CO-GonzaloNeural-Male", "es-CO", "Con entusiasmo"))
        self.assertEqual((title_en, voice_en, lang_en, style_en), (
            "Electricity Explained for Otters", "en-US-AndrewMultilingualNeural-Male", "en-US", ""))
        self.assertEqual((edit_es.language, edit_en.language), ("es-CO", "en-US"))
        self.assertTrue(edit_es.plan_file.endswith("plan.json"))
        self.assertEqual(edit_en.plan_file, "")
        summary = json.loads(stdout)
        self.assertEqual(summary["also"]["language"], "en-US")
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "electricidad.en-US.json")))

    def test_reviewed_translation_and_failures(self):
        reviewed = os.path.join(self.temp_dir, "reviewed.json")
        Path(reviewed).write_text(self._translation(), encoding="utf-8")
        with patch.object(llm, "translate_list_script") as translate, patch.object(
            list_video, "generate_list_video", side_effect=[{"videos": ["a"]}, {"videos": ["b"]}]
        ) as generate:
            code, stdout, _ = self._run([
                "--script", self.script_path, "--also-in", "en-US",
                "--also-script", reviewed, "--also-voice", "en-GB-RyanNeural-Male",
            ])
        translate.assert_not_called()
        self.assertEqual(generate.call_args.args[2].voice_name, "en-GB-RyanNeural-Male")
        self.assertEqual(code, 0)

        with patch.object(list_video, "generate_list_video", side_effect=[{"videos": ["a"]}, list_video.ListVideoError("x")]):
            code, stdout, _ = self._run(["--script", self.script_path, "--also-in", "en-US", "--also-script", reviewed])
        self.assertEqual(code, 1)
        self.assertNotIn("also", json.loads(stdout))

        with patch.object(llm, "translate_list_script", return_value=None):
            self.assertEqual(self._run(["--script", self.script_path, "--also-in", "en-US"])[0], 1)
        self.assertEqual(self._run(["--script", self.script_path, "--also-in", "xx-YY"])[0], 2)
        code, _, stderr = self._run(["--script", self.script_path, "--also-voice", "x"])
        self.assertEqual(code, 2)
        self.assertIn("need --also-in", stderr)


if __name__ == "__main__":
    unittest.main()
