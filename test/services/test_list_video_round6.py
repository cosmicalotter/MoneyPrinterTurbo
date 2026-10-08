"""Round 6: the animatic look (short drawn moments that tell the story), real clips
in a frame, comic reactions, sturdier drawing and icons that fit."""

import io
import json
import os
import shutil
import subprocess
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
from app.models.schema import ListVideoItem, ListVideoScript, MaterialInfo
from app.services import gemini_media, list_video, llm, studio
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_scenes as scenes
from app.utils import utils

FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))


def _icon(path, color=(220, 60, 60, 255), size=120):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((10, 10, size - 10, size - 10), fill=color, outline=(0, 0, 0, 255), width=6)
    image.save(path)
    return path


def _video(path, seconds=1.5, size="320x180"):
    subprocess.run(
        [utils.get_ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
         "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True,
    )
    return path


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


def _png(color="white", size=(64, 36)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class _Error(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def _client(calls, imagen=None, gemini=None):
    """A fake google-genai client whose answers come from the ``imagen``/``gemini`` lists (exceptions are raised)."""
    imagen = list(imagen or [])
    gemini = list(gemini or [])

    class Models:
        def generate_images(self, **kwargs):
            calls.append(("imagen", kwargs))
            answer = imagen.pop(0) if imagen else _png()
            if isinstance(answer, Exception):
                raise answer
            images = [] if answer is None else [type("G", (), {"image": type("I", (), {"image_bytes": answer})})]
            return type("R", (), {"generated_images": images})

        def generate_content(self, **kwargs):
            calls.append(("gemini", kwargs))
            answer = gemini.pop(0) if gemini else _png()
            if isinstance(answer, Exception):
                raise answer
            part = type("P", (), {"inline_data": type("D", (), {"data": answer})})
            content = type("C", (), {"parts": [part] if answer else []})
            return type("R", (), {"candidates": [type("Ca", (), {"content": content})]})

    class Client:
        def __init__(self, **kwargs):
            self.models = Models()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return Client


class TestSturdyDrawing(_TempDirCase):
    def setUp(self):
        super().setUp()
        for patcher in (
            patch.dict(gemini_media._state, {"imagen_failed": "", "last_error": "", "gone": {}, "no_size": set()}),
            patch.object(gemini_media, "RETRY_SECONDS", (0.0, 0.0)),
            patch.object(gemini_media, "_client_kwargs", return_value={}),
            patch.object(gemini_media, "_cache_path", side_effect=lambda model, key: self.path(f"{abs(hash((model, key)))}.png")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    IMAGEN = {"gemini_image_model": "imagen-4.0-fast-generate-001"}  # Imagen only when config.toml names it

    def test_a_broken_imagen_hands_over_to_the_gemini_image_model(self):
        calls = []
        client = _client(calls, imagen=[_Error("404 Publisher model imagen-4.0 not found", code=404)])
        with patch("google.genai.Client", client):
            first = gemini_media.draw("a copper wire", app_config=self.IMAGEN)
            second = gemini_media.draw("a light bulb", app_config=self.IMAGEN)
        self.assertTrue(os.path.isfile(first) and os.path.isfile(second))
        self.assertEqual([kind for kind, _ in calls], ["imagen", "gemini", "gemini"])  # Imagen is not asked again
        self.assertEqual(calls[1][1]["model"], gemini_media.SEQUENCE_DEFAULT_MODEL)
        self.assertIn("not found", gemini_media.imagen_failure())

    def test_filtered_prompts_and_busy_models(self):
        calls = []
        client = _client(calls, imagen=[None], gemini=[_Error("429 RESOURCE_EXHAUSTED", code=429), _png()])
        with patch("google.genai.Client", client):
            path = gemini_media.draw("a storm cloud", app_config=self.IMAGEN)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual([kind for kind, _ in calls], ["imagen", "gemini", "gemini"])  # filtered, then busy, then drawn
        self.assertEqual(gemini_media.imagen_failure(), "")  # one filtered prompt does not condemn Imagen
        calls.clear()
        failing = _client(calls, gemini=[_Error("400 invalid argument")] * 5)
        with patch("google.genai.Client", failing):
            self.assertEqual(gemini_media.draw("x", app_config={"gemini_image_model": "gemini-2.5-flash-image"}), "")
        models = [kwargs["model"] for _, kwargs in calls]
        self.assertEqual(models[0], "gemini-2.5-flash-image")  # the configured model goes first
        self.assertEqual(len(models), len(set(models)))  # a real error is not retried: the next model is asked
        self.assertEqual(set(models), {"gemini-2.5-flash-image", *gemini_media.FLASH_IMAGE_MODELS})
        self.assertIn("invalid argument", gemini_media.last_error())

    def test_next_frame_and_styles(self):
        calls = []
        previous = self.path("previous.png")
        Image.new("RGB", (160, 90), "navy").save(previous)
        with patch("google.genai.Client", _client(calls)):
            gemini_media.draw("the bulb lights up", previous=previous, app_config={})
            gemini_media.draw("a battery", style="ink", app_config={})
            gemini_media.draw("a dark room", scene=True, app_config={})
        frame = calls[0][1]
        self.assertEqual(calls[0][0], "gemini")  # Imagen cannot follow a picture
        self.assertEqual(len(frame["contents"]), 2)
        self.assertIn("previous frame", frame["contents"][1])
        self.assertIn("the bulb lights up", frame["contents"][1])
        self.assertEqual(frame["config"].image_config.aspect_ratio, "16:9")
        self.assertIn("ink line art", calls[1][1]["contents"])
        self.assertIn("cinematic lighting", calls[2][1]["contents"])

    def test_icon_and_clip_checks(self):
        prompt = gemini_media.build_choice_prompt("CORRIENTE CONTINUA", "steady electric current", 3, "es", purpose="icon")
        for words in ("literally depicts", "kiss", "no icon is better than a wrong one", "1 to 3"):
            self.assertIn(words, prompt)
        clip = gemini_media.build_choice_prompt("Una central eléctrica", "power plant", 1, "es", purpose="clip")
        self.assertIn("watermarks", clip)
        self.assertIn("Spanish", clip)


class TestAnimaticPlan(unittest.TestCase):
    def test_normalize_new_shots(self):
        data = {"segments": [{"index": 0, "shots": [
            {"type": "illustration", "at": "a", "draw": "the otter in the dark", "continue": True, "camera": "left"},
            {"type": "illustration", "at": "b", "draw": "x", "continue": "no", "camera": "spin"},
            {"type": "clip", "at": "c", "query": "thunderstorm at night", "label": "Tormenta", "frame": "full", "draw": "a storm"},
            {"type": "clip", "at": "d", "query": "", "draw": "nothing to search"},
            {"type": "clip", "at": "e", "query": "power plant", "frame": "weird"},
            {"type": "meme", "at": "f", "mood": "LAUGH", "text": "jajaja", "draw": "the otter laughing"},
            {"type": "meme", "at": "g", "mood": "unknown"},
            {"type": "meme", "mood": "shock"},
        ]}]}
        shots = llm.normalize_storyboard(data, 1, [])[0]["shots"]
        self.assertEqual([s["type"] for s in shots], ["illustration", "illustration", "clip", "clip", "meme", "meme"])
        self.assertEqual((shots[0]["continue"], shots[0]["camera"]), (True, "left"))
        self.assertNotIn("continue", shots[1])
        self.assertNotIn("camera", shots[1])
        self.assertEqual((shots[2]["frame"], shots[2]["draw"], shots[2]["label"]), ("full", "a storm", "Tormenta"))
        self.assertEqual(shots[3]["frame"], "card")
        self.assertEqual((shots[4]["mood"], shots[4]["draw"]), ("laugh", "the otter laughing"))
        self.assertEqual(shots[5]["mood"], "shock")

    def test_prompt_asks_for_an_animatic(self):
        segments = [{"index": 0, "kind": "intro", "title": "t", "text": "hola", "seconds": 30.0, "shots": 10}]
        prompt = llm.build_storyboard_prompt(segments, ["feliz"], "es-CO")
        for words in ("ANIMATIONS", "every 4 to 7 seconds", '"continue": true', "last drawn picture before it", '"camera"',
                      '"shots": 10', '"type": "clip"', "3%", "cutaway", '"archive"', "At most ONE"):
            self.assertIn(words, prompt)
        self.assertNotIn('"type": "meme"', prompt)
        more = llm.build_storyboard_prompt(segments, [], "es", clips="more", memes=True)
        self.assertIn("7%", more)
        self.assertIn('"type": "meme"', more)
        self.assertIn("one every 60 seconds", more)
        none = llm.build_storyboard_prompt(segments, [], "es", clips="none")
        self.assertNotIn('"type": "clip"', none)
        self.assertEqual(llm.storyboard_shot_target(30, 3), 10)
        self.assertEqual(llm.storyboard_shot_target(1, 3), 1)
        self.assertEqual(llm.storyboard_shot_target(30, 0.5), 20)  # never faster than every 1.5 s

    def test_long_videos_are_planned_in_parallel_chunks(self):
        segments = [{"index": i, "kind": "item", "title": f"t{i}", "text": f"texto {i}", "seconds": 60, "shots": 20} for i in range(5)]
        self.assertEqual([[s["index"] for s in c] for c in llm.storyboard_chunks(segments)], [[0, 1], [2, 3], [4]])

        def reply(prompt, **kwargs):
            if '"index": 2' in prompt and "texto 2" in prompt:
                return "Error: quota"
            indexes = [i for i in range(5) if f"texto {i}" in prompt.split("## Segments:")[1]]
            return json.dumps({"segments": [
                {"index": i, "shots": [{"type": "illustration", "at": "texto", "draw": f"drawing {i}"}]} for i in indexes
            ]})

        with patch.object(llm, "_generate_response", side_effect=reply) as ask:
            board = llm.generate_storyboard(segments, [], "es", review=False)
        self.assertEqual(ask.call_count, 3)
        self.assertEqual([len(entry.get("shots") or []) for entry in board], [1, 1, 0, 0, 1])
        self.assertNotIn("shots", board[2])  # left for the simple fallback
        self.assertEqual(board[4]["shots"][0]["draw"], "drawing 4")
        with patch.object(llm, "_generate_response", return_value="Error: no"):
            self.assertIsNone(llm.generate_storyboard(segments, [], "es", review=False))

    def test_gap_shots(self):
        gaps = [{"index": 1, "text": "la corriente cruza el cable de cobre", "showing": "single: a wire", "seconds": 9, "shots": 2}]
        prompt = llm.build_storyboard_gaps_prompt(gaps, ["feliz"], "es", clips="none")
        self.assertIn("la corriente cruza", prompt)
        self.assertNotIn('"clip"', prompt.split("## Formats:")[0])
        reply = json.dumps({"shots": [
            {"index": 1, "type": "illustration", "at": "cruza el cable", "draw": "electrons crossing a wire", "continue": True},
            {"index": 7, "type": "illustration", "at": "x", "draw": "wrong segment"},
            {"index": 1, "type": "clip", "at": "de cobre", "query": "copper wire"},
            {"index": 1, "type": "meme", "at": "de cobre", "mood": "shock"},
        ]})
        with patch.object(llm, "_generate_response", return_value=reply):
            shots = llm.generate_storyboard_gaps(gaps, [], "es", clips="none")
        self.assertEqual(list(shots), [1])
        self.assertEqual([s["type"] for s in shots[1]], ["illustration"])  # no clips or memes when they are off
        self.assertTrue(shots[1][0]["continue"])
        self.assertEqual(llm.generate_storyboard_gaps([], []), {})


class TestAnimaticTiming(unittest.TestCase):
    TIMES = {"uno": 1.0, "dos": 4.0, "tres": 9.0, "cuatro": 10.0, "cinco": 18.0, "seis": 21.0}

    def test_no_empty_screen_and_reactions_come_back(self):
        specs = [
            {"type": "compare", "items": [{"at": "dos", "label": "a"}, {"at": "tres", "label": "b"}]},
            {"type": "illustration", "at": "cuatro", "draw": "a lab"},
            {"type": "meme", "at": "cinco", "mood": "shock", "text": "¡¿QUÉ?!"},
        ]
        shots = scenes.time_shots(specs, self.TIMES.get, 30.0)
        self.assertEqual([s.type for s in shots], ["compare", "illustration", "meme", "illustration"])
        self.assertAlmostEqual(shots[0].items[0].time, 0.12)  # shown from the start, not after four empty seconds
        meme, back = shots[2], shots[3]
        self.assertAlmostEqual(meme.end - meme.start, scenes.MEME_SECONDS)
        self.assertTrue(back.reprise)
        self.assertIs(back.center, shots[1].center)  # the very same drawing comes back
        self.assertEqual((back.start, back.end), (meme.end, 30.0))
        self.assertEqual((meme.mood, meme.text), ("shock", "¡¿QUÉ?!"))
        short = scenes.time_shots(specs[2:] + [{"type": "single", "at": "seis", "label": "x"}], self.TIMES.get, 25.0)
        self.assertEqual(short[0].end, short[1].start)  # a short reaction keeps its length
        holds = scenes.long_holds(shots, 30.0, 6.0)
        # The second element of the comparison is said nine seconds in: a long hold too.
        self.assertEqual(holds, [(0.12, 9.0), (10.0 - 0.15, 18.0 - 0.15), (scenes.MEME_SECONDS + 18.0 - 0.15, 30.0)])

    def test_illustration_and_clip_specs(self):
        specs = [
            {"type": "illustration", "at": "uno", "draw": "a", "continue": True, "camera": "left", "otter": True},
            {"type": "clip", "at": "dos", "query": "storm", "label": "Tormenta", "frame": "full", "draw": "b"},
        ]
        first, clip = scenes.time_shots(specs, self.TIMES.get, 8.0)
        self.assertEqual((first.follows, first.camera, first.center.otter), (True, "left", True))
        self.assertEqual((clip.frame, clip.center.query, clip.center.label, clip.center.draw), ("full", "storm", "Tormenta", "b"))


class TestAnimaticRenderer(_TempDirCase):
    def _renderer(self, picture):
        theme = fx.Theme(320, 180, FONT, (255, 79, 94))
        return scenes.SceneRenderer(theme, self.path("r"), editor.DOODLE_COLOR, picture, None,
                                    {"boom": "b.wav", "pop": "p.wav", "scribble": "s.wav", "stamp": "t.wav"},
                                    font_path=scenes.hand_font_path(), doodle=True)

    def test_illustrations_move_and_dissolve(self):
        photo = scenes.prepare_picture(_icon(self.path("i.png")), allow_cutout=False)
        renderer = self._renderer(lambda item: photo)
        moves = []
        real = scenes.motion_clip

        def spy(pictures, output, frames, size, path=None, fade=0.2, trim=False):
            moves.append((path(0, 0.0), path(0, 1.0), frames, trim))
            return real(pictures, output, frames, size, path=path, fade=fade, trim=trim)

        shot = scenes.Scene("illustration", 1.0, 3.0, text="05:30", center=scenes.SceneItem(), camera="left", fade_in=0.3, overlap=0.3)
        with patch.object(scenes, "motion_clip", side_effect=spy):
            overlays, sounds = renderer.build(shot, "a")
        (zoom_start, pan_start), (zoom_end, pan_end), frames, trim = moves[0]
        self.assertEqual((zoom_start, zoom_end), (1.06, 1.06))
        self.assertGreater(pan_start, pan_end)  # drifts to the left
        self.assertTrue(trim)  # a drawn page's white border is cut away
        self.assertEqual(frames, int(round(2.3 * scenes.FPS)) + 2)  # long enough to stay under the next shot
        self.assertEqual(overlays[0].fade_in, 0.3)
        self.assertTrue(all(o.end == 3.3 and o.fade_out == 0 for o in overlays))
        caption = overlays[1]
        self.assertGreater(int(float(caption.y.split("+")[0])), 180 * 0.55)  # the caption sits low, off the subject
        lingering = scenes.Scene("illustration", 0, 2, center=scenes.SceneItem(), linger=0.3)
        with patch.object(scenes, "motion_clip", side_effect=spy):
            overlays, _ = renderer.build(lingering, "b")
        self.assertEqual((overlays[0].end, overlays[0].fade_out), (2.3, 0.3))

    def test_clips_in_a_frame_or_full_screen(self):
        stock = _video(self.path("stock.mp4"))
        renderer = self._renderer(lambda item: None)
        card = scenes.Scene("clip", 0, 1.0, center=scenes.SceneItem(label="central eléctrica"), media=stock)
        overlays, sounds = renderer.build(card, "card")
        video = overlays[0]
        self.assertEqual((video.mode, video.start, video.end), ("media", 0, 1.0))
        self.assertTrue(os.path.isfile(video.source))
        self.assertEqual(len(overlays), 2)  # the video and its hand-lettered label
        with Image.open(self.path("r/scenes/card/video-back.png")) as back:
            self.assertEqual(back.size, (320, 180))
        frame = self.path("card.jpg")
        self.assertEqual(fx.extract_frame(utils.get_ffmpeg_binary(), video.source, frame, 0.5), frame)
        with Image.open(frame) as picture:
            corner = picture.convert("RGB").getpixel((3, 3))
            self.assertGreater(corner[0], 180)  # the drawn canvas around the framed video
        full = scenes.Scene("clip", 0, 1.0, frame="full", center=scenes.SceneItem(label="tormenta"), media=stock)
        overlays, _ = renderer.build(full, "full")
        self.assertEqual(overlays[0].mode, "media")
        with self.assertRaises(ValueError):
            renderer.build(scenes.Scene("clip", 0, 1.0, center=scenes.SceneItem()), "none")
        self.assertEqual(fx.extract_frame(utils.get_ffmpeg_binary(), self.path("missing.mp4"), self.path("x.jpg")), "")

    def test_reactions(self):
        otter = scenes.prepare_picture(os.path.join(NUTRIA, "personaje", "sorprendido.png"), trust_alpha=True)
        meme_path = self.path("meme.jpg")
        Image.new("RGB", (200, 260), (90, 120, 160)).save(meme_path)
        meme_photo = scenes.prepare_picture(meme_path, allow_cutout=False)
        meme_photo.info["framed"] = True
        for name, picture, media in (("otter", otter, ""), ("photo", meme_photo, ""), ("video", None, _video(self.path("m.mp4"), 1.0, "180x320"))):
            renderer = self._renderer(lambda item, p=picture: p)
            shot = scenes.Scene("meme", 2.0, 3.5, text="¡¿QUÉ?!", center=scenes.SceneItem(), media=media, mood="shock")
            overlays, sounds = renderer.build(shot, name)
            self.assertEqual(overlays[0].mode, "media", name)
            self.assertIn((2.0, "b.wav", scenes._SFX["boom"]), sounds)
            self.assertEqual(len(overlays), 2)  # the reaction and its caption
        with self.assertRaises(ValueError):
            self._renderer(lambda item: None).build(scenes.Scene("meme", 0, 1, center=scenes.SceneItem()), "empty")

    def test_single_drawings_appear_drawn_on_a_paper_blob(self):
        drawing = scenes.prepare_picture(_icon(self.path("d.png")))
        renderer = self._renderer(lambda item: drawing)
        overlays, sounds = renderer.build(scenes.Scene("single", 0, 3, center=scenes.SceneItem(label="bombilla")), "s")
        self.assertIn("spot", os.path.basename(overlays[0].source))
        self.assertIn("single", os.path.basename(overlays[1].source))
        self.assertIn((0.2, "s.wav", scenes._SFX["scribble"]), [(round(t, 2), p, g) for t, p, g in sounds])
        frames = scenes.reveal_frames(Image.open(self.path("d.png")).convert("RGBA"), frames=6)
        self.assertEqual(len(frames), 6)
        alphas = [sum(f.getchannel("A").getdata()) for f in frames]
        self.assertEqual(alphas, sorted(alphas))  # appears progressively
        self.assertLess(alphas[0], alphas[-1] * 0.6)
        blob = scenes.spot(100, 80, (250, 220, 150), seed=3)
        self.assertEqual(blob.size, (100, 80))
        self.assertEqual(blob.getpixel((50, 40))[3], 255)
        self.assertEqual(blob.getpixel((0, 0))[3], 0)
        rays = scenes.sunburst((120, 80), (240, 120, 130))
        self.assertNotEqual(rays.getpixel((110, 41)), rays.getpixel((110, 53)))
        photo = scenes.prepare_picture(_icon(self.path("p.png")), allow_cutout=False)
        photo.info["framed"] = True
        overlays, _ = self._renderer(lambda item: photo).build(scenes.Scene("single", 0, 3, center=scenes.SceneItem(label="x")), "p")
        self.assertFalse(any("spot" in os.path.basename(o.source) for o in overlays))  # photos keep their frame, no blob

    def test_drawings_keep_wobbling_while_they_fade_out(self):
        drawing = scenes.prepare_picture(_icon(self.path("w.png")))
        renderer = self._renderer(lambda item: drawing)
        shot = scenes.Scene("single", 0, 2.0, center=scenes.SceneItem(label="x"), linger=0.3)
        overlays, _ = renderer.build(shot, "w")
        listing = Path(overlays[1].source).read_text(encoding="utf-8")
        total = sum(float(line.split()[1]) for line in listing.splitlines() if line.startswith("duration"))
        self.assertGreaterEqual(total, 2.3 - 0.08 - 0.15)  # until the end of the fade, not the cut
        self.assertEqual((overlays[1].end, overlays[1].fade_out), (2.3, 0.3))

    def test_compare_divider_waits_for_the_first_element(self):
        renderer = self._renderer(lambda item: None)
        items = [scenes.SceneItem(label="a", time=2.0), scenes.SceneItem(label="b", time=3.0)]
        overlays, _ = renderer.build(scenes.Scene("compare", 0, 5, items=items), "c")
        divider = next(o for o in overlays if "divider" in o.source)
        self.assertEqual(divider.start, 2.0)

    def test_boom_sound(self):
        boom = fx.synthesize_sfx("boom")
        self.assertGreater(len(boom), 0.5 * fx.SFX_SAMPLE_RATE)
        self.assertLessEqual(float(abs(boom).max()), 0.71)
        self.assertIn("boom", fx.SFX_NAMES)


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


class TestAnimaticEditor(_TempDirCase):
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
        options.setdefault("openers", False)
        return editor.Editor(editor.EditOptions(assets_dir=NUTRIA, **options), theme, self.temp_dir, segments, narrations)

    def test_storyboard_targets_and_gap_filling(self):
        ed = self._editor(shot_seconds=3.0, clips="more", memes="otter")
        board = {"segments": [
            {"index": 0, "shots": []},
            {"index": 1, "shots": [
                {"type": "illustration", "at": "palabra0", "draw": "a dark room"},
                {"type": "single", "at": "palabra40", "label": "x", "draw": "a plug"},
            ]},
            {"index": 2, "shots": []},
        ]}
        extra = {1: [{"type": "illustration", "at": "palabra20", "draw": "the room lights up", "continue": True}]}
        with patch.object(editor.llm, "generate_storyboard", return_value=llm.normalize_storyboard(board, 3, [])) as plan, \
                patch.object(editor.llm, "generate_storyboard_gaps", return_value=extra) as gaps:
            ed.make_plan()
            ed._fill_shot_gaps()
        payload = plan.call_args.args[0]
        self.assertEqual((payload[1]["seconds"], payload[1]["shots"]), (24.3, 8))
        self.assertEqual((plan.call_args.kwargs["clips"], plan.call_args.kwargs["memes"]), ("more", True))
        asked = gaps.call_args.args[0]
        self.assertEqual(asked[0]["index"], 1)
        self.assertIn("palabra20", asked[0]["text"])
        self.assertTrue(asked[0]["showing"].startswith("illustration: a dark room"))
        self.assertGreaterEqual(asked[0]["shots"], 2)
        self.assertEqual(ed.plan[1]["shots"][-1]["draw"], "the room lights up")
        saved = json.loads(Path(self.path("edit-plan.json")).read_text(encoding="utf-8"))
        self.assertEqual(len(saved["segments"][1]["shots"]), 3)

    def test_word_times_follow_the_text(self):
        timed = editor.word_times("Hola, nutrias. ¿Qué es V=I·R?", None, 4.0)
        self.assertEqual([word for word, _ in timed], ["Hola,", "nutrias.", "¿Qué", "es", "V=I·R?"])
        times = [t for _, t in timed]
        self.assertEqual(times, sorted(times))
        self.assertEqual(times[0], 0.0)
        self.assertEqual(editor.word_times("", None, 1.0), [])

    def test_continued_illustrations_are_drawn_from_the_previous_frame(self):
        ed = self._editor()
        asked = []

        def draw(description, scene=False, mascot="", previous="", **kwargs):
            asked.append((description, previous, bool(mascot)))
            return _icon(self.path(f"d{len(asked)}.png"))

        Item = scenes.SceneItem
        chain = [
            scenes.Scene("illustration", 0, 2, center=Item(draw="the otter in the dark", otter=True)),
            scenes.Scene("illustration", 2, 4, center=Item(draw="it flips the switch", otter=True), follows=True),
        ]
        with patch.object(editor.gemini_media, "draw", side_effect=draw):
            ed._illustration_chain([(scene, "") for scene in chain])
        self.assertEqual(asked[0], ("the otter in the dark", "", True))
        self.assertEqual(asked[1], ("it flips the switch", self.path("d1.png"), False))  # drawn from frame 1, no extra mascot
        self.assertTrue(all(ed._item_pictures[id(s.center)].info["framed"] for s in chain))

    def test_clips_are_found_checked_and_credited(self):
        ed = self._editor()
        stock = _video(self.path("stock.mp4"))
        found = [MaterialInfo(provider="pexels", url=f"https://v/{n}", duration=8,
                              source_info={"provider": "pexels", "creator": {"name": "Ana"}, "source_page": f"https://pexels.com/{n}"})
                 for n in (1, 2)]
        shot = scenes.Scene("clip", 0, 3, center=scenes.SceneItem(query="power plant", at="palabra3", draw="a power plant"))
        with patch.object(editor.material, "search_videos_pexels", return_value=found) as pexels, \
                patch.object(editor.material, "search_videos_pixabay", return_value=[]), \
                patch.object(editor.material, "save_video", return_value=stock), \
                patch.object(editor.gemini_media, "choose_picture", side_effect=[-1, 0]) as check:
            ed._clip_media(shot, self.TEXT)
        self.assertEqual(pexels.call_args.args[0], "power plant")
        self.assertEqual(check.call_count, 2)  # the first clip did not pass the check
        self.assertEqual(check.call_args.kwargs["purpose"], "clip")
        self.assertEqual(shot.media, stock)
        self.assertIn("Video: Ana / Pexels (https://pexels.com/2)", ed.credits)
        self.assertTrue(ed._has_picture(shot))

        nothing = scenes.Scene("clip", 0, 3, center=scenes.SceneItem(query="power plant", draw="a power plant at dusk"))
        with patch.object(editor.material, "search_videos_pexels", side_effect=ValueError("no key")), \
                patch.object(editor.material, "search_videos_pixabay", return_value=found), \
                patch.object(editor.gemini_media, "draw", return_value=_icon(self.path("drawn.png"))):
            ed._clip_media(nothing)  # both clips were already used
        self.assertEqual(nothing.type, "illustration")  # drawn instead
        self.assertTrue(ed._has_picture(nothing))
        empty = scenes.Scene("clip", 0, 3, center=scenes.SceneItem(query="x"))
        with patch.object(editor.material, "search_videos_pexels", return_value=[]), \
                patch.object(editor.material, "search_videos_pixabay", return_value=[]):
            ed._clip_media(empty)
        self.assertFalse(ed._has_picture(empty))

    def test_memes_from_a_folder_or_the_otter(self):
        folder = self.path("memes")
        os.makedirs(os.path.join(folder, "sorpresa"))
        os.makedirs(os.path.join(folder, "otros"))
        Image.new("RGB", (100, 80), "red").save(os.path.join(folder, "sorpresa", "gato.png"))
        _video(os.path.join(folder, "risa-perro.mp4"), 0.5, "160x120")
        Image.new("RGB", (10, 10)).save(os.path.join(folder, "otros", "x.png"))
        Path(folder, "README.md").write_text("hola", encoding="utf-8")
        library = editor.meme_library(folder)
        self.assertEqual(sorted(library), ["laugh", "shock"])
        self.assertEqual(editor.meme_library(self.path("missing")), {})

        ed = self._editor(memes="folder", memes_dir=folder)
        shock = scenes.Scene("meme", 0, 2, mood="shock", center=scenes.SceneItem(at="palabra1"))
        laugh = scenes.Scene("meme", 5, 7, mood="laugh", center=scenes.SceneItem(at="palabra9"))
        sad = scenes.Scene("meme", 9, 11, mood="sad", center=scenes.SceneItem(at="palabra19", draw="the otter crying"))
        with patch.object(editor.gemini_media, "draw", return_value="") as draw:
            for shot in (shock, laugh, sad):
                ed._meme_picture(shot)
        self.assertTrue(ed._item_pictures[id(shock.center)].info["framed"])
        self.assertTrue(laugh.media.endswith("risa-perro.mp4"))
        draw.assert_called_once()  # no sad meme: the otter is drawn, and its pose when that fails
        self.assertIsNotNone(ed._item_pictures[id(sad.center)])
        self.assertTrue(all(ed._has_picture(s) for s in (shock, laugh, sad)))

        otter = self._editor(memes="otter", memes_dir=folder)
        mind = scenes.Scene("meme", 0, 2, mood="mindblown", center=scenes.SceneItem(draw="the otter's head exploding"))
        with patch.object(editor.gemini_media, "draw", return_value=_icon(self.path("boom.png"))) as draw:
            otter._meme_picture(mind)
        self.assertTrue(draw.call_args.kwargs["mascot"])  # its own reaction, drawn from its picture
        self.assertEqual(mind.media, "")

    def test_crossfades_and_drawing_warnings(self):
        Item = scenes.SceneItem
        shots = [
            scenes.Scene("single", 0, 3, center=Item()),
            scenes.Scene("illustration", 3, 6, center=Item()),
            scenes.Scene("illustration", 6, 7, center=Item()),
            scenes.Scene("sequence", 7, 9, items=[Item()]),
        ]
        editor.Editor._crossfade(shots)
        fade = scenes.CROSSFADE_SECONDS
        self.assertEqual([s.overlap for s in shots], [fade, 0.25, 0.0, 0.0])
        self.assertEqual([s.fade_in for s in shots], [0.0, fade, 0.25, 0.0])
        self.assertEqual(shots[2].linger, fade)  # back to the canvas: it fades out over the drawings

        ed = self._editor(max_drawings=1)
        ed._failed_drawings = 2
        ed._drawings = 1
        with patch.dict(gemini_media._state, {"imagen_failed": "403 denied", "last_error": "timeout"}):
            ed._report_drawings(3)
        text = " | ".join(ed.warnings)
        for words in ("Imagen failed (403 denied)", "2 drawings failed (last error: timeout)", "--max-drawings"):
            self.assertIn(words, text)

    def test_icons_are_checked_before_use(self):
        ed = self._editor(look="footage")
        icon_a, icon_b = _icon(self.path("a.png")), _icon(self.path("b.png"), color=(0, 200, 0, 255))
        paths = {"😘": icon_a, "a steady current": "", "🔋": icon_b}
        item = scenes.SceneItem(label="CORRIENTE CONTINUA", icon="😘", draw="a steady current")
        with patch.object(editor.icons, "fetch", side_effect=lambda name: paths.get(name, "")), \
                patch.object(editor.icons, "alternatives", return_value=["🔋"]), \
                patch.object(editor.gemini_media, "choose_picture", return_value=1) as check:
            image = ed._icon_picture(item)
        self.assertIsNotNone(image)
        self.assertEqual(check.call_args.args[0], [icon_a, icon_b])
        self.assertEqual(check.call_args.kwargs["purpose"], "icon")
        self.assertIn("🔋", ed._used_icons)
        wrong = scenes.SceneItem(label="PRIMER CONTACTO", icon="🀄", draw="first touch")
        with patch.object(editor.icons, "fetch", return_value=icon_a), patch.object(editor.icons, "alternatives", return_value=[]), \
                patch.object(editor.gemini_media, "choose_picture", return_value=-1):
            self.assertIsNone(ed._icon_picture(wrong))  # better no icon than a wrong one
        with patch.object(editor.icons, "fetch", return_value=""), patch.object(editor.icons, "alternatives", side_effect=OSError):
            self.assertIsNone(ed._icon_picture(scenes.SceneItem(label="x", icon="?", draw="y")))


class TestAnimaticOptions(unittest.TestCase):
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

    def test_cli_flags(self):
        _, generate, _ = self._run()
        edit = generate.call_args.kwargs["edit"]
        self.assertEqual((edit.shot_seconds, edit.clips, edit.memes, edit.memes_dir, edit.drawing_style), (5.0, "some", "off", "", "cartoon"))
        _, generate, _ = self._run("--look", "doodle", "--shot-seconds", "1", "--clips", "more", "--memes", "folder",
                                   "--memes-dir", self.temp_dir, "--drawing-style", "ink")
        edit = generate.call_args.kwargs["edit"]
        self.assertEqual((edit.shot_seconds, edit.clips, edit.memes, edit.memes_dir, edit.drawing_style),
                         (3.0, "more", "folder", os.path.abspath(self.temp_dir), "ink"))  # never faster than every 3 s
        code, _, stderr = self._run("--memes-dir", os.path.join(self.temp_dir, "missing"))
        self.assertEqual(code, 2)
        self.assertIn("--memes-dir folder not found", stderr)

    def test_studio_settings(self):
        settings = studio.RenderSettings(look="doodle", shot_seconds=2.5, clips="more", memes="folder", memes_dir="/m", drawing_style="ink")
        argv = studio.build_argv(settings, script_file="s.json")
        for pair in (["--shot-seconds", "2.5"], ["--clips", "more"], ["--memes", "folder"], ["--memes-dir", "/m"], ["--drawing-style", "ink"]):
            index = argv.index(pair[0])
            self.assertEqual(argv[index + 1], pair[1])
        plain = studio.build_argv(studio.RenderSettings(look="footage"), script_file="s.json")
        for flag in ("--shot-seconds", "--clips", "--memes", "--drawing-style"):
            self.assertNotIn(flag, plain)
        otter = studio.build_argv(studio.RenderSettings(memes="otter", memes_dir="/ignored"), script_file="s.json")
        self.assertIn("otter", otter)
        self.assertNotIn("--memes-dir", otter)
        defaults = studio.RenderSettings()
        self.assertEqual((defaults.clips, defaults.memes, defaults.shot_seconds), ("some", "off", 5.0))

    def test_memes_folder_is_documented_and_private(self):
        folder = Path(utils.resource_dir("memes"))
        self.assertIn("Derechos de autor", (folder / "README.md").read_text(encoding="utf-8"))
        self.assertIn("*", (folder / ".gitignore").read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    unittest.main()
