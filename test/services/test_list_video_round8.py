"""Round 8: an animated documentary. Real historical pictures, the best Gemini image models with character
sheets and audits in context, optional Veo shots, colour cards that open each section, the section's label over
the pictures, a calmer pace and a review of every picture before rendering."""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import ListVideoItem, ListVideoScript, VideoParams
from app.services import gemini_media, list_video, llm, studio, web_images
from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import list_video_scenes as scenes
from app.utils import utils

import list_video as list_video_cli  # noqa: E402

FONT = str(Path(utils.font_dir()) / "BeVietnamPro-Bold.ttf")
NUTRIA = utils.resource_dir(os.path.join("characters", "nutria"))
FRESH_STATE = {"imagen_failed": "", "last_error": "", "gone": {}, "no_size": set()}


def _picture(path, color=(220, 60, 60), size=(320, 180)):
    image = Image.new("RGB", size, color)
    ImageDraw.Draw(image).ellipse((40, 30, 160, 150), fill=(250, 240, 200), outline=(0, 0, 0), width=6)
    image.save(path)
    return path


def _png(size=(64, 36), color="white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _video(path, seconds=1.0, size="320x180", source="testsrc2"):
    subprocess.run(
        [utils.get_ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}=size={size}:rate=30",
         "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        check=True,
    )
    return path


def _frames(video):
    probe = subprocess.run([utils.get_ffmpeg_binary(), "-i", video], capture_output=True, text=True).stderr
    width, height = (int(v) for v in re.search(r", (\d+)x(\d+)", probe).groups())
    raw = subprocess.run([utils.get_ffmpeg_binary(), "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def _narration(seconds, frames):
    return editor.Narration(pcm=b"", speech_seconds=seconds, frames=frames)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.temp_dir, name)


class _FakeImageClient:
    """A google-genai client that draws a picture for every generate_content call and records the calls."""

    def __init__(self, calls, answers=None):
        self.calls, self.answers = calls, list(answers or [])

    def __call__(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def models(self):
        return self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0) if self.answers else _png()
        if isinstance(answer, Exception):
            raise answer
        part = SimpleNamespace(inline_data=SimpleNamespace(data=answer))
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


class TestImageModels(_TempDirCase):
    def setUp(self):
        super().setUp()
        for patcher in (
            patch.dict(gemini_media._state, dict(FRESH_STATE, gone={}, no_size=set())),
            patch.object(gemini_media, "RETRY_SECONDS", (0.0, 0.0)),
            patch.object(gemini_media, "_client_kwargs", return_value={}),
            patch.object(gemini_media, "_cache_path", side_effect=lambda model, key: self.path(f"{abs(hash((model, key)))}.png")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_quality_picks_the_models(self):
        self.assertEqual(gemini_media.image_models("standard", app_config={})[0], "gemini-3.1-flash-image")
        self.assertEqual(gemini_media.image_models("high", key_frame=True, app_config={})[0], "gemini-3-pro-image")
        self.assertEqual(gemini_media.image_models("high", key_frame=False, app_config={})[0], "gemini-3.1-flash-image")
        self.assertEqual(gemini_media.image_models("max", key_frame=False, app_config={})[0], "gemini-3-pro-image")
        self.assertEqual(gemini_media.image_models("economy", app_config={})[0], "gemini-2.5-flash-image")
        configured = gemini_media.image_models("high", app_config={"gemini_image_model": "my-model"})
        self.assertEqual(configured[0], "my-model")
        self.assertIn("gemini-3-pro-image", configured)
        gemini_media._state["gone"]["gemini-3-pro-image"] = "404"
        self.assertNotIn("gemini-3-pro-image", gemini_media.image_models("max", app_config={}))
        self.assertTrue(set(gemini_media.IMAGE_QUALITIES) >= {"economy", "standard", "high", "max"})

    def test_a_retired_model_hands_over_to_the_next_one(self):
        calls = []
        client = _FakeImageClient(calls, [RuntimeError("404 NOT_FOUND: model gemini-3-pro-image was not found")])
        with patch("google.genai.Client", client):
            path = gemini_media.draw("a telescope", scene=True, quality="max", app_config={})
        self.assertTrue(os.path.isfile(path))
        self.assertEqual([c["model"] for c in calls], ["gemini-3-pro-image", "gemini-3-pro-image-preview"])
        self.assertIn("gemini-3-pro-image", gemini_media.unavailable_models())
        self.assertEqual(calls[1]["config"].image_config.image_size, "2K")  # a whole scene is drawn sharp

    def test_character_sheets_mascot_and_art_direction_guide_a_drawing(self):
        calls = []
        sheet = _picture(self.path("newton.png"), color=(90, 60, 40))
        otter = _picture(self.path("otter.png"), color=(40, 140, 130))
        with patch("google.genai.Client", _FakeImageClient(calls)):
            gemini_media.draw("Newton reads under an apple tree", scene=True, mascot=otter,
                              characters=[("Isaac Newton", sheet)], art="warm candle-lit interiors", app_config={})
            gemini_media.draw("the same, the apple falls", previous=sheet, characters=[("Isaac Newton", sheet)], app_config={})
        contents = calls[0]["contents"]
        self.assertEqual(contents[0], "Reference picture 1:")  # several pictures are numbered
        self.assertEqual(contents[2], "Reference picture 2:")
        prompt = contents[-1]
        self.assertIn("Reference picture 1 shows the channel's mascot", prompt)
        self.assertIn("Reference picture 2 is the character sheet of Isaac Newton", prompt)
        self.assertIn("Art direction of this video: warm candle-lit interiors", prompt)
        self.assertIn("fewer details rather than wrong ones", prompt)  # simple shapes, drawn right
        follow = calls[1]["contents"][-1]
        self.assertIn("Reference picture 1 is the previous frame", follow)
        self.assertIn("Reference picture 2 is the character sheet of Isaac Newton", follow)

    def test_flat_style_and_character_sheet(self):
        calls = []
        portrait = _picture(self.path("portrait.png"), color=(120, 120, 120), size=(180, 240))
        with patch("google.genai.Client", _FakeImageClient(calls)):
            gemini_media.draw("a voltaic pile", style="flat", app_config={})
            sheet = gemini_media.draw_character("Marie Curie", "a woman in a long black dress, hair in a bun", portrait,
                                                app_config={}, quality="high")
            again = gemini_media.draw_character("Marie Curie", "a woman in a long black dress, hair in a bun", portrait,
                                                app_config={}, quality="high")
        self.assertIn("Flat minimalist vector", calls[0]["contents"])
        self.assertEqual(sheet, again)  # drawn once
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["model"], "gemini-3-pro-image")
        self.assertEqual(calls[1]["config"].image_config.aspect_ratio, "3:4")
        self.assertIn("real portrait of Marie Curie", calls[1]["contents"][-1])
        self.assertIn("full-body", calls[1]["contents"][-1])
        self.assertEqual(gemini_media.draw_character("", "x", app_config={}), "")

    def test_drawings_are_audited_against_what_is_said(self):
        picture = _picture(self.path("telescope.png"))
        prompt = gemini_media.build_drawing_check_prompt("Galileo with his telescope", context="Galileo apuntó al cielo",
                                                         characters=["Galileo Galilei"])
        for words in ('"Galileo apuntó al cielo"', "fits what the narrator says", "Galileo Galilei", "telescope looks like",
                      '"fix"'):
            self.assertIn(words, prompt)
        with patch.object(gemini_media, "_ask", return_value='{"ok": false, "reason": "broken", "fix": "a simple brass tube"}'):
            self.assertEqual(gemini_media.audit_drawing(picture, "x", app_config={}), (False, "a simple brass tube"))
            self.assertFalse(gemini_media.check_drawing(picture, "x", app_config={}))
        with patch.object(gemini_media, "_ask", return_value='{"ok": true}'):
            self.assertEqual(gemini_media.audit_drawing(picture, "x", app_config={}), (True, ""))
        with patch.object(gemini_media, "_ask", side_effect=RuntimeError("quota")):
            self.assertEqual(gemini_media.audit_drawing(picture, "x", app_config={}), (None, ""))

    def test_archive_and_portrait_checks_want_authentic_pictures(self):
        archive = gemini_media.build_choice_prompt("En 1665 llegó la peste", "Great Plague of London 1665 engraving", 3, "es",
                                                   purpose="archive")
        for words in ("REAL document", "reject modern staged", "AI-generated", "answer 0"):
            self.assertIn(words, archive)
        portrait = gemini_media.build_choice_prompt("Isaac Newton", "Isaac Newton portrait", 2, "es", purpose="portrait")
        self.assertIn("real portrait of exactly that person", portrait)
        self.assertIn("archive", gemini_media.PURPOSES)


class _FakeVideoClient:
    """A google-genai client whose generate_videos finishes after one poll."""

    def __init__(self, calls, video=b"MP4DATA", uri=False, fail=None):
        self.calls, self.video, self.uri, self.fail = calls, video, uri, list(fail or [])
        self.models, self.operations, self.files = self, self, self

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def generate_videos(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail.pop(0)
        return SimpleNamespace(done=False, name="op")

    def get(self, operation):
        video = SimpleNamespace(video_bytes=None if self.uri else self.video, uri="https://files/v1" if self.uri else None)
        return SimpleNamespace(done=True, error=None, response=SimpleNamespace(generated_videos=[SimpleNamespace(video=video)]))

    def download(self, file):
        return self.video


class TestVeoShots(_TempDirCase):
    def setUp(self):
        super().setUp()
        for patcher in (
            patch.dict(gemini_media._state, dict(FRESH_STATE, gone={}, no_size=set())),
            patch.object(gemini_media, "VIDEO_POLL_SECONDS", 0.0),
            patch.object(gemini_media, "_video_cache_path", side_effect=lambda model, key: self.path(f"{abs(hash((model, key)))}.mp4")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.drawing = _picture(self.path("apple.png"))

    def test_a_drawing_comes_to_life_once(self):
        calls = []
        client = _FakeVideoClient(calls)
        with patch("google.genai.Client", client), \
                patch.object(gemini_media, "_client_kwargs", return_value={"vertexai": True, "project": "p", "location": "global"}):
            path = gemini_media.animate(self.drawing, "the apple slowly falls", 5.2, app_config={})
            again = gemini_media.animate(self.drawing, "the apple slowly falls", 5.2, app_config={})
        self.assertEqual(path, again)
        self.assertEqual(Path(path).read_bytes(), b"MP4DATA")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["model"], gemini_media.VIDEO_MODELS["standard"][0])
        self.assertEqual(call["config"].duration_seconds, 6)  # the shortest Veo length that covers the shot
        self.assertFalse(call["config"].generate_audio)  # the narration is the sound
        self.assertIn("the apple slowly falls", call["source"].prompt)
        self.assertEqual(call["source"].image.mime_type, "image/png")
        self.assertEqual(gemini_media.animate(self.drawing, "", app_config={}), "")
        self.assertEqual(gemini_media.animate(self.path("missing.png"), "x", app_config={}), "")

    def test_gemini_api_videos_are_downloaded_and_retired_models_skipped(self):
        calls = []
        client = _FakeVideoClient(calls, uri=True, fail=[RuntimeError("404 model veo-3.1-fast-generate-001 not found")])
        with patch("google.genai.Client", client), patch.object(gemini_media, "_client_kwargs", return_value={"api_key": "k"}):
            path = gemini_media.animate(self.drawing, "the candle flickers", 4, app_config={})
        self.assertEqual(Path(path).read_bytes(), b"MP4DATA")
        self.assertEqual([c["model"] for c in calls], list(gemini_media.VIDEO_MODELS["standard"][:2]))
        self.assertIsNone(calls[1]["config"].generate_audio)  # not sent to the Gemini API
        self.assertIn("veo-3.1-fast-generate-001", gemini_media.unavailable_models())

    def test_veo_may_use_its_own_region(self):
        calls = []
        client = _FakeVideoClient(calls)
        settings = {"gemini_video_location": "us-central1"}
        with patch("google.genai.Client", client), \
                patch.object(gemini_media, "_client_kwargs", return_value={"vertexai": True, "project": "p", "location": "global"}):
            self.assertTrue(gemini_media.animate(self.drawing, "the flame flickers", 4, app_config=settings))
        self.assertEqual(client.kwargs["location"], "us-central1")

    def test_lengths_and_models(self):
        self.assertEqual([gemini_media.video_length(s) for s in (2, 4.4, 6, 7, 30)], [4, 4, 6, 8, 8])
        self.assertEqual(gemini_media.video_models("high", {})[0], "veo-3.1-generate-001")
        self.assertEqual(gemini_media.video_models("standard", {"gemini_video_model": "veo-x"})[0], "veo-x")


class TestArchiveSearch(unittest.TestCase):
    def test_real_history_comes_from_commons_bitmaps(self):
        self.assertEqual(web_images.source_order("archive"), ["archive"])
        response = MagicMock()
        response.json.return_value = {"query": {"pages": {}}}
        with patch.object(web_images, "_request", return_value=response) as request:
            web_images.search_archive("Great Plague of London 1665 engraving")
        params = request.call_args.kwargs["params"]
        self.assertEqual(params["gsrsearch"], "Great Plague of London 1665 engraving filetype:bitmap")
        self.assertEqual(params["iiurlwidth"], 1920)


class TestPicturesOnScreen(_TempDirCase):
    def test_formulas_are_written_like_on_a_blackboard(self):
        self.assertEqual(scenes.pretty_formula("F = G*m1*m2/r^2"), "F = G × m1 × m2/r²")
        self.assertEqual(scenes.pretty_formula("E = m*c**2"), "E = m × c²")
        self.assertEqual(scenes.pretty_formula("v = sqrt(2*g*h)"), "v = √(2 × g × h)")
        self.assertEqual(scenes.pretty_formula("a <= b"), "a ≤ b")

    def test_transparent_pictures_never_get_black_corners(self):
        cutout = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
        ImageDraw.Draw(cutout).ellipse((60, 10, 140, 90), fill=(200, 30, 30, 255))
        filled = np.asarray(scenes.fit_cover(cutout, (160, 90)))
        self.assertTrue((filled[0, 0] > 240).all())
        self.assertEqual(scenes.flatten(Image.new("RGB", (4, 4), "red")).mode, "RGB")

    def test_pictures_fill_the_frame_or_sit_over_their_blurred_copy(self):
        wide, tall = Image.new("RGB", (1600, 900), "navy"), Image.new("RGB", (600, 900), "navy")
        self.assertTrue(scenes.fills_frame(wide, (1280, 720)))
        self.assertFalse(scenes.fills_frame(tall, (1280, 720)))
        canvas = scenes.contain_over_blur(tall, (320, 180))
        self.assertEqual(canvas.size, (320, 180))
        pixels = np.asarray(canvas).astype(int)
        self.assertLess(pixels[90, 5].sum(), pixels[90, 160].sum())  # the sides are the darker, blurred copy

    def test_shots_know_their_place_in_the_plan(self):
        times = {"uno": 1.0, "dos": 7.0, "tres": 8.0}
        specs = [
            {"type": "bogus", "at": "uno"},
            {"type": "illustration", "at": "uno", "draw": "a"},
            {"type": "animation", "at": "dos", "frames": [{"draw": "x"}, {"draw": "y", "at": "tres"}]},
        ]
        shots = scenes.time_shots(specs, times.get, 14.0)
        self.assertEqual([s.spec for s in shots], [1, 2])
        self.assertEqual([f.slot for f in shots[1].items], [0, 1])
        timeline = scenes.time_shots([{"type": "timeline", "items": [
            {"at": "tres", "label": "b", "date": "1900"}, {"at": "uno", "label": "a", "date": "1800"}]}], times.get, 14.0)
        self.assertEqual([(i.label, i.slot) for i in timeline[0].items], [("a", 1), ("b", 0)])  # sorted, slots kept

    def test_an_ai_shot_plays_once_and_holds_its_last_frame(self):
        source = _video(self.path("veo.mp4"), seconds=1.0, source="testsrc2")
        output = scenes.video_clip(source, self.path("held.mp4"), 75, (160, 90), hold=True, slow=1.2)
        frames = _frames(output)
        self.assertEqual(len(frames), 75)
        self.assertLess(np.abs(frames[-1].astype(int) - frames[-20].astype(int)).mean(), 1.0)  # held, not looped


class _RendererCase(_TempDirCase):
    def _renderer(self, picture, host=True):
        theme = fx.Theme(320, 180, FONT, (253, 199, 76))
        still = (lambda expression, height: Image.new("RGBA", (height // 2, height), (40, 160, 140, 255))) if host else None
        return scenes.SceneRenderer(theme, self.path("r"), editor.DOODLE_COLOR, picture, still, {},
                                    font_path=scenes.hand_font_path(), doodle=True)


class TestDoodleRendering(_RendererCase):
    def test_archive_shot_with_a_museum_caption(self):
        portrait = scenes.prepare_picture(_picture(self.path("p.png"), size=(120, 180)), allow_cutout=False)
        renderer = self._renderer(lambda item: portrait)
        shot = scenes.Scene("archive", 0.0, 2.0, text="Londres, 1665", center=scenes.SceneItem(), still=True)
        overlays, _ = renderer.build(shot, "a")
        self.assertEqual(overlays[0].mode, "media")
        self.assertEqual(len(_frames(overlays[0].source)), renderer._clip_frames(shot))
        self.assertTrue(any("caption" in o.source for o in overlays[1:]))
        with self.assertRaises(ValueError):
            self._renderer(lambda item: None).build(scenes.Scene("archive", 0, 2, center=scenes.SceneItem()), "b")

    def test_sections_open_on_their_own_colour_with_a_photo(self):
        photo = scenes.prepare_picture(_picture(self.path("einstein.png")), allow_cutout=False)
        renderer = self._renderer(lambda item: photo)
        card = scenes.Scene("opener", 0.0, 3.6, text="Albert Einstein", number=2, enter=False,
                            center=scenes.SceneItem(label="Albert Einstein", query="Albert Einstein portrait"))
        overlays, sounds = renderer.build(card, "o")
        background = Image.open(overlays[0].source).convert("RGB")
        self.assertLess(np.abs(np.asarray(background).astype(int).mean(axis=(0, 1)) - scenes.SECTION_COLORS[1]).max(), 12)
        names = [os.path.basename(o.source) for o in overlays]
        self.assertTrue(any(n.startswith("picture") for n in names))
        self.assertTrue(any(n.startswith("number") for n in names))
        self.assertTrue(any(n.startswith("title") for n in names))

    def test_the_otter_presents_a_formula_from_its_own_column(self):
        renderer = self._renderer(lambda item: None)
        shot = scenes.Scene("equation", 0.0, 4.0, text="F = G*m1*m2/r^2", center=scenes.SceneItem(label="gravedad"),
                            items=[scenes.SceneItem(symbol="F", label="fuerza", time=1.5)])
        overlays, _ = renderer.build(shot, "e")
        host = next(o for o in overlays if os.path.basename(o.source).startswith("host"))
        tokens = [o for o in overlays if os.path.basename(o.source).startswith("token")]
        folder = os.path.dirname(host.source)
        width = max(Image.open(os.path.join(folder, name)).width for name in os.listdir(folder) if re.match(r"host_\d+\.png", name))
        host_right = int(host.x.split("+")[0]) + width
        self.assertTrue(tokens)
        self.assertTrue(all(int(o.x.split("+")[0]) >= host_right for o in tokens))  # the otter never covers the formula
        self.assertIn("×", "".join(scenes.pretty_formula(shot.text)))

    def test_an_illustration_with_an_ai_video_plays_it(self):
        drawing = scenes.prepare_picture(_picture(self.path("d.png")), allow_cutout=False)
        renderer = self._renderer(lambda item: drawing)
        shot = scenes.Scene("illustration", 0.0, 2.0, center=scenes.SceneItem(), media=_video(self.path("v.mp4"), 1.0))
        overlays, _ = renderer.build(shot, "v")
        self.assertTrue(overlays[0].source.endswith("video.mp4"))
        broken = scenes.Scene("illustration", 0.0, 2.0, center=scenes.SceneItem(), media=self.path("missing.mp4"))
        overlays, _ = renderer.build(broken, "w")
        self.assertTrue(overlays[0].source.endswith("clip.mp4"))  # the drawing under a camera move instead
        self.assertEqual(broken.media, "")


class _EditorCase(_TempDirCase):
    TEXT = " ".join(f"palabra{i}" for i in range(60))

    def _editor(self, **options):
        patcher = patch.object(editor.gemini_media, "enabled", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        segments = list_video.build_segments(ListVideoScript(
            title="T", intro="Hola nutrias.", items=[ListVideoItem(name="Isaac Newton", text=self.TEXT, image_term="Newton")],
            outro="Chao."))
        narrations = [_narration(1.0, 45), _narration(24.0, 730), _narration(1.0, 45)]
        theme = fx.Theme(640, 360, FONT, fx.parse_color(fx.DEFAULT_ACCENT))
        options.setdefault("subscribe", "none")
        options.setdefault("look", "doodle")
        options.setdefault("openers", False)
        return editor.Editor(editor.EditOptions(assets_dir=NUTRIA, **options), theme, self.temp_dir, segments, narrations)

    def _found(self, name, url):
        return web_images.WebImage(path=_picture(self.path(name)), source="wikimedia", title=name, author="A", license="PD",
                                   page_url=f"https://commons/{name}", url=url)


class TestDocumentaryEditor(_EditorCase):
    def test_the_same_picture_is_never_planned_twice(self):
        plan = [
            {"shots": [
                {"type": "single", "at": "a", "draw": "a glowing sample of polonium in a small glass vial"},
                {"type": "archive", "at": "b", "query": "Marie Curie laboratory 1904"},
            ]},
            {"shots": [
                {"type": "illustration", "at": "c", "draw": "a small glass vial with a glowing polonium sample"},
                {"type": "archive", "at": "d", "query": "marie curie Laboratory 1904"},
                {"type": "illustration", "at": "e", "draw": "the same vial now glows brighter", "continue": True},
                {"type": "illustration", "at": "f", "draw": "Pierre Curie writes in a notebook by a window"},
            ]},
        ]
        self.assertEqual(editor.drop_repeats(plan), 2)
        self.assertEqual([s["at"] for s in plan[1]["shots"]], ["e", "f"])

    def test_no_icon_ever_stands_in_for_a_picture(self):
        ed = self._editor()
        frame = scenes.SceneItem(draw="the core lights up", icon="💡")
        with patch.object(editor.icons, "fetch", return_value=_picture(self.path("icon.png"))):
            self.assertIsNone(ed._scene_picture(frame))  # the frame is skipped, never a giant icon
            with patch.object(ed, "_drawing", return_value=None), patch.object(ed, "_real_picture", return_value=None):
                ed._doodle_picture(frame, "")
                self.assertIsNone(ed._item_pictures[id(frame)])
                small = scenes.SceneItem(label="1900", icon="📅", draw="a calendar")
                with patch.object(ed, "_checked_icon", return_value=scenes.prepare_picture(self.path("icon.png"))):
                    ed._doodle_picture(small, "", icon=True)  # a small element of a timeline may use one
                self.assertIsNotNone(ed._item_pictures[id(small)])
        footage = self._editor(look="footage")
        with patch.object(footage, "_icon_picture", return_value="icon") as icon:
            self.assertEqual(footage._scene_picture(scenes.SceneItem(icon="💡")), "icon")
        icon.assert_called_once()

    def test_archive_pictures_are_authentic_or_drawn(self):
        ed = self._editor()
        shot = scenes.Scene("archive", 0, 6, query="Great Plague of London 1665 engraving", query_local="peste de Londres",
                            text="Londres, 1665", center=scenes.SceneItem(at="palabra3", draw="a deserted London street",
                                                                         avoid=["https://commons/old.jpg"]))
        found = [self._found("old.jpg", "https://commons/old.jpg"), self._found("plague.jpg", "https://commons/plague.jpg")]
        with patch.object(editor.web_images, "find_candidates", side_effect=lambda q, d, kind, exclude_urls, limit:
                          [f for f in found if f.url not in exclude_urls]) as find, \
                patch.object(editor.gemini_media, "choose_picture", return_value=0) as check:
            ed._archive_picture(shot, self.TEXT)
        self.assertEqual(find.call_args_list[0].kwargs["kind"], "archive")
        self.assertIn("https://commons/old.jpg", find.call_args_list[0].kwargs["exclude_urls"])  # turned down in a review
        self.assertEqual(check.call_args.kwargs["purpose"], "archive")
        self.assertEqual(check.call_args.args[0], [found[1].path])
        image = ed._item_pictures[id(shot.center)]
        self.assertEqual((image.info["archive"], image.info["url"]), (True, "https://commons/plague.jpg"))
        self.assertTrue(ed._has_picture(shot))
        self.assertEqual(ed._stats["archive"], 1)
        drawn = scenes.Scene("archive", 0, 6, query="a painting that does not exist", center=scenes.SceneItem(draw="a lab"))
        with patch.object(editor.web_images, "find_candidates", return_value=[]), \
                patch.object(ed, "_drawing", return_value=scenes.prepare_picture(_picture(self.path("lab.png")), allow_cutout=False)):
            ed._archive_picture(drawn, self.TEXT)
        self.assertEqual(drawn.type, "illustration")  # no authentic picture: the moment is drawn
        self.assertEqual(ed._stats["archive_drawn"], 1)

    def test_recurring_people_are_drawn_from_their_character_sheet(self):
        ed = self._editor(image_quality="high", drawing_style="flat")
        ed.bible = {"style": "candle light", "characters": [
            {"id": "newton", "name": "Isaac Newton", "look": "long brown hair, dark coat", "portrait": "Isaac Newton portrait"}]}
        shot = scenes.Scene("illustration", 0, 5, center=scenes.SceneItem(draw="Newton under a tree", characters=["newton"],
                                                                           at="palabra2"))
        ed._scenes = {1: [shot]}
        portrait = self._found("newton.jpg", "https://commons/newton.jpg")
        asked = []

        def draw(description, **kwargs):
            asked.append((description, kwargs))
            return _picture(self.path(f"d{len(asked)}.png"))

        with patch.object(editor.web_images, "find_candidates", return_value=[portrait]), \
                patch.object(editor.gemini_media, "choose_picture", return_value=0) as check, \
                patch.object(editor.gemini_media, "draw_character", return_value=_picture(self.path("sheet.png"))) as sheet, \
                patch.object(editor.gemini_media, "draw", side_effect=draw), \
                patch.object(editor.gemini_media, "audit_drawing", return_value=(True, "")) as audit:
            ed._prepare_characters()
            ed._illustration_picture(shot, text=self.TEXT)
        self.assertEqual(check.call_args.kwargs["purpose"], "portrait")
        self.assertEqual(sheet.call_args.args[:3], ("Isaac Newton", "long brown hair, dark coat", portrait.path))
        self.assertEqual(sheet.call_args.kwargs["quality"], "high")
        options = asked[0][1]
        self.assertEqual(options["characters"], [("Isaac Newton", self.path("sheet.png"))])
        self.assertEqual((options["art"], options["quality"], options["style"]), ("candle light", "high", "flat"))
        self.assertEqual(audit.call_args.kwargs["characters"], ["Isaac Newton"])
        self.assertIn("palabra2", audit.call_args.kwargs["context"])  # checked against what is said meanwhile
        self.assertEqual(ed._stats["sheets"], 1)

    def test_a_rejected_drawing_is_redrawn_with_its_fix(self):
        ed = self._editor()
        asked = []

        def draw(description, **kwargs):
            asked.append(description)
            return _picture(self.path(f"a{len(asked)}.png"))

        with patch.object(editor.gemini_media, "draw", side_effect=draw), \
                patch.object(editor.gemini_media, "audit_drawing", return_value=(False, "draw the telescope as a simple tube")):
            ed._drawing("Galileo looks through his telescope", scene=True, context="Galileo miró la Luna")
        self.assertEqual(asked[1], "Galileo looks through his telescope. Make sure: draw the telescope as a simple tube")

    def test_the_longest_illustrations_come_to_life(self):
        ed = self._editor(ai_videos=1)
        short = scenes.Scene("illustration", 0, 3, center=scenes.SceneItem(draw="a", motion="leaves move"))
        long = scenes.Scene("illustration", 3, 10, center=scenes.SceneItem(draw="b", motion="the apple falls"))
        still = scenes.Scene("illustration", 10, 18, center=scenes.SceneItem(draw="c"))
        for scene in (short, long, still):
            picture = scenes.prepare_picture(_picture(self.path(f"{scene.center.draw}.png")), allow_cutout=False)
            ed._item_pictures[id(scene.center)] = picture
        ed._scenes = {1: [short, long, still]}
        with patch.object(editor.gemini_media, "animate", return_value=self.path("v.mp4")) as animate:
            ed._bring_to_life()
        animate.assert_called_once()
        self.assertEqual(animate.call_args.args[1:3], ("the apple falls", 7))
        self.assertEqual((long.media, short.media), (self.path("v.mp4"), ""))
        self.assertEqual(ed._stats["videos"], 1)
        none = self._editor()
        none._scenes = {1: [long]}
        with patch.object(editor.gemini_media, "animate") as animate:
            none._bring_to_life()
        animate.assert_not_called()  # off by default (it is paid per second)

    def test_the_section_label_stays_over_pictures_only(self):
        for kind, over in (("illustration", True), ("archive", True), ("animation", True), ("equation", False),
                           ("opener", False), ("timeline", False), ("meme", False)):
            self.assertEqual(editor._pill_over(scenes.Scene(kind, 0, 1)), over, kind)
        self.assertTrue(editor._pill_over(scenes.Scene("clip", 0, 1, frame="full")))
        self.assertFalse(editor._pill_over(scenes.Scene("clip", 0, 1, frame="card")))


class TestReviewBeforeRendering(_EditorCase):
    BOARD = {"segments": [
        {"index": 0, "shots": []},
        {"index": 1, "opener": {"query": "Isaac Newton portrait", "draw": "Isaac Newton"}, "shots": [
            {"type": "archive", "at": "palabra14", "query": "Great Plague of London 1665", "caption": "Londres, 1665",
             "draw": "a deserted street"},
            {"type": "illustration", "at": "palabra28", "draw": "Newton reads under a tree", "motion": "leaves move"},
            {"type": "animation", "at": "palabra42", "frames": [{"draw": "an apple hangs"}, {"draw": "the apple falls"}]},
        ]},
        {"index": 2, "shots": []},
    ]}

    def _drawer(self, asked):
        def draw(description, **kwargs):
            asked.append(description)
            return _picture(self.path(f"drawn{len(asked)}.png"), color=(30 * len(asked) % 255, 90, 160))
        return draw

    def test_review_pins_every_picture_and_the_render_reuses_them(self):
        ed = self._editor(openers=True)
        found = {"archive": self._found("plague.jpg", "https://commons/plague.jpg")}
        portrait = self._found("newton.jpg", "https://commons/newton.jpg")
        asked = []
        board = llm.normalize_storyboard(self.BOARD, 3, sorted(ed.poses))
        with patch.object(editor.llm, "generate_visual_bible", return_value={}), \
                patch.object(editor.llm, "generate_storyboard", return_value=board), \
                patch.object(editor.llm, "generate_storyboard_gaps", return_value={}), \
                patch.object(editor.web_images, "find_candidates",
                             side_effect=lambda q, d, kind="diagram", exclude_urls=None, limit=4:
                             [found["archive"]] if kind == "archive" else [portrait]), \
                patch.object(editor.gemini_media, "choose_picture", return_value=0), \
                patch.object(editor.gemini_media, "draw", side_effect=self._drawer(asked)), \
                patch.object(editor.gemini_media, "audit_drawing", return_value=(True, "")):
            ed.make_plan()
            review_file = ed.write_review()
        review = json.loads(Path(review_file).read_text("utf-8"))
        kinds = {shot["type"]: shot for shot in review["shots"]}
        self.assertEqual(set(kinds), {"opener", "archive", "illustration", "animation"})
        self.assertEqual(kinds["archive"]["pictures"][0]["url"], "https://commons/plague.jpg")
        self.assertEqual(kinds["archive"]["pictures"][0]["kind"], "archive")
        self.assertEqual(len(kinds["animation"]["pictures"]), 2)
        self.assertIn("palabra42", kinds["animation"]["said"])
        self.assertTrue(all(os.path.isfile(p["file"]) and "/review/" in p["file"].replace("\\", "/")
                            for shot in review["shots"] for p in shot["pictures"]))
        plan = json.loads(Path(review["plan_file"]).read_text("utf-8"))
        shots = plan["segments"][1]["shots"]
        self.assertTrue(shots[0]["image"].endswith(".jpg") or shots[0]["image"].endswith(".png"))
        self.assertTrue(all(frame.get("image") for frame in shots[2]["frames"]))
        self.assertTrue(plan["segments"][1]["opener"]["image"])
        self.assertTrue(os.path.isfile(review["reviewed_plan"]))

        # The Studio applies the decisions: the illustration is drawn again from a new description, the archive
        # picture leaves, everything else stays as it was.
        decisions = {
            kinds["illustration"]["id"]: {"action": "redo", "describe": "Newton reads by candle light"},
            kinds["archive"]["id"]: {"action": "remove"},
        }
        with patch.object(studio, "task_outputs", return_value={"folder": self.temp_dir}):
            path = studio.apply_review("task-1", decisions)
            again = studio.apply_review("task-1", decisions)  # saving twice changes nothing more
        self.assertEqual(path, again)
        reviewed = json.loads(Path(path).read_text("utf-8"))["segments"][1]["shots"]
        self.assertEqual([s["type"] for s in reviewed], ["illustration", "animation"])
        self.assertEqual(reviewed[0]["draw"], "Newton reads by candle light")
        self.assertNotIn("image", reviewed[0])

        # The final render with that plan only draws what was asked again.
        final = self._editor(openers=True, plan_file=path)
        asked.clear()
        with patch.object(editor.web_images, "find_candidates", return_value=[]) as find, \
                patch.object(editor.gemini_media, "draw", side_effect=self._drawer(asked)), \
                patch.object(editor.gemini_media, "audit_drawing", return_value=(True, "")):
            final.make_plan()
            final._prepare()
        self.assertEqual(asked, ["Newton reads by candle light"])
        find.assert_not_called()  # the opener's photo was kept
        self.assertGreaterEqual(final._stats["pinned"], 3)
        self.assertEqual([s.type for s in final._scenes[1]], ["opener", "illustration", "animation"])

    def test_a_drawing_can_become_a_real_photo_or_a_new_take(self):
        shots = [
            {"id": "1-00", "segment": 1, "shot": 0, "pictures": [{"slot": "center", "file": "x.png", "url": ""}],
             "query": "", "describe": "an apple falls"},
            {"id": "1-01", "segment": 1, "shot": 1, "pictures": [{"slot": "center", "file": "y.png", "url": "https://c/y"}],
             "query": "Principia 1687", "describe": ""},
            {"id": "1-02", "segment": 1, "shot": 2, "pictures": [], "query": "", "describe": "a / b"},
            {"id": "1-opener", "segment": 1, "shot": -1, "pictures": [{"slot": "opener", "file": "z.png", "url": "https://c/z"}],
             "query": "Newton portrait", "chapter": "Isaac Newton"},
        ]
        plan = {"segments": [{"shots": []}, {"opener": {"query": "Newton portrait", "image": "z.png"}, "shots": [
            {"type": "illustration", "at": "an apple", "draw": "an apple falls", "image": "x.png", "camera": "in"},
            {"type": "archive", "at": "in 1687", "query": "Principia 1687", "draw": "a book", "image": "y.png"},
            {"type": "animation", "at": "then", "frames": [{"draw": "a", "image": "f1.png"}, {"draw": "b", "image": "f2.png"}]},
        ]}]}
        Path(self.path("review.json")).write_text(json.dumps({"shots": shots, "plan_file": self.path("edit-plan.json"),
                                                              "reviewed_plan": self.path("review-plan.json")}), "utf-8")
        Path(self.path("review-plan.json")).write_text(json.dumps(plan), "utf-8")
        decisions = {
            "1-00": {"action": "photo", "query": "Newton apple tree engraving"},
            "1-01": {"action": "redo", "query": "Principia Mathematica title page"},
            "1-02": {"action": "redo", "describe": "a / b"},
            "1-opener": {"action": "redo", "query": "Isaac Newton Kneller portrait"},
        }
        with patch.object(studio, "task_outputs", return_value={"folder": self.temp_dir}):
            data = json.loads(Path(studio.apply_review("t", decisions)).read_text("utf-8"))
        first, archive, animation = data["segments"][1]["shots"]
        self.assertEqual((first["type"], first["query"], first["draw"], first["camera"]),
                         ("archive", "Newton apple tree engraving", "an apple falls", "in"))
        self.assertEqual((archive["query"], archive["avoid"]), ("Principia Mathematica title page", ["https://c/y"]))
        self.assertNotIn("image", archive)
        self.assertEqual([f["draw"] for f in animation["frames"]], ["a (new take 2)", "b (new take 2)"])
        self.assertFalse(any("image" in f for f in animation["frames"]))
        opener = data["segments"][1]["opener"]
        self.assertEqual((opener["query"], opener["avoid"]), ("Isaac Newton Kneller portrait", ["https://c/z"]))
        self.assertNotIn("image", opener)
        self.assertEqual(studio._new_take("a (new take 2)"), "a (new take 3)")
        normalized = llm.normalize_storyboard(data, 2, [])
        self.assertEqual(normalized[1]["shots"][1]["avoid"], ["https://c/y"])  # the plan stays valid
        self.assertEqual(normalized[1]["opener"]["avoid"], ["https://c/z"])
        with patch.object(studio, "task_outputs", return_value={"folder": self.path("nowhere")}):
            with self.assertRaises(ValueError):
                studio.apply_review("t", decisions)

    def test_pins_are_found_next_to_their_plan(self):
        plan = [{"opener": {"query": "x", "image": "review/o.png"}, "shots": [
            {"type": "illustration", "image": "review/a.png", "video": "/abs/v.mp4"},
            {"type": "animation", "frames": [{"draw": "a", "image": "review/f.png"}]}]}]
        editor.pin_paths(plan, "/task")
        self.assertEqual(plan[0]["opener"]["image"], os.path.normpath("/task/review/o.png"))
        self.assertEqual(plan[0]["shots"][0]["video"], "/abs/v.mp4")
        self.assertEqual(plan[0]["shots"][1]["frames"][0]["image"], os.path.normpath("/task/review/f.png"))


class TestReviewStopsBeforeRendering(unittest.TestCase):
    def setUp(self):
        self.task_id = str(uuid4())
        self.addCleanup(shutil.rmtree, os.path.join(utils.storage_dir(), "tasks", self.task_id), True)

    def test_service_returns_the_review_without_rendering(self):
        image = str(Path(__file__).parent.parent / "resources" / "1.png")
        script = ListVideoScript(title="T", intro="Hola.", items=[ListVideoItem(name="A", text="Una frase.", image_file=image)],
                                 outro="Chao.")
        params = VideoParams(video_subject="t", video_source="local", voice_name="no-voice")
        with patch.object(list_video.editor_service.Editor, "make_plan", return_value=[]), \
                patch.object(list_video.editor_service.Editor, "write_review", return_value="/task/review.json"), \
                patch.object(list_video, "render_segment_video") as render:
            result = list_video.generate_list_video(self.task_id, script, params, edit=editor.EditOptions(look="doodle"),
                                                    review=True)
        render.assert_not_called()
        self.assertEqual(result["review_file"], "/task/review.json")
        self.assertTrue(result["plan_file"].endswith("edit-plan.json"))
        with self.assertRaises(list_video.ListVideoError):
            list_video.generate_list_video(self.task_id, script, params, review=True)


class TestCliAndStudio(_TempDirCase):
    def _run(self, *extra):
        script = self.path("s.json")
        Path(script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")
        with patch.object(list_video, "generate_list_video", return_value={"review_file": "/r.json"}) as generate, \
                patch("sys.stdout", io.StringIO()) as out:
            code = list_video_cli.run(["--script", script, "--look", "doodle", *extra])
        return code, generate.call_args.kwargs if generate.called else None, out.getvalue()

    def test_cli_options(self):
        code, kwargs, out = self._run("--review", "--image-quality", "high", "--ai-videos", "3", "--no-director-review",
                                      "--shot-seconds", "20", "--drawing-style", "flat")
        self.assertEqual(code, 0)
        edit = kwargs["edit"]
        self.assertTrue(kwargs["review"])
        self.assertEqual((edit.image_quality, edit.ai_videos, edit.director_review, edit.shot_seconds, edit.drawing_style),
                         ("high", 3, False, 8.0, "flat"))
        self.assertEqual(json.loads(out.strip().splitlines()[-1])["review_file"], "/r.json")
        _, kwargs, _ = self._run()
        self.assertEqual((kwargs["edit"].shot_seconds, kwargs["edit"].image_quality, kwargs["review"]), (5.0, "standard", False))
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            self._run("--review", "--no-edit")

    def test_studio_settings_and_command(self):
        old = studio.RenderSettings.from_dict({"version": 2, "shot_seconds": 3.0, "aspect": "16:9"})
        self.assertEqual((old.shot_seconds, old.image_quality, old.ai_videos, old.version), (5.0, "standard", 0, 3))
        kept = studio.RenderSettings.from_dict({"version": 2, "shot_seconds": 4.5})
        self.assertEqual(kept.shot_seconds, 4.5)
        settings = studio.RenderSettings(image_quality="max", ai_videos=4, director_review=False, drawing_style="flat")
        task_id = str(uuid4())
        argv = studio.build_argv(settings, script_file="s.json", task_id=task_id, review=True)
        for flag in (["--image-quality", "max"], ["--ai-videos", "4"], ["--drawing-style", "flat"], ["--task-id", task_id]):
            self.assertIn(flag[0], argv)
            self.assertEqual(argv[argv.index(flag[0]) + 1], flag[1])
        self.assertIn("--no-director-review", argv)
        self.assertIn("--review", argv)
        self.assertNotIn("--review", studio.build_argv(studio.RenderSettings(), script_file="s.json"))
        self.assertNotIn("--ai-videos", studio.build_argv(studio.RenderSettings(), script_file="s.json"))
        parsed = list_video_cli.build_parser().parse_known_args(argv)[0]  # the CLI understands every flag
        self.assertEqual((parsed.image_quality, parsed.ai_videos, parsed.review), ("max", 4, True))

    def test_progress_counts_pictures_and_reports_a_ready_review(self):
        log = "\n".join([
            "x | INFO | list video segment [1/5] 1. A",
            "x | INFO | m:f - drawn: 'an apple'",
            "x | INFO | m:f - archive picture: 'Principia'",
            "x | INFO | m:f - character sheet drawn: 'Newton'",
        ])
        fraction, stage = studio.progress_from_log(log)
        self.assertEqual(stage, "Preparando las imágenes: 3 listas")
        self.assertAlmostEqual(fraction, 0.38)
        fraction, stage = studio.progress_from_log(log + "\nx | SUCCESS | m:f - list video review ready: /r.json")
        self.assertEqual((fraction, stage), (1.0, "Revisión lista: abre Revisar"))
        fraction, stage = studio.progress_from_log(log + "\nx | INFO | list video segment [2/5] 2. B")
        self.assertEqual(stage, "Montando la sección 2 de 5")

    def test_plans_edited_in_the_studio_stay_storyboards(self):
        data = {"segments": [{"index": 0, "shots": [{"type": "illustration", "at": "a", "draw": "x", "image": "r.png"},
                                                    {"type": "single", "at": "b", "pose": "feliz"}]}],
                "bible": {"style": "s", "characters": []}}
        clean = studio.clean_plan(data)
        self.assertEqual([s["type"] for s in clean["segments"][0]["shots"]], ["illustration", "single"])
        self.assertEqual(clean["segments"][0]["shots"][0]["image"], "r.png")
        self.assertEqual(clean["segments"][0]["shots"][1]["pose"], "feliz")
        self.assertEqual(clean["bible"]["style"], "s")
        footage = studio.clean_plan({"segments": [{"index": 0, "expression": "feliz", "scenes": [], "beats": []}]})
        self.assertNotIn("shots", footage["segments"][0])
        with self.assertRaises(ValueError):
            studio.clean_plan({"nope": 1})

    def test_review_jobs_are_listed(self):
        jobs = [
            {"id": "1", "state": "done", "summary": {"task_id": "t1", "review_file": "/r.json"}, "label": "Revisión", "started": 1},
            {"id": "2", "state": "done", "summary": {"task_id": "t2", "result": {}}, "label": "Render", "started": 2},
            {"id": "3", "state": "running", "summary": None, "label": "x", "started": 3},
        ]
        with patch.object(studio, "list_jobs", return_value=jobs):
            self.assertEqual([r["task_id"] for r in studio.review_jobs("p")], ["t1"])


class TestScriptAndDirector(_TempDirCase):
    def test_the_intro_says_what_the_video_is_about(self):
        otter = llm.build_list_script_prompt("Los científicos más importantes", 5, "es-CO", persona="otter")
        self.assertIn("say clearly what the video is about", otter)
        self.assertIn("even though I am an otter", otter)
        self.assertIn(llm.PERSONAS["otter"], otter)
        plain = llm.build_story_script_prompt("Los científicos más importantes", 5, "es-CO")
        self.assertNotIn("otter", plain)
        for words in ("SAID ALOUD", "do not repeat the name", "Isaac Newton portrait", "real historical people"):
            self.assertIn(words, plain)
        reply = json.dumps({"title": "T", "intro": "Era 1665.", "items": [{"name": "Isaac Newton", "text": "B"}], "outro": "C"})
        with patch.object(llm, "_generate_response", return_value=reply) as ask:
            llm.generate_list_script("Científicos", 3, script_format="story", persona="otter")
        self.assertIn("even though I am an otter", ask.call_args.args[0])

    def test_section_names_are_said_aloud(self):
        script = ListVideoScript(title="T", items=[
            ListVideoItem(name="Isaac Newton", text="En 1665 la peste cerró Cambridge."),
            ListVideoItem(name="Marie Curie", text="Marie Curie llegó a París en 1891."),
        ])
        said = list_video.build_segments(script, say_names=True)
        self.assertEqual(said[0].text, "Isaac Newton. En 1665 la peste cerró Cambridge.")
        self.assertEqual(said[1].text, "Marie Curie llegó a París en 1891.")  # already says it first
        self.assertEqual(list_video.build_segments(script)[0].text, "En 1665 la peste cerró Cambridge.")
        self.assertTrue(list_video.says_name_first("¡ALBERT einstein! nació en Ulm", "Albert Einstein"))
        self.assertFalse(list_video.says_name_first("Nació en Ulm", "Albert Einstein"))

    def test_the_visual_bible(self):
        data = {"style": " warm   candle light ", "characters": [
            {"id": "Isaac Newton", "name": "Isaac Newton", "look": "long brown hair", "portrait": "Isaac Newton portrait"},
            {"id": "isaac-newton", "name": "dup", "look": "x"},
            {"id": "ghost", "name": "no look"},
            "nonsense",
        ], "sections": [
            {"index": 1, "archive": [{"query": "Great Plague of London 1665", "caption": "Londres, 1665"}, {"caption": "no query"}]},
            {"index": "2", "archive": [{"query": "x"}]},
        ]}
        bible = llm.normalize_visual_bible(data)
        self.assertEqual(bible["style"], "warm candle light")
        self.assertEqual([c["id"] for c in bible["characters"]], ["isaac-newton"])
        self.assertEqual(list(bible["sections"]), [1])
        self.assertEqual(bible["sections"][1][0]["caption"], "Londres, 1665")
        with self.assertRaises(ValueError):
            llm.normalize_visual_bible([])
        prompt = llm.build_visual_bible_prompt([{"index": 1, "text": "En 1665..."}], "es-CO", persona="otter")
        for words in ("visual bible", "Wikimedia Commons", "never invent a painting", "do not list it as a character"):
            self.assertIn(words, prompt)
        with patch.object(llm, "_generate_response", return_value=json.dumps(data)):
            self.assertEqual(llm.generate_visual_bible([{"index": 1}], "es")["characters"][0]["name"], "Isaac Newton")
        with patch.object(llm, "_generate_response", return_value="Error: quota"):
            self.assertIsNone(llm.generate_visual_bible([{"index": 1}], "es"))
        part = llm.bible_for(bible, [1])
        self.assertEqual((list(part["archive"]), part["characters"][0]["id"]), (["1"], "isaac-newton"))
        self.assertEqual(llm.bible_for(bible, [3])["archive"], {})
        self.assertEqual(llm.bible_for({}, [1]), {})

    def test_the_film_editor_reviews_the_storyboard(self):
        segments = [{"index": 0, "kind": "item", "title": "Newton", "text": "En 1665 llegó la peste", "seconds": 20, "shots": 4}]
        planned = {"segments": [{"index": 0, "opener": {"query": "Isaac Newton portrait"}, "shots": [
            {"type": "illustration", "at": "llegó la peste", "draw": "the otter walks on a road"}]}]}
        reviewed = {"segments": [{"index": 0, "shots": [
            {"type": "archive", "at": "llegó la peste", "query": "Great Plague of London 1665 engraving", "caption": "Londres",
             "draw": "a deserted plague street"}]}]}
        bible = {"style": "s", "characters": [{"id": "newton", "name": "Isaac Newton", "look": "x", "portrait": ""}], "sections": {}}
        prompts = []

        def reply(prompt, **kwargs):
            prompts.append(prompt)
            return json.dumps(planned if len(prompts) == 1 else reviewed)

        with patch.object(llm, "_generate_response", side_effect=reply):
            board = llm.generate_storyboard(segments, [], "es", bible=bible)
        self.assertEqual(len(prompts), 2)
        self.assertIn("Visual bible of this video", prompts[0])
        self.assertIn("Film editor", prompts[1])
        self.assertIn("the otter walks on a road", prompts[1])  # the editor sees the planned shots
        self.assertEqual(board[0]["shots"][0]["type"], "archive")
        self.assertEqual(board[0]["opener"]["query"], "Isaac Newton portrait")  # kept from the plan
        prompts.clear()
        with patch.object(llm, "_generate_response", side_effect=reply):
            board = llm.generate_storyboard(segments, [], "es", review=False)
        self.assertEqual((len(prompts), board[0]["shots"][0]["type"]), (1, "illustration"))
        prompts.clear()
        with patch.object(llm, "_generate_response", side_effect=reply):
            llm.generate_storyboard(segments, [], "es", reference=planned["segments"])
        self.assertEqual(len(prompts), 1)  # another language of a reviewed video is not reviewed again
        answers = iter([json.dumps(planned)] + ["Error: quota"] * 5)
        with patch.object(llm, "_generate_response", side_effect=lambda prompt, **kwargs: next(answers)):
            board = llm.generate_storyboard(segments, [], "es")
        self.assertEqual(board[0]["shots"][0]["type"], "illustration")  # a failed review keeps the plan
        self.assertEqual(llm.storyboard_shot_target(30, 5), 6)
        self.assertIn("4 to 7 seconds", llm.build_storyboard_review_prompt(segments, planned["segments"], "es"))

    def test_shots_carry_characters_motion_and_archive_details(self):
        data = {"segments": [{"index": 0, "shots": [
            {"type": "archive", "at": "a", "query": "Principia 1687", "caption": "Principia, 1687", "camera": "left",
             "draw": "an old book", "characters": ["Isaac Newton"]},
            {"type": "archive", "at": "b"},
            {"type": "illustration", "at": "c", "draw": "Newton on a hill", "characters": "newton", "motion": " the moon  moves ",
             "text": "LINCOLNSHIRE, 1666 EN LA NOCHE"},
            {"type": "animation", "at": "d", "frames": [{"draw": "one frame", "image": "/r/a.png"}], "characters": list("abcdef")},
            {"type": "clip", "at": "e", "query": "storm", "video": "/r/v.mp4"},
        ]}]}
        shots = llm.normalize_storyboard(data, 1, [])[0]["shots"]
        self.assertEqual([s["type"] for s in shots], ["archive", "illustration", "illustration", "clip"])
        archive, hill, single, clip = shots
        self.assertEqual((archive["text"], archive["camera"], archive["characters"]), ("Principia, 1687", "left", ["isaac-newton"]))
        self.assertEqual((hill["characters"], hill["motion"], hill["text"]), (["newton"], "the moon moves", "LINCOLNSHIRE, 1666 EN LA"))
        self.assertEqual((single["draw"], single["image"], len(single["characters"])), ("one frame", "/r/a.png", 4))
        self.assertEqual(clip["video"], "/r/v.mp4")
        self.assertIn("archive", llm.SHOT_TYPES)

    def test_the_gap_pass_only_adds_full_screen_pictures(self):
        gaps = [{"index": 1, "text": "en 1687 publicó los Principia", "showing": "illustration: x", "seconds": 9, "shots": 2}]
        reply = json.dumps({"shots": [
            {"index": 1, "type": "archive", "at": "publicó los Principia", "query": "Principia 1687 title page"},
            {"index": 1, "type": "stat", "at": "en 1687", "value": 1687},
            {"index": 1, "type": "compare", "items": [{"at": "en", "label": "a"}, {"at": "1687", "label": "b"}]},
        ]})
        with patch.object(llm, "_generate_response", return_value=reply):
            shots = llm.generate_storyboard_gaps(gaps, [], "es")
        self.assertEqual([s["type"] for s in shots[1]], ["archive"])
        self.assertIn("archive", llm.build_storyboard_gaps_prompt(gaps, [], "es"))


class TestNarrationCache(_TempDirCase):
    def _mp3(self, path, seconds=1.0):
        subprocess.run([utils.get_ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=300:duration={seconds}", "-c:a", "libmp3lame", path], check=True)
        return path

    def test_untimed_voices_reuse_their_take(self):
        source = self._mp3(self.path("take.mp3"))
        calls = []

        def tts(text, voice_name, voice_rate, voice_file):
            calls.append(text)
            shutil.copyfile(source, voice_file)
            return list_video.voice.populate_legacy_submaker_with_full_text(
                list_video.voice.ensure_legacy_submaker_fields(list_video.voice.SubMaker()), text, 1.0)

        with patch.object(list_video.voice, "tts", side_effect=tts):
            first = list_video.narrate("Isaac Newton.", "gemini:Schedar", 1.0, self.path("a.mp3"))
            again = list_video.narrate("Isaac Newton.", "gemini:Schedar", 1.0, self.path("b.mp3"))
            with patch.dict(list_video.config.app, {"gemini_tts_style": "otro estilo"}):
                list_video.narrate("Isaac Newton.", "gemini:Schedar", 1.0, self.path("c.mp3"))
        self.assertEqual(len(calls), 2)  # the same text and voice is recorded once; another style records again
        self.assertTrue(os.path.isfile(self.path("b.mp3")))
        self.assertEqual(editor.cue_word_times(again), [])  # the take keeps the untimed path (calm pauses)
        self.assertEqual(len(first.subs), len(again.subs))

    def test_timed_voices_are_always_synthesized(self):
        source = self._mp3(self.path("take.mp3"))
        calls = []

        def tts(text, voice_name, voice_rate, voice_file):
            calls.append(text)
            shutil.copyfile(source, voice_file)
            cue = SimpleNamespace(start=__import__("datetime").timedelta(seconds=0.1))
            return SimpleNamespace(cues=[cue, cue], subs=[], offset=[])

        with patch.object(list_video.voice, "tts", side_effect=tts):
            list_video.narrate("Hola.", "es-CO-GonzaloNeural", 1.0, self.path("a.mp3"))
            list_video.narrate("Hola.", "es-CO-GonzaloNeural", 1.0, self.path("b.mp3"))
        self.assertEqual(len(calls), 2)


class TestEditorDirection(_EditorCase):
    def test_the_bible_guides_the_storyboard_and_is_saved(self):
        ed = self._editor()
        self.assertEqual(ed.persona, "otter")
        bible = {"style": "candle light", "characters": [{"id": "newton", "name": "Isaac Newton", "look": "x", "portrait": ""}],
                 "sections": {1: [{"query": "Great Plague of London", "query_local": "", "caption": "", "about": ""}]}}
        board = llm.normalize_storyboard({"segments": [{"index": 1, "shots": [
            {"type": "illustration", "at": "palabra3", "draw": "Newton reads", "characters": ["newton"]}]}]}, 3, [])
        with patch.object(editor.llm, "generate_visual_bible", return_value=bible) as write_bible, \
                patch.object(editor.llm, "generate_storyboard", return_value=board) as storyboard:
            ed.make_plan()
        self.assertEqual(write_bible.call_args.args[2], "otter")
        self.assertIs(storyboard.call_args.kwargs["bible"], bible)
        self.assertTrue(storyboard.call_args.kwargs["review"])
        saved = json.loads(Path(self.path("edit-plan.json")).read_text("utf-8"))
        self.assertEqual(saved["bible"]["sections"]["1"][0]["query"], "Great Plague of London")
        loaded = editor.load_plan_bible(self.path("edit-plan.json"))
        self.assertEqual((loaded["style"], list(loaded["sections"])), ("candle light", [1]))
        again = self._editor(plan_file=self.path("edit-plan.json"), director_review=False)
        with patch.object(editor.llm, "generate_storyboard") as storyboard:
            again.make_plan()
        storyboard.assert_not_called()
        self.assertEqual(again.bible["characters"][0]["id"], "newton")
        self.assertEqual(editor.load_plan_bible(self.path("missing.json")), {})

    def test_defaults_are_calm(self):
        options = editor.EditOptions()
        self.assertEqual((options.shot_seconds, options.director_review, options.image_quality, options.ai_videos),
                         (5.0, True, "standard", 0))
        self.assertGreaterEqual(scenes.MIN_SHOT_SECONDS, 3.0)
        self.assertGreaterEqual(scenes.COMPOSITION_SECONDS, 5.0)

    def test_cli_passes_the_persona_and_spoken_names(self):
        with patch.object(llm, "generate_list_script", return_value=None) as write, patch("sys.stdout", io.StringIO()), \
                patch("sys.stderr", io.StringIO()):
            list_video_cli.run(["--subject", "Científicos", "--assets", "nutria", "--script-only"])
        self.assertEqual(write.call_args.kwargs["persona"], "otter")
        script = self.path("s.json")
        Path(script).write_text(json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]}), encoding="utf-8")
        with patch.object(list_video, "generate_list_video", return_value={}) as generate, patch("sys.stdout", io.StringIO()):
            list_video_cli.run(["--script", script, "--no-say-names"])
            self.assertFalse(generate.call_args.kwargs["say_names"])
            list_video_cli.run(["--script", script])
            self.assertTrue(generate.call_args.kwargs["say_names"])


class TestStudioReviewPage(_TempDirCase):
    PAGE = """
import json
import sys
from unittest.mock import patch

sys.path.insert(0, {root!r})
from studio import Studio as app

review = json.loads(open({review!r}, encoding="utf-8").read())
with patch.object(app.studio, "review_jobs", return_value=[{{"task_id": "t1", "label": "Revision", "started": 1}}]), \\
        patch.object(app.studio, "load_review", return_value=review), \\
        patch.object(app.studio, "load_settings", return_value=app.studio.RenderSettings()):
    app.st.session_state["project"] = "demo"
    app.page_review()
"""

    def test_the_review_page_shows_every_shot(self):
        from streamlit.testing.v1 import AppTest

        picture = _picture(self.path("a.png"))
        review = {"task_id": "t1", "plan_file": self.path("edit-plan.json"), "report": ["Shots: archive 1"], "warnings": [],
                  "shots": [
                      {"id": "1-opener", "segment": 1, "shot": -1, "chapter": "1. Isaac Newton", "type": "opener",
                       "planned": "opener", "start": 0, "end": 3.6, "said": "Isaac Newton.", "describe": "", "query": "Newton",
                       "caption": "", "pictures": [{"slot": "opener", "file": picture, "url": "u", "kind": "picture"}]},
                      {"id": "1-00", "segment": 1, "shot": 0, "chapter": "1. Isaac Newton", "type": "illustration",
                       "planned": "archive", "start": 3.6, "end": 9, "said": "En 1665 llegó la peste.", "describe": "a street",
                       "query": "plague", "caption": "", "pictures": [{"slot": "center", "file": picture, "url": "",
                                                                         "kind": "picture"}]},
                  ]}
        Path(self.path("review.json")).write_text(json.dumps(review), encoding="utf-8")
        script = self.PAGE.format(root=str(Path(__file__).parent.parent.parent), review=self.path("review.json"))
        page = AppTest.from_string(script, default_timeout=60).run()
        self.assertFalse(page.exception, [e.value for e in page.exception])
        markdown = " ".join(m.value for m in page.markdown)
        self.assertIn("1. Isaac Newton", markdown)
        self.assertIn("en lugar de foto o pintura real", markdown)
        self.assertEqual(len(page.radio), 2)
        self.assertEqual(page.radio[0].options, ["✅ Mantener", "🔁 Otra versión", "📜 Foto real"])  # a card is never removed
        self.assertEqual(len(page.radio[1].options), 4)


if __name__ == "__main__":
    unittest.main()
