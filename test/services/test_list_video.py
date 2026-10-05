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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import list_video as list_video_cli
from app.config import config as app_config
from app.models.schema import ListVideoItem, ListVideoScript, VideoParams
from app.services import list_video, llm, subtitle
from app.utils import utils

RESOURCES = Path(__file__).parent.parent / "resources"


def _script(**overrides):
    data = {
        "title": "Every hormone explained",
        "intro": "Your body runs on chemical messengers.",
        "intro_image_term": "",
        "items": [
            ListVideoItem(name="Adrenaline", text="Adrenaline speeds up your heart.", image_term="racing heart"),
            ListVideoItem(name="Melatonin", text="Melatonin tells your body it is night.", image_term=""),
            ListVideoItem(name="Insulin", text="Insulin lets sugar into your cells.", image_term="sugar cubes"),
        ],
        "outro": "Which one surprised you?",
    }
    data.update(overrides)
    return ListVideoScript(**data)


class TestBuildSegments(unittest.TestCase):
    def test_numbers_items_and_reuses_first_and_last_visual_for_intro_and_outro(self):
        segments = list_video.build_segments(_script())

        self.assertEqual(
            [s.chapter for s in segments],
            ["Intro", "1. Adrenaline", "2. Melatonin", "3. Insulin", "Outro"],
        )
        self.assertEqual(segments[0].label, "")
        self.assertEqual(segments[0].image_term, "racing heart")
        # An item without an image term falls back to its name.
        self.assertEqual(segments[2].image_term, "Melatonin")
        self.assertEqual(segments[-1].image_term, "sugar cubes")

    def test_own_intro_visual_and_unnumbered_items(self):
        script = _script(intro_image_term="messenger molecules", outro="")
        segments = list_video.build_segments(script, number_items=False)

        self.assertEqual(segments[0].image_term, "messenger molecules")
        self.assertEqual(segments[1].label, "Adrenaline")
        self.assertEqual(segments[-1].kind, "item")

    def test_intro_file_is_not_replaced_by_the_item_term(self):
        script = _script(intro_image_file="/tmp/intro.png")
        intro = list_video.build_segments(script)[0]

        self.assertEqual((intro.image_term, intro.image_file), ("", "/tmp/intro.png"))


class TestValidateVisualSources(unittest.TestCase):
    def test_rejects_unsupported_source(self):
        segments = list_video.build_segments(_script())
        with self.assertRaisesRegex(list_video.ListVideoError, "support video_source"):
            list_video.validate_visual_sources(segments, "wavespeed")

    def test_local_source_needs_a_file_for_every_segment(self):
        segments = list_video.build_segments(_script())
        with self.assertRaisesRegex(list_video.ListVideoError, "image_file for every item"):
            list_video.validate_visual_sources(segments, "local")

    def test_reports_missing_image_files(self):
        items = [ListVideoItem(name="A", text="a", image_file="/nonexistent/a.png")]
        segments = list_video.build_segments(_script(items=items))
        with self.assertRaisesRegex(list_video.ListVideoError, "not found"):
            list_video.validate_visual_sources(segments, "pexels")

    def test_openai_image_requires_configuration(self):
        segments = list_video.build_segments(_script())
        with patch.object(list_video.material, "is_openai_image_enabled", return_value=False):
            with self.assertRaisesRegex(list_video.ListVideoError, "openai_image_base_url"):
                list_video.validate_visual_sources(segments, "openai_image")

    def test_accepts_stock_source(self):
        list_video.validate_visual_sources(list_video.build_segments(_script()), "pexels")


class TestTimelineHelpers(unittest.TestCase):
    def test_chapter_format_uses_hours_only_for_long_videos(self):
        chapters = [(0.0, "Intro"), (65.9, "1. A"), (3725.2, "2. B")]
        self.assertEqual(
            list_video.format_chapters(chapters[:2], 120),
            "0:00 Intro\n1:05 1. A",
        )
        self.assertEqual(
            list_video.format_chapters(chapters, 3800),
            "0:00:00 Intro\n0:01:05 1. A\n1:02:05 2. B",
        )

    def test_shifted_subtitles_round_trip_through_srt(self):
        entries = [
            (1, "00:00:00,000 --> 00:00:01,500", "hello"),
            (2, "00:00:01,500 --> 00:00:03,250", "world"),
            (3, "00:00:03,250 --> 00:00:03,250", "empty span"),
        ]
        shifted = list_video.shift_subtitle_entries(entries, 61.0)
        self.assertEqual(shifted, [(61.0, 62.5, "hello"), (62.5, 64.25, "world")])

        with tempfile.TemporaryDirectory() as temp_dir:
            srt = os.path.join(temp_dir, "out.srt")
            self.assertTrue(list_video.write_srt(shifted, srt))
            self.assertEqual(
                subtitle.file_to_subtitles(srt),
                [
                    (1, "00:01:01,000 --> 00:01:02,500", "hello"),
                    (2, "00:01:02,500 --> 00:01:04,250", "world"),
                ],
            )
            self.assertFalse(list_video.write_srt([], srt + ".empty"))

    def test_pcm_is_padded_or_trimmed_to_whole_frames(self):
        frame_bytes = list_video.SAMPLES_PER_FRAME * 2
        padded, trimmed = list_video.pad_pcm_to_frames(b"\x01\x00" * 10, 3)
        self.assertEqual((len(padded), trimmed), (3 * frame_bytes, 0.0))
        self.assertTrue(padded.endswith(b"\x00\x00"))

        cut, trimmed = list_video.pad_pcm_to_frames(b"\x01\x00" * (frame_bytes), 1)
        self.assertEqual(len(cut), frame_bytes)
        self.assertAlmostEqual(trimmed, list_video.SAMPLES_PER_FRAME / list_video.PCM_SAMPLE_RATE)

    def test_montage_loops_short_clips_and_caps_each_piece(self):
        pieces = list_video.plan_montage([("a", 3.0), ("b", 10.0), ("broken", 0.0)], 12.0, 5)
        self.assertEqual(pieces, [("a", 3.0), ("b", 5.0), ("a", 3.0), ("b", 1.0)])
        self.assertEqual(list_video.plan_montage([("broken", 0.0)], 5.0, 5), [])


class TestRenderSegment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.params = VideoParams(
            video_subject="test", video_aspect="16:9", font_name="BeVietnamPro-Bold.ttf"
        )
        self.font = list_video._resolve_font_path(self.params)

    def test_image_segment_has_the_exact_frame_count(self):
        output = os.path.join(self.temp_dir, "segment.mp4")
        list_video.render_segment_video(
            list_video._Visual("image", [str(RESOURCES / "1.png")]),
            17,
            self.params,
            output,
            title="1. Adrenaline",
            font_path=self.font,
            zoom=0.05,
        )

        self.assertEqual(list_video.count_video_frames(output), 17)
        # Temporary still and banner pictures are removed.
        self.assertEqual(os.listdir(self.temp_dir), ["segment.mp4"])

    def test_fallback_background_without_title(self):
        output = os.path.join(self.temp_dir, "plain.mp4")
        list_video.render_segment_video(list_video._Visual("none"), 9, self.params, output)
        self.assertEqual(list_video.count_video_frames(output), 9)

    def test_title_banner_shrinks_to_fit(self):
        banner = list_video.render_title_image("A" * 200, 1080, 1920, self.font)
        self.assertLessEqual(banner.size[0], int(1080 * 0.9))
        self.assertEqual(banner.mode, "RGBA")

    def _test_clip(self, name, size, seconds):
        path = os.path.join(self.temp_dir, name)
        subprocess.run(
            [utils.get_ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i",
             f"testsrc=size={size}:rate=25", "-t", str(seconds), path],
            check=True,
        )
        return path

    def test_short_clips_are_looped_and_joined_without_reencoding(self):
        clips = [self._test_clip("a.mp4", "320x240", 0.5), self._test_clip("b.mp4", "240x320", 0.4)]
        params = VideoParams(video_subject="t", video_aspect="1:1", video_clip_duration=1)
        first = list_video.render_segment_video(
            list_video._Visual("videos", clips), 45, params, os.path.join(self.temp_dir, "s1.mp4")
        )
        contain = VideoParams(video_subject="t", video_aspect="1:1", video_fit_mode="contain")
        second = list_video.render_segment_video(
            list_video._Visual("videos", clips[:1]), 20, contain, os.path.join(self.temp_dir, "s2.mp4")
        )
        self.assertEqual(list_video.count_video_frames(first), 45)

        joined = list_video.concat_segments([first, second], os.path.join(self.temp_dir, "all.mp4"))
        self.assertEqual(list_video.count_video_frames(joined), 65)
        self.assertFalse(os.path.exists(joined + ".txt"))

    def test_unreadable_clips_fall_back_to_a_plain_background(self):
        broken = os.path.join(self.temp_dir, "broken.mp4")
        Path(broken).write_bytes(b"not a video")
        output = os.path.join(self.temp_dir, "fallback.mp4")
        list_video.render_segment_video(list_video._Visual("videos", [broken]), 6, self.params, output)
        self.assertEqual(list_video.count_video_frames(output), 6)

    def test_render_and_join_failures_raise(self):
        with patch.object(list_video.utils, "get_ffmpeg_binary", return_value="/nonexistent/ffmpeg"):
            with self.assertRaises(OSError):
                list_video.render_segment_video(
                    list_video._Visual("none"), 3, self.params, os.path.join(self.temp_dir, "x.mp4")
                )
        with self.assertRaisesRegex(list_video.ListVideoError, "failed to join"):
            list_video.concat_segments(
                [os.path.join(self.temp_dir, "missing.mp4")], os.path.join(self.temp_dir, "y.mp4")
            )

    def test_prepared_still_flattens_transparency_on_white(self):
        from PIL import Image

        source = os.path.join(self.temp_dir, "transparent.png")
        Image.new("RGBA", (40, 20), (255, 0, 0, 0)).save(source)
        output = list_video.prepare_still_image(
            source, os.path.join(self.temp_dir, "still.png"), 64, 64, "contain"
        )
        with Image.open(output) as still:
            self.assertEqual(still.size, (64, 64))
            self.assertEqual(still.getpixel((32, 32)), (255, 255, 255))
            self.assertEqual(still.getpixel((0, 0)), (0, 0, 0))


class TestGenerateListVideo(unittest.TestCase):
    def setUp(self):
        self.task_id = str(uuid4())
        self.addCleanup(
            shutil.rmtree, os.path.join(utils.storage_dir(), "tasks", self.task_id), True
        )
        provider_patch = patch.dict(app_config.app, {"subtitle_provider": "edge"})
        provider_patch.start()
        self.addCleanup(provider_patch.stop)

    def _items(self):
        return [
            ListVideoItem(name=f"Item {n}", text=f"Sentence number {n}.", image_file=str(RESOURCES / f"{n}.png"))
            for n in (1, 2)
        ]

    def test_segments_share_one_timeline_with_chapters_and_subtitles(self):
        script = _script(items=self._items())
        params = VideoParams(
            video_subject="test",
            video_aspect="16:9",
            video_source="local",
            voice_name="no-voice",
            font_name="BeVietnamPro-Bold.ttf",
        )
        rendered = []

        def fake_render(visual, frames, params, output_file, title="", font_path="", zoom=0, **kwargs):
            rendered.append((visual.kind, frames, title))
            Path(output_file).write_bytes(b"segment")
            return output_file

        def fake_concat(segment_files, output_file):
            Path(output_file).write_bytes(b"combined")
            return output_file

        with patch.object(list_video, "render_segment_video", side_effect=fake_render), patch.object(
            list_video, "count_video_frames", side_effect=lambda path: rendered[-1][1] + 1
        ), patch.object(list_video, "concat_segments", side_effect=fake_concat), patch.object(
            list_video.video, "generate_video", return_value=True
        ) as generate_video:
            result = list_video.generate_list_video(self.task_id, script, params, gap_seconds=0.2)

        self.assertEqual([r[2] for r in rendered], ["", "1. Item 1", "2. Item 2", ""])
        self.assertTrue(all(kind == "image" for kind, _, _ in rendered))
        # Narration follows the frames actually written, one extra per segment.
        actual_frames = [frames + 1 for _, frames, _ in rendered]
        with wave.open(result["audio_file"], "rb") as narration:
            self.assertEqual(
                narration.getnframes(), sum(actual_frames) * list_video.voice_polish.HQ_RATE // list_video.FPS
            )
            self.assertEqual(narration.getframerate(), 48000)  # the voice keeps its highs
        self.assertAlmostEqual(result["audio_duration"], sum(actual_frames) / list_video.FPS, places=3)

        starts = [0.0]
        for frames in actual_frames[:-1]:
            starts.append(starts[-1] + frames / list_video.FPS)
        saved = json.loads(Path(utils.task_dir(self.task_id), "script.json").read_text("utf-8"))
        self.assertEqual([c["start"] for c in saved["chapters"]], [round(s, 3) for s in starts])
        self.assertEqual(saved["list_script"]["title"], "Every hormone explained")
        self.assertIn("0:00 Intro", Path(result["chapters_file"]).read_text("utf-8"))

        entries = subtitle.file_to_subtitles(result["subtitle_path"])
        self.assertEqual(entries[1][2], "Sentence number 1")
        self.assertTrue(entries[1][1].startswith(list_video._format_srt_time(starts[1])))

        # Burned-in subtitles go through generate_video with loudness-normalized audio.
        kwargs = generate_video.call_args.kwargs
        self.assertTrue(kwargs["audio_path"].endswith("narration-master.wav"))
        self.assertTrue(os.path.isfile(kwargs["audio_path"]))
        self.assertEqual(result["warnings"], [])
        self.assertFalse(os.path.exists(os.path.join(utils.task_dir(self.task_id), "combined-1.mp4")))

    def test_failed_tts_stops_the_task(self):
        script = _script(items=self._items())
        params = VideoParams(video_subject="t", video_source="local", voice_name="no-voice")
        with patch.object(list_video.voice, "tts", return_value=None):
            with self.assertRaisesRegex(list_video.ListVideoError, "failed to synthesize"):
                list_video.generate_list_video(self.task_id, script, params)

    def test_missing_stock_visual_becomes_a_warning(self):
        warnings, sources = [], []
        segment = list_video.build_segments(_script())[1]
        params = VideoParams(video_subject="t", video_source="pexels")
        with patch.object(list_video.material, "download_videos", return_value=[]):
            visual = list_video._prepare_visual(self.task_id, segment, params, 5.0, sources, warnings)
        self.assertEqual(visual.kind, "none")
        self.assertIn("racing heart", warnings[0])

    def test_stock_sources_are_collected_per_segment(self):
        from app.services import task_artifacts

        task_artifacts.write_script_data(self.task_id, {"material_sources": []})
        segments = list_video.build_segments(_script())
        params = VideoParams(video_subject="t", video_source="pixabay")
        downloads = iter([("clip-a.mp4", {"id": "a"}), ("clip-b.mp4", None)])

        def fake_download(**kwargs):
            path, source = next(downloads)
            self.assertEqual(len(kwargs["search_terms"]), 1)
            self.assertEqual(kwargs["audio_duration"], 5.0 * list_video.STOCK_FOOTAGE_FACTOR)
            if source:
                task_artifacts.patch_script_data(self.task_id, material_sources=[source])
            return [path]

        sources, warnings = [], []
        with patch.object(list_video.material, "download_videos", side_effect=fake_download):
            first = list_video._prepare_visual(self.task_id, segments[1], params, 5.0, sources, warnings)
            # A download that records nothing must not repeat the previous record.
            list_video._prepare_visual(self.task_id, segments[2], params, 5.0, sources, warnings)
        self.assertEqual((first.kind, first.paths), ("videos", ["clip-a.mp4"]))
        self.assertEqual(sources, [{"id": "a"}])
        self.assertEqual(warnings, [])

    def test_generated_image_and_local_files(self):
        segment = list_video.build_segments(_script())[1]
        params = VideoParams(video_subject="t", video_source="openai_image", video_aspect="16:9")
        generated = [SimpleNamespace(url="/tmp/generated.png")]
        with patch.object(list_video.material, "generate_images_openai", return_value=generated) as gen:
            visual = list_video._prepare_visual(self.task_id, segment, params, 4.2, [], [])
        self.assertEqual((visual.kind, visual.paths), ("image", ["/tmp/generated.png"]))
        self.assertEqual(gen.call_args.kwargs["search_term"], "racing heart")
        self.assertEqual(gen.call_args.kwargs["minimum_duration"], 5)

        with patch.object(list_video.material, "generate_images_openai", side_effect=RuntimeError("down")):
            warnings = []
            visual = list_video._prepare_visual(self.task_id, segment, params, 4.2, [], warnings)
        self.assertEqual((visual.kind, len(warnings)), ("none", 1))

        for name, kind in (("a.WEBP", "image"), ("b.mov", "videos")):
            segment.image_file = f"/pics/{name}"
            self.assertEqual(list_video._prepare_visual(self.task_id, segment, params, 1, [], []).kind, kind)


class TestGenerateListScript(unittest.TestCase):
    def test_prompt_contains_count_language_and_subject(self):
        prompt = llm.build_list_script_prompt("Cada hormona", 12, "es-CO", 90)
        self.assertIn("exactly 12 items", prompt)
        self.assertIn("in es-CO", prompt)
        self.assertIn("about 90 words", prompt)
        self.assertTrue(prompt.endswith("Cada hormona"))

    def test_parses_fenced_json_and_drops_unknown_keys(self):
        payload = {
            "title": "Every hormone",
            "intro": "Hook.",
            "items": [{"name": "Insulin", "text": "Text.", "image_term": "sugar", "extra": 1}],
            "outro": "Bye?",
            "notes": "ignored",
        }
        response = "```json\n" + json.dumps(payload) + "\n```"
        with patch.object(llm, "_generate_response", return_value=response):
            script = llm.generate_list_script("Every hormone", 3)
        self.assertEqual(script.items[0].name, "Insulin")
        self.assertEqual(script.items[0].image_file, "")

    def test_retries_invalid_json_and_stops_on_provider_error(self):
        valid = json.dumps({"title": "T", "items": [{"name": "A", "text": "B"}]})
        with patch.object(llm, "_generate_response", side_effect=["not json", "Here: " + valid + " done"]):
            self.assertEqual(llm.generate_list_script("T", 3).title, "T")
        with patch.object(llm, "_generate_response", return_value="Error: quota"):
            self.assertIsNone(llm.generate_list_script("T", 3))

    def test_gives_up_after_retries(self):
        with patch.object(llm, "_generate_response", return_value="{}"):
            self.assertIsNone(llm.generate_list_script("T", 3))


class TestListVideoCli(unittest.TestCase):
    def setUp(self):
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def _run(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = list_video_cli.run(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def _write_script(self):
        os.makedirs(os.path.join(self.temp_dir, "pics"))
        shutil.copy(RESOURCES / "1.png", os.path.join(self.temp_dir, "pics", "1.png"))
        path = os.path.join(self.temp_dir, "script.json")
        data = {
            "title": "T",
            "intro_image_file": "pics/1.png",
            "items": [{"name": "A", "text": "B", "image_file": "pics/1.png"}],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_image_paths_are_relative_to_the_script(self):
        script = list_video_cli.load_script_file(self._write_script())
        expected = os.path.join(self.temp_dir, "pics", "1.png")
        self.assertEqual(script.items[0].image_file, expected)
        self.assertEqual(script.intro_image_file, expected)

    def test_rejects_options_of_the_single_script_pipeline(self):
        code, _, stderr = self._run(["--subject", "x", "--stop-at", "video"])
        self.assertEqual(code, 2)
        self.assertIn("--stop-at", stderr)

    def test_script_only_needs_a_subject(self):
        code, _, stderr = self._run(["--script", "x.json", "--script-only"])
        self.assertEqual(code, 2)
        self.assertIn("--script-only requires --subject", stderr)

    def test_invalid_script_file_exits_with_2(self):
        path = os.path.join(self.temp_dir, "bad.json")
        Path(path).write_text('{"title": "T", "items": []}', encoding="utf-8")
        self.assertEqual(self._run(["--script", path])[0], 2)

    def test_script_only_saves_generated_script(self):
        output = os.path.join(self.temp_dir, "generated.json")
        with patch.object(llm, "generate_list_script", return_value=_script()) as generate:
            code, stdout, _ = self._run(
                ["--subject", "Hormones", "--items", "5", "--video-language", "es-CO",
                 "--script-only", "--output", output]
            )
        self.assertEqual(code, 0)
        self.assertEqual(generate.call_args.kwargs["language"], "es-CO")
        self.assertEqual(generate.call_args.kwargs["item_count"], 5)
        self.assertEqual(json.loads(stdout)["script_file"], output)
        self.assertEqual(list_video_cli.load_script_file(output).title, "Every hormone explained")

    def test_renders_script_with_forwarded_options_and_landscape_default(self):
        path = self._write_script()
        result = {"videos": ["final.mp4"]}
        with patch.object(list_video, "generate_list_video", return_value=result) as generate:
            code, stdout, _ = self._run(
                ["--script", path, "--video-source", "local", "--voice-name", "no-voice",
                 "--zoom", "0", "--no-numbers"]
            )
        self.assertEqual(code, 0)
        args, kwargs = generate.call_args
        params = args[2]
        self.assertEqual(params.video_aspect, "16:9")
        self.assertEqual(params.voice_name, "no-voice")
        self.assertEqual(kwargs["zoom"], 0)
        self.assertFalse(kwargs["number_items"])
        self.assertEqual(json.loads(stdout)["result"], result)

    def test_render_failure_exits_with_1(self):
        path = self._write_script()
        with patch.object(
            list_video, "generate_list_video", side_effect=list_video.ListVideoError("boom")
        ):
            code, _, _ = self._run(["--script", path, "--video-aspect", "9:16"])
        self.assertEqual(code, 1)

    def test_llm_failure_exits_with_1(self):
        with patch.object(llm, "generate_list_script", return_value=None):
            self.assertEqual(self._run(["--subject", "x"])[0], 1)


if __name__ == "__main__":
    unittest.main()
