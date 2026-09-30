import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import research as research_cli
from app.config import config as app_config
from app.services import channel_research as research

NOW = datetime(2026, 9, 30, tzinfo=timezone.utc)


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _video(video_id, title, views, duration="PT12M", published="2026-06-30T12:00:00Z", **extra):
    return {
        "id": video_id,
        "snippet": {"title": title, "publishedAt": published, "tags": ["a"], **extra},
        "statistics": {"viewCount": str(views), "likeCount": str(views // 50), "commentCount": str(views // 500)},
        "contentDetails": {"duration": duration},
    }


class FakeYouTube:
    """Answers the four endpoints the research module uses."""

    def __init__(self):
        self.calls = []
        self.videos = {
            "v1": _video("v1", "Cada veneno explicado", 1_000_000),
            "v2": _video("v2", "El sistema solar", 100_000),
            "v3": _video("v3", "¿Qué pasa si no duermes 10 días?", 90_000, duration="PT25M"),
            "v4": _video("v4", "Dato rápido", 50_000, duration="PT45S"),
            "v5": _video("v5", "En vivo", 10, liveBroadcastContent="live"),
        }

    def __call__(self, url, params, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        self.calls.append((endpoint, dict(params)))
        if endpoint == "channels":
            if params.get("forHandle") == "@missing":
                return _Response({"items": []})
            return _Response({"items": [{
                "id": "UCaaaaaaaaaaaaaaaaaaaaaa",
                "snippet": {"title": "Canal A"},
                "statistics": {"subscriberCount": "200000", "videoCount": "5"},
                "contentDetails": {"relatedPlaylists": {"uploads": "UUaaaa"}},
            }]})
        if endpoint == "playlistItems":
            ids = list(self.videos)
            page = ids[:3] if "pageToken" not in params else ids[3:]
            payload = {"items": [{"contentDetails": {"videoId": v}} for v in page]}
            if "pageToken" not in params:
                payload["nextPageToken"] = "next"
            return _Response(payload)
        if endpoint == "videos":
            return _Response({"items": [self.videos[v] for v in params["id"].split(",")]})
        if endpoint == "search":
            return _Response({"items": [{"id": {"channelId": "UCaaaaaaaaaaaaaaaaaaaaaa"}}, {"id": {}}]})
        return _Response({"error": {"message": "unknown"}}, 404)


class _ResearchCase(unittest.TestCase):
    def setUp(self):
        key_patch = patch.dict(app_config.app, {"youtube_api_key": "test-key"})
        key_patch.start()
        self.addCleanup(key_patch.stop)
        self.fake = FakeYouTube()
        http_patch = patch.object(research.requests, "get", side_effect=self.fake)
        http_patch.start()
        self.addCleanup(http_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)


class TestHelpers(unittest.TestCase):
    def test_channel_references(self):
        self.assertEqual(research.parse_channel_reference("@Capsula"), {"forHandle": "@Capsula"})
        self.assertEqual(research.parse_channel_reference("Capsula"), {"forHandle": "@Capsula"})
        self.assertEqual(
            research.parse_channel_reference("https://www.youtube.com/@C%C3%A1psula_Mental-r9u/videos"),
            {"forHandle": "@Cápsula_Mental-r9u"},
        )
        channel_id = "UC" + "x" * 22
        self.assertEqual(research.parse_channel_reference(f"https://youtube.com/channel/{channel_id}"), {"id": channel_id})
        self.assertEqual(research.parse_channel_reference(channel_id), {"id": channel_id})
        with self.assertRaises(research.ResearchError):
            research.parse_channel_reference(" ")

    def test_durations_formats_and_buckets(self):
        self.assertEqual(research.parse_duration("PT1H2M3S"), 3723)
        self.assertEqual(research.parse_duration("PT45S"), 45)
        self.assertEqual(research.parse_duration("P1DT1S"), 86401)
        self.assertEqual(research.parse_duration("P0D"), 0)
        self.assertEqual(research.parse_duration("garbage"), 0)
        self.assertEqual([research.video_format(s) for s in (30, 120, 900)], ["short", "short?", "long"])
        self.assertEqual(research.duration_bucket(900), "10-20 min")
        self.assertEqual(research.duration_bucket(10_000), "40+ min")

    def test_missing_key_explains_how_to_get_one(self):
        with patch.dict(app_config.app, {"youtube_api_key": ""}):
            with self.assertRaisesRegex(research.ResearchError, "console.cloud.google.com"):
                research.api_key()


class TestCollection(_ResearchCase):
    def test_research_collects_pages_and_computes_outliers(self):
        rows, warnings = research.research(
            ["@CanalA", "@missing"], searches=["explicado"], max_videos=10, now=NOW
        )
        self.assertEqual(warnings, ["channel not found: @missing"])
        # The live stream is skipped and the channel found by search is not read twice.
        self.assertEqual(sorted(r.video_id for r in rows), ["v1", "v2", "v3", "v4"])
        by_id = {r.video_id: r for r in rows}
        # Long videos: medians of 1,000,000 / 100,000 / 90,000 -> 100,000.
        self.assertEqual(by_id["v1"].outlier_score, 10.0)
        self.assertEqual(by_id["v4"].format, "short")
        self.assertEqual(by_id["v4"].outlier_score, 1.0)
        self.assertEqual(by_id["v1"].age_days, 91.5)
        self.assertAlmostEqual(by_id["v1"].views_per_day, round(1_000_000 / 91.5, 1))
        self.assertEqual(by_id["v1"].views_per_subscriber, 5.0)
        self.assertTrue(by_id["v3"].title_has_number and by_id["v3"].title_has_question)
        endpoints = [call[0] for call in self.fake.calls]
        self.assertEqual(endpoints.count("search"), 1)
        self.assertEqual(endpoints.count("playlistItems"), 2)
        self.assertTrue(all(call[1]["key"] == "test-key" for call in self.fake.calls))

    def test_max_videos_limits_pages(self):
        channel = research.get_channel("@CanalA")
        self.assertEqual(research.list_video_ids(channel, 2), ["v1", "v2"])

    def test_api_errors_are_reported(self):
        with patch.object(research.requests, "get", return_value=_Response({"error": {"message": "quota exceeded"}}, 403)):
            with self.assertRaisesRegex(research.ResearchError, "quota exceeded"):
                research.get_channel("@CanalA")

    def test_csv_round_trip_and_summary(self):
        rows, _ = research.research(["@CanalA"], max_videos=10, now=NOW)
        path = research.write_csv(rows, os.path.join(self.temp_dir, "r.csv"))
        with open(path, encoding="utf-8-sig") as fp:
            records = list(csv.DictReader(fp))
        self.assertEqual(records[0]["title"], "Cada veneno explicado")
        self.assertEqual(records[0]["url"], "https://www.youtube.com/watch?v=v1")
        reloaded = research.load_csv(path)
        self.assertEqual({r.video_id: r.outlier_score for r in reloaded}, {r.video_id: r.outlier_score for r in rows})

        summary = research.summarize(rows)
        self.assertEqual(summary["by_format"]["long"]["count"], 3)
        self.assertIn("10-20 min", summary["long_by_duration"])
        self.assertEqual(summary["long_title_features"]["with_question"]["count"], 1)

    def test_analysis_prompt_and_llm_errors(self):
        rows, _ = research.research(["@CanalA"], max_videos=10, now=NOW)
        prompt = research.build_analysis_prompt(rows, research.summarize(rows), "Cabeceando", "Spanish")
        self.assertIn("Cabeceando", prompt)
        self.assertIn("Cada veneno explicado | Canal A | 1000000 | 10.0", prompt)
        self.assertIn("answer in Spanish", prompt)
        with patch.object(research.llm, "_generate_response", return_value="## Ideas"):
            self.assertEqual(research.analyze(rows, "brief"), "## Ideas")
        with patch.object(research.llm, "_generate_response", return_value="Error: quota"):
            with self.assertRaises(research.ResearchError):
                research.analyze(rows, "brief")
        with self.assertRaises(research.ResearchError):
            research.analyze([], "brief")


class TestResearchCli(_ResearchCase):
    def _run(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = research_cli.run(argv)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_collect_analyze_and_reanalyze(self):
        channels = os.path.join(self.temp_dir, "channels.txt")
        Path(channels).write_text("# my list\n@CanalA\n\n", encoding="utf-8")
        output = os.path.join(self.temp_dir, "out", "research.csv")
        with patch.object(research.llm, "_generate_response", return_value="## Patrones"):
            code, stdout, _ = self._run(["--channels-file", channels, "--output", output, "--analyze"])
        self.assertEqual(code, 0)
        summary = json.loads(stdout)
        self.assertEqual(summary["videos"], 4)
        self.assertEqual(Path(summary["analysis"]).read_text(encoding="utf-8"), "## Patrones\n")

        with patch.object(research.llm, "_generate_response", return_value="## Otra vez") as generate:
            code, stdout, _ = self._run(["--from-csv", output, "--analyze", "--language", "Portuguese"])
        self.assertEqual(code, 0)
        self.assertIn("answer in Portuguese", generate.call_args.args[0])

    def test_input_errors(self):
        self.assertEqual(self._run([])[0], 2)
        self.assertEqual(self._run(["--channel", "@a", "--max-videos", "0"])[0], 2)
        with patch.dict(app_config.app, {"youtube_api_key": ""}):
            code, _, _ = self._run(["--channel", "@CanalA"])
        self.assertEqual(code, 1)
        with patch.object(research.requests, "get", return_value=_Response({"items": []})):
            self.assertEqual(self._run(["--channel", "@missing"])[0], 1)


if __name__ == "__main__":
    unittest.main()
