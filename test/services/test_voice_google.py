import base64
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydub import AudioSegment

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import voice_lab
from app.config import config
from app.services import voice
from app.services import voice_google as vg


def _wav(seconds=1.0):
    buffer = io.BytesIO()
    AudioSegment.silent(duration=int(seconds * 1000), frame_rate=24000).export(buffer, format="wav")
    return base64.b64encode(buffer.getvalue()).decode()


class TestDelivery(unittest.TestCase):
    def test_presets_and_pace(self):
        self.assertIn("divulgador científico joven", vg.resolve_style("Divulgador"))
        self.assertEqual(vg.resolve_style("Con calma"), "Con calma")
        self.assertEqual(vg.gemini_delivery("Con calma:", 1.0), "Con calma")
        self.assertEqual(vg.gemini_delivery("Con calma", 1.2), "Con calma, at a fast, energetic pace")
        self.assertEqual(vg.gemini_delivery("", 0.9), "Read this at a slightly relaxed pace")
        self.assertEqual(vg.gemini_delivery("", "x"), "")
        self.assertEqual(vg.gemini_models("gemini-2.5-pro-preview-tts"), ["gemini-2.5-pro-preview-tts", "gemini-2.5-flash-preview-tts"])
        self.assertEqual(vg.gemini_models(""), list(vg.GEMINI_TTS_MODELS))


class TestGeminiAudio(unittest.TestCase):
    TEXT = " ".join(["palabra"] * 50)  # about ten seconds of speech

    def test_retries_then_succeeds(self):
        good = b"\0\0" * vg.SAMPLE_RATE * 9
        request = MagicMock(side_effect=[RuntimeError("503 UNAVAILABLE"), None, good])
        sleep = MagicMock()
        self.assertEqual(vg.gemini_audio(request, self.TEXT, "c", "m", sleep=sleep), good)
        self.assertEqual(request.call_count, 3)
        sleep.assert_called_once_with(2.0)

    def test_missing_model_falls_back_and_short_audio_is_asked_again(self):
        short = b"\0\0" * vg.SAMPLE_RATE  # one second for ten seconds of text
        good = b"\0\0" * vg.SAMPLE_RATE * 8
        calls = []

        def request(model, contents):
            calls.append(model)
            if model == "missing-tts":
                raise RuntimeError("404 NOT_FOUND: model not found")
            return short if len(calls) == 2 else good

        self.assertEqual(vg.gemini_audio(request, self.TEXT, "c", "missing-tts", sleep=lambda s: None), good)
        self.assertEqual(calls, ["missing-tts", vg.GEMINI_TTS_MODELS[0], vg.GEMINI_TTS_MODELS[0]])
        # Always cut short: the longest answer is kept.
        always_short = MagicMock(return_value=short)
        self.assertEqual(vg.gemini_audio(always_short, self.TEXT, "c", "", sleep=lambda s: None), short)
        self.assertIsNone(vg.gemini_audio(MagicMock(side_effect=RuntimeError("boom")), "x", "c", "", sleep=lambda s: None))


class TestCloudTts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def test_names_chunks_and_body(self):
        self.assertTrue(vg.is_gcloud_voice("gcloud:es-US-Chirp3-HD-Charon"))
        self.assertEqual(vg.gcloud_voice_name("gcloud:es-US-Chirp3-HD-Charon-Male"), "es-US-Chirp3-HD-Charon")
        self.assertIn("gcloud:es-US-Chirp3-HD-Puck", vg.get_gcloud_voices())
        text = ("Una frase corta. " * 400).strip()
        chunks = vg.split_text(text, 500)
        self.assertTrue(all(len(c.encode("utf-8")) <= 500 for c in chunks))
        self.assertEqual(" ".join(chunks), text)
        self.assertEqual(vg.split_text("x" * 20 + " y", 10), ["x" * 20, "y"])
        body = vg.gcloud_request_body("hola", "es-US-Chirp3-HD-Charon", 1.1, 0.5, 3)
        self.assertEqual(body["voice"], {"languageCode": "es-US", "name": "es-US-Chirp3-HD-Charon"})
        self.assertEqual(body["audioConfig"]["speakingRate"], 1.1)
        self.assertAlmostEqual(body["audioConfig"]["volumeGainDb"], -6.02, places=2)
        self.assertNotIn("pitch", body["audioConfig"])  # Chirp 3 HD has no pitch control
        self.assertEqual(vg.gcloud_request_body("hola", "es-US-Neural2-B", 9, 1.0, 2)["audioConfig"]["pitch"], 2.0)
        self.assertEqual(vg.gcloud_request_body("hola", "es-US-Neural2-B", 9, 1.0, 0)["audioConfig"]["speakingRate"], 2.0)

    def test_synthesis_with_api_key_and_retry(self):
        busy = MagicMock(status_code=503, text="busy")
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"audioContent": _wav(1.2)}
        out = os.path.join(self.temp_dir, "a", "voice.mp3")
        with patch.object(vg.requests, "post", side_effect=[busy, ok, ok]) as post:
            sub = vg.gcloud_tts(
                "Hola nutrias. " + "Una frase. " * 500, "gcloud:es-US-Chirp3-HD-Charon", 1.0, out,
                app_config={"gcloud_tts_api_key": "k"}, sleep=lambda s: None,
            )
        self.assertIsNotNone(sub)
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(post.call_args.kwargs["headers"], {"X-Goog-Api-Key": "k"})
        self.assertEqual(post.call_count, 3)  # one retry, then two chunks
        self.assertGreater(os.path.getsize(out), 4000)  # two chunks of 1.2 s and the pause between

    def test_failures(self):
        out = os.path.join(self.temp_dir, "v.mp3")
        bad = MagicMock(status_code=403, text="denied")
        with patch.object(vg.requests, "post", return_value=bad):
            self.assertIsNone(vg.gcloud_tts("Hola.", "gcloud:es-US-Chirp3-HD-Charon", 1.0, out, app_config={"gcloud_tts_api_key": "k"}))
        with patch.object(vg, "_gcloud_headers", side_effect=RuntimeError("no credentials")):
            self.assertIsNone(vg.gcloud_tts("Hola.", "gcloud:es-US-Chirp3-HD-Charon", 1.0, out, app_config={}))
        self.assertIsNone(vg.gcloud_tts("Hola.", "gcloud:", 1.0, out, app_config={}))

    def test_application_default_credentials_bill_the_project(self):
        credentials = MagicMock(token="t", quota_project_id=None)
        with patch("google.auth.default", return_value=(credentials, "adc-project")):
            headers = vg._gcloud_headers({"gemini_vertex_project": "mi-proyecto"})
            self.assertEqual(headers, {"Authorization": "Bearer t", "x-goog-user-project": "mi-proyecto"})
            with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": ""}):
                self.assertEqual(vg._gcloud_headers({})["x-goog-user-project"], "adc-project")

    def test_dispatch_from_voice_service(self):
        with patch.object(vg, "gcloud_tts", return_value="sub") as gcloud:
            self.assertEqual(voice.tts("Hola", "gcloud:es-US-Chirp3-HD-Charon", 1.0, "/tmp/x.mp3"), "sub")
        gcloud.assert_called_once()


class TestVoiceLab(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)

    def test_audition_and_cli(self):
        def fake_tts(text, name, rate, path):
            if "bad" in name:
                return None
            Path(path).write_bytes(b"mp3")
            return object()

        with patch.object(voice, "tts", side_effect=fake_tts), patch.dict(config.app, {"gemini_tts_style": "antes"}):
            results = vg.audition(["gemini:Puck-Upbeat", "bad:x"], "Hola", self.temp_dir, style="divulgador")
            self.assertEqual(config.app["gemini_tts_style"], "antes")  # restored
            self.assertEqual([r["ok"] for r in results], [True, False])
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = voice_lab.run(["--voices", "gemini:Puck-Upbeat", "--out", self.temp_dir, "--text", "Hola"])
        self.assertEqual(code, 0)
        self.assertIn("Gemini Flash TTS", stdout.getvalue())
        summary = json.loads(Path(self.temp_dir, "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["results"][0]["voice"], "gemini:Puck-Upbeat")
        listing = io.StringIO()
        with redirect_stdout(listing):
            voice_lab.run(["--list"])
        self.assertIn("divulgador", json.loads(listing.getvalue())["presets"])
        self.assertGreater(vg.estimate_cost(9000, "gemini-flash"), 0)
        self.assertEqual(vg.estimate_cost(9000, "edge"), 0)


if __name__ == "__main__":
    unittest.main()
