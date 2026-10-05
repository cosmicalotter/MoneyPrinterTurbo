import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import list_video_editor as editor
from app.services import list_video_fx as fx
from app.services import voice_polish as vp


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
