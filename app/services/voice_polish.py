"""
Studio-quality narration for list videos.

* The narration is kept at 48 kHz from the TTS file to the final mix (only the
  timing analysis uses a 24 kHz copy), so voices that are born at 44.1 kHz
  keep their highs.
* ``master_voice`` runs a broadcast-style chain: a high-pass for rumble, a
  little less "boxiness", more presence, an exciter that restores the "air"
  of voices generated at 24 kHz (Gemini, Cloud TTS), a gentle de-esser and
  compressor, then a two-pass (linear) loudness normalization to -14 LUFS and
  a true-peak limiter, so it never pumps or distorts.
* ``stretch_pauses`` lengthens the breaths between sentences for a calm,
  unhurried delivery, and ``align_sentences`` finds when each sentence is
  said from those pauses, so pictures land on the right words even when the
  voice gives no word timings.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.utils import utils

HQ_RATE = 48000
ANALYSIS_RATE = 24000
LOUDNESS = {"I": -14.0, "TP": -1.5, "LRA": 11.0}
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def decode(audio_file: str, rate: int = HQ_RATE) -> np.ndarray:
    """Mono 16-bit samples of ``audio_file`` at ``rate``."""
    result = subprocess.run(
        [utils.get_ffmpeg_binary(), "-v", "error", "-i", audio_file, "-vn", "-ac", "1", "-ar", str(rate),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg failed")
    data = result.stdout[: len(result.stdout) - len(result.stdout) % 2]
    return np.frombuffer(data, dtype=np.int16).copy()


def half_rate(samples: np.ndarray) -> np.ndarray:
    """48 kHz -> 24 kHz (pair average, enough for timing analysis)."""
    usable = samples[: len(samples) - len(samples) % 2].astype(np.int32)
    return ((usable[0::2] + usable[1::2]) // 2).astype(np.int16)


def silences(samples: np.ndarray, rate: int, min_seconds: float = 0.12) -> List[Tuple[float, float]]:
    """(start, end) seconds of quiet runs, judged against the voice's own level."""
    hop = max(1, rate // 100)  # 10 ms
    count = len(samples) // hop
    if count < 3:
        return []
    frames = samples[: count * hop].astype(np.float32).reshape(count, hop) / 32768.0
    energy = np.sqrt((frames ** 2).mean(axis=1))
    loud = np.percentile(energy, 90)
    if loud <= 1e-5:
        return []
    quiet = energy < max(loud * 0.06, 2e-4)
    runs, start = [], None
    for index, flag in enumerate(quiet):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, count))
    return [(a * hop / rate, b * hop / rate) for a, b in runs if (b - a) * hop / rate >= min_seconds]


def stretch_pauses(
    samples: np.ndarray, rate: int, pause: float, min_gap: float = 0.24
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Lengthen the pauses between sentences to at least ``pause`` seconds.

    Only inner pauses of ``min_gap`` or more (breaths between sentences, not
    the short ones inside them) grow; returns the samples and (time, added
    seconds) for each insertion.
    """
    if pause <= 0 or len(samples) == 0:
        return samples, []
    gaps = silences(samples, rate, min_gap)
    total = len(samples) / rate
    pieces, inserted, cursor = [], [], 0
    for start, end in gaps:
        if start <= 0.05 or end >= total - 0.05:
            continue  # leading or trailing silence
        missing = pause - (end - start)
        if missing <= 0.02:
            continue
        middle = int((start + end) / 2 * rate)
        pieces.append(samples[cursor:middle])
        pieces.append(np.zeros(int(missing * rate), dtype=samples.dtype))
        inserted.append(((start + end) / 2, missing))
        cursor = middle
    if not inserted:
        return samples, []
    pieces.append(samples[cursor:])
    return np.concatenate(pieces), inserted


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]


def align_sentences(text: str, samples: np.ndarray, rate: int) -> List[Tuple[float, float]]:
    """(start, end) of each sentence of ``text`` in the audio, from its pauses.

    Each boundary between sentences is the pause that best combines being long
    (a breath) with being near where the text says it should be. Falls back
    to spans proportional to the text when the audio has too few pauses.
    """
    parts = sentences(text)
    total = len(samples) / rate if rate else 0.0
    if not parts or total <= 0:
        return []
    gaps = silences(samples, rate, 0.12)
    voiced_start = gaps[0][1] if gaps and gaps[0][0] <= 0.02 else 0.0
    voiced_end = gaps[-1][0] if gaps and gaps[-1][1] >= total - 0.02 else total
    inner = [(a, b) for a, b in gaps if a > voiced_start + 0.05 and b < voiced_end - 0.05]
    lengths = np.array([len(p) + 1 for p in parts], dtype=np.float64)
    expected = voiced_start + (voiced_end - voiced_start) * np.cumsum(lengths)[:-1] / lengths.sum()
    boundaries = len(parts) - 1
    if boundaries == 0:
        return [(voiced_start, voiced_end)]
    if len(inner) < boundaries:
        edges = [voiced_start, *expected.tolist(), voiced_end]
        return [(edges[k], edges[k + 1]) for k in range(len(parts))]
    centers = np.array([(a + b) / 2 for a, b in inner])
    sizes = np.array([b - a for a, b in inner])
    tolerance = max(1.2, 0.12 * (voiced_end - voiced_start))
    # score[k][j]: how well pause j fits boundary k; pick an increasing path.
    score = np.minimum(sizes, 0.9)[None, :] * 3.0 - np.abs(centers[None, :] - expected[:, None]) / tolerance
    best = np.full(score.shape, -np.inf)
    back = np.zeros(score.shape, dtype=int)
    best[0] = score[0]
    for k in range(1, boundaries):
        running, where = -np.inf, -1
        for j in range(len(inner)):
            if j > 0 and best[k - 1][j - 1] > running:
                running, where = best[k - 1][j - 1], j - 1
            if where >= 0:
                best[k][j] = running + score[k][j]
                back[k][j] = where
    j = int(np.argmax(best[-1]))
    chosen = [j]
    for k in range(boundaries - 1, 0, -1):
        j = int(back[k][j])
        chosen.append(j)
    chosen.reverse()
    spans, start = [], voiced_start
    for j in chosen:
        spans.append((start, inner[j][0]))
        start = inner[j][1]
    spans.append((start, voiced_end))
    return spans


def high_band_share(samples: np.ndarray, rate: int, cutoff: float = 12500.0) -> float:
    """Share of the energy above ``cutoff`` Hz (about 0 for voices made at 24 kHz)."""
    if len(samples) < rate // 10:
        return 0.0
    window = samples[: min(len(samples), rate * 20)].astype(np.float32)
    spectrum = np.abs(np.fft.rfft(window)) ** 2
    frequencies = np.fft.rfftfreq(len(window), 1.0 / rate)
    total = spectrum.sum()
    return float(spectrum[frequencies >= cutoff].sum() / total) if total > 0 else 0.0


def polish_chain(add_air: bool) -> str:
    """The ffmpeg filters that make TTS sound like a well-recorded voice."""
    chain = [
        "highpass=f=70:poles=2",
        "equalizer=f=320:t=q:w=1.1:g=-2.5",  # less boxy
        "equalizer=f=140:t=q:w=0.8:g=1.2",  # a little body
        "equalizer=f=3600:t=q:w=1.3:g=2.2",  # presence and clarity
    ]
    if add_air:
        chain.append("aexciter=amount=1.1:drive=6.5:blend=0:freq=5500:ceil=18000")
    chain += [
        "treble=g=1.5:f=9500:t=s",
        "deesser=i=0.35:m=0.5:f=0.5",
        "acompressor=threshold=0.089:ratio=2.5:attack=8:release=160:makeup=1.6:knee=4",
    ]
    return ",".join(chain)


def _measure(audio_file: str, filters: str) -> Optional[dict]:
    target = f"loudnorm=I={LOUDNESS['I']}:TP={LOUDNESS['TP']}:LRA={LOUDNESS['LRA']}:print_format=json"
    result = subprocess.run(
        [utils.get_ffmpeg_binary(), "-hide_banner", "-nostats", "-i", audio_file,
         "-af", f"{filters},{target}" if filters else target, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr or "")
    if result.returncode != 0 or not match:
        return None
    try:
        data = json.loads(match.group(0))
        float(data["input_i"])
    except (ValueError, KeyError, TypeError):
        return None
    if not np.isfinite(float(data["input_i"])):
        return None
    return data


def master_voice(audio_file: str, output_file: str, polish: bool = True) -> str:
    """Polish (optional) and bring the narration to -14 LUFS without pumping; 48 kHz WAV out."""
    samples = decode(audio_file, HQ_RATE)
    filters = polish_chain(high_band_share(samples, HQ_RATE) < 0.002) if polish else ""
    filters = ",".join(f for f in (f"aresample={HQ_RATE}", filters) if f)
    measured = _measure(audio_file, filters)
    if measured is None:
        normalize = f"loudnorm=I={LOUDNESS['I']}:TP={LOUDNESS['TP']}:LRA={LOUDNESS['LRA']}"
    else:
        normalize = (
            f"loudnorm=I={LOUDNESS['I']}:TP={LOUDNESS['TP']}:LRA={LOUDNESS['LRA']}:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:linear=true"
        )
    graph = f"{filters},{normalize},alimiter=limit=0.84:level=false,aresample={HQ_RATE}"
    result = subprocess.run(
        [utils.get_ffmpeg_binary(), "-v", "error", "-y", "-i", audio_file, "-af", graph,
         "-ar", str(HQ_RATE), "-ac", "1", "-c:a", "pcm_s16le", output_file],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"voice mastering failed: {(result.stderr or '').strip()[-1500:]}")
    return output_file


def soft_limit(mix: np.ndarray, knee: float = 0.92) -> np.ndarray:
    """Round off only the peaks above ``knee`` (the voice itself stays untouched)."""
    over = np.abs(mix) > knee
    if over.any():
        room = 1.0 - knee
        mix = mix.copy()
        mix[over] = np.sign(mix[over]) * (knee + room * np.tanh((np.abs(mix[over]) - knee) / room))
    return mix


def shift_time(time: float, inserted: Sequence[Tuple[float, float]]) -> float:
    """Where a moment of the original audio lands after ``stretch_pauses``."""
    return time + sum(added for at, added in inserted if at < time)
