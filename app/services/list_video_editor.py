"""
Automatic editing for list videos.

An LLM "director" decides, per segment, the host character's expression, the
background footage that follows the narration and a few beats anchored to
exact words: a picture of what is being mentioned, a short key fact, or a
reaction of the host. The editor turns that plan into timed ffmpeg overlays
(chapter label, the host popping in and out with lip-flap and bouncy
expression changes, stickers, callouts, subscribe animation, progress bar) and
a list of sound effects.

Every decision is saved to ``edit-plan.json``; editing that file and passing it
back with ``--edit-plan`` re-renders the video with the changes.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger
from PIL import Image

from app.models.schema import VideoAspect
from app.services import gemini_media, icons, llm, material, web_images
from app.services import list_video_fx as fx
from app.services import list_video_host as host
from app.services import list_video_scenes as scenes
from app.utils import utils

BEAT_MODES = ("web", "ai", "none")
SUBSCRIBE_MODES = ("both", "intro", "outro", "none")
_NEUTRAL_EXPRESSIONS = ("explicando", "explaining", "neutral", "normal", "feliz", "happy", "sonriente")
_INTRO_EXPRESSIONS = ("sorprendido", "surprised", "asombrado", "feliz", "happy")
_OUTRO_EXPRESSIONS = ("feliz", "happy", "sonriente", "neutral")
_SFX_GAIN = {"whoosh": 0.55, "pop": 0.7, "tick": 0.8, "click": 0.9, "subscribe": 0.75, "bloop": 0.45}
SUBSCRIBE_SECONDS = 3.6
IMAGE_BEAT_MAX = 5.5
IMAGE_BEAT_MIN = 1.6
TEXT_BEAT_SECONDS = 2.6
MIN_BACKGROUND_SECONDS = 3.0
# Host size as a share of the frame height; the bottom of the sweater stays
# below the frame edge, like a presenter standing behind the screen border.
HOST_HEIGHT = {False: 0.5, True: 0.27}  # by portrait
HOST_SINK = 0.12


@dataclass
class EditOptions:
    assets_dir: str = ""
    beats: str = "web"
    subscribe: str = "both"
    accent: str = fx.DEFAULT_ACCENT
    progress_bar: bool = False
    sound_effects: bool = True
    plan_file: str = ""
    language: str = ""
    host: str = "auto"  # auto (comes and goes), always, none
    lip_sync: bool = False  # swap in the *_habla frames while the voice speaks
    scenes: bool = True  # full-screen explainer scenes between the footage
    illustrations: str = "icons"  # scene pictures: OpenMoji icons, or "ai" (Imagen)
    scene_color: str = ""  # canvas colour; "" is a light tint of the accent
    picture_check: bool = True  # let Gemini vision pick pictures (when configured)
    reference_plan: str = ""  # edit-plan.json of the same video in another language
    sfx_volume: float = 0.65  # scales every sound effect (1.0 = the original loudness)
    host_presence: str = "low"  # low, normal or high share of the video with the host
    openers: bool = True  # open each item with its number, title and a picture of exactly that
    fill_gaps: bool = True  # ask the LLM for pictures for sentences left with only footage
    look: str = "footage"  # "footage" (stock video with pictures and scenes) or "doodle" (all drawn)
    canvas_color: str = ""  # background of the doodle look; "" is a warm yellow
    boil: bool = True  # doodle look: drawings wobble very slightly, like hand-drawn animation
    logo: str = ""  # a corner badge: "nutria" (the bundled otter) or a picture file
    max_drawings: int = 260  # doodle look: AI drawings per video, animation frames included (photos after that)
    seamless: bool = False  # story format: segments flow into each other (no whoosh at the cuts)
    shot_seconds: float = 5.0  # doodle look: a new picture about this often (calm, documentary pace)
    director_review: bool = True  # doodle look: a film editor pass corrects the storyboard before drawing
    image_quality: str = "standard"  # AI drawings: "economy", "standard", "high" (pro key frames) or "max"
    ai_videos: int = 0  # doodle look: illustrations brought to life by Veo (4-8 s videos, paid per second)
    clips: str = "some"  # doodle look: real video clips in a frame now and then ("none", "some", "more")
    memes: str = "off"  # comic reaction cut-ins: "off", "otter" (drawn otter reactions) or "folder" (your memes)
    memes_dir: str = ""  # folder of reaction pictures and videos, by mood ("" is resource/memes)
    drawing_style: str = "cartoon"  # doodle look: "cartoon" (polished 2D animation) or "ink" (pen doodles)


@dataclass
class SoundEvent:
    time: float
    path: str
    gain: float = 1.0


@dataclass
class Narration:
    """What the renderer measured for one segment before drawing it."""

    pcm: bytes  # 24 kHz mono, for timing analysis
    speech_seconds: float
    frames: int
    sub_maker: object = None
    hq_pcm: bytes = b""  # the same narration at 48 kHz, for the final mix
    spans: List[Tuple[float, float]] = field(default_factory=list)  # when each sentence is said


@dataclass
class _Beat:
    kind: str  # "image", "images" (a group shown side by side) or "text"
    start: float
    end: float
    query: str = ""
    look: str = "diagram"
    text: str = ""
    icon: str = ""
    at: str = ""
    members: List["_Beat"] = field(default_factory=list)
    picture: str = ""


@dataclass
class SegmentEdit:
    overlays: List[fx.Overlay] = field(default_factory=list)
    sounds: List[Tuple[float, str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anchors: where in the narration a beat happens
# ---------------------------------------------------------------------------


def normalize_words(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", text)


def _find_words(words: List[str], target: List[str]) -> int:
    size = len(target)
    for index in range(len(words) - size + 1):
        if words[index : index + size] == target:
            return index
    return -1


def cue_word_times(sub_maker) -> List[float]:
    """Word start times from Edge TTS word boundaries; empty for other voices."""
    cues = getattr(sub_maker, "cues", None) or []
    times = []
    for cue in cues:
        start = getattr(cue, "start", None)
        if start is not None and hasattr(start, "total_seconds"):
            times.append(start.total_seconds())
    return times


def anchor_time(
    text: str, anchor: str, sub_maker, speech_seconds: float, spans: Optional[List[Tuple[float, float]]] = None
) -> Optional[float]:
    """Seconds into the segment when ``anchor`` is spoken, or None.

    Word timings (Edge voices) are used when there are; else ``spans``, when
    each sentence is said (found from the pauses of the audio), with the
    words spread over their sentence; else the text's proportions.
    """
    words = normalize_words(text)
    target = normalize_words(anchor)
    if not words or not target:
        return None
    index = _find_words(words, target)
    if index < 0 and len(target) > 2:
        index = _find_words(words, target[:2])
    if index < 0:
        index = next((words.index(w) for w in target if len(w) > 3 and w in words), -1)
    if index < 0:
        return None
    return _word_time(index, words, text, sub_maker, speech_seconds, spans)


def _word_time(
    index: int, words: List[str], text: str, sub_maker, speech_seconds: float, spans: Optional[List[Tuple[float, float]]]
) -> float:
    """When the ``index``-th normalized word of ``text`` is said."""
    times = cue_word_times(sub_maker)
    if times and len(times) >= len(words) * 0.6:
        position = min(len(times) - 1, round(index * len(times) / len(words)))
        return max(0.0, min(times[position], speech_seconds))

    if spans:
        sentence_words = [normalize_words(s) for s in re.split(r"(?<=[.!?…])\s+", text or "") if s.strip()]
        if len(sentence_words) == len(spans) and sum(map(len, sentence_words)) == len(words):
            first = 0
            for (start, end), sentence in zip(spans, sentence_words):
                if index < first + len(sentence):
                    lengths = [len(word) + 1 for word in sentence]
                    inside = index - first
                    return start + (end - start) * sum(lengths[:inside]) / max(1, sum(lengths))
                first += len(sentence)

    lengths = [len(word) + 1 for word in words]
    return speech_seconds * sum(lengths[:index]) / sum(lengths)


GROUP_TAIL = 3.5  # a group of pictures stays this long after its last one
GROUP_MAX = 9.0


def _picture_spans(beat: _Beat) -> Tuple[float, float]:
    """(shortest, longest) seconds a picture beat should stay on screen."""
    if beat.kind == "images":
        last = beat.members[-1].start - beat.start
        return last + IMAGE_BEAT_MIN, min(GROUP_MAX, last + GROUP_TAIL)
    return IMAGE_BEAT_MIN, IMAGE_BEAT_MAX


def schedule_beats(beats: List[dict], text: str, narration: Narration, duration: float) -> List[_Beat]:
    """Give plan beats start/end times that never overlap the same zone."""

    def locate(anchor):
        return anchor_time(text, anchor, narration.sub_maker, narration.speech_seconds, narration.spans)

    def picture(data, start) -> _Beat:
        return _Beat(
            kind="image", start=start, end=0.0, query=data.get("query", ""), look=data.get("look", "diagram"),
            icon=data.get("icon", ""), at=data.get("at", ""),
        )

    timed = []
    for beat in beats:
        kind = beat.get("type")
        if kind == "images":
            members = []
            for data in beat.get("items") or []:
                start = locate(data.get("at", ""))
                if start is not None and start <= duration - 1.0:
                    members.append(picture(data, max(0.15, start - 0.1)))
            members.sort(key=lambda b: b.start)
            for previous, member in zip(members, members[1:]):
                member.start = max(member.start, previous.start + 0.5)
            members = [m for m in members if m.start <= duration - 1.0]
            if len(members) >= 2:
                timed.append(_Beat(kind="images", start=members[0].start, end=0.0, at=members[0].at, members=members))
            elif members:
                timed.append(members[0])
            continue
        if kind not in ("image", "text"):
            continue
        start = locate(beat.get("at", ""))
        if start is None:
            logger.debug(f"beat anchor not found in narration: {beat.get('at')!r}")
            continue
        start = max(0.15, start - 0.1)
        if start > duration - 1.0:
            continue
        if kind == "image":
            timed.append(picture(beat, start))
        else:
            timed.append(_Beat(kind="text", start=start, end=0.0, text=beat.get("text", ""), at=beat.get("at", "")))
    timed.sort(key=lambda b: b.start)

    scheduled: List[_Beat] = []
    for zone in (("image", "images"), ("text",)):
        # Keep the earliest beat and skip any that would cut it short, then
        # let each kept beat run until the next one of the same zone.
        kept: List[_Beat] = []
        for beat in (b for b in timed if b.kind in zone):
            shortest = _picture_spans(beat)[0] if beat.kind != "text" else 1.2
            if kept:
                previous_shortest = _picture_spans(kept[-1])[0] if kept[-1].kind != "text" else 1.2
                if beat.start - kept[-1].start < previous_shortest + 0.15:
                    continue
            if duration - 0.3 - beat.start < min(shortest, IMAGE_BEAT_MIN if beat.kind != "text" else 1.2):
                continue
            kept.append(beat)
        for position, beat in enumerate(kept):
            longest = _picture_spans(beat)[1] if beat.kind != "text" else TEXT_BEAT_SECONDS
            limit = kept[position + 1].start - 0.15 if position + 1 < len(kept) else duration - 0.3
            beat.end = min(beat.start + longest, limit)
            if beat.kind == "images":
                beat.members = [m for m in beat.members if m.start < beat.end - 1.0]
                for member in beat.members:
                    member.end = beat.end
        scheduled += [b for b in kept if b.kind != "images" or b.members]
    return sorted(scheduled, key=lambda b: b.start)


def sentence_at(text: str, anchor: str) -> str:
    """The sentence of ``text`` that contains ``anchor`` (for picture checks)."""
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text or "") if s.strip()]
    target = normalize_words(anchor)
    for sentence in sentences:
        words = normalize_words(sentence)
        if target and _find_words(words, target[: min(2, len(target))]) >= 0:
            return sentence.strip()
    return (text or "").strip()[:300]


def _plan_anchors(entry: dict) -> Tuple[List[str], List[str]]:
    """(beat anchors, scene anchors) of a plan entry."""
    beats = []
    for beat in entry.get("beats") or []:
        if beat.get("type") == "images":
            beats += [item.get("at", "") for item in beat.get("items") or []]
        elif beat.get("type") in ("image", "text"):
            beats.append(beat.get("at", ""))
    shown = []
    for scene in entry.get("scenes") or []:
        shown.append(scene.get("at", ""))
        for key in ("items", "terms", "labels"):
            shown += [item.get("at", "") for item in scene.get(key) or [] if isinstance(item, dict)]
    return [a for a in beats if a], [a for a in shown if a]


def uncovered_sentences(text: str, entry: dict, kind: str, min_words: int = 7) -> List[str]:
    """Sentences of a segment with nothing explanatory on screen, only footage.

    A scene covers its sentence and the next (it lasts several seconds); an
    item's first sentence belongs to its opener.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text or "") if s.strip()]
    words = [normalize_words(sentence) for sentence in sentences]

    def where(anchor: str) -> int:
        target = normalize_words(anchor)[:3]
        if not target:
            return -1
        return next((i for i, sentence in enumerate(words) if _find_words(sentence, target) >= 0), -1)

    covered = {0} if kind == "item" else set()
    beats, shown = _plan_anchors(entry)
    for anchor in beats:
        covered.add(where(anchor))
    for anchor in shown:
        index = where(anchor)
        if index >= 0:
            covered.update((index, index + 1))
    return [s for i, s in enumerate(sentences) if i not in covered and len(s.split()) >= min_words]


def scene_canvas_color(option: str, accent: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """"" is a light tint of the accent (soft brand paper); also "accent", "white" or a hex colour."""
    value = (option or "").strip().lower()
    if value == "accent":
        return accent
    if value == "white":
        return (250, 247, 242)
    if value:
        return fx.parse_color(value)
    return tuple(int(c + (255 - c) * 0.84) for c in accent)


def chip_spans(chip: fx.Overlay, covered: List[Tuple[float, float]], duration: float, travel: int) -> List[fx.Overlay]:
    """The chapter label shown only between scenes: it slides out before each
    scene and back in after it (the first entrance keeps the label's own x)."""
    spans: List[List[float]] = [[0.0, duration]]
    for start, end in sorted(covered):
        pieces = []
        for a, b in spans:
            if b <= start - 0.05 or a >= end + 0.05:
                pieces.append([a, b])
                continue
            if start - 0.05 - a > 0:
                pieces.append([a, start - 0.05])
            if b - (end + 0.05) > 0:
                pieces.append([end + 0.05, b])
        spans = pieces
    overlays = []
    for a, b in spans:
        if b - a < 1.2:
            continue
        leave = f"{travel}*pow(clip((t-{b - 0.3:.3f})/0.3,0,1),2)" if b < duration - 0.05 else "0"
        if a <= 0.0:
            x = f"{chip.x}-{leave}"
        else:
            back = f"{travel}*(1-{fx.ease_expression(0.35, a)})"
            base = chip.x.split("-(", 1)[0]
            x = f"{base}-{back}-{leave}"
        overlays.append(fx.Overlay(chip.source, x=x, y=chip.y, start=a, end=b))
    return overlays


PILL_SHOTS = ("illustration", "animation", "archive", "figure", "zoom")


def _pill_over(scene: scenes.Scene) -> bool:
    """The section's label may stay over this shot (a full-screen picture with nothing written at its top)."""
    return scene.type in PILL_SHOTS or (scene.type == "clip" and scene.frame == "full")


def _after_opener(beat: _Beat, opener_end: float) -> Optional[_Beat]:
    """A beat said while the section opener is on screen waits for it to leave."""
    start = opener_end + 0.2
    if beat.start >= start:
        return beat
    shortest = 1.2 if beat.kind == "text" else IMAGE_BEAT_MIN
    if beat.end - start < shortest:
        return None
    beat.start = start
    for number, member in enumerate(beat.members):
        member.start = max(member.start, start + 0.4 * number)
    return beat


def _before_scenes(beat: _Beat, cover: List[Tuple[float, float]]) -> Optional[_Beat]:
    """Cut a beat short before the next scene; None when it would be hidden (and heard) under one."""
    for start, end in cover:
        if beat.start < end + 0.2 and beat.end > start - 0.2:
            if beat.start >= start - 0.2:
                return None
            beat.end = start - 0.2
    shortest = 1.2 if beat.kind == "text" else IMAGE_BEAT_MIN
    if beat.end - beat.start < shortest:
        return None
    if beat.kind == "images":
        beat.members = [m for m in beat.members if m.start < beat.end - 1.0]
        for member in beat.members:
            member.end = beat.end
        if not beat.members:
            return None
    return beat


def _avoid_cover(windows: List[host.HostWindow], cover: List[Tuple[float, float]]) -> List[host.HostWindow]:
    """Host visits never begin under a scene: the bloop would sound over nothing."""
    kept = []
    for window in windows:
        for start, end in cover:
            if window.enter and start - 0.4 <= window.start < end:
                window.start = end + 0.1
        if window.end - window.start >= host.MIN_WINDOW or not window.enter:
            kept.append(window)
    return kept


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _pick(poses: Dict[str, fx.CharacterPose], preferred) -> str:
    for name in preferred:
        if name in poses:
            return name
    return sorted(poses)[0] if poses else ""


def default_plan(segments, poses) -> List[dict]:
    plan = []
    for index, segment in enumerate(segments):
        preferred = {
            "intro": _INTRO_EXPRESSIONS,
            "outro": _OUTRO_EXPRESSIONS,
        }.get(segment.kind, _NEUTRAL_EXPRESSIONS)
        plan.append(
            {"index": index, "expression": _pick(poses, preferred), "backgrounds": [], "scenes": [], "beats": []}
        )
    return plan


LOOKS = ("footage", "doodle")
DOODLE_COLOR = (244, 194, 79)  # the warm yellow of hand-drawn explainer channels
HOLD_FACTOR = 1.8  # a picture held longer than this many shot lengths gets more shots
DARK_CLIP = 60  # stock clips darker than this (mean 0-255) look like a black box on the canvas
MEME_POSES = {
    "shock": "sorprendido", "mindblown": "sin_palabras", "laugh": "riendo", "facepalm": "preocupado",
    "confused": "pensando", "scared": "preocupado", "sad": "triste", "proud": "feliz", "suspicious": "pensando",
    "panic": "preocupado",
}
# Folder and file names of the memes folder (Spanish or English) and the mood they mean.
MOOD_ALIASES = {
    "shock": "shock", "sorpresa": "shock", "sorprendido": "shock", "asombro": "shock",
    "mindblown": "mindblown", "mente": "mindblown", "explota": "mindblown", "wow": "mindblown",
    "laugh": "laugh", "risa": "laugh", "riendo": "laugh", "jaja": "laugh",
    "facepalm": "facepalm", "verguenza": "facepalm", "palma": "facepalm",
    "confused": "confused", "confundido": "confused", "confusion": "confused",
    "scared": "scared", "miedo": "scared", "susto": "scared",
    "sad": "sad", "triste": "sad", "tristeza": "sad",
    "proud": "proud", "orgullo": "proud", "orgulloso": "proud",
    "suspicious": "suspicious", "sospecha": "suspicious", "sospechoso": "suspicious",
    "panic": "panic", "panico": "panic",
}
MEME_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp") + scenes.VIDEO_EXTENSIONS


def meme_library(folder: str) -> Dict[str, List[str]]:
    """{mood: [files]} of a memes folder: files in a folder named after the mood, or named "<mood>-...".

    Moods and their Spanish names are in ``MOOD_ALIASES``; other files are ignored.
    """
    library: Dict[str, List[str]] = {}
    if not folder or not os.path.isdir(folder):
        return library
    for root, _, files in os.walk(folder):
        relative = os.path.relpath(root, folder)
        parent = "" if relative == "." else normalize_words(relative.split(os.sep)[0])
        for name in sorted(files):
            stem, extension = os.path.splitext(name)
            if extension.lower() not in MEME_EXTENSIONS:
                continue
            words = normalize_words(stem)
            mood = MOOD_ALIASES.get(parent[0] if parent else "") or (MOOD_ALIASES.get(words[0]) if words else None)
            if mood:
                library.setdefault(mood, []).append(os.path.join(root, name))
    return library


def _clip_credit(info) -> str:
    """Attribution line of a stock video (the providers ask for credit when possible)."""
    source = getattr(info, "source_info", None) or {}
    provider = str(source.get("provider") or getattr(info, "provider", "") or "stock").capitalize()
    creator = source.get("creator") if isinstance(source.get("creator"), dict) else {}
    name = str(creator.get("name") or "").strip()
    page = str(source.get("source_page") or "").strip()
    line = f"Video: {name + ' / ' if name else ''}{provider}"
    return f"{line} ({page})" if page else line


def word_times(text: str, sub_maker, speech_seconds: float, spans=None) -> List[Tuple[str, float]]:
    """Every word of ``text`` as written, with when it is said (the same estimate as the anchors)."""
    tokens = (text or "").split()
    words = normalize_words(text)
    if not words:
        return []
    timed = []
    position = 0
    last = 0.0
    for token in tokens:
        count = len(normalize_words(token))
        if count:
            last = _word_time(position, words, text, sub_maker, speech_seconds, spans)
            position += count
        timed.append((token, last))
    return timed


def load_plan_file(path: str, segment_count: int, expressions: List[str]) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    segments = data.get("segments") if isinstance(data, dict) else data
    if isinstance(segments, list) and any(isinstance(e, dict) and "shots" in e for e in segments):
        return llm.normalize_storyboard(data, segment_count, expressions)
    return llm.normalize_edit_plan(data, segment_count, expressions)


def load_plan_bible(path: str) -> dict:
    """The visual bible saved in an edit-plan.json ({} when there is none)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        return {}
    bible = data.get("bible") if isinstance(data, dict) else None
    if not isinstance(bible, dict):
        return {}
    sections = bible.get("sections") or {}
    if isinstance(sections, dict):
        bible = dict(bible, sections=[{"index": int(k), "archive": v} for k, v in sections.items() if str(k).isdigit()])
    try:
        return llm.normalize_visual_bible(bible)
    except ValueError:
        return {}


_STOP_WORDS = frozenset(
    "the and with from into onto over under while their there them they this that these those then than what when "
    "where which shot wide medium close view frame same again very more most some side back front left right".split()
)
REPEAT_SIMILARITY = 0.65  # share of the words two picture descriptions have in common to be the same picture
PICTURE_KINDS = ("archive", "illustration", "animation", "single")


def _subject_words(shot: dict) -> set:
    """The content words of what a picture shot shows (to spot the same picture planned twice)."""
    if shot.get("type") == "archive":
        source = shot.get("query", "")
    elif shot.get("type") == "animation":
        source = " ".join(f.get("draw", "") for f in shot.get("frames") or [] if isinstance(f, dict))
    else:
        source = shot.get("draw", "") or shot.get("query", "")
    return {word for word in normalize_words(source) if len(word) > 3 and word not in _STOP_WORDS}


def drop_repeats(plan: List[dict]) -> int:
    """Leave out picture shots that show again what an earlier shot showed: the same archive search, or a drawing
    described with nearly the same words. A shot that continues the one before it is kept. Returns how many."""
    seen: List[Tuple[str, str, set]] = []
    dropped = 0
    for entry in plan:
        kept = []
        for shot in entry.get("shots") or []:
            kind = shot.get("type")
            if kind not in PICTURE_KINDS or shot.get("continue") or shot.get("image"):
                kept.append(shot)
                continue
            query = " ".join(normalize_words(shot.get("query", ""))) if kind == "archive" else ""
            words = _subject_words(shot)
            repeated = any(query and query == q for _, q, _ in seen) or (
                len(words) >= 4 and any(len(w) >= 4 and len(words & w) / len(words | w) >= REPEAT_SIMILARITY for _, _, w in seen)
            )
            if repeated:
                dropped += 1
                continue
            seen.append((kind, query, words))
            kept.append(shot)
        entry["shots"] = kept
    if dropped:
        logger.info(f"{dropped} shots that showed the same picture again were left out")
    return dropped


_PINNED_KEYS = ("image", "video")


def _pinned_places(plan: List[dict]):
    """Every dict of a plan that may hold a pinned picture or video."""
    for entry in plan:
        if isinstance(entry.get("opener"), dict):
            yield entry["opener"]
        for shot in entry.get("shots") or []:
            if not isinstance(shot, dict):
                continue
            yield shot
            for key in ("frames", "items"):
                yield from (part for part in shot.get(key) or [] if isinstance(part, dict))


def pin_paths(plan: List[dict], base: str) -> None:
    """Pinned pictures and videos given relative to ``base`` (the plan file's folder) become absolute paths."""
    for place in _pinned_places(plan):
        for key in _PINNED_KEYS:
            if place.get(key) and not os.path.isabs(place[key]):
                place[key] = os.path.normpath(os.path.join(base, place[key]))


REVIEW_FILE = "review.json"
REVIEW_PLAN_FILE = "review-plan.json"  # the plan with every picture pinned, as it was reviewed


def llm_part_key(kind: str) -> str:
    """The list of parts of a shot type that has them ("terms" of an equation, "labels" of an annotated picture)."""
    return scenes.PART_KEYS.get(kind, "items")


def doodle_fallback(segments, poses) -> List[dict]:
    """A plain storyboard when the LLM gives none: one drawing per segment."""
    plan = default_plan(segments, poses)
    for entry, segment in zip(plan, segments):
        words = segment.text.split()
        anchor = " ".join(words[:3])
        if segment.kind == "item" and segment.image_term:
            entry["shots"] = [{"type": "single", "at": anchor, "label": segment.name or "", "draw": segment.image_term,
                               "icon": "", "text": ""}]
        elif poses:
            entry["shots"] = [{"type": "single", "at": anchor, "label": "", "pose": entry["expression"], "draw": "",
                               "icon": "", "text": ""}]
        else:
            entry["shots"] = []
    return plan


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


class Editor:
    def __init__(
        self,
        options: EditOptions,
        theme: fx.Theme,
        task_dir: str,
        segments,
        narrations: List[Narration],
    ):
        self.options = options
        self.theme = theme
        self.task_dir = task_dir
        self.segments = segments
        self.narrations = narrations
        self.work_dir = os.path.join(task_dir, "edit")
        os.makedirs(self.work_dir, exist_ok=True)
        self.total = sum(n.frames for n in narrations) / 30.0 or 1.0
        self.doodle = options.look == "doodle"
        # The doodle look draws the otter inside its shots even without a host track.
        self.poses = fx.load_character(options.assets_dir) if options.host != "none" or self.doodle else {}
        self.sfx = fx.resolve_sfx(options.assets_dir, self.work_dir) if options.sound_effects else {}
        self.credits: List[str] = []
        self.warnings: List[str] = []
        self._used_urls: set = set()
        self._subscribe = self._prepare_subscribe()
        self._renderer: Optional[host.HostRenderer] = None
        self._scene_renderer: Optional[scenes.SceneRenderer] = None
        self._beats: Dict[int, List[Tuple[_Beat, str]]] = {}
        self._scenes: Dict[int, List[scenes.Scene]] = {}
        self._scene_pictures: Dict[Tuple[str, str], Optional[object]] = {}
        self._item_pictures: Dict[int, Optional[object]] = {}
        self._gemini = gemini_media.enabled()
        self.host_plan: List[host.HostSegment] = []
        self.plan: List[dict] = []
        self._drawings = 0
        self._drawing_lock = threading.Lock()
        self._drawn: Dict[str, int] = {}
        self._used_icons: set = set()
        self._icon_lock = threading.Lock()
        self._storyboarded = False  # the doodle plan came from the LLM (its long holds can be filled)
        self.bible: dict = {}  # the art direction of the video: style, recurring characters, real pictures
        self._failed_drawings = 0
        self._memes: Optional[Dict[str, List[str]]] = None
        self._used_memes: set = set()
        self._unpictured: List[scenes.Scene] = []  # drawn shots whose drawing failed (rescued after the others)
        self._sheets: Dict[str, Tuple[str, str]] = {}  # character id -> (name, character sheet file)
        # What happened to the pictures, for render-report.txt.
        self._stats: Dict[str, int] = {key: 0 for key in (
            "drawn", "rejected", "budget", "real", "photo", "otter", "items_dropped", "shots_dropped", "build_failed",
            "archive", "archive_drawn", "pinned", "videos", "sheets", "repeats")}

    @property
    def wants_footage(self) -> bool:
        """False in the doodle look, where the drawn canvas replaces stock video."""
        return not self.doodle

    # -- plan ---------------------------------------------------------------

    def make_plan(self) -> List[dict]:
        expressions = sorted(self.poses)
        count = len(self.segments)
        plan = None
        if self.options.plan_file:
            plan = load_plan_file(self.options.plan_file, count, expressions)
            pin_paths(plan, os.path.dirname(os.path.abspath(self.options.plan_file)))
            self.bible = load_plan_bible(self.options.plan_file)
            logger.info(f"using edit plan: {self.options.plan_file}")
        elif expressions or self.options.beats != "none":
            payload = [
                {"index": i, "kind": s.kind, "title": s.chapter, "text": s.text}
                for i, s in enumerate(self.segments)
            ]
            reference = self._reference_plan()
            if self.doodle:
                for entry, segment, narration in zip(payload, self.segments, self.narrations):
                    seconds = narration.frames / 30.0
                    if self.options.openers and segment.kind == "item":
                        seconds = max(0.0, seconds - scenes.OPENER_SECONDS)
                    entry["seconds"] = round(seconds, 1)
                    entry["shots"] = llm.storyboard_shot_target(seconds, self.options.shot_seconds)
                if not reference:
                    # The art director reads the whole script first: style, recurring characters, real pictures.
                    self.bible = llm.generate_visual_bible(
                        [{"index": e["index"], "kind": e["kind"], "title": e["title"], "text": e["text"]} for e in payload],
                        self.options.language, self.persona,
                    ) or {}
                else:
                    self.bible = self._reference_bible()
                plan = llm.generate_storyboard(payload, expressions, self.options.language, reference=reference,
                                               openers=self.options.openers, clips=self.options.clips,
                                               memes=self.options.memes in ("otter", "folder"), bible=self.bible or None,
                                               review=self.options.director_review)
                if plan is None:
                    self.warnings.append("the LLM did not return a storyboard; one drawing per segment was used")
                else:
                    self._storyboarded = not reference
                    self._stats["repeats"] += drop_repeats(plan)
            elif reference:
                plan = llm.generate_edit_plan(payload, expressions, self.options.language, reference=reference,
                                              openers=self.options.openers)
            else:
                plan = llm.generate_edit_plan(payload, expressions, self.options.language, openers=self.options.openers)
            if plan is None and not self.doodle:
                self.warnings.append(
                    "the LLM did not return an edit plan; only the base visuals were used"
                )
            elif plan is not None and not self.doodle and not reference and self.options.fill_gaps and self.options.beats != "none":
                self._fill_gaps(plan)
        fallback = doodle_fallback(self.segments, self.poses) if self.doodle else default_plan(self.segments, self.poses)
        plan = plan or fallback
        for entry, default in zip(plan, fallback):
            if not entry.get("expression"):
                entry["expression"] = default["expression"]
            entry.setdefault("backgrounds", [])
            entry.setdefault("scenes", [])
            if self.doodle:
                entry.setdefault("shots", default.get("shots", []))
            if not self.options.scenes:
                entry["scenes"] = []
            if self.options.beats == "none":
                entry["beats"] = [b for b in entry.get("beats") or [] if b.get("type") == "react"]
        self.plan = plan
        self._save_plan()
        return plan

    def _fill_gaps(self, plan: List[dict]) -> None:
        """A second, short LLM pass: one picture for each sentence left with only footage."""
        gaps = [
            {"index": index, "sentence": sentence}
            for index, (segment, entry) in enumerate(zip(self.segments, plan))
            for sentence in uncovered_sentences(segment.text, entry, segment.kind)
        ]
        if not gaps:
            return
        try:
            extra = llm.generate_gap_beats(gaps[:40], self.options.language)
        except Exception as exc:
            logger.warning(f"could not fill the gaps of the edit plan: {type(exc).__name__}: {exc}")
            return
        added = 0
        for index, beats in extra.items():
            if 0 <= index < len(plan):
                plan[index].setdefault("beats", [])
                plan[index]["beats"] += beats
                added += len(beats)
        logger.info(f"{added} pictures added where only footage was planned ({len(gaps)} sentences)")

    @property
    def persona(self) -> str:
        """"otter" when the bundled otter narrates (the scripts and the art direction know it)."""
        return "otter" if os.path.basename(os.path.normpath(self.options.assets_dir or "")) == "nutria" else ""

    def _reference_bible(self) -> dict:
        path = self.options.reference_plan
        return load_plan_bible(path) if path and os.path.isfile(path) else {}

    def _reference_plan(self) -> Optional[list]:
        path = self.options.reference_plan
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                segments = json.load(fp).get("segments")
        except (OSError, ValueError, AttributeError):
            return None
        if not isinstance(segments, list):
            return None
        # Anchors belong to the other language; the new plan picks its own.
        return [{k: v for k, v in entry.items() if k != "host"} for entry in segments if isinstance(entry, dict)]

    def _save_plan(self) -> None:
        data = {"segments": self.plan}
        if self.bible:
            data["bible"] = {**self.bible, "sections": {str(k): v for k, v in (self.bible.get("sections") or {}).items()}}
        with open(os.path.join(self.task_dir, "edit-plan.json"), "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
            fp.write("\n")

    def _entry(self, index: int) -> dict:
        if index < len(self.plan):
            return self.plan[index]
        return {"expression": "", "backgrounds": [], "scenes": [], "beats": []}

    def background_shots(self, index: int) -> List[Tuple[float, str]]:
        """(start second, stock footage query) scenes for a segment, in order."""
        segment = self.segments[index]
        narration = self.narrations[index]
        duration = narration.frames / 30.0
        timed = []
        for position, background in enumerate(self._entry(index).get("backgrounds") or []):
            start = anchor_time(segment.text, background.get("at", ""), narration.sub_maker, narration.speech_seconds, narration.spans)
            if start is None:
                if position > 0:
                    continue
                start = 0.0
            timed.append((0.0 if position == 0 else max(0.0, start - 0.15), background["query"]))
        shots: List[Tuple[float, str]] = []
        for start, query in sorted(timed, key=lambda shot: shot[0]):
            if not shots:
                shots.append((0.0, query))
            elif start - shots[-1][0] >= MIN_BACKGROUND_SECONDS and duration - start >= MIN_BACKGROUND_SECONDS:
                shots.append((start, query))
        return shots

    def _prepare(self) -> None:
        """Time scenes and beats, fetch every picture, and plan the host over the whole video."""
        if self.host_plan:
            return
        lines: Dict[int, object] = {}
        reactions_by_segment: List[List[Tuple[float, str]]] = []
        covers: List[List[Tuple[float, float]]] = []
        if self.doodle and self._storyboarded and self.options.fill_gaps:
            self._fill_shot_gaps()
        for index, segment in enumerate(self.segments):
            narration = self.narrations[index]
            duration = narration.frames / 30.0
            entry = self._entry(index)
            pauses = host.find_pauses(narration.pcm)

            def locate(anchor, segment=segment, narration=narration):
                return anchor_time(segment.text, anchor, narration.sub_maker, narration.speech_seconds, narration.spans)

            window = self._subscribe_window(index, duration)
            blocked = [window] if window else []
            opener = self._opener(index, pauses)
            if opener is not None:
                blocked.append((opener.start, opener.end + 0.3))
            if self.doodle:
                # Drawn shots one after another, from the end of the opener to the cut.
                timed_scenes = scenes.time_shots(entry.get("shots") or [], locate, duration, opener.end if opener else 0.0)
            else:
                timed_scenes = scenes.time_scenes(entry.get("scenes") or [], locate, pauses, duration, blocked)
            if opener is not None:
                timed_scenes.insert(0, opener)
            self._scenes[index] = timed_scenes
            cover = [(scene.start, scene.end) for scene in timed_scenes]
            covers.append(cover)
            beats = [] if self.doodle else schedule_beats(entry.get("beats") or [], segment.text, narration, duration)
            if opener is not None:
                beats = [beat for beat in (_after_opener(b, opener.end) for b in beats) if beat]
            self._beats[index] = [(beat, "") for beat in (_before_scenes(b, cover) for b in beats) if beat]
            reactions = []
            for beat in entry.get("beats") or []:
                if beat.get("type") != "react":
                    continue
                time = locate(beat.get("at", ""))
                if time is None or time >= duration - 1.5:
                    continue
                if any(start - 0.3 <= time <= end for start, end in cover):
                    continue
                reactions.append((max(0.4, time), beat["expression"]))
            reactions_by_segment.append(reactions)
            lines[index] = segment.text

        self._fetch_pictures(lines)

        # Alternate sideways and vertical entrances across the video.
        count = 0
        infos = []
        for index, segment in enumerate(self.segments):
            narration = self.narrations[index]
            entry = self._entry(index)
            for scene in self._scenes[index]:
                if scene.type == "opener":
                    continue
                scene.vertical = count % 2 == 1
                count += 1
            prepared = []
            for beat, _ in self._beats[index]:
                if beat.kind == "images":
                    beat.members = [m for m in beat.members if m.picture]
                    if len(beat.members) == 1:
                        single = beat.members[0]
                        single.end = beat.end
                        beat = single
                    elif not beat.members:
                        continue
                if beat.kind == "image" and not beat.picture:
                    continue
                prepared.append((beat, beat.picture))
            self._beats[index] = prepared
            infos.append(
                host.SegmentInfo(
                    kind=segment.kind,
                    duration=narration.frames / 30.0,
                    expression=entry.get("expression", ""),
                    reactions=reactions_by_segment[index],
                    pictures=[b.start for b, _ in prepared if b.kind == "image"],
                    pauses=host.find_pauses(narration.pcm),
                    mode=entry.get("host", ""),
                    blocked=[(b.start, b.end) for b, _ in prepared if b.kind == "images"],
                )
            )
        names = sorted(self.poses)
        mode = self.options.host if self.options.host in host.HOST_MODES else "auto"
        if self.doodle and mode != "always":
            mode = "none"  # the otter lives inside the drawings
        seed = fx.safe_seed("".join(s.chapter for s in self.segments)) % len(host.ITEM_PATTERN)
        presence = self.options.host_presence if self.options.host_presence in host.HOST_PRESENCES else "low"
        self.host_plan = host.plan_host(infos, names, mode, seed, presence)
        for planned, cover in zip(self.host_plan, covers):
            planned.windows = _avoid_cover(planned.windows, cover)
        if self.plan and names:
            for entry, planned in zip(self.plan, self.host_plan):
                entry["host"] = planned.mode
            self._save_plan()

    def _opener(self, index: int, pauses: Optional[List[float]] = None) -> Optional[scenes.Scene]:
        segment = self.segments[index]
        if not (self.options.openers and segment.kind == "item"):
            return None
        narration = self.narrations[index]
        if pauses is None:
            pauses = host.find_pauses(narration.pcm)
        return scenes.opener_scene(
            segment.name or segment.chapter, segment.number, self._entry(index).get("opener"), pauses, narration.frames / 30.0
        )

    def _locator(self, index: int):
        segment, narration = self.segments[index], self.narrations[index]
        return lambda anchor: anchor_time(segment.text, anchor, narration.sub_maker, narration.speech_seconds, narration.spans)

    def _fill_shot_gaps(self) -> None:
        """A second, short LLM pass: more shots where one picture would stay too long."""
        limit = max(4.5, self.options.shot_seconds * HOLD_FACTOR)
        gaps = []
        for index, segment in enumerate(self.segments):
            narration = self.narrations[index]
            duration = narration.frames / 30.0
            opener = self._opener(index)
            begin = opener.end if opener else 0.0
            shots = scenes.time_shots(self._entry(index).get("shots") or [], self._locator(index), duration, begin)
            timed_words = word_times(segment.text, narration.sub_maker, narration.speech_seconds, narration.spans)
            for start, end in scenes.long_holds(shots, duration, limit, begin):
                said = " ".join(word for word, time in timed_words if start + 0.2 <= time < end - 0.4)
                if len(said.split()) < 6:
                    continue
                showing = next((s for s in reversed(shots) if s.start <= start + 0.01), None)
                what = ""
                if showing is not None:
                    center = showing.items[-1] if showing.type == "animation" and showing.items else (
                        showing.center or (showing.items[0] if showing.items else None))
                    what = f"{showing.type}: {(center.draw or center.label) if center else showing.text}".strip()
                gaps.append({
                    "index": index, "text": said, "showing": what, "seconds": round(end - start, 1),
                    "shots": max(1, int(round((end - start) / max(1.5, self.options.shot_seconds))) - 1),
                })
        if not gaps:
            return
        try:
            extra = llm.generate_storyboard_gaps(
                gaps[:60], sorted(self.poses), self.options.language, clips=self.options.clips,
                memes=self.options.memes in ("otter", "folder"),
            )
        except Exception as exc:
            logger.warning(f"could not fill the long shots: {type(exc).__name__}: {exc}")
            return
        added = 0
        for index, shots in extra.items():
            if 0 <= index < len(self.plan):
                self.plan[index].setdefault("shots", [])
                self.plan[index]["shots"] += shots
                added += len(shots)
        if added:
            logger.info(f"{added} shots added where a picture stayed too long ({len(gaps)} stretches)")
            self._save_plan()

    def _fetch_pictures(self, texts: Dict[int, str]) -> None:
        """Find every picture of the video at once (searches and checks run in parallel).

        The illustrations go first, so the drawing budget is spent on them; an
        illustration that continues the one before it is drawn from it, so
        each chain of frames is drawn in order.
        """
        jobs = []
        chains: List[List[Tuple[scenes.Scene, str]]] = []
        if self.doodle:
            self._prepare_characters()  # every drawing of a recurring person follows their character sheet
        for index, beats in self._beats.items():
            text = texts[index]
            for beat, _ in beats:
                for member in beat.members if beat.kind == "images" else [beat] if beat.kind == "image" else []:
                    jobs.append(lambda m=member, t=text: setattr(m, "picture", self._beat_picture(m, sentence_at(t, m.at))))
        for index in sorted(self._scenes):
            text = texts[index]
            for scene in self._scenes[index]:
                if scene.reprise:
                    continue  # shows the picture of the shot it repeats
                if scene.type == "figure":
                    jobs.append(lambda sc=scene, t=text: self._figure_picture(sc, t))
                elif scene.type == "annotate":
                    jobs.append(lambda sc=scene, t=text: self._annotate_picture(sc, t))
                    continue
                elif scene.type == "opener":
                    jobs.append(lambda sc=scene, t=text: self._opener_picture(sc, t))
                    continue
                elif scene.type in ("illustration", "animation"):
                    if scene.follows and chains:
                        chains[-1].append((scene, text))
                    else:
                        chains.append([(scene, text)])
                    continue
                elif scene.type == "archive":
                    jobs.append(lambda sc=scene, t=text: self._archive_picture(sc, t))
                    continue
                elif scene.type == "clip":
                    jobs.append(lambda sc=scene, t=text: self._clip_media(sc, t))
                    continue
                elif scene.type == "meme":
                    jobs.append(lambda sc=scene: self._meme_picture(sc))
                    continue
                elif scene.type == "story" and (self.options.illustrations == "ai" or self.doodle) and self._gemini:
                    jobs.append(lambda sc=scene: self._story_pictures(sc))
                # Icons only ever stand in for the small pictures of a composition of several elements.
                small = scene.type in scenes.ITEM_SCENES and len(scene.items) > 1
                for item in scene.items + ([scene.center] if scene.center and scene.type != "figure" else []):
                    if self.doodle and (item.draw or item.icon or item.pose or item.query or item.image):
                        jobs.append(lambda i=item, t=text, s=small and item is not scene.center: self._doodle_picture(i, t, icon=s))
                    elif item.query or item.icon or item.draw:
                        jobs.append(lambda i=item, t=text: self._item_picture(i, sentence_at(t, i.at)))
        first = [lambda c=chain: self._illustration_chain(c) for chain in chains]
        jobs = first + jobs
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(job) for job in jobs]:
                try:
                    future.result()
                except Exception as exc:
                    logger.warning(f"a picture could not be prepared: {type(exc).__name__}: {exc}")
        if self._unpictured:
            self._rescue(texts)
        # Full-screen pictures that were not found leave the footage alone.
        dropped = 0
        for index, timed in self._scenes.items():
            kept = [sc for sc in timed if self._has_picture(sc) and (not self.doodle or self._complete(sc))]
            dropped += len(timed) - len(kept)
            if self.doodle and kept and timed:
                # Shots follow each other: the one before a dropped shot stays longer.
                for shot, following in zip(kept, kept[1:]):
                    if shot.type != "opener":
                        shot.end = following.start
                if kept[-1].type != "opener":
                    kept[-1].end = timed[-1].end
                self._crossfade(kept)
            self._scenes[index] = kept
        if self.doodle:
            self._stats["shots_dropped"] += dropped
            self._bring_to_life()
            self._report_drawings(dropped)

    def _has_picture(self, scene: scenes.Scene) -> bool:
        if scene.type == "animation":
            return any(self._item_pictures.get(id(frame)) is not None for frame in scene.items)
        if scene.type in ("figure", "annotate", "illustration", "archive"):
            return self._item_pictures.get(id(scene.center)) is not None
        if scene.type in ("clip", "meme"):
            return bool(scene.media) or (scene.type == "meme" and self._item_pictures.get(id(scene.center)) is not None)
        return True

    @staticmethod
    def _crossfade(shots: List[scenes.Scene]) -> None:
        """A shot that covers the whole frame dissolves in over the one before it."""
        for shot in shots:
            shot.overlap = shot.fade_in = shot.linger = 0.0
        for shot, following in zip(shots, shots[1:]):
            if shot.type == "opener":
                continue
            fade = min(scenes.CROSSFADE_SECONDS, (following.end - following.start) / 4)
            if following.type in scenes.FULL_FRAME_SHOTS:
                shot.overlap, following.fade_in = fade, fade
            elif shot.type in scenes.FULL_FRAME_SHOTS:
                # Back to the drawn canvas: the picture fades out while the first drawing appears.
                shot.linger = fade

    def _report_drawings(self, dropped: int) -> None:
        """Warnings that say why drawings are missing, so a broken image model is noticed."""
        failure = gemini_media.imagen_failure()
        if failure:
            self.warnings.append(f"Imagen failed ({failure}); the drawings were made with {gemini_media.SEQUENCE_DEFAULT_MODEL}")
        if self._failed_drawings:
            reason = gemini_media.last_error()
            self.warnings.append(
                f"{self._failed_drawings} drawings failed" + (f" (last error: {reason})" if reason else "")
                + "; real pictures or the previous shot were used instead"
            )
        if self._gemini and self._drawings >= self.options.max_drawings > 0:
            self.warnings.append(
                f"the drawing budget ({self.options.max_drawings}) ran out; raise --max-drawings for more drawings"
            )
        if dropped:
            logger.info(f"{dropped} shots without a picture were left out (the shot before them stays longer)")

    # -- assets -------------------------------------------------------------

    def _prepare_subscribe(self) -> Optional[dict]:
        if self.options.subscribe == "none":
            return None
        media = fx.find_asset(self.options.assets_dir, fx.SUBSCRIBE_NAMES, fx.SUBSCRIBE_EXTENSIONS)
        sound = fx.find_asset(self.options.assets_dir, fx.SUBSCRIBE_NAMES, fx.AUDIO_EXTENSIONS)
        if media:
            extension = os.path.splitext(media)[1].lower()
            if extension in (".png", ".webp"):
                return {"mode": "still", "source": media, "click": 0.4, "sound": sound}
            duration = fx.media_duration(utils.get_ffmpeg_binary(), media) or SUBSCRIBE_SECONDS
            return {"mode": "media", "source": media, "click": 0.3, "sound": sound, "duration": duration}
        pattern, click = fx.render_subscribe_frames(
            self.theme, os.path.join(self.work_dir, "subscribe"), self.options.language, 30, SUBSCRIBE_SECONDS
        )
        return {"mode": "frames", "source": pattern, "click": click, "sound": sound}

    def renderer(self) -> host.HostRenderer:
        if self._renderer is None:
            height = int(self.theme.height * HOST_HEIGHT[self.theme.portrait])
            self._renderer = host.HostRenderer(
                self.poses, height, self.work_dir, talking=self.options.lip_sync
            )
        return self._renderer

    def _web_picture(self, query: str, look: str, line: str, purpose: str) -> str:
        """The best web picture for ``query``; Gemini picks among several when it can."""
        save_dir = os.path.join(self.work_dir, "pictures")
        check = self.options.picture_check and self._gemini
        candidates = web_images.find_candidates(
            query, save_dir, kind=look, exclude_urls=self._used_urls, limit=4 if check else 1
        )
        if not candidates:
            return ""
        choice = 0
        if check:
            verdict = gemini_media.choose_picture(
                [c.path for c in candidates], line or query, query, self.options.language, purpose=purpose
            )
            choice = 0 if verdict is None else verdict
        if choice < 0:
            logger.info(f"pictures for {query!r} did not pass the check")
            return ""
        found = candidates[choice]
        self._used_urls.add(found.url)
        self.credits.append(found.credit())
        return found.path

    def _beat_picture(self, beat: _Beat, line: str = "") -> str:
        """A picture for an image beat: AI, or web pictures checked by Gemini, or the icon."""
        save_dir = os.path.join(self.work_dir, "pictures")
        if self.options.beats == "ai":
            if self._gemini:
                path = gemini_media.illustrate(beat.query)
                if path:
                    self.credits.append("Illustrations: generated with Google Gemini")
                    return path
            elif material.is_openai_image_enabled():
                items = material.generate_images_openai(
                    search_term=beat.query, minimum_duration=1, save_dir=save_dir
                )
                if items:
                    self.credits.append(f"{beat.query} — AI-generated illustration")
                    return items[0].url
        path = self._web_picture(beat.query, beat.look, line, "beat") if beat.query else ""
        if path:
            return path
        if beat.icon:
            path = icons.fetch(beat.icon)
            if path:
                self.credits.append(icons.CREDIT)
                return path
        self.warnings.append(f"no picture found for {beat.query!r}")
        return ""

    def _item_picture(self, item: scenes.SceneItem, line: str = "") -> None:
        """Scene element picture: AI doodle, a checked web picture, or the icon."""
        image = None
        if self.options.illustrations == "ai" and self._gemini and item.draw:
            path = gemini_media.illustrate(item.draw)
            if path:
                image = scenes.prepare_picture(path)
                self.credits.append("Illustrations: generated with Google Gemini")
        if image is None and item.query and self.options.picture_check and self._gemini:
            # Real pictures only when Gemini can confirm they fit; icons otherwise.
            path = self._web_picture(item.query, "diagram", line, "scene")
            if path:
                image = scenes.prepare_picture(path)
        if image is None:
            image = self._icon_picture(item)
        self._item_pictures.setdefault(id(item), image)

    def _icon_picture(self, item: scenes.SceneItem):
        """The item's icon; when the video already showed that icon, a different one that fits."""
        key = (id(item), item.icon, item.draw)
        if key not in self._scene_pictures:
            image = None
            if self.options.picture_check and self._gemini:
                image = self._checked_icon(item)
            else:
                for query in self._icon_queries(item):
                    path = icons.fetch(query)
                    if path:
                        image = scenes.prepare_picture(path, trust_alpha=True)
                        self.credits.append(icons.CREDIT)
                        break
            self._scene_pictures[key] = image
        return self._scene_pictures[key]

    def _checked_icon(self, item: scenes.SceneItem):
        """The icon Gemini confirms depicts ``item`` among a few candidates; None when none does
        (a missing icon is better than a wrong one: the label then stands alone, bigger)."""
        wanted = [q for q in (item.icon, item.draw) if q]
        keywords = " ".join(q for q in (item.draw, item.query, item.label) if q)
        with self._icon_lock:
            used = set(self._used_icons)
        names = [q for q in wanted if q not in used]
        if keywords:
            try:
                names += [e for e in icons.alternatives(keywords, exclude=used, limit=6) if e not in names]
            except Exception as exc:
                logger.debug(f"no other icon for {keywords!r}: {exc}")
        names += [q for q in wanted if q not in names]
        candidates: List[Tuple[str, str]] = []
        for name in names:
            path = icons.fetch(name)
            if path and path not in [p for _, p in candidates]:
                candidates.append((name, path))
            if len(candidates) >= 4:
                break
        if not candidates:
            return None
        description = item.draw or item.label or item.icon
        context = " — ".join(t for t in (item.label, item.draw) if t) or description
        verdict = gemini_media.choose_picture(
            [path for _, path in candidates], context, description, self.options.language, purpose="icon"
        )
        choice = 0 if verdict is None else verdict
        if choice < 0:
            logger.info(f"no icon fits {description!r}; the label stands alone")
            return None
        name, path = candidates[choice]
        with self._icon_lock:
            self._used_icons.add(name)
        self.credits.append(icons.CREDIT)
        return scenes.prepare_picture(path, trust_alpha=True)

    def _icon_queries(self, item: scenes.SceneItem) -> List[str]:
        """Icons to try for ``item``, the ones the video has not shown yet first."""
        wanted = [q for q in (item.icon, item.draw) if q]
        with self._icon_lock:
            fresh = [q for q in wanted if q not in self._used_icons]
            if not fresh and wanted:
                # Already used: look for another icon of the same thing.
                keywords = " ".join(q for q in (item.draw, item.query) if q) or item.icon
                try:
                    fresh = [e for e in icons.alternatives(keywords, exclude=self._used_icons) if e not in self._used_icons]
                except Exception as exc:
                    logger.debug(f"no other icon for {keywords!r}: {exc}")
            queries = fresh + [q for q in wanted if q not in fresh]
            if queries:
                self._used_icons.add(queries[0])
        return queries

    def _figure_picture(self, scene: scenes.Scene, text: str) -> None:
        """A full-screen picture, checked by Gemini, which also says how long to show it."""
        if not (self.options.picture_check and self._gemini):
            logger.info("full-screen pictures need Gemini to check them; skipped")
            return
        save_dir = os.path.join(self.work_dir, "pictures")
        candidates = []
        for query in dict.fromkeys(q for q in (scene.query_local, scene.query) if q):
            candidates += web_images.find_candidates(query, save_dir, kind=scene.look, exclude_urls=self._used_urls, limit=3)
        candidates = candidates[:5]
        if not candidates:
            return
        line = sentence_at(text, scene.center.at if scene.center and scene.center.at else "")
        verdict = gemini_media.choose_figure([c.path for c in candidates], line, scene.query, self.options.language)
        choice, seconds = verdict if verdict is not None else (0, 0.0)
        if choice < 0:
            logger.info(f"no full-screen picture for {scene.query!r} passed the check")
            return
        found = candidates[choice]
        self._used_urls.add(found.url)
        self.credits.append(found.credit())
        image = scenes.prepare_picture(found.path, allow_cutout=scene.look != "photo")
        if image is not None and seconds:
            scene.end = min(scene.end, scene.start + seconds + 0.3)
            scene.exit = True
        if scene.center is None:
            scene.center = scenes.SceneItem()
        self._item_pictures[id(scene.center)] = image

    def _annotate_picture(self, scene: scenes.Scene, text: str) -> None:
        """A picture of the structure Gemini confirms shows its parts, and where each part is."""
        if not (self.options.picture_check and self._gemini):
            logger.info("annotated pictures need Gemini to check them; skipped")
            return
        save_dir = os.path.join(self.work_dir, "pictures")
        candidates = []
        for query in dict.fromkeys(q for q in (scene.query_local, scene.query) if q):
            candidates += web_images.find_candidates(query, save_dir, kind=scene.look, exclude_urls=self._used_urls, limit=3)
        candidates = candidates[:5]
        if not candidates:
            return
        labels = [item.label for item in scene.items]
        line = sentence_at(text, scene.center.at if scene.center else "")
        wanted = f"{scene.query} (showing: {', '.join(labels)})" if labels else scene.query
        choice = gemini_media.choose_picture([c.path for c in candidates], line, wanted, self.options.language, purpose="annotate")
        if choice is None or choice < 0:
            logger.info(f"no picture to annotate for {scene.query!r} passed the check")
            return
        found = candidates[choice]
        image = scenes.prepare_picture(found.path, allow_cutout=False)
        if image is None:
            return
        for item, point in zip(scene.items, gemini_media.locate_parts(found.path, labels)):
            item.point = point
        self._used_urls.add(found.url)
        self.credits.append(found.credit())
        self._item_pictures[id(scene.center)] = image

    def _mascot(self) -> str:
        """A picture of the channel's character, as the reference for drawings of it."""
        name = _pick(self.poses, ("explicando", "feliz", "neutral"))
        return self.poses[name].idle if name else ""

    def _pinned(self, item: scenes.SceneItem, framed: bool = True):
        """The picture a review pinned to this element (used as it is), or None."""
        if not item.image:
            return None
        path = item.image
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.abspath(self.options.plan_file)) if self.options.plan_file else self.task_dir
            path = os.path.join(base, path)
        if not os.path.isfile(path):
            logger.warning(f"the pinned picture {item.image!r} is missing; it is made again")
            return None
        picture = scenes.prepare_picture(path, allow_cutout=not framed)
        if picture is not None:
            picture.info["source"] = path
            if framed:
                picture.info["framed"] = True
            with self._drawing_lock:
                self._stats["pinned"] += 1
        return picture

    def _prepare_characters(self) -> None:
        """Draw the character sheet of every recurring person the shots show (from a real portrait when one is found)."""
        if not (self._gemini and self.bible.get("characters")):
            return
        wanted = set()
        for timed in self._scenes.values():
            for scene in timed:
                for item in scene.items + ([scene.center] if scene.center else []):
                    wanted.update(item.characters)
        people = [c for c in self.bible["characters"] if c.get("id") in wanted and c.get("id") not in self._sheets]
        if not people:
            return
        with ThreadPoolExecutor(max_workers=3) as pool:
            for person, path in zip(people, pool.map(self._character_sheet, people)):
                if path:
                    self._sheets[person["id"]] = (person.get("name") or person["id"], path)
        logger.info(f"{len(self._sheets)} character sheets ready for {len(people)} recurring people")

    def _character_sheet(self, person: dict) -> str:
        """The character sheet file of a person of the visual bible ("" when it could not be drawn)."""
        with self._drawing_lock:
            if self._drawings >= max(0, self.options.max_drawings):
                return ""
            self._drawings += 1
        portrait = ""
        if person.get("portrait") and self.options.picture_check:
            save_dir = os.path.join(self.work_dir, "portraits")
            try:
                candidates = web_images.find_candidates(person["portrait"], save_dir, kind="archive", limit=4)
            except Exception as exc:
                logger.debug(f"no portrait of {person.get('name')!r}: {exc}")
                candidates = []
            if candidates:
                choice = gemini_media.choose_picture(
                    [c.path for c in candidates], person.get("name", ""), person["portrait"], self.options.language,
                    purpose="portrait",
                )
                if choice is not None and choice >= 0:
                    portrait = candidates[choice].path
                    self.credits.append(candidates[choice].credit())
        path = gemini_media.draw_character(
            person.get("name", ""), person.get("look", ""), portrait, style=self.options.drawing_style,
            art=self.bible.get("style", ""), quality=self.options.image_quality,
        )
        if path:
            with self._drawing_lock:
                self._stats["sheets"] += 1
        return path

    def _sheets_for(self, ids: List[str]) -> List[Tuple[str, str]]:
        return [self._sheets[key] for key in ids if key in self._sheets]

    def _drawing(
        self, description: str, mascot: bool = False, scene: bool = False, previous: str = "",
        characters: Optional[List[str]] = None, context: str = "",
    ):
        """A Gemini drawing for the doodle look, within the drawing budget; None otherwise.

        ``previous`` is the picture file of the scene this one continues;
        ``characters`` the ids of the recurring people it shows (drawn from
        their character sheets); ``context`` the sentence said while it is on
        screen. With the picture check on, Gemini audits the drawing against
        that sentence and a rejected one is drawn once more with its fix.
        """
        if not (self._gemini and description):
            return None
        with self._drawing_lock:
            if self._drawings >= max(0, self.options.max_drawings):
                self._stats["budget"] += 1
                return None
            self._drawings += 1
            # The same description twice would give the very same drawing.
            key = " ".join(description.lower().split())
            seen = self._drawn.get(key, 0)
            self._drawn[key] = seen + 1
        asked = description
        if seen:
            asked = f"{description}, a different moment, pose and angle from before (variation {seen + 1})"
        sheets = self._sheets_for(characters or [])
        options = {
            "scene": scene, "mascot": self._mascot() if mascot else "", "characters": sheets,
            "art": self.bible.get("style", ""), "quality": self.options.image_quality,
        }
        if previous:
            options["previous"] = previous
        if self.options.drawing_style != "cartoon":
            options["style"] = self.options.drawing_style
        path = gemini_media.draw(asked, **options)
        if path and self.options.picture_check:
            verdict, fix = gemini_media.audit_drawing(
                path, description, mascot=mascot, context=context, characters=[name for name, _ in sheets]
            )
            if verdict is False:
                with self._drawing_lock:
                    self._stats["rejected"] += 1
                    again = self._drawings < max(0, self.options.max_drawings)
                    if again:
                        self._drawings += 1
                if again:
                    retry = f"{asked}. Make sure: {fix}" if fix else f"{asked}, drawn again clearly and simply"
                    redrawn = gemini_media.draw(retry, **options)
                    path = redrawn or path
        if not path:
            with self._drawing_lock:
                self._failed_drawings += 1
            return None
        with self._drawing_lock:
            self._stats["drawn"] += 1
        self.credits.append("Illustrations: drawn with Google Gemini")
        picture = scenes.prepare_picture(path, allow_cutout=not scene)
        if picture is not None:
            picture.info["source"] = path
        return picture

    def _doodle_picture(self, item: scenes.SceneItem, text: str = "", icon: bool = False) -> None:
        """A shot element: a pinned picture, the otter's own pose, a drawing, or a real picture Gemini approves.

        ``icon``: a small element among others, which may fall back to a checked
        icon (never a big picture alone on screen).
        """
        image = self._pinned(item, framed=False)
        if image is None and item.pose and item.pose in self.poses:
            image = scenes.prepare_picture(self.poses[item.pose].idle, trust_alpha=True)
        if image is None and item.draw:
            image = self._drawing(item.draw, mascot=item.otter, characters=item.characters,
                                  context=sentence_at(text, item.at) if text else "")
        if image is None:
            image = self._real_picture(item, text)
        if image is None and icon:
            image = self._icon_picture(item)
        self._item_pictures[id(item)] = image

    def _real_picture(self, item: scenes.SceneItem, text: str = "", look: str = "diagram", purpose: str = "scene"):
        """A real picture of the element (a cut-out PNG when possible), only when Gemini confirms it fits."""
        query = item.query or " ".join((item.draw or "").split()[:6])
        if not (query and self.options.picture_check and self._gemini):
            return None
        path = self._web_picture(query, look, sentence_at(text, item.at) if text else query, purpose)
        image = scenes.prepare_picture(path, allow_cutout=look != "photo") if path else None
        if image is not None:
            with self._drawing_lock:
                self._stats["real"] += 1
        return image

    def _illustration_picture(self, scene: scenes.Scene, previous: str = "", text: str = ""):
        """A whole drawn scene (16:9) for a story moment; drawn from ``previous`` when it continues it."""
        item = scene.center or scenes.SceneItem()
        scene.center = item
        image = self._pinned(item)
        if image is None:
            image = self._drawing(
                item.draw, mascot=item.otter and not previous, scene=True, previous=previous,
                characters=item.characters, context=sentence_at(text, item.at) if text else "",
            )
        if image is not None:
            image.info["framed"] = True
            self._item_pictures[id(item)] = image
        return image

    def _animation_frames(self, scene: scenes.Scene, previous: str = "", text: str = "") -> str:
        """The drawings of an animation, each redrawn from the one before it; the last drawing's file."""
        last = previous
        anchor = scene.center.at if scene.center else ""
        for number, frame in enumerate(scene.items):
            follow = last if number > 0 or scene.follows else ""
            image = self._pinned(frame)
            if image is None:
                context = sentence_at(text, frame.at or anchor) if text else ""
                image = self._drawing(frame.draw, mascot=frame.otter and not follow, scene=True, previous=follow,
                                      characters=frame.characters, context=context)
            if image is None:
                continue  # a missing frame is skipped; the next one follows the last good one
            image.info["framed"] = True
            self._item_pictures[id(frame)] = image
            last = image.info.get("source", "") or last
        return last if any(id(frame) in self._item_pictures for frame in scene.items) else ""

    def _illustration_chain(self, chain: List[Tuple[scenes.Scene, str]]) -> None:
        """Illustrations and animations drawn in order, a continuing one redrawn from the picture before it."""
        previous = ""
        for scene, text in chain:
            if scene.type == "animation":
                drawn = self._animation_frames(scene, previous if scene.follows else "", text)
            else:
                image = self._illustration_picture(scene, previous if scene.follows else "", text)
                drawn = image.info.get("source", "") if image is not None else ""
            if not drawn:
                with self._drawing_lock:
                    self._unpictured.append(scene)
            previous = drawn or previous

    def _rescue(self, texts: Dict[int, str]) -> None:
        """Pictures that are still missing: one more drawing try (the model may have been busy), then a
        real photo of the moment that Gemini approves; a shot without either is left out."""
        where = {id(sc): index for index, timed in self._scenes.items() for sc in timed}
        for scene in list(self._unpictured):
            text = texts.get(where.get(id(scene), -1), "")
            if scene.type == "animation":
                if self._animation_frames(scene, text=text):
                    continue
            elif self._illustration_picture(scene, text=text) is not None:
                continue
            center = scene.center or scenes.SceneItem()
            if not center.query and scene.items:
                center.query = scene.items[0].query or " ".join(scene.items[0].draw.split()[:6])
            elif not center.query:
                center.query = " ".join(center.draw.split()[:6])
            photo = self._real_picture(center, text, look="photo", purpose="beat")
            if photo is not None:
                photo.info["framed"] = True
                scene.type, scene.center, scene.items = "illustration", center, []
                self._item_pictures[id(center)] = photo
                with self._drawing_lock:
                    self._stats["photo"] += 1
        self._unpictured = []

    def _archive_picture(self, scene: scenes.Scene, text: str = "") -> None:
        """A real historical picture (Wikimedia Commons) that Gemini confirms is authentic and shows exactly this
        moment; the moment is drawn instead when none is."""
        item = scene.center or scenes.SceneItem()
        scene.center = item
        image = self._pinned(item)
        if image is None and self.options.picture_check and self._gemini:
            save_dir = os.path.join(self.work_dir, "archive")
            shorter = " ".join(scene.query.split()[:4])
            candidates: list = []
            for query in dict.fromkeys(q for q in (scene.query, scene.query_local, shorter) if q):
                with self._icon_lock:
                    used = set(self._used_urls) | set(item.avoid)
                try:
                    found = web_images.find_candidates(query, save_dir, kind="archive", exclude_urls=used, limit=4)
                except Exception as exc:
                    logger.debug(f"archive search failed for {query!r}: {exc}")
                    found = []
                candidates += [c for c in found if c.url not in {k.url for k in candidates}]
                if len(candidates) >= 4:
                    break
            candidates = candidates[:6]
            if candidates:
                line = sentence_at(text, item.at) if text else scene.query
                wanted = f"{scene.query} ({scene.text})" if scene.text else scene.query
                choice = gemini_media.choose_picture(
                    [c.path for c in candidates], line, wanted, self.options.language, purpose="archive"
                )
                if choice is not None and choice >= 0:
                    chosen = candidates[choice]
                    with self._icon_lock:
                        fresh = chosen.url not in self._used_urls
                        self._used_urls.add(chosen.url)
                    image = scenes.prepare_picture(chosen.path, allow_cutout=False) if fresh else None
                    if image is not None:
                        image.info["framed"] = True
                        image.info["archive"] = True
                        image.info["url"] = chosen.url
                        self.credits.append(chosen.credit())
                        logger.info(f"archive picture: {scene.query!r}")
                        with self._drawing_lock:
                            self._stats["archive"] += 1
                else:
                    logger.info(f"no authentic picture of {scene.query!r} passed the check")
        if image is not None:
            self._item_pictures[id(item)] = image
            return
        if item.draw:
            # No real picture fits: the moment is drawn instead (never a wrong or staged photo).
            scene.type = "illustration"
            if self._illustration_picture(scene, text=text) is not None:
                with self._drawing_lock:
                    self._stats["archive_drawn"] += 1

    def _bring_to_life(self) -> None:
        """The longest illustrations that have a "motion" become short Veo videos (``ai_videos`` at most)."""
        if self.options.ai_videos <= 0 or not self._gemini:
            return
        chosen = []
        for timed in self._scenes.values():
            for scene in timed:
                item = scene.center
                if scene.type != "illustration" or scene.reprise or scene.media or item is None or not item.motion:
                    continue
                picture = self._item_pictures.get(id(item))
                source = picture.info.get("source", "") if picture is not None else ""
                if source and not picture.info.get("archive") and scene.end - scene.start >= 3.5:
                    chosen.append((scene.end - scene.start, scene, source))
        chosen.sort(key=lambda entry: -entry[0])

        def make(entry) -> None:
            seconds, scene, source = entry
            path = gemini_media.animate(source, scene.center.motion, seconds, self.options.image_quality)
            if path:
                scene.media = path
                with self._drawing_lock:
                    self._stats["videos"] += 1

        with ThreadPoolExecutor(max_workers=gemini_media.VIDEO_SLOTS) as pool:
            list(pool.map(make, chosen[: self.options.ai_videos]))
        if self._stats["videos"]:
            self.credits.append("Animated shots: generated with Google Veo")
            logger.info(f"{self._stats['videos']} illustrations brought to life with Veo")

    def _complete(self, scene: scenes.Scene) -> bool:
        """Every composition shows pictures: elements without one are left out, a lone label gets the
        otter; False when the shot has nothing left to show."""
        if scene.type in ("illustration", "figure", "annotate", "clip", "meme", "animation", "opener", "archive") or scene.reprise:
            return True
        picture = self._item_pictures.get
        if scene.type in ("single", "definition", "stat", "zoom"):
            if scene.center is not None and picture(id(scene.center)) is None:
                otter = self._pose_picture("explicando")
                if otter is None:
                    return False
                self._item_pictures[id(scene.center)] = otter
                with self._drawing_lock:
                    self._stats["otter"] += 1
            return True
        if scene.type == "speech" and scene.items:
            if picture(id(scene.items[0])) is None:
                otter = self._pose_picture("explicando")
                if otter is None:
                    return False
                self._item_pictures[id(scene.items[0])] = otter
            scene.items = [scene.items[0]] + [item for item in scene.items[1:] if picture(id(item)) is not None]
            return True
        if scene.type in scenes.ITEM_SCENES and scene.type != "story":
            kept = [item for item in scene.items if picture(id(item)) is not None]
            if len(kept) < len(scene.items):
                with self._drawing_lock:
                    self._stats["items_dropped"] += len(scene.items) - len(kept)
            if not kept or (scene.type == "compare" and len(kept) < 2):
                return False
            if kept[0] is not scene.items[0]:
                kept[0].time = min(kept[0].time, scene.items[0].time)
            scene.items = kept
        return True

    def _stand_in(self, scene: scenes.Scene) -> Optional[scenes.Scene]:
        """The otter with the shot's words, for a shot that could not be drawn (never an empty screen)."""
        otter = self._pose_picture("explicando")
        if otter is None:
            return None
        center = scene.center or scenes.SceneItem()
        item = scenes.SceneItem(label=center.label or (scene.items[0].label if scene.items else ""))
        self._item_pictures[id(item)] = otter
        return scenes.Scene(
            "single", scene.start, scene.end, text=scene.text if scene.type in ("single", "statement") else "",
            center=item, overlap=scene.overlap, linger=scene.linger,
        )

    def _pose_picture(self, preferred: str):
        name = preferred if preferred in self.poses else _pick(self.poses, ("explicando", "feliz", "neutral"))
        return scenes.prepare_picture(self.poses[name].idle, trust_alpha=True) if name else None

    def _clip_media(self, scene: scenes.Scene, text: str = "") -> None:
        """A stock video for a clip shot (Pexels, then Pixabay); its drawing when none is found."""
        item = scene.center or scenes.SceneItem()
        scene.center = item
        if scene.media and os.path.isfile(scene.media):
            return  # kept in a review
        scene.media = ""
        query = item.query
        aspect = VideoAspect.portrait if self.theme.portrait else VideoAspect.landscape
        seconds = max(2, int(scene.end - scene.start + 1))
        candidates = []
        if query:
            for search in (material.search_videos_pexels, material.search_videos_pixabay):
                try:
                    found = search(query, seconds, aspect) or []
                except Exception as exc:
                    logger.debug(f"no {getattr(search, '__name__', 'stock video')} for {query!r}: {exc}")
                    found = []
                with self._icon_lock:
                    candidates += [info for info in found if info.url not in self._used_urls]
                if len(candidates) >= 3:
                    break
        check = self.options.picture_check and self._gemini
        line = sentence_at(text, item.at) if text else query
        for info in candidates[:3]:
            with self._icon_lock:
                if info.url in self._used_urls:
                    continue
                self._used_urls.add(info.url)
            try:
                path = material.save_video(info.url, save_dir=os.path.join(self.work_dir, "clips"))
            except Exception as exc:
                logger.warning(f"clip for {query!r} could not be downloaded: {exc}")
                continue
            if not path:
                continue
            frame = os.path.join(self.work_dir, "clips", f"{os.path.basename(path)}.jpg")
            if fx.extract_frame(utils.get_ffmpeg_binary(), path, frame, 1.0):
                if fx.brightness(frame) < DARK_CLIP:
                    logger.info(f"the clip for {query!r} is too dark")
                    continue
                if check:
                    verdict = gemini_media.choose_picture([frame], line, query, self.options.language, purpose="clip")
                    if verdict is not None and verdict < 0:
                        logger.info(f"the clip for {query!r} did not pass the check")
                        continue
            scene.media = path
            self.credits.append(_clip_credit(info))
            return
        if item.draw:
            # No video fits: the moment is drawn instead.
            logger.info(f"no clip for {query!r}; drawing it instead")
            scene.type = "illustration"
            self._illustration_picture(scene)

    def _meme_files(self) -> Dict[str, List[str]]:
        with self._icon_lock:
            if self._memes is None:
                folder = self.options.memes_dir or utils.resource_dir("memes")
                self._memes = meme_library(folder) if self.options.memes == "folder" else {}
            return self._memes

    def _meme_picture(self, scene: scenes.Scene) -> None:
        """The reaction of a meme shot: a file of the memes folder, the otter drawn reacting, or its pose."""
        item = scene.center or scenes.SceneItem()
        scene.center = item
        mood = scene.mood if scene.mood in llm.MEME_MOODS else "shock"
        files = self._meme_files().get(mood, [])
        with self._icon_lock:
            fresh = [f for f in files if f not in self._used_memes] or files
            chosen = fresh[fx.safe_seed(f"{scene.start}{item.at}") % len(fresh)] if fresh else ""
            if chosen:
                self._used_memes.add(chosen)
        if chosen:
            if chosen.lower().endswith(scenes.VIDEO_EXTENSIONS):
                scene.media = chosen
                return
            image = scenes.prepare_picture(chosen, allow_cutout=False)
            if image is not None:
                image.info["framed"] = True
                self._item_pictures[id(item)] = image
                return
        image = self._drawing(item.draw, mascot=True) if item.draw else None
        if image is None:
            pose = MEME_POSES.get(mood, "sorprendido")
            if pose not in self.poses:
                pose = _pick(self.poses, ("sorprendido", "feliz", "explicando"))
            if pose:
                image = scenes.prepare_picture(self.poses[pose].idle, trust_alpha=True)
        self._item_pictures[id(item)] = image

    def _opener_picture(self, scene: scenes.Scene, text: str) -> None:
        """The section opener's picture, strictly about its title.

        A web picture Gemini confirms shows exactly that topic (in the doodle
        look the real photo or portrait of the section's subject, on its
        coloured card); else a drawing of it; else the plan's icon (footage
        look); else the card shows only the number and the title.
        """
        item = scene.center or scenes.SceneItem(label=scene.text)
        scene.center = item
        image = self._pinned(item)
        if image is None and self.options.picture_check and self._gemini:
            save_dir = os.path.join(self.work_dir, "pictures")
            candidates = []
            for query in dict.fromkeys(q for q in (scene.query_local, scene.query) if q):
                with self._icon_lock:
                    used = set(self._used_urls) | set(item.avoid)
                candidates += web_images.find_candidates(query, save_dir, kind=scene.look, exclude_urls=used, limit=3)
            candidates = candidates[:5]
            if candidates:
                first = re.split(r"(?<=[.!?…])\s+", (text or "").strip())[0][:300]
                context = f"{scene.text}. {first}".strip()
                choice = gemini_media.choose_picture(
                    [c.path for c in candidates], context, scene.query or scene.text, self.options.language, purpose="opener"
                )
                if choice is not None and choice >= 0:
                    found = candidates[choice]
                    with self._icon_lock:
                        self._used_urls.add(found.url)
                    self.credits.append(found.credit())
                    image = scenes.prepare_picture(found.path, allow_cutout=scene.look != "photo" and not self.doodle)
                    if image is not None:
                        image.info["url"] = found.url
                        if self.doodle:
                            image.info["framed"] = True
                else:
                    logger.info(f"no picture for the section {scene.text!r} passed the check")
        if image is None and self.doodle:
            image = self._drawing(item.draw or scene.query or scene.text, mascot=item.otter)
        elif image is None and self._gemini:
            path = gemini_media.illustrate(item.draw or scene.query or scene.text)
            if path:
                image = scenes.prepare_picture(path)
                self.credits.append("Illustrations: generated with Google Gemini")
        if image is None and (item.icon or item.draw) and not self.doodle:
            image = self._icon_picture(item)
        self._item_pictures[id(item)] = image

    def _story_pictures(self, scene: scenes.Scene) -> None:
        """The flipbook drawn by the image model so the moments match each other."""
        frames = gemini_media.illustrate_sequence([item.draw or item.label for item in scene.items])
        if len(frames) != len(scene.items):
            return
        self.credits.append("Illustrations: generated with Google Gemini")
        for item, path in zip(scene.items, frames):
            image = scenes.prepare_picture(path, allow_cutout=False)
            if image is not None:
                image.info["scene"] = True
                self._item_pictures[id(item)] = image

    def _scene_picture(self, item: scenes.SceneItem):
        """Picture for a scene element: prepared in advance, or (footage look) an icon on demand.

        The doodle look never puts an icon in place of a picture that failed:
        the shot shows what it has (a missing animation frame is skipped).
        """
        if id(item) in self._item_pictures:
            return self._item_pictures[id(item)]
        return None if self.doodle else self._icon_picture(item)

    def scene_renderer(self) -> scenes.SceneRenderer:
        if self._scene_renderer is None:
            every = [sc for ss in self._scenes.values() for sc in ss]
            texts = [f"{i.label} {i.unit} {i.link}" for sc in every for i in sc.items]
            texts += [f"{sc.text} {sc.unit} {sc.example} {sc.center.label if sc.center else ''}" for sc in every]
            self._scene_renderer = scenes.SceneRenderer(
                self.theme,
                self.work_dir,
                self.canvas_color(),
                self._scene_picture,
                self._host_still if self.poses else None,
                self.sfx,
                font_path=scenes.hand_font_path(texts),
                doodle=self.doodle,
                boil=self.options.boil,
            )
        return self._scene_renderer

    def canvas_color(self) -> Tuple[int, int, int]:
        if self.doodle:
            if not self.options.canvas_color:
                return DOODLE_COLOR
            try:
                return scene_canvas_color(self.options.canvas_color, self.theme.accent)
            except ValueError:
                return DOODLE_COLOR
        return scene_canvas_color(self.options.scene_color, self.theme.accent)

    def _canvas_overlay(self) -> fx.Overlay:
        path = os.path.join(self.work_dir, "canvas.png")
        if not os.path.isfile(path):
            scenes.paper((self.theme.width, self.theme.height), self.canvas_color(), vignette=0.0).save(path)
        return fx.Overlay(path, x="0", y="0")

    def _logo_overlay(self) -> Optional[fx.Overlay]:
        """The channel's badge in the top-right corner."""
        path = os.path.join(self.work_dir, "logo.png")
        if not os.path.isfile(path):
            source = None
            if self.options.logo == "nutria":
                name = _pick(self.poses, ("feliz", "explicando", "neutral"))
                if name:
                    with Image.open(self.poses[name].idle) as pose:
                        pose = pose.convert("RGBA")
                        bbox = pose.getchannel("A").getbbox() or (0, 0, pose.width, pose.height)
                        pose = pose.crop(bbox)
                        side = int(pose.width * 0.82)
                        left = (pose.width - side) // 2
                        source = pose.crop((left, 0, left + side, side))
            elif self.options.logo and os.path.isfile(self.options.logo):
                with Image.open(self.options.logo) as picture:
                    source = picture.convert("RGBA")
            if source is None:
                return None
            fx.render_badge(source, self.theme.px(118), self.theme.accent).save(path)
        margin = self.theme.px(30)
        return fx.Overlay(path, x=f"W-w-{margin}", y=str(margin))

    def _host_still(self, expression: str, height: int):
        names = list(self.poses)
        if expression not in self.poses:
            expression = _pick(self.poses, ("explicando", "feliz", "neutral"))
        return self.renderer().still(expression or names[0], height)

    # -- layout -------------------------------------------------------------

    def _save(self, image, name: str) -> Tuple[str, int, int]:
        path = os.path.join(self.work_dir, name)
        image.save(path)
        return path, image.width, image.height

    def _sound(self, edit: SegmentEdit, time: float, name: str) -> None:
        if name in self.sfx:
            edit.sounds.append((time, self.sfx[name], _SFX_GAIN.get(name, 0.8)))

    def segment_edit(self, index: int, offset: float, show_titles: bool) -> SegmentEdit:
        self._prepare()
        segment = self.segments[index]
        narration = self.narrations[index]
        duration = narration.frames / 30.0
        theme = self.theme
        width, height = theme.width, theme.height
        margin = theme.px(44)
        edit = SegmentEdit()
        planned = self.host_plan[index] if index < len(self.host_plan) else host.HostSegment("off")
        has_character = bool(self.poses)

        if self.doodle:
            edit.overlays.append(self._canvas_overlay())
        elif index > 0 and not self.options.seamless:
            # Centred on the cut, so the whoosh carries the transition.
            self._sound(edit, -0.2, "whoosh")

        # Chapter label slides in from the left (and stays above the scenes).
        chip_overlay = None
        if show_titles and segment.kind == "item":
            chip = fx.render_chapter_chip(theme, segment.number or None, segment.name or segment.chapter)
            path, chip_w, _ = self._save(chip, f"chip-{index:02d}.png")
            ease = fx.ease_expression(0.45, 0.12)
            # The PNG carries its drop-shadow padding; offset it so the chip
            # itself lands on the margin.
            left = margin - fx.shadow_padding(theme)
            chip_overlay = fx.Overlay(path, x=f"{left}-({chip_w}+{margin})*(1-{ease})", y=str(left))

        subscribe_window = self._subscribe_window(index, duration)
        # Callouts stay right of the host in landscape; in portrait they sit above it.
        character_right = 0
        if has_character and not theme.portrait:
            character_right = int(width * 0.02) + self.renderer().canvas[0]
        for number, (beat, picture) in enumerate(self._beats.get(index, [])):
            if beat.kind in ("image", "images"):
                members = beat.members if beat.kind == "images" else [beat]
                for position, (member, (cx, cy, box_w, box_h)) in enumerate(zip(members, self._picture_slots(len(members)))):
                    try:
                        sticker = fx.make_sticker(
                            theme, member.picture or picture, box_w, box_h, seed=index * 10 + number + position,
                            allow_cutout=member.look != "photo",
                        )
                    except Exception as exc:
                        self.warnings.append(f"picture for {member.query!r} could not be used: {exc}")
                        continue
                    path, w, h = self._save(sticker, f"beat-{index:02d}-{number}-{position}.png")
                    x = int(fx.clamp(cx - w / 2, 0, width - w))
                    y = int(fx.clamp(cy - h / 2, 0, height - h))
                    rise = theme.px(46)
                    ease = fx.ease_expression(0.35, member.start)
                    edit.overlays.append(
                        fx.Overlay(path, x=str(x), y=f"{y}+{rise}*(1-{ease})", start=member.start, end=beat.end, fade_in=0.25, fade_out=0.2)
                    )
                    self._sound(edit, member.start, "pop")
            else:
                if subscribe_window and beat.end > subscribe_window[0] - 0.2 and beat.start < subscribe_window[1]:
                    continue
                callout = fx.render_callout(theme, beat.text)
                path, w, h = self._save(callout, f"text-{index:02d}-{number}.png")
                if theme.portrait:
                    cx, cy = width * 0.5, height * 0.66
                else:
                    cx = (character_right + width) / 2 if has_character else width * 0.5
                    cy = height * 0.82
                x = int(fx.clamp(cx - w / 2, character_right, width - w))
                y = int(fx.clamp(cy - h / 2, 0, height - h - theme.px(14)))
                ease = fx.ease_expression(0.3, beat.start)
                edit.overlays.append(
                    fx.Overlay(path, x=str(x), y=f"{y}+{theme.px(30)}*(1-{ease})", start=beat.start, end=beat.end, fade_in=0.2, fade_out=0.2)
                )
                self._sound(edit, beat.start, "tick")

        if planned.windows:
            edit.overlays.append(self._host_overlay(index, planned))
            for window in planned.windows:
                if window.enter:
                    self._sound(edit, window.start + 0.05, "bloop")

        # Scenes slide over the footage, the stickers and the host.
        for number, scene in enumerate(self._scenes.get(index, [])):
            try:
                overlays, sounds = self.scene_renderer().build(scene, f"{index:02d}-{number}")
            except Exception as exc:
                logger.exception(f"scene {scene.type!r} could not be drawn")
                self.warnings.append(f"a {scene.type} scene could not be drawn ({exc}); the otter stood in for it")
                self._stats["build_failed"] += 1
                stand_in = self._stand_in(scene) if self.doodle else None
                if stand_in is None:
                    continue
                try:
                    overlays, sounds = self.scene_renderer().build(stand_in, f"{index:02d}-{number}-otter")
                except Exception as again:
                    logger.warning(f"the stand-in shot could not be drawn either: {again}")
                    continue
            edit.overlays += overlays
            edit.sounds += sounds

        if chip_overlay is not None:
            # The label hides while a scene fills the screen, so nothing drawn
            # at the top of the scene ends up behind it. In the doodle look it
            # stays over the full-screen pictures and leaves for the opener and
            # the compositions, whose titles and labels use the top of the frame.
            covered = [(sc.start, sc.end) for sc in self._scenes.get(index, []) if not (self.doodle and _pill_over(sc))]
            edit.overlays += chip_spans(chip_overlay, covered, duration, chip_w + margin)

        if subscribe_window:
            edit.overlays.append(self._subscribe_overlay(subscribe_window[0], has_character))
            click = subscribe_window[0] + self._subscribe["click"]
            if self._subscribe.get("sound"):
                edit.sounds.append((subscribe_window[0], self._subscribe["sound"], _SFX_GAIN["subscribe"]))
            else:
                self._sound(edit, click, "click")

        if self.options.logo:
            logo = self._logo_overlay()
            if logo is not None:
                edit.overlays.append(logo)

        volume = max(0.0, float(self.options.sfx_volume))
        edit.sounds = [(time, path, gain * volume) for time, path, gain in edit.sounds if gain * volume > 0]

        if self.options.progress_bar:
            bar = fx.render_progress_bar(theme)
            path, _, bar_h = self._save(bar, "progress.png")
            edit.overlays.append(
                fx.Overlay(path, x=f"-w+w*({offset:.3f}+t)/{self.total:.3f}", y=str(height - bar_h))
            )
        return edit

    def _picture_slots(self, count: int) -> List[Tuple[float, float, int, int]]:
        """(centre x, centre y, box width, box height) for 1-4 pictures shown together.

        One picture sits in the middle; more are spread symmetrically
        (left/right, left/middle/right, ...).
        """
        width, height = self.theme.width, self.theme.height
        if self.theme.portrait:
            if count == 1:
                return [(width * 0.5, height * 0.42, int(width * 0.92), int(height * 0.46))]
            rows = [height * (0.18 + 0.56 * (k + 0.5) / count) for k in range(count)]
            return [(width * 0.5, y, int(width * 0.8), int(height * 0.56 / count)) for y in rows]
        if count == 1:
            return [(width * 0.5, height * 0.48, int(width * 0.62), int(height * 0.8))]
        step = min(0.33, 0.94 / count)
        box_w = int(width * min(0.4, 0.92 / count))
        return [
            (width * (0.5 + (k - (count - 1) / 2) * step), height * 0.48, box_w, int(height * 0.66))
            for k in range(count)
        ]

    def _host_overlay(self, index: int, planned: host.HostSegment) -> fx.Overlay:
        theme = self.theme
        renderer = self.renderer()
        narration = self.narrations[index]
        mouth = fx.mouth_schedule(narration.pcm, 24000, 30) if self.options.lip_sync else None
        frames = host.segment_frames(planned, renderer, narration.frames, mouth)
        track = host.write_concat(frames, os.path.join(renderer.work_dir, f"host-{index:02d}.txt"))
        canvas_w, canvas_h = renderer.canvas
        char_h = renderer.char_size[1]
        x = int(theme.width * 0.02)
        base_y = theme.height - canvas_h + int(char_h * HOST_SINK)
        bob = f"{theme.px(4)}*sin(2*PI*t/2.6)"
        drop = f"{canvas_h}*({host.position_offset(planned.windows)})"
        return fx.Overlay(track, x=str(x), y=f"{base_y}+{bob}+{drop}", mode="concat")

    def _subscribe_window(self, index: int, duration: float) -> Optional[Tuple[float, float]]:
        if not self._subscribe:
            return None
        mode = self.options.subscribe
        kinds = [s.kind for s in self.segments]
        first_item = kinds.index("item") if "item" in kinds else 0
        length = self._subscribe.get("duration", SUBSCRIBE_SECONDS)
        start = None
        if mode in ("both", "intro"):
            if kinds[0] == "intro" and self.narrations[0].frames / 30 >= 4.5:
                if index == 0:
                    start = max(0.3, duration - length - 0.3)
            elif index == first_item:
                start = 1.0
        if mode in ("both", "outro") and index == len(self.segments) - 1 and start is None:
            start = 0.6 if kinds[-1] == "outro" else max(0.5, duration - length - 0.5)
        if start is None or start + 1.0 > duration:
            return None
        return start, min(duration - 0.05, start + length)

    def _subscribe_overlay(self, start: float, has_character: bool) -> fx.Overlay:
        theme = self.theme
        info = self._subscribe
        cx = "(W-w)/2" if not has_character or theme.portrait else f"(W-w)/2+{int(theme.width * 0.08)}"
        y = f"H-h-{theme.px(40)}"
        if self.doodle and not theme.portrait:
            # Over the drawings it waits in the top-right corner, away from the pictures' subject and captions.
            cx, y = f"W-w-{theme.px(30)}", str(theme.px(30))
        if info["mode"] == "still":
            ease = fx.ease_expression(0.35, start)
            return fx.Overlay(
                info["source"], x=cx, y=f"{y}+{theme.px(50)}*(1-{ease})", start=start,
                end=start + SUBSCRIBE_SECONDS, fade_in=0.25, fade_out=0.3,
            )
        return fx.Overlay(
            info["source"], x=cx, y=y, start=start,
            end=start + info.get("duration", SUBSCRIBE_SECONDS), mode=info["mode"],
        )

    # -- review before rendering ---------------------------------------------

    @staticmethod
    def _review_slots(scene: scenes.Scene) -> List[Tuple[str, scenes.SceneItem]]:
        """(where it is pinned in the plan, element) for every picture a shot shows."""
        if scene.type == "opener":
            return [("opener", scene.center)] if scene.center else []
        if scene.type == "animation":
            return [(f"frame:{item.slot}", item) for item in scene.items if item.slot >= 0]
        if scene.type in scenes.ITEM_SCENES or scene.type in scenes.PART_SCENES:
            slots = [(f"item:{item.slot}", item) for item in scene.items if item.slot >= 0]
            return slots + ([("center", scene.center)] if scene.center and scene.type == "diagram" else [])
        if scene.type in ("clip", "meme"):
            return []
        return [("center", scene.center)] if scene.center else []

    @staticmethod
    def _shot_as_made(spec: dict, scene: scenes.Scene) -> dict:
        """The plan entry of a shot as it was finally made: an archive picture, a clip or an animation that had to
        be drawn (or photographed) instead becomes an illustration, so the render does not look for it again."""
        if scene.type == spec.get("type") or scene.type != "illustration":
            return spec
        center = scene.center or scenes.SceneItem()
        frames = [f.get("draw", "") for f in spec.get("frames") or [] if isinstance(f, dict)]
        shot = {
            "type": "illustration", "at": spec.get("at") or center.at,
            "draw": center.draw or spec.get("draw") or (frames[0] if frames else "") or center.query,
            "query": center.query or spec.get("query", ""), "text": spec.get("text", ""),
        }
        for key in ("camera", "characters", "otter", "motion"):
            if spec.get(key):
                shot[key] = spec[key]
        return shot

    def write_review(self) -> str:
        """Make every picture of the video and stop before rendering.

        The pictures are copied to review/ and pinned in edit-plan.json (a
        render with that plan uses them as they are, on the same timing,
        since the narration is cached), and review.json lists the shots, with
        what is said meanwhile, for the Studio's review page. Returns the path
        of review.json.
        """
        import shutil

        self._prepare()
        folder = os.path.join(self.task_dir, "review")
        os.makedirs(folder, exist_ok=True)
        assets = os.path.abspath(self.options.assets_dir) if self.options.assets_dir else ""
        shots = []
        for index, segment in enumerate(self.segments):
            entry = self._entry(index)
            planned = list(entry.get("shots") or [])
            made: Dict[int, dict] = {}
            listed: List[dict] = []
            for scene in self._scenes.get(index, []):
                if scene.reprise or (scene.type != "opener" and not 0 <= scene.spec < len(planned)):
                    continue
                key = "opener" if scene.type == "opener" else f"{scene.spec:02d}"
                pictures = []
                for slot, item in self._review_slots(scene):
                    image = self._item_pictures.get(id(item))
                    source = image.info.get("source", "") if image is not None else ""
                    if not source or not os.path.isfile(source) or (assets and os.path.abspath(source).startswith(assets)):
                        continue  # nothing to review (or the character's own pose)
                    target = os.path.join(folder, f"s{index:02d}-{key}-{slot.replace(':', '')}{os.path.splitext(source)[1] or '.png'}")
                    shutil.copyfile(source, target)
                    pictures.append({"slot": slot, "file": target, "url": image.info.get("url", ""),
                                     "kind": "archive" if image.info.get("archive") else "picture"})
                if scene.media and os.path.isfile(scene.media) and scene.type in ("illustration", "clip"):
                    target = os.path.join(folder, f"s{index:02d}-{key}-video{os.path.splitext(scene.media)[1] or '.mp4'}")
                    shutil.copyfile(scene.media, target)
                    pictures.append({"slot": "video", "file": target, "url": "", "kind": "video"})
                if scene.type == "opener":
                    place = entry.setdefault("opener", {"query": scene.query or scene.text})
                    for picture in pictures:
                        place["image"] = picture["file"]
                else:
                    shot = self._shot_as_made(planned[scene.spec], scene)
                    for picture in pictures:
                        slot = picture["slot"]
                        if slot.startswith(("frame:", "item:")):
                            kind, position = slot.split(":")
                            parts = shot.get("frames" if kind == "frame" else ("items" if "items" in shot else
                                             llm_part_key(shot.get("type", ""))))
                            if isinstance(parts, list) and int(position) < len(parts) and isinstance(parts[int(position)], dict):
                                parts[int(position)]["image"] = picture["file"]
                        elif slot == "center" and isinstance(shot.get("center"), dict):
                            shot["center"]["image"] = picture["file"]  # the hub of a diagram
                        else:
                            shot["video" if slot == "video" else "image"] = picture["file"]
                    made[scene.spec] = shot
                center = scene.center or scenes.SceneItem()
                anchor = center.at or (scene.items[0].at if scene.items else "")
                if scene.type == "opener":
                    said = re.split(r"(?<=[.!?…])\s+", segment.text.strip())[0]
                else:
                    said = sentence_at(segment.text, anchor) if anchor else ""
                listed.append({
                    "id": f"{index}-{key}", "segment": index, "shot": -1 if scene.type == "opener" else scene.spec,
                    "chapter": segment.chapter, "type": scene.type,
                    "planned": planned[scene.spec].get("type", "") if scene.type != "opener" else "opener",
                    "start": round(scene.start, 2), "end": round(scene.end, 2), "said": said,
                    "describe": center.draw or " / ".join(i.draw for i in scene.items if i.draw) or scene.query or center.query,
                    "query": scene.query or center.query, "caption": scene.text, "pictures": pictures,
                })
            if self.doodle and planned:
                # The render shows exactly what was reviewed: shots without a picture are left out of the plan.
                order = sorted(made)
                entry["shots"] = [made[position] for position in order]
                for shot in listed:
                    if shot["shot"] >= 0:
                        shot["shot"] = order.index(shot["shot"])
            shots += listed
        self._save_plan()
        plan_file = os.path.join(self.task_dir, "edit-plan.json")
        reviewed = os.path.join(self.task_dir, REVIEW_PLAN_FILE)
        shutil.copyfile(plan_file, reviewed)  # the plan as reviewed: the review's decisions always start from it
        path = os.path.join(self.task_dir, REVIEW_FILE)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump({"version": 1, "plan_file": plan_file, "reviewed_plan": reviewed,
                       "report": self.report(), "warnings": list(dict.fromkeys(self.warnings)), "shots": shots},
                      fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        logger.success(f"review ready: {len(shots)} shots, {sum(len(s['pictures']) for s in shots)} pictures ({path})")
        return path

    def report(self) -> List[str]:
        """What the editor did, for render-report.txt: shots by type and what happened to the pictures."""
        counts: Dict[str, int] = {}
        for timed in self._scenes.values():
            for scene in timed:
                counts[scene.type] = counts.get(scene.type, 0) + 1
        frames = sum(len(sc.items) for timed in self._scenes.values() for sc in timed if sc.type == "animation")
        lines = ["Shots: " + (", ".join(f"{name} {count}" for name, count in sorted(counts.items(), key=lambda kv: -kv[1])) or "none")]
        if frames:
            lines.append(f"Animation frames: {frames}")
        if self.doodle:
            stats = self._stats
            lines.append(
                f"Drawings: {stats['drawn']} drawn, {self._failed_drawings} failed, {stats['rejected']} redrawn after "
                f"Gemini's check, {self._drawings} of the budget of {self.options.max_drawings} used"
                + (f", {stats['budget']} not drawn because the budget ran out" if stats["budget"] else "")
            )
            lines.append(
                f"Real history: {stats['archive']} archive pictures, {stats['archive_drawn']} drawn because no authentic "
                f"picture passed the check; {stats['sheets']} character sheets; {stats['videos']} AI videos; "
                f"{stats['pinned']} pictures kept from the review; {stats['repeats']} repeated shots left out"
            )
            lines.append(
                f"Stand-ins: {stats['real']} real pictures, {stats['photo']} photos instead of drawings, {stats['otter']} "
                f"otter poses, {stats['items_dropped']} elements and {stats['shots_dropped']} shots left out, "
                f"{stats['build_failed']} shots that failed to render"
            )
            reason = gemini_media.last_error()
            if reason:
                lines.append(f"Last drawing error: {reason}")
        return lines

    def write_credits(self) -> str:
        if not self.credits:
            return ""
        path = os.path.join(self.task_dir, "credits.txt")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(dict.fromkeys(self.credits)) + "\n")
        return path
