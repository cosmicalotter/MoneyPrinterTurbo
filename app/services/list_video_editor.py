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
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

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


@dataclass
class SoundEvent:
    time: float
    path: str
    gain: float = 1.0


@dataclass
class Narration:
    """What the renderer measured for one segment before drawing it."""

    pcm: bytes
    speech_seconds: float
    frames: int
    sub_maker: object = None


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


def anchor_time(text: str, anchor: str, sub_maker, speech_seconds: float) -> Optional[float]:
    """Seconds into the segment when ``anchor`` is spoken, or None."""
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

    times = cue_word_times(sub_maker)
    if times and len(times) >= len(words) * 0.6:
        position = min(len(times) - 1, round(index * len(times) / len(words)))
        return max(0.0, min(times[position], speech_seconds))

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
        return anchor_time(text, anchor, narration.sub_maker, narration.speech_seconds)

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


def load_plan_file(path: str, segment_count: int, expressions: List[str]) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    return llm.normalize_edit_plan(data, segment_count, expressions)


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
        self.poses = fx.load_character(options.assets_dir) if options.host != "none" else {}
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

    # -- plan ---------------------------------------------------------------

    def make_plan(self) -> List[dict]:
        expressions = sorted(self.poses)
        count = len(self.segments)
        plan = None
        if self.options.plan_file:
            plan = load_plan_file(self.options.plan_file, count, expressions)
            logger.info(f"using edit plan: {self.options.plan_file}")
        elif expressions or self.options.beats != "none":
            payload = [
                {"index": i, "kind": s.kind, "title": s.chapter, "text": s.text}
                for i, s in enumerate(self.segments)
            ]
            reference = self._reference_plan()
            if reference:
                plan = llm.generate_edit_plan(payload, expressions, self.options.language, reference=reference)
            else:
                plan = llm.generate_edit_plan(payload, expressions, self.options.language)
            if plan is None:
                self.warnings.append(
                    "the LLM did not return an edit plan; only the base visuals were used"
                )
        fallback = default_plan(self.segments, self.poses)
        plan = plan or fallback
        for entry, default in zip(plan, fallback):
            if not entry.get("expression"):
                entry["expression"] = default["expression"]
            entry.setdefault("backgrounds", [])
            entry.setdefault("scenes", [])
            if not self.options.scenes:
                entry["scenes"] = []
            if self.options.beats == "none":
                entry["beats"] = [b for b in entry.get("beats") or [] if b.get("type") == "react"]
        self.plan = plan
        self._save_plan()
        return plan

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
        with open(os.path.join(self.task_dir, "edit-plan.json"), "w", encoding="utf-8") as fp:
            json.dump({"segments": self.plan}, fp, ensure_ascii=False, indent=2)
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
            start = anchor_time(segment.text, background.get("at", ""), narration.sub_maker, narration.speech_seconds)
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
        for index, segment in enumerate(self.segments):
            narration = self.narrations[index]
            duration = narration.frames / 30.0
            entry = self._entry(index)
            pauses = host.find_pauses(narration.pcm)

            def locate(anchor, segment=segment, narration=narration):
                return anchor_time(segment.text, anchor, narration.sub_maker, narration.speech_seconds)

            window = self._subscribe_window(index, duration)
            timed_scenes = scenes.time_scenes(entry.get("scenes") or [], locate, pauses, duration, [window] if window else [])
            self._scenes[index] = timed_scenes
            cover = [(scene.start, scene.end) for scene in timed_scenes]
            covers.append(cover)
            beats = schedule_beats(entry.get("beats") or [], segment.text, narration, duration)
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
        seed = fx.safe_seed("".join(s.chapter for s in self.segments)) % len(host.ITEM_PATTERN)
        self.host_plan = host.plan_host(infos, names, mode, seed)
        for planned, cover in zip(self.host_plan, covers):
            planned.windows = _avoid_cover(planned.windows, cover)
        if self.plan and names:
            for entry, planned in zip(self.plan, self.host_plan):
                entry["host"] = planned.mode
            self._save_plan()

    def _fetch_pictures(self, texts: Dict[int, str]) -> None:
        """Find every picture of the video at once (searches and checks run in parallel)."""
        jobs = []
        for index, beats in self._beats.items():
            text = texts[index]
            for beat, _ in beats:
                for member in beat.members if beat.kind == "images" else [beat] if beat.kind == "image" else []:
                    jobs.append(lambda m=member, t=text: setattr(m, "picture", self._beat_picture(m, sentence_at(t, m.at))))
        for index, timed in self._scenes.items():
            text = texts[index]
            for scene in timed:
                if scene.type == "figure":
                    jobs.append(lambda sc=scene, t=text: self._figure_picture(sc, t))
                elif scene.type == "story" and self.options.illustrations == "ai" and self._gemini:
                    jobs.append(lambda sc=scene: self._story_pictures(sc))
                for item in scene.items + ([scene.center] if scene.center and scene.type != "figure" else []):
                    if item.query or item.icon or item.draw:
                        jobs.append(lambda i=item, t=text: self._item_picture(i, sentence_at(t, i.at)))
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(job) for job in jobs]:
                try:
                    future.result()
                except Exception as exc:
                    logger.warning(f"a picture could not be prepared: {type(exc).__name__}: {exc}")
        # Full-screen pictures that were not found leave the footage alone.
        for index, timed in self._scenes.items():
            self._scenes[index] = [
                sc for sc in timed if sc.type != "figure" or self._item_pictures.get(id(sc.center)) is not None
            ]

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
                    self.credits.append("Illustrations: generated with Google Imagen")
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
                self.credits.append("Illustrations: generated with Google Imagen")
        if image is None and item.query and self.options.picture_check and self._gemini:
            # Real pictures only when Gemini can confirm they fit; icons otherwise.
            path = self._web_picture(item.query, "diagram", line, "scene")
            if path:
                image = scenes.prepare_picture(path)
        if image is None:
            image = self._icon_picture(item)
        self._item_pictures.setdefault(id(item), image)

    def _icon_picture(self, item: scenes.SceneItem):
        key = (item.icon, item.draw)
        if key not in self._scene_pictures:
            image = None
            for query in (item.icon, item.draw):
                path = icons.fetch(query) if query else ""
                if path:
                    image = scenes.prepare_picture(path)
                    self.credits.append(icons.CREDIT)
                    break
            self._scene_pictures[key] = image
        return self._scene_pictures[key]

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
        """Picture for a scene element (prepared in advance, or an icon on demand)."""
        if id(item) in self._item_pictures:
            return self._item_pictures[id(item)]
        return self._icon_picture(item)

    def scene_renderer(self) -> scenes.SceneRenderer:
        if self._scene_renderer is None:
            texts = [i.label for ss in self._scenes.values() for sc in ss for i in sc.items] + [
                sc.text for ss in self._scenes.values() for sc in ss
            ]
            self._scene_renderer = scenes.SceneRenderer(
                self.theme,
                self.work_dir,
                scene_canvas_color(self.options.scene_color, self.theme.accent),
                self._scene_picture,
                self._host_still if self.poses else None,
                self.sfx,
                font_path=scenes.hand_font_path(texts),
            )
        return self._scene_renderer

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

        if index > 0:
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
                self.warnings.append(f"a {scene.type} scene could not be drawn: {exc}")
                continue
            edit.overlays += overlays
            edit.sounds += sounds

        if chip_overlay is not None:
            edit.overlays.append(chip_overlay)

        if subscribe_window:
            edit.overlays.append(self._subscribe_overlay(subscribe_window[0], has_character))
            click = subscribe_window[0] + self._subscribe["click"]
            if self._subscribe.get("sound"):
                edit.sounds.append((subscribe_window[0], self._subscribe["sound"], _SFX_GAIN["subscribe"]))
            else:
                self._sound(edit, click, "click")

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
                return [(width * 0.5, height * 0.4, int(width * 0.84), int(height * 0.36))]
            rows = [height * (0.18 + 0.56 * (k + 0.5) / count) for k in range(count)]
            return [(width * 0.5, y, int(width * 0.8), int(height * 0.56 / count)) for y in rows]
        if count == 1:
            return [(width * 0.5, height * 0.45, int(width * 0.44), int(height * 0.62))]
        step = min(0.3, 0.9 / count)
        box_w = int(width * min(0.34, 0.86 / count))
        return [
            (width * (0.5 + (k - (count - 1) / 2) * step), height * 0.45, box_w, int(height * 0.5))
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

    def write_credits(self) -> str:
        if not self.credits:
            return ""
        path = os.path.join(self.task_dir, "credits.txt")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(dict.fromkeys(self.credits)) + "\n")
        return path
