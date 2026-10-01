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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from app.services import list_video_fx as fx
from app.services import list_video_host as host
from app.services import llm, material, web_images
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
    progress_bar: bool = True
    sound_effects: bool = True
    plan_file: str = ""
    language: str = ""
    host: str = "auto"  # auto (comes and goes), always, none


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
    kind: str
    start: float
    end: float
    query: str = ""
    look: str = "diagram"
    text: str = ""


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


def schedule_beats(beats: List[dict], text: str, narration: Narration, duration: float) -> List[_Beat]:
    """Give plan beats start/end times that never overlap the same zone."""
    timed = []
    for beat in beats:
        if beat.get("type") not in ("image", "text"):
            continue
        start = anchor_time(text, beat.get("at", ""), narration.sub_maker, narration.speech_seconds)
        if start is None:
            logger.debug(f"beat anchor not found in narration: {beat.get('at')!r}")
            continue
        start = max(0.15, start - 0.1)
        if start > duration - 1.0:
            continue
        timed.append(
            _Beat(
                kind=beat["type"],
                start=start,
                end=0.0,
                query=beat.get("query", ""),
                look=beat.get("look", "diagram"),
                text=beat.get("text", ""),
            )
        )
    timed.sort(key=lambda b: b.start)

    scheduled: List[_Beat] = []
    for kind, longest, shortest in (
        ("image", IMAGE_BEAT_MAX, IMAGE_BEAT_MIN),
        ("text", TEXT_BEAT_SECONDS, 1.2),
    ):
        # Keep the earliest beat and skip any that would cut it short, then
        # let each kept beat run until the next one of the same kind.
        kept: List[_Beat] = []
        for beat in (b for b in timed if b.kind == kind):
            if kept and beat.start - kept[-1].start < shortest + 0.15:
                continue
            if duration - 0.3 - beat.start < shortest:
                continue
            kept.append(beat)
        for position, beat in enumerate(kept):
            limit = kept[position + 1].start - 0.15 if position + 1 < len(kept) else duration - 0.3
            beat.end = min(beat.start + longest, limit)
        scheduled += kept
    return sorted(scheduled, key=lambda b: b.start)


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
            {"index": index, "expression": _pick(poses, preferred), "backgrounds": [], "beats": []}
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
        self._beats: Dict[int, List[Tuple[_Beat, str]]] = {}
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
            if self.options.beats == "none":
                entry["beats"] = [b for b in entry.get("beats") or [] if b.get("type") == "react"]
        self.plan = plan
        self._save_plan()
        return plan

    def _save_plan(self) -> None:
        with open(os.path.join(self.task_dir, "edit-plan.json"), "w", encoding="utf-8") as fp:
            json.dump({"segments": self.plan}, fp, ensure_ascii=False, indent=2)
            fp.write("\n")

    def _entry(self, index: int) -> dict:
        if index < len(self.plan):
            return self.plan[index]
        return {"expression": "", "backgrounds": [], "beats": []}

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
        """Fetch every picture and plan the host over the whole video."""
        if self.host_plan:
            return
        infos = []
        for index, segment in enumerate(self.segments):
            narration = self.narrations[index]
            duration = narration.frames / 30.0
            entry = self._entry(index)
            beats = schedule_beats(entry.get("beats") or [], segment.text, narration, duration)
            prepared = []
            for beat in beats:
                picture = self._beat_picture(beat) if beat.kind == "image" else ""
                if beat.kind == "image" and not picture:
                    continue
                prepared.append((beat, picture))
            self._beats[index] = prepared
            reactions = []
            for beat in entry.get("beats") or []:
                if beat.get("type") != "react":
                    continue
                time = anchor_time(segment.text, beat.get("at", ""), narration.sub_maker, narration.speech_seconds)
                if time is not None and time < duration - 1.5:
                    reactions.append((max(0.4, time), beat["expression"]))
            infos.append(
                host.SegmentInfo(
                    kind=segment.kind,
                    duration=duration,
                    expression=entry.get("expression", ""),
                    reactions=reactions,
                    pictures=[b.start for b, _ in prepared if b.kind == "image"],
                    pauses=host.find_pauses(narration.pcm),
                    mode=entry.get("host", ""),
                )
            )
        names = sorted(self.poses)
        mode = self.options.host if self.options.host in host.HOST_MODES else "auto"
        seed = fx.safe_seed("".join(s.chapter for s in self.segments)) % len(host.ITEM_PATTERN)
        self.host_plan = host.plan_host(infos, names, mode, seed)
        if self.plan and names:
            for entry, planned in zip(self.plan, self.host_plan):
                entry["host"] = planned.mode
            self._save_plan()

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
            self._renderer = host.HostRenderer(self.poses, height, self.work_dir)
        return self._renderer

    def _beat_picture(self, beat: _Beat) -> str:
        save_dir = os.path.join(self.work_dir, "pictures")
        if self.options.beats == "ai" and material.is_openai_image_enabled():
            items = material.generate_images_openai(
                search_term=beat.query, minimum_duration=1, save_dir=save_dir
            )
            if items:
                self.credits.append(f"{beat.query} — AI-generated illustration")
                return items[0].url
        found = web_images.find_image(beat.query, save_dir, kind=beat.look, exclude_urls=self._used_urls)
        if found:
            self.credits.append(found.credit())
            return found.path
        self.warnings.append(f"no picture found for {beat.query!r}")
        return ""

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

        # Chapter label slides in from the left.
        if show_titles and segment.kind == "item":
            chip = fx.render_chapter_chip(theme, segment.number or None, segment.name or segment.chapter)
            path, chip_w, _ = self._save(chip, f"chip-{index:02d}.png")
            ease = fx.ease_expression(0.45, 0.12)
            # The PNG carries its drop-shadow padding; offset it so the chip
            # itself lands on the margin.
            left = margin - fx.shadow_padding(theme)
            edit.overlays.append(
                fx.Overlay(path, x=f"{left}-({chip_w}+{margin})*(1-{ease})", y=str(left))
            )

        subscribe_window = self._subscribe_window(index, duration)
        # Callouts stay right of the host in landscape; in portrait they sit above it.
        character_right = 0
        if has_character and not theme.portrait:
            character_right = int(width * 0.02) + self.renderer().canvas[0]
        for number, (beat, picture) in enumerate(self._beats.get(index, [])):
            if beat.kind == "image":
                if theme.portrait:
                    box_w, box_h, cx, cy = int(width * 0.84), int(height * 0.36), width * 0.5, height * 0.4
                else:
                    box_w, box_h, cx, cy = int(width * 0.38), int(height * 0.6), width * 0.7, height * 0.48
                try:
                    sticker = fx.make_sticker(theme, picture, box_w, box_h, seed=index * 10 + number)
                except Exception as exc:
                    self.warnings.append(f"picture for {beat.query!r} could not be used: {exc}")
                    continue
                path, w, h = self._save(sticker, f"beat-{index:02d}-{number}.png")
                x = int(fx.clamp(cx - w / 2, 0, width - w))
                y = int(fx.clamp(cy - h / 2, 0, height - h))
                rise = theme.px(46)
                ease = fx.ease_expression(0.35, beat.start)
                edit.overlays.append(
                    fx.Overlay(path, x=str(x), y=f"{y}+{rise}*(1-{ease})", start=beat.start, end=beat.end, fade_in=0.25, fade_out=0.2)
                )
                self._sound(edit, beat.start, "pop")
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

    def _host_overlay(self, index: int, planned: host.HostSegment) -> fx.Overlay:
        theme = self.theme
        renderer = self.renderer()
        narration = self.narrations[index]
        mouth = fx.mouth_schedule(narration.pcm, 24000, 30)
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
