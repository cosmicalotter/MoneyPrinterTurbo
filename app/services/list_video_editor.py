"""
Automatic editing for list videos.

An LLM "director" decides, per segment, the host character's expression and a
few beats anchored to exact words of the narration: a picture of what is being
mentioned, or a short key fact. The editor turns that plan into timed ffmpeg
overlays (chapter label, character with lip-flap, stickers, callouts,
subscribe animation, progress bar) and a list of sound effects.

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
from app.services import llm, material, web_images
from app.utils import utils

BEAT_MODES = ("web", "ai", "none")
SUBSCRIBE_MODES = ("both", "intro", "outro", "none")
_NEUTRAL_EXPRESSIONS = ("neutral", "normal", "feliz", "happy", "sonriente")
_INTRO_EXPRESSIONS = ("sorprendido", "surprised", "asombrado", "feliz", "happy")
_OUTRO_EXPRESSIONS = ("feliz", "happy", "sonriente", "neutral")
_SFX_GAIN = {"whoosh": 0.55, "pop": 0.7, "tick": 0.8, "click": 0.9, "subscribe": 0.75}
SUBSCRIBE_SECONDS = 3.6
IMAGE_BEAT_MAX = 5.5
IMAGE_BEAT_MIN = 1.6
TEXT_BEAT_SECONDS = 2.6


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
        plan.append({"index": index, "expression": _pick(poses, preferred), "beats": []})
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
        self.poses = fx.load_character(options.assets_dir)
        self.sfx = fx.resolve_sfx(options.assets_dir, self.work_dir) if options.sound_effects else {}
        self.credits: List[str] = []
        self.warnings: List[str] = []
        self._used_urls: set = set()
        self._character_cache: Dict[Tuple[str, bool], str] = {}
        self._subscribe = self._prepare_subscribe()
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
            if self.options.beats == "none":
                entry["beats"] = []
        self.plan = plan
        with open(os.path.join(self.task_dir, "edit-plan.json"), "w", encoding="utf-8") as fp:
            json.dump({"segments": plan}, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        return plan

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

    def _character_png(self, expression: str, talking: bool) -> str:
        key = (expression, talking)
        if key not in self._character_cache:
            pose = self.poses[expression]
            source = pose.talk if talking and pose.talk else pose.idle
            height = int(self.theme.height * (0.2 if self.theme.portrait else 0.34))
            name = f"character-{fx.safe_seed(expression)}-{'talk' if talking else 'idle'}.png"
            self._character_cache[key] = fx.prepare_character_image(
                source, os.path.join(self.work_dir, name), height
            )
        return self._character_cache[key]

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
        segment = self.segments[index]
        narration = self.narrations[index]
        duration = narration.frames / 30.0
        entry = self.plan[index] if index < len(self.plan) else {"expression": "", "beats": []}
        theme = self.theme
        width, height = theme.width, theme.height
        margin = theme.px(44)
        edit = SegmentEdit()
        has_character = bool(self.poses and entry.get("expression") in self.poses)

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
        beats = schedule_beats(entry.get("beats") or [], segment.text, narration, duration)
        # Callouts stay right of the host in landscape; in portrait they sit above it.
        character_right = int(width * 0.24) if has_character and not theme.portrait else 0
        for number, beat in enumerate(beats):
            if beat.kind == "image":
                picture = self._beat_picture(beat)
                if not picture:
                    continue
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

        if has_character:
            edit.overlays.append(self._character_overlay(index, entry["expression"], narration, segment.kind))

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

    def _character_overlay(self, index, expression, narration: Narration, kind: str) -> fx.Overlay:
        theme = self.theme
        idle = self._character_png(expression, False)
        pose = self.poses[expression]
        source, mode = idle, "still"
        if pose.talk:
            talk = self._character_png(expression, True)
            runs = fx.mouth_schedule(narration.pcm, 24000, 30)
            spoken = sum(count for _, count in runs)
            if narration.frames > spoken:
                runs.append((False, narration.frames - spoken))
            lines = ["ffconcat version 1.0"]
            for is_open, count in runs:
                lines.append(f"file '{os.path.basename(talk if is_open else idle)}'")
                lines.append(f"duration {count / 30:.4f}")
            lines.append(f"file '{os.path.basename(idle)}'")
            source = os.path.join(self.work_dir, f"mouth-{index:02d}.txt")
            with open(source, "w", encoding="utf-8") as fp:
                fp.write("\n".join(lines) + "\n")
            mode = "concat"
        x = str(int(theme.width * 0.025))
        bob = f"{theme.px(5)}*sin(2*PI*t/2.6)"
        y = f"H-h*0.97+{bob}"
        if index == 0:
            y += f"+h*(1-{fx.ease_expression(0.6, 0.15)})"
        elif kind == "item":
            y += f"-{theme.px(18)}*sin(PI*clip(t/0.32,0,1))"
        return fx.Overlay(source, x=x, y=y, mode=mode)

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
