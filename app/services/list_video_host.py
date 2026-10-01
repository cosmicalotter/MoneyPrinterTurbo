"""
The host character's performance in edited list videos.

The host does not sit on screen for the whole video. It pops up from the
bottom edge to introduce some items, jumps in to react to a surprising line,
points at pictures as they pop in, waves hello and goodbye, and leaves the
stage to the footage the rest of the time. Every expression change lands with
a small squash-and-stretch bounce.

The performance is baked into a few PNG frames on one shared canvas (each
expression idle and talking, plus the bounce frames) and sequenced per segment
with an ffconcat list, so ffmpeg composites it with a single overlay whose
position expression adds the entrances and exits.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from app.services import list_video_fx as fx

FPS = 30
HOST_MODES = ("auto", "always", "none")
WAVE_NAMES = ("saludando", "waving", "wave", "hola", "hello")
POINT_NAMES = ("senalando", "señalando", "pointing", "point", "apuntando")
# How items take turns: introduce it, drop in later, both, leave it to the
# footage, ... Intro and outro always keep the host.
ITEM_PATTERN = ("lead", "react", "lead+react", "off", "react", "full")

ENTER_SECONDS = 0.5
EXIT_SECONDS = 0.42
BOUNCE_FRAMES = 10
MIN_WINDOW = 2.2  # shorter visits feel like a glitch
MIN_GAP = 1.8  # shorter absences are merged into one visit
CUE_SPACING = 0.6
REACTION_HOLD = 2.6
POINT_HOLD = 2.0
REACTION_STAY = 3.6  # how long a drop-in visit lasts after the reaction


@dataclass
class HostWindow:
    start: float
    end: float
    enter: bool = True  # rise from the bottom edge at ``start``
    exit: bool = True  # sink out of frame before ``end``


@dataclass
class HostCue:
    time: float
    expression: str
    bounce: bool = True


@dataclass
class SegmentInfo:
    """What the planner needs to know about one segment."""

    kind: str
    duration: float
    expression: str
    reactions: List[Tuple[float, str]] = field(default_factory=list)
    pictures: List[float] = field(default_factory=list)
    pauses: List[float] = field(default_factory=list)
    mode: str = ""


@dataclass
class HostSegment:
    mode: str
    windows: List[HostWindow] = field(default_factory=list)
    cues: List[HostCue] = field(default_factory=list)

    def visible(self, t: float) -> Optional[HostWindow]:
        for window in self.windows:
            if window.start <= t < window.end:
                return window
        return None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def pick_name(names: Sequence[str], wanted: Sequence[str]) -> str:
    lookup = {name.lower(): name for name in names}
    return next((lookup[w] for w in wanted if w in lookup), "")


def find_pauses(pcm: bytes, sample_rate: int = 24000, min_seconds: float = 0.2) -> List[float]:
    """Times where the narrator stops for a breath (start of each silence)."""
    runs = fx.mouth_schedule(pcm, sample_rate, FPS, min_frames=2) if pcm else []
    pauses, frame = [], 0
    for is_open, count in runs:
        if not is_open and count >= min_seconds * FPS and frame > 0:
            pauses.append(frame / FPS)
        frame += count
    return pauses


def _snap(pauses: Sequence[float], target: float, low: float, high: float) -> float:
    """The pause closest to ``target`` inside [low, high], else ``target``."""
    inside = [p for p in pauses if low <= p <= high]
    return min(inside, key=lambda p: abs(p - target)) if inside else target


def segment_mode(kind: str, item_number: int, seed: int) -> str:
    if kind in ("intro", "outro"):
        return "full"
    if item_number == 0:
        return "lead"  # the host hands over from the intro
    return ITEM_PATTERN[(item_number + seed) % len(ITEM_PATTERN)]


def _windows_for(info: SegmentInfo, mode: str, base: str, point: str) -> Tuple[list, list]:
    duration = info.duration
    windows: List[List[float]] = []
    cues: List[Tuple[float, str, str]] = []  # time, expression, role
    if mode == "full":
        windows.append([0.0, duration])
    if mode in ("lead", "lead+react"):
        target = min(max(0.42 * duration, 3.5), 8.5)
        end = _snap(info.pauses, target, target - 1.5, target + 2.0) + 0.45
        windows.append([0.0, end])

    reactions = []
    for time, expression in sorted(info.reactions):
        if not reactions or time - reactions[-1][0] >= 4.0:
            reactions.append((time, expression))
    if mode == "off":
        reactions = reactions[:1]
    if mode in ("react", "lead+react") and not reactions:
        # A cameo: drop in to point at a picture, or just to keep company.
        after = windows[-1][1] + 1.5 if windows else 0.3 * duration
        later = [p for p in info.pictures if after <= p <= duration - 2.5]
        if later and point:
            reactions = [(later[0], point)]
        elif duration - after > 4.0:
            target = max(after, 0.55 * duration)
            reactions = [(_snap(info.pauses, target, target - 1.5, target + 1.5) + 0.3, base)]

    for time, expression in reactions:
        inside = any(w[0] + 0.6 <= time <= w[1] - 0.8 for w in windows)
        if not inside:
            stay = time + REACTION_STAY
            end = _snap(info.pauses, stay, stay - 0.8, stay + 1.8) + 0.45
            windows.append([time - 0.35, end])
        cues.append((time, expression, "react"))
    return windows, cues


def _normalize_windows(windows: List[List[float]], duration: float) -> List[List[float]]:
    windows = sorted([max(0.0, a), min(duration, b)] for a, b in windows)
    merged: List[List[float]] = []
    for start, end in windows:
        if start < 0.9:
            start = 0.0
        if duration - end < 1.4:
            end = duration
        if merged and start - merged[-1][1] < MIN_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [w for w in merged if w[1] - w[0] >= MIN_WINDOW or (w[0] == 0.0 and w[1] == duration)]


def _cues_for(
    info: SegmentInfo,
    windows: List[List[float]],
    events: List[Tuple[float, str, str]],
    base: str,
    wave: str,
    point: str,
) -> List[HostCue]:
    duration = info.duration
    cues: List[HostCue] = []
    for start, end in windows:
        # A visit made for a reaction opens with that face.
        opening = next((e for e in events if start - 0.05 <= e[0] <= start + 0.6), None)
        wanted: List[Tuple[float, str]] = []
        if info.kind == "intro" and start == 0.0 and wave and duration > 4.5:
            first = wave
            wanted.append((min(2.2, duration / 2), base))
        else:
            first = opening[1] if opening else base
        busy = [(opening[0], opening[0] + REACTION_HOLD)] if opening else []
        for time, expression, _ in events:
            if opening is not None and time == opening[0]:
                continue
            if start + 0.6 <= time <= end - 0.8:
                wanted.append((time, expression))
                if time + REACTION_HOLD < end - 0.6:
                    wanted.append((time + REACTION_HOLD, base))
                busy.append((time - 0.3, time + REACTION_HOLD))
        if point and point != base:
            # Point at the first picture that pops in while the host is free.
            for picture in info.pictures:
                if not start + 0.8 <= picture <= end - POINT_HOLD - 0.4:
                    continue
                if any(a - POINT_HOLD <= picture <= b for a, b in busy):
                    continue
                wanted += [(picture, point), (picture + POINT_HOLD, base)]
                break
        if info.kind == "outro" and wave and end == duration and end - start > 4.0:
            wanted.append((max(start + 1.5, duration - 2.8), wave))

        window_cues = [HostCue(start, first, False)]
        for time, expression in sorted(wanted):
            last = window_cues[-1]
            if expression == last.expression or time - last.time < CUE_SPACING or time >= end - 0.3:
                continue
            window_cues.append(HostCue(time, expression, True))
        cues += window_cues
    return cues


def plan_host(
    infos: Sequence[SegmentInfo], names: Sequence[str], host_mode: str = "auto", seed: int = 0
) -> List[HostSegment]:
    """Decide when the host is on screen and which face it makes."""
    if host_mode == "none" or not names:
        return [HostSegment("off") for _ in infos]
    wave = pick_name(names, WAVE_NAMES)
    point = pick_name(names, POINT_NAMES)
    fallback = next((n for n in names if n not in (wave, point)), names[0])
    planned: List[HostSegment] = []
    item_number = 0
    for info in infos:
        base = info.expression if info.expression in names else fallback
        if host_mode == "always":
            mode = "full"
        elif info.mode:
            mode = info.mode
        else:
            mode = segment_mode(info.kind, item_number, seed)
        if info.kind == "item":
            item_number += 1
        if info.duration < 5.0 and mode in ("lead", "lead+react"):
            mode = "full"
        raw, events = _windows_for(info, mode, base, point)
        windows = _normalize_windows(raw, info.duration)
        cues = _cues_for(info, windows, events, base, wave, point)
        planned.append(
            HostSegment(mode, [HostWindow(a, b) for a, b in windows], cues)
        )

    # Stay on stage across a cut instead of leaving and coming straight back.
    for index, segment in enumerate(planned):
        duration = infos[index].duration
        following = planned[index + 1] if index + 1 < len(planned) else None
        if segment.windows and segment.windows[-1].end >= duration:
            last = segment.windows[-1]
            if following is None:
                last.exit = False
            elif following.windows and following.windows[0].start == 0.0:
                last.exit = False
                following.windows[0].enter = False
                if following.cues and following.cues[0].time == 0.0:
                    following.cues[0].bounce = True  # a little hop for the new item
            else:
                last.end = duration - 0.05
        first = segment.windows[0] if segment.windows else None
        if first and first.enter and first.start == 0.0:
            first.start = 0.3 if index == 0 else 0.15
            for cue in segment.cues:
                if cue.time == 0.0:
                    cue.time = first.start
    return planned


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def bounce_shape(frame: int, frames: int = BOUNCE_FRAMES) -> Tuple[float, float, float]:
    """(scale x, scale y, lift as a fraction of the height) for a bounce frame."""
    p = frame / frames
    scale_y = 1 - 0.09 * math.exp(-3.2 * p) * math.cos(3 * math.pi * p)
    scale_x = 1 + 0.6 * (1 - scale_y)
    lift = 0.04 * math.sin(math.pi * min(1.0, max(0.0, (p - 0.1) / 0.6)))
    return scale_x, scale_y, lift


def _file_stem(name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore").decode())
    return f"{stem.strip('-') or 'pose'}-{fx.safe_seed(name)}"


def _bottom_anchor(image: Image.Image) -> float:
    """Horizontal centre of the lowest fifth of the drawing."""
    alpha = image.getchannel("A")
    band = alpha.crop((0, int(image.height * 0.8), image.width, image.height))
    box = band.getbbox()
    return (box[0] + box[2]) / 2 if box else image.width / 2


class HostRenderer:
    """Frames of the host on one shared canvas, written on demand."""

    def __init__(self, poses: Dict[str, fx.CharacterPose], target_height: int, work_dir: str):
        self.work_dir = os.path.join(work_dir, "host")
        os.makedirs(self.work_dir, exist_ok=True)
        sources: Dict[Tuple[str, bool], Image.Image] = {}
        for name, pose in poses.items():
            for talking, path in ((False, pose.idle), (True, pose.talk)):
                if not path:
                    continue
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGBA")
                if not fx.has_transparency(image):
                    image = fx.remove_flat_background(image) or image
                sources[(name, talking)] = image
        self.images = self._align(sources, target_height)
        width, height = next(iter(self.images.values())).size
        self.char_size = (width, height)
        self.canvas = (int(width * 1.1) // 2 * 2 + 2, int(height * 1.1) // 2 * 2 + 2)
        self._files: Dict[Tuple, str] = {}

    @staticmethod
    def _align(sources: Dict[Tuple[str, bool], Image.Image], target_height: int) -> Dict:
        sizes = {image.size for image in sources.values()}
        if len(sizes) == 1:
            # Drawn on one canvas (like the bundled otter): crop them all to
            # the same box, so the body stays put while the arms move.
            box = None
            for image in sources.values():
                bbox = image.getchannel("A").getbbox()
                if bbox:
                    box = bbox if box is None else (
                        min(box[0], bbox[0]), min(box[1], bbox[1]), max(box[2], bbox[2]), max(box[3], bbox[3])
                    )
            cropped = {key: image.crop(box) if box else image for key, image in sources.items()}
        else:
            # Separate drawings: line them up on the bottom of the body.
            trimmed = {}
            for key, image in sources.items():
                bbox = image.getchannel("A").getbbox()
                image = image.crop(bbox) if bbox else image
                scale = target_height / image.height
                trimmed[key] = image.resize(
                    (max(1, int(image.width * scale)), target_height), Image.LANCZOS
                )
            anchors = {key: _bottom_anchor(image) for key, image in trimmed.items()}
            left = max(anchors.values())
            right = max(image.width - anchors[key] for key, image in trimmed.items())
            width = int(left + right) + 1
            cropped = {}
            for key, image in trimmed.items():
                canvas = Image.new("RGBA", (width, target_height), (0, 0, 0, 0))
                canvas.alpha_composite(image, (int(left - anchors[key]), 0))
                cropped[key] = canvas
            return cropped
        sample = next(iter(cropped.values()))
        scale = target_height / sample.height
        size = (max(1, int(sample.width * scale)), target_height)
        return {key: image.resize(size, Image.LANCZOS) for key, image in cropped.items()}

    def _image(self, expression: str, talking: bool) -> Image.Image:
        return self.images.get((expression, talking)) or self.images[(expression, False)]

    def hidden(self) -> str:
        key = ("hidden",)
        if key not in self._files:
            path = os.path.join(self.work_dir, "hidden.png")
            Image.new("RGBA", self.canvas, (0, 0, 0, 0)).save(path, compress_level=1)
            self._files[key] = path
        return self._files[key]

    def frame(self, expression: str, talking: bool, bounce: Optional[int] = None) -> str:
        talking = talking and (expression, True) in self.images
        key = (expression, talking, bounce)
        if key in self._files:
            return self._files[key]
        image = self._image(expression, talking)
        scale_x, scale_y, lift = bounce_shape(bounce) if bounce is not None else (1.0, 1.0, 0.0)
        width, height = image.size
        if bounce is not None:
            image = image.resize((max(1, round(width * scale_x)), max(1, round(height * scale_y))), Image.BILINEAR)
        canvas = Image.new("RGBA", self.canvas, (0, 0, 0, 0))
        x = (self.canvas[0] - image.width) // 2
        y = self.canvas[1] - image.height - int(lift * height)
        canvas.alpha_composite(image, (x, max(0, y)))
        step = f"b{bounce:02d}" if bounce is not None else "still"
        name = f"{_file_stem(expression)}-{'talk' if talking else 'idle'}-{step}.png"
        path = os.path.join(self.work_dir, name)
        canvas.save(path, compress_level=1)
        self._files[key] = path
        return path


# ---------------------------------------------------------------------------
# Per-segment track
# ---------------------------------------------------------------------------


def _ease_out_back(x: str) -> str:
    return f"(1+2.70158*pow({x}-1,3)+1.70158*pow({x}-1,2))"


def _ease_in_back(x: str) -> str:
    return f"(2.70158*pow({x},3)-1.70158*pow({x},2))"


def position_offset(windows: Sequence[HostWindow]) -> str:
    """ffmpeg expression in [~-0.1, 1]: how far below its spot the host is."""
    terms = []
    for window in windows:
        if window.enter:
            x = f"clip((t-{window.start:.3f})/{ENTER_SECONDS},0,1)"
            terms.append(
                f"between(t,{window.start:.3f},{window.start + ENTER_SECONDS:.3f})*(1-{_ease_out_back(x)})"
            )
        if window.exit:
            begin = window.end - EXIT_SECONDS
            x = f"clip((t-{begin:.3f})/{EXIT_SECONDS},0,1)"
            terms.append(f"between(t,{begin:.3f},{window.end:.3f})*{_ease_in_back(x)}")
    return "+".join(terms) or "0"


def segment_frames(
    host: HostSegment, renderer: HostRenderer, frames: int, mouth: List[Tuple[bool, int]]
) -> List[str]:
    """The PNG to show on every frame of a segment."""
    talking = []
    for is_open, count in mouth:
        talking += [is_open] * count
    talking += [False] * max(0, frames - len(talking))
    out = []
    cue_index = -1
    for frame in range(frames):
        t = frame / FPS
        if host.visible(t) is None:
            out.append(renderer.hidden())
            continue
        while cue_index + 1 < len(host.cues) and host.cues[cue_index + 1].time <= t + 1e-6:
            cue_index += 1
        if cue_index < 0:
            out.append(renderer.hidden())
            continue
        cue = host.cues[cue_index]
        bounce = None
        if cue.bounce:
            since = frame - round(cue.time * FPS)
            if 0 <= since < BOUNCE_FRAMES:
                bounce = since
        out.append(renderer.frame(cue.expression, talking[frame], bounce))
    return out


def write_concat(files: List[str], path: str) -> str:
    """Run-length encode per-frame files into an ffconcat list."""
    lines = ["ffconcat version 1.0"]
    index = 0
    while index < len(files):
        run = 1
        while index + run < len(files) and files[index + run] == files[index]:
            run += 1
        lines.append(f"file '{os.path.basename(files[index])}'")
        lines.append(f"duration {run / FPS:.6f}")
        index += run
    if files:
        lines.append(f"file '{os.path.basename(files[-1])}'")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return path
