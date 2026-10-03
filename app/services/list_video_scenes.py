"""
Explainer scenes: full-screen minimalist graphics between the footage.

Every so often the footage gives way to a drawn scene on a paper-textured
canvas in the channel colour, the way explainer channels make one idea
crystal clear and win the viewer's attention back:

* statement - the host alone on the brand colour with a short punchline;
* stat      - a number from the narration as a pie that fills up, or a big
              number that counts up;
* sequence  - two to four things that pop in from left to right as each one
              is named, optionally stamped with a red cross or a green tick
              ("it is not X, nor Y");
* compare   - two situations side by side with a hand-drawn divider;
* diagram   - one central idea and the factors around it, with arrows drawn
              towards it as each factor is named.

The LLM director writes the scenes in the edit plan, anchored to exact words.
Pictures are OpenMoji icons (thick outlines and flat colours, like a doodle)
or doodle-style AI illustrations; labels use a hand-lettered font. Animations
(pop, stamp, arrow drawing, pie sweep, count-up) are short PNG sequences that
ffmpeg holds on their last frame, and the whole scene slides in and out.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.services import list_video_fx as fx
from app.utils import utils

FPS = 30
SCENE_TYPES = ("statement", "stat", "sequence", "compare", "diagram")
SCENE_MARKS = ("cross", "check")
SLIDE_SECONDS = 0.35
MAX_SCENE_SECONDS = 12.0
MIN_SCENE_SECONDS = 2.2
INK = (27, 27, 47)
WHITE = (255, 255, 255)
TEAL = (42, 157, 143)
CROSS = (226, 44, 58)
CHECK = (38, 166, 91)
HAND_FONT = "PatrickHand-Regular.ttf"
_SFX = {"pop": 0.5, "stamp": 0.8, "scribble": 0.45, "whoosh": 0.5, "tick": 0.6}


@dataclass
class SceneItem:
    label: str = ""
    icon: str = ""
    draw: str = ""
    mark: str = ""
    at: str = ""
    time: float = 0.0


@dataclass
class Scene:
    type: str
    start: float
    end: float
    items: List[SceneItem] = field(default_factory=list)
    text: str = ""
    value: float = 0.0
    unit: str = ""
    chart: str = "number"
    expression: str = ""
    center: Optional[SceneItem] = None
    exit: bool = True
    vertical: bool = False  # enter from below and leave upwards instead of sideways


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _item(data: dict) -> SceneItem:
    return SceneItem(
        label=str(data.get("label") or ""),
        icon=str(data.get("icon") or ""),
        draw=str(data.get("draw") or ""),
        mark=str(data.get("mark") or "") if data.get("mark") in SCENE_MARKS else "",
        at=str(data.get("at") or ""),
    )


def _snap(pauses: Sequence[float], target: float, low: float, high: float) -> float:
    inside = [p for p in pauses if low <= p <= high]
    return min(inside, key=lambda p: abs(p - target)) if inside else target


def time_scenes(
    specs: List[dict],
    locate: Callable[[str], Optional[float]],
    pauses: Sequence[float],
    duration: float,
    blocked: Sequence[Tuple[float, float]] = (),
) -> List[Scene]:
    """Turn plan scenes into timed scenes that never overlap each other.

    ``locate`` returns when an anchor is spoken (or None); ``blocked`` are
    intervals reserved for something else, such as the subscribe animation.
    """
    timed: List[Scene] = []
    for spec in specs:
        kind = spec.get("type")
        if kind not in SCENE_TYPES:
            continue
        items: List[SceneItem] = []
        if kind in ("statement", "stat"):
            time = locate(spec.get("at", ""))
            if time is None:
                continue
            times = [time]
        else:
            for data in spec.get("items") or []:
                item = _item(data)
                time = locate(item.at)
                if time is not None:
                    item.time = time
                    items.append(item)
            items.sort(key=lambda i: i.time)
            # Items said in the same breath would pop in on top of each other.
            spaced: List[SceneItem] = []
            for item in items:
                if spaced and item.time - spaced[-1].time < 0.5:
                    item.time = spaced[-1].time + 0.5
                spaced.append(item)
            items = spaced
            needed = 2 if kind != "diagram" else 2
            if len(items) < needed:
                continue
            times = [i.time for i in items]

        start = max(0.0, min(times) - 0.45)
        if start < 0.5:
            start = 0.0
        last = max(times)
        if kind == "statement":
            end = _snap(pauses, last + 2.6, last + 1.6, last + 5.0) + 0.3
        else:
            tail = 3.0 if kind == "stat" else 2.4
            end = _snap(pauses, last + tail, last + 1.8, last + 3.8) + 0.25
        end = min(end, start + MAX_SCENE_SECONDS, duration)
        scene = Scene(
            type=kind,
            start=start,
            end=end,
            items=[i for i in items if i.time < end - 1.2],
            text=str(spec.get("text") or ""),
            value=float(spec.get("value") or 0),
            unit=str(spec.get("unit") or ""),
            chart=spec.get("chart") if spec.get("chart") in ("pie", "number") else "number",
            expression=str(spec.get("expression") or ""),
            center=_item(spec["center"]) if isinstance(spec.get("center"), dict) else None,
        )
        if scene.type == "stat":
            scene.center = SceneItem(icon=str(spec.get("icon") or ""), label=str(spec.get("label") or ""))
        if scene.type in ("sequence", "compare", "diagram") and len(scene.items) < 2:
            continue
        if duration - scene.end < 1.2:
            scene.end, scene.exit = duration, False
        if scene.end - scene.start < MIN_SCENE_SECONDS:
            continue
        for item in scene.items:
            item.time = max(item.time, scene.start + SLIDE_SECONDS + 0.05)
        timed.append(scene)

    kept: List[Scene] = []
    for scene in sorted(timed, key=lambda s: s.start):
        if kept and scene.start < kept[-1].end + 0.6:
            continue
        clash = [(a, b) for a, b in blocked if scene.start < b and scene.end > a]
        if clash:
            # Make room for the blocked moment rather than losing the scene.
            first = min(a for a, _ in clash)
            if first - 0.2 - scene.start < MIN_SCENE_SECONDS or first <= scene.start:
                continue
            scene.end, scene.exit = first - 0.2, True
            scene.items = [i for i in scene.items if i.time < scene.end - 1.2]
            if scene.type in ("sequence", "compare", "diagram") and len(scene.items) < 2:
                continue
        scene.vertical = len(kept) % 2 == 1
        kept.append(scene)
    return kept


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _ease_out_back(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 1 + 2.70158 * (p - 1) ** 3 + 1.70158 * (p - 1) ** 2


def _ease_out_cubic(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


def hand_font_path(texts: Sequence[str] = ()) -> str:
    from app.services import video

    path = os.path.join(utils.font_dir(), HAND_FONT)
    if os.path.isfile(path) and video.subtitle_font_supports_text(path, " ".join(texts)):
        return path
    return ""


def paper(size: Tuple[int, int], color: Tuple[int, int, int], seed: int = 7) -> Image.Image:
    """A flat colour with a little grain and a soft vignette, like a printed page."""
    width, height = size
    rng = np.random.default_rng(seed)
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:] = color
    grain = rng.normal(0, 3.2, (height // 2 + 1, width // 2 + 1)).astype(np.float32)
    grain = np.kron(grain, np.ones((2, 2), dtype=np.float32))[:height, :width]
    base += grain[:, :, None]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.sqrt(((x - width / 2) / (width / 2)) ** 2 + ((y - height / 2) / (height / 2)) ** 2)
    base *= (1 - 0.07 * np.clip(distance - 0.35, 0, None) ** 1.5)[:, :, None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def die_cut(image: Image.Image, outline: int, shadow: bool = True) -> Image.Image:
    """White sticker outline (and a soft shadow) around a transparent picture."""
    pad = outline + 2
    canvas = Image.new("RGBA", (image.width + pad * 2, image.height + pad * 2), (0, 0, 0, 0))
    canvas.alpha_composite(image, (pad, pad))
    alpha = canvas.getchannel("A").point(lambda a: 255 if a > 40 else 0)
    grown = alpha.filter(ImageFilter.MaxFilter(outline * 2 + 1)).filter(ImageFilter.GaussianBlur(1))
    sticker = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    sticker.putalpha(grown)
    sticker.alpha_composite(canvas)
    if not shadow:
        return sticker
    blur = max(2, outline)
    out = Image.new("RGBA", (sticker.width + blur * 4, sticker.height + blur * 4), (0, 0, 0, 0))
    shade = Image.new("RGBA", out.size, (0, 0, 0, 0))
    shade.paste((0, 0, 0, 255), (blur * 2, blur * 2 + blur // 2), sticker.getchannel("A").point(lambda a: a * 70 // 255))
    out.alpha_composite(shade.filter(ImageFilter.GaussianBlur(blur)))
    out.alpha_composite(sticker, (blur * 2, blur * 2))
    return out


def prepare_picture(path: str) -> Optional[Image.Image]:
    """A transparent cut-out of an icon or an illustration drawn on white."""
    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
    except Exception:
        return None
    if not fx.has_transparency(image):
        image = fx.remove_flat_background(image) or image
    bbox = image.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
    return image.crop(bbox) if bbox else image


class _Text:
    def __init__(self, font_path: str):
        self.font_path = font_path

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.font_path, max(8, int(size)))

    def _wrap(self, text: str, font, max_width: int, max_lines: int) -> Optional[List[str]]:
        lines: List[str] = []
        for word in text.split():
            if lines and font.getlength(f"{lines[-1]} {word}") <= max_width:
                lines[-1] = f"{lines[-1]} {word}"
            else:
                lines.append(word)
        if len(lines) > max_lines or any(font.getlength(line) > max_width for line in lines):
            return None
        return lines

    def render(
        self, text: str, size: int, max_width: int, color=INK, max_lines: int = 2, outline: int = 0
    ) -> Image.Image:
        """Hand-lettered text, shrunk until it fits ``max_width`` in ``max_lines``."""
        text = " ".join((text or "").split())
        while True:
            font = self.font(size)
            lines = self._wrap(text, font, max_width, max_lines)
            if lines or size <= 14:
                break
            size = int(size * 0.9)
        lines = lines or [text]
        weight = max(1, size // 28)  # thickens the strokes of the hand font
        stroke = weight + outline
        line_height = int(size * 1.02)
        widths = [int(font.getlength(line)) for line in lines]
        width = max(widths) + stroke * 2 + 4
        height = line_height * len(lines) + int(size * 0.3) + stroke * 2
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        for number, line in enumerate(lines):
            x = (width - widths[number]) / 2
            y = stroke + number * line_height
            if outline:
                draw.text((x, y), line, font=font, fill=WHITE + (255,), stroke_width=stroke, stroke_fill=WHITE + (255,))
            draw.text((x, y), line, font=font, fill=color + (255,), stroke_width=weight, stroke_fill=color + (255,))
        bbox = image.getbbox()
        return image.crop(bbox) if bbox else image


def _supersampled(size: Tuple[int, int], paint: Callable[[ImageDraw.ImageDraw, float], None], scale: int = 3) -> Image.Image:
    big = Image.new("RGBA", (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
    paint(ImageDraw.Draw(big), scale)
    return big.resize(size, Image.LANCZOS)


def draw_mark(kind: str, size: int) -> Image.Image:
    """A thick hand-drawn red cross or green tick with a white outline."""
    color = CROSS if kind == "cross" else CHECK
    width = max(4, size // 7)

    def paint(draw: ImageDraw.ImageDraw, s: float) -> None:
        if kind == "cross":
            strokes = [[(0.14, 0.12), (0.86, 0.88)], [(0.84, 0.1), (0.16, 0.9)]]
        else:
            strokes = [[(0.1, 0.55), (0.38, 0.84), (0.92, 0.16)]]
        for fill, extra in ((WHITE, width * 0.7), (color, 0)):
            for stroke in strokes:
                points = [(x * size * s, y * size * s) for x, y in stroke]
                w = int((width + extra) * s)
                draw.line(points, fill=fill + (255,), width=w, joint="curve")
                for x, y in (points[0], points[-1]):
                    draw.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=fill + (255,))

    margin = width
    image = _supersampled((size + margin * 2, size + margin * 2), lambda d, s: paint(_Shift(d, margin * s), s))
    return image


class _Shift:
    """Wrap ImageDraw so every coordinate is offset by a margin."""

    def __init__(self, draw: ImageDraw.ImageDraw, offset: float):
        self.draw, self.offset = draw, offset

    def _move(self, points):
        return [(x + self.offset, y + self.offset) for x, y in points]

    def line(self, points, **kwargs):
        self.draw.line(self._move(points), **kwargs)

    def ellipse(self, box, **kwargs):
        x0, y0, x1, y1 = box
        o = self.offset
        self.draw.ellipse((x0 + o, y0 + o, x1 + o, y1 + o), **kwargs)


def _bezier(p0, p1, bend: float, steps: int = 48) -> List[Tuple[float, float]]:
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bend, my + dx * bend
    points = []
    for i in range(steps + 1):
        t = i / steps
        points.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1, (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1))
    return points


def arrow_frames(
    p0, p1, width: int, frames: int = 10, bend: float = 0.16, head: bool = True
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """Frames of a hand-drawn arrow (or plain stroke) drawn from p0 to p1; returns (frames, top-left)."""
    points = _bezier(p0, p1, bend)
    head_length = width * 4.2
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = int(head_length + width * 2)
    left, top = int(min(xs)) - pad, int(min(ys)) - pad
    size = (int(max(xs)) + pad - left, int(max(ys)) + pad - top)
    local = [(x - left, y - top) for x, y in points]
    (ax, ay), (bx, by) = local[-4], local[-1]
    angle = math.atan2(by - ay, bx - ax)
    wing = 0.55
    head_points = [
        (bx - head_length * math.cos(angle - wing), by - head_length * math.sin(angle - wing)),
        (bx, by),
        (bx - head_length * math.cos(angle + wing), by - head_length * math.sin(angle + wing)),
    ]
    out = []
    for frame in range(1, frames + 1):
        progress = _ease_out_cubic(frame / frames)
        count = max(2, int(len(local) * progress))

        def paint(draw: ImageDraw.ImageDraw, s: float, count=count, frame=frame) -> None:
            shown = [(x * s, y * s) for x, y in local[:count]]
            draw.line(shown, fill=INK + (255,), width=int(width * s), joint="curve")
            if head and frame == frames:
                draw.line([(x * s, y * s) for x, y in head_points], fill=INK + (255,), width=int(width * s), joint="curve")
                for x, y in (head_points[0], head_points[2]):
                    r = width * s / 2
                    draw.ellipse((x * s - r, y * s - r, x * s + r, y * s + r), fill=INK + (255,))
            r = width * s / 2
            x, y = shown[0]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=INK + (255,))

        out.append(_supersampled(size, paint, scale=2))
    return out, (left, top)


def pie_frames(value: float, radius: int, line: int, frames: int = 18) -> List[Image.Image]:
    """A pie chart whose slice sweeps from 0 to ``value`` percent."""
    size = (radius * 2 + line * 4, radius * 2 + line * 4)
    value = min(100.0, max(0.0, value))
    out = []
    for frame in range(1, frames + 1):
        share = value * _ease_out_cubic(frame / frames)

        def paint(draw: ImageDraw.ImageDraw, s: float, share=share) -> None:
            c = size[0] * s / 2
            r = radius * s
            box = (c - r, c - r, c + r, c + r)
            draw.ellipse(box, fill=WHITE + (255,))
            if share > 0.2:
                draw.pieslice(box, -90, -90 + 360 * share / 100, fill=TEAL + (255,))
                for angle in (-90, -90 + 360 * share / 100):
                    a = math.radians(angle)
                    draw.line([(c, c), (c + r * math.cos(a), c + r * math.sin(a))], fill=INK + (255,), width=int(line * 0.8 * s))
            draw.ellipse(box, outline=INK + (255,), width=int(line * s))

        out.append(_supersampled(size, paint, scale=2))
    return out


def pop_frames(image: Image.Image, frames: int = 9, start_scale: float = 0.35) -> List[Image.Image]:
    """``image`` popping in: grows past its size and settles."""
    pad_w, pad_h = int(image.width * 0.1) + 2, int(image.height * 0.1) + 2
    size = (image.width + pad_w * 2, image.height + pad_h * 2)
    out = []
    for frame in range(1, frames + 1):
        p = frame / frames
        scale = start_scale + (1 - start_scale) * _ease_out_back(p)
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        w, h = max(1, int(image.width * scale)), max(1, int(image.height * scale))
        piece = image.resize((w, h), Image.BILINEAR if frame < frames else Image.LANCZOS)
        if p < 0.35:
            piece.putalpha(piece.getchannel("A").point(lambda a, k=p / 0.35: int(a * k)))
        canvas.alpha_composite(piece, ((size[0] - w) // 2, (size[1] - h) // 2))
        out.append(canvas)
    return out


def stamp_frames(image: Image.Image, frames: int = 6) -> List[Image.Image]:
    """A stamp landing: from big and faint to its size."""
    size = (int(image.width * 1.8) + 2, int(image.height * 1.8) + 2)
    out = []
    for frame in range(1, frames + 1):
        p = _ease_out_cubic(frame / frames)
        scale = 1.75 - 0.75 * p
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        w, h = int(image.width * scale), int(image.height * scale)
        piece = image.resize((w, h), Image.BILINEAR if frame < frames else Image.LANCZOS)
        piece.putalpha(piece.getchannel("A").point(lambda a, k=min(1.0, 0.25 + p): int(a * k)))
        canvas.alpha_composite(piece, ((size[0] - w) // 2, (size[1] - h) // 2))
        out.append(canvas)
    return out


# ---------------------------------------------------------------------------
# Scene layouts
# ---------------------------------------------------------------------------


class SceneRenderer:
    """Turns timed scenes into ffmpeg overlays and sound effects."""

    def __init__(
        self,
        theme: fx.Theme,
        work_dir: str,
        canvas_color: Tuple[int, int, int],
        picture: Callable[[SceneItem], Optional[Image.Image]],
        host_still: Optional[Callable[[str, int], Image.Image]] = None,
        sfx: Optional[Dict[str, str]] = None,
        font_path: str = "",
    ):
        self.theme = theme
        self.work_dir = os.path.join(work_dir, "scenes")
        os.makedirs(self.work_dir, exist_ok=True)
        self.canvas_color = canvas_color
        self.picture = picture
        self.host_still = host_still
        self.sfx = sfx or {}
        self.text = _Text(font_path or theme.font_path)
        self._backgrounds: Dict[Tuple[int, int, int], str] = {}

    # -- helpers -------------------------------------------------------------

    def _background(self, color: Tuple[int, int, int]) -> str:
        if color not in self._backgrounds:
            path = os.path.join(self.work_dir, f"paper-{color[0]:02x}{color[1]:02x}{color[2]:02x}.png")
            paper((self.theme.width, self.theme.height), color).save(path)
            self._backgrounds[color] = path
        return self._backgrounds[color]

    def _sound(self, sounds: list, time: float, name: str) -> None:
        if name in self.sfx:
            sounds.append((time, self.sfx[name], _SFX.get(name, 0.6)))

    @staticmethod
    def _slide(scene: Scene, axis: str = "x") -> str:
        """Offset along ``axis`` while the scene slides in and out (0 otherwise)."""
        if (axis == "y") != scene.vertical:
            return "0"
        size = "H" if scene.vertical else "W"
        s, e, d = scene.start, scene.end, SLIDE_SECONDS
        enter = f"{size}*pow(1-clip((t-{s:.3f})/{d},0,1),3)"
        if not scene.exit:
            return enter
        return f"{enter}-{size}*pow(clip((t-{e - d:.3f})/{d},0,1),3)"

    def _frames(
        self, frames: List[Image.Image], folder: str, name: str, x: float, y: float, start: float, scene: Scene
    ) -> fx.Overlay:
        """Overlay for an animation whose canvas top-left lands at (x, y)."""
        os.makedirs(folder, exist_ok=True)
        for number, frame in enumerate(frames):
            frame.save(os.path.join(folder, f"{name}_{number:03d}.png"), compress_level=1)
        return fx.Overlay(
            os.path.join(folder, f"{name}_%03d.png"),
            x=f"{int(x)}+{self._slide(scene)}",
            y=f"{int(y)}+{self._slide(scene, 'y')}",
            start=start,
            end=scene.end,
            mode="frames",
            hold=True,
        )

    def _centered(self, frames, folder, name, cx, cy, start, scene) -> fx.Overlay:
        w, h = frames[-1].size
        return self._frames(frames, folder, name, cx - w / 2, cy - h / 2, start, scene)

    def _picture(self, item: SceneItem, box: int) -> Optional[Image.Image]:
        image = self.picture(item)
        if image is None:
            return None
        image = image.copy()
        image.thumbnail((box, box), Image.LANCZOS)
        return die_cut(image, self.theme.px(8))

    def _card(
        self, item: SceneItem, box: int, label_size: int, label_width: int, label_on_top: bool, beside: bool = False
    ) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
        """Picture plus hand-lettered label as one element, and where the picture sits in it.

        The label goes above or below the picture, or to its right with ``beside``.
        """
        picture = self._picture(item, box)
        label = self.text.render(item.label.upper(), label_size, label_width) if item.label else None
        if picture is None and label is not None:
            label = self.text.render(item.label.upper(), int(label_size * 1.5), label_width, max_lines=3)
        parts = [p for p in ((label, picture) if label_on_top and not beside else (picture, label)) if p is not None]
        if not parts:
            parts = [Image.new("RGBA", (box, box), (0, 0, 0, 0))]
        gap = self.theme.px(10 if not beside else 24)
        if beside:
            width = sum(p.width for p in parts) + gap * (len(parts) - 1)
            height = max(p.height for p in parts)
        else:
            width = max(p.width for p in parts)
            height = sum(p.height for p in parts) + gap * (len(parts) - 1)
        card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        offset = 0
        where = None
        for part in parts:
            if beside:
                x, y = offset, (height - part.height) // 2
                offset += part.width + gap
            else:
                x, y = (width - part.width) // 2, offset
                offset += part.height + gap
            card.alpha_composite(part, (x, y))
            if part is picture:
                where = (x, y, x + part.width, y + part.height)
        return card, where

    # -- scenes ----------------------------------------------------------------

    def build(self, scene: Scene, key: str) -> Tuple[List[fx.Overlay], List[Tuple[float, str, float]]]:
        folder = os.path.join(self.work_dir, key)
        color = self.theme.accent if scene.type == "statement" else self.canvas_color
        overlays = [
            fx.Overlay(
                self._background(color), x=self._slide(scene), y=self._slide(scene, "y"),
                start=scene.start, end=scene.end,
            )
        ]
        sounds: List[Tuple[float, str, float]] = []
        self._sound(sounds, scene.start - 0.1, "whoosh")
        if scene.exit:
            self._sound(sounds, scene.end - SLIDE_SECONDS - 0.05, "whoosh")
        builder = getattr(self, f"_build_{scene.type}")
        builder(scene, folder, overlays, sounds)
        return overlays, sounds

    def _build_statement(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS * 0.6
        text_bottom = 0
        if scene.text:
            text = self.text.render(scene.text.upper(), int(H * 0.12), int(W * 0.9), max_lines=2)
            frames = pop_frames(text)
            y = H * (0.05 if theme.portrait else 0.045)
            overlays.append(self._frames(frames, folder, "text", (W - frames[-1].width) / 2, y, appear + 0.15, scene))
            text_bottom = y + frames[-1].height
            self._sound(sounds, appear + 0.15, "pop")
        if self.host_still is not None:
            # The host rises from the bottom edge (a tenth of it below the
            # frame), as tall as the space under the text allows.
            room = (H - text_bottom - theme.px(20)) / 0.9
            height = int(min(H * (0.55 if theme.portrait else 0.85), room))
            pose = die_cut(self.host_still(scene.expression, height), theme.px(6), shadow=False)
            frames = pop_frames(pose, frames=10, start_scale=0.55)
            w, h = frames[-1].size
            pad_y = (h - pose.height) / 2
            top = H + 0.1 * pose.height - pose.height - pad_y
            overlays.append(self._frames(frames, folder, "host", (W - w) / 2, top, appear, scene))

    def _build_stat(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS + 0.05
        value = scene.value
        number = f"{value:g}{scene.unit}"
        label = scene.center.label if scene.center else ""
        if theme.portrait:
            chart_center, number_center, label_y = (W * 0.5, H * 0.36), (W * 0.5, H * 0.64), H * 0.73
        elif scene.chart == "pie":
            chart_center, number_center, label_y = (W * 0.32, H * 0.53), (W * 0.68, H * 0.42), H * 0.58
        else:
            chart_center, number_center, label_y = (W * 0.26, H * 0.52), (W * 0.62, H * 0.42), H * 0.58
        count = 18
        if scene.chart == "pie":
            radius = int(H * (0.25 if not theme.portrait else 0.19))
            frames = pie_frames(value, radius, theme.px(9), frames=count)
            overlays.append(self._centered(frames, folder, "pie", *chart_center, appear, scene))
        elif scene.center is not None:
            picture = self._picture(scene.center, int(H * 0.38))
            if picture is not None:
                overlays.append(self._centered(pop_frames(picture), folder, "icon", *chart_center, appear, scene))
        size = int(H * 0.24)
        widest = self.text.render(number, size, int(W * 0.5), max_lines=1)
        digits = []
        for frame in range(1, count + 1):
            shown = value * _ease_out_cubic(frame / count)
            text = f"{round(shown):g}{scene.unit}" if float(value).is_integer() else f"{shown:.1f}{scene.unit}"
            image = self.text.render(text, size, int(W * 0.5), max_lines=1)
            canvas = Image.new("RGBA", (widest.width + 8, widest.height + 8), (0, 0, 0, 0))
            canvas.alpha_composite(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
            digits.append(canvas)
        overlays.append(self._centered(digits, folder, "number", *number_center, appear + 0.1, scene))
        for k in range(0, count, 6):
            self._sound(sounds, appear + 0.1 + k / FPS, "tick")
        if label:
            image = self.text.render(label.upper(), int(H * 0.085), int(W * (0.5 if not theme.portrait else 0.86)))
            frames = pop_frames(image)
            overlays.append(self._frames(frames, folder, "label", number_center[0] - frames[-1].width / 2, label_y, appear + 0.7, scene))
            self._sound(sounds, appear + 0.7, "pop")

    def _row_layout(self, count: int) -> List[Tuple[float, float]]:
        W, H = self.theme.width, self.theme.height
        if self.theme.portrait:
            return [(W * 0.5, H * (0.2 + 0.62 * (i + 0.5) / count)) for i in range(count)]
        margin = W * 0.06
        span = W - margin * 2
        return [(margin + span * (i + 0.5) / count, H * 0.52) for i in range(count)]

    def _build_sequence(self, scene, folder, overlays, sounds, divider: bool = False) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        items = scene.items[:4]
        centers = self._row_layout(len(items))
        if theme.portrait:
            box = int(min(H * 0.16, H * 0.5 / len(items)))
            label_width = int(W * 0.5)
        else:
            box = int(min(H * 0.42, W * 0.78 / len(items)))
            label_width = int(W * 0.9 / len(items))
        label_size = int(H * 0.075)
        if divider and len(items) == 2 and not theme.portrait:
            frames, (left, top) = arrow_frames(
                (W / 2, H * 0.14), (W / 2 + 1, H * 0.88), theme.px(7), frames=8, bend=0.02, head=False
            )
            overlays.append(self._frames(frames, folder, "divider", left, top, scene.start + SLIDE_SECONDS, scene))
            self._sound(sounds, scene.start + SLIDE_SECONDS, "scribble")
        for number, (item, (cx, cy)) in enumerate(zip(items, centers)):
            card, where = self._card(item, box, label_size, label_width, label_on_top=True, beside=theme.portrait)
            frames = pop_frames(card)
            # Line the pictures up, whatever the label sizes.
            card_cy = cy + card.height / 2 - (where[1] + where[3]) / 2 if where and not theme.portrait else cy
            overlays.append(self._centered(frames, folder, f"item{number}", cx, card_cy, item.time, scene))
            self._sound(sounds, item.time, "pop")
            if item.mark:
                if where is not None:
                    x0, y0, x1, y1 = where
                    mark_cx = cx - card.width / 2 + (x0 + x1) / 2
                    mark_cy = card_cy - card.height / 2 + (y0 + y1) / 2
                    size = int(min(box, max(x1 - x0, y1 - y0)) * 0.8)
                else:
                    mark_cx, mark_cy, size = cx, cy, int(box * 0.6)
                when = item.time + 0.45
                overlays.append(
                    self._centered(stamp_frames(draw_mark(item.mark, size)), folder, f"mark{number}", mark_cx, mark_cy, when, scene)
                )
                self._sound(sounds, when, "stamp")

    def _build_compare(self, scene, folder, overlays, sounds) -> None:
        scene.items = scene.items[:2]
        self._build_sequence(scene, folder, overlays, sounds, divider=True)

    def _build_diagram(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        center = (W * 0.5, H * 0.53)
        appear = scene.start + SLIDE_SECONDS + 0.05
        hub = scene.center or SceneItem(label="")
        hub_box = int(H * (0.34 if not theme.portrait else 0.22))
        card, _ = self._card(hub, hub_box, int(H * 0.06), int(W * 0.3), label_on_top=False)
        overlays.append(self._centered(pop_frames(card), folder, "hub", *center, appear, scene))
        self._sound(sounds, appear, "pop")
        hub_radius = max(card.width, card.height) / 2 + theme.px(16)
        items = scene.items[:5]
        rx, ry = (W * 0.36, H * 0.33) if not theme.portrait else (W * 0.3, H * 0.33)
        angles = {
            2: [180, 0], 3: [200, 340, 270], 4: [215, 325, 145, 35], 5: [200, 340, 270, 135, 45],
        }[max(2, len(items))]
        for number, (item, angle) in enumerate(zip(items, angles)):
            a = math.radians(angle)
            nx, ny = center[0] + rx * math.cos(a), center[1] + ry * math.sin(a)
            node_box = int(H * (0.16 if not theme.portrait else 0.08))
            node_label = int(H * (0.064 if not theme.portrait else 0.034))
            node, _ = self._card(item, node_box, node_label, int(W * (0.26 if not theme.portrait else 0.4)), label_on_top=False)
            overlays.append(self._centered(pop_frames(node), folder, f"node{number}", nx, ny, item.time, scene))
            self._sound(sounds, item.time, "pop")
            # Arrow from the node towards the hub, stopping short of both.
            dx, dy = center[0] - nx, center[1] - ny
            length = math.hypot(dx, dy) or 1
            ux, uy = dx / length, dy / length
            node_radius = max(node.width, node.height) / 2 * 0.75
            p0 = (nx + ux * node_radius, ny + uy * node_radius)
            p1 = (center[0] - ux * hub_radius, center[1] - uy * hub_radius)
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < theme.px(40):
                continue
            frames, (left, top) = arrow_frames(p0, p1, theme.px(7), bend=0.12 if number % 2 else -0.12)
            when = item.time + 0.25
            overlays.append(self._frames(frames, folder, f"arrow{number}", left, top, when, scene))
            self._sound(sounds, when, "scribble")
