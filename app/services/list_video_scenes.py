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

The doodle look strings shots together like an animatic: full-screen drawn
frames with a slow camera move that dissolve into each other, compositions
whose drawings appear as if drawn, real video clips in a taped ink frame and
short comic reactions (``clip`` and ``meme``).

The LLM director writes the scenes in the edit plan, anchored to exact words.
Pictures are OpenMoji icons (thick outlines and flat colours, like a doodle)
or doodle-style AI illustrations; labels use a hand-lettered font. Animations
(pop, stamp, arrow drawing, pie sweep, count-up) are short PNG sequences that
ffmpeg holds on their last frame, and the whole scene slides in and out.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.services import list_video_fx as fx
from app.utils import utils

FPS = 30
SCENE_TYPES = (
    "statement", "stat", "sequence", "compare", "diagram", "figure", "zoom", "story",
    "steps", "bars", "grid", "formula", "timeline", "gauge", "question",
    "definition", "equation", "annotate", "chain", "branch",
    "single", "speech", "illustration", "clip", "meme",
)
# Scenes whose elements are anchored one by one to the narration.
ITEM_SCENES = ("sequence", "compare", "diagram", "story", "steps", "bars", "formula", "timeline", "chain", "branch")
# Scenes shown at one anchor whose parts follow by themselves (or at their own anchors).
PART_SCENES = ("equation", "annotate", "speech")
PART_KEYS = {"equation": "terms", "annotate": "labels", "speech": "items"}
# Scenes rendered as a full-frame video clip (a slow zoom) instead of layers.
CLIP_SCENES = ("figure", "zoom", "illustration", "clip", "meme")
# Shots that cover the whole frame: the shot before them stays underneath while they fade in.
FULL_FRAME_SHOTS = CLIP_SCENES
CROSSFADE_SECONDS = 0.3
SCENE_MARKS = ("cross", "check")
SLIDE_SECONDS = 0.35
MAX_SCENE_SECONDS = 12.0
MIN_SCENE_SECONDS = 2.2
MAX_SCENE_DELAY = 4.5  # a scene said under a blocked moment (an opener) may wait this long for it
MAX_CHAIN_DELAY = 3.5  # and this long for the scene before it to leave
DELAYED_SECONDS = 4.5  # a delayed scene keeps up to this much of its length
ITEM_SECONDS = 1.8
INK = (27, 27, 47)
WHITE = (255, 255, 255)
TEAL = (42, 157, 143)
CROSS = (226, 44, 58)
CHECK = (38, 166, 91)
AMBER = (244, 180, 50)
GREY = (190, 190, 198)
HAND_FONT = "PatrickHand-Regular.ttf"
_SFX = {"pop": 0.5, "stamp": 0.8, "scribble": 0.45, "whoosh": 0.5, "tick": 0.6, "boom": 0.75}
# Seconds a scene stays after its last element, by type.
_TAILS = {"statement": 2.6, "stat": 3.0, "bars": 3.0, "grid": 3.4, "gauge": 3.2, "zoom": 3.4,
          "question": 2.8, "story": 2.8, "figure": 5.0, "definition": 5.2, "equation": 2.8, "annotate": 2.6,
          "chain": 2.8, "branch": 2.8, "single": 2.6, "speech": 3.0, "illustration": 3.5, "clip": 3.5, "meme": 2.2}
PART_FIRST = {"equation": 1.2, "annotate": 1.0, "speech": 0.0}  # seconds from the anchor to the first part
PART_STEP = 0.9  # parts without their own anchor follow each other this far apart


@dataclass
class SceneItem:
    label: str = ""
    icon: str = ""
    draw: str = ""
    mark: str = ""
    at: str = ""
    time: float = 0.0
    query: str = ""  # English search for a real picture of the item
    value: float = 0.0
    date: str = ""
    expression: str = ""
    role: str = ""  # "result" for the last term of a formula
    symbol: str = ""  # an equation term's letter ("V")
    unit: str = ""  # an equation term's unit ("voltios")
    link: str = ""  # chain: the verb written on the arrow to the next item
    point: Optional[Tuple[float, float]] = None  # annotate: where the part is (0-1, 0-1 of the picture)
    otter: bool = False  # draw the channel's otter in this picture
    pose: str = ""  # use one of the host's own poses as the picture


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
    total: int = 10
    operator: str = "+"
    cycle: bool = False
    low: str = ""
    high: str = ""
    direction: str = "in"
    look: str = "diagram"
    query: str = ""
    query_local: str = ""
    enter: bool = True  # False: already on screen when the segment starts (no slide in)
    number: int = 0  # section number of an opener
    symbol: str = ""  # definition: the quantity's symbol
    example: str = ""  # equation: a worked example ("12 V = 2 A × 6 Ω")
    still: bool = False  # never slides in or out (the shots of the doodle look)
    camera: str = ""  # illustration: "in", "out", "left" or "right" ("" picks one)
    follows: bool = False  # illustration: redrawn from the illustration before it (the next frame)
    frame: str = "card"  # clip: a framed video on the canvas ("card") or the whole screen ("full")
    mood: str = ""  # meme: the reaction ("shock", "laugh", ...)
    media: str = ""  # clip or meme: the video file to show
    fade_in: float = 0.0  # fades in over the shot before it
    overlap: float = 0.0  # stays this long under the next shot while that one fades in
    linger: float = 0.0  # fades out this long over the start of the next shot (drawn on the canvas)
    reprise: bool = False  # the shot before a reaction, shown again after it (same picture)


OPENER_SECONDS = 3.2  # how long a section opener holds the screen
CAMERA_MOVES = ("in", "out", "left", "right")


def opener_scene(
    title: str, number: int, spec: Optional[dict], pauses: Sequence[float], duration: float
) -> Optional[Scene]:
    """The card that opens a section: its number, its title and one picture of exactly that.

    ``spec`` is the plan's {"query", "query_local", "look", "icon", "draw"};
    without one the title itself is searched. None when the section is too short.
    """
    title = " ".join((title or "").split())
    if not title or duration < OPENER_SECONDS + 4.0:
        return None
    spec = spec if isinstance(spec, dict) else {}
    end = _snap(pauses, OPENER_SECONDS, OPENER_SECONDS - 0.6, OPENER_SECONDS + 1.0) + 0.25
    query = str(spec.get("query") or "").strip()
    query_local = str(spec.get("query_local") or "").strip() or (title if not query else "")
    center = SceneItem(
        label=title, icon=str(spec.get("icon") or ""), draw=str(spec.get("draw") or ""), query=query or title,
    )
    return Scene(
        type="opener", start=0.0, end=end, text=title, center=center, enter=False, number=number,
        look="photo" if spec.get("look") == "photo" else "diagram", query=query or title, query_local=query_local,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _item(data: dict) -> SceneItem:
    try:
        value = float(data.get("value") or 0)
    except (TypeError, ValueError):
        value = 0.0
    return SceneItem(
        label=str(data.get("label") or ""),
        icon=str(data.get("icon") or ""),
        draw=str(data.get("draw") or ""),
        mark=str(data.get("mark") or "") if data.get("mark") in SCENE_MARKS else "",
        at=str(data.get("at") or ""),
        query=str(data.get("query") or ""),
        value=value,
        date=str(data.get("date") or ""),
        expression=str(data.get("expression") or ""),
        role=str(data.get("role") or ""),
        symbol=str(data.get("symbol") or ""),
        unit=str(data.get("unit") or ""),
        link=str(data.get("link") or ""),
        otter=bool(data.get("otter")),
        pose=str(data.get("pose") or ""),
    )


def _snap(pauses: Sequence[float], target: float, low: float, high: float) -> float:
    inside = [p for p in pauses if low <= p <= high]
    return min(inside, key=lambda p: abs(p - target)) if inside else target


def _scene_from(spec: dict, kind: str, start: float, end: float, items: List[SceneItem]) -> Scene:
    def number(key, default=0.0):
        try:
            return float(spec.get(key) if spec.get(key) is not None else default)
        except (TypeError, ValueError):
            return default

    scene = Scene(
        type=kind,
        start=start,
        end=end,
        items=items,
        text=str(spec.get("text") or ""),
        value=number("value"),
        unit=str(spec.get("unit") or ""),
        chart=spec.get("chart") if spec.get("chart") in ("pie", "number") else "number",
        expression=str(spec.get("expression") or ""),
        center=_item(spec["center"]) if isinstance(spec.get("center"), dict) else None,
        total=int(min(100, max(2, number("total", 10)))),
        operator=str(spec.get("operator") or "+")[:2],
        cycle=bool(spec.get("cycle")),
        low=str(spec.get("low") or ""),
        high=str(spec.get("high") or ""),
        direction="out" if spec.get("direction") == "out" else "in",
        look="photo" if spec.get("look") == "photo" else "diagram",
        query=str(spec.get("query") or ""),
        query_local=str(spec.get("query_local") or ""),
        symbol=str(spec.get("symbol") or ""),
        example=str(spec.get("example") or ""),
    )
    if kind == "definition":
        scene.center = SceneItem(
            label=str(spec.get("term") or spec.get("label") or ""), icon=str(spec.get("icon") or ""),
            draw=str(spec.get("draw") or ""), query=str(spec.get("query") or ""), at=str(spec.get("at") or ""),
        )
    elif kind == "equation":
        scene.text = str(spec.get("formula") or spec.get("text") or "")
        scene.center = SceneItem(label=str(spec.get("name") or spec.get("label") or ""))
    elif kind == "annotate":
        scene.center = SceneItem(query=scene.query, at=str(spec.get("at") or ""))
    elif kind in ("single", "illustration", "clip", "meme"):
        item = spec.get("item") if isinstance(spec.get("item"), dict) else spec
        scene.center = _item(dict(item, at=spec.get("at") or ""))
        if kind == "illustration":
            scene.camera = str(spec.get("camera") or "") if spec.get("camera") in CAMERA_MOVES else ""
            scene.follows = bool(spec.get("continue"))
        elif kind == "clip":
            scene.frame = "full" if spec.get("frame") == "full" else "card"
        elif kind == "meme":
            scene.mood = str(spec.get("mood") or "")
            scene.text = str(spec.get("text") or "")
    if kind in ("stat", "grid", "gauge", "zoom", "figure"):
        item = spec.get("item") if isinstance(spec.get("item"), dict) else spec
        scene.center = scene.center or SceneItem(
            icon=str(item.get("icon") or ""), label=str(item.get("label") or spec.get("label") or ""),
            draw=str(item.get("draw") or ""), query=str(item.get("query") or ""), at=str(spec.get("at") or ""),
        )
    return scene


def _delay(scene: Scene, start: float, duration: float) -> bool:
    """Move a scene later, keeping its length; False when it no longer fits."""
    length = min(scene.end - scene.start, DELAYED_SECONDS)
    scene.start = start
    if scene.exit or scene.end < duration:
        scene.end = min(max(scene.end, start + length), duration, start + MAX_SCENE_SECONDS)
        if duration - scene.end < 1.2:
            scene.end, scene.exit = duration, False
    for number, item in enumerate(scene.items):
        item.time = max(item.time, start + SLIDE_SECONDS + 0.05 + 0.5 * number)
    scene.items = [i for i in scene.items if i.time < scene.end - ITEM_SECONDS]
    return scene.end - scene.start >= MIN_SCENE_SECONDS and not (scene.type in ITEM_SCENES and len(scene.items) < 2)


def _spec_times(
    spec: dict, kind: str, locate: Callable[[str], Optional[float]], min_items: int = 2
) -> Optional[Tuple[List[SceneItem], List[float]]]:
    """The timed elements of a plan scene and every moment it shows something; None if it cannot be placed."""
    items: List[SceneItem] = []
    if kind not in ITEM_SCENES:
        time = locate(spec.get("at", ""))
        if time is None:
            return None
        times = [time]
        if kind in PART_SCENES:
            # Each part appears when it is explained, or after the previous one.
            previous = time + PART_FIRST[kind] - PART_STEP
            for data in spec.get(PART_KEYS[kind]) or []:
                if not isinstance(data, dict):
                    continue
                item = _item(data)
                said = locate(item.at) if item.at else None
                item.time = max(previous + PART_STEP, said if said is not None and said < previous + 4.0 else 0.0)
                previous = item.time
                items.append(item)
            times += [i.time for i in items]
        return items, times
    source = list(spec.get("items") or spec.get("frames") or [])
    if kind == "formula" and isinstance(spec.get("result"), dict):
        source.append(dict(spec["result"], role="result"))
    for data in source:
        if not isinstance(data, dict):
            continue
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
    if len(spaced) < min_items:
        return None
    return spaced, [i.time for i in spaced]


MIN_SHOT_SECONDS = 1.6
MEME_SECONDS = 2.4  # a reaction cut-in lasts this long; the shot before it comes back after


def time_shots(
    specs: List[dict], locate: Callable[[str], Optional[float]], duration: float, start: float = 0.0
) -> List[Scene]:
    """Shots of the doodle look: drawn compositions that follow each other with no gap.

    Each shot starts when its first words are said and lasts until the next
    one; the first fills the screen from ``start`` (the end of an opener) and
    the last until the end of the segment. Shots too close to the previous one
    are dropped.
    """
    placed = []
    for spec in specs:
        kind = spec.get("type")
        if kind not in SCENE_TYPES:
            continue
        found = _spec_times(spec, kind, locate, min_items=1)
        if found is None:
            continue
        items, times = found
        placed.append((max(start, min(times) - 0.15), spec, kind, items))
    placed.sort(key=lambda entry: entry[0])
    kept = []
    for at, spec, kind, items in placed:
        if kept and at - kept[-1][0] < MIN_SHOT_SECONDS:
            continue
        if duration - at < MIN_SHOT_SECONDS:
            continue
        kept.append((at, spec, kind, items))
    shots: List[Scene] = []
    for number, (at, spec, kind, items) in enumerate(kept):
        begin = start if number == 0 else at
        end = kept[number + 1][0] if number + 1 < len(kept) else duration
        visible = [i for i in items if i.time < end - 0.8]
        for item in visible:
            item.time = max(item.time, begin + 0.12)
        if visible and kind in ITEM_SCENES:
            # The screen is never left empty waiting for the first element.
            visible[0].time = begin + 0.12
        scene = _scene_from(spec, kind, begin, end, visible)
        scene.exit = False
        previous = shots[-1] if shots else None
        if kind == "meme" and end - begin > MEME_SECONDS + 1.0:
            scene.end = begin + MEME_SECONDS
            if previous is not None and previous.type in ("illustration", "single") and not previous.reprise:
                shots.append(scene)
                # Back to the picture the reaction interrupted (the very same drawing).
                shots.append(replace(previous, start=scene.end, end=end, items=[], reprise=True, follows=False, camera=""))
                continue
            scene.end = end
        shots.append(scene)
    return shots


def long_holds(shots: List[Scene], duration: float, limit: float, start: float = 0.0) -> List[Tuple[float, float]]:
    """Stretches longer than ``limit`` seconds in which nothing new appears on screen."""
    moments = {start} | {s.start for s in shots} | {i.time for s in shots for i in s.items if i.time > 0}
    moments = sorted(m for m in moments if start <= m < duration)
    ends = moments[1:] + [duration]
    return [(a, b) for a, b in zip(moments, ends) if b - a > limit]


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
        found = _spec_times(spec, kind, locate)
        if found is None:
            continue
        items, times = found

        start = max(0.0, min(times) - (0.3 if kind in ("figure", "annotate") else 0.45))
        if start < 0.5:
            start = 0.0
        last = max(times)
        if kind == "figure":
            try:
                seconds = float(spec.get("seconds") or 5)
            except (TypeError, ValueError):
                seconds = 5.0
            seconds = min(9.0, max(3.0, seconds))
            end = _snap(pauses, start + seconds, start + seconds - 1, start + seconds + 1) + 0.25
        elif kind == "statement":
            end = _snap(pauses, last + 2.6, last + 1.6, last + 5.0) + 0.3
        else:
            tail = _TAILS.get(kind, 2.4) + (2.4 if kind == "equation" and spec.get("example") else 0.0)
            end = _snap(pauses, last + tail, last + 1.8, last + tail + 1.4) + 0.25
        end = min(end, start + MAX_SCENE_SECONDS, duration)
        # Every element stays on screen at least ITEM_SECONDS.
        scene = _scene_from(spec, kind, start, end, [i for i in items if i.time < end - ITEM_SECONDS])
        if kind in ITEM_SCENES and len(scene.items) < 2:
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
        inside = [b for a, b in blocked if a <= scene.start < b]
        if inside:
            # Said while something else holds the screen (a section opener):
            # it waits for it, a little late rather than lost.
            if max(inside) - scene.start > MAX_SCENE_DELAY or not _delay(scene, max(inside) + 0.15, duration):
                continue
        if kept and scene.start < kept[-1].end + 0.6:
            # Right after the previous scene (two definitions in a row): it
            # follows it if that is only a moment late.
            if kept[-1].end + 0.6 - scene.start > MAX_CHAIN_DELAY or not _delay(scene, kept[-1].end + 0.6, duration):
                continue
        clash = [(a, b) for a, b in blocked if scene.start < b and scene.end > a]
        if clash:
            # Make room for the blocked moment rather than losing the scene.
            first = min(a for a, _ in clash)
            if first - 0.2 - scene.start < MIN_SCENE_SECONDS or first <= scene.start:
                continue
            scene.end, scene.exit = first - 0.2, True
            scene.items = [i for i in scene.items if i.time < scene.end - ITEM_SECONDS]
            if scene.type in ITEM_SCENES and len(scene.items) < 2:
                continue
        scene.vertical = len(kept) % 2 == 1
        kept.append(scene)
    return kept


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def format_number(value: float) -> str:
    """12 -> "12", 2.5 -> "2.5", 200000000 -> "200,000,000"."""
    if float(value).is_integer():
        return f"{int(value):,}" if abs(value) >= 10000 else str(int(value))
    return f"{value:g}"


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


def paper(size: Tuple[int, int], color: Tuple[int, int, int], seed: int = 7, vignette: float = 0.07) -> Image.Image:
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
    base *= (1 - vignette * np.clip(distance - 0.35, 0, None) ** 1.5)[:, :, None]
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


def prepare_picture(path: str, allow_cutout: bool = True, trust_alpha: bool = False) -> Optional[Image.Image]:
    """A clean transparent cut-out of an icon or drawing, or the photo itself.

    Pictures that cannot be cut out cleanly (photos, dark or busy backdrops)
    come back whole and marked ``info["framed"]``, so scenes show them as a
    framed card instead of a shredded sticker.
    """
    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
    except Exception:
        return None
    if trust_alpha and fx.has_transparency(image):
        # Icons are drawn with their own transparency; use them as they are.
        bbox = image.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
        return image.crop(bbox) if bbox else image
    cutout = fx.cutout_or_none(image, allow_cutout)
    if cutout is None:
        photo = Image.new("RGBA", image.size, (255, 255, 255, 255))
        photo.alpha_composite(image)
        photo.info["framed"] = True
        return photo
    bbox = cutout.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
    return cutout.crop(bbox) if bbox else cutout


def framed_card(image: Image.Image, border: int, radius: int) -> Image.Image:
    """A photo in a rounded white frame with a soft shadow."""
    card = Image.new("RGBA", (image.width + border * 2, image.height + border * 2), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle((0, 0, card.width - 1, card.height - 1), radius, fill=WHITE + (255,))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1), max(1, radius - border // 2), fill=255)
    card.paste(image.convert("RGB"), (border, border), mask)
    blur = max(2, border)
    out = Image.new("RGBA", (card.width + blur * 4, card.height + blur * 4), (0, 0, 0, 0))
    shade = Image.new("RGBA", out.size, (0, 0, 0, 0))
    shade.paste((0, 0, 0, 255), (blur * 2, blur * 2 + blur // 2), card.getchannel("A").point(lambda a: a * 80 // 255))
    out.alpha_composite(shade.filter(ImageFilter.GaussianBlur(blur)))
    out.alpha_composite(card, (blur * 2, blur * 2))
    return out


def ink_frame(image: Image.Image, border: int, radius: int) -> Image.Image:
    """A photo or map with a hand-drawn ink border (the doodle look's frame)."""
    card = Image.new("RGBA", (image.width + border * 2, image.height + border * 2), (0, 0, 0, 0))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1), max(1, radius - border), fill=255)
    card.paste(image.convert("RGB"), (border, border), mask)
    ImageDraw.Draw(card).rounded_rectangle(
        (border // 2, border // 2, card.width - 1 - border // 2, card.height - 1 - border // 2), radius,
        outline=INK + (255,), width=border,
    )
    return card


BOIL_FRAMES = 4  # each wobble lasts this many frames (7.5 drawings per second, like hand animation)


def boil_variants(image: Image.Image, count: int = 3, amplitude: float = 1.3, grid: int = 5, seed: int = 0) -> List[Image.Image]:
    """The same drawing redrawn ``count`` times with tiny smooth warps ("line boil")."""
    width, height = image.size
    if width < 8 or height < 8:
        return [image] * count
    rng = np.random.default_rng(seed)
    variants = []
    for _ in range(count):
        offsets = rng.uniform(-amplitude, amplitude, (grid + 1, grid + 1, 2))
        offsets[0, :, 1] = offsets[-1, :, 1] = 0  # the edges stay put
        offsets[:, 0, 0] = offsets[:, -1, 0] = 0
        xs = [round(width * i / grid) for i in range(grid + 1)]
        ys = [round(height * j / grid) for j in range(grid + 1)]
        mesh = []
        for j in range(grid):
            for i in range(grid):

                def corner(ci, cj):
                    return xs[ci] + offsets[cj, ci, 0], ys[cj] + offsets[cj, ci, 1]

                quad = (*corner(i, j), *corner(i, j + 1), *corner(i + 1, j + 1), *corner(i + 1, j))
                mesh.append(((xs[i], ys[j], xs[i + 1], ys[j + 1]), quad))
        variants.append(image.transform(image.size, Image.MESH, mesh, Image.BICUBIC))
    return variants


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


def burst_frames(radius: int, width: int, frames: int = 8, color=INK) -> List[Image.Image]:
    """Short strokes flying out of a point: "something just happened"."""
    size = (radius * 2 + width * 4, radius * 2 + width * 4)
    out = []
    for frame in range(1, frames + 1):
        p = frame / frames
        fade = 1 - max(0.0, (p - 0.55) / 0.45)

        def paint(draw: ImageDraw.ImageDraw, s: float, p=p, fade=fade) -> None:
            c = size[0] * s / 2
            for k in range(8):
                a = math.radians(k * 45 + 22)
                inner, outer = radius * (0.45 + 0.4 * p), radius * (0.62 + 0.38 * p)
                draw.line(
                    [(c + inner * s * math.cos(a), c + inner * s * math.sin(a)),
                     (c + outer * s * math.cos(a), c + outer * s * math.sin(a))],
                    fill=color + (int(255 * fade),), width=int(width * s),
                )

        out.append(_supersampled(size, paint, scale=2))
    return out


def bar_frames(length: int, thickness: int, line: int, color, frames: int = 15) -> List[Image.Image]:
    """A rounded bar growing from the left to ``length``."""
    size = (length + line * 4, thickness + line * 4)
    out = []
    for frame in range(1, frames + 1):
        shown = max(thickness, int(length * _ease_out_cubic(frame / frames)))

        def paint(draw: ImageDraw.ImageDraw, s: float, shown=shown) -> None:
            box = (line * 2 * s, line * 2 * s, (line * 2 + shown) * s, (line * 2 + thickness) * s)
            draw.rounded_rectangle(box, thickness * s / 2.4, fill=color + (255,), outline=INK + (255,), width=int(line * s))

        out.append(_supersampled(size, paint, scale=2))
    return out


def gauge_frames(value: float, radius: int, line: int, frames: int = 20) -> List[Image.Image]:
    """A half-circle meter whose needle swings from 0 to ``value`` percent."""
    size = (radius * 2 + line * 6, radius + line * 8)
    value = min(100.0, max(0.0, value))
    out = []
    for frame in range(1, frames + 1):
        shown = value * _ease_out_back(frame / frames) if frame < frames else value

        def paint(draw: ImageDraw.ImageDraw, s: float, shown=shown) -> None:
            cx, cy = size[0] * s / 2, (radius + line * 3) * s
            r = radius * s
            box = (cx - r, cy - r, cx + r, cy + r)
            for start, stop, color in ((180, 240, TEAL), (240, 300, AMBER), (300, 360, CROSS)):
                draw.pieslice(box, start, stop, fill=color + (255,))
            inner = r * 0.62
            draw.pieslice((cx - inner, cy - inner, cx + inner, cy + inner), 180, 360, fill=WHITE + (255,))
            draw.arc(box, 180, 360, fill=INK + (255,), width=int(line * s))
            a = math.radians(180 + 180 * min(100.0, max(0.0, shown)) / 100)
            tip = (cx + r * 0.9 * math.cos(a), cy + r * 0.9 * math.sin(a))
            draw.line([(cx, cy), tip], fill=INK + (255,), width=int(line * 1.4 * s))
            k = line * 1.6 * s
            draw.ellipse((cx - k, cy - k, cx + k, cy + k), fill=INK + (255,))

        out.append(_supersampled(size, paint, scale=2))
    return out


def ken_burns_clip(
    still: Image.Image, output: str, frames: int, size: Tuple[int, int], zoom_from: float, zoom_to: float,
    pan: float = 0.0, zoom: str = "",
) -> str:
    """A slow push-in (or pull-out) video of ``still`` with exactly ``frames`` frames.

    ``pan`` (-1 to 1) also drifts the camera sideways (negative: to the left);
    ``zoom`` replaces the linear zoom with an ffmpeg expression of ``on`` (the frame).
    """
    import subprocess

    width, height = size
    source = output.rsplit(".", 1)[0] + "-still.png"
    still.convert("RGB").resize((int(width * 1.5), int(height * 1.5)), Image.LANCZOS).save(source)
    frames = max(2, frames)
    zoom = zoom or f"{zoom_from}+({zoom_to - zoom_from})*on/{frames - 1}"
    pan = max(-1.0, min(1.0, pan))
    x = "iw/2-(iw/zoom/2)" if not pan else f"(iw-iw/zoom)*(0.5+{0.4 * pan:.3f}*(2*on/{frames - 1}-1))"
    command = [
        utils.get_ffmpeg_binary(), "-v", "error", "-y", "-loop", "1", "-i", source,
        "-vf", f"zoompan=z='{zoom}':x='{x}':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={FPS},format=yuv420p",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0 or not os.path.isfile(output):
        raise RuntimeError(f"zoom clip failed: {(result.stderr or '').strip()[-400:]}")
    return output


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".m4v", ".gif")


def video_clip(
    source: str, output: str, frames: int, size: Tuple[int, int], background: Optional[Image.Image] = None,
    box: Optional[Tuple[int, int, int, int]] = None, cover: Optional[Image.Image] = None, contain: bool = False,
    skip: float = 0.0,
) -> str:
    """A full-frame video of exactly ``frames`` frames playing ``source`` (looped when short).

    With ``box`` (x, y, width, height) the video plays inside that box over
    ``background`` with rounded corners, and ``cover`` (a full-frame picture
    with a transparent window, such as an ink frame) goes on top. Without it
    the video fills the frame, cropped or, with ``contain``, whole over a
    blurred copy of itself.
    """
    import subprocess

    width, height = size
    stem = output.rsplit(".", 1)[0]
    frames = max(2, frames)
    gif = source.lower().endswith(".gif")
    looping = ["-ignore_loop", "0"] if gif else ["-stream_loop", "-1"]
    inputs = [*looping, "-ss", f"{max(0.0, skip):.2f}", "-i", source] if skip and not gif else [*looping, "-i", source]
    grade = "eq=saturation=0.92:contrast=1.03"
    filters = []
    if box is not None:
        x, y, w, h = box
        w, h = w - w % 2, h - h % 2
        back = f"{stem}-back.png"
        (background or Image.new("RGB", size, (0, 0, 0))).convert("RGB").save(back)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), max(4, int(min(w, h) * 0.04)), fill=255)
        mask_path = f"{stem}-mask.png"
        mask.save(mask_path)
        inputs = ["-loop", "1", "-i", back, *inputs, "-loop", "1", "-i", mask_path]
        filters.append(
            f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},{grade},format=rgba[v]"
        )
        filters.append("[2:v]format=gray[m]")
        filters.append("[v][m]alphamerge[vm]")
        filters.append(f"[0:v][vm]overlay={x}:{y}:shortest=0[b]")
        last = "b"
        if cover is not None:
            cover_path = f"{stem}-cover.png"
            cover.save(cover_path)
            inputs += ["-loop", "1", "-i", cover_path]
            filters.append("[b][3:v]overlay=0:0[c]")
            last = "c"
        filters.append(f"[{last}]format=yuv420p[out]")
    elif contain:
        filters.append(
            f"[0:v]fps={FPS},split[a][b];"
            f"[a]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=24:2,eq=brightness=-0.12[bg];"
            f"[b]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[out]"
        )
    else:
        filters.append(
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={FPS},{grade},format=yuv420p[out]"
        )
    command = [
        utils.get_ffmpeg_binary(), "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters),
        "-map", "[out]", "-frames:v", str(frames), "-r", str(FPS), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", output,
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0 or not os.path.isfile(output):
        raise RuntimeError(f"video clip failed: {(result.stderr or '').strip()[-400:]}")
    return output


def reveal_frames(image: Image.Image, frames: int = 12) -> List[Image.Image]:
    """``image`` appearing as if drawn: a soft diagonal wipe from the top-left."""
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32)
    height, width = alpha.shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    diagonal = (x / max(1, width) * 0.75 + y / max(1, height) * 0.25)
    soft = 0.12
    out = []
    for frame in range(1, frames + 1):
        progress = _ease_out_cubic(frame / frames) * (1 + soft) - soft
        ramp = np.clip((progress + soft - diagonal) / soft, 0, 1)
        piece = image.copy()
        piece.putalpha(Image.fromarray((alpha * ramp).astype(np.uint8)))
        out.append(piece)
    out[-1] = image.copy()
    return out


def lighter(color: Tuple[int, int, int], share: float = 0.35) -> Tuple[int, int, int]:
    return tuple(int(c + (255 - c) * share) for c in color)


def darker(color: Tuple[int, int, int], share: float = 0.12) -> Tuple[int, int, int]:
    return tuple(int(c * (1 - share)) for c in color)


def spot(width: int, height: int, color: Tuple[int, int, int], seed: int = 0) -> Image.Image:
    """An organic, slightly wobbly blob, like a paper cut-out behind a drawing."""
    rng = np.random.default_rng(seed)
    harmonics = [(k, rng.uniform(0.015, 0.045), rng.uniform(0, 2 * math.pi)) for k in (2, 3, 5)]

    def paint(draw: ImageDraw.ImageDraw, s: float) -> None:
        cx, cy = width * s / 2, height * s / 2
        points = []
        for step in range(120):
            a = 2 * math.pi * step / 120
            r = 1 + sum(amp * math.sin(k * a + phase) for k, amp, phase in harmonics)
            points.append((cx + cx * 0.94 * r * math.cos(a), cy + cy * 0.94 * r * math.sin(a)))
        draw.polygon(points, fill=color + (255,))

    return _supersampled((width, height), paint, scale=2)


def sunburst(size: Tuple[int, int], color: Tuple[int, int, int], rays: int = 18) -> Image.Image:
    """Comic rays from the centre in two tones of ``color``: the backdrop of a reaction."""
    width, height = size
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    cx, cy, reach = width / 2, height / 2, math.hypot(width, height)
    shade = darker(color, 0.1)
    for ray in range(rays):
        a0 = 2 * math.pi * ray / rays
        a1 = a0 + math.pi / rays
        draw.polygon([(cx, cy), (cx + reach * math.cos(a0), cy + reach * math.sin(a0)),
                      (cx + reach * math.cos(a1), cy + reach * math.sin(a1))], fill=shade)
    vignette = paper(size, (0, 0, 0), vignette=0.0)
    return Image.blend(image, vignette, 0.04)


def fit_inside(image: Image.Image, box: int, most: float = 2.5) -> Image.Image:
    """``image`` as large as fits in a ``box`` square (small icons grow up to ``most`` times)."""
    scale = min(box / max(1, image.width), box / max(1, image.height), most)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.LANCZOS) if size != image.size else image


def fit_cover(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    from PIL import ImageOps

    return ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS)


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
        doodle: bool = False,
        boil: bool = True,
    ):
        """``doodle``: everything is drawn on one flat canvas (the doodle look):
        no slides between scenes, no sticker borders, and with ``boil`` the
        drawings wobble very slightly like hand-drawn animation."""
        self.doodle = doodle
        self.boil = boil and doodle
        self.big = 1.3 if doodle else 1.0  # drawings are the whole picture in the doodle look
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
            paper((self.theme.width, self.theme.height), color, vignette=0.0 if self.doodle else 0.07).save(path)
            self._backgrounds[color] = path
        return self._backgrounds[color]

    def _sound(self, sounds: list, time: float, name: str) -> None:
        if name in self.sfx:
            sounds.append((time, self.sfx[name], _SFX.get(name, 0.6)))

    @staticmethod
    def _slide(scene: Scene, axis: str = "x") -> str:
        """Offset along ``axis`` while the scene slides in and out (0 otherwise)."""
        if scene.still:  # shots of the doodle look never slide
            return "0"
        if (axis == "y") != scene.vertical:
            return "0"
        size = "H" if scene.vertical else "W"
        s, e, d = scene.start, scene.end, SLIDE_SECONDS
        enter = f"{size}*pow(1-clip((t-{s:.3f})/{d},0,1),3)" if scene.enter else "0"
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
        if self.boil:
            return self._boiling(frames, folder, name, x, y, start, scene)
        return fx.Overlay(
            os.path.join(folder, f"{name}_%03d.png"),
            x=f"{int(x)}+{self._slide(scene)}",
            y=f"{int(y)}+{self._slide(scene, 'y')}",
            start=start,
            end=scene.end,
            mode="frames",
            hold=True,
        )

    def _boiling(self, frames, folder, name, x, y, start, scene) -> fx.Overlay:
        """The entrance, then the last frame redrawn with tiny wobbles (line boil) until the end."""
        variants = boil_variants(frames[-1], seed=sum(map(ord, name)) + int(start * 10))
        paths = [os.path.join(folder, f"{name}_{number:03d}.png") for number in range(len(frames))]
        for number, variant in enumerate(variants):
            path = os.path.join(folder, f"{name}_boil{number}.png")
            variant.save(path, compress_level=1)
            paths.append(path)
        entries = [(path, 1 / FPS) for path in paths[: len(frames)]]
        # Long enough to keep wobbling (and fading) while it stays over the next shot.
        remaining = max(0.0, scene.end + max(scene.overlap, scene.linger) - start - len(frames) / FPS)
        cycle = paths[len(frames):]
        step = BOIL_FRAMES / FPS
        for count in range(int(remaining / step) + 1):
            entries.append((cycle[count % len(cycle)], step))
        listing = os.path.join(folder, f"{name}.ffconcat")
        fx.write_ffconcat(listing, entries)
        return fx.Overlay(
            listing, x=f"{int(x)}+{self._slide(scene)}", y=f"{int(y)}+{self._slide(scene, 'y')}",
            start=start, end=scene.end, mode="sequence", hold=True,
        )

    def _centered(self, frames, folder, name, cx, cy, start, scene) -> fx.Overlay:
        w, h = frames[-1].size
        return self._frames(frames, folder, name, cx - w / 2, cy - h / 2, start, scene)

    def _picture(self, item: SceneItem, box: int) -> Optional[Image.Image]:
        image = self.picture(item)
        if image is None:
            return None
        framed = bool(image.info.get("framed"))
        image = image.copy()
        if self.doodle:
            # Drawn straight on the canvas: no sticker border; photos get an ink frame.
            if framed:
                border = self.theme.px(5)
                return ink_frame(fit_inside(image, box - border * 2), border, self.theme.px(14))
            return fit_inside(image, box)
        if framed:
            border = self.theme.px(10)
            return framed_card(fit_inside(image, box - border * 2), border, self.theme.px(22))
        return die_cut(fit_inside(image, box), self.theme.px(8))

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
        os.makedirs(folder, exist_ok=True)
        color = self.theme.accent if scene.type in ("statement", "question") else self.canvas_color
        overlays = []
        if self.doodle:
            scene.still, scene.vertical = True, False
        if scene.type not in CLIP_SCENES and not self.doodle:
            overlays.append(
                fx.Overlay(
                    self._background(color), x=self._slide(scene), y=self._slide(scene, "y"),
                    start=scene.start, end=scene.end,
                )
            )
        sounds: List[Tuple[float, str, float]] = []
        if scene.enter and not self.doodle:
            self._sound(sounds, scene.start - 0.1, "whoosh")
        if scene.exit and not self.doodle:
            self._sound(sounds, scene.end - SLIDE_SECONDS - 0.05, "whoosh")
        builder = getattr(self, f"_build_{scene.type}")
        builder(scene, folder, overlays, sounds)
        if self.doodle:
            for overlay in overlays:
                if overlay.end and overlay.end < scene.end - 0.02:
                    continue
                if scene.overlap > 0:
                    # The next shot covers the whole frame: stay underneath while it fades in.
                    overlay.end, overlay.fade_out = scene.end + scene.overlap, 0.0
                else:
                    # Each drawing leaves with a quick fade when the next shot begins.
                    overlay.end = scene.end + scene.linger
                    overlay.fade_out = max(overlay.fade_out, 0.12, scene.linger)
            if scene.fade_in > 0 and scene.type in FULL_FRAME_SHOTS and overlays:
                overlays[0].fade_in = max(overlays[0].fade_in, scene.fade_in)
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
        number = format_number(value) + scene.unit
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
            text = (format_number(round(shown)) if float(value).is_integer() else f"{shown:.1f}") + scene.unit
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
            box = int(min(H * 0.42 * self.big, W * 0.78 / len(items)))
            label_width = int(W * 0.9 / len(items))
        label_size = int(H * 0.075)
        if divider and len(items) == 2 and not theme.portrait:
            frames, (left, top) = arrow_frames(
                (W / 2, H * 0.14), (W / 2 + 1, H * 0.88), theme.px(7), frames=8, bend=0.02, head=False
            )
            # Drawn with the first element, never on an empty screen.
            when = max(scene.start + (0.12 if self.doodle else SLIDE_SECONDS), min(i.time for i in items))
            overlays.append(self._frames(frames, folder, "divider", left, top, when, scene))
            self._sound(sounds, when, "scribble")
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
        hub_box = int(H * (0.34 if not theme.portrait else 0.22) * self.big)
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
            node_box = int(H * (0.16 if not theme.portrait else 0.08) * self.big)
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

    # -- clip scenes ---------------------------------------------------------------

    @staticmethod
    def _clip_frames(scene: Scene) -> int:
        return int(round((scene.end + max(scene.overlap, scene.linger) - scene.start) * FPS)) + 2

    def _clip(
        self, scene: Scene, still: Image.Image, folder: str, zoom_from: float, zoom_to: float, pan: float = 0.0, zoom: str = ""
    ) -> fx.Overlay:
        path = ken_burns_clip(
            still, os.path.join(folder, "clip.mp4"), self._clip_frames(scene), (self.theme.width, self.theme.height),
            zoom_from, zoom_to, pan=pan, zoom=zoom,
        )
        return fx.Overlay(
            path, x=self._slide(scene), y=self._slide(scene, "y"), start=scene.start, end=scene.end, mode="media"
        )

    def _caption(self, canvas: Image.Image, text: str) -> None:
        """A hand-lettered caption on a white pill near the bottom."""
        W, H = canvas.size
        label = self.text.render(text.upper(), int(H * 0.07), int(W * 0.8))
        pad = self.theme.px(18)
        pill = Image.new("RGBA", (label.width + pad * 3, label.height + pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(pill).rounded_rectangle((0, 0, pill.width - 1, pill.height - 1), pill.height // 2, fill=WHITE + (240,))
        pill.alpha_composite(label, ((pill.width - label.width) // 2, pad))
        canvas.alpha_composite(pill, ((W - pill.width) // 2, int(H * 0.95) - pill.height))

    def _build_figure(self, scene, folder, overlays, sounds) -> None:
        """A real picture or diagram filling the screen, slowly zooming."""
        W, H = self.theme.width, self.theme.height
        item = scene.center or SceneItem()
        image = self.picture(item)
        if image is None:
            raise ValueError("the figure has no picture")
        canvas = Image.new("RGBA", (W, H), self.canvas_color + (255,))
        canvas.paste(paper((W, H), self.canvas_color))
        photo = bool(image.info.get("framed")) and scene.look == "photo"
        if photo:
            canvas.paste(fit_cover(image, (W, H)))
        else:
            box = (int(W * 0.9), int(H * (0.78 if item.label else 0.86)))
            picture = image.copy()
            if image.info.get("framed"):
                border = self.theme.px(12)
                picture.thumbnail((box[0] - border * 2, box[1] - border * 2), Image.LANCZOS)
                picture = framed_card(picture, border, self.theme.px(18))
            else:
                picture.thumbnail(box, Image.LANCZOS)
                picture = die_cut(picture, self.theme.px(8))
            top = (H * (0.92 if item.label else 1.0) - picture.height) / 2
            canvas.alpha_composite(picture, ((W - picture.width) // 2, max(0, int(top))))
        if item.label:
            self._caption(canvas, item.label)
        overlays.append(self._clip(scene, canvas, folder, 1.0, 1.12 if photo else 1.05))

    def _build_zoom(self, scene, folder, overlays, sounds) -> None:
        """One thing, big, while the camera slowly pushes in (or pulls out)."""
        W, H = self.theme.width, self.theme.height
        item = scene.center or SceneItem()
        canvas = paper((W, H), self.canvas_color).convert("RGBA")
        box = int(H * (0.62 if not self.theme.portrait else 0.34))
        card, _ = self._card(item, box, int(H * 0.085), int(W * 0.8), label_on_top=False)
        canvas.alpha_composite(card, ((W - card.width) // 2, max(0, (H - card.height) // 2)))
        zoom = (1.0, 1.16) if scene.direction == "in" else (1.16, 1.0)
        overlays.append(self._clip(scene, canvas, folder, *zoom))
        self._sound(sounds, scene.start + SLIDE_SECONDS, "pop")

    # -- story ---------------------------------------------------------------------

    def _build_story(self, scene, folder, overlays, sounds) -> None:
        """A tiny flipbook: 2-4 moments where the host and a prop change."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        frames = scene.items[:4]
        with_host = self.host_still is not None and not theme.portrait
        prop_center = (W * (0.66 if with_host else 0.5), H * 0.44)
        last_expression = None
        for number, item in enumerate(frames):
            until = frames[number + 1].time if number + 1 < len(frames) else scene.end
            image = self.picture(item)
            whole_scene = image is not None and bool(image.info.get("scene"))
            if whole_scene:
                # An illustrated moment (drawn by the image model) fills the stage.
                picture = image.copy()
                border = theme.px(12)
                picture.thumbnail((int(W * 0.8), int(H * 0.72)), Image.LANCZOS)
                card = framed_card(picture, border, theme.px(18))
                overlay = self._centered(pop_frames(card, start_scale=0.9), folder, f"moment{number}", W / 2, H * 0.44, item.time, scene)
            else:
                card, _ = self._card(item, int(H * (0.42 if not theme.portrait else 0.26)), int(H * 0.07), int(W * 0.46), label_on_top=False)
                overlay = self._centered(pop_frames(card), folder, f"prop{number}", *prop_center, item.time, scene)
            overlay.end = until
            overlays.append(overlay)
            self._sound(sounds, item.time, "pop")
            if number > 0:
                burst = burst_frames(int(H * 0.26), theme.px(7))
                center = (W / 2, H * 0.44) if whole_scene else prop_center
                flash = self._centered(burst, folder, f"burst{number}", *center, item.time, scene)
                flash.hold, flash.end = False, item.time + len(burst) / FPS
                overlays.append(flash)
            if with_host and not whole_scene:
                expression = item.expression or scene.expression or "explicando"
                height = int(H * 0.72)
                pose = die_cut(self.host_still(expression, height), theme.px(6), shadow=False)
                motion = pop_frames(pose, frames=8, start_scale=0.55 if last_expression is None else 0.93)
                w, h = motion[-1].size
                pad_y = (h - pose.height) / 2
                top = H + 0.08 * pose.height - pose.height - pad_y
                host_overlay = self._frames(motion, folder, f"host{number}", W * 0.27 - w / 2, top, item.time, scene)
                host_overlay.end = until
                overlays.append(host_overlay)
                last_expression = expression
            if whole_scene and item.label:
                label = self.text.render(item.label.upper(), int(H * 0.075), int(W * 0.8))
                caption = self._centered(pop_frames(label), folder, f"caption{number}", W / 2, H * 0.89, item.time + 0.2, scene)
                caption.end = until
                overlays.append(caption)

    # -- steps, timeline, formula -----------------------------------------------------

    def _badge(self, number: int, size: int) -> Image.Image:
        def paint(draw: ImageDraw.ImageDraw, s: float) -> None:
            draw.ellipse((0, 0, size * s - 1, size * s - 1), fill=self.theme.accent + (255,), outline=INK + (255,), width=int(size * 0.07 * s))

        badge = _supersampled((size, size), paint, scale=2)
        digit = self.text.render(str(number), int(size * 0.72), size, color=WHITE, max_lines=1)
        badge.alpha_composite(digit, ((size - digit.width) // 2, (size - digit.height) // 2))
        return badge

    def _build_steps(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        items = scene.items[:5]
        n = len(items)
        if scene.cycle and n >= 3 and not theme.portrait:
            centers = [
                (W / 2 + W * 0.3 * math.cos(math.radians(-90 + 360 * k / n)), H * 0.53 + H * 0.31 * math.sin(math.radians(-90 + 360 * k / n)))
                for k in range(n)
            ]
            box = int(H * 0.16)
        elif theme.portrait:
            centers = [(W * 0.5, H * (0.18 + 0.68 * (k + 0.5) / n)) for k in range(n)]
            box = int(min(H * 0.13, H * 0.5 / n))
        else:
            centers = [(W * 0.06 + W * 0.88 * (k + 0.5) / n, H * 0.54) for k in range(n)]
            box = int(min(H * 0.3 * self.big, W * 0.62 / n))
        cards = []
        for number, (item, (cx, cy)) in enumerate(zip(items, centers)):
            card, _ = self._card(item, box, int(H * 0.06), int(W * 0.8 / n) if not theme.portrait else int(W * 0.7), label_on_top=False, beside=theme.portrait)
            badge = self._badge(number + 1, theme.px(64))
            combined = Image.new("RGBA", (max(card.width, badge.width), card.height + badge.height // 2), (0, 0, 0, 0))
            combined.alpha_composite(card, ((combined.width - card.width) // 2, badge.height // 2))
            combined.alpha_composite(badge, (0, 0))
            cards.append(combined)
            overlays.append(self._centered(pop_frames(combined), folder, f"step{number}", cx, cy, item.time, scene))
            self._sound(sounds, item.time, "pop")
        pairs = list(zip(range(n - 1), range(1, n)))
        if scene.cycle and n >= 3 and not theme.portrait:
            pairs.append((n - 1, 0))
        for a, b in pairs:
            (ax, ay), (bx, by) = centers[a], centers[b]
            length = math.hypot(bx - ax, by - ay) or 1
            ux, uy = (bx - ax) / length, (by - ay) / length
            ra = max(cards[a].size) * 0.5
            rb = max(cards[b].size) * 0.5
            if length < ra + rb + theme.px(30):
                ra = rb = (length - theme.px(30)) / 2
            frames, (left, top) = arrow_frames(
                (ax + ux * ra, ay + uy * ra), (bx - ux * rb, by - uy * rb), theme.px(6), frames=8,
                bend=0.18 if scene.cycle else 0.0,
            )
            when = items[b].time if b > 0 else items[a].time + 0.4
            overlays.append(self._frames(frames, folder, f"arrow{a}-{b}", left, top, when - 0.15, scene))
            self._sound(sounds, when - 0.15, "scribble")

    def _build_timeline(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        items = scene.items[:5]
        n = len(items)
        y = H * (0.5 if not theme.portrait else 0.5)
        frames, (left, top) = arrow_frames((W * 0.05, y), (W * 0.95, y), theme.px(8), frames=12, bend=0.0)
        overlays.append(self._frames(frames, folder, "line", left, top, scene.start + SLIDE_SECONDS, scene))
        self._sound(sounds, scene.start + SLIDE_SECONDS, "scribble")
        for number, item in enumerate(items):
            cx = W * 0.08 + W * 0.84 * (number + 0.5) / n
            dot = _supersampled((theme.px(34), theme.px(34)), lambda d, s: d.ellipse((0, 0, theme.px(34) * s - 1, theme.px(34) * s - 1), fill=self.theme.accent + (255,), outline=INK + (255,), width=int(theme.px(5) * s)))
            overlays.append(self._centered(stamp_frames(dot), folder, f"dot{number}", cx, y, item.time, scene))
            if item.date:
                date = self.text.render(item.date.upper(), int(H * 0.08), int(W * 0.8 / n), max_lines=1)
                overlays.append(self._centered(pop_frames(date), folder, f"date{number}", cx, y - H * 0.1, item.time, scene))
            card, _ = self._card(item, int(min(H * 0.22, W * 0.7 / n)), int(H * 0.05), int(W * 0.85 / n), label_on_top=False)
            above = number % 2 == 1 and not item.date
            cy = y - H * 0.06 - card.height / 2 if above else y + H * 0.06 + card.height / 2
            overlays.append(self._centered(pop_frames(card), folder, f"event{number}", cx, cy, item.time + 0.1, scene))
            self._sound(sounds, item.time, "pop")

    def _build_formula(self, scene, folder, overlays, sounds) -> None:
        W, H = self.theme.width, self.theme.height
        terms = [i for i in scene.items if i.role != "result"][:3]
        result = next((i for i in scene.items if i.role == "result"), None)
        parts = terms + ([result] if result else [])
        n = len(parts)
        symbols = [scene.operator or "+"] * (len(terms) - 1) + (["="] if result else [])
        slot = W * 0.92 / (n + len(symbols) * 0.45)
        box = int(min(H * 0.32, slot * 0.86))
        x = W * 0.04
        for number, item in enumerate(parts):
            cx = x + slot / 2
            card, _ = self._card(item, box, int(H * 0.06), int(slot * 0.95), label_on_top=False)
            is_result = item is result
            frames = stamp_frames(card) if is_result else pop_frames(card)
            overlays.append(self._centered(frames, folder, f"term{number}", cx, H * 0.5, item.time, scene))
            self._sound(sounds, item.time, "stamp" if is_result else "pop")
            x += slot
            if number < len(symbols):
                sign = self.text.render(symbols[number], int(H * 0.2), int(slot * 0.45), max_lines=1)
                overlays.append(self._centered(pop_frames(sign), folder, f"sign{number}", x + slot * 0.225, H * 0.47, parts[number + 1].time - 0.15, scene))
                x += slot * 0.45

    # -- numbers -------------------------------------------------------------------

    def _build_bars(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        items = scene.items[:5]
        n = len(items)
        top_value = max((abs(i.value) for i in items), default=1) or 1
        row = H * 0.72 / n
        thickness = int(min(row * 0.55, H * 0.11))
        label_width = int(W * 0.26)
        bar_left = W * 0.08 + label_width + theme.px(24)
        longest = int(W * 0.92 - bar_left - W * 0.12)
        for number, item in enumerate(items):
            cy = H * 0.16 + row * (number + 0.5)
            label, _ = self._card(item, int(min(row * 0.8, H * 0.12)), int(min(row * 0.42, H * 0.06)), label_width, label_on_top=False, beside=True)
            overlays.append(self._centered(pop_frames(label), folder, f"label{number}", W * 0.08 + label_width / 2, cy, item.time, scene))
            highlight = abs(item.value) == top_value
            length = max(thickness, int(longest * abs(item.value) / top_value))
            frames = bar_frames(length, thickness, theme.px(5), self.theme.accent if highlight else TEAL)
            overlays.append(self._frames(frames, folder, f"bar{number}", bar_left, cy - frames[-1].height / 2, item.time + 0.1, scene))
            value = self.text.render(f"{format_number(item.value)}{scene.unit}", int(thickness * 0.95), int(W * 0.16), max_lines=1)
            overlays.append(self._frames(pop_frames(value), folder, f"value{number}", bar_left + length + theme.px(18), cy - value.height * 0.6, item.time + 0.5, scene))
            self._sound(sounds, item.time + 0.1, "scribble")

    def _build_grid(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        total = max(2, min(100, scene.total))
        count = int(round(min(total, max(0.0, scene.value))))
        columns = {10: 5, 20: 10, 50: 10, 100: 20}.get(total, min(10, total))
        rows = math.ceil(total / columns)
        area_w, area_h = W * 0.8, H * (0.56 if not theme.portrait else 0.4)
        cell = int(min(area_w / columns, area_h / rows))
        item = scene.center or SceneItem()
        icon_item = SceneItem(icon=item.icon or "🧑", draw=item.draw)
        source = self.picture(icon_item)
        if source is None or source.info.get("framed"):
            source = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(source).ellipse((8, 8, 56, 56), fill=TEAL + (255,), outline=INK + (255,), width=5)
        icon = source.copy()
        icon.thumbnail((int(cell * 0.86), int(cell * 0.86)), Image.LANCZOS)
        faded = icon.convert("LA").convert("RGBA")
        faded.putalpha(faded.getchannel("A").point(lambda a: a * 90 // 255))
        grid_w, grid_h = cell * columns, cell * rows
        steps = 16
        frames = []
        for frame in range(1, steps + 1):
            lit = int(round(count * _ease_out_cubic(frame / steps)))
            canvas = Image.new("RGBA", (grid_w, grid_h), (0, 0, 0, 0))
            for k in range(total):
                r, c = divmod(k, columns)
                piece = icon if k < lit else faded
                canvas.alpha_composite(piece, (c * cell + (cell - piece.width) // 2, r * cell + (cell - piece.height) // 2))
            frames.append(canvas)
        appear = scene.start + SLIDE_SECONDS + 0.05
        overlays.append(self._centered(frames, folder, "grid", W / 2, H * 0.42, appear, scene))
        for k in range(0, steps, 4):
            self._sound(sounds, appear + k / FPS, "tick")
        label = item.label or f"{count} / {total}"
        text = self.text.render(label.upper(), int(H * 0.09), int(W * 0.86))
        overlays.append(self._centered(pop_frames(text), folder, "label", W / 2, H * 0.42 + grid_h / 2 + H * 0.1, appear + 0.6, scene))
        self._sound(sounds, appear + 0.6, "pop")

    def _build_gauge(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS + 0.05
        radius = int(H * (0.32 if not theme.portrait else 0.2))
        frames = gauge_frames(scene.value, radius, theme.px(9))
        cy = H * 0.5
        overlays.append(self._centered(frames, folder, "gauge", W / 2, cy, appear, scene))
        self._sound(sounds, appear, "scribble")
        bottom = cy + frames[-1].height / 2
        for name, text, x in (("low", scene.low, W / 2 - radius), ("high", scene.high, W / 2 + radius)):
            if text:
                image = self.text.render(text.upper(), int(H * 0.055), int(W * 0.25))
                overlays.append(self._centered(pop_frames(image), folder, name, x, bottom + H * 0.04, appear + 0.2, scene))
        label = scene.center.label if scene.center else ""
        if label:
            image = self.text.render(label.upper(), int(H * 0.09), int(W * 0.8))
            overlays.append(self._centered(pop_frames(image), folder, "label", W / 2, H * 0.12, appear + 0.5, scene))
            self._sound(sounds, appear + 0.5, "pop")

    # -- question -------------------------------------------------------------------

    def _build_question(self, scene, folder, overlays, sounds) -> None:
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS * 0.6
        with_host = self.host_still is not None
        text_width = int(W * (0.56 if with_host and not theme.portrait else 0.86))
        text = self.text.render(scene.text.upper() or "?", int(H * 0.11), text_width, max_lines=3)
        cx = W * (0.62 if with_host and not theme.portrait else 0.5)
        overlays.append(self._centered(pop_frames(text), folder, "question", cx, H * 0.42, appear + 0.15, scene))
        self._sound(sounds, appear + 0.15, "pop")
        marks = [(-0.36, -0.3, -12), (0.38, -0.26, 14), (-0.3, 0.34, 8), (0.34, 0.36, -10)]
        for number, (dx, dy, angle) in enumerate(marks):
            mark = self.text.render("?", int(H * 0.14), int(W * 0.2), color=WHITE, max_lines=1)
            mark = mark.rotate(angle, resample=Image.BICUBIC, expand=True)
            when = appear + 0.4 + number * 0.18
            overlays.append(self._centered(pop_frames(mark), folder, f"mark{number}", cx + dx * text_width, H * 0.42 + dy * H * 0.8, when, scene))
        if with_host and not theme.portrait:
            height = int(H * 0.8)
            pose = die_cut(self.host_still(scene.expression or "pensando", height), theme.px(6), shadow=False)
            frames = pop_frames(pose, frames=10, start_scale=0.55)
            w, h = frames[-1].size
            pad_y = (h - pose.height) / 2
            overlays.append(self._frames(frames, folder, "host", W * 0.2 - w / 2, H + 0.1 * pose.height - pose.height - pad_y, appear, scene))

    # -- section opener ---------------------------------------------------------------

    def _fit_picture(self, image: Image.Image, box: Tuple[int, int]) -> Image.Image:
        """A picture as big as ``box``: framed when it is a photo or a diagram, die-cut otherwise."""
        picture = image.copy()
        if image.info.get("framed"):
            border = self.theme.px(12)
            picture.thumbnail((box[0] - border * 2, box[1] - border * 2), Image.LANCZOS)
            return framed_card(picture, border, self.theme.px(20))
        picture.thumbnail(box, Image.LANCZOS)
        return die_cut(picture, self.theme.px(8))

    def _build_opener(self, scene, folder, overlays, sounds) -> None:
        """Section title card: the number, the title and a picture of exactly that topic."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        item = scene.center or SceneItem()
        image = self.picture(item) if (item.query or item.icon or item.draw) else None
        title = (scene.text or item.label).upper()
        picture = None
        if image is not None and theme.portrait:
            picture = self._fit_picture(image, (int(W * 0.86), int(H * 0.42)))
            picture_center, column, middle, title_width = (W / 2, H * 0.3), W / 2, H * 0.7, int(W * 0.86)
        elif image is not None:
            picture = self._fit_picture(image, (int(W * 0.52), int(H * 0.76)))
            picture_center, column, middle, title_width = (W * 0.7, H * 0.5), W * 0.25, H * 0.5, int(W * 0.4)
        else:
            column, middle, title_width = W / 2, H * 0.48, int(W * 0.82)
        size = H * (0.1 if picture is not None and not theme.portrait else 0.12 if picture is None else 0.06)
        digits = None
        if scene.number:
            digits = self.text.render(f"{scene.number:02d}", int(size * 2.6), int(W * 0.4), color=theme.accent, max_lines=1)
        text = self.text.render(title, int(size), title_width, max_lines=3)
        gap, line_gap = theme.px(8), theme.px(14)
        block = (digits.height + gap if digits else 0) + text.height + line_gap + theme.px(10)
        top = middle - block / 2
        if digits is not None:
            overlays.append(self._centered(stamp_frames(digits), folder, "number", column, top + digits.height / 2, scene.start + 0.05, scene))
            top += digits.height + gap
        overlays.append(self._centered(pop_frames(text), folder, "title", column, top + text.height / 2, scene.start + 0.2, scene))
        underline_y = top + text.height + line_gap
        half = text.width / 2
        frames, (left, line_top) = arrow_frames(
            (column - half, underline_y), (column + half, underline_y + 1), theme.px(7), frames=8, bend=0.03, head=False
        )
        overlays.append(self._frames(frames, folder, "underline", left, line_top, scene.start + 0.45, scene))
        if picture is not None:
            when = scene.start + 0.3
            overlays.append(self._centered(pop_frames(picture, start_scale=0.8), folder, "picture", *picture_center, when, scene))
            self._sound(sounds, when, "pop")

    # -- definitions and equations ------------------------------------------------------

    def _pill(self, text: str, size: int, max_width: int, color=INK, border=None) -> Image.Image:
        """Hand-lettered text on a white rounded label (with an optional coloured edge)."""
        label = self.text.render(text, size, max_width, color=color, max_lines=2)
        pad = self.theme.px(16)
        pill = Image.new("RGBA", (label.width + pad * 2, label.height + pad * 2 - pad // 2), (0, 0, 0, 0))
        edge = self.theme.px(4) if border else 0
        ImageDraw.Draw(pill).rounded_rectangle(
            (0, 0, pill.width - 1, pill.height - 1), min(pill.height // 2, self.theme.px(26)),
            fill=WHITE + (245,), outline=(border + (255,)) if border else None, width=edge,
        )
        pill.alpha_composite(label, ((pill.width - label.width) // 2, (pill.height - label.height) // 2))
        return pill

    def _symbol_badge(self, symbol: str, size: int, color) -> Image.Image:
        def paint(draw: ImageDraw.ImageDraw, s: float) -> None:
            draw.ellipse((0, 0, size * s - 1, size * s - 1), fill=color + (255,), outline=INK + (255,), width=int(size * 0.06 * s))

        badge = _supersampled((size, size), paint, scale=2)
        letter = self.text.render(symbol, int(size * 0.62), int(size * 0.8), color=WHITE, max_lines=1)
        badge.alpha_composite(letter, ((size - letter.width) // 2, (size - letter.height) // 2))
        return badge

    def _build_definition(self, scene, folder, overlays, sounds) -> None:
        """A glossary card: the term (and its symbol), what it means and how it is measured."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS + 0.05
        item = scene.center or SceneItem()
        picture = None
        if item.icon or item.query or item.draw:
            picture = self._picture(item, int(min(H * 0.5, W * 0.32) if not theme.portrait else H * 0.28))
        if theme.portrait:
            column, middle, width = W / 2, H * (0.64 if picture is not None else 0.5), int(W * 0.86)
            picture_center = (W / 2, H * 0.27)
        elif picture is not None:
            column, middle, width = W * 0.66, H * 0.5, int(W * 0.5)
            picture_center = (W * 0.22, H * 0.52)
        else:
            column, middle, width = W / 2, H * 0.5, int(W * 0.8)
        big = 1.0 if not theme.portrait else 0.62  # text sizes are fractions of the height
        term = self.text.render(item.label.upper(), int(H * 0.13 * big), int(width * (0.78 if scene.symbol else 1)), color=theme.accent, max_lines=2)
        if scene.symbol:
            badge = self._symbol_badge(scene.symbol, int(min(term.height, H * 0.14 * big)), INK)
            gap = theme.px(22)
            head = Image.new("RGBA", (term.width + gap + badge.width, max(term.height, badge.height)), (0, 0, 0, 0))
            head.alpha_composite(term, (0, (head.height - term.height) // 2))
            head.alpha_composite(badge, (term.width + gap, (head.height - badge.height) // 2))
        else:
            head = term
        meaning = self.text.render(scene.text, int(H * 0.068 * big), width, max_lines=3) if scene.text else None
        unit = self._pill(scene.unit, int(H * 0.058 * big), int(width * 0.9), border=theme.accent) if scene.unit else None
        gap = theme.px(26)
        parts = [p for p in (head, meaning, unit) if p is not None]
        top = middle - (sum(p.height for p in parts) + gap * (len(parts) - 1)) / 2
        when = appear + 0.1
        for name, part, frames, delay in (("term", head, pop_frames, 0.0), ("meaning", meaning, pop_frames, 0.7), ("unit", unit, stamp_frames, 1.5)):
            if part is None:
                continue
            overlays.append(self._centered(frames(part), folder, name, column, top + part.height / 2, when + delay, scene))
            self._sound(sounds, when + delay, "stamp" if name == "unit" else "pop")
            top += part.height + gap
        if picture is not None:
            overlays.append(self._centered(pop_frames(picture), folder, "picture", *picture_center, appear, scene))

    _TERM_COLORS = ((226, 44, 58), TEAL, (205, 128, 18), (112, 76, 196))

    def _build_equation(self, scene, folder, overlays, sounds) -> None:
        """A formula written big, each symbol explained underneath as the narration names it."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS + 0.05
        tokens = re.findall(r"[^\W_]+(?:_[^\W_]+)?|[^\w\s]", scene.text) or [scene.text or "?"]
        palette = (theme.accent,) + self._TERM_COLORS[1:]
        colors = {}
        for number, term in enumerate(t for t in scene.items if t.symbol):
            colors.setdefault(term.symbol, palette[number % len(palette)])
        size = int(H * (0.24 if not theme.portrait else 0.12))
        while True:
            images = [self.text.render(t, size, W, color=colors.get(t, INK), max_lines=1) for t in tokens]
            gap = int(size * 0.18)
            total = sum(i.width for i in images) + gap * (len(images) - 1)
            if total <= W * 0.86 or size < 30:
                break
            size = int(size * 0.88)
        has_terms = any(t.symbol for t in scene.items)
        formula_y = H * (0.42 if has_terms else 0.5)
        x = (W - total) / 2
        centers = []
        for number, image in enumerate(images):
            centers.append(x + image.width / 2)
            overlays.append(self._centered(pop_frames(image, frames=7), folder, f"token{number}", centers[-1], formula_y, appear + 0.08 * number, scene))
            x += image.width + gap
        self._sound(sounds, appear, "pop")
        name = scene.center.label if scene.center else ""
        if name:
            label = self.text.render(name.upper(), int(H * 0.075), int(W * 0.8), max_lines=1)
            overlays.append(self._centered(pop_frames(label), folder, "name", W / 2, H * 0.13, appear + 0.2, scene))
        formula_bottom = formula_y + max(i.height for i in images) / 2
        used = set()
        symbols = [i for i, t in enumerate(tokens) if t in colors]
        spacing = min((b - a for a, b in zip([centers[i] for i in symbols], [centers[i] for i in symbols][1:])), default=W * 0.4)
        for number, term in enumerate(scene.items[:4]):
            index = next((i for i, t in enumerate(tokens) if t == term.symbol and i not in used), None)
            if index is None or not (term.label or term.unit):
                continue
            used.add(index)
            color = colors.get(term.symbol, INK)
            width = int(min(W * 0.3, spacing * 0.96))
            lines = [self.text.render(term.label.upper(), int(H * (0.068 if not theme.portrait else 0.034)), width, color=color)] if term.label else []
            if term.unit:
                lines.append(self.text.render(term.unit, int(H * (0.056 if not theme.portrait else 0.028)), width))
            caption = Image.new("RGBA", (max(i.width for i in lines), sum(i.height for i in lines) + theme.px(6) * (len(lines) - 1)), (0, 0, 0, 0))
            y = 0
            for line in lines:
                caption.alpha_composite(line, ((caption.width - line.width) // 2, y))
                y += line.height + theme.px(6)
            cx = fx.clamp(centers[index], caption.width / 2 + W * 0.02, W * 0.98 - caption.width / 2)
            caption_y = formula_bottom + H * 0.14 + caption.height / 2
            overlays.append(self._centered(pop_frames(caption), folder, f"term{number}", cx, caption_y, term.time, scene))
            frames, (left, top) = arrow_frames(
                (cx, caption_y - caption.height / 2 - theme.px(8)), (centers[index], formula_bottom + theme.px(10)), theme.px(5), frames=7, bend=0.0
            )
            overlays.append(self._frames(frames, folder, f"pointer{number}", left, top, term.time + 0.1, scene))
            self._sound(sounds, term.time, "pop")
        if scene.example:
            when = max([t.time for t in scene.items] + [appear]) + 1.0
            example = self._pill(scene.example, int(H * 0.065), int(W * 0.8), border=theme.accent)
            overlays.append(self._centered(stamp_frames(example), folder, "example", W / 2, H * 0.88, when, scene))
            self._sound(sounds, when, "stamp")

    # -- annotated picture ------------------------------------------------------------------

    def _build_annotate(self, scene, folder, overlays, sounds) -> None:
        """A real picture with labels pointing at its parts as each one is explained."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS * 0.6
        image = self.picture(scene.center or SceneItem())
        if image is None:
            raise ValueError("the annotated picture is missing")
        border = theme.px(12)
        photo = image.convert("RGB").convert("RGBA")
        labels = scene.items[:5]
        if theme.portrait:
            box, center = (W * 0.92, H * 0.5), (W / 2, H * 0.3)
        else:
            box, center = (W * (0.6 if labels else 0.86), H * 0.86), (W * (0.34 if labels else 0.5), H * 0.5)
        photo.thumbnail((int(box[0]) - border * 2, int(box[1]) - border * 2), Image.LANCZOS)
        card = framed_card(photo, border, theme.px(20))
        overlays.append(self._centered(pop_frames(card, start_scale=0.85), folder, "picture", *center, appear, scene))
        inner = max(2, border) * 2 + border
        left_x, top_y = center[0] - card.width / 2 + inner, center[1] - card.height / 2 + inner
        n = len(labels)
        for number, item in enumerate(labels):
            pill = self._pill(item.label.upper(), int(H * (0.052 if not theme.portrait else 0.03)), int(W * (0.3 if not theme.portrait else 0.42)), border=theme.accent)
            if theme.portrait:
                columns = 2 if n > 2 else n
                row, col = divmod(number, columns)
                lx = W * (0.5 if columns == 1 else 0.27 + 0.46 * col)
                ly = H * 0.64 + row * (pill.height + H * 0.04) + pill.height / 2
                anchor = (lx, ly - pill.height / 2)
            else:
                lx = W * 0.82
                ly = H * 0.5 + (number - (n - 1) / 2) * min(H * 0.17, H * 0.8 / max(1, n))
                anchor = (lx - pill.width / 2, ly)
            overlays.append(self._centered(pop_frames(pill), folder, f"label{number}", lx, ly, item.time, scene))
            self._sound(sounds, item.time, "pop")
            if item.point is None:
                continue
            px_, py_ = left_x + item.point[0] * photo.width, top_y + item.point[1] * photo.height
            dot_size = theme.px(30)
            dot = _supersampled((dot_size, dot_size), lambda d, s: d.ellipse((0, 0, dot_size * s - 1, dot_size * s - 1), fill=theme.accent + (255,), outline=WHITE + (255,), width=int(theme.px(6) * s)))
            if math.hypot(anchor[0] - px_, anchor[1] - py_) > theme.px(40):
                frames, (left, top) = arrow_frames(anchor, (px_, py_), theme.px(5), frames=8, bend=0.1 if number % 2 else -0.1, head=False)
                overlays.append(self._frames(frames, folder, f"arrow{number}", left, top, item.time + 0.15, scene))
            overlays.append(self._centered(stamp_frames(dot), folder, f"dot{number}", px_, py_, item.time + 0.4, scene))

    # -- chains and branches ------------------------------------------------------------------

    def _build_chain(self, scene, folder, overlays, sounds) -> None:
        """Real things left to right, joined by arrows that say what each one does to the next."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        items = scene.items[:4]
        n = len(items)
        if theme.portrait:
            centers = [(W * 0.5, H * (0.1 + 0.82 * (k + 0.5) / n)) for k in range(n)]
            box = int(min(H * 0.12, H * 0.44 / n))
        else:
            centers = [(W * 0.04 + W * 0.92 * (k + 0.5) / n, H * 0.5) for k in range(n)]
            box = int(min(H * 0.4 * self.big, W * 0.7 / n))
        cards = []
        for number, (item, (cx, cy)) in enumerate(zip(items, centers)):
            card, _ = self._card(item, box, int(H * 0.058), int(W * 0.8 / n) if not theme.portrait else int(W * 0.5), label_on_top=False, beside=theme.portrait)
            cards.append(card)
            overlays.append(self._centered(pop_frames(card), folder, f"link{number}", cx, cy, item.time, scene))
            self._sound(sounds, item.time, "pop")
        for a in range(n - 1):
            (ax, ay), (bx, by) = centers[a], centers[a + 1]
            if theme.portrait:
                p0, p1 = (ax, ay + cards[a].height / 2 + theme.px(6)), (bx, by - cards[a + 1].height / 2 - theme.px(6))
            else:
                p0, p1 = (ax + cards[a].width / 2 + theme.px(4), ay), (bx - cards[a + 1].width / 2 - theme.px(4), by)
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < theme.px(36):
                continue
            when = items[a + 1].time - 0.25
            frames, (left, top) = arrow_frames(p0, p1, theme.px(6), frames=8, bend=0.0)
            overlays.append(self._frames(frames, folder, f"arrow{a}", left, top, when, scene))
            self._sound(sounds, when, "scribble")
            if items[a].link:
                verb = self.text.render(items[a].link.lower(), int(H * (0.05 if not theme.portrait else 0.03)), int(W * 0.18 if not theme.portrait else W * 0.4), color=theme.accent)
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                spot = (mx + verb.width / 2 + theme.px(16), my) if theme.portrait else (mx, my - verb.height / 2 - theme.px(22))
                overlays.append(self._centered(pop_frames(verb), folder, f"verb{a}", *spot, when + 0.15, scene))

    def _build_branch(self, scene, folder, overlays, sounds) -> None:
        """One cause on the left and what it leads to fanning out on the right."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + SLIDE_SECONDS + 0.05
        source = scene.center or SceneItem()
        items = scene.items[:4]
        n = len(items)
        if theme.portrait:
            origin = (W / 2, H * 0.24)
            hub, _ = self._card(source, int(H * 0.2), int(H * 0.035), int(W * 0.7), label_on_top=False)
            if n == 4:
                spots = [(W * (0.27 + 0.46 * (k % 2)), H * (0.58 + 0.2 * (k // 2))) for k in range(n)]
            else:
                spots = [(W * (k + 0.5) / n, H * 0.66) for k in range(n)]
            box, beside = int(H * 0.11), False
        else:
            origin = (W * 0.2, H * 0.52)
            hub, _ = self._card(source, int(H * 0.38), int(H * 0.06), int(W * 0.3), label_on_top=False)
            spots = [(W * 0.72, H * (0.52 + (k - (n - 1) / 2) * min(0.24, 0.8 / n))) for k in range(n)]
            box, beside = int(min(H * 0.18, H * 0.66 / n)), True
        overlays.append(self._centered(pop_frames(hub), folder, "source", *origin, appear, scene))
        self._sound(sounds, appear, "pop")
        for number, (item, (cx, cy)) in enumerate(zip(items, spots)):
            card, _ = self._card(item, box, int(H * (0.055 if not theme.portrait else 0.03)), int(W * (0.34 if not theme.portrait else 0.42)), label_on_top=False, beside=beside)
            overlays.append(self._centered(pop_frames(card), folder, f"outcome{number}", cx, cy, item.time, scene))
            self._sound(sounds, item.time, "pop")
            if theme.portrait:
                p0 = (origin[0], origin[1] + hub.height / 2 + theme.px(6))
                p1 = (cx, cy - card.height / 2 - theme.px(8))
            else:
                p0 = (origin[0] + hub.width / 2 + theme.px(6), origin[1])
                p1 = (cx - card.width / 2 - theme.px(10), cy)
            frames, (left, top) = arrow_frames(p0, p1, theme.px(6), frames=8, bend=0.12 if cy < origin[1] else -0.12 if cy > origin[1] else 0.0)
            overlays.append(self._frames(frames, folder, f"arrow{number}", left, top, item.time - 0.2, scene))

    # -- doodle shots -------------------------------------------------------------------

    def _title(self, scene, folder, overlays, sounds, text: str, when: float) -> float:
        """A hand-lettered title at the top of a shot; returns where the space below it starts."""
        W, H = self.theme.width, self.theme.height
        if not text:
            return H * 0.06
        image = self.text.render(text.upper(), int(H * (0.072 if not self.theme.portrait else 0.04)), int(W * 0.86), max_lines=2)
        top = H * 0.07
        overlays.append(self._frames(pop_frames(image), folder, "title", (W - image.width) / 2, top, when, scene))
        return top + image.height + H * 0.03

    def _build_single(self, scene, folder, overlays, sounds) -> None:
        """One drawing, big, with its label (and an optional title above).

        In the doodle look a soft paper blob pops in behind it and the drawing
        appears as if it were being drawn.
        """
        theme, W, H = self.theme, self.theme.width, self.theme.height
        appear = scene.start + 0.12
        item = scene.center or SceneItem()
        top = self._title(scene, folder, overlays, sounds, scene.text, appear + 0.25)
        room = H * 0.95 - top
        box = int(min(room * (0.78 if item.label else 0.95), W * (0.62 if not theme.portrait else 0.86)))
        card, where = self._card(item, box, int(H * (0.075 if not theme.portrait else 0.04)), int(W * 0.8), label_on_top=False)
        cy = top + room / 2
        image = self.picture(item) if self.doodle and where is not None else None
        if image is not None and not image.info.get("framed"):
            x0, y0, x1, y1 = where
            side = int(max(x1 - x0, y1 - y0) * 1.12)
            blob = spot(int(side * 1.12), side, lighter(self.canvas_color, 0.3), seed=int(scene.start * 10))
            blob_cy = cy - card.height / 2 + (y0 + y1) / 2
            overlays.append(self._centered(pop_frames(blob, frames=8, start_scale=0.6), folder, "spot", W / 2, blob_cy, appear, scene))
            overlays.append(self._centered(reveal_frames(card), folder, "single", W / 2, cy, appear + 0.08, scene))
            self._sound(sounds, appear + 0.08, "scribble")
        else:
            overlays.append(self._centered(pop_frames(card), folder, "single", W / 2, cy, appear, scene))
            self._sound(sounds, appear, "pop")

    def _bubble(self, text: str, width: int, tail_left: bool = True) -> Image.Image:
        """A comic speech bubble (empty when ``text`` is empty) with its tail at the bottom."""
        theme, H = self.theme, self.theme.height
        label = self.text.render(text, int(H * (0.065 if not theme.portrait else 0.034)), int(width * 0.8), max_lines=4) if text else None
        inner_w = width
        inner_h = max(int(H * 0.24 if not theme.portrait else H * 0.12), (label.height if label else 0) + theme.px(70))
        tail = int(inner_h * 0.35)
        line = theme.px(5)

        def paint(draw: ImageDraw.ImageDraw, s: float) -> None:
            w, h, r = inner_w * s, inner_h * s, min(inner_w, inner_h) * 0.35 * s
            fill, ink = (255, 249, 236, 255), INK + (255,)
            base_x = w * (0.18 if tail_left else 0.82)
            tip = (base_x - (0.1 * w if tail_left else -0.1 * w), h + tail * s)
            draw.polygon([(base_x - 0.06 * w, h - line * s * 2), tip, (base_x + 0.06 * w, h - line * s * 2)], fill=fill, outline=ink)
            draw.rounded_rectangle((line * s, line * s, w - line * s, h - line * s), r, fill=fill, outline=ink, width=int(line * s))
            draw.polygon([(base_x - 0.06 * w + line * s, h - line * s * 1.6), tip, (base_x + 0.06 * w - line * s, h - line * s * 1.6)], fill=fill)
            draw.line([(base_x - 0.06 * w, h - line * s), tip, (base_x + 0.06 * w, h - line * s)], fill=ink, width=int(line * s), joint="curve")

        bubble = _supersampled((inner_w, inner_h + tail), paint, scale=2)
        if label is not None:
            bubble.alpha_composite(label, ((inner_w - label.width) // 2, (inner_h - label.height) // 2))
        return bubble

    def _build_speech(self, scene, folder, overlays, sounds) -> None:
        """Someone talking: the speaker, a speech bubble and maybe who listens."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        people = scene.items[:2]
        if not people:
            return
        speaker = people[0]
        portrait = theme.portrait
        box = int(H * (0.56 if not portrait else 0.3))
        card, where = self._card(speaker, box, int(H * (0.06 if not portrait else 0.032)), int(W * 0.4), label_on_top=False)
        spot = (W * (0.22 if not portrait else 0.32), H * (0.62 if not portrait else 0.66))
        overlays.append(self._centered(pop_frames(card), folder, "speaker", *spot, speaker.time, scene))
        self._sound(sounds, speaker.time, "pop")
        bubble = self._bubble(scene.text, int(W * (0.34 if not portrait else 0.7)))
        bubble_spot = (W * (0.52 if not portrait else 0.5), H * (0.3 if not portrait else 0.32))
        when = speaker.time + 0.45
        overlays.append(self._centered(pop_frames(bubble, start_scale=0.6), folder, "bubble", *bubble_spot, when, scene))
        self._sound(sounds, when, "pop")
        if len(people) > 1:
            other = people[1]
            card, _ = self._card(other, box, int(H * (0.06 if not portrait else 0.032)), int(W * 0.4), label_on_top=False)
            other_spot = (W * (0.8 if not portrait else 0.72), H * (0.62 if not portrait else 0.74))
            overlays.append(self._centered(pop_frames(card), folder, "listener", *other_spot, other.time, scene))
            self._sound(sounds, other.time, "pop")

    def _outlined(self, text: str, size: int, width: int) -> Image.Image:
        """Big white letters with a thick ink outline, like a title card over a scene."""
        font = self.text.font(size)
        while font.getlength(text) > width and size > 20:
            size = int(size * 0.9)
            font = self.text.font(size)
        stroke = max(3, size // 9)
        bbox = font.getbbox(text, stroke_width=stroke)
        image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (0, 0, 0, 0))
        ImageDraw.Draw(image).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=WHITE + (255,), stroke_width=stroke, stroke_fill=INK + (255,))
        return image

    _CAMERA = {"in": (1.0, 1.1, 0.0), "out": (1.1, 1.0, 0.0), "left": (1.1, 1.1, -1.0), "right": (1.1, 1.1, 1.0)}

    def _build_illustration(self, scene, folder, overlays, sounds) -> None:
        """A whole illustrated scene filling the screen, the camera slowly moving, with an optional big caption."""
        W, H = self.theme.width, self.theme.height
        image = self.picture(scene.center or SceneItem())
        if image is None:
            raise ValueError("the illustration was not drawn")
        canvas = fit_cover(image, (W, H)).convert("RGBA")
        camera = scene.camera if scene.camera in self._CAMERA else CAMERA_MOVES[int(scene.start * 7) % len(CAMERA_MOVES)]
        zoom_from, zoom_to, pan = self._CAMERA[camera]
        overlays.append(self._clip(scene, canvas, folder, zoom_from, zoom_to, pan=pan))
        if scene.text:
            text = self._outlined(scene.text.upper(), int(H * 0.13), int(W * 0.8))
            when = scene.start + 0.35
            cy = H * (0.84 if not self.theme.portrait else 0.8) - text.height / 2
            overlays.append(self._centered(stamp_frames(text), folder, "caption", W / 2, cy, when, scene))
            self._sound(sounds, when, "stamp")

    def _build_clip(self, scene, folder, overlays, sounds) -> None:
        """A real video: in an ink frame on the drawn background, or filling the screen."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        if not scene.media:
            raise ValueError("the clip has no video")
        item = scene.center or SceneItem()
        output = os.path.join(folder, "video.mp4")
        frames = self._clip_frames(scene)
        if scene.frame == "full":
            video_clip(scene.media, output, frames, (W, H), skip=0.5)
            overlays.append(fx.Overlay(output, x="0", y="0", start=scene.start, end=scene.end, mode="media"))
            if item.label:
                text = self._outlined(item.label.upper(), int(H * 0.1), int(W * 0.8))
                overlays.append(self._centered(stamp_frames(text), folder, "caption", W / 2, H * 0.86 - text.height / 2, scene.start + 0.3, scene))
                self._sound(sounds, scene.start + 0.3, "stamp")
            return
        label = self.text.render(item.label.upper(), int(H * (0.075 if not theme.portrait else 0.04)), int(W * 0.86)) if item.label else None
        top = H * 0.06 + (label.height + H * 0.04 if label is not None else 0)
        room_w, room_h = W * (0.84 if not theme.portrait else 0.9), H * 0.94 - top
        aspect = 16 / 9 if not theme.portrait else 9 / 14
        w = int(min(room_w, room_h * aspect))
        h = int(w / aspect)
        x, y = int((W - w) / 2), int(top + (room_h - h) / 2)
        border = theme.px(7)
        back = paper((W, H), self.canvas_color, vignette=0.0).convert("RGBA")
        shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        offset = theme.px(10)
        ImageDraw.Draw(shadow).rounded_rectangle((x + offset, y + offset, x + w + offset, y + h + offset), theme.px(24), fill=(0, 0, 0, 70))
        back.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(theme.px(10))))
        cover = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(cover)
        draw.rounded_rectangle((x - border // 2, y - border // 2, x + w + border // 2, y + h + border // 2), theme.px(22), outline=INK + (255,), width=border)
        for corner, angle in ((x + w * 0.08, -8), (x + w * 0.92, 8)):
            # Two pieces of tape holding the picture to the page.
            tape = Image.new("RGBA", (theme.px(150), theme.px(46)), (250, 240, 214, 215))
            tape = tape.rotate(angle, resample=Image.BICUBIC, expand=True)
            cover.alpha_composite(tape, (int(corner - tape.width / 2), int(y - tape.height / 2)))
        video_clip(scene.media, output, frames, (W, H), background=back, box=(x, y, w, h), cover=cover, skip=0.5)
        overlays.append(fx.Overlay(output, x="0", y="0", start=scene.start, end=scene.end, mode="media"))
        if label is not None:
            overlays.append(self._frames(pop_frames(label), folder, "label", (W - label.width) / 2, H * 0.06, scene.start + 0.25, scene))
            self._sound(sounds, scene.start + 0.25, "pop")

    def _build_meme(self, scene, folder, overlays, sounds) -> None:
        """A comic reaction cut-in: a punch-in on the reaction, rays behind it and a big caption."""
        theme, W, H = self.theme, self.theme.width, self.theme.height
        frames = self._clip_frames(scene)
        output = os.path.join(folder, "clip.mp4")
        punch = f"1+0.16*pow(max(0,1-on/7),2)+0.04*on/{max(1, frames - 1)}"
        if scene.media:
            video_clip(scene.media, output, frames, (W, H), contain=True)
            overlays.append(fx.Overlay(output, x="0", y="0", start=scene.start, end=scene.end, mode="media"))
        else:
            image = self.picture(scene.center or SceneItem())
            if image is None:
                raise ValueError("the reaction has no picture")
            if image.info.get("framed"):
                # A meme picture: whole, over a blurred, darker copy of itself.
                canvas = fit_cover(image, (W, H)).filter(ImageFilter.GaussianBlur(theme.px(22))).convert("RGBA")
                canvas.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, 70)))
                picture = image.copy()
                picture.thumbnail((int(W * 0.9), int(H * 0.86)), Image.LANCZOS)
                canvas.alpha_composite(picture.convert("RGBA"), ((W - picture.width) // 2, (H - picture.height) // 2))
            else:
                # The otter's reaction on comic rays.
                canvas = sunburst((W, H), lighter(self.theme.accent, 0.25)).convert("RGBA")
                picture = image.copy()
                picture.thumbnail((int(W * 0.8), int(H * (0.78 if scene.text else 0.9))), Image.LANCZOS)
                picture = die_cut(picture, theme.px(8))
                canvas.alpha_composite(picture, ((W - picture.width) // 2, H - picture.height + theme.px(10)))
            path = ken_burns_clip(canvas, output, frames, (W, H), 1.0, 1.0, zoom=punch)
            overlays.append(fx.Overlay(path, x="0", y="0", start=scene.start, end=scene.end, mode="media"))
        self._sound(sounds, scene.start, "boom")
        if scene.text:
            text = self._outlined(scene.text.upper(), int(H * 0.14), int(W * 0.9))
            overlays.append(self._centered(stamp_frames(text), folder, "caption", W / 2, H * 0.06 + text.height / 2, scene.start + 0.15, scene))
