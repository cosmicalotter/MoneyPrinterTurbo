"""
Build the otter host ("la nutria con lentes") used by list videos.

Every pose is assembled from the same SVG parts (sweater, head, round glasses,
the pencil behind the ear) plus the pieces that carry the emotion (brows, eyes,
mouth, paws and a small head tilt). The script writes the editable SVG sources
to ``svg/`` and the PNGs the renderer uses to ``personaje/``:
``<expression>.png`` with the mouth at rest and ``<expression>_habla.png`` with
the mouth open for lip-sync.

Rasterizing needs a Chromium-based browser (Chromium, Chrome or Playwright's
headless shell); set CHROME=/path/to/binary if it is not found automatically.

    python resource/characters/nutria/build_otter.py
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH, HEIGHT = 680, 720
SCALE = 2  # PNGs are rendered at twice the SVG size before trimming

FUR = "#9C6240"
FUR_DARK = "#6E4129"
FUR_LIGHT = "#B57C55"
CREAM = "#F3DDC3"
CREAM_LIGHT = "#FBEFE1"
NOSE = "#2B1A15"
EYE = "#24160F"
BROW = "#5A321E"
SWEATER = "#2A9D8F"
SWEATER_LIGHT = "#4DBBAC"
SWEATER_DARK = "#1C7A6F"
GLASSES = "#22263A"
BLUSH = "#F4978E"
MOUTH = "#4A1C18"
TONGUE = "#E8736F"
TEAR = "#8ED1FC"
PENCIL = "#FFC745"
PENCIL_DARK = "#E9A92A"

CX = 300
HEAD_CY, HEAD_RX, HEAD_RY = 236, 178, 140
NECK = (CX, 384)
EYE_L, EYE_R, EYE_Y = 236, 364, 206
SHOULDER_L, SHOULDER_R = (222, 444), (378, 444)

BODY = (
    "M 214 430 C 214 384 256 360 300 360 C 344 360 386 384 386 430 "
    "C 420 496 444 574 440 646 C 437 698 404 720 362 720 L 238 720 "
    "C 196 720 163 698 160 646 C 156 574 180 496 214 430 Z"
)


def _stroke(color: str, width: float, opacity: float = 1.0) -> str:
    extra = f' opacity="{opacity}"' if opacity != 1.0 else ""
    return f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{extra}'


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def tail() -> str:
    return (
        f'<path d="M 396 702 C 468 716 544 692 570 630 C 584 596 582 562 568 542 '
        f'C 560 532 546 538 550 552 C 556 590 536 628 492 648 C 462 662 430 666 402 662 Z" fill="{FUR_DARK}"/>'
        f'<path d="M 470 676 C 512 668 542 640 556 606" {_stroke(FUR, 6, 0.45)}/>'
    )


def body() -> str:
    ribs = "".join(
        f'<line x1="{x}" y1="672" x2="{x}" y2="716" {_stroke(SWEATER_DARK, 5, 0.55)}/>'
        for x in range(176, 432, 20)
    )
    collar_ribs = "".join(
        f'<line x1="{CX + dx}" y1="380" x2="{CX + dx}" y2="{402 - abs(dx) // 9}" {_stroke(SWEATER_DARK, 5, 0.45)}/>'
        for dx in range(-80, 81, 20)
    )
    return f"""
  <path d="{BODY}" fill="{SWEATER}"/>
  <g clip-path="url(#bodyclip)">
    <ellipse cx="{CX - 70}" cy="520" rx="60" ry="110" fill="{SWEATER_LIGHT}" opacity="0.28"/>
    <rect x="140" y="664" width="320" height="60" fill="{SWEATER_DARK}" opacity="0.55"/>
    {ribs}
  </g>
  <rect x="{CX - 98}" y="360" width="196" height="48" rx="24" fill="{SWEATER_LIGHT}"/>
  {collar_ribs}
"""


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def pencil() -> str:
    """Tucked behind the right ear: the eraser end pokes out above it."""
    return f"""
  <g transform="rotate(-32 466 176)">
    <rect x="380" y="162" width="132" height="28" rx="5" fill="{PENCIL}"/>
    <rect x="380" y="176" width="132" height="14" rx="5" fill="{PENCIL_DARK}"/>
    <rect x="510" y="160" width="16" height="32" rx="4" fill="#C9CDD6"/>
    <rect x="522" y="160" width="24" height="32" rx="10" fill="#F28B9A"/>
  </g>
"""


def ears(droop: bool = False) -> str:
    out = []
    for side in (-1, 1):
        x = CX + side * 166
        y = 186 + (22 if droop else 0)
        out.append(f'<circle cx="{x}" cy="{y}" r="30" fill="{FUR}"/>')
        out.append(f'<circle cx="{x + side * 3}" cy="{y + 2}" r="15" fill="{FUR_DARK}"/>')
    return "".join(out)


def head() -> str:
    return f"""
  <ellipse cx="{CX}" cy="{HEAD_CY}" rx="{HEAD_RX}" ry="{HEAD_RY}" fill="{FUR}"/>
  <path d="M 274 104 C 266 82 278 62 296 56 C 291 74 297 86 305 98
           C 311 80 328 70 346 74 C 332 84 328 98 328 108 Z" fill="{FUR}"/>
  <g clip-path="url(#headclip)">
    <ellipse cx="{CX}" cy="128" rx="128" ry="42" fill="{FUR_LIGHT}" opacity="0.45"/>
    <ellipse cx="{CX}" cy="326" rx="158" ry="96" fill="{CREAM}"/>
  </g>
  <circle cx="{CX - 38}" cy="300" r="46" fill="{CREAM_LIGHT}"/>
  <circle cx="{CX + 38}" cy="300" r="46" fill="{CREAM_LIGHT}"/>
  {''.join(f'<circle cx="{CX + s * dx}" cy="{y}" r="3.6" fill="{FUR_DARK}" opacity="0.35"/>' for s in (-1, 1) for dx, y in ((52, 292), (64, 306), (46, 312)))}
"""


def whiskers() -> str:
    lines = []
    for side in (-1, 1):
        x0 = CX + side * 82
        for dy, bend in ((-6, -14), (12, 4)):
            y0 = 300 + dy
            lines.append(
                f'<path d="M {x0} {y0} Q {x0 + side * 40} {y0 + bend - 4} {x0 + side * 78} {y0 + bend}" '
                f'{_stroke(FUR_DARK, 3.4, 0.45)}/>'
            )
    return "".join(lines)


def nose() -> str:
    return (
        f'<path d="M 268 262 C 272 246 328 246 332 262 C 334 276 314 290 300 292 '
        f'C 286 290 266 276 268 262 Z" fill="{NOSE}"/>'
        f'<ellipse cx="288" cy="258" rx="10" ry="5" fill="#FFFFFF" opacity="0.5"/>'
    )


def blush(strength: float) -> str:
    return "".join(
        f'<ellipse cx="{CX + side * 118}" cy="292" rx="28" ry="15" fill="{BLUSH}" opacity="{strength}"/>'
        for side in (-1, 1)
    )


def glasses(dy: float = 0, tilt: float = 0) -> str:
    lenses = []
    for cx in (EYE_L, EYE_R):
        lenses.append(
            f'<circle cx="{cx}" cy="{EYE_Y}" r="54" fill="#FFFFFF" fill-opacity="0.18" '
            f'stroke="{GLASSES}" stroke-width="10"/>'
        )
        lenses.append(
            f'<path d="M {cx - 34} {EYE_Y - 14} Q {cx - 28} {EYE_Y - 34} {cx - 10} {EYE_Y - 40}" '
            f'{_stroke("#FFFFFF", 7, 0.75)}/>'
        )
        lenses.append(f'<circle cx="{cx + 30}" cy="{EYE_Y + 28}" r="4.5" fill="#FFFFFF" opacity="0.6"/>')
    return f"""
  <g transform="translate(0 {dy}) rotate({tilt} {CX} {EYE_Y})">
    <path d="M {EYE_L - 54} {EYE_Y - 8} L {CX - HEAD_RX + 14} {EYE_Y - 18}" {_stroke(GLASSES, 9)}/>
    <path d="M {EYE_R + 54} {EYE_Y - 8} L {CX + HEAD_RX - 14} {EYE_Y - 18}" {_stroke(GLASSES, 9)}/>
    {''.join(lenses)}
    <path d="M {CX - 12} {EYE_Y - 4} Q {CX} {EYE_Y - 16} {CX + 12} {EYE_Y - 4}" {_stroke(GLASSES, 9)}/>
  </g>
"""


BROW_SHAPES = {
    # (outer, inner) height offsets; negative raises the brow.
    "rest": (0, 0),
    "up": (-12, -12),
    "high": (-22, -22),
    "worried": (6, -18),
    "sad": (10, -14),
    "focus": (-6, 10),
}


def brows(kind: str) -> str:
    base = EYE_Y - 74
    if kind == "quizzical":
        offsets = {-1: (-16, -20), 1: (4, 6)}
    else:
        offsets = {side: BROW_SHAPES[kind] for side in (-1, 1)}
    paths = []
    for side in (-1, 1):
        cx = EYE_L if side == -1 else EYE_R
        outer, inner = offsets[side]
        x_outer, x_inner = cx + side * 30, cx - side * 26
        top = base + min(outer, inner) - 9
        paths.append(
            f'<path d="M {x_outer} {base + outer} Q {cx} {top} {x_inner} {base + inner}" {_stroke(BROW, 12)}/>'
        )
    return "".join(paths)


def _eye(kind: str, x: float, y: float, side: int) -> str:
    if kind == "normal":
        return (
            f'<ellipse cx="{x}" cy="{y}" rx="17" ry="21" fill="{EYE}"/>'
            f'<circle cx="{x + 6}" cy="{y - 8}" r="6.5" fill="#FFFFFF"/>'
            f'<circle cx="{x - 6}" cy="{y + 9}" r="2.6" fill="#FFFFFF" opacity="0.85"/>'
        )
    if kind == "wide":
        return (
            f'<circle cx="{x}" cy="{y}" r="24" fill="{EYE}"/>'
            f'<circle cx="{x + 8}" cy="{y - 9}" r="8" fill="#FFFFFF"/>'
            f'<circle cx="{x - 8}" cy="{y + 10}" r="3.4" fill="#FFFFFF"/>'
        )
    if kind == "dots":
        return f'<circle cx="{x}" cy="{y}" r="7.5" fill="{EYE}"/>'
    if kind == "happy":
        return f'<path d="M {x - 19} {y + 7} Q {x} {y - 17} {x + 19} {y + 7}" {_stroke(EYE, 8.5)}/>'
    if kind == "laugh":
        return (
            f'<path d="M {x - side * 16} {y - 13} L {x + side * 13} {y} L {x - side * 16} {y + 13}" '
            f'{_stroke(EYE, 8.5)}/>'
        )
    if kind == "sad":
        return (
            f'<ellipse cx="{x}" cy="{y + 5}" rx="16" ry="17" fill="{EYE}"/>'
            f'<path d="M {x - 24} {y - 12} L {x + 24} {y - 12} L {x + 24} {y - 2 + side * 6} '
            f'Q {x} {y + 2} {x - 24} {y - 2 - side * 6} Z" fill="{FUR}"/>'
            f'<path d="M {x - 22} {y - 1 - side * 6} Q {x} {y + 3} {x + 22} {y - 1 + side * 6}" {_stroke(EYE, 4)}/>'
            f'<circle cx="{x + 5}" cy="{y + 8}" r="4.5" fill="#FFFFFF"/>'
        )
    if kind == "sparkle":
        star = " ".join(
            f"{x + 3 + r * math.cos(math.radians(a))},{y - 3 + r * math.sin(math.radians(a))}"
            for a, r in zip(range(-90, 270, 45), (15, 5.5) * 4)
        )
        return f'<circle cx="{x}" cy="{y}" r="23" fill="{EYE}"/><polygon points="{star}" fill="#FFFFFF"/>'
    raise ValueError(kind)


def eyes(kind: str, look=(0, 0)) -> str:
    lx, ly = look
    if kind == "wink":
        return _eye("normal", EYE_L + lx, EYE_Y + ly, -1) + _eye("happy", EYE_R, EYE_Y, 1)
    return "".join(_eye(kind, cx + lx, EYE_Y + ly, side) for cx, side in ((EYE_L, -1), (EYE_R, 1)))


def mouth(kind: str, open_: bool) -> str:
    philtrum = f'<path d="M {CX} 290 L {CX} 300" {_stroke(NOSE, 5)}/>'
    talking_sizes = {
        "smile": (19, 15), "flat": (15, 12), "side": (15, 12), "frown": (15, 11),
        "wavy": (15, 11), "o": (15, 19), "drop": (20, 38), "grin": (30, 24), "laugh": (34, 30),
    }
    resting_open = {"o": (12, 15), "drop": (18, 32), "grin": (28, 21), "laugh": (32, 27)}
    if open_ or kind in resting_open:
        rx, ry = talking_sizes[kind] if open_ else resting_open[kind]
        top = 302
        if kind in ("grin", "laugh", "smile"):
            shape = (
                f'<path d="M {CX - rx} {top} Q {CX} {top + 6} {CX + rx} {top} '
                f'Q {CX + rx} {top + ry * 2} {CX} {top + ry * 2} Q {CX - rx} {top + ry * 2} {CX - rx} {top} Z" fill="{MOUTH}"/>'
            )
        else:
            shape = f'<ellipse cx="{CX}" cy="{top + ry}" rx="{rx}" ry="{ry}" fill="{MOUTH}"/>'
        tongue = f'<ellipse cx="{CX}" cy="{top + ry * 1.55}" rx="{rx * 0.58}" ry="{ry * 0.4}" fill="{TONGUE}"/>'
        outline = shape.replace(f' fill="{MOUTH}"', "")
        clip = f'<clipPath id="mouthclip">{outline}</clipPath>'
        return philtrum + clip + shape + f'<g clip-path="url(#mouthclip)">{tongue}</g>'
    shapes = {
        "smile": f"M {CX - 24} 298 Q {CX - 12} 314 {CX} 300 Q {CX + 12} 314 {CX + 24} 298",
        "flat": f"M {CX - 16} 306 Q {CX} 310 {CX + 16} 306",
        "side": f"M {CX - 8} 308 Q {CX + 8} 310 {CX + 22} 300",
        "frown": f"M {CX - 20} 316 Q {CX} 300 {CX + 20} 316",
        "wavy": f"M {CX - 22} 310 Q {CX - 15} 302 {CX - 7} 310 Q {CX} 318 {CX + 7} 310 Q {CX + 15} 302 {CX + 22} 310",
    }
    return philtrum + f'<path d="{shapes[kind]}" {_stroke(NOSE, 5.5)}/>'


def tear() -> str:
    x, y = EYE_R + 26, EYE_Y + 34
    return f'<path d="M {x} {y} Q {x + 13} {y + 22} {x} {y + 31} Q {x - 13} {y + 22} {x} {y} Z" fill="{TEAR}"/>'


def sweat() -> str:
    return f'<path d="M 452 112 Q 472 140 459 156 Q 444 163 440 146 Q 440 132 452 112 Z" fill="{TEAR}"/>'


def wave_lines() -> str:
    return "".join(
        f'<path d="M {x} {y} q 14 -16 0 -32" {_stroke(BROW, 6, 0.55)}/>' for x, y in ((566, 212), (588, 232))
    )


def shine() -> str:
    """Little sparkles around an excited host."""
    out = []
    for x, y, r in ((96, 86, 16), (520, 70, 12), (88, 330, 10)):
        points = " ".join(
            f"{x + rr * math.cos(math.radians(a))},{y + rr * math.sin(math.radians(a))}"
            for a, rr in zip(range(-90, 270, 45), (r, r * 0.32) * 4)
        )
        out.append(f'<polygon points="{points}" fill="{PENCIL}"/>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Arms and paws
# ---------------------------------------------------------------------------


def sleeve(shoulder, bend, wrist) -> str:
    sx, sy = shoulder
    bx, by = bend
    wx, wy = wrist
    path = f"M {sx} {sy} Q {bx} {by} {wx} {wy}"
    dx, dy = wx - bx, wy - by
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    cuff = f"M {wx - ux * 14} {wy - uy * 14} L {wx} {wy}"
    return (
        f'<path d="{path}" {_stroke(SWEATER_DARK, 60)}/>'
        f'<path d="{path}" {_stroke(SWEATER, 47)}/>'
        f'<path d="{cuff}" {_stroke(SWEATER_DARK, 60)}/>'
        f'<path d="{cuff}" {_stroke(SWEATER_LIGHT, 48)}/>'
    )


def mitten(x, y, angle=90.0) -> str:
    """A closed paw; ``angle`` points from the wrist towards the toes."""
    toes = "".join(
        f'<line x1="14" y1="{dy}" x2="26" y2="{dy * 1.15}" {_stroke(FUR_DARK, 3.2, 0.6)}/>'
        for dy in (-8, 0, 8)
    )
    return (
        f'<g transform="translate({x} {y}) rotate({angle})">'
        f'<ellipse cx="6" cy="0" rx="28" ry="25" fill="{FUR}"/>{toes}</g>'
    )


def palm(x, y, angle=-90.0) -> str:
    """An open paw facing the viewer, toes towards ``angle``."""
    beans = "".join(
        f'<ellipse cx="{24 * math.cos(math.radians(a))}" cy="{24 * math.sin(math.radians(a))}" '
        f'rx="7" ry="8" fill="{CREAM}"/>'
        for a in (-52, -18, 18, 52)
    )
    nubs = "".join(
        f'<circle cx="{25 * math.cos(math.radians(a))}" cy="{25 * math.sin(math.radians(a))}" r="12" fill="{FUR}"/>'
        for a in (-52, -18, 18, 52)
    )
    return (
        f'<g transform="translate({x} {y}) rotate({angle})">'
        f'{nubs}<ellipse cx="0" cy="0" rx="28" ry="30" fill="{FUR}"/>'
        f'<ellipse cx="-4" cy="0" rx="13" ry="15" fill="{CREAM}"/>{beans}</g>'
    )


def pointing(x, y) -> str:
    """Paw pointing to the right with the index toe."""
    return (
        f'<rect x="{x}" y="{y - 22}" width="56" height="20" rx="10" fill="{FUR}"/>'
        f'<ellipse cx="{x}" cy="{y}" rx="30" ry="27" fill="{FUR}"/>'
        f'<path d="M {x + 8} {y + 6} q 14 2 20 -4 M {x + 6} {y + 16} q 14 2 18 -4" {_stroke(FUR_DARK, 3.2, 0.6)}/>'
        f'<ellipse cx="{x - 4}" cy="{y - 22}" rx="12" ry="9" fill="{FUR}"/>'
    )


def arms(kind: str) -> tuple[str, str]:
    """(below the head, over the head) parts for a paw pose."""
    L, R = SHOULDER_L, SHOULDER_R
    if kind == "rest":
        part = (
            sleeve(L, (192, 548), (282, 570)) + sleeve(R, (408, 548), (318, 570))
            + mitten(290, 584, 20) + mitten(312, 590, 160)
        )
        return part, ""
    if kind == "present":
        part = (
            sleeve(L, (196, 548), (288, 574)) + mitten(296, 588, 20)
            + sleeve(R, (440, 540), (478, 470)) + palm(492, 446, -60)
        )
        return part, ""
    if kind == "point":
        part = (
            sleeve(L, (196, 548), (288, 574)) + mitten(296, 588, 20)
            + sleeve(R, (420, 500), (482, 456)) + pointing(500, 448)
        )
        return part, ""
    if kind == "cheeks":
        return "", (
            sleeve(L, (172, 440), (170, 352)) + sleeve(R, (428, 440), (430, 352))
            + palm(172, 312, -96) + palm(428, 312, -84)
        )
    if kind == "up":
        return "", (
            sleeve(L, (150, 400), (120, 290)) + sleeve(R, (450, 400), (480, 290))
            + palm(112, 252, -110) + palm(488, 252, -70)
        )
    if kind == "chin":
        below = sleeve(L, (230, 560), (340, 548)) + mitten(352, 548, 0)
        over = sleeve(R, (430, 520), (360, 420)) + mitten(342, 396, -120)
        return below, over
    if kind == "belly":
        part = sleeve(L, (180, 520), (232, 576)) + sleeve(R, (420, 520), (368, 576))
        part += mitten(244, 588, 50) + mitten(356, 588, 130)
        return part, ""
    if kind == "limp":
        part = sleeve(L, (196, 520), (190, 600)) + sleeve(R, (404, 520), (410, 600))
        part += mitten(188, 622, 95) + mitten(412, 622, 85)
        return part, ""
    if kind == "clutch":
        part = (
            sleeve(L, (186, 520), (282, 494)) + sleeve(R, (414, 520), (318, 494))
            + mitten(292, 480, -70) + mitten(310, 484, -110)
        )
        return part, ""
    if kind == "wave":
        below = sleeve(L, (196, 548), (288, 574)) + mitten(296, 588, 20)
        over = sleeve(R, (470, 420), (500, 320)) + palm(508, 280, -78)
        return below, over
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------

EXPRESSIONS = {
    # name: brows, eyes, mouth, arms, extras
    "neutral": dict(brows="rest", eyes="normal", mouth="smile", arms="rest"),
    "feliz": dict(brows="up", eyes="happy", mouth="smile", arms="rest", blush=0.75, tilt=-5),
    "explicando": dict(brows="quizzical", eyes="normal", mouth="smile", arms="present", tilt=-4),
    "senalando": dict(brows="up", eyes="normal", look=(8, 2), mouth="smile", arms="point", tilt=-3),
    "saludando": dict(brows="up", eyes="wink", mouth="grin", arms="wave", blush=0.7, tilt=-6, wave=True),
    "sorprendido": dict(brows="high", eyes="wide", mouth="o", arms="cheeks"),
    "sin_palabras": dict(brows="high", eyes="dots", mouth="drop", arms="limp", glasses_dy=16, glasses_tilt=-6),
    "pensando": dict(brows="quizzical", eyes="normal", look=(9, -9), mouth="side", arms="chin", tilt=6),
    "triste": dict(brows="sad", eyes="sad", mouth="frown", arms="rest", droop=True, tear=True, blush=0.3, tilt=4, sink=8),
    "preocupado": dict(brows="worried", eyes="normal", look=(0, 3), mouth="wavy", arms="clutch", sweat=True),
    "emocionado": dict(brows="high", eyes="sparkle", mouth="grin", arms="up", blush=0.75, shine=True),
    "riendo": dict(brows="up", eyes="laugh", mouth="laugh", arms="belly", blush=0.8, tilt=-7),
}


def build_svg(name: str, talking: bool) -> str:
    spec = EXPRESSIONS[name]
    below, over = arms(spec["arms"])
    tilt = spec.get("tilt", 0)
    sink = spec.get("sink", 0)
    face_extras = (tear() if spec.get("tear") else "") + (sweat() if spec.get("sweat") else "")
    extras = (wave_lines() if spec.get("wave") else "") + (shine() if spec.get("shine") else "")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <!-- Cabeceando's otter host: {name}{' (talking)' if talking else ''} -->
  <defs>
    <clipPath id="bodyclip"><path d="{BODY}"/></clipPath>
    <clipPath id="headclip"><ellipse cx="{CX}" cy="{HEAD_CY}" rx="{HEAD_RX}" ry="{HEAD_RY}"/></clipPath>
  </defs>
  {tail()}
  {body()}
  {below}
  <g transform="translate(0 {sink}) rotate({tilt} {NECK[0]} {NECK[1]})">
    {pencil()}
    {ears(spec.get('droop', False))}
    {head()}
    {whiskers()}
    {blush(spec.get('blush', 0.5))}
    {eyes(spec['eyes'], spec.get('look', (0, 0)))}
    {glasses(spec.get('glasses_dy', 0), spec.get('glasses_tilt', 0))}
    {brows(spec['brows'])}
    {nose()}
    {mouth(spec['mouth'], talking)}
    {face_extras}
  </g>
  {over}
  {extras}
</svg>
"""


# ---------------------------------------------------------------------------
# Rasterizing
# ---------------------------------------------------------------------------


def find_chrome() -> str:
    candidates = [os.environ.get("CHROME", "")]
    candidates += sorted(glob.glob("/opt/pw-browsers/chromium_headless_shell-*/chrome-linux/headless_shell"))
    candidates += [shutil.which(name) or "" for name in ("chromium", "chromium-browser", "google-chrome", "chrome")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise SystemExit("No Chromium-based browser found; set CHROME=/path/to/chrome")


def rasterize(svg_path: str, png_path: str, chrome: str) -> None:
    with open(svg_path, encoding="utf-8") as fp:
        scaled = fp.read().replace(
            f'width="{WIDTH}" height="{HEIGHT}"', f'width="{WIDTH * SCALE}" height="{HEIGHT * SCALE}"', 1
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        source = os.path.join(temp_dir, "pose.svg")
        with open(source, "w", encoding="utf-8") as fp:
            fp.write(scaled)
        subprocess.run(
            [
                chrome, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
                "--default-background-color=00000000",
                f"--window-size={WIDTH * SCALE},{HEIGHT * SCALE}",
                f"--screenshot={png_path}", f"file://{source}",
            ],
            check=True,
            capture_output=True,
        )


def add_cutout_border(png_path: str, width: int = 14) -> None:
    """White die-cut outline, matching the collage stickers."""
    with Image.open(png_path) as image:
        image = image.convert("RGBA")
    # Same padding for every pose, so they keep sharing one canvas.
    pad = width + 4
    canvas = Image.new("RGBA", (image.width + pad * 2, image.height + pad * 2), (0, 0, 0, 0))
    canvas.alpha_composite(image, (pad, pad))
    alpha = canvas.getchannel("A").point(lambda a: 255 if a > 30 else 0)
    grown = alpha.filter(ImageFilter.MaxFilter(width * 2 + 1)).filter(ImageFilter.GaussianBlur(1.2))
    outlined = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    outlined.putalpha(grown)
    outlined.alpha_composite(canvas)
    outlined.save(png_path)


def crop_to_union(paths, margin: int = 4) -> None:
    """Crop every pose to the same box, so the body stays put between poses."""
    box = None
    for path in paths:
        with Image.open(path) as image:
            bbox = image.getchannel("A").getbbox()
        if bbox:
            box = bbox if box is None else (
                min(box[0], bbox[0]), min(box[1], bbox[1]), max(box[2], bbox[2]), max(box[3], bbox[3])
            )
    if box is None:
        return
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGBA")
        left, top = max(0, box[0] - margin), max(0, box[1] - margin)
        right, bottom = min(image.width, box[2] + margin), min(image.height, box[3] + margin)
        image.crop((left, top, right, bottom)).save(path, optimize=True)


def main() -> int:
    chrome = find_chrome()
    svg_dir = os.path.join(HERE, "svg")
    png_dir = os.path.join(HERE, "personaje")
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    written = []
    for name in EXPRESSIONS:
        for talking in (False, True):
            stem = f"{name}_habla" if talking else name
            svg_path = os.path.join(svg_dir, f"{stem}.svg")
            with open(svg_path, "w", encoding="utf-8") as fp:
                fp.write(build_svg(name, talking))
            png_path = os.path.join(png_dir, f"{stem}.png")
            rasterize(svg_path, png_path, chrome)
            add_cutout_border(png_path)
            written.append(png_path)
    crop_to_union(written)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
