"""
Graphics, sound effects and asset handling for edited list videos.

Everything drawn here is rendered once with Pillow into transparent PNGs and
then animated by ffmpeg overlays, so a long video never pays for per-frame
Python drawing. Sound effects are synthesized with numpy, so the default look
works offline and without licensing questions; a user asset folder can replace
any of them.

Asset folder layout (all optional):

    assets/
      personaje/            character expressions, e.g. feliz.png, pensando.png
                            plus optional talking frames: feliz_habla.png
      sfx/                  whoosh / pop / tick / click / bloop / stamp / scribble
                            (.wav, .mp3 or .ogg)
      suscribete.gif        subscribe animation (.gif, .webm, .mov or .png)
      suscribete.mp3        sound played with the subscribe animation
"""

from __future__ import annotations

import math
import os
import random
import re
import subprocess
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

SFX_NAMES = ("whoosh", "pop", "tick", "click", "bloop", "stamp", "scribble")
SFX_SAMPLE_RATE = 24000
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".m4a", ".flac")
IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
SUBSCRIBE_EXTENSIONS = (".gif", ".webm", ".mov", ".mp4", ".png", ".webp")
CHARACTER_DIRS = ("personaje", "character")
SUBSCRIBE_NAMES = ("suscribete", "subscribe")
TALK_SUFFIXES = ("_habla", "_talk", "_hablando", "_open")

DEFAULT_ACCENT = "#FF4F5E"
_INK = (27, 27, 47)


@dataclass
class Overlay:
    """A picture or animation composited over a segment by ffmpeg.

    ``mode`` selects how ``source`` is read: "still" (one picture held in
    memory), "frames" (an image-sequence pattern starting at ``start``),
    "concat" (an ffconcat list covering the whole segment), "sequence" (an
    ffconcat list starting at ``start``) or "media" (a GIF or video starting at
    ``start``). ``x``/``y`` are ffmpeg overlay expressions.
    """

    source: str
    x: str
    y: str
    start: float = 0.0
    end: float = 0.0  # 0 keeps it until the segment ends
    mode: str = "still"
    fade_in: float = 0.0
    fade_out: float = 0.0
    hold: bool = False  # "frames": keep the last frame until ``end``


def render_badge(picture: Image.Image, size: int, ring: Tuple[int, int, int]) -> Image.Image:
    """A round channel badge: the picture in a white disc with a coloured ring."""
    scale = 3
    big = size * scale
    badge = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    line = int(big * 0.07)
    draw.ellipse((0, 0, big - 1, big - 1), fill=ring + (255,))
    draw.ellipse((line, line, big - 1 - line, big - 1 - line), fill=(255, 255, 255, 255))
    inner = big - line * 2
    fitted = ImageOps.contain(picture.convert("RGBA"), (int(inner * 0.92), int(inner * 0.92)), Image.LANCZOS)
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    layer.alpha_composite(fitted, ((big - fitted.width) // 2, line + inner - fitted.height))
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse((line, line, big - 1 - line, big - 1 - line), fill=255)
    clipped = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    clipped.paste(layer, (0, 0), Image.composite(layer.getchannel("A"), Image.new("L", (big, big), 0), mask))
    badge.alpha_composite(clipped)
    return badge.resize((size, size), Image.LANCZOS)


def write_ffconcat(path: str, entries) -> str:
    """An ffconcat list playing (picture, seconds) entries one after another."""
    lines = ["ffconcat version 1.0"]
    for picture, seconds in entries:
        lines.append(f"file '{os.path.abspath(picture)}'")
        lines.append(f"duration {seconds:.4f}")
    if entries:
        # The concat demuxer ignores the last duration unless the file repeats.
        lines.append(f"file '{os.path.abspath(entries[-1][0])}'")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return path


def media_duration(ffmpeg_binary: str, media_file: str) -> float:
    """Duration reported by ffmpeg, or 0.0 when it cannot be read."""
    result = subprocess.run(
        [ffmpeg_binary, "-hide_banner", "-i", media_file],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@dataclass
class Theme:
    width: int
    height: int
    font_path: str
    accent: Tuple[int, int, int]

    @property
    def unit(self) -> float:
        return min(self.width, self.height) / 1080

    @property
    def portrait(self) -> bool:
        return self.height > self.width

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.unit)))


def parse_color(value: str) -> Tuple[int, int, int]:
    text = (value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"invalid color: {value!r}; use #RRGGBB")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _font(theme: Theme, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(theme.font_path, max(8, int(size)))


def _with_shadow(image: Image.Image, radius: int, offset: int, opacity: int = 110) -> Image.Image:
    """Return ``image`` on a larger canvas with a soft drop shadow."""
    pad = radius * 3 + offset
    canvas = Image.new("RGBA", (image.width + pad * 2, image.height + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    mask = image.getchannel("A").point(lambda a: a * opacity // 255)
    shadow.paste((0, 0, 0, 255), (pad, pad + offset), mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(image, (pad, pad))
    return canvas


def _fit_text(theme: Theme, text: str, size: float, max_width: int) -> ImageFont.FreeTypeFont:
    font = _font(theme, size)
    while font.size > 12 and font.getlength(text) > max_width:
        font = _font(theme, font.size * 0.92)
    return font


def render_chapter_chip(theme: Theme, number: Optional[int], title: str) -> Image.Image:
    """Top-left chapter label: accent number block plus the item name."""
    height = theme.px(88)
    radius = theme.px(20)
    pad = theme.px(26)
    max_title = int(theme.width * (0.62 if not theme.portrait else 0.72))
    title_font = _fit_text(theme, title, theme.px(42), max_title)
    number_text = f"{number:02d}" if number is not None else ""
    number_font = _font(theme, theme.px(46))
    number_width = height + theme.px(10) if number_text else 0
    title_width = int(title_font.getlength(title))
    width = number_width + pad + title_width + pad

    chip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(chip)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius, fill=(255, 255, 255, 245))
    if number_text:
        draw.rounded_rectangle((0, 0, number_width, height - 1), radius, fill=theme.accent + (255,))
        # Square off the inner corners of the number block.
        draw.rectangle((number_width - radius, 0, number_width, height - 1), fill=theme.accent + (255,))
        draw.text(
            (number_width / 2, height / 2),
            number_text,
            font=number_font,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )
    draw.text(
        (number_width + pad, height / 2),
        title,
        font=title_font,
        fill=_INK + (255,),
        anchor="lm",
    )
    return _with_shadow(chip, theme.px(10), theme.px(6))


def shadow_padding(theme: Theme) -> int:
    """Transparent border that render_chapter_chip adds around the chip."""
    return theme.px(10) * 3 + theme.px(6)


def render_callout(theme: Theme, text: str) -> Image.Image:
    """Bold key-fact label shown when the narration mentions it."""
    max_width = int(theme.width * (0.62 if not theme.portrait else 0.86))
    font = _fit_text(theme, text, theme.px(70), max_width)
    pad_x, pad_y = theme.px(40), theme.px(22)
    left, top, right, bottom = font.getbbox(text, stroke_width=0)
    width = (right - left) + pad_x * 2
    height = (bottom - top) + pad_y * 2
    pill = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pill)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), theme.px(24), fill=theme.accent + (255,))
    # A thin lighter band on top gives the pill some depth.
    highlight = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(highlight).rounded_rectangle(
        (theme.px(6), theme.px(5), width - theme.px(6), height // 2),
        theme.px(18),
        fill=(255, 255, 255, 38),
    )
    pill.alpha_composite(highlight)
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=(255, 255, 255, 255))
    return _with_shadow(pill, theme.px(12), theme.px(8))


def render_progress_bar(theme: Theme) -> Image.Image:
    height = theme.px(9)
    bar = Image.new("RGBA", (theme.width, height), theme.accent + (235,))
    return bar


def remove_flat_background(image: Image.Image, tolerance: int = 26) -> Optional[Image.Image]:
    """Cut out a uniform background that touches the image border.

    Diagrams and illustrations usually sit on a plain white or coloured field.
    When at least 85% of the border has one colour, pixels of that colour that
    are connected to the border become transparent. Photos are left alone.
    """
    rgb = image.convert("RGB")
    small = rgb.copy()
    small.thumbnail((360, 360))
    pixels = np.asarray(small).astype(np.int16)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    background = np.median(border, axis=0)
    close = np.abs(border - background).max(axis=1) <= tolerance
    if close.mean() < 0.85:
        return None
    # A light page (diagrams, clip art) or a perfectly flat digital colour.
    # Dark or noisy backdrops are photos: cutting them shreds the subject.
    luminance = float(np.dot(background, (0.299, 0.587, 0.114)))
    flatness = float(np.abs(border - background).mean())
    if luminance < 200 and flatness > 2.5:
        return None

    similar = np.abs(pixels - background).max(axis=2) <= tolerance
    height, width = similar.shape
    reached = np.zeros_like(similar)
    stack = [(y, x) for y in range(height) for x in (0, width - 1)]
    stack += [(y, x) for x in range(width) for y in (0, height - 1)]
    while stack:
        y, x = stack.pop()
        if reached[y, x] or not similar[y, x]:
            continue
        reached[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y < height - 1:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x < width - 1:
            stack.append((y, x + 1))
    if reached.mean() > 0.97 or reached.mean() < 0.03:
        return None
    if main_shape_share(~reached) < 0.85:
        return None  # the "subject" would fall apart into pieces

    mask = Image.fromarray(np.where(reached, 0, 255).astype(np.uint8))
    mask = mask.resize(rgb.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2))
    cutout = rgb.convert("RGBA")
    cutout.putalpha(ImageChops.multiply(mask, image.convert("RGBA").getchannel("A")))
    return cutout


def main_shape_share(mask: np.ndarray, side: int = 160) -> float:
    """Share of the ``True`` pixels that belong to the largest connected shape."""
    from collections import deque

    mask = np.asarray(mask, dtype=bool)
    step = max(1, int(math.ceil(max(mask.shape) / side)))
    grid = mask[::step, ::step]
    total = int(grid.sum())
    if total == 0:
        return 0.0
    seen = np.zeros_like(grid)
    height, width = grid.shape
    largest = 0
    for y, x in zip(*np.nonzero(grid)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue, size = deque([(y, x)]), 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        largest = max(largest, size)
    return largest / total


def has_transparency(image: Image.Image) -> bool:
    return image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 245


def cutout_or_none(image: Image.Image, allow_cutout: bool = True) -> Optional[Image.Image]:
    """A clean transparent cut-out of ``image``, or None when it should stay a photo."""
    image = image.convert("RGBA")
    if has_transparency(image):
        alpha = np.asarray(image.getchannel("A")) > 40
        # Scattered transparent fragments look like confetti as stickers.
        return image if main_shape_share(alpha) >= 0.6 else None
    if not allow_cutout:
        return None
    return remove_flat_background(image)


def _fit_box(image: Image.Image, width: int, height: int, most: float = 2.0) -> Image.Image:
    """As large as fits in width x height; small pictures grow up to ``most`` times."""
    scale = min(width / max(1, image.width), height / max(1, image.height), most)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.LANCZOS) if size != image.size else image


def make_sticker(
    theme: Theme,
    image_path: str,
    max_width: int,
    max_height: int,
    seed: int = 0,
    allow_cutout: bool = True,
) -> Image.Image:
    """Turn a picture into an on-screen element.

    Pictures with transparency (or a light, flat background that comes off as
    one clean shape) become die-cut stickers with a white outline; photos and
    anything that would cut out badly become rounded cards with a white
    frame, tilted slightly. ``allow_cutout=False`` always makes a card.
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    rng = random.Random(seed)

    cutout = cutout_or_none(image, allow_cutout)
    if cutout is None and has_transparency(image):
        flat = Image.new("RGBA", image.size, (255, 255, 255, 255))
        flat.alpha_composite(image)
        image = flat
    elif cutout is not None:
        image = cutout

    if cutout is not None:
        bbox = image.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
        if bbox:
            image = image.crop(bbox)
        outline = theme.px(9)
        image = _fit_box(image, max_width - outline * 2, max_height - outline * 2)
        canvas = Image.new("RGBA", (image.width + outline * 2, image.height + outline * 2), (0, 0, 0, 0))
        canvas.paste(image, (outline, outline), image)
        alpha = canvas.getchannel("A").point(lambda a: 255 if a > 40 else 0)
        grown = alpha.filter(ImageFilter.MaxFilter(outline * 2 + 1)).filter(ImageFilter.GaussianBlur(1))
        sticker = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        sticker.putalpha(grown)
        sticker.alpha_composite(canvas)
        return _with_shadow(sticker, theme.px(14), theme.px(8), opacity=120)

    border = theme.px(12)
    radius = theme.px(26)
    image = _fit_box(image, max_width - border * 2, max_height - border * 2)
    card = Image.new("RGBA", (image.width + border * 2, image.height + border * 2), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle(
        (0, 0, card.width - 1, card.height - 1), radius, fill=(255, 255, 255, 255)
    )
    photo_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(photo_mask).rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1), max(1, radius - border // 2), fill=255
    )
    card.paste(image, (border, border), photo_mask)
    card = card.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True)
    return _with_shadow(card, theme.px(16), theme.px(10), opacity=130)


# ---------------------------------------------------------------------------
# Subscribe animation
# ---------------------------------------------------------------------------


def subscribe_labels(language: str) -> Tuple[str, str]:
    lang = (language or "").lower()
    if lang.startswith("en"):
        return "SUBSCRIBE", "SUBSCRIBED"
    if lang.startswith("pt"):
        return "INSCREVA-SE", "INSCRITO"
    return "SUSCRÍBETE", "SUSCRITO"


def _ease_out(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 1 - (1 - value) ** 3


def _draw_cursor(draw: ImageDraw.ImageDraw, x: float, y: float, size: float) -> None:
    points = [
        (0, 0), (0, 0.78), (0.2, 0.6), (0.34, 0.92), (0.47, 0.86), (0.33, 0.55), (0.58, 0.55),
    ]
    shape = [(x + px * size, y + py * size) for px, py in points]
    draw.polygon(shape, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))
    draw.line(shape + [shape[0]], fill=(20, 20, 20, 255), width=max(2, int(size * 0.05)))


def render_subscribe_frames(
    theme: Theme, out_dir: str, language: str, fps: int, duration: float = 3.6
) -> Tuple[str, float]:
    """Draw a subscribe button that gets clicked; returns (pattern, click_time)."""
    label, done_label = subscribe_labels(language)
    font = _font(theme, theme.px(46))
    pad_x, height = theme.px(46), theme.px(104)
    icon = theme.px(46)
    width = int(max(font.getlength(label), font.getlength(done_label)) + icon + pad_x * 2 + theme.px(22))
    frame_w, frame_h = width + theme.px(160), height + theme.px(150)
    click_time = 1.35
    os.makedirs(out_dir, exist_ok=True)
    frames = int(round(duration * fps))
    for index in range(frames):
        t = index / fps
        frame = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
        appear = _ease_out(t / 0.35)
        vanish = 1 - _ease_out((t - (duration - 0.35)) / 0.35)
        opacity = min(appear, vanish)
        if opacity <= 0:
            frame.save(os.path.join(out_dir, f"sub_{index:04d}.png"))
            continue
        clicked = t >= click_time
        press = 0.94 if click_time <= t < click_time + 0.12 else 1.0
        button = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(button)
        fill = (120, 120, 128, 255) if clicked else (230, 33, 23, 255)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), height // 2, fill=fill)
        # Play-button icon (or a check mark once subscribed).
        cx, cy = pad_x + icon / 2, height / 2
        if clicked:
            draw.line(
                [(cx - icon * 0.35, cy), (cx - icon * 0.08, cy + icon * 0.28), (cx + icon * 0.4, cy - icon * 0.3)],
                fill=(255, 255, 255, 255),
                width=theme.px(9),
                joint="curve",
            )
        else:
            draw.rounded_rectangle(
                (cx - icon / 2, cy - icon * 0.36, cx + icon / 2, cy + icon * 0.36),
                theme.px(10),
                fill=(255, 255, 255, 255),
            )
            draw.polygon(
                [(cx - icon * 0.12, cy - icon * 0.17), (cx - icon * 0.12, cy + icon * 0.17), (cx + icon * 0.2, cy)],
                fill=fill,
            )
        draw.text(
            (pad_x + icon + theme.px(22), height / 2),
            done_label if clicked else label,
            font=font,
            fill=(255, 255, 255, 255),
            anchor="lm",
        )
        if press != 1.0:
            button = button.resize((int(width * press), int(height * press)), Image.LANCZOS)
        button = _with_shadow(button, theme.px(10), theme.px(6))
        bx = (frame_w - button.width) // 2
        by = int((frame_h - button.height) // 2 + (1 - appear) * theme.px(60))
        frame.alpha_composite(button, (max(0, bx), max(0, by)))

        # Cursor glides in, clicks, and leaves with the button.
        if 0.55 <= t:
            travel = _ease_out((t - 0.55) / 0.7)
            target_x = frame_w / 2 + width * 0.18
            target_y = frame_h / 2 + height * 0.1
            cursor_x = frame_w + theme.px(10) - (frame_w + theme.px(10) - target_x) * travel
            cursor_y = frame_h - (frame_h - target_y) * travel
            _draw_cursor(ImageDraw.Draw(frame), cursor_x, cursor_y, theme.px(58) * press)

        if opacity < 1:
            alpha = frame.getchannel("A").point(lambda a: int(a * opacity))
            frame.putalpha(alpha)
        frame.save(os.path.join(out_dir, f"sub_{index:04d}.png"))
    return os.path.join(out_dir, "sub_%04d.png"), click_time


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------


@dataclass
class CharacterPose:
    idle: str
    talk: str = ""


def load_character(assets_dir: str) -> Dict[str, CharacterPose]:
    """Map expression name -> pose from ``assets/personaje``."""
    if not assets_dir:
        return {}
    folder = ""
    for name in CHARACTER_DIRS:
        candidate = os.path.join(assets_dir, name)
        if os.path.isdir(candidate):
            folder = candidate
            break
    if not folder:
        return {}
    files = {}
    for entry in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(entry)
        if ext.lower() in IMAGE_EXTENSIONS:
            files[stem.lower()] = os.path.join(folder, entry)
    poses: Dict[str, CharacterPose] = {}
    for stem, path in files.items():
        if any(stem.endswith(suffix) for suffix in TALK_SUFFIXES):
            continue
        talk = next((files[stem + s] for s in TALK_SUFFIXES if stem + s in files), "")
        poses[stem] = CharacterPose(idle=path, talk=talk)
    return poses


def prepare_character_image(source: str, output: str, target_height: int, mirror: bool = False) -> str:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
    if not has_transparency(image):
        # Characters drawn by image generators usually come on a plain
        # background; cut it out so no box shows around the host.
        image = remove_flat_background(image) or image
    bbox = image.getchannel("A").getbbox()
    if bbox:
        image = image.crop(bbox)
    scale = target_height / image.height
    image = image.resize((max(1, int(image.width * scale)), target_height), Image.LANCZOS)
    if mirror:
        image = ImageOps.mirror(image)
    image.save(output)
    return output


def mouth_schedule(
    pcm: bytes, sample_rate: int, fps: int, min_frames: int = 2
) -> List[Tuple[bool, int]]:
    """Open/closed mouth runs (in frames) following the narration volume."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    per_frame = sample_rate // fps
    frames = len(samples) // per_frame
    if frames == 0:
        return []
    energy = np.sqrt(
        np.mean(samples[: frames * per_frame].reshape(frames, per_frame) ** 2, axis=1)
    )
    voiced = energy[energy > 1]
    if voiced.size == 0:
        return [(False, frames)]
    threshold = max(250.0, float(np.percentile(voiced, 75)) * 0.35)
    states = energy > threshold
    runs: List[List] = []
    for state in states:
        if runs and runs[-1][0] == bool(state):
            runs[-1][1] += 1
        else:
            runs.append([bool(state), 1])
    # Merge flickers shorter than min_frames into the previous run.
    merged: List[List] = []
    for state, count in runs:
        if merged and (count < min_frames or merged[-1][0] == state):
            merged[-1][1] += count
        else:
            merged.append([state, count])
    return [(state, count) for state, count in merged]


# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------


def synthesize_sfx(name: str, sample_rate: int = SFX_SAMPLE_RATE) -> np.ndarray:
    """Return a float32 mono sound effect in [-1, 1]."""
    rng = np.random.default_rng(7)
    if name == "whoosh":
        duration = 0.55
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        noise = rng.standard_normal(t.size)
        # Sweep a one-pole low-pass from dark to bright and back.
        cutoff = 0.02 + 0.2 * np.sin(np.pi * t / duration) ** 2
        out = np.zeros_like(noise)
        state = 0.0
        for i, sample in enumerate(noise):
            state += cutoff[i] * (sample - state)
            out[i] = state
        envelope = np.sin(np.pi * np.clip(t / duration, 0, 1)) ** 2
        sound = out * envelope
        return (0.55 * sound / (np.abs(sound).max() or 1)).astype(np.float32)
    if name == "pop":
        duration = 0.14
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq = 280 + 620 * np.exp(-t * 40)
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        sound = np.sin(phase) * np.exp(-t * 32)
        sound[: int(0.004 * sample_rate)] *= np.linspace(0, 1, int(0.004 * sample_rate))
        return (0.5 * sound).astype(np.float32)
    if name == "bloop":
        # A water-drop "bloop" for the otter popping into frame.
        duration = 0.18
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq = 240 * (1 + 3.4 * (t / duration) ** 1.4)
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        attack = np.clip(t / 0.008, 0, 1)
        sound = np.sin(phase) * attack * np.exp(-t * 20)
        return (0.5 * sound).astype(np.float32)
    if name == "stamp":
        # A rubber stamp landing: a short low thump with a papery slap.
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        thump = np.sin(2 * np.pi * (95 + 60 * np.exp(-t * 40)) * t) * np.exp(-t * 28)
        slap = rng.standard_normal(t.size) * np.exp(-t * 90)
        sound = 0.8 * thump + 0.35 * slap
        return (0.6 * sound / (np.abs(sound).max() or 1)).astype(np.float32)
    if name == "scribble":
        # A marker drawing a line: band-limited noise with a stroke rhythm.
        duration = 0.36
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        noise = rng.standard_normal(t.size)
        smooth = np.convolve(noise, np.ones(6) / 6, mode="same")
        bright = noise - smooth
        rhythm = 0.55 + 0.45 * np.sin(2 * np.pi * 11 * t) ** 2
        envelope = np.sin(np.pi * t / duration) ** 0.7
        sound = bright * rhythm * envelope
        return (0.3 * sound / (np.abs(sound).max() or 1)).astype(np.float32)
    if name == "tick":
        duration = 0.09
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sound = (np.sin(2 * np.pi * 1760 * t) + 0.5 * np.sin(2 * np.pi * 2640 * t)) * np.exp(-t * 55)
        return (0.22 * sound).astype(np.float32)
    if name == "click":
        duration = 0.12
        samples = int(sample_rate * duration)
        sound = np.zeros(samples, dtype=np.float32)
        for start in (0.0, 0.055):
            begin = int(start * sample_rate)
            length = int(0.012 * sample_rate)
            burst = rng.standard_normal(length) * np.exp(-np.linspace(0, 6, length))
            sound[begin : begin + length] += burst[: max(0, samples - begin)]
        return (0.35 * sound / (np.abs(sound).max() or 1)).astype(np.float32)
    raise ValueError(f"unknown sound effect: {name}")


def write_wav(samples: np.ndarray, path: str, sample_rate: int = SFX_SAMPLE_RATE) -> str:
    data = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data.tobytes())
    return path


def find_asset(folder: str, names, extensions) -> str:
    if not folder or not os.path.isdir(folder):
        return ""
    wanted = {name.lower() for name in names}
    for entry in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(entry)
        if stem.lower() in wanted and ext.lower() in extensions:
            return os.path.join(folder, entry)
    return ""


def resolve_sfx(assets_dir: str, work_dir: str) -> Dict[str, str]:
    """Sound effect files: user overrides from assets/sfx, else synthesized."""
    sfx_dir = os.path.join(assets_dir, "sfx") if assets_dir else ""
    resolved = {}
    for name in SFX_NAMES:
        custom = find_asset(sfx_dir, [name], AUDIO_EXTENSIONS)
        if custom:
            resolved[name] = custom
        else:
            resolved[name] = write_wav(synthesize_sfx(name), os.path.join(work_dir, f"sfx-{name}.wav"))
    return resolved


# ---------------------------------------------------------------------------
# Demo assets
# ---------------------------------------------------------------------------

DEMO_EXPRESSIONS = ("neutral", "feliz", "sorprendido", "pensando", "preocupado")


def _draw_mascot(expression: str, talking: bool, size: int = 900) -> Image.Image:
    """A friendly red blood cell host, drawn at 3x and downsampled."""
    scale = 3
    s = size * scale
    image = Image.new("RGBA", (s, int(s * 1.08)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx, cy, r = s / 2, s * 0.56, s * 0.42

    # Body with a darker rim and lighter centre, like a blood cell.
    draw.ellipse((cx - r, cy - r * 0.92, cx + r, cy + r * 0.92), fill=(196, 30, 58, 255))
    draw.ellipse((cx - r * 0.93, cy - r * 0.86, cx + r * 0.93, cy + r * 0.84), fill=(233, 58, 78, 255))
    draw.ellipse((cx - r * 0.55, cy - r * 0.5, cx + r * 0.55, cy + r * 0.42), fill=(242, 92, 104, 255))
    # Glint near the rim, clear of the eyebrows.
    draw.ellipse((cx - r * 0.8, cy - r * 0.66, cx - r * 0.6, cy - r * 0.48), fill=(255, 170, 178, 170))

    # Eyes.
    eye_y = cy - r * 0.18
    eye_dx = r * 0.33
    eye_w, eye_h = r * 0.2, r * (0.3 if expression == "sorprendido" else 0.26)
    look = {"pensando": (-0.05, -0.08), "preocupado": (0, 0.03)}.get(expression, (0, 0))
    for side in (-1, 1):
        ex = cx + side * eye_dx
        draw.ellipse((ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h), fill=(255, 255, 255, 255))
        px = ex + look[0] * r
        py = eye_y + look[1] * r
        pr = eye_w * 0.62
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(30, 22, 40, 255))
        draw.ellipse((px - pr * 0.2, py - pr * 0.75, px + pr * 0.45, py - pr * 0.1), fill=(255, 255, 255, 255))

    # Eyebrows carry most of the emotion.
    # (lift, tilt): a positive tilt raises the inner ends (worried, curious).
    brow = {
        "feliz": (0.06, 0.0),
        "sorprendido": (0.16, 0.0),
        "pensando": (0.1, 0.07),
        "preocupado": (0.08, 0.13),
    }.get(expression, (0.08, 0.0))
    width = int(r * 0.07)
    for side in (-1, 1):
        ex = cx + side * eye_dx
        lift = brow[0] * r
        tilt = brow[1] * r * side
        draw.line(
            [(ex - eye_w * 1.1, eye_y - eye_h - lift - tilt), (ex + eye_w * 1.1, eye_y - eye_h - lift + tilt)],
            fill=(110, 12, 30, 255),
            width=width,
            joint="curve",
        )

    # Cheeks.
    for side in (-1, 1):
        bx = cx + side * r * 0.56
        draw.ellipse((bx - r * 0.12, cy + r * 0.02, bx + r * 0.12, cy + r * 0.12), fill=(255, 140, 150, 170))

    # Mouth.
    my = cy + r * 0.22
    ink = (70, 10, 26, 255)
    if expression == "sorprendido":
        mw, mh = r * 0.16, r * (0.2 if talking else 0.16)
        draw.ellipse((cx - mw, my - mh * 0.6, cx + mw, my + mh), fill=ink)
        draw.ellipse((cx - mw * 0.6, my + mh * 0.25, cx + mw * 0.6, my + mh * 0.9), fill=(235, 110, 120, 255))
    elif talking:
        # Open smile: the lower half of an ellipse, with a tongue.
        mw, mh = r * (0.2 if expression != "preocupado" else 0.15), r * 0.2
        draw.pieslice((cx - mw, my - mh, cx + mw, my + mh), 0, 180, fill=ink)
        draw.pieslice((cx - mw * 0.55, my + mh * 0.25, cx + mw * 0.55, my + mh * 0.95), 180, 360, fill=(235, 110, 120, 255))
    elif expression == "feliz" or expression == "neutral":
        spread = 0.24 if expression == "feliz" else 0.18
        draw.arc(
            (cx - r * spread, my - r * 0.16, cx + r * spread, my + r * 0.12),
            20,
            160,
            fill=ink,
            width=int(r * 0.06),
        )
    elif expression == "pensando":
        draw.line([(cx - r * 0.1, my + r * 0.02), (cx + r * 0.14, my - r * 0.03)], fill=ink, width=int(r * 0.06))
    else:
        draw.arc((cx - r * 0.16, my, cx + r * 0.16, my + r * 0.2), 200, 340, fill=ink, width=int(r * 0.06))

    return image.resize((size, int(size * 1.08)), Image.LANCZOS)


def create_demo_assets(folder: str) -> List[str]:
    """Write a sample character (a blood-cell host) using the expected layout."""
    character_dir = os.path.join(folder, "personaje")
    os.makedirs(character_dir, exist_ok=True)
    written = []
    for expression in DEMO_EXPRESSIONS:
        for talking in (False, True):
            name = f"{expression}_habla.png" if talking else f"{expression}.png"
            path = os.path.join(character_dir, name)
            _draw_mascot(expression, talking).save(path)
            written.append(path)
    os.makedirs(os.path.join(folder, "sfx"), exist_ok=True)
    return written


def ease_expression(duration: float, delay: float = 0.0) -> str:
    """ffmpeg expression for an ease-out progress from 0 to 1."""
    return f"(1-pow(1-clip((t-{delay:.3f})/{max(duration, 0.01):.3f},0,1),3))"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_seed(text: str) -> int:
    return sum(ord(ch) * (i + 1) for i, ch in enumerate(text)) % 100000

