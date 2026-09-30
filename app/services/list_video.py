"""
List-format long videos ("every X explained").

The regular pipeline narrates the whole script in one TTS pass and loops the
material clips until they cover the narration, so a visual never knows which
sentence it belongs to. A list video needs the opposite: while item 3 is being
narrated, item 3's picture and title must be on screen.

This renderer therefore works segment by segment (intro, every item, outro):

1. synthesize every segment narration on its own and measure it;
2. optionally let ``list_video_editor`` plan overlays for each segment
   (chapter label, host character, pictures and key facts timed to words,
   subscribe animation, progress bar) plus sound effects;
3. render the segment's visual (generated image, stock clips or a local file)
   and its overlays with ffmpeg for exactly that length;
4. pad the narration with silence to the rendered frame count, so audio and
   video share one timeline with no accumulated drift;
5. shift the segment subtitles, chapter marks and sound effects by the running
   offset.

Without burned-in subtitles the segments are joined and muxed with ffmpeg only
(background music ducked under the voice, loudness normalized to -14 LUFS).
With subtitles the result goes through ``video.generate_video`` as usual.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.config import config
from app.models import const
from app.models.schema import (
    ListVideoScript,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
)
from app.services import bgm as bgm_service
from app.services import list_video_editor as editor_service
from app.services import list_video_fx as fx
from app.services import material, subtitle, task_artifacts, video, voice
from app.utils import file_security, utils

LIST_VIDEO_SOURCES = ("pexels", "pixabay", "coverr", "openai_image", "local")
STOCK_VIDEO_SOURCES = ("pexels", "pixabay", "coverr")
# Still pictures are fitted once with Pillow; everything else is read by ffmpeg.
IMAGE_EXTENSIONS = frozenset({*const.FILE_TYPE_IMAGES, "webp"})

FPS = video.fps
# Narration is assembled as 16-bit mono PCM. At 24 kHz and 30 fps every video
# frame is exactly 800 samples, so padding to a frame count is exact.
PCM_SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = PCM_SAMPLE_RATE // FPS
DEFAULT_GAP_SECONDS = 0.4
DEFAULT_ZOOM = 0.08
INTRO_CHAPTER = "Intro"
OUTRO_CHAPTER = "Outro"
_FALLBACK_BACKGROUND = (24, 24, 32)
_SRT_TIME_PATTERN = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


class ListVideoError(RuntimeError):
    """Raised when a list video cannot be produced."""


@dataclass
class ListSegment:
    kind: str  # "intro", "item" or "outro"
    text: str
    chapter: str
    label: str
    image_term: str
    image_file: str
    number: int = 0  # 1-based position of an item, 0 for intro/outro
    name: str = ""


@dataclass
class _Visual:
    kind: str  # "image", "videos" or "none"
    paths: List[str] = field(default_factory=list)


def build_segments(
    script: ListVideoScript, number_items: bool = True
) -> List[ListSegment]:
    """Flatten a list script into ordered segments.

    Intro and outro reuse the first and last item's visual when they have none
    of their own, so a hand-written script only needs item visuals.
    """
    items = script.items
    segments: List[ListSegment] = []

    def fallback_visual(term: str, image_file: str, item) -> tuple[str, str]:
        if term.strip() or image_file.strip():
            return term.strip(), image_file.strip()
        return (item.image_term.strip() or item.name.strip()), item.image_file.strip()

    if script.intro.strip():
        term, image_file = fallback_visual(
            script.intro_image_term, script.intro_image_file, items[0]
        )
        segments.append(
            ListSegment("intro", script.intro.strip(), INTRO_CHAPTER, "", term, image_file)
        )

    for index, item in enumerate(items, start=1):
        name = item.name.strip()
        label = f"{index}. {name}" if number_items else name
        segments.append(
            ListSegment(
                "item",
                item.text.strip(),
                label,
                label,
                item.image_term.strip() or name,
                item.image_file.strip(),
                number=index if number_items else 0,
                name=name,
            )
        )

    if script.outro.strip():
        term, image_file = fallback_visual(
            script.outro_image_term, script.outro_image_file, items[-1]
        )
        segments.append(
            ListSegment("outro", script.outro.strip(), OUTRO_CHAPTER, "", term, image_file)
        )
    return segments


def validate_visual_sources(segments: List[ListSegment], video_source: str) -> None:
    """Fail before any paid or slow work when visuals cannot be resolved."""
    if video_source not in LIST_VIDEO_SOURCES:
        raise ListVideoError(
            f"list videos support video_source {', '.join(LIST_VIDEO_SOURCES)}; "
            f"got {video_source!r}"
        )
    missing_files = [
        segment.image_file
        for segment in segments
        if segment.image_file and not os.path.isfile(segment.image_file)
    ]
    if missing_files:
        raise ListVideoError(f"image files not found: {', '.join(missing_files)}")
    if video_source == "local":
        without_file = [s.chapter for s in segments if not s.image_file]
        if without_file:
            raise ListVideoError(
                "video_source local needs an image_file for every item: "
                f"{', '.join(without_file)}"
            )
    if video_source == "openai_image" and not material.is_openai_image_enabled():
        raise ListVideoError(
            "openai_image requires openai_image_base_url and openai_image_model "
            "in config.toml"
        )


def format_chapter_time(seconds: float, use_hours: bool = False) -> str:
    total = max(0, int(math.floor(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if use_hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes + hours * 60}:{secs:02d}"


def format_chapters(chapters: List[tuple[float, str]], total_duration: float) -> str:
    """Return YouTube chapter lines ("0:00 Intro") for a video description."""
    use_hours = total_duration >= 3600
    return "\n".join(
        f"{format_chapter_time(start, use_hours)} {title}" for start, title in chapters
    )


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _parse_srt_time(value: str) -> float:
    match = _SRT_TIME_PATTERN.search(value)
    if not match:
        raise ValueError(f"invalid subtitle timestamp: {value!r}")
    hours, minutes, secs, fraction = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(secs)
        + int(fraction) / (10 ** len(fraction))
    )


def shift_subtitle_entries(entries, offset: float) -> List[tuple[float, float, str]]:
    """Convert ``subtitle.file_to_subtitles`` entries to absolute times."""
    shifted = []
    for _, times, text in entries:
        start_raw, _, end_raw = times.partition("-->")
        start = _parse_srt_time(start_raw) + offset
        end = _parse_srt_time(end_raw) + offset
        if text.strip() and end > start:
            shifted.append((start, end, text.strip()))
    return shifted


def write_srt(entries: List[tuple[float, float, str]], subtitle_file: str) -> bool:
    if not entries:
        return False
    with open(subtitle_file, "w", encoding="utf-8") as fp:
        for index, (start, end, text) in enumerate(entries, start=1):
            fp.write(
                f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
                f"{text}\n\n"
            )
    return True


def pad_pcm_to_frames(pcm: bytes, frames: int) -> tuple[bytes, float]:
    """Pad or trim 16-bit mono PCM to exactly ``frames`` video frames.

    Returns the adjusted PCM and how many seconds of it were cut off.
    """
    target_bytes = frames * SAMPLES_PER_FRAME * 2
    if len(pcm) >= target_bytes:
        trimmed = (len(pcm) - target_bytes) / 2 / PCM_SAMPLE_RATE
        return pcm[:target_bytes], trimmed
    return pcm + b"\x00" * (target_bytes - len(pcm)), 0.0


def _decode_pcm(audio_file: str) -> bytes:
    command = [
        utils.get_ffmpeg_binary(),
        "-v",
        "error",
        "-i",
        audio_file,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(PCM_SAMPLE_RATE),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise ListVideoError(f"failed to decode narration {audio_file}: {error}")
    pcm = result.stdout
    return pcm[: len(pcm) - (len(pcm) % 2)]


def count_video_frames(video_file: str) -> int:
    """Count the frames ffmpeg actually wrote; 0 when it cannot be read."""
    # Decoding is required: FFmpeg 7 reports no frame count for stream copies.
    command = [
        utils.get_ffmpeg_binary(),
        "-hide_banner",
        "-i",
        video_file,
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    matches = re.findall(r"frame=\s*(\d+)", result.stderr or "")
    return int(matches[-1]) if matches else 0


def _read_material_sources(task_id: str) -> list:
    script_file = os.path.join(utils.task_dir(task_id), "script.json")
    try:
        with open(script_file, "r", encoding="utf-8") as fp:
            sources = json.load(fp).get("material_sources") or []
    except (OSError, ValueError, AttributeError):
        return []
    return sources if isinstance(sources, list) else []


def _prepare_visual(
    task_id: str,
    segment: ListSegment,
    params: VideoParams,
    duration: float,
    material_sources: list,
    warnings: List[str],
) -> _Visual:
    if segment.image_file:
        if utils.parse_extension(segment.image_file) in IMAGE_EXTENSIONS:
            return _Visual("image", [segment.image_file])
        return _Visual("videos", [segment.image_file])

    source = params.video_source
    term = segment.image_term
    try:
        if source == "openai_image":
            items = material.generate_images_openai(
                search_term=term,
                minimum_duration=max(1, math.ceil(duration)),
                video_aspect=VideoAspect(params.video_aspect),
                save_dir=utils.task_dir(task_id),
            )
            if items:
                return _Visual("image", [items[0].url])
        elif source in STOCK_VIDEO_SOURCES:
            # download_videos replaces the task's source list on every call.
            # Clear it first so a failed call cannot repeat the previous
            # segment's records, then keep this segment's attribution.
            task_artifacts.patch_script_data(task_id, material_sources=[])
            paths = material.download_videos(
                task_id=task_id,
                search_terms=[term],
                source=source,
                video_aspect=VideoAspect(params.video_aspect),
                video_concat_mode=VideoConcatMode.sequential,
                audio_duration=duration,
                max_clip_duration=params.video_clip_duration,
            )
            material_sources.extend(_read_material_sources(task_id))
            if paths:
                return _Visual("videos", paths)
    except Exception as exc:
        logger.error(
            f"failed to prepare visual for {segment.chapter!r}: "
            f"{type(exc).__name__}: {exc}"
        )

    warnings.append(
        f"no visual found for {segment.chapter!r} (term: {term!r}); "
        "a plain background was used"
    )
    return _Visual("none")


def render_title_image(
    text: str, canvas_width: int, canvas_height: int, font_path: str
) -> Image.Image:
    """Draw ``text`` as a rounded, semi-transparent RGBA banner."""
    font_size = max(24, int(min(canvas_width, canvas_height) * 0.06))
    max_width = int(canvas_width * 0.9)
    while True:
        font = ImageFont.truetype(font_path, font_size)
        stroke = max(1, font_size // 25)
        left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
        pad_x = int(font_size * 0.6)
        pad_y = int(font_size * 0.35)
        box_width = (right - left) + 2 * pad_x
        if box_width <= max_width or font_size <= 20:
            break
        font_size = max(20, int(font_size * 0.9))

    box_width = min(box_width, max_width)
    box_height = (bottom - top) + 2 * pad_y
    image = Image.new("RGBA", (box_width, box_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, box_width - 1, box_height - 1),
        radius=int(font_size * 0.4),
        fill=(0, 0, 0, 170),
    )
    draw.text(
        (pad_x - left, pad_y - top),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )
    return image


def prepare_still_image(
    image_path: str,
    output_file: str,
    width: int,
    height: int,
    fit_mode: VideoFitMode,
) -> str:
    """Fit a picture to the canvas once, so ffmpeg never rescales per frame."""
    with Image.open(image_path) as source:
        source = ImageOps.exif_transpose(source)
        if source.mode in ("RGBA", "LA", "P"):
            # Transparent illustrations look best on white, not black.
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")
    if VideoFitMode(fit_mode) == VideoFitMode.contain:
        fitted = ImageOps.pad(image, (width, height), Image.LANCZOS, color=(0, 0, 0))
    else:
        fitted = ImageOps.fit(image, (width, height), Image.LANCZOS)
    fitted.save(output_file)
    return output_file


def _probe_duration(media_file: str) -> float:
    result = subprocess.run(
        [utils.get_ffmpeg_binary(), "-hide_banner", "-i", media_file],
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


def plan_montage(
    clip_durations: List[tuple[str, float]], duration: float, max_clip_duration: float
) -> List[tuple[str, float]]:
    """Choose (clip, seconds) pieces that cover ``duration``, looping if needed."""
    usable = [(path, length) for path, length in clip_durations if length > 1 / FPS]
    pieces: List[tuple[str, float]] = []
    total = 0.0
    while usable and total < duration - 1 / FPS and len(pieces) < 200:
        for path, length in usable:
            remaining = duration - total
            if remaining <= 1 / FPS:
                break
            take = min(length, max_clip_duration, remaining)
            pieces.append((path, take))
            total += take
    return pieces


def _fit_filter(width: int, height: int, fit_mode: VideoFitMode) -> str:
    if VideoFitMode(fit_mode) == VideoFitMode.contain:
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )


def _still_filter() -> str:
    # Decode a picture once and repeat it in memory: much cheaper than
    # re-reading the file for every frame with "-loop 1".
    return f"loop=loop=-1:size=1:start=0,setpts=N/{FPS}/TB"


def _overlay_input(overlay: fx.Overlay) -> List[str]:
    if overlay.mode == "frames":
        return ["-framerate", str(FPS), "-i", overlay.source]
    if overlay.mode == "concat":
        return ["-f", "concat", "-safe", "0", "-i", overlay.source]
    if overlay.mode == "media" and overlay.source.lower().endswith(".webm"):
        # libvpx keeps the alpha channel of transparent WebM animations.
        return ["-c:v", "libvpx-vp9", "-i", overlay.source]
    return ["-i", overlay.source]


def _overlay_prefilter(overlay: fx.Overlay, duration: float) -> str:
    if overlay.mode == "still":
        chain = [_still_filter()]
    elif overlay.mode == "concat":
        chain = [f"fps={FPS}"]
    else:
        chain = [f"fps={FPS}", f"setpts=PTS-STARTPTS+{overlay.start:.3f}/TB"]
    chain.append("format=rgba")
    end = overlay.end or duration
    if overlay.fade_in > 0:
        chain.append(f"fade=t=in:st={overlay.start:.3f}:d={overlay.fade_in:.3f}:alpha=1")
    if overlay.fade_out > 0:
        chain.append(
            f"fade=t=out:st={max(overlay.start, end - overlay.fade_out):.3f}"
            f":d={overlay.fade_out:.3f}:alpha=1"
        )
    return ",".join(chain)


def render_segment_video(
    visual: _Visual,
    frames: int,
    params: VideoParams,
    output_file: str,
    title: str = "",
    font_path: str = "",
    zoom: float = DEFAULT_ZOOM,
    overlays: List[fx.Overlay] | None = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> str:
    """Render one silent segment of exactly ``frames`` frames with ffmpeg."""
    width, height = VideoAspect(params.video_aspect).to_resolution()
    fit_mode = VideoFitMode(params.video_fit_mode)
    duration = frames / FPS
    stem = os.path.splitext(output_file)[0]
    temp_files: List[str] = []
    inputs: List[str] = []
    filters: List[str] = []
    overlays = list(overlays or [])

    if visual.kind == "videos":
        clip_durations = [(path, _probe_duration(path)) for path in visual.paths]
        pieces = plan_montage(
            clip_durations, duration, max(1, int(params.video_clip_duration or 5))
        )
        if not pieces:
            logger.warning(f"no readable clips for {output_file}; using a plain background")
            visual = _Visual("none")
        else:
            for index, (path, take) in enumerate(pieces):
                inputs += ["-t", f"{take:.3f}", "-i", path]
                filters.append(
                    f"[{index}:v]{_fit_filter(width, height, fit_mode)},"
                    f"fps={FPS},setsar=1,format=yuv420p[p{index}]"
                )
            labels = "".join(f"[p{index}]" for index in range(len(pieces)))
            # tpad holds the last frame if the clips end a little early.
            filters.append(
                f"{labels}concat=n={len(pieces)}:v=1:a=0,"
                f"tpad=stop_mode=clone:stop_duration={duration:.3f}[base]"
            )

    if visual.kind == "image":
        # Zooming needs spare pixels; 1.5x keeps the zoom smooth and cheap.
        oversample = 1.5 if zoom > 0 else 1.0
        still = prepare_still_image(
            visual.paths[0],
            f"{stem}-still.png",
            int(round(width * oversample)),
            int(round(height * oversample)),
            fit_mode,
        )
        temp_files.append(still)
        inputs += ["-i", still]
        if zoom > 0:
            filters.append(
                f"[0:v]{_still_filter()},zoompan=z='1+{zoom}*on/{frames}'"
                ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d=1:s={width}x{height}:fps={FPS},setsar=1[base]"
            )
        else:
            filters.append(f"[0:v]{_still_filter()},setsar=1[base]")
    elif visual.kind == "none":
        red, green, blue = _FALLBACK_BACKGROUND
        inputs += [
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x{red:02x}{green:02x}{blue:02x}:s={width}x{height}:r={FPS}",
        ]
        filters.append("[0:v]setsar=1[base]")

    if title and font_path:
        banner = f"{stem}-title.png"
        render_title_image(title, width, height, font_path).save(banner)
        temp_files.append(banner)
        overlays.insert(0, fx.Overlay(banner, x="(W-w)/2", y=str(int(height * 0.05))))

    current = "base"
    for number, overlay in enumerate(overlays):
        input_index = inputs.count("-i")
        inputs += _overlay_input(overlay)
        filters.append(f"[{input_index}:v]{_overlay_prefilter(overlay, duration)}[o{number}]")
        options = [f"x='{overlay.x}'", f"y='{overlay.y}'", "eval=frame"]
        if overlay.start > 0 or overlay.end > 0:
            options.append(
                f"enable='between(t,{overlay.start:.3f},{(overlay.end or duration):.3f})'"
            )
        if overlay.mode in ("frames", "media"):
            options.append("eof_action=pass")
        filters.append(f"[{current}][o{number}]overlay={':'.join(options)}[m{number}]")
        current = f"m{number}"

    finish = []
    if fade_in > 0:
        finish.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        finish.append(f"fade=t=out:st={max(0.0, duration - fade_out):.3f}:d={fade_out:.3f}")
    finish.append("format=yuv420p")
    filters.append(f"[{current}]{','.join(finish)}[v]")

    command = [
        utils.get_ffmpeg_binary(),
        "-v",
        "error",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-frames:v",
        str(frames),
        "-r",
        str(FPS),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        output_file,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or not os.path.isfile(output_file):
            error = (result.stderr or "").strip()[-2000:]
            raise ListVideoError(f"failed to render {output_file}: {error}")
        return output_file
    finally:
        video.delete_files(temp_files)


def concat_segments(segment_files: List[str], output_file: str) -> str:
    """Join segments rendered with identical settings without re-encoding."""
    list_file = f"{output_file}.txt"
    with open(list_file, "w", encoding="utf-8") as fp:
        for segment_file in segment_files:
            fp.write(f"file '{video._format_ffmpeg_concat_path(segment_file)}'\n")
    try:
        result = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                output_file,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        video.delete_files(list_file)
    if result.returncode != 0 or not os.path.isfile(output_file):
        raise ListVideoError(f"failed to join segments: {(result.stderr or '').strip()}")
    return output_file


def _resolve_font_path(params: VideoParams) -> str:
    font_name = params.font_name or "STHeitiMedium.ttc"
    return file_security.resolve_path_within_directory(utils.font_dir(), font_name)


DESIGN_FONT = "BeVietnamPro-Bold.ttf"
LOUDNESS_TARGET = "loudnorm=I=-14:TP=-1.5:LRA=11"


def _design_font(params: VideoParams, texts: List[str]) -> str:
    """Bold Latin font for labels, unless the texts need the subtitle font."""
    candidate = os.path.join(utils.font_dir(), DESIGN_FONT)
    sample = " ".join(texts)
    if os.path.isfile(candidate) and video.subtitle_font_supports_text(candidate, sample):
        return candidate
    return _resolve_font_path(params)


def mix_sound_effects(
    narration_file: str, events: List[tuple], output_file: str
) -> str:
    """Add (time, file, gain) sound effects onto the narration."""
    with wave.open(narration_file, "rb") as source:
        voice_samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
    mix = voice_samples.astype(np.float32) / 32768
    cache = {}
    for time, path, gain in events:
        if path not in cache:
            cache[path] = np.frombuffer(_decode_pcm(path), dtype=np.int16).astype(np.float32) / 32768
        effect = cache[path]
        begin = max(0, int(round(time * PCM_SAMPLE_RATE)))
        if begin >= len(mix):
            continue
        piece = effect[: len(mix) - begin]
        mix[begin : begin + len(piece)] += piece * gain
    # Soft-limit the rare peaks where an effect lands on loud speech.
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    with wave.open(output_file, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(PCM_SAMPLE_RATE)
        target.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
    return output_file


def _run_ffmpeg(command: List[str], what: str) -> None:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if result.returncode != 0:
        raise ListVideoError(f"failed to {what}: {(result.stderr or '').strip()[-2000:]}")


def normalize_loudness(audio_file: str, output_file: str) -> str:
    """Bring the narration to YouTube's loudness (-14 LUFS)."""
    _run_ffmpeg(
        [
            utils.get_ffmpeg_binary(), "-v", "error", "-y", "-i", audio_file,
            "-af", LOUDNESS_TARGET, "-ar", "48000", output_file,
        ],
        "normalize loudness",
    )
    return output_file


def mux_final_video(
    video_file: str,
    audio_file: str,
    output_file: str,
    duration: float,
    bgm_file: str = "",
    bgm_volume: float = 0.2,
) -> str:
    """Attach the audio without re-encoding the picture.

    Background music is looped, ducked under the voice with a sidechain
    compressor and faded out; the mix is normalized to -14 LUFS.
    """
    inputs = ["-i", video_file, "-i", audio_file]
    if bgm_file:
        inputs += ["-stream_loop", "-1", "-i", bgm_file]
        fade_start = max(0.0, duration - 3)
        graph = (
            "[1:a]aformat=channel_layouts=stereo,asplit=2[voice][key];"
            f"[2:a]aformat=channel_layouts=stereo,volume={bgm_volume:.3f},"
            f"atrim=0:{duration:.3f},afade=t=out:st={fade_start:.3f}:d=3[music];"
            "[music][key]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=350[ducked];"
            f"[voice][ducked]amix=inputs=2:duration=first:normalize=0,{LOUDNESS_TARGET}[a]"
        )
    else:
        graph = f"[1:a]{LOUDNESS_TARGET},aformat=channel_layouts=stereo[a]"
    _run_ffmpeg(
        [
            utils.get_ffmpeg_binary(), "-v", "error", "-y", *inputs,
            "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", output_file,
        ],
        "write the final video",
    )
    return output_file


def _list_bgm_file(params: VideoParams, warnings: List[str]) -> str:
    if str(params.bgm_type or "").strip().lower() == "none":
        return ""
    if not bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume):
        return ""
    if params.bgm_type not in ("random", "custom"):
        warnings.append(
            f"background music type {params.bgm_type!r} is not supported for list videos"
        )
        return ""
    return video.get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file) or ""


def generate_list_video(
    task_id: str,
    script: ListVideoScript,
    params: VideoParams,
    *,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    number_items: bool = True,
    show_item_titles: bool = True,
    zoom: float = DEFAULT_ZOOM,
    edit: Optional[editor_service.EditOptions] = None,
) -> dict:
    """Render a list-format video and return its files and chapters.

    With ``edit`` the video is edited automatically: chapter labels, a host
    character, pictures and key facts timed to the narration, a subscribe
    animation, sound effects and a progress bar. Without it each segment
    shows its visual and an "N. name" title.

    ``image_file`` paths in the script are used as given; callers that accept
    scripts from untrusted clients must restrict them first.
    """
    segments = build_segments(script, number_items=number_items)
    validate_visual_sources(segments, params.video_source)
    if not utils.check_ffmpeg_ready():
        raise ListVideoError(
            "ffmpeg is not available; install ffmpeg or set app.ffmpeg_path"
        )
    font_path = ""
    if show_item_titles or params.subtitle_enabled:
        font_path = _resolve_font_path(params)

    task_dir = utils.task_dir(task_id)
    task_artifacts.write_script_data(
        task_id,
        {"list_script": script.model_dump(), "material_sources": []},
    )

    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    word_level = getattr(params, "subtitle_display_mode", "sentence") == "word_by_word"
    voice_name = voice.parse_voice_name(params.voice_name)
    warnings: List[str] = []

    # Pass 1: narrate every segment, so the editor knows the whole timeline.
    narrations: List[editor_service.Narration] = []
    for index, segment in enumerate(segments):
        step = f"[{index + 1}/{len(segments)}] {segment.chapter}"
        logger.info(f"list video narration {step}")
        audio_file = os.path.join(task_dir, f"segment-{index:02d}.mp3")
        sub_maker = voice.tts(
            text=segment.text,
            voice_name=voice_name,
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            raise ListVideoError(f"failed to synthesize narration for {step}")
        pcm = _decode_pcm(audio_file)
        speech_seconds = len(pcm) / 2 / PCM_SAMPLE_RATE
        if speech_seconds <= 0:
            raise ListVideoError(f"narration for {step} is empty")
        narrations.append(
            editor_service.Narration(
                pcm=pcm,
                speech_seconds=speech_seconds,
                frames=math.ceil((speech_seconds + max(0.0, gap_seconds)) * FPS),
                sub_maker=sub_maker,
            )
        )

    editor = None
    if edit is not None:
        width, height = VideoAspect(params.video_aspect).to_resolution()
        theme = fx.Theme(
            width,
            height,
            _design_font(params, [s.chapter for s in segments]),
            fx.parse_color(edit.accent),
        )
        editor = editor_service.Editor(edit, theme, task_dir, segments, narrations)
        editor.make_plan()

    narration_file = os.path.join(task_dir, "narration.wav")
    segment_videos: List[str] = []
    subtitle_entries: List[tuple[float, float, str]] = []
    chapters: List[tuple[float, str]] = []
    material_sources: list = []
    sound_events: List[tuple] = []
    offset = 0.0

    # Pass 2: draw each segment and assemble one timeline.
    with wave.open(narration_file, "wb") as narration:
        narration.setnchannels(1)
        narration.setsampwidth(2)
        narration.setframerate(PCM_SAMPLE_RATE)

        for index, segment in enumerate(segments):
            step = f"[{index + 1}/{len(segments)}] {segment.chapter}"
            logger.info(f"list video segment {step}")
            measured = narrations[index]
            target_frames = measured.frames
            visual = _prepare_visual(
                task_id,
                segment,
                params,
                target_frames / FPS,
                material_sources,
                warnings,
            )
            overlays: List[fx.Overlay] = []
            if editor is not None:
                segment_edit = editor.segment_edit(index, offset, show_item_titles)
                overlays = segment_edit.overlays
                sound_events += [
                    (offset + time, path, gain) for time, path, gain in segment_edit.sounds
                ]
            segment_video = os.path.join(task_dir, f"segment-{index:02d}.mp4")
            render_segment_video(
                visual,
                target_frames,
                params,
                segment_video,
                title=segment.label if show_item_titles and editor is None else "",
                font_path=font_path,
                zoom=zoom,
                overlays=overlays,
                fade_in=0.5 if editor is not None and index == 0 else 0.0,
                fade_out=0.6 if editor is not None and index == len(segments) - 1 else 0.0,
            )
            segment_videos.append(segment_video)

            # Pad the narration to the frames actually written, so an encoder
            # that rounds a frame differently cannot shift later segments.
            frames = count_video_frames(segment_video) or target_frames
            pcm, trimmed = pad_pcm_to_frames(measured.pcm, frames)
            if trimmed > gap_seconds + 1 / FPS:
                warnings.append(f"narration for {step} was cut by {trimmed:.2f}s")
            narration.writeframes(pcm)

            if params.subtitle_enabled and subtitle_provider == "edge":
                segment_srt = os.path.join(task_dir, f"segment-{index:02d}.srt")
                voice.create_subtitle(
                    sub_maker=measured.sub_maker,
                    text=segment.text,
                    subtitle_file=segment_srt,
                    word_level=word_level,
                )
                entries = subtitle.file_to_subtitles(segment_srt)
                if entries:
                    subtitle_entries.extend(shift_subtitle_entries(entries, offset))
                else:
                    warnings.append(f"subtitles could not be aligned for {step}")

            chapters.append((offset, segment.chapter))
            offset += frames / FPS

    combined_video = concat_segments(
        segment_videos, os.path.join(task_dir, "combined-1.mp4")
    )

    audio_file = narration_file
    if sound_events:
        audio_file = mix_sound_effects(
            narration_file, sound_events, os.path.join(task_dir, "narration-sfx.wav")
        )

    subtitle_path = ""
    if params.subtitle_enabled and subtitle_provider:
        subtitle_path = os.path.join(task_dir, "subtitle.srt")
        if subtitle_provider == "whisper":
            subtitle.create(
                audio_file=narration_file,
                subtitle_file=subtitle_path,
                word_level=word_level,
            )
            if not word_level:
                full_text = "\n".join(segment.text for segment in segments)
                subtitle.correct(subtitle_file=subtitle_path, video_script=full_text)
            if not subtitle.file_to_subtitles(subtitle_path):
                subtitle_path = ""
        elif not write_srt(subtitle_entries, subtitle_path):
            subtitle_path = ""

    final_video = os.path.join(task_dir, "final-1.mp4")
    if subtitle_path:
        # Burned-in subtitles need the MoviePy compositor, which also mixes
        # the background music; hand it narration already at -14 LUFS.
        normalized = normalize_loudness(audio_file, os.path.join(task_dir, "narration-mix.wav"))
        bgm_ok = video.generate_video(
            video_path=combined_video,
            audio_path=normalized,
            subtitle_path=subtitle_path,
            output_file=final_video,
            params=params,
        )
        if not bgm_ok:
            warnings.append("background music could not be mixed")
    else:
        mux_final_video(
            combined_video,
            audio_file,
            final_video,
            offset,
            bgm_file=_list_bgm_file(params, warnings),
            bgm_volume=params.bgm_volume if params.bgm_volume is not None else 0.2,
        )

    chapters_text = format_chapters(chapters, offset)
    chapters_file = os.path.join(task_dir, "chapters.txt")
    with open(chapters_file, "w", encoding="utf-8") as fp:
        fp.write(f"{script.title}\n\n{chapters_text}\n")

    credits_file = ""
    if editor is not None:
        warnings += editor.warnings
        credits_file = editor.write_credits()

    task_artifacts.patch_script_data(
        task_id,
        material_sources=material_sources,
        chapters=[{"start": round(start, 3), "title": title} for start, title in chapters],
    )
    # Segment files are only intermediate renders of the final video.
    video.delete_files([*segment_videos, combined_video])

    for warning in warnings:
        logger.warning(warning)
    logger.success(f"list video finished: {final_video}")
    return {
        "videos": [final_video],
        "chapters": chapters_text,
        "chapters_file": chapters_file,
        "credits_file": credits_file,
        "audio_file": narration_file,
        "audio_duration": round(offset, 3),
        "subtitle_path": subtitle_path,
        "warnings": warnings,
    }
