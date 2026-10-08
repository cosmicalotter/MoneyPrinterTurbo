"""
Gemini as the picture editor and the illustrator of list videos.

``choose_picture`` shows a few candidate pictures to a Gemini vision model
together with the line being narrated and keeps the one a viewer understands
at a glance: it rejects dense diagrams, collages, watermarks and text in
another language. ``illustrate`` draws a picture in the channel's doodle style
with Imagen (or a Gemini image model), so scenes can show exactly what the
narration says. Both use the same credentials as the Gemini LLM (an API key or
Vertex AI Application Default Credentials).

Costs (2025 list prices, check yours): a picture check sends a few small JPEGs to a
Flash model, a fraction of a cent; an Imagen 4 Fast illustration costs about
US$0.02. Illustrations are cached by prompt, so re-renders and other languages
reuse them for free.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import threading
import time
from typing import List, Optional, Tuple

from loguru import logger
from PIL import Image, ImageOps

from app.config import config
from app.services import gemini_auth
from app.utils import utils

VISION_DEFAULT_MODEL = "gemini-2.5-flash"
IMAGE_DEFAULT_MODEL = "imagen-4.0-fast-generate-001"
STYLE_PROMPT = (
    "Minimalist hand-drawn doodle illustration for an educational explainer video: {subject}. "
    "Thick uniform dark outlines, flat soft colours, simple rounded shapes, friendly and clean, "
    "one centred subject with generous empty space around it, plain pure white background. "
    "No text, no letters, no numbers, no labels, no watermark, no frame, no shadows, no gradients, "
    "not photorealistic."
)
_LANGUAGE_NAMES = {"es": "Spanish", "en": "English", "pt": "Portuguese", "fr": "French", "de": "German", "it": "Italian"}


def _app(app_config=None):
    return app_config if app_config is not None else config.app


def _client_kwargs(app_config) -> dict:
    return gemini_auth.client_kwargs(app_config, str(app_config.get("gemini_api_key", "") or ""))


def enabled(app_config=None) -> bool:
    """True when Gemini credentials (key or Vertex AI) are configured."""
    try:
        _client_kwargs(_app(app_config))
        return True
    except ValueError:
        return False


def _jpeg(path: str, side: int = 640) -> bytes:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = image.convert("RGB")
    image.thumbnail((side, side), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


def language_name(language: str) -> str:
    return _LANGUAGE_NAMES.get((language or "").split("-")[0].lower(), "the narration's language")


PURPOSES = ("beat", "scene", "figure", "opener", "annotate", "icon", "clip")


def build_choice_prompt(line: str, query: str, count: int, language: str = "", purpose: str = "beat") -> str:
    """What the picture editor is asked; ``purpose`` is where the picture will be shown.

    * beat: big, over the footage, for a couple of seconds;
    * scene: small, beside icons in a minimalist drawn scene;
    * figure: filling the screen, long enough to read a diagram;
    * opener: big, beside the title of a new section; ``line`` is that title
      followed by the section's first sentence;
    * annotate: big, with labels pointing at its parts; ``query`` names the parts;
    * icon: a small flat icon beside its label in a drawn scene;
    * clip: a frame of a stock video shown for a few seconds in a drawn video.
    """
    language = language_name(language)
    said = f'The narrator says: "{line}"'
    if purpose == "icon":
        said = f'An icon will stand for "{query}" in a drawn explainer scene. Context: "{line}"'
        rules = """- literally depicts that thing, so a viewer recognises it at once without reading the label;
- reject an icon of a different thing that only shares a word (a face blowing a kiss is not "an energy wave",
  a mahjong tile is not "first contact", a plain coloured circle is not "a thick cable");
- a close, recognisable symbol of the idea is fine (a battery for "energy stored", a snail for "slow")."""
        answer = f'{{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}'
        return f"""
You are the art director of an educational YouTube channel.
{said}
Below are {count} candidate icons, numbered 1 to {count} in order.
Choose the one icon that:
{rules}
If no icon clearly fits, answer 0: no icon is better than a wrong one.
Return only JSON: {answer}
""".strip()
    if purpose == "clip":
        rules = f"""- clearly shows what the narrator talks about, so a viewer gets it at a glance;
- is a clean, bright, well-lit, sharp shot of a real scene (reject dark or night shots where little is visible);
- has no text, titles, logos or watermarks;
- nothing disturbing, no gore; any visible text is in {language} or English."""
        answer = f'{{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}'
    elif purpose == "opener":
        said = (
            f'This picture opens a new section of the video, shown for three seconds beside its title. '
            f'The section title and first sentence: "{line}"'
        )
        rules = f"""- shows exactly the topic of the section title, so a viewer who sees only the picture would guess the title;
  reject pictures of a related but different subject (for "voltage and current", not a random power plant);
- is the most representative and explanatory image of that topic: a clear photo, illustration or simple labelled diagram;
- is clean and readable on a TV: one main subject or one simple diagram, not a collage, a page of text or a screenshot;
- any text is in {language} or in English; never text in another language or alphabet;
- has no watermark, logo, gore or anything disturbing, and is sharp.
If you are not sure a picture is strictly about the title, answer 0."""
        answer = f'{{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}'
    elif purpose == "annotate":
        rules = f"""- shows exactly the structure, object or process the narrator explains, large and clear;
- the parts we will point at are clearly visible in it (they are listed in the request);
- is clean: a clear illustration, medical or scientific diagram, or photo; few or no labels of its own
  (any labels in {language} or in English); not a collage, a page of text or a screenshot;
- has no watermark, logo, gore or anything disturbing, and is sharp."""
        answer = f'{{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}'
    elif purpose == "figure":
        rules = f"""- explains or shows exactly what the narrator says (a diagram, chart, infographic or a striking photo);
- is clean and readable on a TV: large shapes, little clutter, sharp, not a page of text or a screenshot;
- any text is in {language} or in English and readable; never text in another language or alphabet;
- has no watermark, logo, gore or anything disturbing."""
        answer = (
            f'{{"choice": <number from 0 to {count}>, "seconds": <how long a viewer needs to understand it: '
            '3 or 4 for a photo, 5 to 8 for a diagram with text>, "reason": "<a few words>"}'
        )
    else:
        small = (
            "- it will be shown small next to simple icons, so it must be one isolated object or figure "
            "(clip art, a cut-out on a plain background, or a clean photo of only that thing);\n"
            "- it must be very closely related to the words: if you are not sure, answer 0;\n"
            if purpose == "scene"
            else ""
        )
        explain = (
            "- when the narrator explains how something works or what it is made of, a clean explanatory diagram or "
            "illustration of exactly that is best (electrons moving through a wire, charges attracting, an organ's parts);\n"
            if purpose == "beat"
            else ""
        )
        rules = f"""- clearly and literally shows what the narrator says, so a viewer gets it in two or three seconds;
{small}{explain}- is simple and readable on a TV: one main subject or one clear diagram, not a dense textbook figure, collage, page of text or screenshot;
- has no text or labels, except a few large words in {language} or English, and never text in another language or alphabet;
- has no watermark, logo, gore or anything disturbing, and is sharp."""
        answer = f'{{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}'
    return f"""
You are the picture editor of an educational YouTube channel for a general audience.
{said}
We want a picture of: "{query}"
Below are {count} candidate pictures, numbered 1 to {count} in order.
Choose the one picture that:
{rules}
If none of the pictures meets every rule, answer 0.
Return only JSON: {answer}
""".strip()


def _parse_answer(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        answer = json.loads(match.group(0))
    except ValueError:
        return None
    return answer if isinstance(answer, dict) else None


def _parse_choice(text: str, count: int) -> Optional[int]:
    answer = _parse_answer(text)
    if answer is None:
        return None
    try:
        choice = int(answer.get("choice"))
    except (ValueError, TypeError):
        return None
    if choice == 0:
        return -1
    if 1 <= choice <= count:
        return choice - 1
    return None


def _ask(paths: List[str], prompt: str, app_config) -> str:
    from google import genai
    from google.genai import types

    kwargs = _client_kwargs(app_config)
    model = str(app_config.get("gemini_vision_model", "") or "").strip() or VISION_DEFAULT_MODEL
    contents: list = [prompt]
    for number, path in enumerate(paths, 1):
        contents.append(f"Picture {number}:")
        contents.append(types.Part.from_bytes(data=_jpeg(path), mime_type="image/jpeg"))
    with genai.Client(**kwargs) as client:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
        )
    return response.text


def choose_picture(
    paths: List[str], line: str, query: str, language: str = "", app_config=None, purpose: str = "beat"
) -> Optional[int]:
    """Index of the best picture, -1 when none is good enough, None when the check failed."""
    if not paths:
        return -1
    app_config = _app(app_config)
    try:
        text = _ask(paths, build_choice_prompt(line, query, len(paths), language, purpose), app_config)
        choice = _parse_choice(text, len(paths))
        if choice is None:
            logger.warning(f"picture check returned an unexpected answer: {text!r}")
        return choice
    except Exception as exc:
        logger.warning(f"picture check failed ({type(exc).__name__}: {exc}); using the first candidate")
        return None


def choose_figure(
    paths: List[str], line: str, query: str, language: str = "", app_config=None
) -> Optional[Tuple[int, float]]:
    """(index, seconds to show it) for a full-screen picture; index -1 when none fits; None if the check failed."""
    if not paths:
        return -1, 0.0
    app_config = _app(app_config)
    try:
        text = _ask(paths, build_choice_prompt(line, query, len(paths), language, "figure"), app_config)
    except Exception as exc:
        logger.warning(f"figure check failed ({type(exc).__name__}: {exc}); using the first candidate")
        return None
    choice = _parse_choice(text, len(paths))
    if choice is None:
        logger.warning(f"figure check returned an unexpected answer: {text!r}")
        return None
    try:
        seconds = float((_parse_answer(text) or {}).get("seconds") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    return choice, min(9.0, max(3.0, seconds)) if seconds else 0.0


def _cache_path(model: str, prompt: str) -> str:
    folder = utils.storage_dir(os.path.join("cache", "illustrations"), create=True)
    key = hashlib.sha1(f"{model}\n{prompt}".encode("utf-8")).hexdigest()[:20]
    return os.path.join(folder, f"{key}.png")


def illustrate(subject: str, app_config=None) -> str:
    """A doodle-style illustration of ``subject`` on white; "" when it fails."""
    subject = (subject or "").strip()
    if not subject:
        return ""
    app_config = _app(app_config)
    model = _usable_model(str(app_config.get("gemini_image_model", "") or "").strip() or IMAGE_DEFAULT_MODEL)
    prompt = STYLE_PROMPT.format(subject=subject)
    path = _cache_path(model, prompt)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    data = _generate_image(model, prompt, "1:1", [], app_config, subject)
    if not data:
        return ""
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(path)
    logger.info(f"illustration drawn: {subject!r}")
    return path


SEQUENCE_DEFAULT_MODEL = "gemini-2.5-flash-image"
# Waits before each new try of a busy image model (quotas are per minute, so the last waits are long).
RETRY_SECONDS = (5.0, 15.0, 30.0, 60.0)
NO_IMAGE_RETRIES = 1  # an answer without a picture is asked once more
IMAGE_SLOTS = 2  # pictures drawn at the same time (more only hits the per-minute quota)
_slots = threading.BoundedSemaphore(IMAGE_SLOTS)
_TRANSIENT = ("429", "resource_exhausted", "resource exhausted", "503", "unavailable", "500", "internal",
              "deadline", "timeout", "timed out", "temporarily", "overloaded", "rate limit")
_state = {"imagen_failed": "", "last_error": ""}


class _NoImage(RuntimeError):
    """The model answered without a picture (a filtered prompt, or text only)."""


def last_error() -> str:
    """Why the last drawing failed ("" when none did), for the render's warnings."""
    return _state["last_error"]


def imagen_failure() -> str:
    """Why Imagen was given up for this session ("" while it works)."""
    return _state["imagen_failed"]


def _transient(exc: Exception) -> bool:
    if getattr(exc, "code", None) in (429, 500, 502, 503, 504):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(word in text for word in _TRANSIENT)


def _usable_model(model: str) -> str:
    """``model``, or the Gemini image model once Imagen failed in this session."""
    if model.startswith("imagen") and _state["imagen_failed"]:
        return SEQUENCE_DEFAULT_MODEL
    return model


def _image_bytes(client, model: str, prompt: str, aspect: str, references: List[bytes]) -> bytes:
    from google.genai import types

    if model.startswith("imagen"):
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=aspect, output_mime_type="image/png"),
        )
        images = list(getattr(response, "generated_images", None) or [])
        if not images:
            raise _NoImage("Imagen returned no picture (the prompt may have been filtered)")
        return images[0].image.image_bytes
    contents: list = [types.Part.from_bytes(data=data, mime_type="image/png") for data in references]
    response = client.models.generate_content(
        model=model,
        contents=contents + [prompt] if contents else prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"], image_config=types.ImageConfig(aspect_ratio=aspect)
        ),
    )
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    raise _NoImage(f"{model} answered without a picture")


def _generate_image(
    model: str, prompt: str, aspect: str, references: List[bytes], app_config, subject: str
) -> bytes:
    """The picture's bytes; b"" when every try failed.

    Busy models (429, 503) are tried again after a short wait. When Imagen
    fails (a region or project without it, a filtered prompt) the Gemini
    image model draws instead, and a broken Imagen is not asked again.
    """
    from google import genai

    attempt = empty = 0
    while True:
        try:
            with _slots, genai.Client(**_client_kwargs(app_config)) as client:
                return _image_bytes(client, model, prompt, aspect, references)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:300]
            if model.startswith("imagen") and not _transient(exc):
                if not isinstance(exc, _NoImage) and not _state["imagen_failed"]:
                    _state["imagen_failed"] = reason
                    logger.warning(f"Imagen failed ({reason}); drawing with {SEQUENCE_DEFAULT_MODEL} from now on")
                model = SEQUENCE_DEFAULT_MODEL
                continue
            if isinstance(exc, _NoImage) and empty < NO_IMAGE_RETRIES:
                empty += 1
                continue
            if _transient(exc) and attempt < len(RETRY_SECONDS):
                wait = RETRY_SECONDS[attempt] * random.uniform(0.8, 1.25)
                logger.info(f"image model busy ({reason[:80]}); trying again in {wait:.0f} s")
                time.sleep(wait)
                attempt += 1
                continue
            _state["last_error"] = reason
            logger.warning(f"drawing failed for {subject!r}: {reason}")
            return b""
SEQUENCE_PROMPT = (
    "Minimalist hand-drawn doodle illustration for an educational explainer video, "
    "thick uniform dark outlines, flat soft colours, plain pure white background, no text: {subject}"
)
SEQUENCE_NEXT = (
    "Redraw the same scene with the same style, characters, objects and framing, changing only this: "
    "{subject}. No text."
)


def illustrate_sequence(descriptions: List[str], app_config=None) -> List[str]:
    """Consecutive frames of one little scene, each drawn from the previous one.

    Uses a Gemini image model that can edit pictures (Imagen cannot), so the
    characters and props stay the same from frame to frame. Frames are cached
    by their whole story. Returns [] when the model is not available.
    """
    descriptions = [d.strip() for d in descriptions if d and d.strip()]
    if len(descriptions) < 2:
        return []
    app_config = _app(app_config)
    model = str(app_config.get("gemini_image_model", "") or "").strip()
    if not model or model.startswith("imagen"):
        model = SEQUENCE_DEFAULT_MODEL
    story = "\n".join(descriptions)
    paths = [_cache_path(model, f"{story}\n#{n}") for n in range(len(descriptions))]
    if all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths):
        return paths
    try:
        from google import genai
        from google.genai import types

        kwargs = _client_kwargs(app_config)
        previous = None
        with genai.Client(**kwargs) as client:
            for number, subject in enumerate(descriptions):
                if previous is None:
                    contents: list = [SEQUENCE_PROMPT.format(subject=subject)]
                else:
                    contents = [
                        types.Part.from_bytes(data=previous, mime_type="image/png"),
                        SEQUENCE_NEXT.format(subject=subject),
                    ]
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                data = next(
                    part.inline_data.data
                    for part in response.candidates[0].content.parts
                    if getattr(part, "inline_data", None) and part.inline_data.data
                )
                with Image.open(io.BytesIO(data)) as image:
                    image.convert("RGB").save(paths[number])
                with open(paths[number], "rb") as fp:
                    previous = fp.read()
        logger.info(f"story drawn in {len(paths)} frames")
        return paths
    except Exception as exc:
        logger.warning(f"story frames failed: {type(exc).__name__}: {exc}")
        return []


def build_points_prompt(labels: List[str]) -> str:
    return f"""
Point to each of these parts in the picture: {json.dumps(labels, ensure_ascii=False)}.
Return only JSON: {{"points": [{{"label": "<the label exactly as given>", "point": [y, x]}}]}}
with y and x normalised to 0-1000 (0, 0 is the top-left corner) at the centre of that part.
Use "point": null for a part that is not clearly visible. Keep the order of the list.
""".strip()


def locate_parts(path: str, labels: List[str], app_config=None) -> List[Optional[Tuple[float, float]]]:
    """Where each labelled part is in the picture, as (x, y) fractions; None for parts not found."""
    if not path or not labels:
        return [None] * len(labels)
    app_config = _app(app_config)
    try:
        answer = _parse_answer(_ask([path], build_points_prompt(labels), app_config)) or {}
    except Exception as exc:
        logger.warning(f"locating parts failed ({type(exc).__name__}: {exc})")
        return [None] * len(labels)
    found = {}
    entries = answer.get("points") if isinstance(answer.get("points"), list) else []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        point = entry.get("point")
        try:
            y, x = (min(1000.0, max(0.0, float(v))) / 1000.0 for v in point)
        except (TypeError, ValueError):
            continue
        label = str(entry.get("label") or "").strip().lower()
        found.setdefault(label, (x, y))
        found.setdefault(position, (x, y))
    return [found.get(label.strip().lower(), found.get(position)) for position, label in enumerate(labels)]


# ---------------------------------------------------------------------------
# The doodle look: every picture is drawn
# ---------------------------------------------------------------------------

DOODLE_PROMPT = (
    "Hand-drawn cartoon doodle for a calm educational YouTube animation: {subject}. "
    "Black ink line art with natural, slightly uneven pen strokes and a little cross-hatching for shading, "
    "flat muted colours (cream, warm grey, dusty blue, brick red, mustard), simple, friendly and expressive, "
    "like a hand-made editorial illustration. One subject, centred, with generous empty space around it, "
    "on a plain pure white background. No text, no letters, no numbers, no frame, no ground shadow, "
    "not photorealistic, not 3D, no gradients."
)
SCENE_PROMPT = (
    "Wide 16:9 hand-drawn cartoon scene for a calm educational YouTube animation: {subject}. "
    "Clean black ink outlines, flat muted colours, simple characters with round heads and expressive faces, "
    "an uncluttered composition with one clear focal point, soft light, a hand-made look. "
    "No text, no letters, no numbers, no watermark, not photorealistic, not 3D."
)
# The polished 2D animation look (the default): bold outlines, soft cel shading
# and cinematic light, like a frame of a professional animated explainer.
CARTOON_PROMPT = (
    "Modern 2D cartoon illustration for a calm, professional educational YouTube animation: {subject}. "
    "Clean bold dark outlines, soft cel shading with gentle light, a muted harmonious palette "
    "(teal, warm brown, cream, dusty blue, soft yellow, brick red), rounded friendly shapes, expressive and simple. "
    "One subject, whole, centred and isolated, with generous empty space around it, on a plain pure white "
    "background. No text, no letters, no numbers, no frame, no ground shadow, not photorealistic, not 3D."
)
CARTOON_SCENE_PROMPT = (
    "Wide 16:9 frame of a modern 2D cartoon animation for a calm, professional educational YouTube channel: "
    "{subject}. Clean bold dark outlines, soft cel shading, cinematic lighting with a clear mood, a muted "
    "harmonious palette, a simple background with depth, one clear focal point and an uncluttered composition "
    "that reads at a glance, like a frame of a polished animated explainer. "
    "No text, no letters, no numbers, no captions, no watermark, not photorealistic, not 3D."
)
DRAWING_STYLES = ("cartoon", "ink")
_PROMPTS = {"cartoon": (CARTOON_PROMPT, CARTOON_SCENE_PROMPT), "ink": (DOODLE_PROMPT, SCENE_PROMPT)}
MASCOT_NOTE = (
    "The reference picture shows the channel's mascot, an otter with round glasses, a teal sweater and a pencil "
    "behind its ear. Draw this same otter, keeping its design and colours, in the style described. "
)
NEXT_FRAME_PROMPT = (
    "The reference picture is the previous frame of an animation. Draw the next frame: keep exactly the same "
    "drawing style, setting, characters, colours and lighting, and change only this: {subject}. "
    "Everything not mentioned stays as it was. Wide 16:9 frame. No text, no letters, no numbers, no captions."
)


def _reference_bytes(path: str) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.getchannel("A"))
    canvas.thumbnail((768, 768), Image.LANCZOS)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def draw(
    subject: str, scene: bool = False, mascot: str = "", app_config=None, previous: str = "", style: str = "cartoon"
) -> str:
    """A drawing for the doodle look: one subject on white, or a whole 16:9 scene.

    ``mascot`` is a picture of the channel's character; when given, a Gemini
    image model (which can follow a reference) draws that same character.
    ``previous`` is the scene drawn just before: the new one is redrawn from it
    with only ``subject`` changed, like the next frame of an animation.
    ``style`` is "cartoon" (polished 2D animation) or "ink" (pen doodles).
    Drawings are cached by their prompt. Returns "" when drawing fails.
    """
    subject = " ".join((subject or "").split())
    if not subject:
        return ""
    app_config = _app(app_config)
    configured = str(app_config.get("gemini_image_model", "") or "").strip() or IMAGE_DEFAULT_MODEL
    model = _usable_model(configured)
    single, whole = _PROMPTS.get(style, _PROMPTS["cartoon"])
    reference = b""
    if previous and os.path.isfile(previous):
        reference = _reference_bytes(previous)
        prompt = NEXT_FRAME_PROMPT.format(subject=subject)
        scene = True
    else:
        prompt = (whole if scene else single).format(subject=subject)
        if mascot and os.path.isfile(mascot):
            reference = _reference_bytes(mascot)
            prompt = MASCOT_NOTE + prompt
    if reference and model.startswith("imagen"):
        model = SEQUENCE_DEFAULT_MODEL  # Imagen cannot follow a reference picture
    key = prompt + ("\n#ref" + hashlib.sha1(reference).hexdigest()[:12] if reference else "")
    path = _cache_path(model, key)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    data = _generate_image(model, prompt, "16:9" if scene else "1:1", [reference] if reference else [], app_config, subject)
    if not data:
        return ""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.convert("RGB").save(path)
    except Exception as exc:
        _state["last_error"] = f"unreadable picture: {exc}"
        logger.warning(f"drawing for {subject!r} could not be read: {exc}")
        return ""
    logger.info(f"drawn: {subject!r}")
    return path


def build_drawing_check_prompt(description: str, mascot: bool = False) -> str:
    otter = (
        "- the channel's mascot, an otter with round glasses and a teal sweater, looks like itself (not a different animal);\n"
        if mascot else ""
    )
    return f"""
You are the art director of an educational cartoon channel. An illustrator was asked to draw:
"{description}"
Check the picture below. It passes when:
- it clearly shows what was asked, so a viewer recognises it at a glance;
{otter}- it has no text, letters, numbers, labels or watermarks drawn in it;
- nothing is broken: no melted or extra limbs, no garbled shapes, no half-drawn objects.
Return only JSON: {{"ok": true or false, "reason": "<a few words>"}}
""".strip()


def check_drawing(path: str, description: str, mascot: bool = False, app_config=None) -> Optional[bool]:
    """True when Gemini confirms a drawing shows ``description`` cleanly, False when not, None if the check failed."""
    if not path or not os.path.isfile(path):
        return False
    app_config = _app(app_config)
    try:
        answer = _parse_answer(_ask([path], build_drawing_check_prompt(description, mascot), app_config)) or {}
    except Exception as exc:
        logger.warning(f"drawing check failed ({type(exc).__name__}: {exc})")
        return None
    verdict = answer.get("ok")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower() in ("true", "yes", "1")
    if verdict is None:
        return None
    if not verdict:
        logger.info(f"drawing rejected for {description!r}: {answer.get('reason', '')}")
    return bool(verdict)
