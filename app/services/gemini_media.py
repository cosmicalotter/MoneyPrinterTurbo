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
import re
from typing import List, Optional

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


def build_choice_prompt(line: str, query: str, count: int, language: str = "") -> str:
    return f"""
You are the picture editor of an educational YouTube channel for a general audience.
The narrator says: "{line}"
We want a picture of: "{query}"
Below are {count} candidate pictures, numbered 1 to {count} in order.
Choose the one picture that:
- clearly and literally shows what the narrator says, so a viewer gets it in under two seconds;
- is simple: one main subject on a clean background, not a dense scientific diagram, collage, chart or infographic;
- has no text or labels, except at most a few words in {language_name(language)} or English, and never text in another language or alphabet;
- has no watermark, logo, gore or anything disturbing, and is sharp.
If none of the pictures meets every rule, answer 0.
Return only JSON: {{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}
""".strip()


def _parse_choice(text: str, count: int) -> Optional[int]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        choice = int(json.loads(match.group(0)).get("choice"))
    except (ValueError, TypeError, AttributeError):
        return None
    if choice == 0:
        return -1
    if 1 <= choice <= count:
        return choice - 1
    return None


def choose_picture(
    paths: List[str], line: str, query: str, language: str = "", app_config=None
) -> Optional[int]:
    """Index of the best picture, -1 when none is good enough, None when the check failed."""
    if not paths:
        return -1
    app_config = _app(app_config)
    try:
        from google import genai
        from google.genai import types

        kwargs = _client_kwargs(app_config)
        model = str(app_config.get("gemini_vision_model", "") or "").strip() or VISION_DEFAULT_MODEL
        contents: list = [build_choice_prompt(line, query, len(paths), language)]
        for number, path in enumerate(paths, 1):
            contents.append(f"Picture {number}:")
            contents.append(types.Part.from_bytes(data=_jpeg(path), mime_type="image/jpeg"))
        with genai.Client(**kwargs) as client:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
            )
        choice = _parse_choice(response.text, len(paths))
        if choice is None:
            logger.warning(f"picture check returned an unexpected answer: {response.text!r}")
        return choice
    except Exception as exc:
        logger.warning(f"picture check failed ({type(exc).__name__}: {exc}); using the first candidate")
        return None


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
    model = str(app_config.get("gemini_image_model", "") or "").strip() or IMAGE_DEFAULT_MODEL
    prompt = STYLE_PROMPT.format(subject=subject)
    path = _cache_path(model, prompt)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    try:
        from google import genai
        from google.genai import types

        kwargs = _client_kwargs(app_config)
        with genai.Client(**kwargs) as client:
            if model.startswith("imagen"):
                response = client.models.generate_images(
                    model=model,
                    prompt=prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1, aspect_ratio="1:1", output_mime_type="image/png"
                    ),
                )
                data = response.generated_images[0].image.image_bytes
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
                data = next(
                    part.inline_data.data
                    for part in response.candidates[0].content.parts
                    if getattr(part, "inline_data", None) and part.inline_data.data
                )
        with Image.open(io.BytesIO(data)) as image:
            image.convert("RGB").save(path)
        logger.info(f"illustration drawn: {subject!r}")
        return path
    except Exception as exc:
        logger.warning(f"illustration failed for {subject!r}: {type(exc).__name__}: {exc}")
        return ""
