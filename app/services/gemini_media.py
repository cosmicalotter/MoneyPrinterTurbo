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


PURPOSES = ("beat", "scene", "figure", "opener", "annotate")


def build_choice_prompt(line: str, query: str, count: int, language: str = "", purpose: str = "beat") -> str:
    """What the picture editor is asked; ``purpose`` is where the picture will be shown.

    * beat: big, over the footage, for a couple of seconds;
    * scene: small, beside icons in a minimalist drawn scene;
    * figure: filling the screen, long enough to read a diagram;
    * opener: big, beside the title of a new section; ``line`` is that title
      followed by the section's first sentence;
    * annotate: big, with labels pointing at its parts; ``query`` names the parts.
    """
    language = language_name(language)
    said = f'The narrator says: "{line}"'
    if purpose == "opener":
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


SEQUENCE_DEFAULT_MODEL = "gemini-2.5-flash-image"
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
