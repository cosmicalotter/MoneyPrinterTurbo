"""
Gemini as the picture editor, the illustrator and the animator of list videos.

``choose_picture`` shows a few candidate pictures to a Gemini vision model
together with the line being narrated and keeps the one a viewer understands
at a glance: it rejects dense diagrams, collages, watermarks and text in
another language (and, for real historical pictures, anything staged or
modern). ``draw`` draws the doodle look with a Gemini image model (Gemini 3.1
Flash Image by default, Gemini 3 Pro Image for the key pictures in the
"high" quality), following reference pictures: the frame before, the
channel's mascot and the character sheets of the people of the video.
``audit_drawing`` checks every drawing against the line it illustrates, and
``animate`` turns a drawing into a short Veo video. All use the same
credentials as the Gemini LLM (an API key or Vertex AI Application Default
Credentials).

Costs (2026 list prices, check yours): a picture check sends a few small JPEGs to a
Flash model, a fraction of a cent; a drawing costs about US$0.04 (Gemini 2.5
Flash Image), US$0.07 (3.1 Flash Image) or US$0.13 (3 Pro Image); a Veo 3.1
Fast video about US$0.10 per second without sound. Drawings and videos are
cached by prompt and references, so re-renders and other languages reuse them
for free.
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
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger
from PIL import Image, ImageOps

from app.config import config
from app.services import gemini_auth
from app.utils import utils

VISION_DEFAULT_MODEL = "gemini-2.5-flash"
IMAGE_DEFAULT_MODEL = "gemini-3.1-flash-image"  # Imagen 4 is only used when config.toml names it
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


PURPOSES = ("beat", "scene", "figure", "opener", "annotate", "icon", "clip", "archive", "portrait")


def build_choice_prompt(line: str, query: str, count: int, language: str = "", purpose: str = "beat") -> str:
    """What the picture editor is asked; ``purpose`` is where the picture will be shown.

    * beat: big, over the footage, for a couple of seconds;
    * scene: small, beside icons in a minimalist drawn scene;
    * figure: filling the screen, long enough to read a diagram;
    * opener: big, beside the title of a new section; ``line`` is that title
      followed by the section's first sentence;
    * annotate: big, with labels pointing at its parts; ``query`` names the parts;
    * icon: a small flat icon beside its label in a drawn scene;
    * clip: a frame of a stock video shown for a few seconds in a drawn video;
    * archive: a real historical picture filling the screen of a documentary;
    * portrait: a real portrait of a person, the model for their cartoon.
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
    if purpose in ("archive", "portrait"):
        if purpose == "archive":
            said = f'This real picture fills the screen of a history documentary while the narrator says: "{line}"'
            what = """- is a REAL document of exactly what is asked: a painting, engraving or drawing of the period, a historical
  photograph, a manuscript, a map, a title page, or a museum photograph of the real object or the real place;
- shows the right person, event, object or place: reject a portrait of someone else, a different event or a
  related but different object;"""
        else:
            said = f'We need a real portrait of this person to draw them as a cartoon. Context: "{line}"'
            what = """- is a real portrait of exactly that person (a painting, an engraving or a photograph), the face clearly
  visible and large; reject statues, caricatures, group pictures where the face is tiny, and other people;"""
        rules = f"""{what}
- reject modern staged or stock photos, re-enactments, costumes, film stills, AI-generated or 3D images, cartoons,
  clip art, collages and screenshots;
- is sharp and large; no watermark, no big modern text, no museum label or wide frame taking much of the picture;
  any text is in {language} or English, or is part of the historical document;
- nothing gory or disturbing.
If you are not sure a picture is authentic and about exactly that, answer 0."""
        answer = f'{{"choice": <number from 0 to {count}>, "reason": "<a few words>"}}'
    elif purpose == "clip":
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
    models = image_models("standard", True, app_config)
    prompt = STYLE_PROMPT.format(subject=subject)
    path = _cache_path(models[0], prompt)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    data = _generate_image(models, prompt, "1:1", [], app_config, subject)
    if not data:
        return ""
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(path)
    logger.info(f"illustration drawn: {subject!r}")
    return path


# Gemini image models, newest first. Imagen was retired in 2026: it is only used when
# config.toml names it explicitly (gemini_image_model), and never for edits.
FLASH_IMAGE_MODELS = ("gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-2.5-flash-image")
PRO_IMAGE_MODELS = ("gemini-3-pro-image", "gemini-3-pro-image-preview") + FLASH_IMAGE_MODELS
ECONOMY_IMAGE_MODELS = ("gemini-2.5-flash-image", "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview")
# economy: the cheapest; standard: Gemini 3.1 Flash Image; high: Gemini 3 Pro Image for the key pictures
# (new scenes, first frames, character sheets) and Flash for the frames redrawn from them; max: Pro everywhere.
IMAGE_QUALITIES = ("economy", "standard", "high", "max")
SEQUENCE_DEFAULT_MODEL = FLASH_IMAGE_MODELS[0]
SHARP_SIZE = "2K"  # Gemini 3 image models draw scenes at 2K, sharp on a 1080p video
RETRY_SECONDS = (5.0, 15.0, 30.0, 60.0)  # waits before each new try of a busy model (quotas are per minute)
NO_IMAGE_RETRIES = 1  # an answer without a picture is asked once more
IMAGE_SLOTS = 2  # pictures drawn at the same time (more only hits the per-minute quota)
_slots = threading.BoundedSemaphore(IMAGE_SLOTS)
_TRANSIENT = ("429", "resource_exhausted", "resource exhausted", "503", "unavailable", "500", "internal",
              "deadline", "timeout", "timed out", "temporarily", "overloaded", "rate limit")
_GONE = ("404", "not_found", "not found", "does not exist", "is not supported", "not supported for", "was retired",
         "is retired", "deprecated", "no longer available", "unknown model", "invalid model", "permission_denied")
_state = {"imagen_failed": "", "last_error": "", "gone": {}, "no_size": set()}


class _NoImage(RuntimeError):
    """The model answered without a picture (a filtered prompt, or text only)."""


class _TooSlow(RuntimeError):
    """A video was still not ready after ``VIDEO_TIMEOUT_SECONDS`` (not worth asking again)."""


def last_error() -> str:
    """Why the last drawing failed ("" when none did), for the render's warnings."""
    return _state["last_error"]


def imagen_failure() -> str:
    """Why Imagen was given up for this session ("" while it works or is not used)."""
    return _state["imagen_failed"]


def unavailable_models() -> Dict[str, str]:
    """{model: why} for the image models this project cannot use (retired, not enabled, wrong region)."""
    return dict(_state["gone"])


def _transient(exc: Exception) -> bool:
    if getattr(exc, "code", None) in (429, 500, 502, 503, 504):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(word in text for word in _TRANSIENT)


def _gone(exc: Exception) -> bool:
    """The model itself cannot be used (as opposed to this one request failing)."""
    if getattr(exc, "code", None) in (403, 404):
        return True
    text = f"{exc}".lower()
    return any(word in text for word in _GONE)


def image_models(quality: str = "standard", key_frame: bool = True, app_config=None) -> Tuple[str, ...]:
    """The image models to try, best first, for a picture of ``quality``.

    ``key_frame`` is a picture drawn from scratch (a new scene, a first
    frame, a character sheet); the frames redrawn from it use Flash under
    "high". A model named in config.toml (gemini_image_model) is tried first.
    """
    app_config = _app(app_config)
    if quality == "max" or (quality == "high" and key_frame):
        chain = PRO_IMAGE_MODELS
    elif quality == "economy":
        chain = ECONOMY_IMAGE_MODELS
    else:
        chain = FLASH_IMAGE_MODELS
    configured = str(app_config.get("gemini_image_model", "") or "").strip()
    if configured:
        chain = (configured,) + tuple(m for m in chain if m != configured)
    usable = tuple(m for m in chain if m not in _state["gone"])
    return usable or chain[-1:]


def _usable_model(model: str) -> str:
    """``model``, or the default Gemini image model once ``model`` failed in this session."""
    if model in _state["gone"] or (model.startswith("imagen") and _state["imagen_failed"]):
        return SEQUENCE_DEFAULT_MODEL
    return model


def _image_bytes(client, model: str, prompt: str, aspect: str, references: Sequence[bytes], size: str = "") -> bytes:
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
    contents: list = []
    for number, data in enumerate(references, 1):
        if len(references) > 1:
            contents.append(f"Reference picture {number}:")
        contents.append(types.Part.from_bytes(data=data, mime_type="image/png"))
    image_config = types.ImageConfig(aspect_ratio=aspect, image_size=size) if size else types.ImageConfig(aspect_ratio=aspect)
    response = client.models.generate_content(
        model=model,
        contents=contents + [prompt] if contents else prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=image_config),
    )
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    raise _NoImage(f"{model} answered without a picture")


def _generate_image(
    models, prompt: str, aspect: str, references: Sequence[bytes], app_config, subject: str, size: str = ""
) -> bytes:
    """The picture's bytes from the first model that draws it; b"" when every try failed.

    ``models`` is one model or several, best first. A model this project
    cannot use (retired, not enabled, wrong region) is skipped for the rest
    of the session; busy models (429, 503) are tried again after a wait; a
    picture refused by one model is asked of the next one.
    """
    from google import genai

    chain = [models] if isinstance(models, str) else list(models)
    if any(m.startswith("imagen") for m in chain) and not any(not m.startswith("imagen") for m in chain):
        chain.append(SEQUENCE_DEFAULT_MODEL)  # Imagen alone: the Gemini model takes over when it fails
    reason = "no image model could be used"
    for model in chain:
        if model in _state["gone"] or (references and model.startswith("imagen")):
            continue
        if model.startswith("imagen") and _state["imagen_failed"]:
            continue
        wanted = size if size and model.startswith("gemini-3") and model not in _state["no_size"] else ""
        attempt = empty = 0
        while True:
            try:
                with _slots, genai.Client(**_client_kwargs(app_config)) as client:
                    return _image_bytes(client, model, prompt, aspect, references, wanted)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"[:300]
                if wanted and "size" in str(exc).lower() and not _transient(exc):
                    _state["no_size"].add(model)
                    wanted = ""
                    continue
                if isinstance(exc, _NoImage):
                    if empty < NO_IMAGE_RETRIES and not model.startswith("imagen"):
                        empty += 1
                        continue
                    break  # refused: the next model may draw it
                if _transient(exc):
                    if attempt < len(RETRY_SECONDS):
                        wait = RETRY_SECONDS[attempt] * random.uniform(0.8, 1.25)
                        logger.info(f"image model busy ({reason[:80]}); trying again in {wait:.0f} s")
                        time.sleep(wait)
                        attempt += 1
                        continue
                    break
                if model.startswith("imagen") or _gone(exc):
                    if model.startswith("imagen"):
                        _state["imagen_failed"] = _state["imagen_failed"] or reason
                    _state["gone"][model] = reason
                    logger.warning(f"{model} cannot be used ({reason[:120]}); trying the next image model")
                break
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
# What every drawing must get right: simple shapes drawn correctly read better than rich details drawn wrong.
CLEAN_SHAPES = (
    "People have simple, well-proportioned bodies and hands; every object has a simple, correct and recognisable shape "
    "(fewer details rather than wrong ones)."
)
# The polished look of a 2D animated documentary (the default): simplified shapes, bold outlines, soft cel shading.
CARTOON_PROMPT = (
    "Simple 2D cartoon illustration for a calm, professional animated documentary on YouTube: {subject}. "
    "Clean bold dark outlines, simplified shapes with few details, flat colours with soft cel shading, a limited "
    "harmonious palette (teal, warm brown, cream, dusty blue, soft yellow, brick red), friendly and expressive. "
    + CLEAN_SHAPES + " One subject, whole, centred and isolated, with generous empty space around it, on a plain pure "
    "white background. No text, no letters, no numbers, no frame, no ground shadow, not photorealistic, not 3D."
)
CARTOON_SCENE_PROMPT = (
    "Wide 16:9 frame of a simple 2D cartoon animation for a calm, professional animated documentary on YouTube: "
    "{subject}. Clean bold dark outlines, simplified shapes with few details, flat colours with soft cel shading, "
    "cinematic lighting with a clear mood, a limited harmonious palette, a simple background with depth, one clear "
    "focal point and an uncluttered composition that reads at a glance. " + CLEAN_SHAPES + " "
    "No text, no letters, no numbers, no captions, no watermark, not photorealistic, not 3D."
)
# The most simplified look: flat vector shapes without outlines or shading, like a motion-graphics explainer.
FLAT_PROMPT = (
    "Flat minimalist vector illustration for an educational YouTube animation: {subject}. Simple geometric shapes, "
    "flat colours without shading or outlines, a limited palette of four or five muted colours, very few details, "
    "friendly and clear. " + CLEAN_SHAPES + " One subject, whole and centred, with generous empty space around it, "
    "on a plain pure white background. No text, no letters, no numbers, no frame, not photorealistic, not 3D."
)
FLAT_SCENE_PROMPT = (
    "Wide 16:9 flat minimalist vector illustration, like a frame of a modern motion-graphics explainer: {subject}. "
    "Simple geometric shapes, flat colours without shading, a limited palette of four or five muted colours and one "
    "accent colour, very few details, a simple background, one clear focal point and lots of breathing room. "
    + CLEAN_SHAPES + " No text, no letters, no numbers, no captions, no watermark, not photorealistic, not 3D."
)
DRAWING_STYLES = ("cartoon", "flat", "ink")
_PROMPTS = {
    "cartoon": (CARTOON_PROMPT, CARTOON_SCENE_PROMPT),
    "flat": (FLAT_PROMPT, FLAT_SCENE_PROMPT),
    "ink": (DOODLE_PROMPT, SCENE_PROMPT),
}
MASCOT_NOTE = (
    "{picture} shows the channel's mascot, an otter with round glasses, a teal sweater and a pencil behind its ear. "
    "Draw this same otter, keeping its design and colours, in the style described. "
)
CHARACTER_NOTE = "{picture} is the character sheet of {name}: draw {name} exactly like that (face, hair, clothes, colours). "
NEXT_FRAME_PROMPT = (
    "{picture} is the previous frame of an animation. Draw the next frame: keep exactly the same drawing style, "
    "setting, characters, colours and lighting, and change only this: {subject}. Everything not mentioned stays as it "
    "was. Wide 16:9 frame. No text, no letters, no numbers, no captions."
)
ART_NOTE = " Art direction of this video: {art}."
MAX_REFERENCES = 4  # pictures a drawing follows at most (the frame before, the mascot, character sheets)


def _reference_bytes(path: str) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.getchannel("A"))
    canvas.thumbnail((768, 768), Image.LANCZOS)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _picture_name(number: int, count: int) -> str:
    return "The reference picture" if count == 1 else f"Reference picture {number}"


def draw(
    subject: str, scene: bool = False, mascot: str = "", app_config=None, previous: str = "", style: str = "cartoon",
    characters: Sequence[Tuple[str, str]] = (), art: str = "", quality: str = "standard",
) -> str:
    """A drawing for the doodle look: one subject on white, or a whole 16:9 scene.

    ``mascot`` is a picture of the channel's character and ``characters``
    [(name, character sheet file)] the people of the video: the drawing keeps
    their design. ``previous`` is the scene drawn just before: the new one is
    redrawn from it with only ``subject`` changed, like the next frame of an
    animation. ``style`` is "cartoon" (a simple 2D animated documentary),
    "flat" (flat vector shapes) or "ink" (pen doodles); ``art`` is the art
    direction of the video; ``quality`` picks the image models (see
    ``image_models``). Drawings are cached by their prompt and references.
    Returns "" when drawing fails.
    """
    subject = " ".join((subject or "").split())
    if not subject:
        return ""
    app_config = _app(app_config)
    single, whole = _PROMPTS.get(style, _PROMPTS["cartoon"])
    follows = bool(previous and os.path.isfile(previous))
    sheets = [(name, path) for name, path in characters if name and path and os.path.isfile(path)]
    sources: List[Tuple[str, str]] = []  # (kind, file) in the order the prompt names them
    if follows:
        sources.append(("previous", previous))
        scene = True
    elif mascot and os.path.isfile(mascot):
        sources.append(("mascot", mascot))
    sources += [(name, path) for name, path in sheets]
    sources = sources[:MAX_REFERENCES]
    count = len(sources)
    notes = []
    for number, (kind, _) in enumerate(sources, 1):
        picture = _picture_name(number, count)
        if kind == "mascot":
            notes.append(MASCOT_NOTE.format(picture=picture))
        elif kind != "previous":
            notes.append(CHARACTER_NOTE.format(picture=picture, name=kind))
    if follows:
        prompt = NEXT_FRAME_PROMPT.format(picture=_picture_name(1, count), subject=subject) + " " + "".join(notes)
    else:
        prompt = "".join(notes) + (whole if scene else single).format(subject=subject)
    art = " ".join((art or "").split())
    if art and style != "ink":
        prompt = prompt.rstrip() + ART_NOTE.format(art=art)
    references = [_reference_bytes(path) for _, path in sources]
    models = image_models(quality, key_frame=not follows, app_config=app_config)
    size = SHARP_SIZE if scene else ""
    key = prompt + "".join(f"\n#ref{hashlib.sha1(data).hexdigest()[:12]}" for data in references) + (f"\n#{size}" if size else "")
    path = _cache_path(models[0], key)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    data = _generate_image(models, prompt, "16:9" if scene else "1:1", references, app_config, subject, size)
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


CHARACTER_SHEET = (
    "character sheet of {name}: {look}. One full-body figure standing in a relaxed three-quarter view, the whole "
    "figure visible from head to shoes, a calm friendly expression"
)
PORTRAIT_NOTE = (
    "The reference picture is a real portrait of {name}: draw the cartoon version keeping what makes them recognisable "
    "(face shape, hairstyle, beard or moustache, typical clothes). "
)


def draw_character(
    name: str, look: str, portrait: str = "", app_config=None, style: str = "cartoon", art: str = "",
    quality: str = "standard",
) -> str:
    """The character sheet of a recurring person: one full-body drawing that every later drawing of them follows.

    ``portrait`` is a real portrait of them (a painting or a photograph): the
    cartoon keeps their recognisable features. Returns "" when drawing fails.
    """
    name, look = " ".join((name or "").split()), " ".join((look or "").split())
    if not name:
        return ""
    single = _PROMPTS.get(style, _PROMPTS["cartoon"])[0]
    prompt = single.format(subject=CHARACTER_SHEET.format(name=name, look=look or name))
    references = []
    if portrait and os.path.isfile(portrait):
        prompt = PORTRAIT_NOTE.format(name=name) + prompt
        references.append(_reference_bytes(portrait))
    art = " ".join((art or "").split())
    if art and style != "ink":
        prompt += ART_NOTE.format(art=art)
    app_config = _app(app_config)
    models = image_models(quality, key_frame=True, app_config=app_config)
    key = prompt + "".join(f"\n#ref{hashlib.sha1(data).hexdigest()[:12]}" for data in references)
    path = _cache_path(models[0], key)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    data = _generate_image(models, prompt, "3:4", references, app_config, f"character sheet of {name}")
    if not data:
        return ""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.convert("RGB").save(path)
    except Exception as exc:
        logger.warning(f"character sheet of {name!r} could not be read: {exc}")
        return ""
    logger.info(f"character sheet drawn: {name!r}")
    return path


def build_drawing_check_prompt(
    description: str, mascot: bool = False, context: str = "", characters: Sequence[str] = ()
) -> str:
    otter = (
        "- the channel's mascot, an otter with round glasses and a teal sweater, looks like itself (not a different animal);\n"
        if mascot else ""
    )
    people = (
        f"- {', '.join(characters)} look like real people of their period, drawn consistently (not caricatures);\n"
        if characters else ""
    )
    said = f'It will be on screen while the narrator says: "{context}"\n' if context else ""
    fits = " and it fits what the narrator says at that moment" if context else ""
    return f"""
You are the art director of an animated documentary channel. An illustrator was asked to draw:
"{description}"
{said}Check the picture below. It passes when:
- it clearly shows what was asked, so a viewer recognises it at a glance{fits};
{otter}{people}- people have normal anatomy (two arms, two legs, at most five fingers, faces not distorted) and every object has a
  correct, recognisable shape (a telescope looks like a real telescope); nothing melted, merged, floating or half-drawn;
- it has no text, letters, numbers, labels or watermarks drawn in it (a few illegible marks on a page are fine).
Return only JSON: {{"ok": true or false, "reason": "<a few words>", "fix": "<when not ok: what to draw differently, in a few words>"}}
""".strip()


def audit_drawing(
    path: str, description: str, mascot: bool = False, app_config=None, context: str = "",
    characters: Sequence[str] = (),
) -> Tuple[Optional[bool], str]:
    """(verdict, what to fix): True when Gemini confirms a drawing shows ``description`` cleanly and fits the
    narrated ``context``, False when not, None when the check failed."""
    if not path or not os.path.isfile(path):
        return False, ""
    app_config = _app(app_config)
    try:
        prompt = build_drawing_check_prompt(description, mascot, context, characters)
        answer = _parse_answer(_ask([path], prompt, app_config)) or {}
    except Exception as exc:
        logger.warning(f"drawing check failed ({type(exc).__name__}: {exc})")
        return None, ""
    verdict = answer.get("ok")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower() in ("true", "yes", "1")
    if verdict is None:
        return None, ""
    fix = " ".join(str(answer.get("fix") or answer.get("reason") or "").split())[:160]
    if not verdict:
        logger.info(f"drawing rejected for {description!r}: {answer.get('reason', '')}")
    return bool(verdict), fix if not verdict else ""


def check_drawing(
    path: str, description: str, mascot: bool = False, app_config=None, context: str = "",
    characters: Sequence[str] = (),
) -> Optional[bool]:
    """True when Gemini confirms a drawing shows ``description`` cleanly, False when not, None if the check failed."""
    return audit_drawing(path, description, mascot, app_config, context, characters)[0]


# ---------------------------------------------------------------------------
# Short AI videos (Veo): a drawing brought to life
# ---------------------------------------------------------------------------

# Veo 3.1 models by quality, best first; the "-001" names are Vertex AI's, the "-preview" ones the Gemini API's.
VIDEO_MODELS = {
    "economy": ("veo-3.1-lite-generate-001", "veo-3.1-fast-generate-001", "veo-3.1-fast-generate-preview"),
    "standard": ("veo-3.1-fast-generate-001", "veo-3.1-fast-generate-preview", "veo-3.1-lite-generate-001"),
    "high": ("veo-3.1-generate-001", "veo-3.1-generate-preview", "veo-3.1-fast-generate-001", "veo-3.1-fast-generate-preview"),
}
VIDEO_SECONDS = (4, 6, 8)  # the lengths Veo 3.1 makes
VIDEO_POLL_SECONDS = 10.0
VIDEO_TIMEOUT_SECONDS = 600.0
VIDEO_SLOTS = 2
_video_slots = threading.BoundedSemaphore(VIDEO_SLOTS)
MOTION_PROMPT = (
    "Bring this illustration to life as one shot of a calm 2D animated documentary: {motion}. Keep exactly the same "
    "drawing style, characters, colours and framing; gentle, smooth, slow movement; the camera stays still or moves very "
    "slowly. No new objects, no text, no morphing, no cuts."
)
VIDEO_NEGATIVE = (
    "text, letters, subtitles, watermark, morphing, distorted faces, extra limbs, flicker, fast motion, scene cut, "
    "photorealistic, 3D render"
)


def video_models(quality: str = "standard", app_config=None) -> Tuple[str, ...]:
    """The Veo models to try, best first (a model named in config.toml as gemini_video_model goes first)."""
    app_config = _app(app_config)
    chain = VIDEO_MODELS.get("high" if quality == "max" else quality, VIDEO_MODELS["standard"])
    configured = str(app_config.get("gemini_video_model", "") or "").strip()
    if configured:
        chain = (configured,) + tuple(m for m in chain if m != configured)
    usable = tuple(m for m in chain if m not in _state["gone"])
    return usable or chain[-1:]


def video_length(seconds: float) -> int:
    """The shortest Veo length that covers ``seconds`` (the longest when none does)."""
    return next((length for length in VIDEO_SECONDS if length >= seconds - 0.5), VIDEO_SECONDS[-1])


def _video_cache_path(model: str, key: str) -> str:
    folder = utils.storage_dir(os.path.join("cache", "videos"), create=True)
    name = hashlib.sha1(f"{model}\n{key}".encode("utf-8")).hexdigest()[:20]
    return os.path.join(folder, f"{name}.mp4")


def _video_bytes(client, model: str, prompt: str, image: bytes, seconds: int, vertex: bool) -> bytes:
    from google.genai import types

    settings = {
        "number_of_videos": 1, "duration_seconds": seconds, "aspect_ratio": "16:9", "negative_prompt": VIDEO_NEGATIVE,
    }
    if vertex:
        settings["generate_audio"] = False  # the narration and the music are the sound (and silent video costs less)
    operation = client.models.generate_videos(
        model=model,
        source=types.GenerateVideosSource(prompt=prompt, image=types.Image(image_bytes=image, mime_type="image/png")),
        config=types.GenerateVideosConfig(**settings),
    )
    waited = 0.0
    while not operation.done:
        if waited >= VIDEO_TIMEOUT_SECONDS:
            raise _TooSlow(f"{model}: the video was not ready after {int(VIDEO_TIMEOUT_SECONDS)} s")
        time.sleep(VIDEO_POLL_SECONDS)
        waited += VIDEO_POLL_SECONDS
        operation = client.operations.get(operation)
    if getattr(operation, "error", None):
        raise RuntimeError(f"{model} failed: {operation.error}")
    result = getattr(operation, "response", None) or getattr(operation, "result", None)
    videos = list(getattr(result, "generated_videos", None) or [])
    if not videos or getattr(videos[0], "video", None) is None:
        reasons = getattr(result, "rai_media_filtered_reasons", None) or ""
        raise _NoImage(f"{model} returned no video {reasons}".strip())
    video = videos[0].video
    data = getattr(video, "video_bytes", None) or b""
    if not data and getattr(video, "uri", None):
        data = client.files.download(file=video)  # the Gemini API hands a link to the file
    if not data:
        raise _NoImage(f"{model} returned an empty video")
    return data


def animate(image: str, motion: str, seconds: float = 4.0, quality: str = "standard", app_config=None) -> str:
    """A short Veo video of the drawing ``image`` doing ``motion`` (an .mp4 file); "" when it fails.

    The video starts on the drawing itself, so it can replace a still shot.
    Videos are cached by drawing, motion and length.
    """
    motion = " ".join((motion or "").split())
    if not image or not os.path.isfile(image) or not motion:
        return ""
    app_config = _app(app_config)
    length = video_length(seconds)
    with Image.open(image) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((1920, 1920), Image.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
    data = buffer.getvalue()
    prompt = MOTION_PROMPT.format(motion=motion)
    models = video_models(quality, app_config)
    path = _video_cache_path(models[0], f"{prompt}\n{length}\n{hashlib.sha1(data).hexdigest()}")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    try:
        kwargs = _client_kwargs(app_config)
    except ValueError as exc:
        _state["last_error"] = str(exc)
        return ""
    location = str(app_config.get("gemini_video_location", "") or "").strip()
    if location and kwargs.get("vertexai"):
        kwargs["location"] = location  # Veo may live in another region than the Gemini models (us-central1)
    from google import genai

    reason = "no video model could be used"
    for model in models:
        if model in _state["gone"]:
            continue
        attempt = 0
        while True:
            try:
                with _video_slots, genai.Client(**kwargs) as client:
                    video = _video_bytes(client, model, prompt, data, length, bool(kwargs.get("vertexai")))
                with open(path, "wb") as fp:
                    fp.write(video)
                logger.info(f"video made with {model}: {motion!r}")
                return path
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"[:300]
                if _transient(exc) and not isinstance(exc, _TooSlow) and attempt < 2:
                    wait = RETRY_SECONDS[min(attempt + 1, len(RETRY_SECONDS) - 1)] * random.uniform(0.8, 1.25)
                    logger.info(f"video model busy ({reason[:80]}); trying again in {wait:.0f} s")
                    time.sleep(wait)
                    attempt += 1
                    continue
                if _gone(exc) and not isinstance(exc, _NoImage):
                    _state["gone"][model] = reason
                    logger.warning(f"{model} cannot be used ({reason[:120]}); trying the next video model")
                break
    _state["last_error"] = reason
    logger.warning(f"the video failed for {motion!r}: {reason}")
    return ""
