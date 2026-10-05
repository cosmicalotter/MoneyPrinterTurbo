"""
Google voices for narration: Gemini TTS done robustly, Google Cloud
Text-to-Speech (Chirp 3 HD, Neural2, Studio...), delivery presets and a small
voice lab to compare voices on the same text.

Gemini TTS ("gemini:<Voice>") follows natural-language directions placed
before the text, so a preset such as "divulgador" sets the tone, energy and
pace. Requests are retried on transient errors, a model that does not exist
falls back to the next one, and audio that comes back cut short is asked for
again.

Google Cloud TTS ("gcloud:<voice name>", e.g. "gcloud:es-US-Chirp3-HD-Charon")
uses the Text-to-Speech REST API with Application Default Credentials (the same
``gcloud auth application-default login`` as Vertex AI) or with
``gcloud_tts_api_key``. Speed comes from the voice rate; pitch
(``gcloud_tts_pitch``, in semitones) works on Neural2/Studio/WaveNet voices.

Approximate 2025 list prices, check yours: Gemini 2.5 Flash TTS about
US$0.015 per minute of audio (Pro about US$0.03); Cloud TTS Chirp 3 HD about
US$30 per million characters (about US$0.03 per minute) with a monthly free
tier; ElevenLabs uses about 1 credit per character.
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional, Sequence

import requests
from loguru import logger

from app.config import config

GEMINI_TTS_MODELS = ("gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts")
GEMINI_TTS_ATTEMPTS = 3
GCLOUD_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
GCLOUD_TTS_ATTEMPTS = 3
GCLOUD_CHUNK_BYTES = 4500  # the API takes at most 5000 bytes of text per request
SAMPLE_RATE = 24000

# Delivery directions for Gemini voices (Spanish, since the channel narrates in
# Spanish; Gemini follows them in any language).
VOICE_STYLE_PRESETS: Dict[str, str] = {
    "divulgador": (
        "Habla como un divulgador científico joven latinoamericano, cercano y curioso, que le explica algo "
        "fascinante a un amigo: voz cálida y clara, energía natural, sonrisa en la voz, ritmo ágil pero sin prisa, "
        "pausas breves antes de los datos importantes"
    ),
    "entusiasta": (
        "Habla como un youtuber joven y entusiasta que no puede esperar a contar lo que descubrió: mucha energía, "
        "ritmo rápido, sorpresa genuina en los datos curiosos, pero siempre claro"
    ),
    "profe": (
        "Habla como un profesor joven y paciente que explica paso a paso: tono amable y seguro, ritmo pausado, "
        "remarca los términos técnicos y las unidades para que se entiendan"
    ),
    "calmado": (
        "Habla con calma y cercanía, como un narrador de documental joven que conversa en voz baja: voz cálida, "
        "suave y natural, ritmo tranquilo, una pausa clara al terminar cada idea, sin prisa y sin dramatizar"
    ),
    "sereno": (
        "Narra como un divulgador sereno de un canal de animación educativa: voz cálida y un poco grave, muy natural, "
        "ritmo lento y pausado, silencios breves entre frases para que cada idea se asiente, tono amable y seguro, "
        "como si le explicaras algo importante a un amigo en una noche tranquila"
    ),
    "narrador": (
        "Narra como un documental de ciencia: voz profunda y segura, ritmo medido, misterio y asombro en las preguntas"
    ),
}
VOICE_STYLE_PRESETS["divulgador-es"] = VOICE_STYLE_PRESETS["divulgador"]

# Young, clear male voices that explain well in Spanish, to audition first.
SPANISH_MALE_SHORTLIST = (
    "gemini:Puck-Upbeat",
    "gemini:Charon-Informative",
    "gemini:Achird-Friendly",
    "gemini:Sadachbia-Lively",
    "gemini:Iapetus-Clear",
    "gemini:Fenrir-Excitable",
    "gcloud:es-US-Chirp3-HD-Puck",
    "gcloud:es-US-Chirp3-HD-Charon",
    "gcloud:es-US-Chirp3-HD-Achird",
    "gcloud:es-US-Chirp3-HD-Iapetus",
    "es-CO-GonzaloNeural-Male",
    "es-MX-JorgeNeural-Male",
)
CHIRP_VOICE_NAMES = (
    "Achird", "Algenib", "Algieba", "Alnilam", "Charon", "Enceladus", "Fenrir", "Iapetus", "Orus", "Puck",
    "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Umbriel", "Zubenelgenubi",
    "Achernar", "Aoede", "Autonoe", "Callirrhoe", "Despina", "Erinome", "Gacrux", "Kore", "Laomedeia",
    "Leda", "Pulcherrima", "Sulafat", "Vindemiatrix", "Zephyr",
)
GCLOUD_LOCALES = ("es-US", "es-ES")


def resolve_style(style: str) -> str:
    """A preset name ("divulgador") becomes its directions; any other text is kept as is."""
    text = (style or "").strip()
    return VOICE_STYLE_PRESETS.get(text.lower(), text)


def gemini_delivery(style: str, rate: float = 1.0) -> str:
    """The directions sent before the text: the style plus the pace asked by the voice rate."""
    style = resolve_style(style).rstrip(":").strip()
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 1.0
    if not math.isfinite(rate) or rate <= 0:
        rate = 1.0
    pace = ""
    if rate >= 1.15:
        pace = "at a fast, energetic pace"
    elif rate >= 1.05:
        pace = "at a slightly brisk pace"
    elif rate <= 0.85:
        pace = "slowly and calmly"
    elif rate <= 0.95:
        pace = "at a slightly relaxed pace"
    if pace:
        return f"{style}, {pace}" if style else f"Read this {pace}"
    return style


def gemini_models(configured: str) -> List[str]:
    """The configured TTS model first, then the known ones (used when a model does not exist)."""
    models = [configured.strip()] if configured and configured.strip() else []
    return models + [m for m in GEMINI_TTS_MODELS if m not in models]


def _missing_model(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text or "not_found" in text or "not found" in text or "is not supported" in text


def expected_seconds(text: str) -> float:
    """A rough lower bound for how long the narration of ``text`` lasts."""
    return len(re.findall(r"\w+", text or "")) / 5.0


def gemini_audio(
    request: Callable[[str, str], Optional[bytes]],
    text: str,
    contents: str,
    model: str,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[bytes]:
    """PCM audio from Gemini, retrying transient failures and audio that was cut short.

    ``request(model, contents)`` sends one request and returns the raw PCM
    (16-bit mono at 24 kHz) or None when the answer had no audio.
    """
    best = None
    for candidate in gemini_models(model):
        for attempt in range(GEMINI_TTS_ATTEMPTS):
            try:
                audio = request(candidate, contents)
            except Exception as exc:
                if _missing_model(exc):
                    logger.warning(f"Gemini TTS model {candidate!r} is not available ({exc}); trying the next one")
                    break
                logger.warning(f"Gemini TTS attempt {attempt + 1} failed: {type(exc).__name__}: {exc}")
                if attempt + 1 < GEMINI_TTS_ATTEMPTS:
                    sleep(2.0 * (attempt + 1))
                continue
            if not audio:
                logger.warning(f"Gemini TTS attempt {attempt + 1} returned no audio")
                continue
            seconds = len(audio) / (2 * SAMPLE_RATE)
            if best is None or len(audio) > len(best):
                best = audio
            if seconds >= expected_seconds(text) * 0.5:
                return audio
            logger.warning(f"Gemini TTS audio looks cut short ({seconds:.1f} s for {len(text)} characters); asking again")
        if best is not None:
            return best
    return best


# ---------------------------------------------------------------------------
# Google Cloud Text-to-Speech
# ---------------------------------------------------------------------------


def is_gcloud_voice(voice_name: str) -> bool:
    return (voice_name or "").startswith("gcloud:")


def gcloud_voice_name(voice_name: str) -> str:
    """"gcloud:es-US-Chirp3-HD-Charon-Male" -> "es-US-Chirp3-HD-Charon"."""
    name = (voice_name or "").split(":", 1)[-1].strip()
    return re.sub(r"-(Male|Female)$", "", name)


def get_gcloud_voices(locales: Sequence[str] = GCLOUD_LOCALES) -> List[str]:
    """Chirp 3 HD voices for ``locales`` (the most natural Cloud TTS voices)."""
    return [f"gcloud:{locale}-Chirp3-HD-{name}" for locale in locales for name in CHIRP_VOICE_NAMES]


def _gcloud_headers(app_config) -> dict:
    key = str(app_config.get("gcloud_tts_api_key", "") or "").strip() or os.environ.get("GOOGLE_TTS_API_KEY", "").strip()
    if key:
        return {"X-Goog-Api-Key": key}
    import google.auth
    from google.auth.transport.requests import Request

    credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    headers = {"Authorization": f"Bearer {credentials.token}"}
    quota = (
        str(app_config.get("gemini_vertex_project", "") or "").strip()
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or getattr(credentials, "quota_project_id", None)
        or project
        or ""
    )
    if quota:
        # User credentials need a project to bill the request to.
        headers["x-goog-user-project"] = quota
    return headers


def split_text(text: str, limit: int = GCLOUD_CHUNK_BYTES) -> List[str]:
    """Pieces of at most ``limit`` UTF-8 bytes, cut between sentences (or words)."""
    chunks: List[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?…])\s+", " ".join((text or "").split())):
        pieces = [sentence]
        if len(sentence.encode("utf-8")) > limit:
            pieces, word_chunk = [], ""
            for word in sentence.split():
                if word_chunk and len(f"{word_chunk} {word}".encode("utf-8")) > limit:
                    pieces.append(word_chunk)
                    word_chunk = word
                else:
                    word_chunk = f"{word_chunk} {word}".strip()
            pieces.append(word_chunk)
        for piece in pieces:
            if current and len(f"{current} {piece}".encode("utf-8")) > limit:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def gcloud_request_body(text: str, voice: str, rate: float, volume: float, pitch: float) -> dict:
    language = "-".join(voice.split("-")[:2])
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        rate = 1.0
    audio = {
        "audioEncoding": "LINEAR16",
        "sampleRateHertz": SAMPLE_RATE,
        "speakingRate": min(2.0, max(0.25, rate if math.isfinite(rate) and rate > 0 else 1.0)),
    }
    if volume and volume > 0 and abs(volume - 1.0) > 1e-3:
        audio["volumeGainDb"] = round(min(16.0, max(-96.0, 20 * math.log10(volume))), 2)
    if pitch and "Chirp" not in voice:
        audio["pitch"] = min(20.0, max(-20.0, float(pitch)))
    return {"input": {"text": text}, "voice": {"languageCode": language, "name": voice}, "audioConfig": audio}


def gcloud_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
    app_config=None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Narration from Google Cloud TTS saved as MP3; the SubMaker for subtitles, or None."""
    from pydub import AudioSegment

    from app.services import voice as voice_service

    voice_service._configure_pydub_ffmpeg(AudioSegment)
    app_config = app_config if app_config is not None else config.app
    voice = gcloud_voice_name(voice_name)
    if not voice:
        logger.error(f"invalid Google Cloud voice: {voice_name!r}")
        return None
    try:
        headers = _gcloud_headers(app_config)
    except Exception as exc:
        logger.error(
            f"Google Cloud TTS is not configured ({type(exc).__name__}: {exc}); run "
            "`gcloud auth application-default login` or set gcloud_tts_api_key"
        )
        return None
    pitch = app_config.get("gcloud_tts_pitch", 0) or 0
    try:
        pitch = float(pitch)
    except (TypeError, ValueError):
        pitch = 0.0
    audio = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    for number, chunk in enumerate(split_text(text)):
        body = gcloud_request_body(chunk, voice, voice_rate, voice_volume, pitch)
        content = None
        for attempt in range(GCLOUD_TTS_ATTEMPTS):
            try:
                response = requests.post(GCLOUD_TTS_URL, json=body, headers=headers, timeout=120)
            except requests.RequestException as exc:
                logger.warning(f"Google Cloud TTS request failed: {exc}")
                sleep(2.0 * (attempt + 1))
                continue
            if response.status_code == 200:
                content = response.json().get("audioContent")
                break
            message = response.text[:300]
            if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < GCLOUD_TTS_ATTEMPTS:
                logger.warning(f"Google Cloud TTS busy ({response.status_code}); retrying")
                sleep(2.0 * (attempt + 1))
                continue
            logger.error(f"Google Cloud TTS failed ({response.status_code}): {message}")
            return None
        if not content:
            logger.error("Google Cloud TTS returned no audio")
            return None
        piece = AudioSegment.from_file(io.BytesIO(base64.b64decode(content)), format="wav")
        if number:
            audio += AudioSegment.silent(duration=250, frame_rate=piece.frame_rate)
        audio += piece
    voice_service.ensure_file_path_exists(voice_file)
    exported = audio.export(voice_file, format="mp3", bitrate="160k")
    exported.close()
    logger.info(f"Google Cloud TTS completed: {voice_file}")
    sub_maker = voice_service.ensure_legacy_submaker_fields(voice_service.SubMaker())
    return voice_service.populate_legacy_submaker_with_full_text(
        sub_maker=sub_maker, text=text, audio_duration_seconds=len(audio) / 1000.0
    )


# ---------------------------------------------------------------------------
# Voice lab
# ---------------------------------------------------------------------------

LAB_TEXT = (
    "¿Sabías que tu corazón tiene su propio marcapasos? Se llama nodo sinusal, y cada segundo lanza una "
    "pequeña chispa eléctrica que hace latir al corazón. El voltaje se mide en voltios, y la corriente, "
    "en amperios. Y aquí viene lo curioso: esa chispa es tan pequeña que no encendería ni un bombillo."
)


def safe_name(voice_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", voice_name).strip("_") or "voice"


def audition(
    voices: Sequence[str], text: str, out_dir: str, rate: float = 1.0, style: str = ""
) -> List[dict]:
    """Narrate the same text with each voice so they can be compared side by side."""
    from app.services import voice as voice_service

    os.makedirs(out_dir, exist_ok=True)
    previous = config.app.get("gemini_tts_style")
    results = []
    try:
        if style:
            config.app["gemini_tts_style"] = style
        for name in voices:
            path = os.path.join(out_dir, f"{safe_name(name)}.mp3")
            started = time.time()
            try:
                made = voice_service.tts(text, name, rate, path)
            except Exception as exc:
                made, error = None, f"{type(exc).__name__}: {exc}"
            else:
                error = "" if made is not None else "no audio (see the log)"
            ok = made is not None and os.path.isfile(path)
            results.append({
                "voice": name, "file": path if ok else "", "ok": ok, "error": error,
                "seconds": round(time.time() - started, 1),
            })
    finally:
        if style:
            if previous is None:
                config.app.pop("gemini_tts_style", None)
            else:
                config.app["gemini_tts_style"] = previous
    return results


def estimate_cost(characters: int, provider: str) -> float:
    """Rough US$ cost of narrating ``characters`` (2025 list prices; 15 characters per second)."""
    minutes = characters / 15.0 / 60.0
    per_minute = {"gemini-flash": 0.015, "gemini-pro": 0.03, "gcloud-chirp": 0.027, "edge": 0.0}
    if provider == "elevenlabs":
        return round(characters * 0.00018, 3)  # Creator plan: about US$0.18 per 1,000 credits
    return round(minutes * per_minute.get(provider, 0.0), 3)
