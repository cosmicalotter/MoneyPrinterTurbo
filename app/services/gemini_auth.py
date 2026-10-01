"""
How the google-genai client authenticates for Gemini (LLM and TTS).

By default Gemini uses ``gemini_api_key`` from Google AI Studio. With
``gemini_use_vertexai = true`` it calls Vertex AI instead and authenticates
with Application Default Credentials (``gcloud auth application-default
login`` on a workstation, or the service account of a cloud machine), so no
key is stored in config.toml.
"""

from __future__ import annotations

import os

DEFAULT_VERTEX_LOCATION = "us-central1"
_TRUE = ("1", "true", "yes", "on")


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUE


def use_vertexai(app_config) -> bool:
    """Vertex AI is used when config.toml or GOOGLE_GENAI_USE_VERTEXAI asks for it."""
    configured = app_config.get("gemini_use_vertexai")
    if configured is not None and str(configured).strip() != "":
        return _is_true(configured)
    return _is_true(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", ""))


def client_kwargs(app_config, api_key: str = "") -> dict:
    """Keyword arguments for ``google.genai.Client``.

    Raises ValueError with the setting to fix when authentication is
    incomplete, so callers can report it before sending a request.
    """
    if use_vertexai(app_config):
        project = str(app_config.get("gemini_vertex_project", "") or "").strip() or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        ).strip()
        if not project:
            raise ValueError(
                "gemini_vertex_project is not set; put your Google Cloud project ID in "
                "config.toml (or GOOGLE_CLOUD_PROJECT) to use Vertex AI"
            )
        location = (
            str(app_config.get("gemini_vertex_location", "") or "").strip()
            or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
            or DEFAULT_VERTEX_LOCATION
        )
        return {"vertexai": True, "project": project, "location": location}
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError(
            "gemini_api_key is not set; add it to config.toml or set "
            "gemini_use_vertexai = true to use Vertex AI credentials"
        )
    return {"api_key": api_key}
