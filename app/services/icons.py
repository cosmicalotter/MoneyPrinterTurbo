"""
Icons for scene diagrams, from OpenMoji (https://openmoji.org).

OpenMoji draws every emoji with the same thick dark outline and flat colours,
which is exactly the hand-drawn, minimalist look of explainer channels, and
its licence (CC BY-SA 4.0) allows use in videos with attribution. The LLM
names an icon with an emoji ("🕯️") or a couple of English words ("candle");
both resolve to an OpenMoji picture that is downloaded once and cached.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import List, Optional

import requests
from loguru import logger

from app.utils import utils

OPENMOJI_VERSION = "15.1.0"
OPENMOJI_BASE = f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/{OPENMOJI_VERSION}"
CREDIT = "Icons: OpenMoji (openmoji.org), CC BY-SA 4.0"
_SKIPPED_GROUPS = ("flags", "component")
_VARIATION = re.compile("[︎️]")
_lock = threading.Lock()
_index: Optional[List[dict]] = None


def _cache_dir() -> str:
    return utils.storage_dir(os.path.join("cache", "openmoji"), create=True)


def _get(url: str) -> requests.Response:
    response = requests.get(url, timeout=(10, 60))
    response.raise_for_status()
    return response


def load_index() -> List[dict]:
    """OpenMoji's catalogue (emoji, hexcode, annotation, tags), cached on disk."""
    global _index
    with _lock:
        if _index is not None:
            return _index
        path = os.path.join(_cache_dir(), f"openmoji-{OPENMOJI_VERSION}.json")
        data = None
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except (OSError, ValueError):
                data = None
        if data is None:
            try:
                data = _get(f"{OPENMOJI_BASE}/data/openmoji.json").json()
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False)
            except Exception as exc:
                logger.warning(f"OpenMoji catalogue could not be downloaded: {exc}")
                return []
        _index = [
            entry
            for entry in data
            if isinstance(entry, dict)
            and entry.get("hexcode")
            and not entry.get("skintone")
            and entry.get("group") not in _SKIPPED_GROUPS
        ]
        return _index


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _plain(emoji: str) -> str:
    return _VARIATION.sub("", emoji or "")


def find(icon: str, index: Optional[List[dict]] = None) -> Optional[dict]:
    """The catalogue entry for an emoji or an English keyword, or None."""
    icon = (icon or "").strip()
    if not icon:
        return None
    index = load_index() if index is None else index
    if any(ord(ch) > 0x2000 for ch in icon):
        wanted = _plain(icon)
        for entry in index:
            if _plain(entry.get("emoji", "")) == wanted:
                return entry
        # A sequence the catalogue lacks: fall back to its first character.
        first = _plain(icon)[:1]
        for entry in index:
            if _plain(entry.get("emoji", "")) == first:
                return entry
        return None

    ranked = rank(icon, index)
    return ranked[0] if ranked else None


def rank(text: str, index: Optional[List[dict]] = None, minimum: float = 2.4) -> List[dict]:
    """Catalogue entries matching English keywords, best first."""
    query = _words(text)
    if not query:
        return []
    index = load_index() if index is None else index
    scored = []
    for entry in index:
        annotation = _words(entry.get("annotation", ""))
        tags = set(_words(entry.get("tags", "")))
        curated = set(_words(entry.get("openmoji_tags", "")))
        score = 0.0
        if annotation == query:
            score += 20
        for word in query:
            if word in annotation:
                score += 4
            elif word in curated:
                score += 3.5
            elif word in tags:
                score += 3
        if not score:
            continue
        # Prefer specific, standard emoji over long descriptions.
        score -= 0.6 * max(0, len(annotation) - len(query))
        if entry.get("group", "").startswith("extras"):
            score -= 0.5
        if score >= minimum:
            scored.append((score, entry))
    scored.sort(key=lambda pair: -pair[0])
    return [entry for _, entry in scored]


def alternatives(text: str, exclude: Optional[set] = None, limit: int = 5) -> List[str]:
    """Other emoji that fit ``text`` (English keywords), leaving out ``exclude``."""
    exclude = {_plain(e) for e in (exclude or set())}
    found = []
    for entry in rank(text):
        emoji = entry.get("emoji", "")
        if emoji and _plain(emoji) not in exclude and emoji not in found:
            found.append(emoji)
        if len(found) >= limit:
            break
    return found


def fetch(icon: str, size: int = 618) -> str:
    """Local PNG path of the icon, downloading it once; "" when unavailable."""
    entry = find(icon)
    if entry is None:
        return ""
    hexcode = entry["hexcode"]
    path = os.path.join(_cache_dir(), f"{hexcode}.png")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    try:
        content = _get(f"{OPENMOJI_BASE}/color/{size}x{size}/{hexcode}.png").content
    except Exception as exc:
        logger.warning(f"OpenMoji icon {hexcode} could not be downloaded: {exc}")
        return ""
    temp = path + ".part"
    with open(temp, "wb") as fp:
        fp.write(content)
    os.replace(temp, path)
    return path
