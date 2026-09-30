"""
Find pictures on the web for on-screen beats, with attribution.

Only sources whose licences allow reuse in videos are queried:

* Wikimedia Commons (no key): diagrams, anatomy, historical pictures. Most
  files need attribution, which is collected for the video description.
* Pexels photos and Pixabay images reuse the stock-video API keys from
  config.toml when they are set.

Downloads are size-limited, must decode as images, and are re-saved as PNG so
nothing but pixel data reaches the renderer.
"""

from __future__ import annotations

import hashlib
import html
import io
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests
from loguru import logger
from PIL import Image, ImageOps

from app.config import config
from app.services import material

USER_AGENT = (
    "MoneyPrinterTurbo/1.x (https://github.com/harry0703/MoneyPrinterTurbo; "
    "list video beats)"
)
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MIN_IMAGE_SIDE = 320
MAX_ASPECT_RATIO = 3.5
_WIKIMEDIA_MIMES = {"image/jpeg", "image/png", "image/svg+xml", "image/webp", "image/tiff"}


@dataclass
class WebImage:
    path: str
    source: str
    title: str
    author: str
    license: str
    page_url: str

    def credit(self) -> str:
        parts = [self.title or "Image", self.author, self.license, self.page_url]
        return " — ".join(part for part in parts if part)


@dataclass
class _Candidate:
    url: str
    source: str
    title: str
    author: str
    license: str
    page_url: str


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def _request(url: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    return requests.get(
        url,
        headers=headers,
        proxies=config.proxy,
        verify=material._get_tls_verify(),
        timeout=(15, 30),
        **kwargs,
    )


def _api_key(name: str) -> str:
    try:
        return material.get_api_key(name)
    except ValueError:
        return ""


def search_wikimedia(query: str, limit: int = 8) -> List[_Candidate]:
    response = _request(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1280,
        },
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    candidates = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("mime") not in _WIKIMEDIA_MIMES:
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        meta = info.get("extmetadata") or {}
        candidates.append(
            _Candidate(
                url=url,
                source="wikimedia",
                title=re.sub(r"^File:", "", page.get("title", "")).rsplit(".", 1)[0],
                author=_strip_html((meta.get("Artist") or {}).get("value", "")),
                license=_strip_html((meta.get("LicenseShortName") or {}).get("value", "")),
                page_url=info.get("descriptionurl", ""),
            )
        )
    return candidates


def search_pexels_photos(query: str, limit: int = 6) -> List[_Candidate]:
    key = _api_key("pexels_api_keys")
    if not key:
        return []
    response = _request(
        "https://api.pexels.com/v1/search",
        params={"query": query, "per_page": limit},
        headers={"Authorization": key},
    )
    response.raise_for_status()
    candidates = []
    for photo in response.json().get("photos") or []:
        src = photo.get("src") or {}
        url = src.get("large2x") or src.get("large") or src.get("original")
        if url:
            candidates.append(
                _Candidate(
                    url=url,
                    source="pexels",
                    title=photo.get("alt") or "Photo",
                    author=photo.get("photographer", ""),
                    license="Pexels License",
                    page_url=photo.get("url", ""),
                )
            )
    return candidates


def search_pixabay_images(query: str, limit: int = 6) -> List[_Candidate]:
    key = _api_key("pixabay_api_keys")
    if not key:
        return []
    response = _request(
        "https://pixabay.com/api/",
        params={"key": key, "q": query[:100], "per_page": max(3, limit), "safesearch": "true"},
    )
    response.raise_for_status()
    candidates = []
    for hit in response.json().get("hits") or []:
        url = hit.get("largeImageURL") or hit.get("webformatURL")
        if url:
            candidates.append(
                _Candidate(
                    url=url,
                    source="pixabay",
                    title=hit.get("tags") or "Image",
                    author=hit.get("user", ""),
                    license="Pixabay Content License",
                    page_url=hit.get("pageURL", ""),
                )
            )
    return candidates


SEARCHERS: Dict[str, Callable[[str], List[_Candidate]]] = {
    "wikimedia": search_wikimedia,
    "pexels": search_pexels_photos,
    "pixabay": search_pixabay_images,
}


def source_order(kind: str) -> List[str]:
    """Diagrams are best on Wikimedia Commons, real-life scenes on Pexels."""
    if kind == "photo":
        return ["pexels", "pixabay", "wikimedia"]
    return ["wikimedia", "pixabay", "pexels"]


def download_image(url: str, save_dir: str) -> str:
    """Download ``url`` into ``save_dir`` as PNG; raise ValueError if unusable."""
    with _request(url, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"not an image: {content_type}")
        data = io.BytesIO()
        for chunk in response.iter_content(64 * 1024):
            data.write(chunk)
            if data.tell() > MAX_DOWNLOAD_BYTES:
                raise ValueError("image is larger than 15 MiB")
    try:
        with Image.open(io.BytesIO(data.getvalue())) as image:
            image.load()
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    except Exception as exc:
        raise ValueError(f"cannot decode image: {exc}") from exc
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image too small: {width}x{height}")
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"image too narrow: {width}x{height}")
    os.makedirs(save_dir, exist_ok=True)
    name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(save_dir, f"web-{name}.png")
    image.save(path)
    return path


def find_image(
    query: str,
    save_dir: str,
    kind: str = "diagram",
    exclude_urls: Optional[set] = None,
) -> Optional[WebImage]:
    """Return the first usable picture for ``query``, or None."""
    exclude_urls = exclude_urls if exclude_urls is not None else set()
    for source in source_order(kind):
        try:
            candidates = SEARCHERS[source](query)
        except Exception as exc:
            logger.warning(f"image search failed: source={source}, query={query!r}, error={exc}")
            continue
        for candidate in candidates[:5]:
            if candidate.url in exclude_urls:
                continue
            try:
                path = download_image(candidate.url, save_dir)
            except Exception as exc:
                logger.debug(f"skip image {candidate.url}: {exc}")
                continue
            exclude_urls.add(candidate.url)
            logger.info(f"image found: source={source}, query={query!r}")
            return WebImage(
                path=path,
                source=candidate.source,
                title=candidate.title,
                author=candidate.author,
                license=candidate.license,
                page_url=candidate.page_url,
            )
    logger.warning(f"no usable image found for {query!r}")
    return None
