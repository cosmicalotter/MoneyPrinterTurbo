"""
Competitor research: which videos worked for channels like yours, and why.

Uses the official YouTube Data API v3 (a free API key from Google Cloud, set
as ``youtube_api_key`` in config.toml). Only public data exists for other
channels: views, likes, comments, dates, durations and titles. Retention and
click-through rate are private to each channel's owner.

The most useful number derived from public data is the *outlier score*: a
video's views divided by the median views of its channel's other videos of
the same format. A score of 10 means the topic and packaging pulled ten times
more than that channel usually does, independent of channel size.

Quota: a channel costs about 1 unit plus 2 units per 50 videos; a channel
search costs 100 units. The free daily quota is 10,000 units.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote

import requests
from loguru import logger

from app.config import config
from app.services import llm, material

API_BASE = "https://www.googleapis.com/youtube/v3"
SHORT_MAX_SECONDS = 60
MAYBE_SHORT_MAX_SECONDS = 180
DURATION_BUCKETS = (
    (60, "0-1 min"),
    (300, "1-5 min"),
    (600, "5-10 min"),
    (1200, "10-20 min"),
    (2400, "20-40 min"),
    (float("inf"), "40+ min"),
)
CSV_FIELDS = (
    "channel",
    "channel_subscribers",
    "title",
    "url",
    "published",
    "age_days",
    "duration_seconds",
    "format",
    "views",
    "likes",
    "comments",
    "views_per_day",
    "outlier_score",
    "engagement_rate",
    "comments_per_1k_views",
    "views_per_subscriber",
    "title_length",
    "title_has_number",
    "title_has_question",
)


class ResearchError(RuntimeError):
    """Raised when the YouTube API cannot be used."""


@dataclass
class Channel:
    id: str
    title: str
    subscribers: int
    video_count: int
    uploads_playlist: str


@dataclass
class VideoRow:
    channel: str
    channel_subscribers: int
    video_id: str
    title: str
    published: str
    age_days: float
    duration_seconds: int
    views: int
    likes: int
    comments: int
    format: str = ""
    views_per_day: float = 0.0
    outlier_score: float = 0.0
    engagement_rate: float = 0.0
    comments_per_1k_views: float = 0.0
    views_per_subscriber: float = 0.0
    title_length: int = 0
    title_has_number: bool = False
    title_has_question: bool = False
    tags: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------


def api_key() -> str:
    key = str(config.app.get("youtube_api_key", "") or "").strip()
    if not key:
        raise ResearchError(
            "youtube_api_key is not set in config.toml; create a free key for the "
            "YouTube Data API v3 at https://console.cloud.google.com/apis/credentials"
        )
    return key


def _get(endpoint: str, params: dict) -> dict:
    response = requests.get(
        f"{API_BASE}/{endpoint}",
        params={**params, "key": api_key()},
        proxies=config.proxy,
        verify=material._get_tls_verify(),
        timeout=(15, 30),
    )
    if response.status_code != 200:
        try:
            message = response.json()["error"]["message"]
        except Exception:
            message = response.text[:200]
        raise ResearchError(f"YouTube API {endpoint} failed ({response.status_code}): {message}")
    return response.json()


def parse_channel_reference(reference: str) -> dict:
    """Turn a handle, channel URL or channel ID into API lookup params."""
    value = (reference or "").strip()
    match = re.search(r"youtube\.com/(@[\w.\-%]+|channel/(UC[\w-]{20,}))", value)
    if match:
        value = match.group(2) or match.group(1)
    # Handles with accents arrive percent-encoded in copied URLs.
    value = unquote(value)
    if re.fullmatch(r"UC[\w-]{20,}", value):
        return {"id": value}
    if value and not value.startswith("@"):
        value = f"@{value}"
    if len(value) < 2:
        raise ResearchError(f"not a channel handle, URL or ID: {reference!r}")
    return {"forHandle": value}


def get_channel(reference: str) -> Channel:
    data = _get("channels", {"part": "snippet,statistics,contentDetails", **parse_channel_reference(reference)})
    items = data.get("items") or []
    if not items:
        raise ResearchError(f"channel not found: {reference}")
    item = items[0]
    statistics_ = item.get("statistics") or {}
    return Channel(
        id=item["id"],
        title=(item.get("snippet") or {}).get("title", reference),
        subscribers=int(statistics_.get("subscriberCount") or 0),
        video_count=int(statistics_.get("videoCount") or 0),
        uploads_playlist=item["contentDetails"]["relatedPlaylists"]["uploads"],
    )


def search_channels(query: str, limit: int = 10) -> List[str]:
    """Channel IDs matching ``query`` (100 quota units per call)."""
    data = _get(
        "search",
        {"part": "snippet", "type": "channel", "q": query, "maxResults": max(1, min(limit, 50))},
    )
    return [item["id"]["channelId"] for item in data.get("items") or [] if item.get("id", {}).get("channelId")]


def list_video_ids(channel: Channel, max_videos: int) -> List[str]:
    ids: List[str] = []
    token = None
    while len(ids) < max_videos:
        params = {"part": "contentDetails", "playlistId": channel.uploads_playlist, "maxResults": 50}
        if token:
            params["pageToken"] = token
        data = _get("playlistItems", params)
        ids += [item["contentDetails"]["videoId"] for item in data.get("items") or []]
        token = data.get("nextPageToken")
        if not token:
            break
    return ids[:max_videos]


def parse_duration(value: str) -> int:
    """ISO 8601 duration (PT1H2M3S) to seconds."""
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", value or "")
    if not match:
        return 0
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def fetch_videos(channel: Channel, video_ids: List[str], now: Optional[datetime] = None) -> List[VideoRow]:
    now = now or datetime.now(timezone.utc)
    rows = []
    for start in range(0, len(video_ids), 50):
        data = _get(
            "videos",
            {"part": "snippet,statistics,contentDetails", "id": ",".join(video_ids[start : start + 50])},
        )
        for item in data.get("items") or []:
            snippet = item.get("snippet") or {}
            stats = item.get("statistics") or {}
            if snippet.get("liveBroadcastContent") in ("live", "upcoming"):
                continue
            published = snippet.get("publishedAt", "")
            try:
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
                age_days = max((now - published_at).total_seconds() / 86400, 0.5)
            except ValueError:
                age_days = 0.5
            rows.append(
                VideoRow(
                    channel=channel.title,
                    channel_subscribers=channel.subscribers,
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    published=published[:10],
                    age_days=round(age_days, 1),
                    duration_seconds=parse_duration((item.get("contentDetails") or {}).get("duration", "")),
                    views=int(stats.get("viewCount") or 0),
                    likes=int(stats.get("likeCount") or 0),
                    comments=int(stats.get("commentCount") or 0),
                    tags=list(snippet.get("tags") or [])[:15],
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def video_format(seconds: int) -> str:
    # The API does not say whether a video is a Short; Shorts can run up to
    # three minutes, so that range is marked as uncertain.
    if seconds <= SHORT_MAX_SECONDS:
        return "short"
    if seconds <= MAYBE_SHORT_MAX_SECONDS:
        return "short?"
    return "long"


def duration_bucket(seconds: int) -> str:
    return next(label for limit, label in DURATION_BUCKETS if seconds <= limit)


def add_metrics(rows: List[VideoRow]) -> List[VideoRow]:
    """Fill derived metrics; outliers are measured within channel and format."""
    groups: Dict[tuple, List[VideoRow]] = {}
    for row in rows:
        row.format = video_format(row.duration_seconds)
        groups.setdefault((row.channel, row.format), []).append(row)
    for group in groups.values():
        median = statistics.median(r.views for r in group) or 1
        for row in group:
            row.outlier_score = round(row.views / median, 2)
    for row in rows:
        views = max(row.views, 1)
        row.views_per_day = round(row.views / row.age_days, 1)
        row.engagement_rate = round((row.likes + row.comments) / views, 4)
        row.comments_per_1k_views = round(row.comments * 1000 / views, 2)
        row.views_per_subscriber = round(row.views / row.channel_subscribers, 3) if row.channel_subscribers else 0.0
        row.title_length = len(row.title)
        row.title_has_number = bool(re.search(r"\d", row.title))
        row.title_has_question = "?" in row.title
    return rows


def _median(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.median(values), 1) if values else 0.0


def summarize(rows: List[VideoRow]) -> dict:
    """Aggregate patterns that are cheaper to compute than to ask a model for."""
    long_rows = [r for r in rows if r.format == "long"]
    summary = {
        "videos": len(rows),
        "channels": sorted({r.channel for r in rows}),
        "by_format": {},
        "long_by_duration": {},
        "long_title_features": {},
    }
    for fmt in ("short", "short?", "long"):
        group = [r for r in rows if r.format == fmt]
        if group:
            summary["by_format"][fmt] = {
                "count": len(group),
                "median_views": _median(r.views for r in group),
                "median_views_per_day": _median(r.views_per_day for r in group),
            }
    for _, label in DURATION_BUCKETS:
        group = [r for r in long_rows if duration_bucket(r.duration_seconds) == label]
        if group:
            summary["long_by_duration"][label] = {
                "count": len(group),
                "median_outlier_score": _median(r.outlier_score for r in group),
                "median_views": _median(r.views for r in group),
            }
    for name, test in (
        ("with_number", lambda r: r.title_has_number),
        ("without_number", lambda r: not r.title_has_number),
        ("with_question", lambda r: r.title_has_question),
        ("short_title_under_50", lambda r: r.title_length < 50),
        ("long_title_50_plus", lambda r: r.title_length >= 50),
    ):
        group = [r for r in long_rows if test(r)]
        if group:
            summary["long_title_features"][name] = {
                "count": len(group),
                "median_outlier_score": _median(r.outlier_score for r in group),
            }
    return summary


def research(
    channels: List[str],
    searches: List[str] = (),
    max_videos: int = 300,
    search_limit: int = 5,
    now: Optional[datetime] = None,
) -> tuple[List[VideoRow], List[str]]:
    """Collect videos from channels (and channels found by searches)."""
    references = list(channels)
    for query in searches:
        found = search_channels(query, search_limit)
        logger.info(f"search {query!r}: {len(found)} channels")
        references += found
    rows: List[VideoRow] = []
    warnings: List[str] = []
    seen = set()
    for reference in references:
        try:
            channel = get_channel(reference)
        except ResearchError as exc:
            warnings.append(str(exc))
            continue
        if channel.id in seen:
            continue
        seen.add(channel.id)
        ids = list_video_ids(channel, max_videos)
        videos = fetch_videos(channel, ids, now=now)
        logger.info(f"{channel.title}: {len(videos)} videos, {channel.subscribers} subscribers")
        rows += videos
    return add_metrics(rows), warnings


def write_csv(rows: List[VideoRow], path: str) -> str:
    ordered = sorted(rows, key=lambda r: r.outlier_score, reverse=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in ordered:
            data = asdict(row)
            data["url"] = row.url
            writer.writerow({key: data[key] for key in CSV_FIELDS})
    return path


# ---------------------------------------------------------------------------
# AI analysis
# ---------------------------------------------------------------------------


def _table(rows: List[VideoRow]) -> str:
    lines = ["title | channel | views | outlier | age_days | minutes | engagement"]
    for r in rows:
        lines.append(
            f"{r.title} | {r.channel} | {r.views} | {r.outlier_score} | {r.age_days:.0f} | "
            f"{r.duration_seconds / 60:.1f} | {r.engagement_rate:.3f}"
        )
    return "\n".join(lines)


def build_analysis_prompt(rows: List[VideoRow], summary: dict, brief: str, language: str) -> str:
    long_rows = sorted((r for r in rows if r.format == "long"), key=lambda r: r.outlier_score, reverse=True)
    shorts = sorted((r for r in rows if r.format == "short"), key=lambda r: r.outlier_score, reverse=True)
    return f"""
# Role: YouTube strategist for an educational channel

## My channel:
{brief}

## Data:
Public statistics of videos from channels similar to mine. "outlier" is the
video's views divided by the median views of the same channel's videos of the
same format: above 3 means the topic and packaging clearly beat that channel's
normal level. Retention and click-through rate are not public.

### Aggregated patterns (computed, trust these numbers):
{json.dumps(summary, ensure_ascii=False)}

### Top long-form outliers:
{_table(long_rows[:40])}

### Weakest long-form videos:
{_table(long_rows[-15:])}

### Top Shorts outliers:
{_table(shorts[:15])}

## Task (answer in {language}, markdown):
1. Topic patterns: which themes and emotions (fear, curiosity, awe, money, disgust...) the outliers share, and which topics flop.
2. Title patterns: structures, lengths and words that recur in outliers; compare with the aggregated numbers.
3. Format: the duration that works best, and what Shorts teach about hooks.
4. Gaps: topics with proven demand that these channels have not covered well.
5. 20 video ideas for my channel, ranked by expected potential, each with a title in my channel's style, a thumbnail concept (3-4 words of text plus the main image) and which outlier(s) inspired it.
Base every claim on the data above; say when the evidence is weak.
""".strip()


def analyze(rows: List[VideoRow], brief: str, language: str = "Spanish") -> str:
    if not rows:
        raise ResearchError("no videos to analyze")
    prompt = build_analysis_prompt(rows, summarize(rows), brief, language)
    response = llm._generate_response(prompt)
    if not response or response.startswith("Error: "):
        raise ResearchError(f"the LLM could not analyze the data: {response}")
    return response.strip()


def load_csv(path: str) -> List[VideoRow]:
    """Read a CSV written by write_csv, so it can be analyzed again."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        for record in csv.DictReader(fp):
            match = re.search(r"v=([\w-]+)", record.get("url", ""))
            rows.append(
                VideoRow(
                    channel=record.get("channel", ""),
                    channel_subscribers=int(float(record.get("channel_subscribers") or 0)),
                    video_id=match.group(1) if match else "",
                    title=record.get("title", ""),
                    published=record.get("published", ""),
                    age_days=float(record.get("age_days") or 0.5),
                    duration_seconds=int(float(record.get("duration_seconds") or 0)),
                    views=int(float(record.get("views") or 0)),
                    likes=int(float(record.get("likes") or 0)),
                    comments=int(float(record.get("comments") or 0)),
                )
            )
    return add_metrics(rows)
