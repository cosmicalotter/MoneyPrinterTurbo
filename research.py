"""Research what works for channels like yours (YouTube Data API v3).

Collect the public statistics of every video of the channels you choose,
rank them by outlier score (views compared with the channel's own median)
and optionally ask the configured LLM for patterns and video ideas.

    uv run python research.py --channel @SomeChannel --channel @Another \\
      --search "every X explained" --analyze \\
      --brief "Cabeceando: general topics explained for otters, long list videos"

Set youtube_api_key in config.toml first (free key for the YouTube Data API v3
at https://console.cloud.google.com/apis/credentials). Retention and
click-through rate are private to each channel and are not available.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Sequence

from loguru import logger

DEFAULT_BRIEF = (
    "A faceless educational YouTube channel publishing long list videos "
    '("every X explained") with a mascot host and a recurring title formula.'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect public statistics of channels like yours, rank their videos by "
            "outlier score and optionally ask the LLM for patterns and ideas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--channel", action="append", default=[], help="handle, channel URL or ID; repeatable")
    parser.add_argument("--channels-file", help="text file with one channel per line")
    parser.add_argument("--search", action="append", default=[], help="also research channels found by this query (100 quota units); repeatable")
    parser.add_argument("--search-limit", type=int, default=5, help="channels taken from each search")
    parser.add_argument("--max-videos", type=int, default=300, help="most recent videos read per channel")
    parser.add_argument("--from-csv", help="analyze an existing research CSV instead of calling the API")
    parser.add_argument("--output", help="CSV path (default: storage/research/research-<date>.csv)")
    parser.add_argument("--analyze", action="store_true", help="ask the configured LLM for patterns and 20 video ideas")
    parser.add_argument("--brief", default=DEFAULT_BRIEF, help="short description of your channel for the analysis")
    parser.add_argument("--language", default="Spanish", help="language of the analysis")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    channels = list(args.channel)
    if args.channels_file:
        with open(args.channels_file, "r", encoding="utf-8-sig") as fp:
            channels += [line.strip() for line in fp if line.strip() and not line.startswith("#")]
    if not args.from_csv and not channels and not args.search:
        parser.error("give --channel, --channels-file, --search or --from-csv")
    if args.max_videos < 1 or args.search_limit < 1:
        parser.error("--max-videos and --search-limit must be positive")

    from app.services import channel_research as research
    from app.utils import utils

    warnings = []
    try:
        if args.from_csv:
            rows = research.load_csv(args.from_csv)
            csv_path = os.path.abspath(args.from_csv)
        else:
            rows, warnings = research.research(
                channels, args.search, max_videos=args.max_videos, search_limit=args.search_limit
            )
            if not rows:
                logger.error("no videos were collected")
                return 1
            csv_path = os.path.abspath(
                args.output
                or os.path.join(
                    utils.storage_dir("research", create=True),
                    f"research-{datetime.now():%Y%m%d-%H%M}.csv",
                )
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            research.write_csv(rows, csv_path)
        summary = {"csv": csv_path, "videos": len(rows), "warnings": warnings}
        if args.analyze:
            analysis = research.analyze(rows, args.brief, args.language)
            analysis_path = os.path.splitext(csv_path)[0] + "-analysis.md"
            with open(analysis_path, "w", encoding="utf-8") as fp:
                fp.write(analysis + "\n")
            summary["analysis"] = analysis_path
    except (research.ResearchError, OSError, ValueError) as exc:
        logger.error(f"research failed: {exc}")
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
