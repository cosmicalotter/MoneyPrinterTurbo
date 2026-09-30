"""Command line entry for list-format videos ("every X explained").

Two-step workflow:

1. Generate an editable JSON script from a topic and review it:
     uv run python list_video.py --subject "Every hormone explained" --items 12 \\
       --video-language en-US --script-only --output hormones.json
2. Render the reviewed script:
     uv run python list_video.py --script hormones.json

Each item gets its own picture, an on-screen "N. name" title and a YouTube
chapter, all synchronized with its narration. Every video, voice, subtitle and
music option of cli.py is accepted as well.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from loguru import logger

import cli

# cli.py options that belong to the single-script pipeline. Accepting them
# silently would suggest they affect the list video.
_UNSUPPORTED_CLI_OPTIONS = (
    "--video-subject",
    "--video-script",
    "--video-terms",
    "--batch-file",
    "--stop-at",
    "--video-materials",
    "--video-count",
    "--custom-audio-file",
    "--paragraph-number",
    "--video-script-prompt",
    "--custom-system-prompt",
    "--match-materials-to-script",
    "--video-concat-mode",
    "--video-transition-mode",
)
_IMAGE_FILE_FIELDS = ("intro_image_file", "outro_image_file")
_DEFAULT_ITEM_COUNT = 12
_MAX_SCRIPT_BYTES = 1024 * 1024


def _item_count(value: str) -> int:
    from app.models.schema import MAX_LIST_VIDEO_ITEMS
    from app.services.llm import MIN_LIST_ITEM_COUNT

    number = int(value)
    if not MIN_LIST_ITEM_COUNT <= number <= MAX_LIST_VIDEO_ITEMS:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_LIST_ITEM_COUNT} and {MAX_LIST_VIDEO_ITEMS}"
        )
    return number


def _words_per_item(value: str) -> int:
    from app.services.llm import MAX_LIST_WORDS_PER_ITEM, MIN_LIST_WORDS_PER_ITEM

    number = int(value)
    if not MIN_LIST_WORDS_PER_ITEM <= number <= MAX_LIST_WORDS_PER_ITEM:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_LIST_WORDS_PER_ITEM} and {MAX_LIST_WORDS_PER_ITEM}"
        )
    return number


def _zoom(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 0.5:
        raise argparse.ArgumentTypeError("must be between 0 and 0.5")
    return number


def _gap(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 5:
        raise argparse.ArgumentTypeError("must be between 0 and 5 seconds")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a long list-format video (\"every X explained\"): one picture, "
            "title and chapter per item, synchronized with its narration."
        ),
        epilog="""
Examples:
  Write an editable script, review it, then render it:
    uv run python list_video.py --subject "Every hormone explained" --items 12 \\
      --video-language en-US --script-only --output hormones.json
    uv run python list_video.py --script hormones.json --video-source openai_image

  Generate and render in one go with a Colombian Spanish voice:
    uv run python list_video.py --subject "Cada fobia explicada" --items 10 \\
      --video-language es-CO --voice-name es-CO-GonzaloNeural-Male

Script format (JSON):
  {"title": "...", "intro": "...", "intro_image_term": "...",
   "items": [{"name": "...", "text": "...", "image_term": "...",
              "image_file": "optional/local/picture.png"}],
   "outro": "...", "outro_image_term": "..."}
  image_term is an English picture description for stock search or image
  generation; image_file (relative to the script file) overrides it.

Supported --video-source values: pexels, pixabay, coverr, openai_image and
local (local needs an image_file for every item). The aspect ratio defaults to
16:9. All other video, voice, subtitle and music options of cli.py are accepted
(see: uv run python cli.py --help). Files are written to
storage/tasks/<task-id>/, including chapters.txt for the YouTube description.
""",
        formatter_class=cli._CliHelpFormatter,
        # Unknown options are forwarded to cli.py; abbreviations could make
        # this parser claim them instead.
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subject", help="video topic used to write the script with the LLM")
    source.add_argument("--script", help="path to a reviewed list script JSON file")
    parser.add_argument(
        "--items",
        type=_item_count,
        default=_DEFAULT_ITEM_COUNT,
        help="number of items the LLM writes (with --subject)",
    )
    parser.add_argument(
        "--words-per-item",
        type=_words_per_item,
        default=None,
        help="approximate narration words per item (default: 110, about 45 seconds)",
    )
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="only write the generated script so it can be reviewed before rendering",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="where to save the generated script (default: storage/tasks/<task-id>/list-script.json)",
    )
    parser.add_argument(
        "--gap",
        type=_gap,
        default=None,
        help="silence after each segment in seconds (default: 0.4)",
    )
    parser.add_argument(
        "--zoom",
        type=_zoom,
        default=None,
        help="slow zoom applied to pictures over each segment, 0 disables it (default: 0.08)",
    )
    parser.add_argument(
        "--no-numbers",
        action="store_true",
        help='show "name" instead of "N. name" on screen and in chapters',
    )
    parser.add_argument(
        "--no-item-titles",
        action="store_true",
        help="do not draw the item title on screen",
    )
    parser.add_argument("--task-id", type=cli._task_id, default=None, help="task id to reuse")
    return parser


def _find_unsupported_options(forwarded: Sequence[str]) -> list[str]:
    found = []
    for arg in forwarded:
        name = arg.split("=", 1)[0]
        if name in _UNSUPPORTED_CLI_OPTIONS and name not in found:
            found.append(name)
    return found


def _has_option(forwarded: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in forwarded)


def load_script_file(path: str):
    """Load a list script and resolve image paths relative to its directory."""
    from app.models.schema import ListVideoScript

    script_path = os.path.abspath(path)
    if os.path.getsize(script_path) > _MAX_SCRIPT_BYTES:
        raise ValueError(f"script file is larger than 1 MiB: {path}")
    with open(script_path, "r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("list script must be a JSON object")

    base_dir = os.path.dirname(script_path)

    def resolve(value):
        if isinstance(value, str) and value.strip():
            value = os.path.expanduser(value.strip())
            if not os.path.isabs(value):
                value = os.path.join(base_dir, value)
            return os.path.abspath(value)
        return value

    for key in _IMAGE_FILE_FIELDS:
        if key in data:
            data[key] = resolve(data[key])
    for item in data.get("items") or []:
        if isinstance(item, dict) and "image_file" in item:
            item["image_file"] = resolve(item["image_file"])
    return ListVideoScript.model_validate(data)


def save_script_file(script, path: str) -> str:
    output = os.path.abspath(path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(script.model_dump(), fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return output


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if args.script_only and not args.subject:
        parser.error("--script-only requires --subject")
    unsupported = _find_unsupported_options(forwarded)
    if unsupported:
        parser.error(f"not available for list videos: {', '.join(unsupported)}")

    forwarded = list(forwarded)
    if not _has_option(forwarded, "--video-aspect"):
        forwarded += ["--video-aspect", "16:9"]
    # cli.py needs a subject to accept the options; stopping at "script" skips
    # checks that only apply to the single-script material stage.
    cli_args = cli.parse_args(
        [*forwarded, "--video-subject", args.subject or "list video", "--stop-at", "script"]
    )

    from pydantic import ValidationError

    try:
        params = cli.build_video_params(cli_args)
        cli.prepare_cli_files(params, stop_at="video")
        script = load_script_file(args.script) if args.script else None
    except (ValueError, OSError, ValidationError) as exc:
        logger.error(f"invalid list video input: {exc}")
        return 2

    from app.services import list_video, llm
    from app.utils import utils

    task_id = args.task_id or utils.get_uuid()
    script_file = os.path.abspath(args.script) if args.script else ""
    if script is None:
        words = args.words_per_item or llm.DEFAULT_LIST_WORDS_PER_ITEM
        script = llm.generate_list_script(
            video_subject=args.subject,
            item_count=args.items,
            language=params.video_language or "",
            words_per_item=words,
        )
        if script is None:
            logger.error("the LLM did not return a valid list script")
            return 1
        script_file = save_script_file(
            script,
            args.output or os.path.join(utils.task_dir(task_id), "list-script.json"),
        )
        logger.info(f"list script saved: {script_file}")
        if args.script_only:
            print(json.dumps({"task_id": task_id, "script_file": script_file}, ensure_ascii=False))
            return 0

    options = {}
    if args.gap is not None:
        options["gap_seconds"] = args.gap
    if args.zoom is not None:
        options["zoom"] = args.zoom
    try:
        result = list_video.generate_list_video(
            task_id,
            script,
            params,
            number_items=not args.no_numbers,
            show_item_titles=not args.no_item_titles,
            **options,
        )
    except (list_video.ListVideoError, ValueError) as exc:
        logger.error(f"list video failed: task_id={task_id}, error={exc}")
        return 1
    except Exception as exc:
        logger.exception(f"list video failed unexpectedly: task_id={task_id}, error={exc}")
        return 1

    print(
        json.dumps(
            {"task_id": task_id, "script_file": script_file, "result": result},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    cli._force_utf8_console()
    raise SystemExit(run())
