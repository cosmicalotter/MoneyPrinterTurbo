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
DEMO_ASSETS = "demo"


def built_in_assets_dir(name: str) -> str:
    """A host bundled in resource/characters (e.g. "nutria"), or ""."""
    from app.utils import utils

    if not name or os.sep in name or "/" in name or name.startswith("."):
        return ""
    folder = utils.resource_dir(os.path.join("characters", name))
    return folder if os.path.isdir(os.path.join(folder, "personaje")) else ""


def demo_assets_dir() -> str:
    """The built-in red blood cell host, drawn on first use."""
    from app.services.list_video_fx import create_demo_assets
    from app.utils import utils

    folder = utils.storage_dir("demo-assets", create=True)
    if not os.path.isdir(os.path.join(folder, "personaje")):
        create_demo_assets(folder)
    return folder
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


def _accent(value: str) -> str:
    from app.services.list_video_fx import parse_color

    try:
        parse_color(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _scene_color(value: str) -> str:
    if value.strip().lower() in ("accent", "white"):
        return value.strip().lower()
    return _accent(value)


def _languages(value: str) -> list:
    languages = []
    for part in value.split(","):
        language = part.strip()
        if language and language not in languages:
            languages.append(language)
    if not languages:
        raise argparse.ArgumentTypeError("give at least one language, e.g. en-US")
    return languages


def per_language(values: Sequence[str], languages: Sequence[str], option: str) -> dict:
    """Map "LANG=VALUE" entries (or one bare VALUE for a single language)."""
    mapping = {}
    for value in values:
        prefix, separator, rest = value.partition("=")
        if separator and prefix.strip() in languages:
            mapping[prefix.strip()] = rest.strip()
        elif len(languages) == 1:
            mapping[languages[0]] = value.strip()
        else:
            raise ValueError(
                f"{option} needs LANG=VALUE with one of {', '.join(languages)}: {value!r}"
            )
    return mapping


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

Automatic editing (on by default, --no-edit turns it off):
  The LLM plans each segment: the host character's expression plus pictures
  and key facts that pop in when the narration mentions them. The result has
  chapter labels, sound effects, a subscribe animation, a progress bar and
  audio normalized to -14 LUFS. The plan is saved as edit-plan.json; edit it
  and pass it back with --edit-plan to re-render with your changes.

  Explainer scenes interrupt the footage to make one idea obvious: the host
  alone with a punchline, a number as a filling pie or a count-up, things
  listed (stamped with a red cross when the narration denies them), two
  situations side by side, or a diagram whose arrows are drawn as each factor
  is named. They use OpenMoji icons, or doodles drawn by Imagen with
  --illustrations ai, on a paper canvas in the channel colour (--scene-color).
  Pictures that pop in are checked by Gemini (simple, on topic, no foreign
  text) when Gemini credentials are configured; an icon replaces a picture
  that fails.

  Background footage follows the narration: the plan picks a new stock scene
  every 6-8 seconds and each scene is cut into shots of --video-clip-duration
  seconds (default 5), so the picture changes every 4-6 seconds.

  The host is not on screen all the time (--host auto): it pops up from the
  bottom to introduce some items, drops in to react to a surprising line,
  points at pictures as they appear and leaves the stage to the footage the
  rest of the time; expression changes cross-fade with a small bounce. Use
  --host always to keep it on screen, or --host none to hide it. The poses
  are still pictures; --lip-sync swaps in the *_habla frames while it talks.

  --assets nutria uses the bundled otter host (resource/characters/nutria).
  --assets also takes a folder with your own material (all optional):
    personaje/feliz.png, personaje/feliz_habla.png, ...  character expressions;
        *_habla.png is the open-mouth frame used while the voice is speaking;
        saludando.png waves in the intro and outro, senalando.png points at
        the pictures that pop in
    sfx/whoosh.wav, pop.wav, tick.wav, click.wav, bloop.wav  replace the
        built-in sounds (bloop plays when the host pops up)
    suscribete.gif (or .webm/.mov/.png) and suscribete.mp3  subscribe animation
  --assets demo is a simpler sample character (a red blood cell).

Two languages at once:
  --also-in en-US makes a second video in English from the same script: the
  LLM adapts the text (title formulas and jokes included), pictures stay the
  same, and the English version gets its own edit plan and voice. Several
  languages work too (--also-in en-US,pt-BR); per-language options then use
  LANG=VALUE, e.g. --also-voice pt-BR=pt-BR-AntonioNeural-Male. Review the
  translations first with --script-only; each is saved next to the script as
  <name>.<LANG>.json and can be passed back with --also-script.
    uv run python list_video.py --subject "La electricidad explicada para nutrias" \\
      --video-language es-CO --also-in en-US --script-only --output electricidad.json
    uv run python list_video.py --script electricidad.json --also-in en-US \\
      --also-script electricidad.en-US.json --voice-name es-CO-GonzaloNeural-Male

Supported --video-source values: pexels, pixabay, coverr, openai_image and
local (local needs an image_file for every item). List videos default to 16:9,
no burned-in subtitles and no background music; pass --subtitle-enabled or
--bgm-type random to add them. All other video, voice, subtitle and music
options of cli.py are accepted (see: uv run python cli.py --help). Files are
written to storage/tasks/<task-id>/: final-1.mp4, chapters.txt and credits.txt
for the YouTube description, and edit-plan.json.
""",
        formatter_class=cli._CliHelpFormatter,
        # Unknown options are forwarded to cli.py; abbreviations could make
        # this parser claim them instead.
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group()
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

    edit_group = parser.add_argument_group("automatic editing")
    edit_group.add_argument(
        "--no-edit",
        action="store_true",
        help="plain look: only the visual and an item title per segment",
    )
    edit_group.add_argument(
        "--assets",
        default="",
        help='folder with character, sound and subscribe assets, "nutria" for the bundled otter host '
        'or "demo" for the sample character',
    )
    edit_group.add_argument(
        "--host",
        choices=["auto", "always", "none"],
        default="auto",
        help="when the character is on screen: comes and goes (auto), the whole video, or never",
    )
    edit_group.add_argument(
        "--no-scenes",
        action="store_true",
        help="no full-screen explainer scenes (statements, numbers, lists, comparisons, diagrams)",
    )
    edit_group.add_argument(
        "--illustrations",
        choices=["icons", "ai"],
        default="icons",
        help="pictures in scenes: OpenMoji icons (free) or doodle illustrations drawn by Imagen "
        "(about US$0.02 each, needs Gemini credentials)",
    )
    edit_group.add_argument(
        "--scene-color",
        type=_scene_color,
        default=None,
        help='scene canvas colour: a hex colour, "accent" or "white" (default: a light tint of --accent)',
    )
    edit_group.add_argument(
        "--no-picture-check",
        action="store_true",
        help="use the first picture found instead of letting Gemini choose a simple, relevant one",
    )
    edit_group.add_argument(
        "--lip-sync",
        action="store_true",
        help="animate the mouth with the *_habla frames (default: still poses)",
    )
    edit_group.add_argument(
        "--beats",
        choices=["web", "ai", "none"],
        default="web",
        help="pictures that pop in: searched on the web, AI-generated (openai_image), or none",
    )
    edit_group.add_argument(
        "--subscribe",
        choices=["both", "intro", "outro", "none"],
        default="both",
        help="when the subscribe animation appears",
    )
    edit_group.add_argument("--accent", type=_accent, default=None, help="accent colour, e.g. #FF4F5E")
    edit_group.add_argument("--edit-plan", default="", help="reuse an edited edit-plan.json")
    edit_group.add_argument("--no-progress-bar", action="store_true", help="hide the progress bar")
    edit_group.add_argument("--no-sfx", action="store_true", help="no sound effects")
    edit_group.add_argument(
        "--voice-style",
        default=None,
        help='delivery instructions for Gemini voices, e.g. "Narra con entusiasmo y curiosidad"',
    )
    edit_group.add_argument(
        "--also-in",
        metavar="LANGS",
        type=_languages,
        default=[],
        help="also make the video in these languages, comma-separated (e.g. en-US,pt-BR)",
    )
    edit_group.add_argument(
        "--also-voice",
        action="append",
        default=[],
        help="[LANG=]VOICE for an --also-in version (default: a free Edge voice); repeatable",
    )
    edit_group.add_argument(
        "--also-voice-style",
        action="append",
        default=[],
        help="[LANG=]Gemini delivery instructions for an --also-in version (default: none); repeatable",
    )
    edit_group.add_argument(
        "--also-script",
        action="append",
        default=[],
        help="[LANG=]reviewed translation to use instead of translating again; repeatable",
    )
    edit_group.add_argument(
        "--create-demo-assets",
        metavar="DIR",
        default=None,
        help="write a sample character (a red blood cell) into DIR and exit",
    )
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
    if args.create_demo_assets:
        from app.services.list_video_fx import create_demo_assets

        written = create_demo_assets(args.create_demo_assets)
        print(json.dumps({"assets": os.path.abspath(args.create_demo_assets), "files": written}, ensure_ascii=False))
        return 0
    if not args.subject and not args.script:
        parser.error("one of --subject or --script is required")
    if args.script_only and not args.subject:
        parser.error("--script-only requires --subject")
    unsupported = _find_unsupported_options(forwarded)
    if unsupported:
        parser.error(f"not available for list videos: {', '.join(unsupported)}")
    if args.assets == DEMO_ASSETS:
        args.assets = demo_assets_dir()
    elif args.assets and not os.path.isdir(args.assets) and built_in_assets_dir(args.assets):
        args.assets = built_in_assets_dir(args.assets)
    if args.assets and not os.path.isdir(args.assets):
        parser.error(f"--assets folder not found: {args.assets}")
    if args.edit_plan and not os.path.isfile(args.edit_plan):
        parser.error(f"--edit-plan file not found: {args.edit_plan}")
    if not args.also_in and (args.also_voice or args.also_voice_style or args.also_script):
        parser.error("--also-voice, --also-voice-style and --also-script need --also-in")
    try:
        also_voices = per_language(args.also_voice, args.also_in, "--also-voice")
        also_styles = per_language(args.also_voice_style, args.also_in, "--also-voice-style")
        also_script_paths = per_language(args.also_script, args.also_in, "--also-script")
    except ValueError as exc:
        parser.error(str(exc))

    forwarded = list(forwarded)
    if not _has_option(forwarded, "--video-aspect"):
        forwarded += ["--video-aspect", "16:9"]
    # Long-form list videos read best without burned-in subtitles or music;
    # both stay available when asked for explicitly.
    if not any(_has_option(forwarded, o) for o in ("--subtitle-enabled", "--no-subtitle-enabled")):
        forwarded += ["--no-subtitle-enabled"]
    if not any(_has_option(forwarded, o) for o in ("--bgm-type", "--bgm-file")):
        forwarded += ["--bgm-type", "none"]
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
        also_scripts = {
            language: load_script_file(path) for language, path in also_script_paths.items()
        }
    except (ValueError, OSError, ValidationError) as exc:
        logger.error(f"invalid list video input: {exc}")
        return 2

    from dataclasses import replace

    from app.config import config
    from app.services import list_video, llm, voice
    from app.services.list_video_editor import EditOptions
    from app.utils import utils

    for language in args.also_in:
        also_voices.setdefault(language, voice.default_edge_voice(language))
        if not also_voices[language]:
            logger.error(f"no default voice for {language}; pass --also-voice {language}=VOICE")
            return 2

    if args.voice_style is not None:
        config.app["gemini_tts_style"] = args.voice_style

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

    also_script_files = {
        language: os.path.abspath(path) for language, path in also_script_paths.items()
    }
    for language in args.also_in:
        if language in also_scripts:
            continue
        translated = llm.translate_list_script(script, language)
        if translated is None:
            logger.error(f"the LLM did not return a valid {language} script")
            return 1
        stem, extension = os.path.splitext(script_file)
        also_scripts[language] = translated
        also_script_files[language] = save_script_file(
            translated, f"{stem}.{language}{extension or '.json'}"
        )
        logger.info(f"{language} script saved: {also_script_files[language]}")

    if args.script_only:
        summary = {"task_id": task_id, "script_file": script_file}
        if args.also_in:
            summary["also"] = [
                {"language": language, "script_file": also_script_files[language]}
                for language in args.also_in
            ]
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    options = {}
    if args.gap is not None:
        options["gap_seconds"] = args.gap
    if args.zoom is not None:
        options["zoom"] = args.zoom
    if not args.no_edit:
        options["edit"] = EditOptions(
            assets_dir=os.path.abspath(args.assets) if args.assets else "",
            beats=args.beats,
            subscribe=args.subscribe,
            accent=args.accent or EditOptions.accent,
            progress_bar=not args.no_progress_bar,
            sound_effects=not args.no_sfx,
            plan_file=os.path.abspath(args.edit_plan) if args.edit_plan else "",
            language=params.video_language or "",
            host=args.host,
            lip_sync=args.lip_sync,
            scenes=not args.no_scenes,
            illustrations=args.illustrations,
            scene_color=args.scene_color or "",
            picture_check=not args.no_picture_check,
        )

    def render(render_task_id, render_script, render_params, render_options):
        try:
            return list_video.generate_list_video(
                render_task_id,
                render_script,
                render_params,
                number_items=not args.no_numbers,
                show_item_titles=not args.no_item_titles,
                **render_options,
            )
        except (list_video.ListVideoError, ValueError) as exc:
            logger.error(f"list video failed: task_id={render_task_id}, error={exc}")
        except Exception as exc:
            logger.exception(
                f"list video failed unexpectedly: task_id={render_task_id}, error={exc}"
            )
        return None

    result = render(task_id, script, params, options)
    if result is None:
        return 1
    summary = {"task_id": task_id, "script_file": script_file, "result": result}

    if args.also_in:
        summary["also"] = []
    for language in args.also_in:
        also_task_id = utils.get_uuid()
        also_params = params.model_copy(
            update={"voice_name": also_voices[language], "video_language": language}
        )
        also_options = dict(options)
        if "edit" in options:
            # Edit plans anchor beats to words of one language, so every
            # version gets its own plan, built on the first version's visuals
            # (same scenes, icons and pictures, and cached illustrations).
            reference = os.path.join(utils.task_dir(task_id), "edit-plan.json")
            also_options["edit"] = replace(
                options["edit"], language=language, plan_file="", reference_plan=reference
            )
        # A delivery style is written for one language; never reuse it.
        config.app["gemini_tts_style"] = also_styles.get(language, "")
        logger.info(f"rendering the {language} version with voice {also_voices[language]}")
        also_result = render(also_task_id, also_scripts[language], also_params, also_options)
        if also_result is None:
            print(json.dumps(summary, ensure_ascii=False))
            return 1
        summary["also"].append(
            {
                "language": language,
                "task_id": also_task_id,
                "script_file": also_script_files[language],
                "result": also_result,
            }
        )

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    cli._force_utf8_console()
    raise SystemExit(run())
