"""
Voice lab: narrate the same text with several voices and compare them.

  python voice_lab.py                      # the Spanish young male shortlist
  python voice_lab.py --voices gemini:Puck-Upbeat,gcloud:es-US-Chirp3-HD-Charon --style divulgador
  python voice_lab.py --list               # the voice ids you can use

Gemini voices ("gemini:*") use Gemini TTS with gemini_api_key or Vertex AI;
"gcloud:*" voices use Google Cloud Text-to-Speech (Chirp 3 HD) with the same
Application Default Credentials; "elevenlabs:<voice id>" uses ElevenLabs; plain
names such as es-CO-GonzaloNeural-Male are the free Edge voices.
The files land in storage/voice-lab/ with a summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice_lab.py",
        description="Narrate the same text with several voices and compare them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--voices", default="", help="comma-separated voice ids (default: Spanish young male shortlist)")
    parser.add_argument("--text", default="", help="text to narrate (default: a short science explanation)")
    parser.add_argument("--text-file", default="", help="read the text from a file")
    parser.add_argument(
        "--style", default="divulgador",
        help="Gemini delivery: divulgador, entusiasta, profe, calmado, narrador or your own directions",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="speaking rate (1.0 normal, 1.1 a bit faster)")
    parser.add_argument("--out", default="", help="output folder (default: storage/voice-lab)")
    parser.add_argument("--list", action="store_true", help="print the voice ids and presets and exit")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from app.services import voice, voice_google
    from app.utils import utils

    if args.list:
        print(json.dumps({
            "shortlist": list(voice_google.SPANISH_MALE_SHORTLIST),
            "gemini": voice.get_gemini_voices(),
            "gcloud": voice_google.get_gcloud_voices(),
            "presets": voice_google.VOICE_STYLE_PRESETS,
        }, ensure_ascii=False, indent=2))
        return 0
    text = args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as fp:
            text = fp.read()
    text = " ".join((text or voice_google.LAB_TEXT).split())
    voices = [v.strip() for v in args.voices.split(",") if v.strip()] or list(voice_google.SPANISH_MALE_SHORTLIST)
    out = args.out or utils.storage_dir("voice-lab", create=True)
    results = voice_google.audition(voices, text, out, rate=args.rate, style=args.style)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump({"text": text, "style": args.style, "rate": args.rate, "results": results}, fp, ensure_ascii=False, indent=2)
    width = max(len(r["voice"]) for r in results)
    for result in results:
        status = result["file"] if result["ok"] else f"FAILED: {result['error']}"
        print(f"{result['voice']:<{width}}  {status}")
    minute = len(text) / 15.0 / 60.0
    print(
        f"\n{len(text)} characters (~{minute:.1f} min). Rough cost per 10-minute video: "
        f"Gemini Flash TTS ${voice_google.estimate_cost(9000, 'gemini-flash')}, "
        f"Gemini Pro TTS ${voice_google.estimate_cost(9000, 'gemini-pro')}, "
        f"Chirp 3 HD ${voice_google.estimate_cost(9000, 'gcloud-chirp')}, "
        f"ElevenLabs ~${voice_google.estimate_cost(9000, 'elevenlabs')} (9,000 credits), Edge free."
    )
    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(run())
