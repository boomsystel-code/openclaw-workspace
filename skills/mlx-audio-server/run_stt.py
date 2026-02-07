#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=2.17.0",
# ]
# ///

import argparse
import os
import sys
from openai import OpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8899/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "mlx-community/glm-asr-nano-2512-8bit")


def cli():
    parser = argparse.ArgumentParser(
        prog="OpenAI Transcription Client",
        description="Transcribe audio with OpenAI Compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--base_url",
        default=BASE_URL,
        help="Base URL of the MLX Audio server",
    )
    parser.add_argument(
        "--api_key",
        default=API_KEY,
        help="API Key (dummy for local server)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Model to use for transcription",
    )

    return parser.parse_args()


def main():
    args = cli()
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)

    print(f"Connecting to {args.base_url}...")
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print(f"Transcribing with {args.model}: {args.audio_file}...")

    try:
        with open(args.audio_file, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model=args.model,
                file=audio,
                response_format="json",  # or verbose_json, text, srt, vtt
            )

        if args.verbose:
            print("\nFull Response:")
            print(transcript)
            print("\nTranscription Text:")

        # Depending on response_format, transcript might be an object or raw text
        if hasattr(transcript, 'text'):
            print(transcript.text)
        else:
            print(transcript)

    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
