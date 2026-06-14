#!/usr/bin/env python3
"""
Local test: run the full pipeline with one photo + audio + script.

Usage:
    python test_local.py \
        --photo path/to/face.jpg \
        --audio path/to/voice.wav \
        --script "Hello, this is a test of the AI cloning pipeline." \
        --skip-enhance          # optional: skip GFPGAN for faster testing

Requires:
    - All models installed (run inside Docker or on a GPU machine)
    - .env file with R2 credentials (or export env vars)
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def encode_file(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Test AI cloning pipeline locally")
    parser.add_argument("--photo", required=True, help="Path to reference face image (JPG/PNG)")
    parser.add_argument("--audio", required=True, help="Path to reference audio (WAV/MP3, ≥6s)")
    parser.add_argument("--script", required=True, help="Text to synthesize")
    parser.add_argument("--emotion", default="neutral", choices=["neutral", "happy", "sad", "angry"])
    parser.add_argument("--skip-enhance", action="store_true", help="Skip GFPGAN (faster)")
    parser.add_argument("--still", action="store_true", help="SadTalker still mode (less head movement)")
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.photo, "photo"), (args.audio, "audio")]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    print("── Encoding inputs ───────────────────────────────────────────")
    print(f"  Photo : {args.photo} ({Path(args.photo).stat().st_size // 1024} KB)")
    print(f"  Audio : {args.audio} ({Path(args.audio).stat().st_size // 1024} KB)")
    print(f"  Script: {args.script[:80]}{'...' if len(args.script) > 80 else ''}")
    print()

    job = {
        "id": "local_test_001",
        "input": {
            "photo_b64": encode_file(args.photo),
            "audio_b64": encode_file(args.audio),
            "script": args.script,
            "emotion": args.emotion,
            "skip_enhance": args.skip_enhance,
            "still_mode": args.still,
        }
    }

    print("── Running handler ───────────────────────────────────────────")
    # Import handler from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from handler import handler

    result = handler(job)

    print()
    print("── Result ────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2))

    if result.get("status") == "success":
        print()
        print(f"✓ Video URL: {result['video_url']}")
        sys.exit(0)
    else:
        print()
        print(f"✗ Error: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
