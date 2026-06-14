#!/usr/bin/env python3
"""
Remote test: submit one job to a deployed RunPod serverless endpoint and poll for result.

Usage:
    python test_remote.py \
        --photo path/to/face.jpg \
        --audio path/to/voice.wav \
        --script "Hello, this is a test." \
        --endpoint-id YOUR_RUNPOD_ENDPOINT_ID \
        --skip-enhance

Env vars:
    RUNPOD_API_KEY — your RunPod API key
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
BASE_URL = "https://api.runpod.io/v2"


def encode_file(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def submit_job(endpoint_id: str, payload: dict) -> str:
    url = f"{BASE_URL}/{endpoint_id}/run"
    resp = requests.post(
        url,
        json={"input": payload},
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"  Job submitted → {job_id}")
    return job_id


def poll_job(endpoint_id: str, job_id: str, timeout: int = 600, interval: int = 5) -> dict:
    url = f"{BASE_URL}/{endpoint_id}/status/{job_id}"
    start = time.time()
    last_status = None

    while time.time() - start < timeout:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        if status != last_status:
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:3d}s] Status: {status}")
            last_status = status

        if status == "COMPLETED":
            return data.get("output", {})
        elif status == "FAILED":
            raise RuntimeError(f"Job failed: {data.get('error', 'unknown error')}")

        time.sleep(interval)

    raise TimeoutError(f"Job did not complete within {timeout}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--skip-enhance", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="Poll timeout in seconds")
    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)

    print("── Encoding inputs ───────────────────────────────────────────")
    payload = {
        "photo_b64": encode_file(args.photo),
        "audio_b64": encode_file(args.audio),
        "script": args.script,
        "emotion": args.emotion,
        "skip_enhance": args.skip_enhance,
    }
    print(f"  Photo : {args.photo}")
    print(f"  Audio : {args.audio}")
    print(f"  Script: {args.script[:80]}")
    print()

    print("── Submitting job ────────────────────────────────────────────")
    job_id = submit_job(args.endpoint_id, payload)

    print()
    print("── Polling for result ────────────────────────────────────────")
    result = poll_job(args.endpoint_id, job_id, timeout=args.timeout)

    print()
    print("── Result ────────────────────────────────────────────────────")
    print(json.dumps(result, indent=2))

    if result.get("status") == "success":
        print(f"\n✓ Video URL: {result['video_url']}")
    else:
        print(f"\n✗ Error: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
