"""
RunPod serverless handler for AI cloning pipeline.
Pipeline: Chatterbox (voice) → SadTalker (video) → GFPGAN (enhance) → R2 upload
"""

import runpod
import os
import uuid
import base64
import tempfile
import subprocess
import shutil
from pathlib import Path

from voice import clone_voice
from video import synthesize_video
from enhance import enhance_video
from storage import upload_to_r2


def handler(job):
    job_input = job["input"]

    # ── Validate inputs ──────────────────────────────────────────────────────
    required = ["audio_b64", "photo_b64", "script"]
    for field in required:
        if field not in job_input:
            return {"error": f"Missing required field: {field}"}

    job_id = job.get("id", str(uuid.uuid4()))
    skip_enhance = job_input.get("skip_enhance", False)  # set True for faster v1 testing

    workdir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
    print(f"[{job_id}] Working dir: {workdir}")

    try:
        # ── Decode inputs ────────────────────────────────────────────────────
        audio_path = workdir / "reference.wav"
        photo_path = workdir / "reference.jpg"

        audio_path.write_bytes(base64.b64decode(job_input["audio_b64"]))
        photo_path.write_bytes(base64.b64decode(job_input["photo_b64"]))

        script = job_input["script"]
        emotion = job_input.get("emotion", "neutral")   # neutral | happy | sad | angry

        print(f"[{job_id}] Inputs decoded. Script length: {len(script)} chars")

        # ── Stage 1: Voice cloning ───────────────────────────────────────────
        print(f"[{job_id}] Stage 1: Cloning voice with Chatterbox...")
        cloned_wav = workdir / "cloned_speech.wav"
        clone_voice(
            reference_audio=str(audio_path),
            script=script,
            output_path=str(cloned_wav),
            emotion=emotion,
        )
        print(f"[{job_id}] Stage 1 done → {cloned_wav}")

        # ── Stage 2: Video synthesis ─────────────────────────────────────────
        print(f"[{job_id}] Stage 2: Synthesizing video with SadTalker...")
        raw_video = workdir / "raw_video.mp4"
        synthesize_video(
            photo_path=str(photo_path),
            audio_path=str(cloned_wav),
            output_path=str(raw_video),
        )
        print(f"[{job_id}] Stage 2 done → {raw_video}")

        # ── Stage 3: Enhancement (optional) ─────────────────────────────────
        if skip_enhance:
            final_video = raw_video
            print(f"[{job_id}] Stage 3: Skipped (skip_enhance=True)")
        else:
            print(f"[{job_id}] Stage 3: Enhancing with GFPGAN...")
            enhanced_video = workdir / "enhanced_video.mp4"
            enhance_video(
                input_path=str(raw_video),
                output_path=str(enhanced_video),
            )
            final_video = enhanced_video
            print(f"[{job_id}] Stage 3 done → {final_video}")

        # ── Upload to R2 ─────────────────────────────────────────────────────
        print(f"[{job_id}] Uploading to R2...")
        r2_key = f"outputs/{job_id}/output.mp4"
        public_url = upload_to_r2(str(final_video), r2_key)
        print(f"[{job_id}] Upload done → {public_url}")

        return {
            "status": "success",
            "job_id": job_id,
            "video_url": public_url,
            "stages": {
                "voice": "chatterbox",
                "video": "sadtalker",
                "enhance": "skipped" if skip_enhance else "gfpgan",
            },
        }

    except Exception as e:
        import traceback
        print(f"[{job_id}] ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e), "job_id": job_id}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"[{job_id}] Cleaned up workdir")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
