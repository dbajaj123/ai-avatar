"""
RunPod serverless handler for AI cloning pipeline.
Pipeline: Chatterbox (voice) → SadTalker (video) → GFPGAN (enhance) → uguu.se (upload)
Returns public video URL.
"""

import runpod
import os
import uuid
import base64
import tempfile
import shutil
from pathlib import Path

from voice import clone_voice
from video import synthesize_video
from enhance import enhance_video
from storage import upload_to_uguu


def handler(job):
    job_input = job["input"]

    required = ["audio_b64", "photo_b64", "script"]
    for field in required:
        if field not in job_input:
            return {"error": f"Missing required field: {field}"}

    job_id = job.get("id", str(uuid.uuid4()))
    skip_enhance = job_input.get("skip_enhance", False)

    workdir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
    print(f"[{job_id}] Working dir: {workdir}")

    try:
        # Decode inputs
        audio_path = workdir / "reference.wav"
        photo_path = workdir / "reference.jpg"
        audio_path.write_bytes(base64.b64decode(job_input["audio_b64"]))
        photo_path.write_bytes(base64.b64decode(job_input["photo_b64"]))

        script  = job_input["script"]
        emotion = job_input.get("emotion", "neutral")

        # Stage 1: Voice cloning
        print(f"[{job_id}] Stage 1: Cloning voice...")
        cloned_wav = workdir / "cloned_speech.wav"
        clone_voice(
            reference_audio=str(audio_path),
            script=script,
            output_path=str(cloned_wav),
            emotion=emotion,
        )

        # Stage 2: Video synthesis
        print(f"[{job_id}] Stage 2: Synthesizing video...")
        raw_video = workdir / "raw_video.mp4"
        synthesize_video(
            photo_path=str(photo_path),
            audio_path=str(cloned_wav),
            output_path=str(raw_video),
        )

        # Stage 3: Enhancement (optional)
        if skip_enhance:
            final_video = raw_video
            print(f"[{job_id}] Stage 3: Skipped")
        else:
            print(f"[{job_id}] Stage 3: Enhancing...")
            enhanced_video = workdir / "enhanced_video.mp4"
            enhance_video(
                input_path=str(raw_video),
                output_path=str(enhanced_video),
            )
            final_video = enhanced_video

        # Stage 4: Upload to uguu.se
        print(f"[{job_id}] Stage 4: Uploading to uguu.se...")
        video_url = upload_to_uguu(str(final_video))

        return {
            "status": "success",
            "job_id": job_id,
            "video_url": video_url,
            "stages": {
                "voice":   "chatterbox",
                "video":   "sadtalker",
                "enhance": "skipped" if skip_enhance else "gfpgan",
                "storage": "uguu.se",
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