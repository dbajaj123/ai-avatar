"""
Stage 3: Face enhancement via GFPGAN (optional)
Repo: https://github.com/TencentARC/GFPGAN

Runs frame extraction → batch GFPGAN → reassemble video.
On A100 with batching: ~30–60s for a 30s clip.
"""

import subprocess
import sys
import shutil
from pathlib import Path

GFPGAN_DIR = Path("/workspace/GFPGAN")
GFPGAN_MODEL = Path("/workspace/models/GFPGANv1.4.pth")


def enhance_video(
    input_path: str,
    output_path: str,
    upscale: int = 1,       # 1 = same resolution (faster), 2 = 2× upscale
    only_center_face: bool = True,  # True = only enhance the main talking face
):
    """
    Enhance a raw SadTalker video using GFPGAN.
    Extracts frames, runs GFPGAN in batch, reassembles with original audio.
    
    Args:
        input_path: Raw MP4 from SadTalker
        output_path: Enhanced MP4
        upscale: Output resolution multiplier
        only_center_face: Restrict enhancement to the center/largest face
    """
    import tempfile
    import os

    workdir = Path(tempfile.mkdtemp(prefix="gfpgan_"))
    frames_in  = workdir / "frames_in"
    frames_out = workdir / "frames_out"
    frames_in.mkdir()
    frames_out.mkdir()

    try:
        # 1. Extract frames from raw video
        print(f"[enhance/gfpgan] Extracting frames from {input_path}...")
        _run(["ffmpeg", "-i", input_path, "-q:v", "2", str(frames_in / "frame_%06d.jpg"), "-loglevel", "error"])

        frame_count = len(list(frames_in.glob("*.jpg")))
        print(f"[enhance/gfpgan] {frame_count} frames extracted")

        # 2. Run GFPGAN on all frames
        print(f"[enhance/gfpgan] Running GFPGAN (upscale={upscale})...")
        gfpgan_cmd = [
            sys.executable,
            str(GFPGAN_DIR / "inference_gfpgan.py"),
            "-i", str(frames_in),
            "-o", str(frames_out),
            "-v", "1.4",
            "-s", str(upscale),
            "--model_path", str(GFPGAN_MODEL),
            "--bg_upsampler", "none",
        ]
        if only_center_face:
            gfpgan_cmd.append("--only_center_face")

        _run(gfpgan_cmd, cwd=str(GFPGAN_DIR))

        # GFPGAN writes to frames_out/restored_imgs/
        restored_dir = frames_out / "restored_imgs"
        if not restored_dir.exists():
            raise RuntimeError(f"GFPGAN output dir not found: {restored_dir}")

        # 3. Get original video FPS
        fps = _get_fps(input_path)
        print(f"[enhance/gfpgan] Reassembling at {fps} fps...")

        # 4. Reassemble frames → video (no audio yet)
        silent_video = workdir / "enhanced_silent.mp4"
        _run([
            "ffmpeg",
            "-framerate", str(fps),
            "-i", str(restored_dir / "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(silent_video),
            "-loglevel", "error",
        ])

        # 5. Mux original audio back in
        _run([
            "ffmpeg",
            "-i", str(silent_video),
            "-i", input_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
            "-loglevel", "error",
        ])

        print(f"[enhance/gfpgan] Saved → {output_path}")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}\n{result.stderr}")
    return result


def _get_fps(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    num, den = result.stdout.strip().split("/")
    return round(int(num) / int(den), 3)
