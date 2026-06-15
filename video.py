"""
Stage 2: Video synthesis via SadTalker
Repo: https://github.com/OpenTalker/SadTalker

Expected install path: /workspace/SadTalker
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

SADTALKER_DIR = Path(os.environ.get("SADTALKER_DIR", "/workspace/SadTalker"))


def synthesize_video(
    photo_path: str,
    audio_path: str,
    output_path: str,
    still_mode: bool = False,
    preprocess: str = "crop",  # 'crop' confirmed working; 'full' causes numpy array errors
    size: int = 256,
    pose_style: int = 0,
):
    """
    Synthesize a lip-synced talking-head video from a photo + audio.

    Args:
        photo_path: Reference face image (JPG/PNG, front-facing recommended)
        audio_path: Cloned speech WAV from Stage 1
        output_path: Final output MP4 path
        still_mode: Minimal head motion
        preprocess: 'crop' (confirmed working) or 'full' (broken with numpy)
        size: 256 for speed, 512 for quality
        pose_style: Head pose variation seed (0 = neutral)
    """
    result_dir = Path(output_path).parent / "sadtalker_out"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SADTALKER_DIR / "inference.py"),
        "--driven_audio", audio_path,
        "--source_image", photo_path,
        "--result_dir", str(result_dir),
        "--preprocess", preprocess,
        "--size", str(size),
        "--pose_style", str(pose_style),
        "--enhancer", "none",
    ]

    if still_mode:
        cmd.append("--still")

    print(f"[video/sadtalker] Running inference...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SADTALKER_DIR))

    if result.returncode != 0:
        print(f"[video/sadtalker] STDOUT:\n{result.stdout}")
        print(f"[video/sadtalker] STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"SadTalker failed (exit {result.returncode})\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    # SadTalker outputs:
    # - result_dir/TIMESTAMP.mp4  (final merged video)
    # - result_dir/TIMESTAMP/input##audio.mp4  (intermediate)
    # We want the top-level timestamped mp4
    mp4_files = list(result_dir.glob("*.mp4"))
    if not mp4_files:
        # Fall back to subdirectory
        mp4_files = list(result_dir.glob("*/*.mp4"))
    if not mp4_files:
        print(f"[video/sadtalker] Dir contents: {list(result_dir.rglob('*'))}")
        raise RuntimeError("SadTalker produced no MP4 output")

    # Prefer top-level mp4 (the final one with audio merged)
    top_level = [f for f in mp4_files if f.parent == result_dir]
    chosen = top_level[0] if top_level else mp4_files[0]
    shutil.move(str(chosen), output_path)
    print(f"[video/sadtalker] Saved → {output_path}")