"""
Stage 1: Voice cloning via XTTS v2 (Coqui)
Supports all languages: en, hi, fr, es, de, zh, ja, ko, etc.
"""

import torch
from pathlib import Path

# Patch torch.load to use weights_only=False for torch 2.6+ compatibility
# XTTS model weights require this since they use custom classes
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def clone_voice(
    reference_audio: str,
    script: str,
    output_path: str,
    language: str = "en",
    emotion: str = "neutral",  # kept for API compatibility, XTTS doesn't use emotion
    **kwargs,
):
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[voice/xtts] Loading XTTS v2 on {device}, language={language}...")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"[voice/xtts] Generating speech ({len(script)} chars)...")
    tts.tts_to_file(
        text=script,
        speaker_wav=reference_audio,
        language=language,
        file_path=output_path,
    )
    print(f"[voice/xtts] Saved → {output_path}")