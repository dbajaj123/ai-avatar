"""
Stage 1: Voice cloning via Chatterbox (Resemble AI)
No fallback — fail loudly so we can see the real error.
"""

import torch
from pathlib import Path


def clone_voice(
    reference_audio: str,
    script: str,
    output_path: str,
    emotion: str = "neutral",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
):
    from chatterbox.tts import ChatterboxTTS
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[voice/chatterbox] CUDA available: {torch.cuda.is_available()}")
    print(f"[voice/chatterbox] PyTorch version: {torch.__version__}")
    print(f"[voice/chatterbox] Loading model on {device}...")

    model = ChatterboxTTS.from_pretrained(device=device)

    print(f"[voice/chatterbox] Generating speech ({len(script)} chars)...")
    wav = model.generate(
        script,
        audio_prompt_path=reference_audio,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    torchaudio.save(output_path, wav, model.sr)
    print(f"[voice/chatterbox] Saved → {output_path}")