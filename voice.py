"""
Stage 1: Voice cloning via Chatterbox (Resemble AI)
Fallback: F5-TTS if Chatterbox unavailable

Chatterbox: https://github.com/resemble-ai/chatterbox
Install: pip install chatterbox-tts
"""

import torch
from pathlib import Path


def clone_voice(
    reference_audio: str,
    script: str,
    output_path: str,
    emotion: str = "neutral",
    exaggeration: float = 0.5,   # 0.0–1.0, emotion intensity
    cfg_weight: float = 0.5,     # classifier-free guidance weight
):
    """
    Clone voice from reference_audio and synthesize script.
    
    Args:
        reference_audio: Path to reference WAV/MP3 (≥6 seconds recommended)
        script: Text to synthesize in the cloned voice
        output_path: Where to write the output WAV
        emotion: One of neutral | happy | sad | angry
        exaggeration: How strongly to apply the emotion (0.0 = flat, 1.0 = very expressive)
        cfg_weight: Higher = more faithful to reference timbre, lower = more natural
    """
    try:
        _clone_with_chatterbox(reference_audio, script, output_path, exaggeration, cfg_weight)
    except ImportError:
        print("[voice] Chatterbox not found, falling back to F5-TTS")
        _clone_with_f5tts(reference_audio, script, output_path)


def _clone_with_chatterbox(reference_audio, script, output_path, exaggeration, cfg_weight):
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[voice/chatterbox] Loading model on {device}...")

    model = ChatterboxTTS.from_pretrained(device=device)

    print(f"[voice/chatterbox] Generating speech ({len(script)} chars)...")
    wav = model.generate(
        script,
        audio_prompt_path=reference_audio,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    import torchaudio
    torchaudio.save(output_path, wav, model.sr)
    print(f"[voice/chatterbox] Saved → {output_path}")


def _clone_with_f5tts(reference_audio, script, output_path):
    """F5-TTS fallback via CLI"""
    import subprocess
    cmd = [
        "f5-tts_infer-cli",
        "--model", "F5TTS_v1_Base",
        "--ref_audio", reference_audio,
        "--ref_text", "",          # auto-transcribe reference
        "--gen_text", script,
        "--output_dir", str(Path(output_path).parent),
        "--output_file", Path(output_path).name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"F5-TTS failed: {result.stderr}")
    print(f"[voice/f5tts] Saved → {output_path}")
