"""
Stage 1: Voice cloning via Chatterbox (Resemble AI)
Fallback: F5-TTS
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
    try:
        _clone_with_chatterbox(reference_audio, script, output_path, exaggeration, cfg_weight)
    except ImportError as e:
        print(f"[voice] Chatterbox ImportError: {e} — falling back to F5-TTS")
        _clone_with_f5tts(reference_audio, script, output_path)
    except Exception as e:
        print(f"[voice] Chatterbox failed: {e} — falling back to F5-TTS")
        _clone_with_f5tts(reference_audio, script, output_path)


def _clone_with_chatterbox(reference_audio, script, output_path, exaggeration, cfg_weight):
    from chatterbox.tts import ChatterboxTTS
    import torchaudio

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
    torchaudio.save(output_path, wav, model.sr)
    print(f"[voice/chatterbox] Saved → {output_path}")


def _clone_with_f5tts(reference_audio, script, output_path):
    import subprocess
    cmd = [
        "f5-tts_infer-cli",
        "--model", "F5TTS_v1_Base",
        "--ref_audio", reference_audio,
        "--ref_text", "",
        "--gen_text", script,
        "--output_dir", str(Path(output_path).parent),
        "--output_file", Path(output_path).name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"F5-TTS failed: {result.stderr}")
    print(f"[voice/f5tts] Saved → {output_path}")