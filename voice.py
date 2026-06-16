"""
Stage 1: Voice cloning
- English: Chatterbox (Resemble AI) - best quality for English
- Other languages: XTTS v2 (Coqui) - multilingual support
"""

import torch
from pathlib import Path


def clone_voice(
    reference_audio: str,
    script: str,
    output_path: str,
    emotion: str = "neutral",
    language: str = "en",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
):
    if language == "en":
        _clone_chatterbox(reference_audio, script, output_path, emotion, exaggeration, cfg_weight)
    else:
        _clone_xtts(reference_audio, script, output_path, language)


def _clone_chatterbox(reference_audio, script, output_path, emotion, exaggeration, cfg_weight):
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


def _clone_xtts(reference_audio, script, output_path, language):
    from TTS.api import TTS
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[voice/xtts] Loading XTTS v2 on {device} for language: {language}...")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"[voice/xtts] Generating speech ({len(script)} chars)...")
    tts.tts_to_file(
        text=script,
        speaker_wav=reference_audio,
        language=language,
        file_path=output_path,
    )
    print(f"[voice/xtts] Saved → {output_path}")