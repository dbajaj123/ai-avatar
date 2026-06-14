# Base: RunPod PyTorch image with CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── SadTalker ────────────────────────────────────────────────────────────────
RUN git clone https://github.com/OpenTalker/SadTalker.git /workspace/SadTalker
WORKDIR /workspace/SadTalker
RUN pip install -r requirements.txt

# Download SadTalker weights
RUN mkdir -p /workspace/SadTalker/checkpoints && \
    wget -q -O /workspace/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" && \
    wget -q -O /workspace/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors"

# Download 3DMM shape model
RUN mkdir -p /workspace/SadTalker/gfpgan/weights && \
    wget -q -O /workspace/SadTalker/gfpgan/weights/shape_predictor_68_face_landmarks.dat.bz2 \
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" && \
    bunzip2 /workspace/SadTalker/gfpgan/weights/shape_predictor_68_face_landmarks.dat.bz2

# ── GFPGAN ───────────────────────────────────────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/TencentARC/GFPGAN.git /workspace/GFPGAN
WORKDIR /workspace/GFPGAN
RUN pip install -r requirements.txt && pip install basicsr facexlib

RUN mkdir -p /workspace/models && \
    wget -q -O /workspace/models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

# ── Chatterbox ───────────────────────────────────────────────────────────────
WORKDIR /workspace
RUN pip install chatterbox-tts

# ── F5-TTS (fallback) ────────────────────────────────────────────────────────
RUN pip install f5-tts

# ── Python deps ──────────────────────────────────────────────────────────────
RUN pip install \
    runpod \
    boto3 \
    torchaudio \
    Pillow \
    numpy

# ── Worker code ──────────────────────────────────────────────────────────────
WORKDIR /workspace/worker
COPY handler.py voice.py video.py enhance.py storage.py ./

ENV SADTALKER_DIR=/workspace/SadTalker
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
