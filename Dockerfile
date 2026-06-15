FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg git wget curl bzip2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Chatterbox first (most likely to have conflicts) ─────────────────────────
RUN pip install chatterbox-tts

# ── SadTalker ────────────────────────────────────────────────────────────────
RUN git clone https://github.com/OpenTalker/SadTalker.git /workspace/SadTalker
WORKDIR /workspace/SadTalker
RUN pip install -r requirements.txt --no-deps

# Download SadTalker weights
RUN mkdir -p /workspace/SadTalker/checkpoints && \
    wget -q -O /workspace/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" && \
    wget -q -O /workspace/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors"

# Download 3DMM shape model
RUN mkdir -p /workspace/SadTalker/gfpgan/weights && \
    curl -L "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sfd/shape_predictor_68_face_landmarks.dat" \
    -o /workspace/SadTalker/gfpgan/weights/shape_predictor_68_face_landmarks.dat

# ── GFPGAN ───────────────────────────────────────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/TencentARC/GFPGAN.git /workspace/GFPGAN
WORKDIR /workspace/GFPGAN
RUN pip install basicsr facexlib gfpgan

RUN mkdir -p /workspace/models && \
    wget -q -O /workspace/models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

# ── Python deps ──────────────────────────────────────────────────────────────
RUN pip install runpod boto3 Pillow numpy torchaudio

# ── Worker code ──────────────────────────────────────────────────────────────
WORKDIR /workspace/worker
COPY handler.py voice.py video.py enhance.py storage.py ./

ENV SADTALKER_DIR=/workspace/SadTalker
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]