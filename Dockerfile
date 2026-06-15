FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# ── System deps ──────────────────────────────────────────────────────────────
# Note: ffmpeg not available via apt on this image — installed via static binary below
RUN apt-get update && apt-get install -y \
    git wget curl bzip2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── ffmpeg static binary (apt ffmpeg not available on this base image) ────────
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    -O /tmp/ffmpeg.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp && \
    cp /tmp/ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    cp /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg*

# ── Core torch stack (confirmed: torch 2.6.0 + torchvision 0.21.0 cu124) ─────
RUN pip install \
    torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# torchvision MUST be 0.21.0 — the base image ships 0.19.1 which breaks
# torchvision::nms operator registration with transformers 5.2.0
RUN pip install torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Chatterbox TTS ────────────────────────────────────────────────────────────
# Install chatterbox without deps first, then pin its requirements manually
# to avoid conflicts with SadTalker's older pins
RUN pip install chatterbox-tts --no-deps && \
    pip install \
    transformers==5.2.0 \
    safetensors==0.5.3 \
    librosa==0.11.0 \
    pyloudnorm \
    pykakasi==2.3.0 \
    spacy-pkuseg \
    gradio==6.8.0 \
    diffusers==0.29.0 \
    resemble-perth==1.0.1 \
    omegaconf==2.3.0 \
    conformer==0.3.2 \
    s3tokenizer \
    "spacy>=3.4.0" \
    "numpy>=1.24.0,<2.0.0" \
    resampy==0.4.3

# ── SadTalker ────────────────────────────────────────────────────────────────
RUN git clone https://github.com/OpenTalker/SadTalker.git /workspace/SadTalker

WORKDIR /workspace/SadTalker

# Install SadTalker deps without overriding our torch stack
RUN pip install -r requirements.txt --no-deps

# SadTalker's requirements.txt downgrades numpy, librosa, joblib, scipy, imageio
# Restore confirmed working versions after
RUN pip install \
    "numpy>=1.24.0,<2.0.0" \
    librosa==0.11.0 \
    safetensors==0.5.3 \
    tifffile \
    PyWavelets \
    opencv-python \
    "joblib>=1.4.0" \
    "imageio==2.22.4" \
    "imageio-ffmpeg>=0.4.8"

# ── SadTalker source patches (numpy compat) ───────────────────────────────────
# Patch 1: np.float removed in numpy 1.20+ 
RUN sed -i 's/preds.astype(np.float,/preds.astype(float,/g' \
    /workspace/SadTalker/src/face3d/util/my_awing_arch.py

# Patch 2: inhomogeneous array from t[0], t[1] being arrays not scalars
RUN sed -i \
    's/trans_params = np.array(\[w0, h0, s, t\[0\], t\[1\]\])/trans_params = np.array([w0, h0, s, float(t[0]), float(t[1])])/' \
    /workspace/SadTalker/src/face3d/util/preprocess.py

# ── Download SadTalker model weights ─────────────────────────────────────────
RUN mkdir -p /workspace/SadTalker/checkpoints && \
    wget -q -O /workspace/SadTalker/checkpoints/mapping_00109-model.pth.tar \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar" && \
    wget -q -O /workspace/SadTalker/checkpoints/mapping_00229-model.pth.tar \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar" && \
    wget -q -O /workspace/SadTalker/checkpoints/SadTalker_V0.0.2_256.safetensors \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors" && \
    wget -q -O /workspace/SadTalker/checkpoints/SadTalker_V0.0.2_512.safetensors \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors"

# Download GFPGAN/facexlib weights that SadTalker needs
RUN mkdir -p /workspace/SadTalker/gfpgan/weights && \
    wget -q -O /workspace/SadTalker/gfpgan/weights/alignment_WFLW_4HG.pth \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth" && \
    wget -q -O /workspace/SadTalker/gfpgan/weights/detection_Resnet50_Final.pth \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -q -O /workspace/SadTalker/gfpgan/weights/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" && \
    wget -q -O /workspace/SadTalker/gfpgan/weights/parsing_parsenet.pth \
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

# ── GFPGAN / basicsr ─────────────────────────────────────────────────────────
WORKDIR /workspace

RUN pip install basicsr --upgrade && \
    pip install gfpgan facexlib

# Patch basicsr: torchvision.transforms.functional_tensor removed in torchvision 0.21+
RUN sed -i \
    's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py

# ── Final re-pins ─────────────────────────────────────────────────────────────
# basicsr --upgrade pulls numpy 2.x; re-pin to <2 which is required by scikit-image 0.19.3
RUN pip install "numpy>=1.24.0,<2.0.0" librosa==0.11.0 safetensors==0.5.3

# Re-pin torchvision in case anything above downgraded it
RUN pip install torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── RunPod worker deps ────────────────────────────────────────────────────────
RUN pip install runpod boto3 Pillow accelerate requests

# ── Worker code ──────────────────────────────────────────────────────────────
WORKDIR /workspace/worker
COPY handler.py voice.py video.py enhance.py storage.py ./

ENV SADTALKER_DIR=/workspace/SadTalker
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]