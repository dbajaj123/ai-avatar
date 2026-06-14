# RunPod Worker — AI Cloning Pipeline

**Stack:** Chatterbox → SadTalker → GFPGAN → Cloudflare R2

## Files

```
handler.py      — RunPod entrypoint
voice.py        — Stage 1: voice cloning (Chatterbox / F5-TTS fallback)
video.py        — Stage 2: video synthesis (SadTalker)
enhance.py      — Stage 3: face enhancement (GFPGAN, optional)
storage.py      — R2 upload
Dockerfile      — Full image build
test_local.py   — Run pipeline locally (no RunPod needed)
test_remote.py  — Submit job to deployed RunPod endpoint
.env.example    — Copy to .env and fill in
```

## Setup

```bash
cp .env.example .env
# fill in .env
```

## Testing

### Option A — Local (fastest for dev, needs GPU + models installed)

```bash
python test_local.py \
  --photo  assets/face.jpg \
  --audio  assets/voice.wav \
  --script "Hello, this is a test of the AI cloning pipeline." \
  --skip-enhance
```

Remove `--skip-enhance` when you want GFPGAN in the loop.

### Option B — Remote (against deployed RunPod endpoint)

```bash
python test_remote.py \
  --photo       assets/face.jpg \
  --audio       assets/voice.wav \
  --script      "Hello, this is a test." \
  --endpoint-id YOUR_RUNPOD_ENDPOINT_ID \
  --skip-enhance
```

## Docker build + push

```bash
docker build -t your-dockerhub-user/ai-cloning-worker:latest .
docker push your-dockerhub-user/ai-cloning-worker:latest
```

Then in RunPod console:
- New Serverless Endpoint → Docker image: `your-dockerhub-user/ai-cloning-worker:latest`
- GPU: A100 80GB
- Min workers: 0 (scale to zero when idle)
- Max workers: 3

## Input schema

```json
{
  "audio_b64":    "<base64-encoded WAV/MP3>",
  "photo_b64":    "<base64-encoded JPG/PNG>",
  "script":       "Text to speak in the cloned voice.",
  "emotion":      "neutral",
  "skip_enhance": false,
  "still_mode":   false
}
```

## Output schema

```json
{
  "status":    "success",
  "job_id":    "abc123",
  "video_url": "https://assets.yourdomain.com/outputs/abc123/output.mp4",
  "stages": {
    "voice":   "chatterbox",
    "video":   "sadtalker",
    "enhance": "gfpgan"
  }
}
```

## Upgrade path

- Voice: Chatterbox → already SOTA, no change needed
- Video: swap `video.py` to call Hallo when it stabilises (same interface)
- Enhancement: increase GFPGAN `upscale=2` for higher output resolution
# ai-avatar
