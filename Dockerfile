FROM dronabajaj/ai-cloning-base:latest

WORKDIR /workspace/worker
COPY handler.py voice.py video.py enhance.py storage.py ./

ENV SADTALKER_DIR=/workspace/SadTalker
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]