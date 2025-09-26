FROM python:3.10.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/home/user/nltk_data \
    HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 cmake rsync \
  && rm -rf /var/lib/apt/lists/* \
  && git lfs install --system

WORKDIR /home/user/app
RUN useradd -m -u 1000 user || true && chown -R 1000:1000 /home/user
USER 1000:1000

COPY --chown=1000:1000 requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab || true

COPY --chown=1000:1000 . .

CMD ["python", "app.py"]