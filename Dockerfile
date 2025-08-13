# Use slim to reduce image size (keeps Debian trixie)
FROM python:3.10-slim

# ---- Base settings (quieter, faster, safer) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/home/user/nltk_data

# System deps (no recommends = leaner). Note: libgl1 replaces removed libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 cmake rsync \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install --system

# ---- App dir & user permissions ----
WORKDIR /home/user/app
# (Optional) ensure /home/user exists and is writable on some runners
RUN useradd -m -u 1000 user || true && chown -R 1000:1000 /home/user
USER 1000:1000

# ---- Dependency layer (max cache reuse) ----
# Copy just requirements first so pip install layer can cache
COPY --chown=1000:1000 requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip wheel \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data at build time (avoids runtime writes)
RUN python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab || \
    python -m nltk.downloader -d "$NLTK_DATA" punkt

# ---- App code ----
# Now copy the rest of your app
COPY --chown=1000:1000 . .

# (If you have a Gradio app, HF will expose port; no need for EXPOSE)
CMD ["python", "app.py"]