# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm

# ---- Environment & runtime tuning ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Create a non-root user WITH a real home so libs can cache safely
ARG APP_USER=app
ARG APP_GROUP=app
ARG APP_UID=10001
RUN addgroup --system ${APP_GROUP} \
 && adduser --system --uid ${APP_UID} --ingroup ${APP_GROUP} --home /home/${APP_USER} ${APP_USER}

# Cache locations
ENV HOME=/home/${APP_USER} \
    XDG_CACHE_HOME=/home/${APP_USER}/.cache \
    TORCH_HOME=/home/${APP_USER}/.cache/torch
RUN mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME"

WORKDIR /code

# ---- Copy only requirements first (better caching) ----
COPY requirements.txt .

# ---- Build & runtime system deps ----
# - build-essential, g++, cmake: required to build insightface C/C++ extension
# - libgl1, libglib2.0-0, libsm6, libxext6, ffmpeg: common OpenCV/mediapipe runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ cmake \
      libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Optional: remove heavy build tools after wheels are built to slim the image
RUN apt-get purge -y --auto-remove build-essential g++ cmake || true && \
    rm -rf /var/lib/apt/lists/*

# ---- App code ----
COPY . .

# Permissions for non-root user
RUN chown -R ${APP_USER}:${APP_GROUP} /code /home/${APP_USER}
USER ${APP_USER}

EXPOSE 8000

# Start FastAPI (adjust the module:path if your ASGI app variable is named differently)
CMD uvicorn app.api.server:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}
