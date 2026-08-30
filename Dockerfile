FROM node:24-bookworm-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    DOCINTEL_DATA_DIR=/data \
    DOCINTEL_FRONTEND_DIR=/app/frontend/dist \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 tesseract-ocr tesseract-ocr-eng \
  && rm -rf /var/lib/apt/lists/* \
  && python -m venv /opt/venv

COPY pyproject.toml README.md ./
COPY backend/src ./backend/src
RUN python -m pip install --upgrade pip \
  && python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130 \
  && python -m pip install .

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 CMD curl --fail --silent --output /dev/null http://127.0.0.1:8000/api/health || exit 1
CMD ["uvicorn", "docintel.api:app", "--host", "0.0.0.0", "--port", "8000"]
