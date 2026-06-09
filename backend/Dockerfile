# NBA Prediction backend.
# Builds a slim image with uvicorn + FastAPI serving the API on :8000.
FROM python:3.11-slim

# OpenMP (XGBoost) is straightforward on Debian — apt provides the right arch automatically.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer caches across code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The DB, uploaded files, and trained models all live under /data so they
# persist via a volume. NBA_MODELS_DIR is set explicitly rather than relying on
# the path resolving to /data/models by accident of the COPY layout.
ENV NBA_DB_PATH=/data/nba_predictions.db \
    NBA_UPLOADS_DIR=/data/uploads \
    NBA_MODELS_DIR=/data/models

RUN mkdir -p /data/uploads /data/models

EXPOSE 8000

# DB-only liveness probe (no external calls). Uses python so we don't need curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
