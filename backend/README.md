# NBA Prediction Backend

FastAPI + scikit-learn + SQLite. See [`../README.md`](../README.md) for the
architecture overview and feature list, and [`../DEPLOY.md`](../DEPLOY.md) for
the deployment story.

## Quick start

```bash
pip install -r requirements.txt
python3 -m uvicorn api:app --port 8000 --reload
# → http://localhost:8000   (Swagger: /docs)
```

## Tests

```bash
python3 -m pytest tests/
# 78 tests, tmp DB, never touches data/nba_predictions.db.
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `NBA_DB_PATH` | `../data/nba_predictions.db` | SQLite path |
| `NBA_UPLOADS_DIR` | `../data/uploads` | CSV upload landing dir |
| `NBA_MODELS_DIR` | `../data/models` | Where trained `.pkl` files go |
| `NBA_API_KEY` | unset (dev) | If set, required as `X-API-Key` on POST endpoints |
| `NBA_CORS_ORIGINS` | `http://localhost:8080` | Comma-separated allowed origins |
