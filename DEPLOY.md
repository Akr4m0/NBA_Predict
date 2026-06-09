# Deployment Guide

Practical steps to run NBA Predict in production. For architecture/dev, see
`CLAUDE.md`.

---

## 1. Prerequisites

- Docker + Docker Compose (the supported deploy path), **or** Python 3.11 +
  Node 20 for a bare-metal run.
- A `balldontlie` API key (free tier = scores only) — only needed if you use the
  data-freshness pipeline.

---

## 2. Configure secrets

Copy the example and fill it in (`.env` is gitignored — never commit it):

```bash
cp .env.example .env
```

| Variable | Purpose | Prod recommendation |
|---|---|---|
| `NBA_API_KEY` | Require `X-API-Key` on mutating endpoints | **Set it** (unset = open) |
| `BALLDONTLIE_API_KEY` | Data-freshness feed | Set if using freshness |
| `NBA_CORS_ORIGINS` | Allowed browser origins (comma-sep) | Set to your real frontend URL |
| `VITE_API_BASE_URL` | API URL baked into the SPA **at build time** | Set to your public API URL |

---

## 3. Docker deploy (recommended)

`VITE_API_BASE_URL` is compiled into the frontend bundle, so it must be correct
**at build time** — pass it as a build arg / compose variable, not at runtime.

```bash
# From the project root, with .env populated:
docker compose build
docker compose up -d
```

This brings up:
- **backend** (uvicorn) on `:8000` — DB, uploads, and trained models persist
  under `./data` via the bind mount.
- **frontend** (nginx) on `:8080`.

### Verify the deploy (the smoke test)

```bash
# 1. Backend health (DB ping) — expect {"status":"ok","database":true,...}
curl -s http://localhost:8000/api/health

# 2. Container healthcheck flips to "healthy" within ~30s
docker inspect --format '{{.State.Health.Status}}' nba-predict-backend

# 3. Frontend serves
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/        # 200

# 4. (auth on) a mutating call without the key is rejected
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/api/train \
  -H 'Content-Type: application/json' -d '{}'                          # 401
```

If health stays `503`/`starting`, check `docker compose logs backend`.

---

## 4. Load data & train

```bash
# Option A — upload via the UI: open http://localhost:8080/import, upload
#   data/games.csv (or data/balldontlie_finals.csv), then train on /train.

# Option B — refresh to current games via balldontlie, then retrain:
docker compose exec backend python3 backfill_freshness.py \
  --seasons 2022 2023 2024 2025 --retrain random_forest
```

Regenerate the combined CSV any time:
`docker compose exec backend python3 export_balldontlie_csv.py --seasons 2022 2023 2024 2025`

---

## 5. Backups

```bash
# Consistent, WAL-safe snapshot (writes data/nba_predictions.db.backup-<ts>)
docker compose exec backend python3 backup_db.py
```

Schedule it (cron/systemd timer) and copy the backup off-host.

---

## 6. Bare-metal (no Docker)

```bash
# Backend
cd backend && pip install -r requirements.txt
NBA_API_KEY=... BALLDONTLIE_API_KEY=... \
  uvicorn api:app --host 0.0.0.0 --port 8000

# Frontend (build with the real API URL, serve the static dist/)
cd front && npm ci
VITE_API_BASE_URL=https://api.yourdomain.com npm run build
#   then serve front/dist/ with nginx / any static host
```

---

## 7. Known limitations (plan around these)

- **Rate limiter keys on the direct client IP.** Behind nginx/Docker that's the
  proxy, so per-IP limits become effectively global. For real per-client limiting,
  enforce it at your proxy/CDN.
- **`VITE_API_BASE_URL` is baked at build time** — no runtime override. Rebuild
  the frontend image to change the API URL.
- **SQLite + WAL** is single-writer. Fine for one backend replica; for multiple
  replicas or heavy concurrent writes, migrate to Postgres.
- **Identity is honor-system handles** (no auth on the user leaderboard). Add real
  auth before treating the user board as competitive.
- **Don't run the container as non-root** against the `./data` bind mount without
  matching uids — it breaks DB/model writes.
- **balldontlie free tier = scores only**; box-score features mean-fallback for
  new games until a GOAT key + re-run enriches them.
