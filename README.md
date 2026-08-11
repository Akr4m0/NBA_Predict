# NBA Game Prediction System

Predicts the winner of NBA games from pre-game information only. A FastAPI +
scikit-learn backend trains and serves the models; a React single-page app
handles import, training, prediction, and verification.

Built as the software component of a diploma thesis.

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, scikit-learn, XGBoost, pandas, SQLite (WAL) |
| Frontend | React 18, TypeScript, Vite, Tailwind, shadcn/ui, Recharts, Framer Motion |
| Serving | uvicorn (`:8000`), nginx (`:8080`), Docker Compose |

## Quick start

Docker Compose is the supported path:

```bash
cp .env.example .env      # fill in keys; .env is gitignored
docker compose build
docker compose up -d
```

- Frontend → http://localhost:8080
- API + Swagger docs → http://localhost:8000/docs

Then open `/import` to upload `data/games.csv`, and `/train` to fit the models.

Running without Docker:

```bash
# Backend
cd backend && pip install -r requirements.txt
python3 -m uvicorn api:app --port 8000 --reload

# Frontend
cd front && npm ci
npm run dev
```

See [DEPLOY.md](DEPLOY.md) for production configuration, backups, the deploy
smoke test, and known limitations.

## Project layout

```
backend/
  api.py                    FastAPI app — 17 endpoints, API-key auth, rate limiting
  database.py               SQLite schema, migrations, WAL handling
  data_importer.py          CSV/Excel ingest, column mapping, validation, cleaning
  predictive_models.py      Feature engineering, training, calibration, prediction
  balldontlie_client.py     External scores feed client
  data_freshness.py         Fetch → import → retrain → promote → resolve pipeline
  team_locations.py         Arena coordinates and team-ID mapping
  backfill_freshness.py     One-shot backfill over past seasons
  export_balldontlie_csv.py Regenerate the combined CSV
  backup_db.py              WAL-safe database snapshot
  tests/                    78 pytest tests

front/
  src/pages/                Home, Dashboard, Import, Train, Predictions,
                            Leaderboard, Analysis, Verify, About
  src/components/           UI components (shadcn/ui)
  src/lib/                  API client, team data, utilities (20 vitest tests)

data/
  games.csv                 Historical game data
  Games_most_recent.csv     Extended dataset through the 2026 Finals

docs/                       Database schema and design notes
```

## Models

Four classifiers are trained and compared:

- **Decision Tree** — interpretable decision paths, used for feature-importance readouts
- **Random Forest** — bagged ensemble, the default recommendation
- **XGBoost** — gradient boosting
- **Baseline** — most-frequent-class predictor, the benchmark the others must beat

Probabilities are calibrated with `CalibratedClassifierCV`. Evaluation uses
`TimeSeriesSplit` rather than random k-fold, so no future game is ever used to
predict a past one.

## Features

Features are derived strictly from information available **before tip-off** —
team identity and encoding, temporal fields (month, day of week, season),
rolling form and win/loss streaks, rest days, home-court advantage, and
travel distance between arenas.

Post-game box-score statistics are deliberately excluded from the feature set.
[docs/LEAKAGE_INVESTIGATION.md](docs/LEAKAGE_INVESTIGATION.md) documents a
latent leakage trap found during development and how it was removed.

## Frontend routes

| Route | Purpose |
|---|---|
| `/` | Landing page |
| `/dashboard` | Overview and navigation hub |
| `/import` | Upload and validate CSV/Excel datasets |
| `/train` | Configure and train models |
| `/predictions` | Generate and browse game predictions |
| `/leaderboard` | User prediction leaderboard |
| `/analysis` | Model performance charts and feature importances |
| `/verify` | Compare predictions against actual results |
| `/about` | Project information |

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `NBA_API_KEY` | If set, required as `X-API-Key` on mutating endpoints |
| `NBA_CORS_ORIGINS` | Comma-separated allowed browser origins |
| `BALLDONTLIE_API_KEY` | Scores feed for the data-freshness pipeline |
| `VITE_API_BASE_URL` | API URL, compiled into the SPA **at build time** |
| `NBA_DB_PATH` | SQLite path (default `../data/nba_predictions.db`) |

## Tests

```bash
cd backend && python3 -m pytest tests/    # 78 tests
cd front && npx vitest run                # 20 tests
```

Backend tests run against a temporary database and never touch
`data/nba_predictions.db`.

## Documentation

- [DEPLOY.md](DEPLOY.md) — production deployment, backups, limitations
- [backend/README.md](backend/README.md) — backend usage and environment variables
- [front/README.md](front/README.md) — frontend development notes
- [docs/DATABASE_SCHEMA.txt](docs/DATABASE_SCHEMA.txt) — table definitions
- [docs/LEAKAGE_INVESTIGATION.md](docs/LEAKAGE_INVESTIGATION.md) — leakage case study
- [docs/PHASE9_RESEARCH.md](docs/PHASE9_RESEARCH.md) — design decisions
