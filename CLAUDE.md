# NBA Prediction System — Claude Code Context

Full-stack NBA game-outcome predictor.

- **Backend**: Python 3.9 / FastAPI / scikit-learn / SQLite (`backend/`)
- **Frontend**: React 18 / TypeScript / Vite / Tailwind / shadcn-ui / Recharts / Framer Motion (`front/`)
- **Data**: `data/games.csv` — 26,651 rows / 26,622 unique games / 30 teams, spanning SEASON 2003–2022 (date range Jan 2005 → Sep 2020)

## Running the project

```bash
# Terminal 1 — backend
cd backend
python3 -m uvicorn api:app --port 8000 --reload
# → http://localhost:8000   (Swagger: /docs)

# Terminal 2 — frontend
cd front
npm install   # first time only
npm run dev
# → http://localhost:8080
```

> Port 8000 may conflict with another local app on this machine. If `uvicorn` errors with "address already in use", run on `--port 8001` and set `VITE_API_BASE_URL=http://localhost:8001` for the frontend (or update the default in `front/src/lib/api.ts:3`).

### Running tests

```bash
# Backend (pytest, 77 tests — tmp DB, never touches prod data)
cd backend && python3 -m pytest tests/

# Frontend (vitest, 20 tests — pure-function + fetch-mocked tests for lib/)
cd front && npm test
```

### Running with Docker

```bash
# From project root:
docker compose up --build
# → frontend at http://localhost:8080, backend at http://localhost:8000.
# DB + uploads + trained models persist under ./data via a bind mount.
# Set NBA_API_KEY=... in the environment (or .env) to require X-API-Key on mutating endpoints.
```

---

## Architecture

### Database — `backend/database.py`
SQLite at `data/nba_predictions.db`. Tables:
- `import_records` — each CSV upload session
- `game_data` — individual games (`home_team`, `away_team` as NBA team IDs, integers e.g. `1610612738` = Boston Celtics)
- `models` — trained model registry (id, name, type, parameters, model_path, created_date)
- `prediction_results` — per-model metrics: accuracy, precision_score, recall, f1_score, confidence, feature_importance (JSON), **cv_accuracy_mean**, **cv_accuracy_std**, **calibration_method**, predictions/actual_results arrays
- `predictions` — handle-keyed saved user predictions: `user_handle, model_id, home_team, away_team, season, predicted_label, predicted_confidence, model_label, created_at` + resolution fields `game_date, resolved, actual_label, correct` (filled by the freshness pipeline against real finals)

`init_database` sets **WAL mode** (concurrent reads during writes). Two migration paths: legacy `_migrate()` (`ALTER TABLE … ADD/DROP COLUMN` in try/except; DROP needs SQLite 3.35+) and a forward-looking versioned helper `_apply_migrations()` keyed on `PRAGMA user_version` (the `predictions` table is migration v1).

### Models — `backend/predictive_models.py`
`PredictiveModels` class. Four model types:

| API name | DB type | Class | Calibration |
|---|---|---|---|
| `decision_tree` | `DecisionTree` | `DecisionTreeClassifier` | sigmoid |
| `random_forest` | `RandomForest` | `RandomForestClassifier` | isotonic |
| `xgboost` | `XGBoost` | `XGBClassifier` | isotonic |
| `baseline` | `Baseline` | `DummyClassifier(most_frequent)` | none |

**Features** (34 total, all leakage-safe — strict `< game_date` lookups via `np.searchsorted` on pre-built per-team numpy arrays):

- Categorical: `home_team_encoded`, `away_team_encoded`, `season_encoded`
- Calendar: `month`, `day_of_week`, `day_of_year`
- Rolling form: `home_team_last5_winrate`, `away_team_last5_winrate`, `h2h_home_wins_rate`
- **Elo** (FiveThirtyEight-style, K=20, HCA=100, 25% season-boundary regression toward 1505): `home_elo_pre`, `away_elo_pre`, `elo_diff`
- **Rest**: `home_rest_days`, `away_rest_days`, `home_b2b`, `away_b2b`, `rest_diff`
- **Venue win-rate**: `home_team_home_winrate`, `away_team_away_winrate`
- **Rolling box-score** (10 features — each side's last-10-games mean for FG%, FT%, 3P%, AST, REB; sourced from each team's *own* perspective regardless of venue): `home_fg_pct_l10`, `away_fg_pct_l10`, `home_ft_pct_l10`, `away_ft_pct_l10`, `home_fg3_pct_l10`, `away_fg3_pct_l10`, `home_ast_l10`, `away_ast_l10`, `home_reb_l10`, `away_reb_l10`
- **Travel & circadian** (using `team_locations.py` — lat/lon + UTC offset for all 30 arenas): `home_travel_dist`, `away_travel_dist` (haversine miles since each team's prior game), `home_tz_shift`, `away_tz_shift` (signed hours, +east)
- **Season type**: `is_playoff` (1 for playoffs or play-in, 0 otherwise — derived from `GAME_ID` leading digit in `data_importer._clean_data`)

`elo_diff` is the single most important feature (~19% of RF importance); `home_fg_pct_l10` / `away_fg_pct_l10` each contribute ~3-3.5%.

**Training**: Every model except Baseline is wrapped in `CalibratedClassifierCV` so `predict_proba` is well-calibrated. Each train also runs `TimeSeriesSplit(n_splits=5)` cross-validation (leakage-safe temporal CV) and persists `cv_accuracy_mean` / `cv_accuracy_std` to `prediction_results`.

**Persistence**: `data/models/{type}_{import_id}_{YYYYMMDD_HHMMSS}.pkl` + sidecar `.json` with feature_names, label_encoder classes, feature_means. `feature_importances_` is accessed through `_averaged_feature_importances()` because the model is now wrapped in `CalibratedClassifierCV`.

**Confidence**: `mean(max(predict_proba))`. Baseline capped at 0.99; `confidence_reliable: false` for Baseline only.

**LabelEncoder convention** (relevant for the confusion matrix in `/api/verify`): `1 = home_win`, `0 = away_win` (alphabetical).

### Data importer — `backend/data_importer.py`
Public `validate_csv(df)` → `{valid, errors, warnings}`.
- Required columns: `home_team`, `away_team`, `game_date` (errors on null)
- Score nulls → warnings (unplayed/future games skipped)
- Ties → treated as Away Win (with warning)

Optional CSV columns are recognised and persisted to `game_data` when present:
- Box-score stats per side: `FG_PCT_home/away`, `FT_PCT_home/away`, `FG3_PCT_home/away`, `AST_home/away`, `REB_home/away`. Used to compute rolling pre-game stat features.
- `GAME_ID` — first digit (1/2/4/5 = preseason/regular/playoffs/play-in) is converted to a `season_type` string column.

### REST API — `backend/api.py`
CORS: `http://localhost:8080` by default; override via `NBA_CORS_ORIGINS=https://a.com,https://b.com`.

**Auth**: set `NBA_API_KEY=<secret>` to require an `X-API-Key` header on the mutating endpoints (`POST /api/imports/upload`, `DELETE /api/imports/{id}`, `POST /api/train`, `POST /api/predict`, `POST /api/predictions`, `POST /api/freshness/sync`). Reads are always open. Unset env var = no auth (dev mode).

**Rate limits** (slowapi, per remote IP):
- `POST /api/predict` — 60/minute
- `POST /api/predictions` — 60/minute
- `POST /api/train` — 5/hour
- `POST /api/freshness/sync` — 5/hour
- `POST /api/imports/upload` — 20/hour
- `DELETE /api/imports/{id}` — 30/hour

Exceeding any returns `429 {"detail": "Rate limit exceeded: ..."}`.

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | `{status, database, timestamp}` — 200 ok / 503 if DB unreachable (DB ping only, no external calls) |
| GET | `/api/imports` | List all import records |
| POST | `/api/imports/upload` | Upload CSV/XLSX → validate → import |
| DELETE | `/api/imports/{id}` | Delete an import + its games/results/models (keeps user predictions); removes model files |
| GET | `/api/imports/{id}/teams` | Unique team IDs for an import |
| GET | `/api/imports/{id}/preview` | First 10 rows |
| GET | `/api/models` | All models with latest metrics |
| POST | `/api/train` | Sync train `{import_id, model_type, params}` |
| GET | `/api/results` | Model comparison table |
| POST | `/api/predict` | `{home_team, away_team, season, model_id?, model_type?}` |
| POST | `/api/predictions` | Save a handle-keyed prediction (computed server-side) |
| GET | `/api/predictions?user=&limit=` | List saved predictions (newest first), `model_name` joined |
| GET | `/api/leaderboard?sort=&import_id=` | Rank models by accuracy/`cv`/`f1`; `import_id` scopes to one import's test split |
| GET | `/api/user-leaderboard?user=` | Per-handle accuracy over **resolved** predictions |
| GET | `/api/freshness/status` | `{configured, latest_import_id, latest_game_date, total_games}` |
| POST | `/api/freshness/sync` | Small incremental balldontlie sync (requires `max_games ≤ 250`; full backfill = CLI) |
| GET | `/api/verify/{import_id}` | Per-model accuracy + real confusion matrix (`tp`, `tn`, `fp`, `fn`) |

### Frontend pages

| Path | Source | Status |
|---|---|---|
| `/` | `pages/Index.tsx` | Landing |
| `/dashboard` | `pages/Dashboard.tsx` | KPIs, grouped bar chart, line chart (wired) |
| `/import` | `pages/Import.tsx` | Upload with progress, preview, validation panel (wired) |
| `/train` | `pages/Train.tsx` | Per-model train cards with sliders (wired) |
| `/predictions` | `pages/Predictions.tsx` | Team combobox + model picker + result card (wired) |
| `/analysis` | `pages/Analysis.tsx` | Radar, feature importance, estimated confusion matrix (wired) |
| `/leaderboard` | `pages/Leaderboard.tsx` | Medal-ranked model board (sort by accuracy/CV/F1), import scope, freshness status (wired) |
| `/verify` | `pages/Verify.tsx` | KPIs, per-model accuracy bars, **real** confusion matrices (wired) |
| `/about` | `pages/About.tsx` | About |

**Shared infra**: `lib/api.ts` (typed fetch + `ApiError`), `lib/teams.ts` (NBA team ID → name map, 30 teams), `lib/utils.ts` (`cn`). Components: `GlassCard`, `EmptyState`, `TopLoadingBar`, `ErrorBoundary`, Navbar with Framer Motion `layoutId="navbar-indicator"` sliding underline. `App.tsx` renders `<Navbar />` and `<Footer />` **once, outside `<Suspense>`** so they stay mounted across lazy-route transitions (pages never re-render the chrome). It also installs global QueryClient `onError` hooks that surface API errors via `sonner` toast.

---

## Reference accuracy (RF, 34 features, temporal split)

> Historical measurement on the original **26,552-game** set (2003→2022). The live DB was reset + re-uploaded with the balldontlie-refreshed set (now ~31,353 games through 2026-06-03), so newly-trained models will differ — check `/leaderboard` for current numbers.


| Metric | Value |
|---|---|
| Test accuracy | **64.2%** |
| **TimeSeriesSplit CV accuracy** (honest) | **66.0%** |
| Baseline test accuracy | 55.6% |
| Lift over baseline | **+8.6pp** |
| F1 | 0.626 |

History: 19-feature RF (before box-score / travel / playoff additions) got 63.4% test / 65.6% CV.

Vegas closing lines land around 67–68% as a reference ceiling for game-winner prediction.

---

## Phase status — all done

- **Phase 1** (model accuracy): Elo, rest-days, venue win-rates, `CalibratedClassifierCV`, `TimeSeriesSplit` CV, DB migrations.
- **Phase 2** (Verify.tsx): wired to `/api/verify/{id}` with real `tp`/`tn`/`fp`/`fn` per model.
- **Phase 3** (team names): `lib/teams.ts` + Predictions combobox displays team names + searchable by name or ID.
- **Phase 4** (cleanup): pytest suite in `backend/tests/`. Dead stat-extraction code removed; see `docs/LEAKAGE_INVESTIGATION.md`.
- **Phase 5** (lift candidates + production hardening, 2026-05-27): rolling box-score stat features (10), travel distance + timezone shift (4), playoff flag (1) — feature count 19 → 34, +0.4pp CV accuracy. Production hardening: `NBA_API_KEY` shared-secret auth, slowapi rate limits, Dockerfile + docker-compose for the full stack.
- **Phase 6** (performance polish, 2026-05-29): vectorized `_add_historical_features` via `_precompute_team_pregame_features` + per-pair h2h cumsum + vectorized haversine — `prepare_features` on 26K rows went from 8.8s → 1.3s (6.6×). Frontend route-split with `React.lazy` — initial JS bundle 1,033 KB → 459 KB (302 KB → 147 KB gzipped); Recharts now its own lazy chunk loaded on `/dashboard` and `/analysis`. Added frontend vitest suite (`front/src/lib/{teams,utils,api}.test.ts` — 20 tests).
- **Phase 7** (UX fixes + test isolation, 2026-06-02): lifted `<Navbar />` + `<Footer />` from per-page render into `App.tsx` outside `<Suspense>` — fixes the "menu needs multiple clicks" bug that happened during lazy-route chunk loads (the old per-page navbar unmounted with the page and the fallback covered the viewport). `RouteFallback` now sits below the (still-visible) navbar. `predict_single` builds X as a `pd.DataFrame(columns=feature_names)` instead of a raw `np.array`, silencing sklearn's "X does not have valid feature names" UserWarning. **Test pollution fix**: `MODELS_DIR` now reads from `NBA_MODELS_DIR` env var; conftest's autouse `_isolate_models_dir` fixture sets it per-test. `api_client` switched from `sys.modules.pop`+re-import to `importlib.reload` so it doesn't strand other test files' references to the original `predictive_models` module (which was the source of stray `baseline_1_*.pkl` files leaking into `data/models/`).
- **Phase 8** (review-driven cleanup, 2026-06-03): three correctness fixes + a sweep of debt found by the final review.
  - `_build_history_index` now `drop_duplicates(['game_date','home_team','away_team'], keep='first')` so the 29 duplicate `GAME_ID`s in `games.csv` no longer double-count in `_team_wins` / rolling stats.
  - `_compute_travel_tz_vectorized` guards against missing `*_prev_venue` columns (the empty-pregame edge case).
  - `_migrate` checks `sqlite3.sqlite_version_info` and logs a clear warning rather than silently failing the `DROP COLUMN` statements on SQLite < 3.35.
  - `App.tsx` moved Navbar and Footer **outside** `<ErrorBoundary>` so a page render error still leaves the chrome usable.
  - New test `test_vectorized_and_per_call_paths_agree` asserts row-by-row equivalence between the per-call helpers (used by `predict_single`) and the vectorized path (used by training) — protects the Phase 6 split from silent drift.
  - `pytest.ini` filterwarnings narrowed: stops blanket-suppressing `UserWarning` — which immediately surfaced a real bug (`validate_csv`'s `warnings: List[str]` local was shadowing the stdlib `warnings` module). Renamed the local to `warning_msgs`.
  - Misc: stale marketing copy on `Index.tsx` / `About.tsx` (78% → 66%, neural nets → "coming soon"); `/api/health` uses `datetime.now(timezone.utc)`; `backend/README.md` rewritten to point at this doc; `front/Dockerfile` calls out the baked-in `VITE_API_BASE_URL` foot-gun; unused `StandardScaler` import dropped; `joblib` pinned in `requirements.txt`; `save_game_data` switched to `executemany`. Deleted abandoned `PROJECT_ORGANIZATION.md`, `prompr.md`, and the md5-duplicate `data/uploads/games.csv`.
- **Phase 9** (saved predictions, leaderboard, data freshness, prod hardening, 2026-06-05): research + decisions in `docs/PHASE9_RESEARCH.md`.
  - **Saved predictions**: `predictions` table (WAL + `user_version` versioned migrations); `POST/GET /api/predictions` (handle-keyed, computed server-side); `Predictions.tsx` gains a handle field (localStorage), Save button, server-backed history with Hit/Miss/Pending resolution badges + per-user accuracy.
  - **Model leaderboard**: `GET /api/leaderboard` (sort accuracy/cv/f1, `import_id`-scopable so cross-import test splits aren't mixed); `/leaderboard` page (medals, metric toggle, import scope, freshness status). `GET /api/user-leaderboard` seeds a future user board from resolved picks.
  - **Data freshness (balldontlie)**: `balldontlie_client.py` (cursor-paginated, 13s throttle + escalating 429 backoff for the free 5 req/min tier), `data_freshness.py` (`FreshnessPipeline`: map via `team_locations.ABBR_TO_ID`, dedup-append, resolve predictions, optional retrain), CLIs `backfill_freshness.py` (DB) and `export_balldontlie_csv.py` (clone games.csv → `balldontlie_finals.csv`). **Free tier = scores only** (box scores GOAT-only → those 10 features mean-fallback for new games; tier-agnostic + enrich-on-upgrade). `POST /api/freshness/sync` is incremental-only (`max_games ≤ 250`); full backfill is the CLI.
  - **Delete imports**: `DELETE /api/imports/{id}` cascades games + results + models-trained-only-on-it (+ their files); Import page trash button → AlertDialog confirm.
  - **Prod hardening**: deep `/api/health` (DB ping, 503 on failure) + Docker `HEALTHCHECK`; compose passes `BALLDONTLIE_API_KEY`; `NBA_MODELS_DIR` set explicitly in the Dockerfile; `.env` auto-loaded (`api._load_dotenv`) + `.env.example`; `backup_db.py` (`VACUUM INTO`); `logging.basicConfig` (level via `NBA_LOG_LEVEL`); Import-page upload now honours `VITE_API_BASE_URL`. 77 backend + 20 frontend tests.

---

## Key decisions (locked)

| Decision | Choice |
|---|---|
| Training endpoint | Sync (blocks, returns metrics directly) |
| Neural Network card | Greyed out with "Coming Soon" badge |
| DB migrations | Inline `ALTER TABLE` with try/except at startup |
| File upload component | Custom `DropZone` (no `react-dropzone`) |
| `/api/predict` model selection | Accepts both `model_id` (explicit) and `model_type` (latest) |
| CORS | Default `http://localhost:8080`; configurable via `NBA_CORS_ORIGINS` |
| API auth | Shared secret in `NBA_API_KEY` env (X-API-Key header on mutating endpoints). No auth if unset |
| Rate limiter | slowapi keyed on remote IP; limits hardcoded per-endpoint in `api.py` |
| Deployment | docker-compose: `backend` (uvicorn) + `frontend` (nginx). DB persists under `./data` bind mount |
| Tie games | Treated as Away Win, warning logged |
| Probability calibration | `CalibratedClassifierCV` — isotonic for RF/XGB, sigmoid for DT, none for Baseline |
| Cross-validation | `TimeSeriesSplit(n_splits=5)` — never random KFold (would leak future into past) |

## Gotchas

- **Team IDs**: integers in `games.csv` (e.g. `1610612738` = Boston Celtics). Frontend maps via `front/src/lib/teams.ts`; backend stores raw integers.
- **XGBoost on macOS**: needs `libomp.dylib`. If `import xgboost` fails with "Library not loaded: @rpath/libomp.dylib", the wheel's hardcoded rpath (`/usr/local/opt/libomp/lib`) doesn't match your machine. Two fixes that work without sudo:
  1. `brew install libomp` if you have Intel Homebrew at `/usr/local/`.
  2. On Apple Silicon w/ x86_64 Python (Rosetta), patch the rpath to reuse sklearn's bundled libomp:
     ```bash
     install_name_tool -add_rpath \
       "$(python3 -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')" \
       "$(python3 -c 'import xgboost, os; print(os.path.join(os.path.dirname(xgboost.__file__), "lib", "libxgboost.dylib"))')"
     ```
     Re-run after any `pip install --force-reinstall xgboost`.
- **Removed features**: `score_difference`, `total_score`, and constant `home_advantage=1.0` are gone — first two were 100% leakage (encoded the outcome), the third carried zero information.
- **Per-game stat columns** (`FG_PCT_home`, `AST_home`, `REB_home`, etc.) are post-game outcomes — using them flat would be 100% leakage. The naive direct-use path (`_extract_team_stats` / `_extract_statistical_features`) was removed on 2026-05-20; see `docs/LEAKAGE_INVESTIGATION.md`. The stats are now persisted to `game_data` *only* to build pre-game rolling means in `_rolling_stat_mean`, which uses `np.searchsorted(side='left')` to ensure no future-game values bleed in. See `tests/test_predictive_models.py::test_rolling_stat_mean_does_not_leak_future` for the verification.
- **Duplicate games**: the CSV has 26,622 unique `GAME_ID`s but 26,651 rows (29 duplicates). `_compute_elo_history` deduplicates on `(game_date, home_team, away_team)` before its forward pass.
- **`data/models/`** and **`data/uploads/`** are auto-created on first run.
- **balldontlie free tier = scores only.** `/games` gives final scores/dates/teams/season; box scores (FG%/AST/REB…) are GOAT-tier ($39.99/mo). New games ingested on a free key leave the 10 box-score columns blank → those features mean-fallback. The pipeline is tier-agnostic; a GOAT key + re-run enriches them. balldontlie uses its own team IDs — we join on `abbreviation` via `team_locations.ABBR_TO_ID`.
- **Known prod limitations (acknowledged, not bugs):** (1) the slowapi rate limiter keys on the *direct* client IP — behind nginx/Docker that's the proxy, so per-IP limits become eff_global; put real per-client limiting at the proxy if needed. (2) `VITE_API_BASE_URL` is baked into the SPA at **build** time (no runtime override) — set the compose `args`/`--build-arg` for your deploy. (3) Don't run the container as non-root against the `./data` bind mount without matching uids — it breaks DB/model writes.
- **Pre-Phase-5 artifacts**: cleared on 2026-05-29. Backup at `data/nba_predictions.db.before-reset-2026-05-29`. Pre-React dev tools (`nba_predictor.py`, `dashboard.py`, `performance_evaluator.py`) and their deps (`dash`, `dash-bootstrap-components`, `matplotlib`, `seaborn`, `plotly`, `sqlalchemy`) were removed at the same time.
- **2026-06-05 full reset + re-upload**: the DB was wiped to empty (backup at `data/nba_predictions.db.backup-20260605_184348`), then `data/balldontlie_finals.csv` (games.csv clone + balldontlie finals) was uploaded. Current live state: **import `id=2`, 31,353 games spanning 2003-10-05 → 2026-06-03**, plus whichever models you train on it. The 2022→2026 rows have blank box-score columns (free-tier). `balldontlie_finals.csv` is regenerable via `backend/export_balldontlie_csv.py`.
