# Phase 9 — Research & Findings

Goal for this work cycle: **save user predictions, build a leaderboard, refresh to recent NBA data, and production-harden the whole system.**

This document is research only — it surfaces options and the decisions that block design. No architecture is committed here.

---

## 0. The decision that governs everything: what does the leaderboard rank?

There are two readings, with wildly different scope. They produce **different `predictions` table schemas**, so this blocks design.

### Reading A — Model leaderboard (small, nearly free)
Rank trained ML models by accuracy. The data already exists: `prediction_results` stores per-game `predictions` + `actual_results` arrays plus `accuracy`, `f1_score`, `cv_accuracy_mean`, etc. `/api/verify/{id}` already computes per-model confusion matrices. A model leaderboard is mostly a new read endpoint + a frontend page.
- No user concept, no game-resolution loop, no freshness dependency.
- "Save user predictions" would just mean persisting the client-side `history` list (currently in-memory in `Predictions.tsx`, lost on refresh).

### Reading B — User leaderboard (a whole system)
Rank **humans** by how accurate their saved picks turn out.
- **Trap:** the model is deterministic — same matchup → same output. If a "user prediction" is merely the model's answer persisted, every user gets an identical pick and the leaderboard measures nothing. For a user leaderboard to mean anything, the human must enter *their own* pick (agree with or override the model) **on a game not yet played**.
- That forces **resolution against real outcomes** → which forces **data freshness** (you need upcoming games to predict, and a job to fill in actuals later).
- So under Reading B, all four features are **one connected system**:
  `freshness → upcoming schedule + resolves picks → saved predictions reference unresolved games → leaderboard scores the resolved ones.`

**Consequence:** under Reading A the four features are independent and small; under Reading B they are a pipeline and freshness is load-bearing. **This must be answered before the predictions schema is designed.**

---

## 1. Current state (verified in code)

| Area | Finding |
|---|---|
| Predictions | Computed live in `predict_single` (`predictive_models.py:1052`) and returned; **never persisted**. Frontend keeps a `history` array in React state only (`Predictions.tsx:289`) — lost on refresh. |
| Users / identity | **No user concept anywhere.** Auth is a single shared secret `NBA_API_KEY` via `X-API-Key` (`api.py:46`) — gates mutations, does not identify a user. |
| DB | SQLite at `data/nba_predictions.db`. Tables: `import_records`, `game_data`, `models`, `prediction_results`. Migrations = inline `ALTER TABLE … try/except` (`database.py:93`). No `predictions`/`users` table. |
| Model leaderboard data | Already present: `prediction_results.predictions` + `.actual_results` (JSON arrays), per-model metrics, `/api/verify/{id}`. |
| Data recency | Training DB (import `id=2`) tops out at **SEASON 2022** (games to ~Dec 2022). |
| Importer format | Expects **wide** rows (`home_team, away_team, home_score, away_score, game_date` + optional box-score cols). Column-variation mapping in `data_importer.py:155`. |

---

## 2. Data freshness — options & reality checks

### Option 1 — Local 2010–2024 dataset (already on disk, cheapest)
`NBA-Data-2010-2024-main/` (source: github.com/NocturneBear/NBA-Data-2010-2024) — regular-season + playoff **totals** and **box scores**, 2010 through 2024.
- **Format mismatch:** totals are **long** (one row per team per game: `SEASON_YEAR, TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, WL, FG_PCT, FT_PCT, FG3_PCT, AST, REB, PTS, …`). The importer wants wide. → needs a **pivot/join** (pair the two rows of each `GAME_ID` into home/away using `MATCHUP` "X vs Y" = home, "X @ Y" = away), not a plain upload.
- Gets the model to **2024**, +2 seasons over current. One-time effort. No network dependency.

### Option 2 — `nba_api` (live, free, scrapes stats.nba.com)
`pip install nba_api`; `from nba_api.live.nba.endpoints import scoreboard` for today's games; `scoreboardv2` for historical by date.
- **Production caveat to VERIFY, not assume:** nba_api hits stats.nba.com, which is widely reported to **block datacenter/cloud IPs** and require specific headers. Must confirm it actually reaches stats.nba.com **from the deploy target** before relying on it.
- Richest data (full box scores) and truly current.

### Option 3 — balldontlie API (live, real API key, documented limits)
`/nba/v1/games`, `/box_scores`, `/standings`, plus odds/props on paid tiers.
- Free tier: **5 req / 60 s**; paid $9.99/mo → 60 req/min; $39.99/mo → 600 req/min.
- Real API key + documented limits → **more production-friendly** than scraping. Good for schedule + final scores; box-score depth varies by tier.

**Discriminating test:** if deploying to cloud, balldontlie is the safer live source; nba_api only if it provably reaches stats.nba.com from the host.

### The part that's easy to miss
**Fresh data is inert without a retraining loop.** Ingesting new games does not improve predictions by itself — label encoders + the 34-feature model must be **re-fit and the new model promoted**. "Data freshness" is a **pipeline** (fetch → pivot → import → retrain → swap active model), not a single endpoint. Cadence (nightly? after each game day?) is a decision.

---

## 3. Saving user predictions — design sketch (pending §0)

New `predictions` table, roughly:
`id, user_handle/user_id, model_id, home_team, away_team, season, game_date (the game being predicted),
predicted_label, predicted_confidence, model_label (what the model said), created_at,
resolved (bool), actual_label, correct (bool)`.

- Endpoints: `POST /api/predictions` (save), `GET /api/predictions?user=` (list), and a resolution job that fills `actual_label`/`correct` once the game is final.
- Under Reading A, drop everything from `game_date`/`resolved` onward and it's just a persisted history.
- Frontend hook: replace the in-memory `history` in `Predictions.tsx` with server reads/writes.

---

## 4. Leaderboard — design sketch (pending §0)

- **Reading A:** `GET /api/leaderboard` = models ranked by accuracy/F1/CV — read over `prediction_results`. ~A page + endpoint.
- **Reading B:** rank `user_handle` by resolved-prediction accuracy (with a minimum-N guard so a 1-for-1 user doesn't top a 60-for-100 user). Requires the resolution loop from §2.

---

## 5. Production hardening — only the items that change because of new write paths

(Full hardening list deferred — these are the ones the new features *force*.)

- **SQLite write concurrency.** Single-file SQLite + user-prediction writes + a resolution job + training all writing = classic contention. **WAL mode is the minimum**; Postgres is the real production answer. Flag now.
- **Migrations.** New `predictions`/`users` tables stretch the `ALTER TABLE try/except` pattern past comfort — adopt a real migration step (even a lightweight versioned one).
- **Identity model.** A leaderboard needs distinguishable users; the shared `X-API-Key` does not identify anyone. Options: (a) lightweight **handle/nickname** MVP — fine for a demo; (b) real auth (sessions/JWT) — needed for a *competitive* leaderboard (else people spoof handles). Decision.
- **Secret management** for the external data-API key — separate from `NBA_API_KEY`; never bake into the frontend image (the `front/Dockerfile` `VITE_*` foot-gun already noted in CLAUDE.md).
- Deferred but noted: structured logging/observability, healthcheck depth, backups for the new tables, CI.

---

## 6. Decisions — LOCKED (2026-06-04)

1. **Leaderboard semantics → MODELS.** Rank trained models by accuracy/F1/CV over `prediction_results`. Nearly free; the four features stay independent. No user-scoring resolution loop required for the leaderboard.
2. **Identity → HANDLE/NICKNAME MVP.** Saved predictions are keyed by a user-typed handle; no passwords. Honor-system. (Upgrade path to real auth left open.)
3. **Freshness → balldontlie LIVE API.** Real key + documented limits (free 5 req/60s). Wire an ingest job for recent games → pivot to wide → existing importer → **retrain & promote** the active model. Cadence TBD (manual button first, scheduled later).
4. **DB → SQLite + WAL for this cycle.** ⚠️ **Postgres reminder:** revisit Postgres when any of these become true — (a) the leaderboard becomes user-based / multi-writer concurrency rises, (b) the resolution/ingest job runs concurrently with user writes under real traffic, or (c) we deploy to more than one backend replica. Flag it at that point.

## 7. Resulting build sequence (model leaderboard + handle-keyed saved predictions + balldontlie freshness)

Because the leaderboard is model-based, the saved-predictions feature and the leaderboard are **decoupled**. Proposed order:

1. **DB + hardening base:** enable WAL; add `predictions` table (`id, user_handle, model_id, home_team, away_team, season, predicted_label, predicted_confidence, model_label, created_at` — plus optional `game_date/resolved/actual_label/correct` so personal history can be resolved against balldontlie finals later). Adopt a small versioned-migration helper.
2. **Save predictions API + UI:** `POST /api/predictions`, `GET /api/predictions?user=`; swap `Predictions.tsx`'s in-memory `history` for server reads/writes; add a handle field.
3. **Model leaderboard:** `GET /api/leaderboard` (rank models by accuracy/F1/CV from `prediction_results`); new frontend page + navbar entry.
4. **balldontlie freshness pipeline:** client module (keyed by `BALLDONTLIE_API_KEY` env, rate-limit aware) → fetch recent finals → map to wide importer rows → import → retrain → promote active model. Manual trigger endpoint first.
   - **LOCKED sub-decisions (2026-06-05):**
     - **Tier: start FREE, upgrade later.** Free `/games` gives scores/dates/teams/season but **NOT box scores** (FG%/AST/REB/FT%/3P% are GOAT $39.99 only). Build **tier-agnostic**: ingest whatever the key allows; on a free key the 10 box-score features mean-fallback for new games (24/34 features incl. Elo stay fresh). Add an **enrich step** so that after upgrading to a GOAT key + re-run, box-score columns backfill onto already-ingested games. Zero code change to switch tiers.
     - **Backfill: fill the 2022→now gap** (~SEASON 2022..current, ~5k games), paginated with sleeps for the 5 req/min free limit. One-time job via the manual endpoint.
     - **Resolve saved predictions: YES.** Match each saved prediction to its real final (by teams+date), fill `resolved`/`actual_label`/`correct`, and add a per-user accuracy view. Foundation for a future user leaderboard (board itself stays model-based for now).
     - **Cross-import gate (from step 3):** resolve leaderboard scoping by `import_id` as part of this step (new imports/models will otherwise be ranked on incomparable test splits).
   - **Secrets:** `BALLDONTLIE_API_KEY` in a gitignored `.env` + `.env.example`; never committed.
   - ⚠️ **GATE (carried from step 3 review):** the leaderboard (`GET /api/leaderboard`) currently ranks **all** models across **all** imports together. With one import that's apples-to-apples (RF & DT both eval on 5,311 games of import 2). The moment step 4 creates a *new* import → retrains → new model rows, the board would compare accuracies measured on **different test splits** as if equal. Resolve as part of step 4: either **scope the board by `import_id`** (reuse the param `/api/results` already has + an import selector like Verify), or surface each model's import + eval-game count so the basis is visible. Don't let freshness silently make the board misleading.
5. **Prod hardening sweep:** secret management for the balldontlie key, structured logging, healthcheck depth, backups for new tables, CI, deferred items from §5.
