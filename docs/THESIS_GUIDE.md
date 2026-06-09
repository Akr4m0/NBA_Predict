# Thesis Guide — NBA Game-Outcome Prediction System

**Purpose.** Everything the next session needs to write an academic thesis (CS / Data-Science / ML, ~50–90 pp) about this project: the chapter outline (Part A), an actionable writing guide (Part B), and a technical evidence inventory mapping every claim to code (Part C). Open this one file and go.

> Drafted 2026-06-09 from a two-agent sweep of the repo (Explore inventory + general-purpose outline). No thesis text written yet — this is the plan.

---

## Framing note (read before anything else)

**The contribution is the methodology, not the accuracy number.** The system tops out at ~64% test / ~66% CV accuracy — *below* the ~67–68% Vegas closing-line ceiling. Framed as "a system that predicts NBA winners," the obvious committee question is "so it loses to the market — what's the contribution?" Don't walk into that.

The genuine research core is **leakage-safe temporal feature construction + honest temporal evaluation**, anchored by the documented near-miss in `docs/LEAKAGE_INVESTIGATION.md`: a one-line `json.loads` "fix" would have silently produced ~95% accuracy that is *useless on real future games*. Most student sports-ML projects contain exactly that bug and ship it. This one catches it, explains the mechanism, and prevents it by construction (`np.searchsorted(side='left')` strictly-before-game lookups) and by validation (`TimeSeriesSplit`, never random KFold). That is the spine of the thesis.

Weighting: **methodology (primary) → feature-ablation empirics (backbone) → production system / MLOps (clearly demarcated engineering artifact, not a research claim).**

**Fact sources, in priority order:** (1) code under `backend/` and `front/` — authoritative; (2) `CLAUDE.md`, `docs/PHASE9_RESEARCH.md`, `docs/LEAKAGE_INVESTIGATION.md` — authoritative for design/decisions; (3) root `README.md` — **stale, pre-rewrite** (mentions Dash, a CLI, port 5000, "78%") — historical artifact only, never a fact source.

**Two numbers to re-establish at write time (do NOT transcribe):**
- All accuracy/CV/F1 figures in the docs were measured on the **original 26,552-game set (2003–2022)**. The live DB is now **import `id=2`, 31,353 games, 2003-10-05 → 2026-06-03**. Re-measure everything (Part B §4).
- Current test count is **78 backend (pytest) + 20 frontend (vitest)** — `CLAUDE.md` still says "77" in places (written before the model-path portability fix); confirm with a live run.

---

# PART A — THESIS OUTLINE

Organized around research questions, **not** the project's Phase 1–9 build history (that changelog → Appendix E). Per-chapter page budgets in Part B §7.

## Candidate Research Questions (state 3–4; lead with RQ4)

- **RQ1 — Achievable ceiling from public pre-game data.** How much NBA game-winner accuracy is achievable from *leakage-safe, publicly available pre-game features alone* (no betting-market inputs), and how large is the residual gap to the market ceiling?
- **RQ2 — Marginal value of feature families.** What is the incremental predictive contribution of each engineered family — Elo, rest/back-to-back, rolling box-score, travel & circadian, venue win-rate, head-to-head — over a majority-class baseline?
- **RQ3 — Calibration.** Does `CalibratedClassifierCV` (isotonic for RF/XGB, sigmoid for DT) measurably improve probability *reliability* (Brier / ECE / reliability diagrams), independent of accuracy?
- **RQ4 — Leakage discipline (primary, methodological).** What classes of data leakage arise in temporal sports-outcome prediction, and what construction-time and validation-time disciplines provably prevent them while keeping the model deployable on genuinely unseen future games?

## Candidate framings of the central contribution (lead with one, mention the others)

- **(a) Methodological case study [LEAD].** Rigorous, reproducible leakage-safe temporal feature construction + honest evaluation, motivated by a concrete documented near-miss.
- **(b) Empirical feature-ablation study [BACKBONE].** Controlled measurement of how much each engineered family contributes, vs. a majority baseline and a market reference.
- **(c) Reproducible production system / MLOps artifact [ENGINEERING].** End-to-end containerized system with a live data-freshness/retrain pipeline and prediction resolution — presented as engineering, explicitly *not* a research result.

## Chapters

**Front matter.** Title, declaration, acknowledgements, abstract, TOC, list of figures/tables, abbreviations (Elo, HCA, CV, ECE, B2B, WAL, ROC).

**Ch. 1 — Abstract** (≈1 p, written last). Problem; 4 models over 34 leakage-safe features; calibrated probabilities; honest `TimeSeriesSplit`; headline result (re-measured) vs. 55.6% baseline and 67–68% market ceiling; the methodological contribution + the leakage near-miss; one sentence on the engineering artifact.

**Ch. 2 — Introduction** (≈5 pp). 2.1 Motivation (basketball: high game count, dense schedule → rest/travel signal, mature Elo tradition; the market gap quantifies signal *absent* from public data). 2.2 Problem statement (binary `home_win`=1 / `away_win`=0 before tip-off, info strictly before `game_date`; tie→away-win, disclosed). 2.3 RQs. 2.4 Contributions. 2.5 Scope/non-goals (out: spreads/totals, player/injury, live in-game, deep learning, market inputs as features). 2.6 Structure.

**Ch. 3 — Background & Related Work** (≈10 pp). 3.1 Sports-outcome prediction (winner vs. spread/total; home advantage). 3.2 Elo / FiveThirtyEight (K-factor, logistic expected score, HCA, season regression → tie to K=20, HCA=100, 25% regression toward 1505). 3.3 ML for sports (RF, XGBoost; classical ML over DL on tabular modest-N data). 3.4 Calibration (Platt/sigmoid vs. isotonic; reliability diagrams, Brier, ECE). 3.5 Leakage in time-series ML (target leakage; post-game stats hazard; why random K-fold leaks future→past; `TimeSeriesSplit`). 3.6 Betting market as benchmark (~67–68% closing-line accuracy; a *literature* benchmark, not a head-to-head here — disclose). 3.7 Gap analysis.

**Ch. 4 — Data** (≈7 pp). 4.1 Sources (Kaggle-style `games.csv` 2003–2022; balldontlie live → `data/balldontlie_finals.csv`; live snapshot import id=2, 31,353 games, 2003→2026, 30 teams — verify at write). 4.2 Schema (`game_data`; team IDs as ints e.g. `1610612738`=Boston; optional box-score cols; `GAME_ID`→`season_type`). 4.3 EDA (class balance ≈ 55.6% home; games/season; home-win-rate over time; rest-day/B2B distribution; box-score coverage by season). 4.4 The leakage trap with post-game stats (worked example from `LEAKAGE_INVESTIGATION.md`: `FG_PCT_home > FG_PCT_away` trivially reveals the winner; foreshadow the ~95% fake-accuracy near-miss). 4.5 Data quality (29 dup `GAME_ID`s deduped on `(game_date,home,away)`; ties→away; unplayed skipped; 2022→2026 rows have blank box-score cols → mean-imputed; **validity threat, disclose**).

**Ch. 5 — Feature Engineering (methodological core)** (≈13 pp). 5.1 Design principle: strictly-before-game info only (`np.searchsorted(side='left')` over per-team date-sorted arrays — the invariant the chapter defends). 5.2 The 34 features by family (see Part C table). 5.3 The Elo sub-model in detail (K=20, HCA=100, 25% season regression toward 1505; forward pass with dedup; worth a figure). 5.4 Leakage-safe construction, proved (`_rolling_stat_mean`/`searchsorted` pattern; cold-start league-mean fallback; tests `test_rolling_stat_mean_does_not_leak_future`; vectorized training path vs. per-call inference path + the equivalence test `test_vectorized_and_per_call_paths_agree`). **5.5 Case study: the Chekhov's-gun leakage near-miss** (full treatment of `LEAKAGE_INVESTIGATION.md`: dead `_extract_statistical_features` path; `pd.json_normalize` on JSON *strings* silently returns 0 columns; the "obvious" `json.loads` fix → ~95% CV accuracy that collapses in prod; why this is *more* dangerous than the `score_difference` 100% leak — "great but plausible." **The thesis's signature evidence — give it room.**). 5.6 Removed features (`score_difference`, `total_score` = 100% leakage; constant `home_advantage` = zero info).

**Ch. 6 — Methodology / Models** (≈9 pp). 6.1 Models (DT, RF, XGB, Dummy baseline). 6.2 Calibration (`CalibratedClassifierCV` isotonic RF/XGB, sigmoid DT, none baseline; importance via `_averaged_feature_importances`). 6.3 Honest temporal evaluation (`TimeSeriesSplit(5)`; why random KFold is invalid; persisted cv mean/std). 6.4 Metrics (acc/P/R/F1; confusion convention `1=home_win`,`0=away_win`). 6.5 Confidence (`mean(max(predict_proba))`; baseline capped 0.99, `confidence_reliable=false`). 6.6 Persistence (`.pkl` + sidecar `.json`; DB registry).

**Ch. 7 — System Design & Implementation (engineering artifact — keep it under the research chapters)** (≈11 pp). 7.1 Architecture overview (+ diagram). 7.2 Database (SQLite+WAL; 5 tables; two migration paths). 7.3 REST API (endpoint catalogue; `X-API-Key`; rate limits; deep `/api/health`). 7.4 Frontend (page map; typed fetch; Recharts; lazy routes). 7.5 Data-freshness / live-retrain pipeline (the MLOps highlight; client → `FreshnessPipeline`; tier-agnostic; CLI vs. incremental endpoint). 7.6 Prediction resolution & leaderboards (handle-keyed; import-scoped). 7.7 Reproducibility & deployment (Docker; `.env`; `backup_db.py`). 7.8 Testing (78 backend + 20 frontend; leakage + equivalence regression tests).

**Ch. 8 — Experiments & Evaluation (empirical backbone)** (≈12 pp). **All numbers freshly re-measured (Part B §4).** 8.1 Setup (snapshot, temporal split, CV, versions, seeds). 8.2 Main results (per-model table + baseline). 8.3 CV-vs-test gap. 8.4 Calibration (RQ3 — reliability diagrams, Brier/ECE; *must run*). 8.5 Confusion matrices (real tp/tn/fp/fn). 8.6 Feature importance (`elo_diff` ~19% leads). 8.7 Feature-family ablation (RQ2 — leave-one-out / add-one-in; *must run*). 8.8 Baseline + market comparison (RQ1). 8.9 (Optional) per-season / playoff-vs-regular and clean-subset-vs-imputed breakdown.

**Ch. 9 — Discussion** (≈5 pp). Interpret vs. RQs: market gap → info content of public data; why Elo dominates; whether calibration paid off; the leakage lesson (honest CV "costs" paper accuracy but is the only number that transfers); classical-ML-vs-DL suitability.

**Ch. 10 — Limitations & Threats to Validity** (≈4 pp — don't skimp; this earns trust). Imputed box-score features (~5k recent games, 10/34 features mean-filled → report clean pre-2022 subset too); Vegas = literature benchmark not head-to-head; single league, no injuries/lineups; tie→away convention; engineering-vs-research boundary; temporal distribution shift; stale-baseline caution.

**Ch. 11 — Future Work** (≈2 pp). Acquire historical closing lines for a true market head-to-head; GOAT-tier box-score backfill to remove imputation; player/injury features; spread & total prediction; scheduled retrain; Postgres for multi-writer; sequence/DL benchmark; per-season recalibration.

**Ch. 12 — Conclusion** (≈2 pp). Restate methodological contribution, measured ceiling, verified artifact. No new info.

**Appendices.** A. Full 34-feature list (formula, source cols, leakage note each). B. API reference. C. DB schema + migration versions. D. Reproducibility (exact commands, snapshot row count/hash, versions, Docker). E. Development methodology / Phase 1–9 history (here, not the body). F. Test inventory.

---

# PART B — WRITING GUIDE FOR THE NEXT SESSION

## 1. Writing ORDER (inside-out, evidence-first — not front-to-back)

1. **Ch. 5 Feature Engineering + leakage case study (5.4–5.5) — FIRST** (the contribution; material already documented; crystallizes the argument).
2. **Ch. 6 Methodology / Models** (stable; from code).
3. **Ch. 4 Data** (pairs with the 5.5 leakage set-up).
4. **Ch. 8 Experiments — only after running §4 measurements** (no results prose against stale numbers).
5. **Ch. 7 System Design** (mostly transcription; low-cognitive-load session).
6. **Ch. 3 Background & Related Work** (once technical vocabulary is fixed).
7. **Ch. 9 → 10 → 11 → 12.**
8. **Ch. 2 Introduction, then Ch. 1 Abstract — LAST.**

## 2. CHAPTER → EVIDENCE map

| Chapter | Pull facts/figures from |
|---|---|
| 3 Background | External literature (§5); cross-check Elo params + market ceiling vs. `CLAUDE.md`. |
| 4 Data | `data/balldontlie_finals.csv`, the live DB (row count), `backend/data_importer.py`, `docs/LEAKAGE_INVESTIGATION.md` §1, `CLAUDE.md`. |
| 5 Feature Eng. | `backend/predictive_models.py` (feature builders), `backend/team_locations.py`, `docs/LEAKAGE_INVESTIGATION.md` (the whole near-miss), `CLAUDE.md` feature table. |
| 6 Methodology | `backend/predictive_models.py` (`CalibratedClassifierCV`, `TimeSeriesSplit`, `_averaged_feature_importances`, confidence), `CLAUDE.md`. |
| 7 System | `backend/api.py`, `database.py`, `balldontlie_client.py`, `data_freshness.py`, `backfill_freshness.py`, `front/src/`, `Dockerfile`/`docker-compose.yml`, `DEPLOY.md`, `docs/PHASE9_RESEARCH.md`. |
| 8 Experiments | **Freshly measured** outputs (§4) + `prediction_results` + `/api/verify`, `/api/leaderboard`, `/api/results`. |
| 9–12 | Own analysis; `CLAUDE.md` "Key decisions" + "Gotchas". |
| Appendices | `backend/api.py` (endpoints), `database.py` (schema), `backend/tests/`, `CLAUDE.md` Phase status (→ App. E). |

> **Routing rule:** root `README.md` is pre-rewrite (wrong on numbers/ports/architecture/page list). Cite only when discussing the project's own evolution.

## 3. FIGURES & TABLES (and where the data lives)

| # | Artifact | Type | Data source |
|---|---|---|---|
| F1 | System-architecture diagram | Hand-drawn | `backend/`, `front/` |
| F2 | Freshness pipeline data-flow (fetch→pivot→import→retrain→promote→resolve) | Hand-drawn | `data_freshness.py`, `balldontlie_client.py`, `PHASE9_RESEARCH.md` §7 |
| F3 | Leakage timeline (strictly-before window; post-game stats off-limits) | Hand-drawn | `LEAKAGE_INVESTIGATION.md`, `searchsorted(side='left')` |
| F4 | Elo trajectory for 1–2 teams across a season (season-boundary regression) | Line chart | Elo history pass in `predictive_models.py` |
| F5 | Feature-importance bar chart, grouped by family | Bar | `prediction_results.feature_importance` / `/api/results` |
| F6 | Reliability diagrams, calibrated vs. uncalibrated, per model | Calibration plot | **Run §4.3** |
| F7 | Confusion matrices per model | Heatmap | `/api/verify/{import_id}` |
| F8 | Home-win rate over time + class balance | Line/bar | `game_data` query |
| F9 | Box-score coverage by season (free-tier gap) | Bar | `game_data` (null FG_PCT by season) |
| T1 | Model comparison: acc / CV mean±std / P / R / F1 / confidence + baseline | Table | **Run §4.1**, `/api/leaderboard` |
| T2 | CV-vs-test accuracy | Table | **Run §4.1** |
| T3 | Feature-family ablation (ΔCV per family) | Table | **Run §4.4** |
| T4 | Full 34-feature list (formula, source, leakage note) | Table (App. A) | `predictive_models.py`, `CLAUDE.md` |
| T5 | Lift: model vs. majority baseline vs. Vegas literature ceiling | Table | §4.1 + literature |

## 4. EXPERIMENTS / MEASUREMENTS to run BEFORE Chapter 8

**Hard rule: every doc number is stale (old 26,552-game set; live DB is now ~31,353 games through 2026). Re-measure on the current dataset; transcribe nothing. Do this in a dedicated measurement session.**

1. Re-train + re-measure all four models on import `id=2`: test accuracy, CV mean±std, P/R/F1, confidence → T1, T2, T5.
2. Real confusion matrices per model via `/api/verify/{import_id}` → F7.
3. **Calibration evidence (new):** reliability diagrams + Brier and/or ECE, calibrated vs. uncalibrated (sklearn `calibration_curve`, `brier_score_loss`) → F6, §8.4.
4. **Feature-family ablation (new):** leave-one-family-out and add-one-family-in over the 6 families (Elo, rest, rolling box-score, travel/circadian, venue, H2H), ΔCV-accuracy → T3.
5. **Clean-subset vs. full-set:** metrics on the pre-2022 clean-box-score subset vs. the full imputed set; report both (neutralizes the imputation threat) → §8.9.
6. Per-season + playoff-vs-regular breakdown (optional, strong) → §8.9, F8.
7. Confirm current majority-baseline accuracy + state lift; cite (don't measure) the ~67–68% market ceiling.
8. Confirm test counts from a live `pytest` + `vitest` run (currently 78 + 20).

## 5. CITATIONS to gather (concrete starting points)

- **Elo / FiveThirtyEight.** Elo, *The Rating of Chessplayers* (1978); FiveThirtyEight "How We Calculate NBA Elo Ratings" (Silver/Paine); Glickman on rating systems.
- **Tree ensembles / boosting.** Breiman, "Random Forests" (2001); Chen & Guestrin, "XGBoost" (KDD 2016).
- **scikit-learn.** Pedregosa et al. (JMLR 2011).
- **Calibration.** Platt (1999); Zadrozny & Elkan (2002, isotonic); Niculescu-Mizil & Caruana (2005); Guo et al. (2017, ECE); Brier (1950).
- **Time-series CV / leakage.** Bergmeir & Benítez (2012); Kaufman et al. (2012, "Leakage in Data Mining"); scikit-learn `TimeSeriesSplit` docs.
- **Sports prediction / market efficiency.** Representative NBA/NFL ML-prediction papers; closing-line-efficiency literature.
- **Data sources.** balldontlie API docs; the games dataset provenance (Kaggle); haversine reference.

Use a reference manager (Zotero/BibTeX); cite exact scikit-learn / XGBoost / pandas versions measured with (App. D).

## 6. Academic conventions & tone

- Past tense for what you did; present for established facts.
- **No marketing language** ("state-of-the-art", "powerful", "professional analytics" — all over the stale README; keep it out).
- Hedge: "achieves ~64% on a temporal test split", never "predicts NBA games" unqualified; report ±std/CI.
- Define before use; one symbol per concept (`home_win`=1 always).
- Figures/tables referenced from text with self-contained captions.
- Separate claim from evidence (method ≠ result).
- Demarcate engineering from research in prose.
- Reproducibility statement near every result (split, seed, snapshot).

## 7. Page budget & milestones (~71 pp body; scale to your institution)

| Chapter | Pages | | Chapter | Pages |
|---|---|---|---|---|
| 1 Abstract | 1 | | 7 System Design | 11 |
| 2 Introduction | 5 | | **8 Experiments** | **12** |
| 3 Background | 10 | | 9 Discussion | 5 |
| 4 Data | 7 | | 10 Limitations | 4 |
| **5 Feature Eng.** | **13** | | 11 Future Work | 2 |
| 6 Methodology | 9 | | 12 Conclusion | 2 |

Deliberate weighting: Feature Eng. + Methodology + Evaluation ≈ half the body; System Design capped ~11 pp. Guard against the implementation chapter swelling past the research chapters.

**Milestones.** M0: measurement session (run all §4; save every figure/table). *Gate: no Ch. 8 prose until done.* M1: Ch. 5 (+ leakage case study) + Ch. 6. M2: Ch. 4 + Ch. 8. M3: Ch. 7 + Ch. 3. M4: Ch. 9–12, then 2, then 1. M5: appendices, figures, references, full read-through, supervisor review.

## 8. Pitfalls specific to THIS project

1. **Don't overclaim accuracy.** ~64%/66% is *below* the market ceiling. Frame as "achievable ceiling from public pre-game features"; always show next to the 55.6% baseline (lift) *and* the 67–68% ceiling (honesty).
2. **Get the leakage story exactly right** — it's the thesis. `searchsorted(side='left')` = strictly-before; the `json.loads` "fix" is a *trap*; ~95% fake accuracy is *worse* than useless because it's plausible. Cite the actual tests.
3. **Never transcribe doc numbers** — re-measure (§4). "64.2%/66.0%" and "26,552 games" are historical.
4. **Disclose the imputed box-score features** — report the clean-subset comparison.
5. **Keep Vegas a literature benchmark** — no head-to-head table implying you evaluated closing lines on your own games; put a real head-to-head in Future Work.
6. **Separate engineering from research** everywhere — the freshness pipeline, leaderboards, Docker, ~98 tests are excellent *artifacts*; the *contribution* is methodological.
7. **Don't structure as Phases 1–9** — that's a changelog → Appendix E; the body is by RQ.
8. **Reconcile the test count** (77 vs. 78 in docs) from a live run.

---

# PART C — TECHNICAL EVIDENCE INVENTORY

Dense `file:line` map so the writer can pull facts fast. (Line numbers approximate — confirm at write time.)

## C.1 The 34 features → implementing functions (`backend/predictive_models.py` unless noted)

| Group | # | Features | Implementation |
|---|---|---|---|
| Categorical (encoded) | 3 | `home_team_encoded`, `away_team_encoded`, `season_encoded` | `prepare_features()` (~83–107), LabelEncoder on unique teams/seasons |
| Calendar | 3 | `month`, `day_of_week`, `day_of_year` | `prepare_features()` (~95–99) |
| Rolling form | 3 | `home/away_team_last5_winrate`, `h2h_home_wins_rate` | `_rolling_winrate()` (~232), `_h2h_winrate()` (~255), `_compute_h2h_vectorized()` (~537) |
| Elo (538-style) | 3 | `home_elo_pre`, `away_elo_pre`, `elo_diff` | `_compute_elo_history()` (~305–370); K=20, HCA=100, init 1500, 25% season regression toward 1505 |
| Rest & B2B | 5 | `home/away_rest_days`, `home/away_b2b`, `rest_diff` | `_days_since_last()` (~391), vectorized in `_precompute_team_pregame_features()` |
| Venue win-rate | 2 | `home_team_home_winrate`, `away_team_away_winrate` | `_rolling_venue_winrate()` (~403), vectorized (~495–510) |
| Rolling box-score (L10) | 10 | `home/away_{fg_pct,ft_pct,fg3_pct,ast,reb}_l10` | `_rolling_stat_mean()` (~424–444), vectorized shift+roll (~512–522); league-mean cold-start |
| Travel & circadian | 4 | `home/away_travel_dist`, `home/away_tz_shift` | `_compute_travel_tz_vectorized()` (~590–638); `team_locations.py` (30 arenas) |
| Season type | 1 | `is_playoff` | `data_importer._clean_data()` (~238–241), GAME_ID leading digit |

Orchestrator: `_add_historical_features()` (~640–715). History index: `_build_history_index()` (~140–230, dedups on `(game_date,home,away)`). **Leakage invariant:** `np.searchsorted(side='left')` at lines ~243, 274, 396, 418, 436, 503, 556.

## C.2 ML pipeline

- `_prepare_split()` (~789–818): chronological full set for CV; temporal 80/20 if `use_temporal_split=True`, else random stratified (`random_state=42`).
- `_train_and_eval()` (~868–966): `CalibratedClassifierCV(method=…, cv=5)` (~888); test metrics; `TimeSeriesSplit(5)` CV if ≥100 rows (~911–925); `_averaged_feature_importances()` (~849–866); persist `.pkl`+`.json` (~721–749).
- Model trainers: `train_decision_tree/random_forest/xgboost/baseline` (~981–1041).
- `predict_single()` (~1052–1242): server-side feature build; **`load_model()` (~751) resolves model_path by basename under `MODELS_DIR` if the stored absolute path is missing — the Docker-portability fix.**
- Calibration: isotonic (RF/XGB), sigmoid (DT), none (baseline). Confidence = `mean(max(predict_proba))`, baseline capped 0.99. Label convention `1=home_win`, `0=away_win`.

## C.3 REST API (`backend/api.py`) — endpoints

`/api/health` (DB ping, 503 on fail) · `/api/imports` (GET) · `/api/imports/upload` (POST, auth, 20/h) · `/api/imports/{id}` (DELETE, auth, 30/h, cascades) · `/api/imports/{id}/teams` · `/api/imports/{id}/preview` · `/api/models` · `/api/train` (POST, auth, 5/h) · `/api/results` · `/api/leaderboard` (sort acc/f1/cv, `import_id`-scoped) · `/api/predict` (POST, auth, 60/min) · `/api/predictions` (POST auth 60/min; GET) · `/api/user-leaderboard` · `/api/freshness/status` · `/api/freshness/sync` (POST, auth, 5/h, `max_games ≤ 250`) · `/api/verify/{id}` (real tp/tn/fp/fn). Auth = `X-API-Key` (`NBA_API_KEY`); slowapi per-IP (proxy caveat).

## C.4 Data / DB / freshness

- DB `backend/database.py`: tables `import_records`, `game_data`, `models`, `prediction_results`, `predictions`(v1). WAL (~62); legacy `_migrate()` ALTER try/except + versioned `_apply_migrations()` via `PRAGMA user_version`. Dedup-append (`append_game_data` ~307–334), `resolve_predictions`, `ping()`.
- `balldontlie_client.py`: cursor pagination (`iter_finals`), 13s throttle (free 5/min), escalating 429 backoff (20→40→60→75s, ≤8 retries), `iter_box_scores` (GOAT-only → free key degrades).
- `data_freshness.py` `FreshnessPipeline`: `map_game()` (abbr→ID via `team_locations.ABBR_TO_ID`), `ingest()`, `sync()` (ingest→resolve→optional retrain→promote).
- CLIs: `backfill_freshness.py` (DB ingest), `export_balldontlie_csv.py` (clone games.csv → balldontlie_finals.csv), `backup_db.py` (`VACUUM INTO`).
- `data_importer.py`: `validate_csv()`, column mapping, `season_type` from GAME_ID, ties→away, dedup note (26,622 unique / 26,651 rows).

## C.5 Reference numbers (HISTORICAL — 26,552-game set; re-measure)

| Metric | Value |
|---|---|
| RF test accuracy (temporal 80/20) | 64.2% |
| RF TimeSeriesSplit CV accuracy | 66.0% (±~0.5%) |
| Majority baseline | 55.6% |
| Lift over baseline | +8.6 pp |
| RF F1 | 0.626 |
| Vegas closing-line ceiling (literature) | ~67–68% |

`elo_diff` ≈ 19% RF importance (top feature); each FG%-L10 ≈ 3–3.5%.

## C.6 Tests (regression evidence)

Backend pytest (**78 tests**, tmp-DB isolation): leakage — `test_rolling_stat_mean_does_not_leak_future`, `test_elo_pre_does_not_use_future_games`, `test_rolling_winrate_does_not_leak_future`; equivalence — `test_vectorized_and_per_call_paths_agree`; leaderboard import-scoping + reorder; model-path portability fallback; CSV validation (`test_validate_csv.py`); `conftest.py` (sample 4-team/2-season/24-game DF, per-test MODELS_DIR isolation). Frontend vitest (**20 tests**): API error handling, fetch mocking, types.

## C.7 Gaps / Threats to Validity (for Ch. 10 / 11)

Handle-based identity (unverified, spoofable); SQLite single-writer scaling (→ Postgres trigger: user-based leaderboard / multi-replica); free-tier box-score gap (10/34 features mean-imputed for ~5k 2022→2026 games — quantify clean-subset delta); rate-limiter-behind-proxy (per-IP → global); `VITE_API_BASE_URL` baked at build; single data source (no independent test set); **reference numbers measured on old 26.5K set, not current 31.3K**; team-level only (no player/injury features); implicit 2003-forward scope.

## C.8 Evidence index (the ~15 most thesis-relevant files)

`backend/predictive_models.py` (whole ML pipeline) · `backend/api.py` (endpoints/auth/verify) · `backend/database.py` (schema/WAL/migrations) · `backend/data_importer.py` (validation/cleaning) · `backend/balldontlie_client.py` (ingest client) · `backend/data_freshness.py` (pipeline) · `backend/team_locations.py` (arenas/ABBR_TO_ID) · `backend/tests/test_predictive_models.py` (leakage + equivalence tests) · `backend/tests/conftest.py` (fixtures) · `docs/LEAKAGE_INVESTIGATION.md` (**the near-miss case study**) · `docs/PHASE9_RESEARCH.md` (design decisions) · `docker-compose.yml` + `backend/Dockerfile` · `DEPLOY.md` · `front/src/lib/api.ts` + `front/src/pages/Leaderboard.tsx` · `CLAUDE.md` (architecture + history).
