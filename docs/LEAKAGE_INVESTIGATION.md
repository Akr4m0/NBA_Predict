# Per-Game Stats — Dead Code & Latent Leakage Trap

**Date investigated:** 2026-05-19
**Status:** Investigated, recommendation pending.
**TL;DR:** The codebase has a feature-engineering pathway for per-game team box-score stats (`FG_PCT_home`, `AST_home`, `REB_home`, etc.) that **silently does nothing** today. The data flows from CSV into the SQLite DB but never makes it into the feature matrix because of a `pd.json_normalize` bug. Current 65.6% CV accuracy is legitimate. **However**, a future "fix" would introduce massive (~95%+) leakage because those columns are post-game outcomes, not pre-game features.

---

## 1. Background — what's in `games.csv`

The CSV has 21 columns. The relevant ones for this investigation:

| Column | Type | When known | Risk |
|---|---|---|---|
| `home_team`, `away_team` | id | Pre-game | Safe |
| `game_date`, `SEASON` | date | Pre-game | Safe |
| `home_score`, `away_score` | int | **Post-game** | Removed already (`score_difference`, `total_score` were the leakage features Phase 0 deleted) |
| `FG_PCT_home`, `FG_PCT_away` | float | **Post-game** | The stats this doc is about |
| `FT_PCT_home`, `FT_PCT_away` | float | **Post-game** | "" |
| `FG3_PCT_home`, `FG3_PCT_away` | float | **Post-game** | "" |
| `AST_home`, `AST_away` | int | **Post-game** | "" |
| `REB_home`, `REB_away` | int | **Post-game** | "" |
| `HOME_TEAM_WINS` | bool | **Post-game** | This is the **target**, not a feature |

Every `*_home` / `*_away` stat is the actual recorded box score from the game being predicted. **If used as a feature, the model would already know who won.**

Sample row (12/22/2022, Pelicans @ Spurs):
```
game_date,home_team,away_team,home_score,FG_PCT_home,AST_home,REB_home,away_score,FG_PCT_away,AST_away,REB_away,HOME_TEAM_WINS
12/22/2022,1610612740,1610612759,126,0.484,25,46,117,0.478,23,44,1
```

A model that sees `FG_PCT_home=0.484, FG_PCT_away=0.478` can trivially predict home win (0.484 > 0.478) with very high accuracy.

---

## 2. The intended pipeline

There are two halves:

### Half A — Importer packs stats into JSON (`backend/data_importer.py:218-246`)

In `_clean_data`:

```python
# data_importer.py:218-223
stat_columns = [col for col in df.columns if any(stat in col.lower()
               for stat in ['fg%', 'fga', 'fgm', '3p', 'ft', 'reb', 'ast', 'stl', 'blk', 'to'])]

if stat_columns:
    df = self._extract_team_stats(df, stat_columns)
```

In `_extract_team_stats`:

```python
# data_importer.py:227-246
home_stat_cols = [col for col in stat_columns if 'home' in col.lower()]
away_stat_cols = [col for col in stat_columns if 'away' in col.lower() or 'visitor' in col.lower()]

if home_stat_cols:
    df.loc[:, 'home_stats'] = df[home_stat_cols].to_dict('records')
else:
    df.loc[:, 'home_stats'] = [{}] * len(df)

if away_stat_cols:
    df.loc[:, 'away_stats'] = df[away_stat_cols].to_dict('records')
else:
    df.loc[:, 'away_stats'] = [{}] * len(df)
```

For each row, `to_dict('records')` produces a dict like `{"FT_PCT_home": 0.926, "AST_home": 25.0, "REB_home": 46.0}`. That dict goes into the `home_stats` cell.

### Half A.1 — DB persists as JSON strings (`backend/database.py:69-70, 153-164`)

```python
# database.py:69-70 (schema)
home_stats TEXT,
away_stats TEXT,

# database.py:163-164 (save)
json.dumps(row.get('home_stats', {})),
json.dumps(row.get('away_stats', {})),
```

`json.dumps()` turns the dict into a string like `'{"FT_PCT_home": 0.926, "AST_home": 25.0, "REB_home": 46.0}'`. SQLite stores that string.

Verified in the live DB:

```
home_stats row 0: '{"FT_PCT_home": 0.926, "AST_home": 25.0, "REB_home": 46.0}'
home_stats row 1: '{"FT_PCT_home": 0.952, "AST_home": 16.0, "REB_home": 40.0}'
home_stats row 2: '{"FT_PCT_home": 0.786, "AST_home": 22.0, "REB_home": 37.0}'
```

### Half B — Trainer tries to unpack into features (`backend/predictive_models.py:123-137`)

```python
def _extract_statistical_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if 'home_stats' in df.columns:
        home_stats_df = pd.json_normalize(df['home_stats'])
        home_stats_df.columns = ['home_' + c for c in home_stats_df.columns]
        features = pd.concat([features, home_stats_df], axis=1)

    if 'away_stats' in df.columns:
        away_stats_df = pd.json_normalize(df['away_stats'])
        away_stats_df.columns = ['away_' + c for c in away_stats_df.columns]
        features = pd.concat([features, away_stats_df], axis=1)

    return features
```

Called once from `prepare_features` at line 94.

The **intended** outcome: 6 new feature columns added to the feature matrix:
- `home_FT_PCT_home`, `home_AST_home`, `home_REB_home`
- `away_FT_PCT_away`, `away_AST_away`, `away_REB_away`

---

## 3. The actual behavior

`pd.json_normalize()` expects an **iterable of dicts**, not strings. When given a `Series` of JSON strings, it silently returns an **empty DataFrame** instead of parsing them.

### Empirical proof

```python
result = pd.json_normalize(df['home_stats'].head(3))
# Shape: (3, 0)
# Columns: []
```

So `_extract_statistical_features` adds **zero columns**. The `pd.concat` with an empty-column DataFrame is a no-op.

### Verified via feature count

After a Random Forest training run with the full pipeline, `feature_names` contains exactly **19 features**:

```
home_team_encoded, away_team_encoded,
month, day_of_week, day_of_year,
season_encoded,
home_team_last5_winrate, away_team_last5_winrate, h2h_home_wins_rate,
home_elo_pre, away_elo_pre, elo_diff,
home_rest_days, away_rest_days, home_b2b, away_b2b, rest_diff,
home_team_home_winrate, away_team_away_winrate
```

Breakdown: 2 (teams) + 3 (date) + 1 (season) + 13 (historical/Elo/rest/venue) = 19. **Zero from `_extract_statistical_features`.** If it were working, we'd see 25 features.

---

## 4. Why it's a Chekhov's gun

The trap: the code **looks** like it should work. A reasonable engineer reading this might:

1. Run training, notice the stats aren't in `feature_importances_`.
2. Add a print statement, see `_extract_statistical_features` returning empty.
3. "Fix" it with one line: `df['home_stats'] = df['home_stats'].apply(json.loads)` before the call.
4. Run training. **Now accuracy jumps to 95%+.**
5. Celebrate. Ship.

That model would be entirely useless on new games (where you don't know post-game FG_PCT) but appear amazing on cross-validation. This is the worst kind of leakage — silent, invisible to the engineer who introduced it, and devastating in production.

The `score_difference` / `total_score` leakage was caught early because it was 100% accuracy (obviously wrong). A `FG_PCT_home` based model gets ~95%, which looks "great but plausible" — exactly the danger zone.

---

## 5. What's wrong about it — itemized

1. **Half-built feature with no working path to the model.** Dead code by definition.
2. **It stores ~50 KB of useless JSON per 5000 rows in the DB.** Trivial but real.
3. **The unused stats are exactly the columns that would be most dangerous if accidentally enabled.** Of all the things to leave half-wired, this is the worst.
4. **The substring filter on line 219** (`['fg%', 'fga', 'fgm', '3p', 'ft', 'reb', 'ast', 'stl', 'blk', 'to']`) is itself buggy. `FG_PCT_home` doesn't match `fg%` (no percent sign) or `fga`/`fgm` — so it's silently dropped. `FG3_PCT_home` doesn't match `3p` because the substring is `3_p` not `3p`. So out of 10 stat columns in the CSV, only 3 are even being picked up (`AST_*`, `REB_*`, `FT_PCT_*`). This makes the latent landmine selective rather than total — but the principle stands.
5. **`pd.json_normalize` on strings is silent.** It doesn't warn, doesn't error, doesn't log. The bug is invisible at runtime.

---

## 6. Risk analysis — removing the code

### Code dependency check

Full grep across `backend/`, `front/`, docs:

| Site | File:line | Action if removed |
|---|---|---|
| Schema columns | `database.py:69-70` | Keep columns (data exists; old imports work). New imports will write `'{}'` going forward. Or migrate to drop columns — riskier. |
| Save INSERT | `database.py:153-164` | If we keep schema, leave alone. If we drop the columns, must update. |
| Importer packing | `data_importer.py:218-225, 227-246` | Remove. No external caller. |
| Trainer extraction | `predictive_models.py:94, 123-137` | Remove. No external caller. |
| CLAUDE.md note | `CLAUDE.md:149` | Update to reflect the removal. |
| Frontend | (none) | No risk. |

**No external callers.** Both `_extract_team_stats` and `_extract_statistical_features` are private (underscore-prefixed) and only called from within their own files.

### Behavioral risk

- **Training output**: identical. Same 19 features today, same 19 features after removal. Same 65.6% CV accuracy.
- **DB rows already saved**: untouched. The `home_stats` / `away_stats` columns still hold their JSON strings; nothing reads them.
- **New imports**: will save `'{}'` (empty JSON) into those columns going forward, which is fine.
- **Loading old persisted models**: unaffected — the model pickles store their own `feature_names` list, which never included the stats.

### Schema risk

If you choose to also drop the `home_stats` / `away_stats` columns from the table:
- SQLite supports `ALTER TABLE DROP COLUMN` since 3.35.0 (2021). Likely supported on your machine, but check.
- Adds a one-way migration. Existing DB files still have the columns; deleting them is destructive.
- Recommendation: **leave the columns**, just stop writing meaningful data to them. Reversibility costs almost nothing.

### Migration / rollback risk

Pure removal of the importer and trainer code is fully reversible by `git revert`. No data is destroyed. No models break.

---

## 7. What replaces it

Three different framings of "replace," depending on what you wanted from the stats:

### Replacement A — for the dead code itself
**Nothing.** The code did nothing observable; removing it changes nothing observable. There's no replacement to write.

### Replacement B — for "encode team strength"
Already done in Phase 1. The features:

- `home_elo_pre`, `away_elo_pre`, `elo_diff` — rolling Elo (most predictive)
- `home_team_last5_winrate`, `away_team_last5_winrate` — recent form
- `home_team_home_winrate`, `away_team_away_winrate` — venue-specific form
- `h2h_home_wins_rate` — head-to-head history

These capture team strength **leakage-safely** (strict `< game_date` lookups) and are the reason CV accuracy lifted from 61.7% → 65.6%.

### Replacement C — if you actually want box-score-style features
The honest version of "use AST/REB/FG%" is **rolling pre-game averages** of those box-score stats. Example: "home team's average AST in their last 10 games before this one." That's leakage-safe and adds real signal.

To do this:

```python
# Pseudocode for a new helper in predictive_models.py:
def _rolling_stat(self, team: str, before_date, stat: str, n: int = 10) -> float:
    """
    Average value of `stat` for `team` across its last `n` games strictly before `before_date`.
    Cold-start: league mean for that stat.
    """
    ...
```

Features to add (10 per side = 20 total):
- `home_fg_pct_l10`, `home_ft_pct_l10`, `home_3pt_pct_l10`, `home_ast_l10`, `home_reb_l10` (and away)

Expected lift: ~+1 to +2pp on top of current 65.6% CV accuracy. Effort: ~2 hours.

This is the natural Phase-5 follow-on if you want to push past 67% toward the Vegas line (~68%). Note: this requires the DB to actually have those columns. Today it discards `FG_PCT_*` (substring filter doesn't match). So Replacement C also requires fixing the substring filter — or just reading the raw CSV columns directly.

---

## 8. Recommended action

Remove the dead code now. Specifically:

1. **`backend/data_importer.py`** — Delete lines 218-225 (the `stat_columns` block in `_clean_data` and the call to `_extract_team_stats`) and the entire `_extract_team_stats` method (lines 227-246).
2. **`backend/predictive_models.py`** — Delete the `_extract_statistical_features` call in `prepare_features` (line 94) and the entire method (lines 123-137).
3. **`backend/database.py`** — Leave schema and save logic alone. New imports will write `json.dumps({})` = `'{}'` for both stats columns, which is harmless.
4. **`CLAUDE.md:149`** — Update the gotcha bullet to say "Stat-extraction code was removed in 2026-05-19; see `docs/LEAKAGE_INVESTIGATION.md` for why."
5. **Optional**: replace with Replacement C if you want the +1-2pp accuracy push.

Total: ~50 lines of code deleted, zero behavior change, eliminates the Chekhov's gun. Fully reversible via `git revert`.

---

## 9. Appendix — exact file:line index

```
backend/database.py:69          home_stats TEXT,
backend/database.py:70          away_stats TEXT,
backend/database.py:153         game_date, season, home_stats, away_stats, result)
backend/database.py:163         json.dumps(row.get('home_stats', {})),
backend/database.py:164         json.dumps(row.get('away_stats', {})),

backend/data_importer.py:218-225   stat_columns filter in _clean_data
backend/data_importer.py:227-246   _extract_team_stats method body

backend/predictive_models.py:94    _extract_statistical_features call site
backend/predictive_models.py:123-137  _extract_statistical_features method body

CLAUDE.md:149                       gotcha note (to be updated)
```

No frontend references. No tests reference these. No external callers.
