#!/usr/bin/env python3
"""
Clone games.csv and append recent balldontlie finals → data/balldontlie_finals.csv.

Produces a single CSV in the exact games.csv schema: every original row is kept
verbatim, and newer finals fetched from balldontlie are appended (deduped against
what's already there). The source games.csv is never modified.

Free-tier balldontlie gives scores only, so the 11 box-score columns
(FG_PCT_home, AST_home, REB_home, … + away) are left blank for the new rows —
the importer accepts that, and the box-score features mean-fallback for them.

Examples:
    cd backend
    python3 export_balldontlie_csv.py --seasons 2022 2023 2024 2025
    python3 export_balldontlie_csv.py --seasons 2024 --max-games 50   # quick test

Reads BALLDONTLIE_API_KEY from the environment or the project-root .env.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

# games.csv column order (must match exactly).
GAMES_CSV_COLUMNS = [
    "game_date", "GAME_ID", "GAME_STATUS_TEXT", "home_team", "away_team", "SEASON",
    "TEAM_ID_home", "home_score", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home",
    "AST_home", "REB_home", "TEAM_ID_away", "away_score", "FG_PCT_away", "FT_PCT_away",
    "FG3_PCT_away", "AST_away", "REB_away", "HOME_TEAM_WINS",
]
_BLANK_BOXSCORE = [
    "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",
]

_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_path = _ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _dedup_key(game_date, home, away) -> tuple:
    """Normalise a (date, home, away) triple to a comparable key. Handles both
    M/D/YYYY (games.csv) and YYYY-MM-DD (balldontlie) dates."""
    ts = pd.to_datetime(str(game_date), errors="coerce")
    date_str = ts.strftime("%Y-%m-%d") if pd.notna(ts) else str(game_date)
    return (date_str, str(home).strip(), str(away).strip())


def _game_to_row(game: dict, abbr_to_team_id) -> dict | None:
    home = game.get("home_team") or {}
    away = game.get("visitor_team") or {}
    home_id = abbr_to_team_id(home.get("abbreviation"))
    away_id = abbr_to_team_id(away.get("abbreviation"))
    if not home_id or not away_id:
        return None
    hs, as_ = game.get("home_team_score"), game.get("visitor_team_score")
    if hs is None or as_ is None:
        return None
    ts = pd.to_datetime(str(game.get("date")), errors="coerce")
    if pd.isna(ts):
        return None
    date_str = f"{ts.month}/{ts.day}/{ts.year}"          # M/D/YYYY like games.csv
    lead = "4" if game.get("postseason") else "2"        # season_type leading digit
    row = {c: "" for c in GAMES_CSV_COLUMNS}
    row.update({
        "game_date": date_str,
        "GAME_ID": f"{lead}{game.get('id')}",
        "GAME_STATUS_TEXT": "Final",
        "home_team": home_id,
        "away_team": away_id,
        "SEASON": str(game.get("season")),
        "TEAM_ID_home": home_id,
        "home_score": int(hs),
        "TEAM_ID_away": away_id,
        "away_score": int(as_),
        "HOME_TEAM_WINS": 1 if int(hs) > int(as_) else 0,
    })
    # _BLANK_BOXSCORE columns stay "" (free tier has no box scores).
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone games.csv + append balldontlie finals.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=str, default=str(_ROOT / "data" / "games.csv"))
    parser.add_argument("--out", type=str, default=str(_ROOT / "data" / "balldontlie_finals.csv"))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv()

    from balldontlie_client import BalldontlieClient
    from team_locations import abbr_to_team_id

    # 1. Clone the source (read every column as a string so nothing is reformatted).
    src = pd.read_csv(args.source, dtype=str, keep_default_na=False)
    logging.info("Loaded source %s: %d rows", args.source, len(src))
    seen = {
        _dedup_key(r["game_date"], r["home_team"], r["away_team"])
        for _, r in src.iterrows()
    }

    # 2. Fetch + map newer finals, skipping anything already present.
    client = BalldontlieClient()
    if not client.configured:
        raise SystemExit("BALLDONTLIE_API_KEY is not set (env or .env). Aborting.")

    new_rows = []
    fetched = 0
    for game in client.iter_finals(args.seasons, start_date=args.start_date, max_games=args.max_games):
        fetched += 1
        row = _game_to_row(game, abbr_to_team_id)
        if row is None:
            continue
        key = _dedup_key(row["game_date"], row["home_team"], row["away_team"])
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(row)

    logging.info("Fetched %d finals; %d are new (not already in source).", fetched, len(new_rows))

    # 3. Concatenate (original rows verbatim + new rows) and write.
    if new_rows:
        combined = pd.concat([src, pd.DataFrame(new_rows, columns=GAMES_CSV_COLUMNS)],
                             ignore_index=True)
    else:
        combined = src
    combined.to_csv(args.out, index=False)
    logging.info("Wrote %s: %d rows (%d original + %d new).",
                 args.out, len(combined), len(src), len(new_rows))


if __name__ == "__main__":
    main()
