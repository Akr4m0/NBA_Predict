#!/usr/bin/env python3
"""
One-time / periodic data-freshness backfill via balldontlie.

The free tier is rate-limited to 5 requests/minute, so a multi-season backfill
takes minutes — run it here as a CLI rather than through the (timeout-bound,
5/hour) HTTP endpoint. The endpoint is for small incremental syncs.

Examples:
    cd backend
    # Fill the 2022 -> now gap and retrain the random forest afterwards:
    python3 backfill_freshness.py --seasons 2022 2023 2024 2025 --retrain random_forest

    # Quick smoke (15 games, no retrain), into a specific import:
    python3 backfill_freshness.py --seasons 2024 --max-games 15 --import-id 3

Reads BALLDONTLIE_API_KEY from the environment or the project-root .env.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load project-root .env for BALLDONTLIE_API_KEY (real env vars win)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill recent NBA finals via balldontlie.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="Season starting years, e.g. 2022 2023 2024 2025")
    parser.add_argument("--import-id", type=int, default=None,
                        help="Target import to extend (default: latest)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="YYYY-MM-DD server-side filter (incremental syncs)")
    parser.add_argument("--retrain", type=str, nargs="*", default=None,
                        help="Model types to retrain after ingest, e.g. random_forest")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Cap fetched games (smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv()

    from database import NBADatabase
    from data_importer import DataImporter
    from predictive_models import PredictiveModels
    from data_freshness import FreshnessPipeline

    db_path = os.environ.get(
        "NBA_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "nba_predictions.db"),
    )
    db = NBADatabase(db_path=db_path)
    pipe = FreshnessPipeline(db=db, importer=DataImporter(db), pm=PredictiveModels(db=db))

    if not pipe.configured:
        raise SystemExit("BALLDONTLIE_API_KEY is not set (env or .env). Aborting.")

    logging.info("Syncing seasons=%s import_id=%s retrain=%s ...",
                 args.seasons, args.import_id, args.retrain)
    result = pipe.sync(
        seasons=args.seasons,
        target_import_id=args.import_id,
        start_date=args.start_date,
        retrain=args.retrain,
        max_games=args.max_games,
    )
    logging.info("Done: fetched=%d inserted=%d import_id=%s resolved_predictions=%d",
                 result["fetched"], result["inserted"], result["import_id"],
                 result["resolved_predictions"])
    for r in result.get("retrained", []):
        if "error" in r:
            logging.warning("  retrain %s FAILED: %s", r["model_type"], r["error"])
        else:
            logging.info("  retrained %s -> model_id=%s accuracy=%s",
                         r["model_type"], r["model_id"], r.get("accuracy"))


if __name__ == "__main__":
    main()
