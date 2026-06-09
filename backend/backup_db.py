#!/usr/bin/env python3
"""
Back up the SQLite database to a timestamped file.

Uses `VACUUM INTO`, which writes a single consistent copy (committed WAL data
included) without needing the app to be stopped. Deliberately a CLI, not an
HTTP endpoint — an endpoint that dumps the whole DB is a data-exfiltration
surface.

Examples:
    cd backend
    python3 backup_db.py                      # -> data/nba_predictions.db.backup-<ts>
    python3 backup_db.py --out /backups/nba.db

Honors NBA_DB_PATH (same as the app); falls back to data/nba_predictions.db.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up the NBA prediction SQLite DB.")
    parser.add_argument(
        "--db",
        default=os.environ.get("NBA_DB_PATH", str(_ROOT / "data" / "nba_predictions.db")),
        help="Path to the SQLite DB (default: NBA_DB_PATH or data/nba_predictions.db)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Destination path (default: <db>.backup-YYYYmmdd_HHMMSS)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else db_path.with_name(f"{db_path.name}.backup-{ts}")
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {out}")

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("VACUUM INTO ?", (str(out),))

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Backed up {db_path} -> {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
