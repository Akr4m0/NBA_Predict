import sqlite3
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

# Versioned schema migrations applied after the base CREATE TABLE statements.
# Each entry is (version, [sql, ...]); statements run in order when the DB's
# PRAGMA user_version is below `version`, then user_version is bumped to it.
# This is the path for all schema changes going forward — the legacy
# ALTER-TABLE/try-except block in _migrate() is frozen for backward compat.
SCHEMA_MIGRATIONS = [
    (
        1,
        [
            # Per-user saved predictions (handle-keyed, honor-system MVP).
            # game_date/resolved/actual_label/correct support resolving a
            # saved pick against a real final later (balldontlie pipeline).
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_handle TEXT NOT NULL,
                model_id INTEGER,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                season TEXT,
                predicted_label TEXT NOT NULL,
                predicted_confidence REAL,
                model_label TEXT,
                game_date DATE,
                resolved INTEGER NOT NULL DEFAULT 0,
                actual_label TEXT,
                correct INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_handle)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)",
        ],
    ),
]


class NBADatabase:
    def __init__(self, db_path: str = "nba_predictions.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Write-Ahead Logging: lets readers run concurrently with a writer
            # (resolution job / training reading while a prediction is saved).
            # Persists on the DB file once set; the pragma is a cheap no-op after.
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                logger.warning("Could not enable WAL mode: %s", exc)

            # Import records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS import_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    record_count INTEGER,
                    description TEXT
                )
            ''')

            # Models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    parameters TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Prediction results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    import_record_id INTEGER,
                    model_id INTEGER,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    predictions TEXT,
                    actual_results TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (import_record_id) REFERENCES import_records (id),
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            ''')

            # Historical game data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS game_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    import_record_id INTEGER,
                    home_team TEXT,
                    away_team TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    game_date DATE,
                    season TEXT,
                    result TEXT,
                    season_type TEXT,
                    fg_pct_home REAL,
                    ft_pct_home REAL,
                    fg3_pct_home REAL,
                    ast_home REAL,
                    reb_home REAL,
                    fg_pct_away REAL,
                    ft_pct_away REAL,
                    fg3_pct_away REAL,
                    ast_away REAL,
                    reb_away REAL,
                    FOREIGN KEY (import_record_id) REFERENCES import_records (id)
                )
            ''')

            conn.commit()

        # Run migrations for new columns (legacy, frozen)
        self._migrate()
        # Run versioned schema migrations (new tables/indexes going forward)
        self._apply_migrations()

    def _migrate(self):
        """Add or drop columns on existing tables. Each statement is wrapped
        in try/except so reruns on already-migrated DBs are no-ops.

        DROP COLUMN requires SQLite 3.35+ (Mar 2021). On older runtimes the
        drop is skipped with a logger warning rather than silently failing —
        the dead columns simply linger; predictive_models doesn't read them."""
        import sqlite3 as _sqlite3
        if _sqlite3.sqlite_version_info < (3, 35, 0):
            logger.warning(
                "SQLite %s is older than 3.35 — DROP COLUMN migrations will be "
                "skipped. Legacy `home_stats` / `away_stats` columns on game_data "
                "will remain but are never read.",
                _sqlite3.sqlite_version,
            )
        migrations = [
            "ALTER TABLE prediction_results ADD COLUMN confidence REAL DEFAULT NULL",
            "ALTER TABLE prediction_results ADD COLUMN feature_importance TEXT DEFAULT NULL",
            "ALTER TABLE models ADD COLUMN model_path TEXT DEFAULT NULL",
            "ALTER TABLE prediction_results ADD COLUMN cv_accuracy_mean REAL DEFAULT NULL",
            "ALTER TABLE prediction_results ADD COLUMN cv_accuracy_std REAL DEFAULT NULL",
            "ALTER TABLE prediction_results ADD COLUMN calibration_method TEXT DEFAULT NULL",
            # Dead stat columns — see docs/LEAKAGE_INVESTIGATION.md. Removed 2026-05-27.
            "ALTER TABLE game_data DROP COLUMN home_stats",
            "ALTER TABLE game_data DROP COLUMN away_stats",
            # Box-score stats + season type for new pre-game-rolling features.
            "ALTER TABLE game_data ADD COLUMN season_type TEXT DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN fg_pct_home REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN ft_pct_home REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN fg3_pct_home REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN ast_home REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN reb_home REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN fg_pct_away REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN ft_pct_away REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN fg3_pct_away REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN ast_away REAL DEFAULT NULL",
            "ALTER TABLE game_data ADD COLUMN reb_away REAL DEFAULT NULL",
        ]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for sql in migrations:
                try:
                    cursor.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already migrated
            conn.commit()

    def _apply_migrations(self):
        """Apply versioned schema migrations from SCHEMA_MIGRATIONS in order.

        Uses SQLite's `PRAGMA user_version` as the applied-version marker, so a
        migration runs at most once per DB file and reruns are no-ops. This is
        the forward path for all new schema changes; the legacy _migrate()
        ALTER block is kept only so existing DBs keep their column tweaks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current = cursor.execute("PRAGMA user_version").fetchone()[0]
            for version, statements in SCHEMA_MIGRATIONS:
                if version <= current:
                    continue
                for sql in statements:
                    cursor.execute(sql)
                # user_version takes a literal, not a bound parameter.
                cursor.execute(f"PRAGMA user_version = {int(version)}")
                current = version
            conn.commit()

    def save_prediction(self, user_handle: str, home_team: str, away_team: str,
                        predicted_label: str, season: Optional[str] = None,
                        predicted_confidence: Optional[float] = None,
                        model_id: Optional[int] = None,
                        model_label: Optional[str] = None,
                        game_date: Optional[str] = None) -> int:
        """Persist a user's saved prediction (handle-keyed). Returns its id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions
                (user_handle, model_id, home_team, away_team, season,
                 predicted_label, predicted_confidence, model_label, game_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_handle, model_id, home_team, away_team, season,
                  predicted_label, predicted_confidence, model_label, game_date))
            return cursor.lastrowid

    def get_predictions(self, user_handle: Optional[str] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """Return saved predictions, newest first, optionally filtered by handle."""
        query = "SELECT * FROM predictions"
        params: List[Any] = []
        if user_handle:
            query += " WHERE user_handle = ?"
            params.append(user_handle)
        query += " ORDER BY created_at DESC, id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def resolve_predictions(self) -> int:
        """Match unresolved saved predictions to real finals in game_data and
        fill resolved / actual_label / correct. A prediction resolves against
        the earliest game between the same home/away teams whose game_date is
        on or after the day the pick was made. Returns the number resolved."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            preds = cur.execute(
                "SELECT * FROM predictions WHERE resolved = 0"
            ).fetchall()
            resolved = 0
            for p in preds:
                made_on = (p["created_at"] or "")[:10]  # 'YYYY-MM-DD'
                row = cur.execute(
                    """
                    SELECT result, game_date FROM game_data
                    WHERE home_team = ? AND away_team = ?
                      AND result IS NOT NULL
                      AND date(game_date) >= date(?)
                    ORDER BY date(game_date) ASC
                    LIMIT 1
                    """,
                    (str(p["home_team"]), str(p["away_team"]), made_on or "0001-01-01"),
                ).fetchone()
                if row is None:
                    continue
                actual = row["result"]
                correct = 1 if p["predicted_label"] == actual else 0
                cur.execute(
                    "UPDATE predictions SET resolved = 1, actual_label = ?, correct = ? "
                    "WHERE id = ?",
                    (actual, correct, p["id"]),
                )
                resolved += 1
            conn.commit()
            return resolved

    def get_user_accuracy(self, user_handle: Optional[str] = None) -> pd.DataFrame:
        """Per-handle resolved-prediction accuracy (a user 'leaderboard' seed).
        Only counts resolved picks."""
        query = (
            "SELECT user_handle, COUNT(*) AS resolved_count, "
            "SUM(correct) AS correct_count, "
            "AVG(correct) AS accuracy "
            "FROM predictions WHERE resolved = 1"
        )
        params: List[Any] = []
        if user_handle:
            query += " AND user_handle = ?"
            params.append(user_handle)
        query += " GROUP BY user_handle ORDER BY accuracy DESC, resolved_count DESC"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_existing_game_keys(self, import_record_id: int) -> set:
        """Set of (game_date, home_team, away_team) already stored for an import
        — used to dedup incremental appends."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT game_date, home_team, away_team FROM game_data "
                "WHERE import_record_id = ?",
                (import_record_id,),
            ).fetchall()
        return {(str(r[0]), str(r[1]), str(r[2])) for r in rows}

    def append_game_data(self, import_record_id: int, games_df: pd.DataFrame) -> int:
        """Append only games not already present for this import (dedup on
        game_date + home_team + away_team). Returns the count inserted, and
        bumps the import's record_count by that amount."""
        if games_df.empty:
            return 0
        existing = self.get_existing_game_keys(import_record_id)
        new_rows = [
            row for _, row in games_df.iterrows()
            if (str(row.get("game_date")), str(row.get("home_team")), str(row.get("away_team")))
            not in existing
        ]
        if not new_rows:
            return 0
        new_df = pd.DataFrame(new_rows)
        self.save_game_data(import_record_id, new_df)
        # Keep import_records.record_count in sync with the live row count.
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM game_data WHERE import_record_id = ?",
                (import_record_id,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE import_records SET record_count = ? WHERE id = ?",
                (total, import_record_id),
            )
            conn.commit()
        return len(new_df)

    def ping(self) -> bool:
        """Cheap connectivity check for the health endpoint — SELECT 1."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def get_latest_import_id(self) -> Optional[int]:
        """The most recent import's id, or None if there are no imports."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM import_records ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    def delete_import(self, import_id: int) -> Dict[str, Any]:
        """Delete an import and everything derived from it: its game_data, its
        prediction_results, and any models whose results were *only* for this
        import (so a model evaluated on another import is kept). Saved user
        predictions are left intact — they're user-owned, not import-owned.

        Returns a summary including `model_paths` (the .pkl paths of deleted
        models) so the caller can remove the files from disk."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()

            # Models that have any result tied to this import …
            candidate_models = [
                r[0] for r in cur.execute(
                    "SELECT DISTINCT model_id FROM prediction_results WHERE import_record_id = ?",
                    (import_id,),
                ).fetchall()
            ]
            # … but only delete the ones with NO results for any other import.
            models_to_delete = []
            for mid in candidate_models:
                others = cur.execute(
                    "SELECT COUNT(*) FROM prediction_results "
                    "WHERE model_id = ? AND import_record_id <> ?",
                    (mid, import_id),
                ).fetchone()[0]
                if others == 0:
                    models_to_delete.append(mid)

            model_paths = []
            for mid in models_to_delete:
                row = cur.execute("SELECT model_path FROM models WHERE id = ?", (mid,)).fetchone()
                if row and row[0]:
                    model_paths.append(row[0])

            games_deleted = cur.execute(
                "SELECT COUNT(*) FROM game_data WHERE import_record_id = ?", (import_id,)
            ).fetchone()[0]
            results_deleted = cur.execute(
                "SELECT COUNT(*) FROM prediction_results WHERE import_record_id = ?", (import_id,)
            ).fetchone()[0]

            cur.execute("DELETE FROM prediction_results WHERE import_record_id = ?", (import_id,))
            cur.execute("DELETE FROM game_data WHERE import_record_id = ?", (import_id,))
            for mid in models_to_delete:
                cur.execute("DELETE FROM models WHERE id = ?", (mid,))
            cur.execute("DELETE FROM import_records WHERE id = ?", (import_id,))
            conn.commit()

        return {
            "import_id": import_id,
            "games_deleted": games_deleted,
            "results_deleted": results_deleted,
            "models_deleted": len(models_to_delete),
            "model_paths": model_paths,
        }

    def register_import(self, filename: str, file_path: str, record_count: int, description: str = None) -> int:
        """Register a new data import and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO import_records (filename, file_path, record_count, description)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_path, record_count, description))
            return cursor.lastrowid

    def register_model(self, name: str, model_type: str, parameters: Dict[str, Any],
                       model_path: Optional[str] = None) -> int:
        """Register a new model and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO models (name, type, parameters, model_path)
                VALUES (?, ?, ?, ?)
            ''', (name, model_type, json.dumps(parameters), model_path))
            return cursor.lastrowid

    def save_prediction_results(self, import_record_id: int, model_id: int,
                                accuracy: float, precision: float, recall: float,
                                f1: float, predictions: List, actual_results: List = None,
                                confidence: Optional[float] = None,
                                feature_importance: Optional[str] = None,
                                cv_accuracy_mean: Optional[float] = None,
                                cv_accuracy_std: Optional[float] = None,
                                calibration_method: Optional[str] = None):
        """Save prediction results to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_results
                (import_record_id, model_id, accuracy, precision_score, recall, f1_score,
                 predictions, actual_results, confidence, feature_importance,
                 cv_accuracy_mean, cv_accuracy_std, calibration_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (import_record_id, model_id, accuracy, precision, recall, f1,
                  json.dumps(predictions),
                  json.dumps(actual_results) if actual_results else None,
                  confidence, feature_importance,
                  cv_accuracy_mean, cv_accuracy_std, calibration_method))
            return cursor.lastrowid
    
    def save_game_data(self, import_record_id: int, games_df: pd.DataFrame):
        """Save game data to database. Uses executemany for a ~10× speedup over
        per-row execute on 26K-row imports."""
        stat_cols = ('fg_pct_home', 'ft_pct_home', 'fg3_pct_home', 'ast_home', 'reb_home',
                     'fg_pct_away', 'ft_pct_away', 'fg3_pct_away', 'ast_away', 'reb_away')

        def _row_tuple(row):
            return (
                import_record_id,
                row.get('home_team'),
                row.get('away_team'),
                row.get('home_score'),
                row.get('away_score'),
                row.get('game_date'),
                row.get('season'),
                row.get('result'),
                row.get('season_type'),
                *(row.get(c) if pd.notna(row.get(c)) else None for c in stat_cols),
            )

        rows = [_row_tuple(row) for _, row in games_df.iterrows()]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany('''
                INSERT INTO game_data
                (import_record_id, home_team, away_team, home_score, away_score,
                 game_date, season, result, season_type,
                 fg_pct_home, ft_pct_home, fg3_pct_home, ast_home, reb_home,
                 fg_pct_away, ft_pct_away, fg3_pct_away, ast_away, reb_away)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?)
            ''', rows)
    
    def get_import_records(self) -> pd.DataFrame:
        """Get all import records"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM import_records", conn)
    
    def get_models(self) -> pd.DataFrame:
        """Get all models"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM models", conn)
    
    def get_prediction_results(self, import_record_id: int = None, model_id: int = None) -> pd.DataFrame:
        """Get prediction results with optional filtering"""
        query = "SELECT * FROM prediction_results"
        params = []
        
        if import_record_id or model_id:
            query += " WHERE "
            conditions = []
            if import_record_id:
                conditions.append("import_record_id = ?")
                params.append(import_record_id)
            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)
            query += " AND ".join(conditions)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_game_data(self, import_record_id: int = None) -> pd.DataFrame:
        """Get game data with optional filtering by import record"""
        query = "SELECT * FROM game_data"
        params = []

        if import_record_id:
            query += " WHERE import_record_id = ?"
            params.append(import_record_id)

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Return a single model row as a dict, or None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_model_by_type(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Return the most recently trained model of a given type, or None."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM models WHERE type = ? AND model_path IS NOT NULL "
                "ORDER BY created_date DESC LIMIT 1",
                (model_type,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_unique_teams(self, import_record_id: int) -> List[str]:
        """Return sorted list of unique team names for a given import."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT home_team FROM game_data WHERE import_record_id = ? "
                "UNION SELECT DISTINCT away_team FROM game_data WHERE import_record_id = ? "
                "ORDER BY 1",
                (import_record_id, import_record_id)
            )
            return [r[0] for r in cursor.fetchall() if r[0]]