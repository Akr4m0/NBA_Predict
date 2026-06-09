"""Schema, migration, and CRUD round-trip tests for NBADatabase."""
import json
import sqlite3

import pandas as pd
import pytest

from database import NBADatabase


def _columns(db_path: str, table: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def test_fresh_db_has_expected_tables(db, tmp_db_path):
    with sqlite3.connect(tmp_db_path) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert {"import_records", "models", "prediction_results", "game_data"}.issubset(names)


def test_dead_stat_columns_are_absent_on_fresh_db(db, tmp_db_path):
    """game_data must NOT carry home_stats / away_stats anymore."""
    cols = _columns(tmp_db_path, "game_data")
    assert "home_stats" not in cols
    assert "away_stats" not in cols


def test_migration_drops_stat_columns_from_legacy_db(tmp_db_path):
    """Simulate a legacy DB that still has home_stats/away_stats, then init NBADatabase."""
    with sqlite3.connect(tmp_db_path) as conn:
        conn.execute("""
            CREATE TABLE game_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                import_record_id INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                game_date DATE,
                season TEXT,
                home_stats TEXT,
                away_stats TEXT,
                result TEXT
            )
        """)
        conn.commit()

    # Constructor runs _migrate(), which should DROP the dead columns.
    NBADatabase(db_path=tmp_db_path)

    cols = _columns(tmp_db_path, "game_data")
    assert "home_stats" not in cols, f"home_stats still present: {cols}"
    assert "away_stats" not in cols, f"away_stats still present: {cols}"


def test_migration_is_idempotent(db, tmp_db_path):
    # Initialising a second NBADatabase on the same path must not error.
    NBADatabase(db_path=tmp_db_path)
    NBADatabase(db_path=tmp_db_path)
    cols = _columns(tmp_db_path, "prediction_results")
    # New columns from the migration should be present.
    assert "cv_accuracy_mean" in cols
    assert "calibration_method" in cols


def test_register_import_round_trip(db):
    iid = db.register_import("foo.csv", "/tmp/foo.csv", record_count=10, description="d")
    assert iid is not None
    df = db.get_import_records()
    assert iid in df["id"].values


def test_register_model_and_save_prediction_results(db):
    iid = db.register_import("foo.csv", "/tmp/foo.csv", record_count=5)
    mid = db.register_model("rf", "RandomForest", {"n_estimators": 100}, model_path="/tmp/m.pkl")

    db.save_prediction_results(
        import_record_id=iid, model_id=mid,
        accuracy=0.65, precision=0.6, recall=0.7, f1=0.65,
        predictions=[0, 1], actual_results=[1, 1],
        confidence=0.55, feature_importance=json.dumps({"elo_diff": 0.24}),
        cv_accuracy_mean=0.66, cv_accuracy_std=0.01, calibration_method="isotonic",
    )
    results = db.get_prediction_results(import_record_id=iid, model_id=mid)
    assert len(results) == 1
    row = results.iloc[0]
    assert row["accuracy"] == pytest.approx(0.65)
    assert row["cv_accuracy_mean"] == pytest.approx(0.66)
    assert row["calibration_method"] == "isotonic"


def test_save_game_data_round_trip(db, sample_games_df):
    iid = db.register_import("games.csv", "/tmp/games.csv", record_count=len(sample_games_df))
    df_to_save = sample_games_df.copy()
    df_to_save["game_date"] = df_to_save["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, df_to_save)

    loaded = db.get_game_data(iid)
    assert len(loaded) == len(sample_games_df)
    # Schema only has the live columns
    assert "home_stats" not in loaded.columns
    assert "away_stats" not in loaded.columns


def test_get_unique_teams(db, sample_games_df):
    iid = db.register_import("games.csv", "/tmp/games.csv", record_count=len(sample_games_df))
    df_to_save = sample_games_df.copy()
    df_to_save["game_date"] = df_to_save["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, df_to_save)

    teams = db.get_unique_teams(iid)
    expected = sorted(set(sample_games_df["home_team"]) | set(sample_games_df["away_team"]))
    assert teams == expected


# ── Phase 9 step 1: WAL, versioned migrations, predictions table ──────────────

def test_wal_mode_enabled(db, tmp_db_path):
    with sqlite3.connect(tmp_db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_predictions_table_exists(db, tmp_db_path):
    cols = _columns(tmp_db_path, "predictions")
    for expected in ("user_handle", "home_team", "away_team", "predicted_label",
                     "predicted_confidence", "model_label", "resolved",
                     "actual_label", "correct", "created_at"):
        assert expected in cols, f"missing column {expected}: {cols}"


def test_user_version_set_to_latest_migration(db, tmp_db_path):
    from database import SCHEMA_MIGRATIONS
    latest = max(v for v, _ in SCHEMA_MIGRATIONS)
    with sqlite3.connect(tmp_db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == latest


def test_versioned_migration_is_idempotent_and_preserves_rows(db, tmp_db_path):
    """Re-initialising must not rerun migrations or wipe existing predictions."""
    pid = db.save_prediction(
        user_handle="akram", home_team="1610612738", away_team="1610612739",
        predicted_label="home_win", season="2022", predicted_confidence=0.61,
        model_id=None, model_label="home_win",
    )
    assert pid is not None
    # Second construction on the same file must be a no-op for migrations.
    NBADatabase(db_path=tmp_db_path)
    rows = db.get_predictions(user_handle="akram")
    assert len(rows) == 1
    assert rows.iloc[0]["id"] == pid


def test_delete_import_cascades(db, sample_games_df):
    """Deleting an import removes its games + results + models trained on it,
    keeps user predictions, and reports the deleted model file paths."""
    iid = db.register_import("games.csv", "/tmp/games.csv", len(sample_games_df))
    df = sample_games_df.copy()
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, df)
    mid = db.register_model("rf", "RandomForest", {}, model_path="/tmp/rf.pkl")
    db.save_prediction_results(iid, mid, accuracy=0.6, precision=0.6, recall=0.6,
                               f1=0.6, predictions=[1], actual_results=[1])
    # a user pick must survive
    db.save_prediction(user_handle="akram", home_team="1610612738",
                       away_team="1610612739", predicted_label="home_win")

    summary = db.delete_import(iid)
    assert summary["games_deleted"] == len(sample_games_df)
    assert summary["results_deleted"] == 1
    assert summary["models_deleted"] == 1
    assert summary["model_paths"] == ["/tmp/rf.pkl"]

    assert db.get_game_data(iid).empty
    assert db.get_model_by_id(mid) is None
    assert iid not in db.get_import_records().get("id", [])
    # user prediction is untouched
    assert len(db.get_predictions(user_handle="akram")) == 1


def test_delete_import_keeps_model_used_by_other_import(db, sample_games_df):
    """A model with results on another import must NOT be deleted."""
    i1 = db.register_import("a.csv", "/tmp/a", 0)
    i2 = db.register_import("b.csv", "/tmp/b", 0)
    mid = db.register_model("rf", "RandomForest", {}, model_path="/tmp/rf.pkl")
    db.save_prediction_results(i1, mid, accuracy=0.6, precision=0.6, recall=0.6, f1=0.6,
                               predictions=[1], actual_results=[1])
    db.save_prediction_results(i2, mid, accuracy=0.7, precision=0.7, recall=0.7, f1=0.7,
                               predictions=[1], actual_results=[1])

    summary = db.delete_import(i1)
    assert summary["models_deleted"] == 0  # still referenced by i2
    assert db.get_model_by_id(mid) is not None


def test_save_and_get_predictions_round_trip(db):
    db.save_prediction(
        user_handle="akram", home_team="1610612738", away_team="1610612739",
        predicted_label="home_win", season="2022", predicted_confidence=0.61,
        model_label="home_win",
    )
    db.save_prediction(
        user_handle="other", home_team="1610612740", away_team="1610612741",
        predicted_label="away_win", season="2022", predicted_confidence=0.55,
    )
    mine = db.get_predictions(user_handle="akram")
    assert len(mine) == 1
    row = mine.iloc[0]
    assert row["home_team"] == "1610612738"
    assert row["predicted_label"] == "home_win"
    assert row["predicted_confidence"] == pytest.approx(0.61)
    assert int(row["resolved"]) == 0
    # No filter → all handles.
    assert len(db.get_predictions()) == 2
    # Limit respected.
    assert len(db.get_predictions(limit=1)) == 1
