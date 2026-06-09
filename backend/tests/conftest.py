"""
Shared pytest fixtures.

All fixtures use a tmp DB so tests never touch data/nba_predictions.db.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

# Make backend/ importable for tests (run from any cwd).
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@pytest.fixture(autouse=True)
def _isolate_models_dir(tmp_path, monkeypatch):
    """Every test gets its own MODELS_DIR so `train_*` never writes a .pkl to
    the real data/models/ directory. Patches both the env var (picked up by
    fresh imports) and the already-loaded module attribute (if any)."""
    test_models = tmp_path / "models"
    test_models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NBA_MODELS_DIR", str(test_models))
    if "predictive_models" in sys.modules:
        monkeypatch.setattr(sys.modules["predictive_models"], "MODELS_DIR", test_models)


@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
def db(tmp_db_path):
    from database import NBADatabase
    return NBADatabase(db_path=tmp_db_path)


@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """
    A tiny but realistic chronological dataset: 4 teams, 2 seasons, 24 games.
    Enough for prepare_features to produce all 19 features without NaNs falling
    back to means, and enough rows for TimeSeriesSplit(n_splits=2) in CV tests.
    """
    teams = ["1610612737", "1610612738", "1610612739", "1610612740"]
    rows = []
    # Season 2018-19: ~12 games, alternating winners
    base = pd.Timestamp("2018-10-15")
    for i in range(12):
        h, a = teams[i % 4], teams[(i + 1) % 4]
        if h == a:
            a = teams[(i + 2) % 4]
        home_win = (i % 3 != 0)
        rows.append({
            "home_team": h,
            "away_team": a,
            "game_date": base + pd.Timedelta(days=i * 3),
            "home_score": 110 if home_win else 95,
            "away_score": 95 if home_win else 110,
            "season": "2018-19",
            "result": "home_win" if home_win else "away_win",
        })
    # Season 2019-20: ~12 more
    base2 = pd.Timestamp("2019-10-15")
    for i in range(12):
        h, a = teams[(i + 2) % 4], teams[(i + 3) % 4]
        if h == a:
            a = teams[i % 4]
        home_win = (i % 2 == 0)
        rows.append({
            "home_team": h,
            "away_team": a,
            "game_date": base2 + pd.Timedelta(days=i * 3),
            "home_score": 110 if home_win else 95,
            "away_score": 95 if home_win else 110,
            "season": "2019-20",
            "result": "home_win" if home_win else "away_win",
        })
    return pd.DataFrame(rows)


def _reload_app_modules():
    """Reload api + its deps IN PLACE so module-level env-var reads (DB_PATH,
    UPLOADS_DIR, MODELS_DIR) pick up new values. Critical: we use
    `importlib.reload` rather than `sys.modules.pop`+`import` — the latter
    creates a *new* module object, leaving any class already imported by another
    test file referencing the old module's globals (which still holds the
    pre-patch MODELS_DIR → tests pollute data/models/).
    """
    import importlib
    import database, data_importer, predictive_models, api
    for mod in (database, data_importer, predictive_models, api):
        importlib.reload(mod)
    return api


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """FastAPI TestClient with isolated DB + uploads dir, no auth enforced."""
    monkeypatch.setenv("NBA_DB_PATH", str(tmp_path / "api_test.db"))
    monkeypatch.setenv("NBA_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("NBA_API_KEY", raising=False)
    # Force balldontlie "unconfigured" in tests even if a real key lives in .env
    # (api._load_dotenv won't overwrite an already-set var; "" → configured False).
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "")

    api_module = _reload_app_modules()
    from fastapi.testclient import TestClient
    return TestClient(api_module.app)


@pytest.fixture
def api_client_with_key(tmp_path, monkeypatch):
    """Same TestClient but with NBA_API_KEY set. Header X-API-Key must be 'secret'."""
    monkeypatch.setenv("NBA_DB_PATH", str(tmp_path / "api_auth_test.db"))
    monkeypatch.setenv("NBA_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("NBA_API_KEY", "secret")
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "")

    api_module = _reload_app_modules()
    from fastapi.testclient import TestClient
    return TestClient(api_module.app)
