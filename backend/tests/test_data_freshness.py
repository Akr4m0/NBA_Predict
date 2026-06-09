"""Tests for the balldontlie freshness pipeline (mapping, dedup-append,
resolution) using a fake client — no live API calls."""
import pandas as pd
import pytest

from data_freshness import FreshnessPipeline


def _game(gid, date, season, home_abbr, away_abbr, hs, as_, postseason=False, status="Final"):
    return {
        "id": gid, "date": date, "season": season, "status": status,
        "postseason": postseason,
        "home_team_score": hs, "visitor_team_score": as_,
        "home_team": {"abbreviation": home_abbr},
        "visitor_team": {"abbreviation": away_abbr},
    }


class FakeClient:
    """Mimics BalldontlieClient.iter_finals — yields only 'Final' games."""
    def __init__(self, games):
        self.games = games
        self.configured = True

    def iter_finals(self, seasons, start_date=None, per_page=100, max_games=None):
        n = 0
        for g in self.games:
            if str(g.get("status", "")).strip().lower() != "final":
                continue
            yield g
            n += 1
            if max_games is not None and n >= max_games:
                return


# NBA Stats IDs for the abbreviations used below.
BOS, NYK, LAL, MIN = "1610612738", "1610612752", "1610612747", "1610612750"


def test_map_game_translates_to_wide_row():
    row = FreshnessPipeline.map_game(_game(101, "2024-10-22", 2024, "BOS", "NYK", 132, 109))
    assert row["home_team"] == BOS
    assert row["away_team"] == NYK
    assert row["home_score"] == 132 and row["away_score"] == 109
    assert row["season"] == "2024"
    assert row["GAME_ID"].startswith("2")  # regular season leading digit


def test_map_game_playoff_leading_digit():
    row = FreshnessPipeline.map_game(
        _game(102, "2024-05-01", 2023, "LAL", "MIN", 100, 99, postseason=True)
    )
    assert row["GAME_ID"].startswith("4")  # playoffs


def test_map_game_returns_none_for_unmappable():
    # Unknown abbreviation (defunct team)
    assert FreshnessPipeline.map_game(_game(1, "2024-10-22", 2024, "ZZZ", "NYK", 1, 2)) is None
    # Missing score
    g = _game(2, "2024-10-22", 2024, "BOS", "NYK", None, 2)
    assert FreshnessPipeline.map_game(g) is None


def test_ingest_appends_and_dedups(db):
    iid = db.register_import("seed.csv", "/tmp/seed.csv", 0)
    games = [
        _game(101, "2024-10-22", 2024, "BOS", "NYK", 132, 109),
        _game(102, "2024-10-22", 2024, "LAL", "MIN", 110, 103),
    ]
    pipe = FreshnessPipeline(db=db, client=FakeClient(games))

    r = pipe.ingest(seasons=[2024], target_import_id=iid)
    assert r["inserted"] == 2
    assert r["import_id"] == iid

    # Re-running the same games must insert nothing (dedup).
    r2 = pipe.ingest(seasons=[2024], target_import_id=iid)
    assert r2["inserted"] == 0

    gd = db.get_game_data(iid)
    assert len(gd) == 2
    bos = gd[gd["home_team"] == BOS].iloc[0]
    assert bos["away_team"] == NYK
    assert bos["home_score"] == 132
    assert bos["result"] == "home_win"
    assert bos["season_type"] == "regular"
    # record_count was bumped to match.
    imp = db.get_import_records()
    assert int(imp[imp["id"] == iid].iloc[0]["record_count"]) == 2


def test_sync_resolves_predictions_correct_and_wrong(db):
    iid = db.register_import("seed.csv", "/tmp/seed.csv", 0)
    # Far-future dates so they're >= the picks' created_at (now).
    db.save_prediction(user_handle="akram", home_team=BOS, away_team=NYK,
                       predicted_label="home_win", season="2099")   # will be RIGHT
    db.save_prediction(user_handle="akram", home_team=LAL, away_team=MIN,
                       predicted_label="home_win", season="2099")   # will be WRONG
    games = [
        _game(201, "2099-12-30", 2099, "BOS", "NYK", 120, 100),  # home won
        _game(202, "2099-12-31", 2099, "LAL", "MIN", 90, 110),   # home lost
    ]
    pipe = FreshnessPipeline(db=db, client=FakeClient(games))
    res = pipe.sync(seasons=[2099], target_import_id=iid)

    assert res["inserted"] == 2
    assert res["resolved_predictions"] == 2

    preds = db.get_predictions(user_handle="akram")
    by_home = {p["home_team"]: p for _, p in preds.iterrows()}
    assert int(by_home[BOS]["correct"]) == 1 and by_home[BOS]["actual_label"] == "home_win"
    assert int(by_home[LAL]["correct"]) == 0 and by_home[LAL]["actual_label"] == "away_win"

    # User accuracy: 1 of 2 → 0.5
    acc = db.get_user_accuracy("akram")
    assert len(acc) == 1
    assert int(acc.iloc[0]["resolved_count"]) == 2
    assert acc.iloc[0]["accuracy"] == pytest.approx(0.5)


def test_sync_retrains_when_requested(db, sample_games_df):
    from predictive_models import PredictiveModels
    iid = db.register_import("seed.csv", "/tmp/seed.csv", len(sample_games_df))
    seed = sample_games_df.copy()
    seed["game_date"] = seed["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, seed)

    games = [
        _game(401, "2099-01-01", 2099, "BOS", "NYK", 120, 100),
        _game(402, "2099-01-02", 2099, "LAL", "MIN", 100, 90),
    ]
    pm = PredictiveModels(db=db)
    pipe = FreshnessPipeline(db=db, client=FakeClient(games), pm=pm)
    res = pipe.sync(seasons=[2099], target_import_id=iid, retrain=["baseline"])

    assert res["inserted"] == 2
    assert len(res["retrained"]) == 1
    assert res["retrained"][0]["model_type"] == "baseline"
    assert res["retrained"][0]["model_id"] is not None


def test_resolve_skips_games_before_pick_date(db):
    iid = db.register_import("seed.csv", "/tmp/seed.csv", 0)
    db.save_prediction(user_handle="akram", home_team=BOS, away_team=NYK,
                       predicted_label="home_win", season="2003")
    # A 2003 game is before the pick's created_at (now) → must NOT resolve.
    games = [_game(301, "2003-11-01", 2003, "BOS", "NYK", 120, 100)]
    pipe = FreshnessPipeline(db=db, client=FakeClient(games))
    res = pipe.sync(seasons=[2003], target_import_id=iid)
    assert res["resolved_predictions"] == 0
    p = db.get_predictions(user_handle="akram").iloc[0]
    assert int(p["resolved"]) == 0
