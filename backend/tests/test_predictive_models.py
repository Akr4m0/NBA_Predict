"""
Tests for PredictiveModels.

The most important test here is `test_rolling_winrate_does_not_leak_future` — it
verifies the core invariant that pre-game features for game N never see games
≥ N. If this regresses, train metrics will look artificially great and predictions
will silently degrade in production.
"""
import numpy as np
import pandas as pd
import pytest

from predictive_models import PredictiveModels


def test_prepare_features_produces_expected_columns(db, sample_games_df):
    pm = PredictiveModels(db=db)
    features, target = pm.prepare_features(sample_games_df)

    # All 34 features documented in CLAUDE.md must be present — equality, not
    # subset, so a regression that drops a feature is caught.
    expected = {
        # Identity / calendar (6)
        "home_team_encoded", "away_team_encoded", "season_encoded",
        "month", "day_of_week", "day_of_year",
        # Rolling form (3)
        "home_team_last5_winrate", "away_team_last5_winrate", "h2h_home_wins_rate",
        # Elo (3)
        "home_elo_pre", "away_elo_pre", "elo_diff",
        # Rest (5)
        "home_rest_days", "away_rest_days", "home_b2b", "away_b2b", "rest_diff",
        # Venue (2)
        "home_team_home_winrate", "away_team_away_winrate",
        # Box-score rolling (10)
        "home_fg_pct_l10", "away_fg_pct_l10",
        "home_ft_pct_l10", "away_ft_pct_l10",
        "home_fg3_pct_l10", "away_fg3_pct_l10",
        "home_ast_l10", "away_ast_l10",
        "home_reb_l10", "away_reb_l10",
        # Travel & circadian (4)
        "home_travel_dist", "away_travel_dist", "home_tz_shift", "away_tz_shift",
        # Season type (1)
        "is_playoff",
    }
    actual = set(features.columns)
    assert actual == expected, (
        f"missing: {expected - actual}\nextra: {actual - expected}"
    )
    assert len(features.columns) == 34

    # Target is binary
    assert target is not None
    assert set(target.unique()).issubset({0, 1})
    assert len(features) == len(sample_games_df)


def test_rolling_winrate_does_not_leak_future(db):
    """
    Build a tiny chronological history for team A: 4 losses, then 1 win.
    Querying the rolling winrate at the date of game5 must NOT include game5.
    """
    d = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 3) for i in range(5)]
    # Explicit rows. In each row, scores + result are internally consistent
    # AND A's outcome is what the test name claims (L, L, L, L, W).
    df = pd.DataFrame([
        # A home, A lost (B wins as away)
        {"home_team": "A", "away_team": "B", "home_score":  95, "away_score": 110, "result": "away_win", "game_date": d[0], "season": "2019-20"},
        # A away, A lost (B wins as home)
        {"home_team": "B", "away_team": "A", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[1], "season": "2019-20"},
        # A home, A lost
        {"home_team": "A", "away_team": "B", "home_score":  95, "away_score": 110, "result": "away_win", "game_date": d[2], "season": "2019-20"},
        # A away, A lost
        {"home_team": "B", "away_team": "A", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[3], "season": "2019-20"},
        # A home, A finally wins
        {"home_team": "A", "away_team": "B", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[4], "season": "2019-20"},
    ])

    pm = PredictiveModels(db=db)
    pm._build_history_index(df)

    # Sanity: A has exactly 5 games in the index, with wins [0,0,0,0,1].
    assert len(pm._team_dates["A"]) == 5
    assert list(pm._team_wins["A"]) == [0, 0, 0, 0, 1]

    # At the date of game 5 (A's only win): only the 4 prior losses count.
    rate_at_game5 = pm._rolling_winrate("A", d[4], n=5)
    assert rate_at_game5 == pytest.approx(0.0), (
        f"Leakage: winrate at game5 should be 0/4=0.0 but got {rate_at_game5}"
    )

    # One day AFTER game 5, the win counts: 1/5 = 0.2.
    rate_after = pm._rolling_winrate("A", d[4] + pd.Timedelta(days=1), n=5)
    assert rate_after == pytest.approx(0.2)


def test_elo_pre_does_not_use_future_games(db):
    """The pre-game Elo for game N in _elo_lookup must reflect post-game Elo of N-1, never N."""
    rows = []
    base = pd.Timestamp("2020-01-01")
    for i in range(6):
        h, a = ("A", "B") if i % 2 == 0 else ("B", "A")
        # A wins all 6 games.
        result = "home_win" if h == "A" else "away_win"
        score_h = 110 if (h == "A") else 95
        score_a = 95 if (h == "A") else 110
        rows.append({
            "home_team": h, "away_team": a,
            "game_date": base + pd.Timedelta(days=i * 3),
            "home_score": score_h, "away_score": score_a,
            "season": "2019-20", "result": result,
        })
    df = pd.DataFrame(rows)

    pm = PredictiveModels(db=db)
    pm._build_history_index(df)
    lookup = pm._elo_lookup.sort_values("game_date").reset_index(drop=True)

    # A wins everything, so A's pre-game Elo must be monotonically non-decreasing.
    a_elos = []
    for _, row in lookup.iterrows():
        if row["home_team"] == "A":
            a_elos.append(row["home_elo_pre"])
        else:
            a_elos.append(row["away_elo_pre"])
    for prev, curr in zip(a_elos, a_elos[1:]):
        assert curr >= prev, f"A's pre-Elo decreased: {a_elos}"
    # First game: A starts at the initialization value.
    assert a_elos[0] == pytest.approx(PredictiveModels._ELO_INIT)


def test_rolling_stat_mean_does_not_leak_future(db):
    """
    Build a tiny history for team A with deterministic FG_PCT values, then verify
    that querying at game N's date averages ONLY games 0..N-1, never N or later.
    """
    d = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 3) for i in range(5)]
    df = pd.DataFrame([
        # A always at home for simplicity. fg_pct_home rises: 0.40, 0.45, 0.50, 0.55, 0.99 (sentinel for leak detection).
        {"home_team": "A", "away_team": "B", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[0], "season": "2019-20", "fg_pct_home": 0.40, "fg_pct_away": 0.40, "ft_pct_home": 0.7, "ft_pct_away": 0.7, "fg3_pct_home": 0.3, "fg3_pct_away": 0.3, "ast_home": 20, "ast_away": 20, "reb_home": 40, "reb_away": 40},
        {"home_team": "A", "away_team": "B", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[1], "season": "2019-20", "fg_pct_home": 0.45, "fg_pct_away": 0.40, "ft_pct_home": 0.7, "ft_pct_away": 0.7, "fg3_pct_home": 0.3, "fg3_pct_away": 0.3, "ast_home": 20, "ast_away": 20, "reb_home": 40, "reb_away": 40},
        {"home_team": "A", "away_team": "B", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[2], "season": "2019-20", "fg_pct_home": 0.50, "fg_pct_away": 0.40, "ft_pct_home": 0.7, "ft_pct_away": 0.7, "fg3_pct_home": 0.3, "fg3_pct_away": 0.3, "ast_home": 20, "ast_away": 20, "reb_home": 40, "reb_away": 40},
        {"home_team": "A", "away_team": "B", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[3], "season": "2019-20", "fg_pct_home": 0.55, "fg_pct_away": 0.40, "ft_pct_home": 0.7, "ft_pct_away": 0.7, "fg3_pct_home": 0.3, "fg3_pct_away": 0.3, "ast_home": 20, "ast_away": 20, "reb_home": 40, "reb_away": 40},
        # Game 5 is the would-be leak: if it bleeds into earlier rolling means we'll see 0.99 pollute the result.
        {"home_team": "A", "away_team": "B", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[4], "season": "2019-20", "fg_pct_home": 0.99, "fg_pct_away": 0.40, "ft_pct_home": 0.7, "ft_pct_away": 0.7, "fg3_pct_home": 0.3, "fg3_pct_away": 0.3, "ast_home": 20, "ast_away": 20, "reb_home": 40, "reb_away": 40},
    ])

    pm = PredictiveModels(db=db)
    pm._build_history_index(df)

    # At d[4] (game 5), only the four prior values 0.40/0.45/0.50/0.55 should average.
    mean = pm._rolling_stat_mean("A", d[4], "fg_pct")
    assert mean == pytest.approx((0.40 + 0.45 + 0.50 + 0.55) / 4), (
        f"Leakage: fg_pct rolling mean at d[4] should be 0.475 (no game5) but was {mean}"
    )

    # AFTER d[4] the 0.99 value is allowed.
    mean_after = pm._rolling_stat_mean("A", d[4] + pd.Timedelta(days=1), "fg_pct")
    assert mean_after == pytest.approx((0.40 + 0.45 + 0.50 + 0.55 + 0.99) / 5)


def test_last_venue_returns_prior_host(db):
    """_last_venue must return the previous game's host team, not the current one."""
    d = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 3) for i in range(3)]
    df = pd.DataFrame([
        # Game 1: B hosts.    Game 2: C hosts.   Game 3 (the one we're predicting): A hosts.
        {"home_team": "B", "away_team": "A", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[0], "season": "2019-20"},
        {"home_team": "C", "away_team": "A", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[1], "season": "2019-20"},
        {"home_team": "A", "away_team": "D", "home_score": 110, "away_score":  95, "result": "home_win", "game_date": d[2], "season": "2019-20"},
    ])

    pm = PredictiveModels(db=db)
    pm._build_history_index(df)

    # At the time of game 3 (A vs D in A's arena), A's last venue was C (game 2 host).
    assert pm._last_venue("A", d[2]) == "C"
    # Before any game, A has no prior venue.
    assert pm._last_venue("A", d[0]) is None


def test_vectorized_and_per_call_paths_agree(db, sample_games_df):
    """
    The vectorized path (`_precompute_team_pregame_features` + merges) must
    produce the same numbers as the per-call helpers (`_rolling_winrate`,
    `_h2h_winrate`, `_days_since_last`, `_rolling_venue_winrate`,
    `_rolling_stat_mean`, `_last_venue` + distance/tz). If they drift, training
    metrics silently move without any feature-list change.

    Per-call helpers are still used by `predict_single` for one-game inference,
    so this is exercising the equivalence we rely on across both code paths.
    """
    from team_locations import distance_between, tz_shift_hours

    pm = PredictiveModels(db=db)
    features, _ = pm.prepare_features(sample_games_df)

    # _build_history_index has already been called by prepare_features and the
    # per-call helpers will hit the same indices.
    df = sample_games_df.reset_index(drop=True).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    for i, row in df.iterrows():
        ht, at, d = row["home_team"], row["away_team"], row["game_date"]

        # Last-5 win rate
        assert features["home_team_last5_winrate"].iloc[i] == pytest.approx(
            pm._rolling_winrate(ht, d), rel=1e-6
        ), f"home_team_last5_winrate drift on row {i}"
        assert features["away_team_last5_winrate"].iloc[i] == pytest.approx(
            pm._rolling_winrate(at, d), rel=1e-6
        ), f"away_team_last5_winrate drift on row {i}"

        # H2H
        assert features["h2h_home_wins_rate"].iloc[i] == pytest.approx(
            pm._h2h_winrate(ht, at, d), rel=1e-6
        ), f"h2h_home_wins_rate drift on row {i}"

        # Rest
        assert int(features["home_rest_days"].iloc[i]) == pm._days_since_last(ht, d)
        assert int(features["away_rest_days"].iloc[i]) == pm._days_since_last(at, d)

        # Venue win rate
        assert features["home_team_home_winrate"].iloc[i] == pytest.approx(
            pm._rolling_venue_winrate(ht, d, "home"), rel=1e-6
        ), f"home_team_home_winrate drift on row {i}"
        assert features["away_team_away_winrate"].iloc[i] == pytest.approx(
            pm._rolling_venue_winrate(at, d, "away"), rel=1e-6
        ), f"away_team_away_winrate drift on row {i}"

        # Rolling stat means (cover one of each side to keep the loop quick)
        for stat in ("fg_pct", "ast", "reb"):
            assert features[f"home_{stat}_l10"].iloc[i] == pytest.approx(
                pm._rolling_stat_mean(ht, d, stat), rel=1e-6
            ), f"home_{stat}_l10 drift on row {i}"
            assert features[f"away_{stat}_l10"].iloc[i] == pytest.approx(
                pm._rolling_stat_mean(at, d, stat), rel=1e-6
            ), f"away_{stat}_l10 drift on row {i}"

        # Travel + tz against current venue (home team's arena)
        h_prev = pm._last_venue(ht, d)
        a_prev = pm._last_venue(at, d)
        expected_h_travel = distance_between(h_prev, ht) if h_prev is not None else 0.0
        expected_a_travel = distance_between(a_prev, ht) if a_prev is not None else 0.0
        assert features["home_travel_dist"].iloc[i] == pytest.approx(expected_h_travel, abs=1e-6)
        assert features["away_travel_dist"].iloc[i] == pytest.approx(expected_a_travel, abs=1e-6)
        expected_h_tz = tz_shift_hours(h_prev, ht) if h_prev is not None else 0
        expected_a_tz = tz_shift_hours(a_prev, ht) if a_prev is not None else 0
        assert int(features["home_tz_shift"].iloc[i]) == expected_h_tz
        assert int(features["away_tz_shift"].iloc[i]) == expected_a_tz


def test_training_pipeline_runs_end_to_end(db, sample_games_df, tmp_db_path):
    """Smoke test: a baseline model trains + persists metrics with no crashes."""
    iid = db.register_import("games.csv", "/tmp/games.csv", record_count=len(sample_games_df))
    df = sample_games_df.copy()
    df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
    db.save_game_data(iid, df)

    pm = PredictiveModels(db=db)
    model_id, metrics = pm.train_baseline(iid, strategy="most_frequent")
    assert isinstance(model_id, int)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
