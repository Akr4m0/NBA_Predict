import pandas as pd
import pytest

from data_importer import validate_csv


def _good_df(**overrides) -> pd.DataFrame:
    df = pd.DataFrame({
        "home_team": ["1610612737", "1610612738"],
        "away_team": ["1610612738", "1610612737"],
        "game_date": ["2019-10-15", "2019-10-17"],
        "home_score": [110, 95],
        "away_score": [95, 100],
    })
    for k, v in overrides.items():
        df[k] = v
    return df


def test_valid_csv_passes():
    result = validate_csv(_good_df())
    assert result["valid"] is True
    assert result["errors"] == []


def test_missing_required_column():
    df = _good_df().drop(columns=["home_team"])
    result = validate_csv(df)
    assert result["valid"] is False
    assert any("home_team" in e for e in result["errors"])


def test_null_team_is_error():
    df = _good_df()
    df.loc[0, "home_team"] = None
    result = validate_csv(df)
    assert result["valid"] is False
    assert any("home_team" in e and "null" in e for e in result["errors"])


def test_null_score_is_warning_not_error():
    df = _good_df()
    df.loc[1, "home_score"] = None
    result = validate_csv(df)
    assert result["valid"] is True
    assert any("home_score" in w for w in result["warnings"])


def test_negative_score_is_error():
    df = _good_df()
    df.loc[0, "home_score"] = -5
    result = validate_csv(df)
    assert result["valid"] is False
    assert any("home_score" in e and "negative" in e for e in result["errors"])


def test_empty_team_name_is_error():
    df = _good_df()
    df.loc[0, "home_team"] = "   "
    result = validate_csv(df)
    assert result["valid"] is False
    assert any("home_team" in e for e in result["errors"])


def test_unparseable_date_is_error():
    df = _good_df()
    df.loc[0, "game_date"] = "not-a-date"
    result = validate_csv(df)
    assert result["valid"] is False
    assert any("game_date" in e for e in result["errors"])


def test_tie_game_emits_warning():
    df = _good_df()
    df.loc[0, "home_score"] = 100
    df.loc[0, "away_score"] = 100
    result = validate_csv(df)
    # Ties don't invalidate, but they must be flagged.
    assert result["valid"] is True
    assert any("tie" in w.lower() for w in result["warnings"])
