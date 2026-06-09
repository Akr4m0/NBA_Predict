"""Sanity tests for the team-location helpers used by travel/tz features."""
import pytest

from team_locations import (
    distance_between,
    get_location,
    haversine_miles,
    tz_shift_hours,
)


# Known NBA team IDs for cross-coast and same-city checks.
BOSTON = "1610612738"
GSW = "1610612744"
LAL = "1610612747"
LAC = "1610612746"
TOR = "1610612761"
DEN = "1610612743"


def test_distance_known_pair_boston_to_gsw_within_tolerance():
    # Boston (~Logan) to SF Bay is ~2700 mi great-circle. Anything in 2600–2800 is fine.
    d = distance_between(BOSTON, GSW)
    assert 2600 <= d <= 2800, f"unexpected BOS→GSW distance: {d}"


def test_distance_same_team_is_zero():
    assert distance_between(BOSTON, BOSTON) == pytest.approx(0.0, abs=0.01)


def test_distance_lal_lac_share_arena_distance_zero():
    # Both teams share Crypto.com / Arena coordinates — distance must be ~0.
    assert distance_between(LAL, LAC) == pytest.approx(0.0, abs=0.5)


def test_distance_unknown_team_returns_zero():
    assert distance_between("9999999999", BOSTON) == 0.0
    assert distance_between(BOSTON, "9999999999") == 0.0


def test_tz_shift_pt_to_et_is_positive_three():
    # GSW (PT, offset 8) → BOS (ET, offset 5) = 8 - 5 = 3 hours east.
    assert tz_shift_hours(GSW, BOSTON) == 3


def test_tz_shift_et_to_pt_is_negative_three():
    assert tz_shift_hours(BOSTON, GSW) == -3


def test_tz_shift_within_same_zone_is_zero():
    # TOR is in Eastern alongside BOS.
    assert tz_shift_hours(BOSTON, TOR) == 0


def test_tz_shift_unknown_team_returns_zero():
    assert tz_shift_hours("9999999999", BOSTON) == 0


def test_haversine_known_lat_lon_pair():
    # NYC (40.71, -74.01) to LA (34.05, -118.24) is ~2451 miles.
    d = haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
    assert 2400 <= d <= 2500, f"unexpected NYC→LA distance: {d}"


def test_get_location_returns_expected_fields():
    loc = get_location(BOSTON)
    assert loc is not None
    assert loc.abbr == "BOS"
    assert loc.tz_offset == 5
    assert 42 < loc.lat < 43
