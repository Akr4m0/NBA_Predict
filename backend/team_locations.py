"""
Arena coordinates and standard-time UTC offsets for all 30 NBA franchises.

Used to derive two pre-game features:
- travel_distance: haversine miles between a team's previous game venue and the
  current game venue.
- tz_shift: integer hours between the previous and current venue (signed,
  positive = travelled east).

DST is intentionally ignored — the magnitude (1, 2, 3 hours) is what matters
for fatigue/circadian-shift purposes; a single-hour DST blip wouldn't survive
any reasonable feature scaler anyway.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class TeamLocation:
    abbr: str
    name: str
    lat: float
    lon: float
    tz_offset: int  # hours west of UTC (e.g. ET = 5, PT = 8). Standard time only.


# Keyed by the NBA Stats team_id (string).
_LOCATIONS: Dict[str, TeamLocation] = {
    "1610612737": TeamLocation("ATL", "Atlanta Hawks",         33.7573,  -84.3963, 5),
    "1610612738": TeamLocation("BOS", "Boston Celtics",        42.3662,  -71.0621, 5),
    "1610612739": TeamLocation("CLE", "Cleveland Cavaliers",   41.4965,  -81.6882, 5),
    "1610612740": TeamLocation("NOP", "New Orleans Pelicans",  29.9490,  -90.0815, 6),
    "1610612741": TeamLocation("CHI", "Chicago Bulls",         41.8807,  -87.6742, 6),
    "1610612742": TeamLocation("DAL", "Dallas Mavericks",      32.7905,  -96.8103, 6),
    "1610612743": TeamLocation("DEN", "Denver Nuggets",        39.7487, -105.0077, 7),
    "1610612744": TeamLocation("GSW", "Golden State Warriors", 37.7679, -122.3873, 8),
    "1610612745": TeamLocation("HOU", "Houston Rockets",       29.7508,  -95.3621, 6),
    "1610612746": TeamLocation("LAC", "LA Clippers",           34.0431, -118.2675, 8),
    "1610612747": TeamLocation("LAL", "Los Angeles Lakers",    34.0431, -118.2675, 8),
    "1610612748": TeamLocation("MIA", "Miami Heat",            25.7814,  -80.1870, 5),
    "1610612749": TeamLocation("MIL", "Milwaukee Bucks",       43.0451,  -87.9173, 6),
    "1610612750": TeamLocation("MIN", "Minnesota Timberwolves",44.9795,  -93.2761, 6),
    "1610612751": TeamLocation("BKN", "Brooklyn Nets",         40.6826,  -73.9754, 5),
    "1610612752": TeamLocation("NYK", "New York Knicks",       40.7505,  -73.9934, 5),
    "1610612753": TeamLocation("ORL", "Orlando Magic",         28.5392,  -81.3839, 5),
    "1610612754": TeamLocation("IND", "Indiana Pacers",        39.7639,  -86.1556, 5),
    "1610612755": TeamLocation("PHI", "Philadelphia 76ers",    39.9012,  -75.1719, 5),
    "1610612756": TeamLocation("PHX", "Phoenix Suns",          33.4457, -112.0712, 7),
    "1610612757": TeamLocation("POR", "Portland Trail Blazers",45.5316, -122.6668, 8),
    "1610612758": TeamLocation("SAC", "Sacramento Kings",      38.5800, -121.4997, 8),
    "1610612759": TeamLocation("SAS", "San Antonio Spurs",     29.4271,  -98.4375, 6),
    "1610612760": TeamLocation("OKC", "Oklahoma City Thunder", 35.4634,  -97.5151, 6),
    "1610612761": TeamLocation("TOR", "Toronto Raptors",       43.6435,  -79.3791, 5),
    "1610612762": TeamLocation("UTA", "Utah Jazz",             40.7683, -111.9011, 7),
    "1610612763": TeamLocation("MEM", "Memphis Grizzlies",     35.1382,  -90.0505, 6),
    "1610612764": TeamLocation("WAS", "Washington Wizards",    38.8981,  -77.0209, 5),
    "1610612765": TeamLocation("DET", "Detroit Pistons",       42.3411,  -83.0553, 5),
    "1610612766": TeamLocation("CHA", "Charlotte Hornets",     35.2251,  -80.8392, 5),
}


def get_location(team_id) -> Optional[TeamLocation]:
    return _LOCATIONS.get(str(team_id))


# Reverse map: team abbreviation (e.g. "BOS") → NBA Stats team_id (string).
# Used to translate external feeds (balldontlie) that key on abbreviation into
# the NBA Stats IDs this system stores everywhere. All 30 current franchises.
ABBR_TO_ID: Dict[str, str] = {loc.abbr: tid for tid, loc in _LOCATIONS.items()}


def abbr_to_team_id(abbr: Optional[str]) -> Optional[str]:
    """NBA Stats team_id for an abbreviation, or None if unrecognised
    (e.g. a defunct franchise in very old data)."""
    if not abbr:
        return None
    return ABBR_TO_ID.get(abbr.strip().upper())


_EARTH_RADIUS_MILES = 3958.7613


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in statute miles between two lat/lon points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def distance_between(team_a_id, team_b_id) -> float:
    """Distance in miles between two teams' home arenas. 0.0 if either is unknown."""
    a = get_location(team_a_id)
    b = get_location(team_b_id)
    if a is None or b is None:
        return 0.0
    return haversine_miles(a.lat, a.lon, b.lat, b.lon)


def tz_shift_hours(from_team_id, to_team_id) -> int:
    """
    Hours of timezone shift when travelling from `from_team_id`'s venue to
    `to_team_id`'s venue. Positive = travelled east (e.g. PT → ET = +3).
    Returns 0 if either location is unknown.
    """
    a = get_location(from_team_id)
    b = get_location(to_team_id)
    if a is None or b is None:
        return 0
    return a.tz_offset - b.tz_offset
