import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    xgb = None

from database import NBADatabase
from team_locations import get_location, distance_between, tz_shift_hours

logger = logging.getLogger(__name__)

# Columns used for rolling pre-game stat features.
# Each "team-perspective stat" is taken from `<stat>_home` when the team played
# at home, or `<stat>_away` when they played away — see _build_history_index.
_STAT_COLUMNS = ('fg_pct', 'ft_pct', 'fg3_pct', 'ast', 'reb')

# Rolling window for stat means (number of prior games per team).
_STAT_ROLL_N = 10

# Directory where trained models are persisted. Override via NBA_MODELS_DIR
# (used by the test suite so train_* doesn't pollute the real data/models/).
MODELS_DIR = Path(
    os.environ.get(
        "NBA_MODELS_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "models"),
    )
)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical DB type names used in models.type
DB_TYPE = {
    "decision_tree": "DecisionTree",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "baseline": "Baseline",
}


class PredictiveModels:
    def __init__(self, db: NBADatabase):
        self.db = db
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, pd.DataFrame] = {}
        self.feature_means: Dict[str, float] = {}
        # In-memory caches built by _build_history_index
        self._team_games: Dict[str, pd.DataFrame] = {}
        self._history_cache: Dict[tuple, float] = {}

    # ------------------------------------------------------------------ #
    # Feature preparation                                                  #
    # ------------------------------------------------------------------ #

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for ML models.
        Returns: (features_df, target_series)
        """
        df = df.copy()
        features = pd.DataFrame()

        # Team encoding
        if 'home_team' in df.columns and 'away_team' in df.columns:
            all_teams = sorted(
                set(df['home_team'].dropna().unique()) |
                set(df['away_team'].dropna().unique())
            )
            team_encoder = LabelEncoder()
            team_encoder.fit(all_teams)
            features['home_team_encoded'] = team_encoder.transform(df['home_team'])
            features['away_team_encoded'] = team_encoder.transform(df['away_team'])
            self.label_encoders['teams'] = team_encoder

        # Date-based features
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            features['month'] = df['game_date'].dt.month
            features['day_of_week'] = df['game_date'].dt.dayofweek
            features['day_of_year'] = df['game_date'].dt.dayofyear

        # Season features
        if 'season' in df.columns:
            season_encoder = LabelEncoder()
            features['season_encoded'] = season_encoder.fit_transform(
                df['season'].astype(str)
            )
            self.label_encoders['season'] = season_encoder

        # Rolling & h2h features (leakage-safe)
        features = self._add_historical_features(df, features)

        # Target variable
        target = None
        if 'result' in df.columns:
            result_encoder = LabelEncoder()
            target = pd.Series(result_encoder.fit_transform(df['result'].astype(str)))
            self.label_encoders['result'] = result_encoder
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            result = df.apply(
                lambda r: 'home_win' if float(r['home_score']) > float(r['away_score']) else 'away_win',
                axis=1
            )
            result_encoder = LabelEncoder()
            target = pd.Series(result_encoder.fit_transform(result))
            self.label_encoders['result'] = result_encoder

        # Fill and store feature names + means
        features = features.fillna(features.mean(numeric_only=True))
        self.feature_names = list(features.columns)
        self.feature_means = {
            col: float(features[col].mean()) for col in features.columns
        }

        return features, target

    # ------------------------------------------------------------------ #
    # Leakage-safe rolling / h2h features                                 #
    # ------------------------------------------------------------------ #

    def _build_history_index(self, all_games: pd.DataFrame):
        """
        Build per-team numpy arrays (sorted dates + win indicators) for O(log n) lookups.
        Also stores per (home, away) pair h2h data, per-team home/away venue indices,
        per-team rolling box-score stats, per-team venue history, and a chronological
        Elo timeline.
        """
        self._team_dates: Dict[str, np.ndarray] = {}   # team -> sorted datetime64 array
        self._team_wins: Dict[str, np.ndarray] = {}    # team -> int8 array (1=win, 0=loss)
        self._team_home_dates: Dict[str, np.ndarray] = {}  # team -> sorted dates when AT HOME
        self._team_home_wins: Dict[str, np.ndarray] = {}   # team -> wins on those home games
        self._team_away_dates: Dict[str, np.ndarray] = {}  # team -> sorted dates when AWAY
        self._team_away_wins: Dict[str, np.ndarray] = {}   # team -> wins on those away games
        self._h2h_dates: Dict[tuple, np.ndarray] = {}  # (home,away) -> sorted datetime64
        self._h2h_wins: Dict[tuple, np.ndarray] = {}   # (home,away) -> int8 (1=home won)
        self._team_stats: Dict[str, Dict[str, np.ndarray]] = {}  # team -> {stat -> float64 arr}
        self._team_venues: Dict[str, np.ndarray] = {}  # team -> team_id of host arena per game
        self._history_cache = {}

        # Deduplicate identical (date, home, away) rows BEFORE any per-team work
        # so rolling-winrate / venue-winrate / stat-mean don't double-count the 29
        # duplicate GAME_IDs in games.csv. _compute_elo_history does the same at
        # its own scope; doing it here keeps the team/h2h/venue paths consistent.
        all_games = all_games.drop_duplicates(
            subset=['game_date', 'home_team', 'away_team'], keep='first'
        ).reset_index(drop=True)

        # Pre-compute league-wide stat means as cold-start fallbacks.
        self._league_stat_means: Dict[str, float] = {}
        for stat in _STAT_COLUMNS:
            both = []
            if f'{stat}_home' in all_games.columns:
                both.append(pd.to_numeric(all_games[f'{stat}_home'], errors='coerce'))
            if f'{stat}_away' in all_games.columns:
                both.append(pd.to_numeric(all_games[f'{stat}_away'], errors='coerce'))
            if both:
                combined = pd.concat(both).dropna().values
                self._league_stat_means[stat] = float(combined.mean()) if combined.size else 0.0
            else:
                self._league_stat_means[stat] = 0.0

        all_teams = pd.concat(
            [all_games['home_team'], all_games['away_team']]
        ).dropna().unique()

        for team in all_teams:
            mask = (all_games['home_team'] == team) | (all_games['away_team'] == team)
            tg = all_games[mask].sort_values('game_date').reset_index(drop=True)

            dates = tg['game_date'].values.astype('datetime64[ns]')
            result_home_win = (tg['result'] == 'home_win').values
            result_away_win = (tg['result'] == 'away_win').values
            is_home = (tg['home_team'] == team).values
            is_away = (tg['away_team'] == team).values

            wins = ((result_home_win & is_home) | (result_away_win & is_away)).astype(np.int8)
            self._team_dates[team] = dates
            self._team_wins[team] = wins

            # Venue-specific indices (already in chronological order)
            self._team_home_dates[team] = dates[is_home]
            self._team_home_wins[team] = (result_home_win[is_home]).astype(np.int8)
            self._team_away_dates[team] = dates[is_away]
            self._team_away_wins[team] = (result_away_win[is_away]).astype(np.int8)

            # Venue history: home_team column IS the host arena holder.
            self._team_venues[team] = tg['home_team'].values

            # Box-score stats from this team's perspective. When the team played
            # at home, take *_home; when away, take *_away.
            team_stats: Dict[str, np.ndarray] = {}
            for stat in _STAT_COLUMNS:
                h_col, a_col = f'{stat}_home', f'{stat}_away'
                if h_col in tg.columns and a_col in tg.columns:
                    h_vals = pd.to_numeric(tg[h_col], errors='coerce').values
                    a_vals = pd.to_numeric(tg[a_col], errors='coerce').values
                    team_stats[stat] = np.where(is_home, h_vals, a_vals).astype(np.float64)
                else:
                    team_stats[stat] = np.full(len(tg), np.nan, dtype=np.float64)
            self._team_stats[team] = team_stats

        # Build h2h index for each (home_team, away_team) pair
        for (ht, at), grp in all_games.groupby(['home_team', 'away_team']):
            grp_s = grp.sort_values('game_date')
            dates_ht_at = grp_s['game_date'].values.astype('datetime64[ns]')
            wins_ht = (grp_s['result'] == 'home_win').astype(np.int8).values
            self._h2h_dates[(ht, at)] = dates_ht_at
            self._h2h_wins[(ht, at)] = wins_ht

        # Elo timeline (forward pass, leakage-safe)
        self._compute_elo_history(all_games)

    def _rolling_winrate(self, team: str, before_date, n: int = 5) -> float:
        """Win rate of `team` in its last `n` games strictly before `before_date` (O(log n))."""
        key = (team, str(before_date), n)
        if key in self._history_cache:
            return self._history_cache[key]

        if team not in self._team_dates:
            self._history_cache[key] = 0.5
            return 0.5

        ts = np.datetime64(before_date, 'ns')
        idx = int(np.searchsorted(self._team_dates[team], ts, side='left'))

        if idx < 2:
            self._history_cache[key] = 0.5
            return 0.5

        start = max(0, idx - n)
        wins_slice = self._team_wins[team][start:idx]
        rate = float(wins_slice.mean())
        self._history_cache[key] = rate
        return rate

    def _h2h_winrate(self, home_team: str, away_team: str, before_date) -> float:
        """
        Rate of home_team wins in all prior matchups between home_team and away_team,
        regardless of venue. Strictly before `before_date`.
        """
        key = ('h2h', home_team, away_team, str(before_date))
        if key in self._history_cache:
            return self._history_cache[key]

        ts = np.datetime64(before_date, 'ns')
        total = 0
        home_wins = 0

        for pair, home_is_home in (
            ((home_team, away_team), True),
            ((away_team, home_team), False),
        ):
            if pair not in self._h2h_dates:
                continue
            idx = int(np.searchsorted(self._h2h_dates[pair], ts, side='left'))
            if idx == 0:
                continue
            w = self._h2h_wins[pair][:idx]
            total += idx
            if home_is_home:
                home_wins += int(w.sum())
            else:
                # wins array is 1 when the row's home team won; away_team was home here,
                # so home_team (our perspective) winning means row's away team won → w==0
                home_wins += int((1 - w).sum())

        if total == 0:
            self._history_cache[key] = 0.5
            return 0.5

        rate = home_wins / total
        self._history_cache[key] = rate
        return rate

    # ------------------------------------------------------------------ #
    # Elo rating (FiveThirtyEight-style)                                  #
    # ------------------------------------------------------------------ #

    # Elo hyperparameters
    _ELO_K = 20
    _ELO_HCA = 100          # home-court advantage in Elo points
    _ELO_INIT = 1500.0      # cold-start Elo for unseen teams
    _ELO_MEAN = 1505.0      # mean for season-boundary regression
    _ELO_REGRESSION = 0.25  # fraction regressed toward mean at season boundary

    def _compute_elo_history(self, all_games: pd.DataFrame):
        """
        Forward pass over all games (chronological) computing pre-game Elo for both teams.
        Builds:
        - self._elo_lookup: DataFrame indexed by (game_date, home_team, away_team) with
          home_elo_pre and away_elo_pre columns (used for training features).
        - self._team_last_elo: dict team -> most recent post-game Elo (for predict_single).
        - self._team_last_season: dict team -> most recent season seen (for predict_single).
        """
        # Drop any duplicate (date, home, away) rows so the Elo timeline is well-defined
        # and the merge later won't produce cross-product rows.
        games_sorted = (
            all_games
            .drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='first')
            .sort_values('game_date')
            .reset_index(drop=True)
        )
        n = len(games_sorted)

        home_teams = games_sorted['home_team'].values
        away_teams = games_sorted['away_team'].values
        results = games_sorted['result'].values
        if 'season' in games_sorted.columns:
            seasons = games_sorted['season'].values
        else:
            seasons = np.array([None] * n)

        home_elo_pre = np.empty(n, dtype=np.float64)
        away_elo_pre = np.empty(n, dtype=np.float64)
        elo: Dict[str, float] = {}
        last_season: Dict[str, Any] = {}

        for i in range(n):
            ht = home_teams[i]
            at = away_teams[i]
            season = seasons[i]

            r_home = elo.get(ht, self._ELO_INIT)
            r_away = elo.get(at, self._ELO_INIT)

            # Season-boundary regression toward the mean
            if season is not None and not (isinstance(season, float) and np.isnan(season)):
                if ht in last_season and str(last_season[ht]) != str(season):
                    r_home = (1.0 - self._ELO_REGRESSION) * r_home + self._ELO_REGRESSION * self._ELO_MEAN
                if at in last_season and str(last_season[at]) != str(season):
                    r_away = (1.0 - self._ELO_REGRESSION) * r_away + self._ELO_REGRESSION * self._ELO_MEAN
                last_season[ht] = season
                last_season[at] = season

            home_elo_pre[i] = r_home
            away_elo_pre[i] = r_away

            # Update post-game Elo
            home_won = 1.0 if results[i] == 'home_win' else 0.0
            expected_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - self._ELO_HCA) / 400.0))
            delta = self._ELO_K * (home_won - expected_home)
            elo[ht] = r_home + delta
            elo[at] = r_away - delta

        lookup = games_sorted[['game_date', 'home_team', 'away_team']].copy()
        lookup['game_date'] = pd.to_datetime(lookup['game_date'])
        lookup['home_elo_pre'] = home_elo_pre
        lookup['away_elo_pre'] = away_elo_pre
        self._elo_lookup = lookup
        self._team_last_elo = elo
        self._team_last_season = last_season

    def _pre_game_elo(self, team: str, before_date, current_season=None) -> float:
        """
        Elo a team carries INTO a game starting on `before_date`, with optional
        season-boundary regression applied if `current_season` differs from the
        team's most recent season. Used by predict_single (training uses the
        pre-computed _elo_lookup directly).
        """
        if not hasattr(self, '_team_last_elo') or team not in self._team_last_elo:
            return self._ELO_INIT
        r = float(self._team_last_elo[team])
        if current_season is not None and team in self._team_last_season:
            if str(self._team_last_season[team]) != str(current_season):
                r = (1.0 - self._ELO_REGRESSION) * r + self._ELO_REGRESSION * self._ELO_MEAN
        return r

    # ------------------------------------------------------------------ #
    # Rest days and venue-specific win-rate                               #
    # ------------------------------------------------------------------ #

    def _days_since_last(self, team: str, before_date, default: int = 7) -> int:
        """Days since the team last played, strictly before `before_date`. Capped at `default`."""
        if team not in self._team_dates:
            return default
        ts = np.datetime64(before_date, 'ns')
        idx = int(np.searchsorted(self._team_dates[team], ts, side='left'))
        if idx == 0:
            return default
        prev_ts = self._team_dates[team][idx - 1]
        days = int((ts - prev_ts) / np.timedelta64(1, 'D'))
        return min(days, default)

    def _rolling_venue_winrate(self, team: str, before_date, venue: str, n: int = 30) -> float:
        """
        Win-rate of `team` in its last `n` games at the given venue ('home' or 'away'),
        strictly before `before_date`. Cold-start 0.5.
        """
        if venue == 'home':
            dates_arr = self._team_home_dates.get(team)
            wins_arr = self._team_home_wins.get(team)
        else:
            dates_arr = self._team_away_dates.get(team)
            wins_arr = self._team_away_wins.get(team)

        if dates_arr is None or len(dates_arr) == 0:
            return 0.5
        ts = np.datetime64(before_date, 'ns')
        idx = int(np.searchsorted(dates_arr, ts, side='left'))
        if idx < 2:
            return 0.5
        start = max(0, idx - n)
        return float(wins_arr[start:idx].mean())

    def _rolling_stat_mean(self, team: str, before_date, stat: str, n: int = _STAT_ROLL_N) -> float:
        """
        Mean of `stat` for `team`'s last `n` games strictly before `before_date`.
        Falls back to the pre-computed league mean when there is no prior history.
        """
        league_mean = self._league_stat_means.get(stat, 0.0)
        if team not in self._team_dates or team not in self._team_stats:
            return league_mean
        stat_arr = self._team_stats[team].get(stat)
        if stat_arr is None or len(stat_arr) == 0:
            return league_mean
        ts = np.datetime64(before_date, 'ns')
        idx = int(np.searchsorted(self._team_dates[team], ts, side='left'))
        if idx == 0:
            return league_mean
        start = max(0, idx - n)
        window = stat_arr[start:idx]
        finite = window[np.isfinite(window)]
        if finite.size == 0:
            return league_mean
        return float(finite.mean())

    def _last_venue(self, team: str, before_date):
        """team_id of the arena where `team` played their most recent game strictly
        before `before_date`. Returns None if no prior game."""
        if team not in self._team_dates:
            return None
        ts = np.datetime64(before_date, 'ns')
        idx = int(np.searchsorted(self._team_dates[team], ts, side='left'))
        if idx == 0:
            return None
        return self._team_venues[team][idx - 1]

    # ------------------------------------------------------------------ #
    # Vectorized fast path                                                #
    # ------------------------------------------------------------------ #

    def _precompute_team_pregame_features(self) -> pd.DataFrame:
        """One-shot vectorized build of per-(team, game_date) PRE-game features.

        Each numeric column reflects the team's state STRICTLY before that game —
        same invariant as the per-call helpers, but built with cumsum + searchsorted
        instead of an O(N) Python loop per query.
        """
        frames: List[pd.DataFrame] = []
        for team, dates in self._team_dates.items():
            n = len(dates)
            if n == 0:
                continue
            wins = self._team_wins[team]
            venues = self._team_venues[team]

            df_t = pd.DataFrame({
                'team': team,
                'game_date': pd.to_datetime(dates),
            })

            # Last-5 win rate (pre-game) — shift(1) so row i sees games < i.
            wins_s = pd.Series(wins.astype(float))
            df_t['_last5_wr'] = (
                wins_s.shift(1).rolling(5, min_periods=2).mean().fillna(0.5).values
            )

            # Rest days since previous game (capped at 7, default 7 for first game).
            delta_days = (pd.Series(dates) - pd.Series(dates).shift(1)).dt.days
            rest = delta_days.clip(upper=7).fillna(7).astype(int).values
            df_t['_rest_days'] = rest
            df_t['_b2b'] = (rest <= 1).astype(np.int8)

            # Venue-specific rolling win rates (last 30 home/away games).
            # cumsum trick + searchsorted = fully vectorized.
            for venue, dates_v, wins_v in (
                ('home', self._team_home_dates.get(team), self._team_home_wins.get(team)),
                ('away', self._team_away_dates.get(team), self._team_away_wins.get(team)),
            ):
                col = f'_{venue}_venue_wr'
                if dates_v is None or len(dates_v) == 0:
                    df_t[col] = 0.5
                    continue
                idx = np.searchsorted(dates_v, dates, side='left')
                cumsum_v = np.concatenate([[0], np.cumsum(wins_v.astype(np.int64))])
                window = 30
                upper = idx
                lower = np.maximum(0, idx - window)
                sum_wins = cumsum_v[upper] - cumsum_v[lower]
                count = upper - lower
                df_t[col] = np.where(count >= 2, sum_wins / np.maximum(count, 1), 0.5)

            # Rolling box-score stat means (window = _STAT_ROLL_N).
            team_stats = self._team_stats.get(team, {})
            for stat in _STAT_COLUMNS:
                league_mean = self._league_stat_means.get(stat, 0.0)
                stat_arr = team_stats.get(stat)
                if stat_arr is None or len(stat_arr) == 0:
                    df_t[f'_{stat}_l10'] = league_mean
                    continue
                s = pd.Series(stat_arr)
                roll = s.shift(1).rolling(_STAT_ROLL_N, min_periods=1).mean()
                df_t[f'_{stat}_l10'] = roll.fillna(league_mean).values

            # Previous venue (the team_id that hosted the team's prior game).
            prev = pd.Series(venues, dtype=object).shift(1)
            df_t['_prev_venue'] = prev.where(prev.notna(), None).values

            frames.append(df_t)

        if not frames:
            return pd.DataFrame(columns=['team', 'game_date'])
        out = pd.concat(frames, ignore_index=True)
        # Dedupe defensively — if a team somehow has two games on one date,
        # keep the first to avoid cross-product blowups in the merge.
        return out.drop_duplicates(subset=['team', 'game_date'], keep='first')

    def _compute_h2h_vectorized(self, df_work: pd.DataFrame) -> np.ndarray:
        """Pre-game h2h home-wins rate, per row of df_work. Aggregates both
        (ht, at) and (at, ht) orientations to mirror `_h2h_winrate`."""
        # Pre-cumulate per pair.
        pair_cumsum: Dict[tuple, tuple] = {}
        for pair, dates in self._h2h_dates.items():
            wins = self._h2h_wins[pair].astype(np.int64)
            pair_cumsum[pair] = (dates, np.concatenate([[0], np.cumsum(wins)]))

        rates = np.full(len(df_work), 0.5, dtype=np.float64)
        # Group input rows by pair so we can vectorize per-pair lookups.
        for (ht, at), grp in df_work.groupby(['home_team', 'away_team'], sort=False):
            idx_rows = grp.index.to_numpy()
            qdates = grp['game_date'].values.astype('datetime64[ns]')

            # Forward orientation: (ht, at). ht wins = cumsum value.
            fwd = pair_cumsum.get((ht, at))
            if fwd is not None:
                fdates, fcum = fwd
                f_idx = np.searchsorted(fdates, qdates, side='left')
                fwd_total = f_idx
                fwd_ht_wins = fcum[f_idx]
            else:
                fwd_total = np.zeros(len(grp), dtype=np.int64)
                fwd_ht_wins = np.zeros(len(grp), dtype=np.int64)

            # Reverse orientation: (at, ht) — at was home. ht wins = total - at_wins.
            rev = pair_cumsum.get((at, ht))
            if rev is not None:
                rdates, rcum = rev
                r_idx = np.searchsorted(rdates, qdates, side='left')
                rev_total = r_idx
                rev_at_wins = rcum[r_idx]
                rev_ht_wins = rev_total - rev_at_wins
            else:
                rev_total = np.zeros(len(grp), dtype=np.int64)
                rev_ht_wins = np.zeros(len(grp), dtype=np.int64)

            total = fwd_total + rev_total
            ht_wins = fwd_ht_wins + rev_ht_wins
            grp_rate = np.where(total > 0, ht_wins / np.maximum(total, 1), 0.5)
            # `rates` is indexed by df_work.index order; map via integer positions.
            # df_work has a fresh range index in callers, so idx_rows are positional.
            rates[idx_rows] = grp_rate
        return rates

    @staticmethod
    def _haversine_vec(lat1, lon1, lat2, lon2):
        rlat1 = np.radians(lat1); rlat2 = np.radians(lat2)
        dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
        return 2 * 3958.7613 * np.arcsin(np.sqrt(a))

    def _compute_travel_tz_vectorized(self, df_work: pd.DataFrame, features: pd.DataFrame) -> None:
        """Build home/away travel-distance and tz-shift columns on `features`."""
        # If pregame was empty (no game history), the *_prev_venue columns were
        # never produced by the merge. Fall back to zeros instead of KeyError'ing.
        if 'home_prev_venue' not in df_work.columns or 'away_prev_venue' not in df_work.columns:
            n = len(df_work)
            features['home_travel_dist'] = np.zeros(n, dtype=np.float64)
            features['away_travel_dist'] = np.zeros(n, dtype=np.float64)
            features['home_tz_shift'] = np.zeros(n, dtype=np.int8)
            features['away_tz_shift'] = np.zeros(n, dtype=np.int8)
            return

        # Collect every team id we might look up (home, away, both prev_venues).
        unique_ids = set()
        for col in ('home_team', 'away_team', 'home_prev_venue', 'away_prev_venue'):
            if col in df_work.columns:
                unique_ids.update(df_work[col].dropna().tolist())

        lat_map: Dict[Any, float] = {}
        lon_map: Dict[Any, float] = {}
        tz_map: Dict[Any, float] = {}
        for tid in unique_ids:
            loc = get_location(tid)
            if loc is not None:
                lat_map[tid] = loc.lat
                lon_map[tid] = loc.lon
                tz_map[tid] = float(loc.tz_offset)

        cur_lat = df_work['home_team'].map(lat_map).astype(float).values
        cur_lon = df_work['home_team'].map(lon_map).astype(float).values
        cur_tz = df_work['home_team'].map(tz_map).astype(float).values

        h_lat = df_work['home_prev_venue'].map(lat_map).astype(float).values
        h_lon = df_work['home_prev_venue'].map(lon_map).astype(float).values
        h_tz = df_work['home_prev_venue'].map(tz_map).astype(float).values
        a_lat = df_work['away_prev_venue'].map(lat_map).astype(float).values
        a_lon = df_work['away_prev_venue'].map(lon_map).astype(float).values
        a_tz = df_work['away_prev_venue'].map(tz_map).astype(float).values

        with np.errstate(invalid='ignore'):
            h_travel = self._haversine_vec(h_lat, h_lon, cur_lat, cur_lon)
            a_travel = self._haversine_vec(a_lat, a_lon, cur_lat, cur_lon)
        features['home_travel_dist'] = np.where(np.isnan(h_travel), 0.0, h_travel)
        features['away_travel_dist'] = np.where(np.isnan(a_travel), 0.0, a_travel)

        h_shift = h_tz - cur_tz
        a_shift = a_tz - cur_tz
        features['home_tz_shift'] = np.where(np.isnan(h_shift), 0, h_shift).astype(np.int8)
        features['away_tz_shift'] = np.where(np.isnan(a_shift), 0, a_shift).astype(np.int8)

    def _add_historical_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add leakage-safe rolling, h2h, Elo, rest, and venue-winrate features.

        Vectorized end-to-end: one precompute over all teams + two merges + two
        groupby passes for h2h/travel. No per-row Python loops over searchsorted.
        """
        if not {'home_team', 'away_team', 'game_date'}.issubset(df.columns):
            return features

        # Load ALL game data from DB for history.
        all_games = self.db.get_game_data()
        if len(all_games) == 0:
            all_games = df.copy()

        all_games['game_date'] = pd.to_datetime(all_games['game_date'])
        df_work = df.copy().reset_index(drop=True)
        df_work['game_date'] = pd.to_datetime(df_work['game_date'])

        self._build_history_index(all_games)
        pregame = self._precompute_team_pregame_features()

        # --- Merge per-team pregame features onto BOTH sides ----------------
        pregame_h = pregame.add_suffix('_h').rename(columns={'team_h': 'home_team', 'game_date_h': 'game_date'})
        pregame_a = pregame.add_suffix('_a').rename(columns={'team_a': 'away_team', 'game_date_a': 'game_date'})

        df_work = df_work.merge(pregame_h, on=['home_team', 'game_date'], how='left')
        df_work = df_work.merge(pregame_a, on=['away_team', 'game_date'], how='left')

        # --- Populate the simple per-team features ---------------------------
        features['home_team_last5_winrate'] = df_work['_last5_wr_h'].fillna(0.5).values
        features['away_team_last5_winrate'] = df_work['_last5_wr_a'].fillna(0.5).values

        home_rest = df_work['_rest_days_h'].fillna(7).astype(int).values
        away_rest = df_work['_rest_days_a'].fillna(7).astype(int).values
        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['home_b2b'] = df_work['_b2b_h'].fillna(0).astype(np.int8).values
        features['away_b2b'] = df_work['_b2b_a'].fillna(0).astype(np.int8).values
        features['rest_diff'] = home_rest - away_rest

        features['home_team_home_winrate'] = df_work['_home_venue_wr_h'].fillna(0.5).values
        features['away_team_away_winrate'] = df_work['_away_venue_wr_a'].fillna(0.5).values

        for stat in _STAT_COLUMNS:
            league = self._league_stat_means.get(stat, 0.0)
            features[f'home_{stat}_l{_STAT_ROLL_N}'] = df_work[f'_{stat}_l10_h'].fillna(league).values
            features[f'away_{stat}_l{_STAT_ROLL_N}'] = df_work[f'_{stat}_l10_a'].fillna(league).values

        # --- Vectorized h2h --------------------------------------------------
        features['h2h_home_wins_rate'] = self._compute_h2h_vectorized(df_work)

        # --- Elo (already a single merge) ------------------------------------
        elo_key = df_work[['game_date', 'home_team', 'away_team']]
        elo_merged = elo_key.merge(
            self._elo_lookup, on=['game_date', 'home_team', 'away_team'], how='left'
        )
        home_elo = elo_merged['home_elo_pre'].fillna(self._ELO_INIT).values
        away_elo = elo_merged['away_elo_pre'].fillna(self._ELO_INIT).values
        features['home_elo_pre'] = home_elo
        features['away_elo_pre'] = away_elo
        features['elo_diff'] = home_elo - away_elo

        # --- Travel + tz (vectorized haversine + integer offset diff) --------
        df_work = df_work.rename(columns={'_prev_venue_h': 'home_prev_venue',
                                          '_prev_venue_a': 'away_prev_venue'})
        self._compute_travel_tz_vectorized(df_work, features)

        # --- Playoff flag ----------------------------------------------------
        if 'season_type' in df_work.columns:
            features['is_playoff'] = (
                df_work['season_type'].isin(['playoffs', 'play_in']).astype(np.int8).values
            )
        else:
            features['is_playoff'] = np.zeros(len(df_work), dtype=np.int8)

        return features

    # ------------------------------------------------------------------ #
    # Model persistence                                                    #
    # ------------------------------------------------------------------ #

    def _persist_model(self, model, model_type: str, import_id: int) -> str:
        """
        Save model .pkl and sidecar .json to MODELS_DIR.
        Returns the absolute path to the .pkl file.
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        pkl_path = str(MODELS_DIR / f"{model_type}_{import_id}_{ts}.pkl")
        json_path = pkl_path.replace('.pkl', '.json')

        joblib.dump(model, pkl_path)

        # Serialize label encoders (classes only, not the objects)
        encoders_meta: Dict[str, List] = {}
        for key, enc in self.label_encoders.items():
            if hasattr(enc, 'classes_'):
                encoders_meta[key] = enc.classes_.tolist()

        sidecar = {
            "feature_names": self.feature_names,
            "label_encoders": encoders_meta,
            "feature_means": self.feature_means,
            "model_type": model_type,
            "import_id": import_id,
            "trained_at": datetime.now().isoformat(),
        }
        with open(json_path, 'w') as f:
            json.dump(sidecar, f)

        return pkl_path

    def load_model(self, model_id: int) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a persisted model by its DB id.
        Returns (estimator, metadata_dict).
        Raises FileNotFoundError if the pkl file is missing.
        """
        row = self.db.get_model_by_id(model_id)
        if row is None:
            raise ValueError(f"No model found with id={model_id}")

        pkl_path = row.get('model_path')
        # model_path may have been stored as an absolute path on a different host
        # (e.g. trained on the host, then loaded inside a container where models
        # live at MODELS_DIR). If the stored path is missing, resolve by filename
        # under the current MODELS_DIR so models stay portable across machines.
        if pkl_path and not os.path.exists(pkl_path):
            candidate = MODELS_DIR / os.path.basename(pkl_path)
            if candidate.exists():
                pkl_path = str(candidate)
        if not pkl_path or not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Model file not found for model_id={model_id}: {pkl_path}"
            )

        json_path = pkl_path.replace('.pkl', '.json')
        model = joblib.load(pkl_path)

        meta: Dict[str, Any] = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)

        return model, meta

    # ------------------------------------------------------------------ #
    # Training helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare_split(self, import_record_id: int, use_temporal_split: bool):
        """
        Shared data loading + split logic for all train methods.
        Returns (X_train, X_test, y_train, y_test, X_full, y_full) where the
        full pair is chronologically sorted for use with TimeSeriesSplit CV.
        """
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")

        # Always build a chronologically sorted (X_full, y_full) for CV
        df_sorted = df.copy()
        if 'game_date' in df_sorted.columns:
            df_sorted['game_date'] = pd.to_datetime(df_sorted['game_date'])
            df_sorted = df_sorted.sort_values('game_date').reset_index(drop=True)
        X_full, y_full = self.prepare_features(df_sorted)
        if y_full is None:
            raise ValueError("No target variable could be created from the data")

        if use_temporal_split and 'game_date' in df_sorted.columns:
            split_idx = int(len(X_full) * 0.8)
            X_train = X_full.iloc[:split_idx]
            X_test = X_full.iloc[split_idx:]
            y_train = y_full.iloc[:split_idx]
            y_test = y_full.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
            )
        return X_train, X_test, y_train, y_test, X_full, y_full

    # ------------------------------------------------------------------ #
    # Calibration & feature-importance access helpers                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _calibration_method_for(model_type: str) -> Optional[str]:
        """Return 'isotonic' / 'sigmoid' / None depending on the base estimator."""
        mt = model_type.lower()
        if mt == 'baseline':
            return None  # DummyClassifier probabilities aren't meaningful — skip
        if mt == 'decision_tree':
            return 'sigmoid'  # isotonic overfits on small DT leaves
        return 'isotonic'  # RF, XGB

    @staticmethod
    def _base_estimator(fitted_model):
        """
        Return the underlying tree-based estimator from a fitted model,
        unwrapping CalibratedClassifierCV if present. Used for feature_importances_.
        """
        if isinstance(fitted_model, CalibratedClassifierCV):
            calibrated = fitted_model.calibrated_classifiers_
            if not calibrated:
                return None
            base = calibrated[0]
            return getattr(base, 'estimator', None)
        return fitted_model

    @staticmethod
    def _averaged_feature_importances(fitted_model) -> Optional[np.ndarray]:
        """
        For a CalibratedClassifierCV: average feature_importances_ across CV folds.
        For a plain estimator: return its feature_importances_.
        Returns None if not available.
        """
        if isinstance(fitted_model, CalibratedClassifierCV):
            importances = []
            for cc in fitted_model.calibrated_classifiers_:
                est = getattr(cc, 'estimator', None)
                if est is not None and hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            if not importances:
                return None
            return np.mean(importances, axis=0)
        if hasattr(fitted_model, 'feature_importances_'):
            return fitted_model.feature_importances_
        return None

    def _train_and_eval(
        self,
        model,
        model_type: str,
        db_model_type: str,
        model_name: str,
        import_record_id: int,
        X_train, X_test, y_train, y_test,
        X_full=None, y_full=None,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Fit model (with probability calibration where applicable),
        compute metrics + confidence + TimeSeriesSplit CV, persist, register in DB.
        Returns (model_id, metrics).
        """
        is_baseline = model_type.lower() == 'baseline'
        calibration_method = self._calibration_method_for(model_type)

        # Wrap with isotonic/sigmoid calibration where appropriate
        if calibration_method is not None:
            fitted = CalibratedClassifierCV(model, method=calibration_method, cv=5)
        else:
            fitted = model
        fitted.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)
        proba = fitted.predict_proba(X_test)
        mean_confidence = float(np.mean(np.max(proba, axis=1)))
        if is_baseline:
            mean_confidence = min(mean_confidence, 0.99)

        metrics: Dict[str, Any] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confidence': round(mean_confidence, 4),
            'calibration_method': calibration_method,
        }

        # TimeSeriesSplit cross-validation (leakage-safe: trains on past, tests on future)
        cv_mean: Optional[float] = None
        cv_std: Optional[float] = None
        if X_full is not None and y_full is not None and len(X_full) >= 100:
            try:
                tscv = TimeSeriesSplit(n_splits=5)
                # Use a fresh unfitted clone for CV so we don't reuse the already-fitted estimator
                if calibration_method is not None:
                    cv_estimator = CalibratedClassifierCV(clone(model), method=calibration_method, cv=3)
                else:
                    cv_estimator = clone(model)
                cv_scores = cross_val_score(cv_estimator, X_full, y_full, cv=tscv, scoring='accuracy', n_jobs=1)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception as e:
                logger.warning(f"TimeSeriesSplit CV failed for {model_name}: {e}")
        metrics['cv_accuracy_mean'] = cv_mean
        metrics['cv_accuracy_std'] = cv_std

        # Feature importance (averaged across calibration folds for trees)
        importance_json = None
        importances = self._averaged_feature_importances(fitted)
        if importances is not None and len(importances) == len(self.feature_names):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances,
            }).sort_values('importance', ascending=False)
            self.feature_importances[model_name] = importance_df
            importance_json = importance_df.to_json(orient='records')

        # Persist fitted (possibly calibrated) model to disk
        pkl_path = self._persist_model(fitted, model_type, import_record_id)

        # Register model row
        model_id = self.db.register_model(
            name=model_name,
            model_type=db_model_type,
            parameters=model.get_params(),
            model_path=pkl_path,
        )

        # Save prediction results
        self.db.save_prediction_results(
            import_record_id=import_record_id,
            model_id=model_id,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1_score'],
            predictions=y_pred.tolist(),
            actual_results=y_test.tolist(),
            confidence=metrics['confidence'],
            feature_importance=importance_json,
            cv_accuracy_mean=cv_mean,
            cv_accuracy_std=cv_std,
            calibration_method=calibration_method,
        )

        return model_id, metrics

    # ------------------------------------------------------------------ #
    # Public train methods                                                 #
    # ------------------------------------------------------------------ #

    def train_decision_tree(self, import_record_id: int, use_temporal_split: bool = False,
                            **params) -> Tuple[int, Dict[str, float]]:
        X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_split(import_record_id, use_temporal_split)
        model_params = {
            'random_state': 42,
            'max_depth': params.get('max_depth', 10),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
        }
        model = DecisionTreeClassifier(**model_params)
        return self._train_and_eval(
            model, 'decision_tree', 'DecisionTree',
            f"Decision Tree - Import {import_record_id}",
            import_record_id, X_train, X_test, y_train, y_test,
            X_full=X_full, y_full=y_full,
        )

    def train_random_forest(self, import_record_id: int, use_temporal_split: bool = False,
                            **params) -> Tuple[int, Dict[str, float]]:
        X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_split(import_record_id, use_temporal_split)
        model_params = {
            'random_state': 42,
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 10),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1),
        }
        model = RandomForestClassifier(**model_params)
        return self._train_and_eval(
            model, 'random_forest', 'RandomForest',
            f"Random Forest - Import {import_record_id}",
            import_record_id, X_train, X_test, y_train, y_test,
            X_full=X_full, y_full=y_full,
        )

    def train_xgboost(self, import_record_id: int, use_temporal_split: bool = False,
                      **params) -> Tuple[int, Dict[str, float]]:
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_split(import_record_id, use_temporal_split)
        n_classes = len(np.unique(y_train))
        model_params = {
            'random_state': 42,
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
            'base_score': 0.5,
        }
        model = xgb.XGBClassifier(**model_params)
        return self._train_and_eval(
            model, 'xgboost', 'XGBoost',
            f"XGBoost - Import {import_record_id}",
            import_record_id, X_train, X_test, y_train, y_test,
            X_full=X_full, y_full=y_full,
        )

    def train_baseline(self, import_record_id: int, use_temporal_split: bool = False,
                       strategy: str = 'most_frequent') -> Tuple[int, Dict[str, float]]:
        X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_split(import_record_id, use_temporal_split)
        model_params = {'strategy': strategy, 'random_state': 42}
        model = DummyClassifier(**model_params)
        return self._train_and_eval(
            model, 'baseline', 'Baseline',
            f"Baseline ({strategy}) - Import {import_record_id}",
            import_record_id, X_train, X_test, y_train, y_test,
            X_full=X_full, y_full=y_full,
        )

    # ------------------------------------------------------------------ #
    # Temporal split                                                       #
    # ------------------------------------------------------------------ #

    def temporal_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2,
                                  date_column: str = 'game_date'):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df_sorted = df.sort_values(date_column)
        X, y = self.prepare_features(df_sorted)
        split_idx = int(len(df_sorted) * (1 - test_size))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #

    def predict_single(self, home_team: str, away_team: str, season: str,
                       model_id: int = None, model_type: str = None) -> Dict[str, Any]:
        """
        Predict outcome for a single matchup using a persisted model.
        model_id takes precedence over model_type.
        model_type falls back to the latest persisted model of that type.
        """
        if model_id is not None:
            model, meta = self.load_model(model_id)
            row = self.db.get_model_by_id(model_id)
            model_name = row['name'] if row else f"Model {model_id}"
        elif model_type is not None:
            # Convert API-style type to DB-style type
            db_type = DB_TYPE.get(model_type.lower(), model_type)
            row = self.db.get_latest_model_by_type(db_type)
            if row is None:
                raise FileNotFoundError(
                    f"No trained model found for type '{model_type}'"
                )
            model_id = row['id']
            model, meta = self.load_model(model_id)
            model_name = row['name']
        else:
            raise ValueError("Must provide model_id or model_type")

        feature_names = meta.get('feature_names', [])
        feature_means = meta.get('feature_means', {})

        # Start with training means as defaults
        feature_dict = {f: feature_means.get(f, 0.0) for f in feature_names}

        # Team encoding
        enc_meta = meta.get('label_encoders', {})
        if 'teams' in enc_meta:
            team_classes = enc_meta['teams']
            team_idx = {t: i for i, t in enumerate(team_classes)}
            if home_team in team_idx and 'home_team_encoded' in feature_dict:
                feature_dict['home_team_encoded'] = float(team_idx[home_team])
            if away_team in team_idx and 'away_team_encoded' in feature_dict:
                feature_dict['away_team_encoded'] = float(team_idx[away_team])

        # Date features (use today)
        today = datetime.now()
        if 'month' in feature_dict:
            feature_dict['month'] = float(today.month)
        if 'day_of_week' in feature_dict:
            feature_dict['day_of_week'] = float(today.weekday())
        if 'day_of_year' in feature_dict:
            feature_dict['day_of_year'] = float(today.timetuple().tm_yday)

        # Season encoding
        if 'season' in enc_meta and 'season_encoded' in feature_dict:
            season_classes = enc_meta['season']
            season_idx = {str(s): i for i, s in enumerate(season_classes)}
            if str(season) in season_idx:
                feature_dict['season_encoded'] = float(season_idx[str(season)])

        # Rolling / h2h / Elo / rest / venue features
        all_games = self.db.get_game_data()
        if len(all_games) > 0:
            all_games['game_date'] = pd.to_datetime(all_games['game_date'])
            self._history_cache = {}
            self._build_history_index(all_games)
            today_ts = pd.Timestamp(today.date())

            if 'home_team_last5_winrate' in feature_dict:
                feature_dict['home_team_last5_winrate'] = self._rolling_winrate(
                    home_team, today_ts, n=5
                )
            if 'away_team_last5_winrate' in feature_dict:
                feature_dict['away_team_last5_winrate'] = self._rolling_winrate(
                    away_team, today_ts, n=5
                )
            if 'h2h_home_wins_rate' in feature_dict:
                feature_dict['h2h_home_wins_rate'] = self._h2h_winrate(
                    home_team, away_team, today_ts
                )

            # Elo (with season-boundary regression applied if the team's last season differs)
            home_elo = self._pre_game_elo(home_team, today_ts, current_season=season)
            away_elo = self._pre_game_elo(away_team, today_ts, current_season=season)
            if 'home_elo_pre' in feature_dict:
                feature_dict['home_elo_pre'] = home_elo
            if 'away_elo_pre' in feature_dict:
                feature_dict['away_elo_pre'] = away_elo
            if 'elo_diff' in feature_dict:
                feature_dict['elo_diff'] = home_elo - away_elo

            # Rest days and back-to-back flags
            home_rest = self._days_since_last(home_team, today_ts)
            away_rest = self._days_since_last(away_team, today_ts)
            if 'home_rest_days' in feature_dict:
                feature_dict['home_rest_days'] = float(home_rest)
            if 'away_rest_days' in feature_dict:
                feature_dict['away_rest_days'] = float(away_rest)
            if 'home_b2b' in feature_dict:
                feature_dict['home_b2b'] = float(1 if home_rest <= 1 else 0)
            if 'away_b2b' in feature_dict:
                feature_dict['away_b2b'] = float(1 if away_rest <= 1 else 0)
            if 'rest_diff' in feature_dict:
                feature_dict['rest_diff'] = float(home_rest - away_rest)

            # Real venue-specific win-rates
            if 'home_team_home_winrate' in feature_dict:
                feature_dict['home_team_home_winrate'] = self._rolling_venue_winrate(
                    home_team, today_ts, 'home'
                )
            if 'away_team_away_winrate' in feature_dict:
                feature_dict['away_team_away_winrate'] = self._rolling_venue_winrate(
                    away_team, today_ts, 'away'
                )

            # Rolling pre-game stat means
            for stat in _STAT_COLUMNS:
                h_key = f'home_{stat}_l{_STAT_ROLL_N}'
                a_key = f'away_{stat}_l{_STAT_ROLL_N}'
                if h_key in feature_dict:
                    feature_dict[h_key] = self._rolling_stat_mean(home_team, today_ts, stat)
                if a_key in feature_dict:
                    feature_dict[a_key] = self._rolling_stat_mean(away_team, today_ts, stat)

            # Travel + tz: current venue is `home_team`'s arena. Use each side's
            # last known venue; cold-start to that team's own city.
            current_venue = home_team
            h_prev = self._last_venue(home_team, today_ts) or home_team
            a_prev = self._last_venue(away_team, today_ts) or away_team
            if 'home_travel_dist' in feature_dict:
                feature_dict['home_travel_dist'] = distance_between(h_prev, current_venue)
            if 'away_travel_dist' in feature_dict:
                feature_dict['away_travel_dist'] = distance_between(a_prev, current_venue)
            if 'home_tz_shift' in feature_dict:
                feature_dict['home_tz_shift'] = float(tz_shift_hours(h_prev, current_venue))
            if 'away_tz_shift' in feature_dict:
                feature_dict['away_tz_shift'] = float(tz_shift_hours(a_prev, current_venue))

        # is_playoff is unknown for an arbitrary user-supplied matchup. Default to
        # regular season — caller can manually set if predicting a playoff game.
        if 'is_playoff' in feature_dict:
            feature_dict['is_playoff'] = 0.0

        # Build ordered feature row as a DataFrame (with column names) so the
        # estimator's "X does not have valid feature names" UserWarning stays quiet.
        X = pd.DataFrame(
            [[feature_dict.get(f, 0.0) for f in feature_names]],
            columns=feature_names,
        )

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        confidence = float(np.max(y_proba[0]))

        # Decode prediction label
        if 'result' in enc_meta:
            result_classes = enc_meta['result']
            pred_label = result_classes[int(y_pred[0])]
        else:
            pred_label = 'home_win' if int(y_pred[0]) == 1 else 'away_win'

        prediction_display = "Home Win" if pred_label == 'home_win' else "Away Win"

        is_baseline = meta.get('model_type', '').lower() == 'baseline'
        if is_baseline:
            confidence = min(confidence, 0.99)

        # Feature importance top-10 (unwrap CalibratedClassifierCV if needed)
        feature_importance = None
        importances = self._averaged_feature_importances(model)
        if importances is not None and feature_names and len(importances) == len(feature_names):
            pairs = sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1], reverse=True
            )[:10]
            feature_importance = {k: round(v, 6) for k, v in pairs}

        return {
            "prediction": prediction_display,
            "predicted_label": pred_label,  # 'home_win' | 'away_win'
            "confidence": round(confidence, 4),
            "confidence_reliable": not is_baseline,
            "model_id": model_id,
            "model_name": model_name,
            "feature_importance": feature_importance,
        }

    # ------------------------------------------------------------------ #
    # Comparison / reporting                                               #
    # ------------------------------------------------------------------ #

    def get_feature_importance(self, model, model_name: str) -> pd.DataFrame:
        importances = self._averaged_feature_importances(model)
        if importances is None or len(importances) != len(self.feature_names):
            return pd.DataFrame()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        self.feature_importances[model_name] = importance_df
        return importance_df

    def get_model_comparison(self, import_record_id: int = None) -> pd.DataFrame:
        results_df = self.db.get_prediction_results(import_record_id=import_record_id)
        models_df = self.db.get_models()

        comparison = results_df.merge(models_df, left_on='model_id', right_on='id', suffixes=('', '_model'))
        cols = ['name', 'type', 'accuracy', 'precision_score', 'recall', 'f1_score',
                'confidence', 'created_date', 'feature_importance']
        available = [c for c in cols if c in comparison.columns]
        numeric_cols = [c for c in ['accuracy', 'precision_score', 'recall', 'f1_score', 'confidence'] if c in available]
        result = comparison[available].copy()
        result[numeric_cols] = result[numeric_cols].round(4)
        return result.sort_values('accuracy', ascending=False)
