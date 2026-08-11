import pandas as pd
import os
import logging
import warnings
from typing import Dict, Any, Tuple, List
from database import NBADatabase

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['home_team', 'away_team', 'game_date']
SCORE_COLUMNS = ['home_score', 'away_score']


def validate_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a DataFrame of NBA game data.
    Returns {"valid": bool, "errors": list[str], "warnings": list[str]}.
    """
    errors: List[str] = []
    # Named `warning_msgs` rather than `warnings` so it doesn't shadow the
    # stdlib `warnings` module we need below.
    warning_msgs: List[str] = []

    # Check required columns (team names + date are hard required)
    all_required = REQUIRED_COLUMNS + SCORE_COLUMNS
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return {"valid": False, "errors": errors, "warnings": warning_msgs}

    # Null checks — identity columns are errors, score nulls are warnings
    for col in REQUIRED_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null value(s)")

    for col in SCORE_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            warning_msgs.append(
                f"Column '{col}' has {null_count} null value(s) — "
                f"those rows will be skipped during import"
            )

    # Date parseability — `format=mixed` lets pandas handle ISO + US/EU strings
    # without falling back to per-element dateutil (which warns loudly).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pd.to_datetime(df['game_date'], errors='raise', format='mixed')
    except Exception:
        errors.append("Column 'game_date' contains values that cannot be parsed as dates")

    # Team name non-empty
    for col in ('home_team', 'away_team'):
        bad = df[col].dropna().astype(str).str.strip().eq('').sum()
        if bad > 0:
            errors.append(f"Column '{col}' has {bad} empty/whitespace team name(s)")

    # Scores must be non-negative integers (or NaN — nulls handled above)
    for col in ('home_score', 'away_score'):
        non_null = df[col].dropna()
        negative = (pd.to_numeric(non_null, errors='coerce').fillna(0) < 0).sum()
        if negative > 0:
            bad_indices = df[pd.to_numeric(df[col], errors='coerce') < 0].index.tolist()[:5]
            errors.append(
                f"Column '{col}' has {negative} negative value(s) "
                f"(first rows: {bad_indices})"
            )

    # Tie warnings (home_score == away_score)
    try:
        home = pd.to_numeric(df['home_score'], errors='coerce')
        away = pd.to_numeric(df['away_score'], errors='coerce')
        ties = df[(home == away) & home.notna() & away.notna()]
        for idx in ties.index:
            row = df.loc[idx]
            warning_msgs.append(
                f"Row {idx}: tie game ({row['home_team']} vs {row['away_team']}, "
                f"score {row['home_score']}–{row['away_score']}) will be treated as Away Win"
            )
    except Exception:
        pass

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warning_msgs}


class DataImporter:
    def __init__(self, db: NBADatabase):
        self.db = db

    def import_historical_data(self, file_path: str, description: str = None) -> Tuple[int, pd.DataFrame]:
        """
        Import historical NBA data from CSV or Excel file.
        Returns: (import_record_id, dataframe)
        Raises ValueError with validation errors if the file fails validation.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(filename)[1].lower()

        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Map column variations before validation
        df = self._map_column_variations(df)

        # Validate with the public validator
        validation = validate_csv(df)
        if not validation['valid']:
            raise ValueError(f"Validation failed: {validation['errors']}")

        # Log warnings
        for w in validation['warnings']:
            logger.warning(w)

        # Clean and standardize data
        df = self._clean_data(df)

        # Register import in database
        import_record_id = self.db.register_import(
            filename=filename,
            file_path=file_path,
            record_count=len(df),
            description=description
        )

        # Convert timestamps to strings before saving to database
        df_to_save = df.copy()
        if 'game_date' in df_to_save.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_to_save['game_date']):
                df_to_save['game_date'] = pd.to_datetime(df_to_save['game_date'], errors='coerce')
            df_to_save['game_date'] = df_to_save['game_date'].dt.strftime('%Y-%m-%d')

        # Save game data to database
        self.db.save_game_data(import_record_id, df_to_save)

        return import_record_id, df

    def _validate_data_structure(self, df: pd.DataFrame) -> None:
        """Legacy: kept for backward compatibility. Calls validate_csv internally."""
        result = validate_csv(df)
        if not result['valid']:
            raise ValueError(
                f"Missing required columns. Available columns: {list(df.columns)}"
            )
    
    def _map_column_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variations to standard names"""
        column_mappings = {
            # Home team variations
            'Home Team': 'home_team',
            'home': 'home_team',
            'Home': 'home_team',
            'team_home': 'home_team',

            # Away team variations
            'Away Team': 'away_team',
            'away': 'away_team',
            'Away': 'away_team',
            'team_away': 'away_team',
            'visitor': 'away_team',
            'Visitor': 'away_team',

            # Date variations
            'Date': 'game_date',
            'date': 'game_date',
            'Game Date': 'game_date',
            'GAME_DATE': 'game_date',

            # Score variations
            'Home Score': 'home_score',
            'home_points': 'home_score',
            'PTS_home': 'home_score',
            'Away Score': 'away_score',
            'away_points': 'away_score',
            'PTS_away': 'away_score',
            'visitor_points': 'away_score',

            # Season variations
            'Season': 'season',
            'SEASON': 'season',
            'season_year': 'season',

            # Game ID (used to derive season_type from its leading digit)
            'GAME_ID': 'game_id',
            'Game ID': 'game_id',
            'gameId': 'game_id',

            # Games_most_recent.csv column names
            'hometeamId': 'home_team',
            'awayteamId': 'away_team',
            'gameDate': 'game_date',
            'homeScore': 'home_score',
            'awayScore': 'away_score',

            # Box-score stats (preserved in DB for pre-game rolling features)
            'FG_PCT_home': 'fg_pct_home',
            'FT_PCT_home': 'ft_pct_home',
            'FG3_PCT_home': 'fg3_pct_home',
            'AST_home':    'ast_home',
            'REB_home':    'reb_home',
            'FG_PCT_away': 'fg_pct_away',
            'FT_PCT_away': 'ft_pct_away',
            'FG3_PCT_away': 'fg3_pct_away',
            'AST_away':    'ast_away',
            'REB_away':    'reb_away',
        }

        df = df.rename(columns=column_mappings)
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        df = df.copy()

        # Convert game_date to datetime
        if 'game_date' in df.columns:
            df.loc[:, 'game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

        # Create result column if scores are available.
        # Ties (home_score == away_score) are treated as Away Win per project spec.
        if 'home_score' in df.columns and 'away_score' in df.columns:
            def _result(row):
                try:
                    h, a = float(row['home_score']), float(row['away_score'])
                except (TypeError, ValueError):
                    return None
                if h > a:
                    return 'home_win'
                return 'away_win'  # away win or tie → away win

            df.loc[:, 'result'] = df.apply(_result, axis=1)

        # Derive season_type from GAME_ID's leading digit:
        #   1 = preseason, 2 = regular, 4 = playoffs, 5 = play-in.
        # (NBA Stats API convention.) Anything else / missing → None.
        if 'game_id' in df.columns:
            season_type_map = {'1': 'preseason', '2': 'regular', '4': 'playoffs', '5': 'play_in'}
            df.loc[:, 'season_type'] = (
                df['game_id'].astype(str).str[0].map(season_type_map)
            )

        # Remove rows with missing critical data (including scores — unplayed games)
        drop_cols = ['home_team', 'away_team', 'game_date']
        if 'home_score' in df.columns and 'away_score' in df.columns:
            drop_cols += ['home_score', 'away_score']
        df = df.dropna(subset=drop_cols)

        return df

    def get_data_summary(self, import_record_id: int) -> Dict[str, Any]:
        """Get summary statistics for imported data"""
        df = self.db.get_game_data(import_record_id)
        
        summary = {
            'total_games': len(df),
            'date_range': {
                'start': df['game_date'].min() if 'game_date' in df.columns else None,
                'end': df['game_date'].max() if 'game_date' in df.columns else None
            },
            'unique_teams': set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()),
            'seasons': df['season'].unique().tolist() if 'season' in df.columns else [],
        }
        
        if 'result' in df.columns:
            summary['results_distribution'] = df['result'].value_counts().to_dict()
        
        return summary