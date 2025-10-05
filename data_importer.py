import pandas as pd
import os
from typing import Dict, Any, Tuple, List
from database import NBADatabase

class DataImporter:
    def __init__(self, db: NBADatabase):
        self.db = db
    
    def import_historical_data(self, file_path: str, description: str = None) -> Tuple[int, pd.DataFrame]:
        """
        Import historical NBA data from CSV or Excel file
        Returns: (import_record_id, dataframe)
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
        
        # Validate required columns
        self._validate_data_structure(df)
        
        # Clean and standardize data
        df = self._clean_data(df)
        
        # Register import in database
        import_record_id = self.db.register_import(
            filename=filename,
            file_path=file_path,
            record_count=len(df),
            description=description
        )
        
        # Save game data to database
        self.db.save_game_data(import_record_id, df)
        
        return import_record_id, df
    
    def _validate_data_structure(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe has required columns for NBA data"""
        required_columns = ['home_team', 'away_team', 'game_date']
        
        # Check if any required columns are missing
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to map common column variations
            df = self._map_column_variations(df)
            
            # Check again after mapping
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}. "
                               f"Available columns: {list(df.columns)}")
    
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
            'season_year': 'season'
        }
        
        df = df.rename(columns=column_mappings)
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        df = df.copy()
        
        # Convert game_date to datetime
        if 'game_date' in df.columns:
            df.loc[:, 'game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        
        # Create result column if scores are available
        if 'home_score' in df.columns and 'away_score' in df.columns:
            df.loc[:, 'result'] = df.apply(lambda row: 
                'home_win' if row['home_score'] > row['away_score'] 
                else 'away_win' if row['away_score'] > row['home_score'] 
                else 'tie', axis=1)
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['home_team', 'away_team', 'game_date'])
        
        # Extract team statistics if available
        stat_columns = [col for col in df.columns if any(stat in col.lower() 
                       for stat in ['fg%', 'fga', 'fgm', '3p', 'ft', 'reb', 'ast', 'stl', 'blk', 'to'])]
        
        if stat_columns:
            df = self._extract_team_stats(df, stat_columns)
        
        return df
    
    def _extract_team_stats(self, df: pd.DataFrame, stat_columns: List[str]) -> pd.DataFrame:
        """Extract team statistics into structured format"""
        df = df.copy()
        
        # Separate home and away stats
        home_stat_cols = [col for col in stat_columns if 'home' in col.lower()]
        away_stat_cols = [col for col in stat_columns if 'away' in col.lower() or 'visitor' in col.lower()]
        
        # Create structured stats dictionaries
        if home_stat_cols:
            df.loc[:, 'home_stats'] = df[home_stat_cols].to_dict('records')
        else:
            df.loc[:, 'home_stats'] = [{}] * len(df)
            
        if away_stat_cols:
            df.loc[:, 'away_stats'] = df[away_stat_cols].to_dict('records')
        else:
            df.loc[:, 'away_stats'] = [{}] * len(df)
        
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