import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import Dict, Any, Tuple, List
import json
from database import NBADatabase

class PredictiveModels:
    def __init__(self, db: NBADatabase):
        self.db = db
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning models
        Returns: (features_df, target_series)
        """
        df = df.copy()
        
        # Create basic features
        features = pd.DataFrame()
        
        # Team encoding
        if 'home_team' in df.columns and 'away_team' in df.columns:
            # Encode teams
            all_teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
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
            features['season_encoded'] = season_encoder.fit_transform(df['season'])
            self.label_encoders['season'] = season_encoder
        
        # Statistical features from home_stats and away_stats
        features = self._extract_statistical_features(df, features)
        
        # Historical performance features (if we have enough data)
        features = self._add_historical_features(df, features)
        
        # Target variable
        target = None
        if 'result' in df.columns:
            result_encoder = LabelEncoder()
            target = pd.Series(result_encoder.fit_transform(df['result']))
            self.label_encoders['result'] = result_encoder
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            # Create result from scores
            result = df.apply(lambda row: 
                'home_win' if row['home_score'] > row['away_score'] 
                else 'away_win' if row['away_score'] > row['home_score'] 
                else 'tie', axis=1)
            result_encoder = LabelEncoder()
            target = pd.Series(result_encoder.fit_transform(result))
            self.label_encoders['result'] = result_encoder
        
        # Fill missing values
        features = features.fillna(features.mean())
        
        return features, target
    
    def _extract_statistical_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from team stats"""
        
        # Parse JSON stats if they exist
        if 'home_stats' in df.columns:
            home_stats_df = pd.json_normalize(df['home_stats'])
            # Add prefix to avoid column conflicts
            home_stats_df.columns = ['home_' + col for col in home_stats_df.columns]
            features = pd.concat([features, home_stats_df], axis=1)
        
        if 'away_stats' in df.columns:
            away_stats_df = pd.json_normalize(df['away_stats'])
            # Add prefix to avoid column conflicts  
            away_stats_df.columns = ['away_' + col for col in away_stats_df.columns]
            features = pd.concat([features, away_stats_df], axis=1)
        
        # Basic score-based features if available
        if 'home_score' in df.columns and 'away_score' in df.columns:
            features['total_score'] = df['home_score'] + df['away_score']
            features['score_difference'] = df['home_score'] - df['away_score']
        
        return features
    
    def _add_historical_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add historical performance features for teams"""
        
        if 'home_team' in df.columns and 'away_team' in df.columns and 'game_date' in df.columns:
            # Sort by date
            df_sorted = df.sort_values('game_date')
            
            # Calculate recent performance for each team
            features['home_recent_wins'] = 0
            features['away_recent_wins'] = 0
            features['home_recent_games'] = 0
            features['away_recent_games'] = 0
            
            # This is a simplified version - in a real implementation, 
            # you'd calculate rolling windows of team performance
            for team in df['home_team'].unique():
                team_games = df_sorted[(df_sorted['home_team'] == team) | (df_sorted['away_team'] == team)]
                # Add basic team strength indicator
                team_win_rate = self._calculate_team_win_rate(team_games, team)
                features.loc[df['home_team'] == team, 'home_team_strength'] = team_win_rate
                features.loc[df['away_team'] == team, 'away_team_strength'] = team_win_rate
        
        return features
    
    def _calculate_team_win_rate(self, team_games: pd.DataFrame, team: str) -> float:
        """Calculate win rate for a specific team"""
        if len(team_games) == 0:
            return 0.5  # Default to 50% if no data
        
        wins = 0
        total_games = len(team_games)
        
        for _, game in team_games.iterrows():
            if 'result' in game:
                if (game['home_team'] == team and game['result'] == 'home_win') or \
                   (game['away_team'] == team and game['result'] == 'away_win'):
                    wins += 1
            elif 'home_score' in game and 'away_score' in game:
                if (game['home_team'] == team and game['home_score'] > game['away_score']) or \
                   (game['away_team'] == team and game['away_score'] > game['home_score']):
                    wins += 1
        
        return wins / total_games if total_games > 0 else 0.5
    
    def train_decision_tree(self, import_record_id: int, **params) -> Tuple[int, Dict[str, float]]:
        """
        Train a decision tree model on the specified dataset
        Returns: (model_id, performance_metrics)
        """
        # Get data
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if y is None:
            raise ValueError("No target variable could be created from the data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model_params = {
            'random_state': 42,
            'max_depth': params.get('max_depth', 10),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1)
        }
        
        dt_model = DecisionTreeClassifier(**model_params)
        dt_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = dt_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Register model in database
        model_id = self.db.register_model(
            name=f"Decision Tree - Import {import_record_id}",
            model_type="DecisionTree",
            parameters=model_params
        )
        
        # Save results
        self.db.save_prediction_results(
            import_record_id=import_record_id,
            model_id=model_id,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1_score'],
            predictions=y_pred.tolist(),
            actual_results=y_test.tolist()
        )
        
        return model_id, metrics
    
    def train_random_forest(self, import_record_id: int, **params) -> Tuple[int, Dict[str, float]]:
        """
        Train a random forest model on the specified dataset
        Returns: (model_id, performance_metrics)
        """
        # Get data
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if y is None:
            raise ValueError("No target variable could be created from the data")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model_params = {
            'random_state': 42,
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 10),
            'min_samples_split': params.get('min_samples_split', 2),
            'min_samples_leaf': params.get('min_samples_leaf', 1)
        }
        
        rf_model = RandomForestClassifier(**model_params)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Register model in database
        model_id = self.db.register_model(
            name=f"Random Forest - Import {import_record_id}",
            model_type="RandomForest", 
            parameters=model_params
        )
        
        # Save results
        self.db.save_prediction_results(
            import_record_id=import_record_id,
            model_id=model_id,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1_score'],
            predictions=y_pred.tolist(),
            actual_results=y_test.tolist()
        )
        
        return model_id, metrics
    
    def get_model_comparison(self, import_record_id: int = None) -> pd.DataFrame:
        """Compare performance of all models"""
        results_df = self.db.get_prediction_results(import_record_id=import_record_id)
        models_df = self.db.get_models()
        
        # Merge with model information
        comparison = results_df.merge(models_df, left_on='model_id', right_on='id', suffixes=('', '_model'))
        
        # Select relevant columns
        comparison = comparison[[
            'name', 'type', 'accuracy', 'precision_score', 'recall', 'f1_score', 'created_date'
        ]].round(4)
        
        return comparison.sort_values('accuracy', ascending=False)