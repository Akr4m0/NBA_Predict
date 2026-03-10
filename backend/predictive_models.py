import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier
from typing import Dict, Any, Tuple, List
import json
from database import NBADatabase
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class PredictiveModels:
    def __init__(self, db: NBADatabase):
        self.db = db
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importances = {}
        
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

        # Store feature names for later use
        self.feature_names = list(features.columns)

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

    def temporal_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, date_column: str = 'game_date'):
        """
        Perform temporal train/test split based on date
        Returns: (X_train, X_test, y_train, y_test)
        """
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Sort by date
        df_sorted = df.sort_values(date_column)

        # Prepare features
        X, y = self.prepare_features(df_sorted)

        # Calculate split index
        split_idx = int(len(df_sorted) * (1 - test_size))

        # Split temporally
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def get_feature_importance(self, model, model_name: str) -> pd.DataFrame:
        """
        Extract feature importance from a trained model
        Returns: DataFrame with feature names and importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store for later retrieval
        self.feature_importances[model_name] = importance_df

        return importance_df
    
    def train_decision_tree(self, import_record_id: int, use_temporal_split: bool = False, **params) -> Tuple[int, Dict[str, float]]:
        """
        Train a decision tree model on the specified dataset
        Returns: (model_id, performance_metrics)
        """
        # Get data
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")

        # Split data
        if use_temporal_split and 'game_date' in df.columns:
            X_train, X_test, y_train, y_test = self.temporal_train_test_split(df, test_size=0.2)
        else:
            # Prepare features
            X, y = self.prepare_features(df)

            if y is None:
                raise ValueError("No target variable could be created from the data")

            # Random split
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

        # Get feature importance
        importance_df = self.get_feature_importance(dt_model, f"DecisionTree_{import_record_id}")

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
    
    def train_random_forest(self, import_record_id: int, use_temporal_split: bool = False, **params) -> Tuple[int, Dict[str, float]]:
        """
        Train a random forest model on the specified dataset
        Returns: (model_id, performance_metrics)
        """
        # Get data
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")

        # Split data
        if use_temporal_split and 'game_date' in df.columns:
            X_train, X_test, y_train, y_test = self.temporal_train_test_split(df, test_size=0.2)
        else:
            # Prepare features
            X, y = self.prepare_features(df)

            if y is None:
                raise ValueError("No target variable could be created from the data")

            # Random split
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

        # Get feature importance
        importance_df = self.get_feature_importance(rf_model, f"RandomForest_{import_record_id}")

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

    def train_xgboost(self, import_record_id: int, use_temporal_split: bool = False, **params) -> Tuple[int, Dict[str, float]]:
        """
        Train an XGBoost model on the specified dataset
        Returns: (model_id, performance_metrics)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")

        # Get data
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")

        # Split data
        if use_temporal_split and 'game_date' in df.columns:
            X_train, X_test, y_train, y_test = self.temporal_train_test_split(df, test_size=0.2)
        else:
            # Prepare features
            X, y = self.prepare_features(df)

            if y is None:
                raise ValueError("No target variable could be created from the data")

            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # Train model
        n_classes = len(np.unique(y_train))
        model_params = {
            'random_state': 42,
            'n_estimators': params.get('n_estimators', 100),
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
            'base_score': 0.5
        }

        xgb_model = xgb.XGBClassifier(**model_params)
        xgb_model.fit(X_train, y_train)

        # Make predictions
        y_pred = xgb_model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # Get feature importance
        importance_df = self.get_feature_importance(xgb_model, f"XGBoost_{import_record_id}")

        # Register model in database
        model_id = self.db.register_model(
            name=f"XGBoost - Import {import_record_id}",
            model_type="XGBoost",
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

    def train_baseline(self, import_record_id: int, use_temporal_split: bool = False, strategy: str = 'most_frequent') -> Tuple[int, Dict[str, float]]:
        """
        Train a baseline model (DummyClassifier) for comparison
        strategy can be: 'most_frequent', 'stratified', 'uniform', 'constant'
        Returns: (model_id, performance_metrics)
        """
        # Get data
        df = self.db.get_game_data(import_record_id)
        if len(df) == 0:
            raise ValueError("No data found for the specified import record")

        # Split data
        if use_temporal_split and 'game_date' in df.columns:
            X_train, X_test, y_train, y_test = self.temporal_train_test_split(df, test_size=0.2)
        else:
            # Prepare features
            X, y = self.prepare_features(df)

            if y is None:
                raise ValueError("No target variable could be created from the data")

            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # Train baseline model
        model_params = {
            'strategy': strategy,
            'random_state': 42
        }

        baseline_model = DummyClassifier(**model_params)
        baseline_model.fit(X_train, y_train)

        # Make predictions
        y_pred = baseline_model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # Register model in database
        model_id = self.db.register_model(
            name=f"Baseline ({strategy}) - Import {import_record_id}",
            model_type="Baseline",
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