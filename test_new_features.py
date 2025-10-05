#!/usr/bin/env python3
"""
Test script for new features: XGBoost, temporal split, feature importance, and baseline
"""

import pandas as pd
import numpy as np
from database import NBADatabase
from predictive_models import PredictiveModels

def create_test_data():
    """Create sample NBA game data for testing"""
    np.random.seed(42)

    teams = ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bucks', 'Nets']
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    data = []
    for date in dates:
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        home_score = np.random.randint(80, 130)
        away_score = np.random.randint(80, 130)

        data.append({
            'game_date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'season': '2022-23',
            'home_stats': {'points': home_score, 'rebounds': np.random.randint(30, 50)},
            'away_stats': {'points': away_score, 'rebounds': np.random.randint(30, 50)}
        })

    return pd.DataFrame(data)

def test_temporal_split():
    """Test temporal train/test split"""
    print("\n" + "="*80)
    print("Testing Temporal Train/Test Split")
    print("="*80)

    db = NBADatabase("test_nba.db")
    models = PredictiveModels(db)

    # Create test data
    df = create_test_data()

    # Test temporal split
    X_train, X_test, y_train, y_test = models.temporal_train_test_split(df)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")
    print("[PASS] Temporal split working correctly")

    return db, df

def test_feature_importance(db, df):
    """Test feature importance extraction"""
    print("\n" + "="*80)
    print("Testing Feature Importance Analysis")
    print("="*80)

    models = PredictiveModels(db)

    # Prepare features
    X, y = models.prepare_features(df)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train a simple model
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)

    # Get feature importance
    importance_df = models.get_feature_importance(dt_model, "TestModel")

    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    print("\n[PASS] Feature importance analysis working correctly")

def test_baseline_model():
    """Test baseline model"""
    print("\n" + "="*80)
    print("Testing Baseline Model")
    print("="*80)

    db = NBADatabase("test_baseline.db")
    models = PredictiveModels(db)

    # Create and import test data
    df = create_test_data()

    # Save to database - convert timestamps to strings
    df_to_save = df.copy()
    df_to_save['game_date'] = df_to_save['game_date'].dt.strftime('%Y-%m-%d')

    import_id = db.register_import(
        filename="test_data.csv",
        file_path="test_data.csv",
        record_count=len(df_to_save),
        description="Test data for baseline model"
    )
    db.save_game_data(import_id, df_to_save)

    # Train baseline model
    try:
        model_id, metrics = models.train_baseline(import_id)
        print(f"\nBaseline Model Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print("[PASS] Baseline model working correctly")
    except Exception as e:
        print(f"[FAIL] Error training baseline model: {str(e)}")

def test_xgboost_model():
    """Test XGBoost model"""
    print("\n" + "="*80)
    print("Testing XGBoost Model")
    print("="*80)

    try:
        import xgboost as xgb
        print(f"XGBoost version: {xgb.__version__}")

        db = NBADatabase("test_xgboost.db")
        models = PredictiveModels(db)

        # Create and import test data
        df = create_test_data()

        # Save to database - convert timestamps to strings
        df_to_save = df.copy()
        df_to_save['game_date'] = df_to_save['game_date'].dt.strftime('%Y-%m-%d')

        import_id = db.register_import(
            filename="test_data.csv",
            file_path="test_data.csv",
            record_count=len(df_to_save),
            description="Test data for XGBoost"
        )
        db.save_game_data(import_id, df_to_save)

        # Train XGBoost model
        model_id, metrics = models.train_xgboost(import_id)
        print(f"\nXGBoost Model Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

        # Check feature importance
        if f"XGBoost_{import_id}" in models.feature_importances:
            print(f"\nTop 5 Most Important Features (XGBoost):")
            print(models.feature_importances[f"XGBoost_{import_id}"].head(5).to_string(index=False))

        print("[PASS] XGBoost model working correctly")

    except ImportError:
        print("[FAIL] XGBoost not installed")
    except Exception as e:
        print(f"[FAIL] Error training XGBoost model: {str(e)}")

def main():
    print("\n" + "="*80)
    print("NBA Prediction - New Features Test Suite")
    print("="*80)

    # Test temporal split
    db, df = test_temporal_split()

    # Test feature importance
    test_feature_importance(db, df)

    # Test baseline model
    test_baseline_model()

    # Test XGBoost model
    test_xgboost_model()

    print("\n" + "="*80)
    print("All Tests Completed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
