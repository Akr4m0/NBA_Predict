# NBA Prediction Backend

Python backend for NBA game prediction using machine learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run from project root
python -m backend.nba_predictor --help

# Or run from backend directory
cd backend
python nba_predictor.py --help
```

## Available Commands

```bash
# Import data
python nba_predictor.py import ../data/games.csv --description "NBA 2023 Season"

# Train models
python nba_predictor.py train 1 --models decision_tree random_forest xgboost

# Compare models
python nba_predictor.py compare

# Launch dashboard
python nba_predictor.py dashboard

# Auto (import + train + compare)
python nba_predictor.py auto ../data/games.csv
```

## Files

- `nba_predictor.py` - Main CLI application
- `database.py` - SQLite database operations
- `data_importer.py` - CSV/Excel data import
- `predictive_models.py` - ML models (Decision Tree, Random Forest, XGBoost)
- `performance_evaluator.py` - Model evaluation and comparison
- `dashboard.py` - Dash web dashboard
- `real_data_loader.py` - Prediction verification
- `requirements.txt` - Python dependencies

## Database Location

By default, the database is created in `../data/nba_predictions.db`

You can specify a custom location:
```bash
python nba_predictor.py --db /path/to/database.db
```
