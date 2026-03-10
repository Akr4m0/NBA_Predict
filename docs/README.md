# NBA Game Prediction Application

A comprehensive NBA game prediction system that allows you to import historical data, train machine learning models, compare their performance, and verify predictions against real results.

## Features

### Core Functionality
- **Historical Data Import**: Import NBA game data from CSV/Excel files
- **Multiple ML Models**: Decision Tree and Random Forest classifiers
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Interactive Dashboard**: Web-based visualization and analysis tool
- **Real Data Verification**: Compare predictions with actual game results

### Data Management
- SQLite database for storing imports, models, and results
- Automatic data cleaning and feature engineering
- Flexible data schema supporting various NBA data formats
- Import tracking and metadata management

### Model Analysis
- Cross-model performance comparison
- Detailed statistical reports
- Confusion matrix analysis
- Feature importance insights

## Installation

1. **Clone/Download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Import Your NBA Data
```bash
python nba_predictor.py import "your_nba_data.csv" --description "2023 NBA Season"
```

### 2. Train Models and Compare Results (All-in-One)
```bash
python nba_predictor.py auto "your_nba_data.csv" --description "2023 NBA Season"
```

### 3. Launch Interactive Dashboard
```bash
python nba_predictor.py dashboard
```
Then visit http://localhost:8050 in your browser.

## Detailed Usage

### Data Import

The application accepts CSV or Excel files with NBA game data. Required columns:
- `home_team` (or variations like "Home Team", "Home")
- `away_team` (or variations like "Away Team", "Visitor") 
- `game_date` (or variations like "Date", "Game Date")

Optional but recommended columns:
- `home_score`, `away_score` 
- `season`
- Team statistics (FG%, rebounds, assists, etc.)

```bash
# Import data
python nba_predictor.py import "nba_games_2023.csv" --description "2023 Regular Season"
```

### Model Training

Train specific models on imported data:

```bash
# Train all models (default)
python nba_predictor.py train 1

# Train specific models
python nba_predictor.py train 1 --models decision_tree
python nba_predictor.py train 1 --models random_forest
```

### Performance Analysis

```bash
# Compare all models
python nba_predictor.py compare

# Compare models for specific dataset
python nba_predictor.py compare --import_id 1

# Generate detailed report
python nba_predictor.py report 1 --output "detailed_analysis.txt"
```

### Real Data Verification

To verify your predictions against actual results:

1. **Import real game results**:
   ```bash
   python nba_predictor.py import "real_results_2023.csv" --description "Actual 2023 Results"
   ```

2. **Use Python to compare** (interactive mode):
   ```python
   from nba_predictor import NBAPredictor
   from real_data_loader import RealDataLoader
   
   predictor = NBAPredictor()
   loader = RealDataLoader(predictor.db)
   
   # Compare predictions (import_id 1) with real results (import_id 2)
   verification = loader.compare_predictions_with_reality(1, 2)
   loader.generate_verification_report(verification)
   ```

### Dashboard Features

The interactive dashboard provides:

- **Overview**: Summary statistics and recent activity
- **Model Comparison**: Visual performance comparisons 
- **Dataset Analysis**: Detailed dataset information
- **Detailed Results**: In-depth analysis for specific datasets

Launch with:
```bash
python nba_predictor.py dashboard --port 8080
```

## Data Format Examples

### Basic CSV Format
```csv
home_team,away_team,game_date,home_score,away_score
Lakers,Warriors,2023-01-15,112,108
Heat,Celtics,2023-01-15,95,102
```

### Advanced Format with Statistics
```csv
home_team,away_team,game_date,home_score,away_score,home_fg_pct,away_fg_pct,home_reb,away_reb
Lakers,Warriors,2023-01-15,112,108,0.456,0.432,45,42
Heat,Celtics,2023-01-15,95,102,0.398,0.478,38,51
```

## Command Reference

### Main Commands

| Command | Description | Example |
|---------|-------------|---------|
| `import` | Import historical data | `python nba_predictor.py import data.csv` |
| `train` | Train predictive models | `python nba_predictor.py train 1` |
| `compare` | Compare model performance | `python nba_predictor.py compare` |
| `list` | List imported datasets | `python nba_predictor.py list` |
| `report` | Generate detailed report | `python nba_predictor.py report 1` |
| `dashboard` | Launch web dashboard | `python nba_predictor.py dashboard` |
| `auto` | Import + train + compare | `python nba_predictor.py auto data.csv` |

### Options

- `--db`: Specify database file (default: `nba_predictions.db`)
- `--description`: Add description to data import
- `--models`: Choose specific models to train
- `--import_id`: Filter by specific dataset
- `--output`: Specify output file for reports
- `--port`: Set dashboard port

## File Structure

```
NBA_Prediction_Decision_tree/
├── nba_predictor.py          # Main application and CLI
├── database.py               # Database operations and schema
├── data_importer.py          # Data import and cleaning
├── predictive_models.py      # ML model implementations
├── performance_evaluator.py  # Model evaluation and comparison
├── dashboard.py              # Interactive web dashboard
├── real_data_loader.py       # Real data verification
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Model Details

### Decision Tree Classifier
- Interpretable model showing decision paths
- Good for understanding feature importance
- Default parameters optimized for NBA data

### Random Forest Classifier  
- Ensemble method combining multiple trees
- Generally higher accuracy than single decision tree
- Robust to overfitting

### Feature Engineering
The application automatically creates features from your data:
- Team encoding (converts team names to numbers)
- Date-based features (month, day of week, day of year)
- Historical performance metrics
- Statistical features from team stats
- Home/away advantage indicators

## Performance Metrics

The application tracks multiple performance metrics:

- **Accuracy**: Overall correct prediction rate
- **Precision**: Positive predictive value
- **Recall**: True positive rate  
- **F1-Score**: Harmonic mean of precision and recall

## Troubleshooting

### Common Issues

1. **"Missing required columns" error**
   - Ensure your data has `home_team`, `away_team`, and `game_date` columns
   - Check column name variations (the app tries to auto-map common variations)

2. **"No data found" error**
   - Verify the import was successful with `python nba_predictor.py list`
   - Check that the import_id exists

3. **Dashboard not loading**
   - Ensure all dependencies are installed
   - Try a different port: `python nba_predictor.py dashboard --port 8080`
   - Check for firewall restrictions

### Data Quality Tips

- Ensure dates are in a recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- Remove or handle missing values in critical columns
- Use consistent team name formatting
- Include as many statistical features as possible for better predictions

## Contributing

This application is designed to be extensible. You can:
- Add new machine learning models in `predictive_models.py`
- Enhance feature engineering in the `prepare_features()` method
- Add new visualization components to the dashboard
- Extend the database schema for additional data types

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Support

For issues or questions:
1. Check this README for common solutions
2. Review the command help: `python nba_predictor.py --help`
3. Examine the generated log files and error messages

---

**Happy Predicting!**