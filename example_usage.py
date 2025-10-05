#!/usr/bin/env python3
"""
Example usage script for the NBA Game Prediction Application

This script demonstrates how to use the NBA predictor programmatically
for various tasks like data import, model training, and analysis.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

# Import our application modules
from nba_predictor import NBAPredictor
from real_data_loader import RealDataLoader

def create_sample_data():
    """
    Create sample NBA data for demonstration purposes
    In real usage, you would use actual NBA data from sources like:
    - basketball-reference.com
    - NBA API
    - ESPN API
    - Your own data collection
    """
    
    # Sample teams
    teams = [
        'Lakers', 'Warriors', 'Celtics', 'Heat', 'Nets', 'Clippers',
        'Bucks', 'Suns', 'Nuggets', 'Mavericks', 'Sixers', 'Knicks'
    ]
    
    # Generate sample historical data
    np.random.seed(42)  # For reproducible results
    sample_data = []
    
    start_date = datetime(2023, 10, 1)
    
    for i in range(200):  # Generate 200 sample games
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        game_date = start_date + timedelta(days=i // 3)  # ~3 games per day
        
        # Generate realistic-ish scores
        home_score = np.random.randint(95, 130)
        away_score = np.random.randint(95, 130)
        
        # Add some team strength bias
        team_strengths = {
            'Warriors': 5, 'Lakers': 4, 'Celtics': 4, 'Heat': 3,
            'Nets': 3, 'Clippers': 3, 'Bucks': 4, 'Suns': 3,
            'Nuggets': 4, 'Mavericks': 3, 'Sixers': 3, 'Knicks': 2
        }
        
        home_strength = team_strengths.get(home_team, 2)
        away_strength = team_strengths.get(away_team, 2)
        
        # Home court advantage
        home_score += 2 + home_strength
        away_score += away_strength
        
        # Generate some basic stats
        home_fg_pct = np.random.normal(0.45, 0.05)
        away_fg_pct = np.random.normal(0.45, 0.05)
        home_reb = np.random.randint(40, 55)
        away_reb = np.random.randint(40, 55)
        home_ast = np.random.randint(20, 35)
        away_ast = np.random.randint(20, 35)
        
        sample_data.append({
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date.strftime('%Y-%m-%d'),
            'home_score': home_score,
            'away_score': away_score,
            'season': '2023-24',
            'home_fg_pct': round(home_fg_pct, 3),
            'away_fg_pct': round(away_fg_pct, 3),
            'home_reb': home_reb,
            'away_reb': away_reb,
            'home_ast': home_ast,
            'away_ast': away_ast
        })
    
    return pd.DataFrame(sample_data)

def example_basic_workflow():
    """
    Example 1: Basic workflow - import data, train models, compare results
    """
    print("Example 1: Basic NBA Prediction Workflow")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = NBAPredictor("example_nba_predictions.db")
    
    # Create and save sample data
    print("Creating sample NBA data...")
    sample_df = create_sample_data()
    sample_df.to_csv("sample_nba_data.csv", index=False)
    print(f"   Created {len(sample_df)} sample games")
    
    # Import the data
    print("\nImporting data...")
    result = predictor.import_data("sample_nba_data.csv", "Sample NBA 2023-24 data")
    
    if result["success"]:
        import_id = result["import_id"]
        
        # Train models
        print(f"\nTraining models on import ID {import_id}...")
        training_results = predictor.train_models(import_id)
        
        # Compare model performance
        print("\nComparing model performance...")
        predictor.compare_models(import_id)
        
        # Generate detailed report
        print(f"\nGenerating performance report...")
        predictor.generate_report(import_id, "basic_workflow_report.txt")
        
    else:
        print(f"Failed to import data: {result['error']}")

def example_advanced_analysis():
    """
    Example 2: Advanced analysis with real data verification
    """
    print("\n\nExample 2: Advanced Analysis with Verification")
    print("=" * 50)
    
    # Initialize components
    predictor = NBAPredictor("example_nba_predictions.db")
    loader = RealDataLoader(predictor.db)
    
    # Create training data (earlier games)
    print("Creating training data...")
    training_data = create_sample_data()
    training_data = training_data.iloc[:150]  # Use first 150 games for training
    training_data.to_csv("training_data.csv", index=False)
    
    # Create "real results" data (later games)
    print("Creating real results data...")
    real_data = create_sample_data()
    real_data = real_data.iloc[150:]  # Use remaining games as "real results"
    real_data.to_csv("real_results.csv", index=False)
    
    # Import training data and train models
    print("\nImporting training data...")
    training_result = predictor.import_data("training_data.csv", "Training Data")
    
    if training_result["success"]:
        training_import_id = training_result["import_id"]
        
        print(f"Training models...")
        predictor.train_models(training_import_id)
        
        # Import real results
        print("\nImporting real results data...")
        real_import_id = loader.load_real_results("real_results.csv", "Real Game Results")
        
        # Verify predictions against real results
        print(f"\nVerifying predictions...")
        verification_results = loader.compare_predictions_with_reality(
            training_import_id, real_import_id
        )
        
        # Generate verification report
        print("Generating verification report...")
        loader.generate_verification_report(verification_results, "verification_report.txt")
        
        # Print summary
        summary = loader.create_verification_summary_table(verification_results)
        print("\nVerification Summary:")
        print(summary)

def example_batch_processing():
    """
    Example 3: Batch processing multiple datasets
    """
    print("\n\nExample 3: Batch Processing Multiple Datasets")
    print("=" * 50)
    
    predictor = NBAPredictor("example_nba_predictions.db")
    
    # Simulate multiple datasets (different seasons)
    seasons = ["2021-22", "2022-23", "2023-24"]
    import_ids = []
    
    for season in seasons:
        print(f"\nProcessing {season} season...")
        
        # Create season-specific data
        season_data = create_sample_data()
        season_data['season'] = season
        
        # Add some season-specific variations
        if season == "2021-22":
            # Simulate lower-scoring season
            season_data['home_score'] *= 0.95
            season_data['away_score'] *= 0.95
        
        filename = f"nba_data_{season.replace('-', '_')}.csv"
        season_data.to_csv(filename, index=False)
        
        # Import and train
        result = predictor.import_data(filename, f"NBA {season} Season")
        if result["success"]:
            import_id = result["import_id"]
            import_ids.append(import_id)
            
            # Train models for this season
            predictor.train_models(import_id)
    
    # Compare across all seasons
    print(f"\nComparing models across all seasons...")
    predictor.compare_models()  # Compare all models
    
    # Generate reports for each season
    for i, import_id in enumerate(import_ids):
        season = seasons[i]
        report_file = f"season_report_{season.replace('-', '_')}.txt"
        predictor.generate_report(import_id, report_file)
        print(f"   Report for {season}: {report_file}")

def example_custom_analysis():
    """
    Example 4: Custom analysis using the database directly
    """
    print("\n\nExample 4: Custom Analysis")
    print("=" * 50)
    
    from database import NBADatabase
    from performance_evaluator import PerformanceEvaluator
    
    # Initialize database connection
    db = NBADatabase("example_nba_predictions.db")
    evaluator = PerformanceEvaluator(db)
    
    # Get all model results
    print("Custom performance analysis...")
    all_results = evaluator.compare_all_models()
    
    if len(all_results) > 0:
        print(f"\nPerformance Statistics:")
        print(f"   Total models trained: {len(all_results)}")
        print(f"   Best accuracy: {all_results['Accuracy'].max():.4f}")
        print(f"   Average accuracy: {all_results['Accuracy'].mean():.4f}")
        print(f"   Best F1-score: {all_results['F1_Score'].max():.4f}")
        
        # Find best performing model type
        best_by_type = all_results.groupby('Model_Type')[['Accuracy', 'F1_Score']].mean()
        print(f"\nAverage performance by model type:")
        print(best_by_type.round(4))
        
        # Get best models by different metrics
        print(f"\nBest models by metric:")
        metrics = ['accuracy', 'f1_score']
        for metric in metrics:
            best = evaluator.get_best_models_by_metric(metric)
            if len(best) > 0:
                print(f"   Best {metric}: {best.iloc[0]['Model_Name']} ({best.iloc[0][metric.capitalize() if metric != 'f1_score' else 'F1_Score']:.4f})")
    
    else:
        print("No model results found. Run previous examples first!")

def cleanup_example_files():
    """Clean up example files"""
    files_to_remove = [
        "sample_nba_data.csv", "training_data.csv", "real_results.csv",
        "nba_data_2021_22.csv", "nba_data_2022_23.csv", "nba_data_2023_24.csv",
        "example_nba_predictions.db"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

def main():
    """
    Run all examples
    """
    print("NBA Prediction Application - Example Usage")
    print("=" * 60)
    print("This script demonstrates various ways to use the NBA predictor")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_workflow()
        example_advanced_analysis()
        example_batch_processing()
        example_custom_analysis()
        
        print("\n\nAll examples completed successfully!")
        print("\nGenerated files:")
        print("- basic_workflow_report.txt")
        print("- verification_report.txt")
        print("- season_report_*.txt")
        print("- example_nba_predictions.db")
        
        print(f"\nTo explore the data interactively, run:")
        print(f"python -c \"from nba_predictor import NBAPredictor; p = NBAPredictor('example_nba_predictions.db'); p.launch_dashboard()\"")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Make sure all required dependencies are installed:")
        print("pip install -r requirements.txt")
    
    # Clean up option
    response = input("\nWould you like to clean up example files? (y/n): ")
    if response.lower() == 'y':
        cleanup_example_files()

if __name__ == "__main__":
    main()