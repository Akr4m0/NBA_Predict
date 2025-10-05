#!/usr/bin/env python3
"""
NBA Game Prediction Application

This application provides comprehensive functionality for NBA game prediction including:
- Historical data import from CSV/Excel files
- Predictive model training (Decision Tree, Random Forest)  
- Model performance evaluation and comparison
- Interactive dashboard for results analysis
"""

import argparse
import sys
import os
from typing import Dict, Any

from database import NBADatabase
from data_importer import DataImporter
from predictive_models import PredictiveModels
from performance_evaluator import PerformanceEvaluator
from dashboard import NBADashboard

class NBAPredictor:
    def __init__(self, db_path: str = "nba_predictions.db"):
        self.db = NBADatabase(db_path)
        self.importer = DataImporter(self.db)
        self.models = PredictiveModels(self.db)
        self.evaluator = PerformanceEvaluator(self.db)
        
    def import_data(self, file_path: str, description: str = None) -> Dict[str, Any]:
        """Import historical NBA data from file"""
        try:
            import_id, df = self.importer.import_historical_data(file_path, description)
            summary = self.importer.get_data_summary(import_id)
            
            result = {
                "success": True,
                "import_id": import_id,
                "records_imported": len(df),
                "summary": summary
            }
            
            print(f"Successfully imported {len(df)} records from {os.path.basename(file_path)}")
            print(f"   Import ID: {import_id}")
            print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            print(f"   Unique teams: {len(summary['unique_teams'])}")
            
            return result
            
        except Exception as e:
            print(f"Error importing data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def train_models(self, import_id: int, models: list = None, use_temporal_split: bool = False) -> Dict[str, Any]:
        """Train predictive models on imported data"""
        if models is None:
            models = ['decision_tree', 'random_forest', 'xgboost', 'baseline']

        results = {}

        for model_type in models:
            try:
                print(f"Training {model_type}...")

                if model_type == 'decision_tree':
                    model_id, metrics = self.models.train_decision_tree(import_id, use_temporal_split=use_temporal_split)
                elif model_type == 'random_forest':
                    model_id, metrics = self.models.train_random_forest(import_id, use_temporal_split=use_temporal_split)
                elif model_type == 'xgboost':
                    model_id, metrics = self.models.train_xgboost(import_id, use_temporal_split=use_temporal_split)
                elif model_type == 'baseline':
                    model_id, metrics = self.models.train_baseline(import_id, use_temporal_split=use_temporal_split)
                else:
                    print(f"Unknown model type: {model_type}")
                    continue
                
                results[model_type] = {
                    "model_id": model_id,
                    "metrics": metrics
                }
                
                print(f"{model_type} trained successfully:")
                print(f"   Model ID: {model_id}")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   F1-Score: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                print(f"Error training {model_type}: {str(e)}")
                results[model_type] = {"error": str(e)}
        
        return results
    
    def compare_models(self, import_id: int = None) -> None:
        """Compare model performance"""
        try:
            comparison = self.evaluator.compare_all_models(import_id)
            
            if len(comparison) == 0:
                print("No model results found for comparison")
                return
            
            print("\nModel Performance Comparison")
            print("=" * 80)
            print(f"{'Model':<30} {'Type':<15} {'Accuracy':<10} {'F1-Score':<10}")
            print("-" * 80)
            
            for _, row in comparison.iterrows():
                print(f"{row['Model_Name']:<30} {row['Model_Type']:<15} {row['Accuracy']:<10.4f} {row['F1_Score']:<10.4f}")
            
            # Best models
            best_accuracy = comparison.loc[comparison['Accuracy'].idxmax()]
            best_f1 = comparison.loc[comparison['F1_Score'].idxmax()]
            
            print("\nBest Performing Models:")
            print(f"   Best Accuracy: {best_accuracy['Model_Name']} ({best_accuracy['Accuracy']:.4f})")
            print(f"   Best F1-Score: {best_f1['Model_Name']} ({best_f1['F1_Score']:.4f})")
            
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
    
    def list_imports(self) -> None:
        """List all data imports"""
        try:
            imports = self.db.get_import_records()
            
            if len(imports) == 0:
                print("No data imports found")
                return
            
            print("\nImported Datasets")
            print("=" * 80)
            print(f"{'ID':<5} {'Filename':<30} {'Records':<10} {'Import Date':<20}")
            print("-" * 80)
            
            for _, row in imports.iterrows():
                print(f"{row['id']:<5} {row['filename']:<30} {row['record_count']:<10} {row['import_date']:<20}")
                
        except Exception as e:
            print(f"Error listing imports: {str(e)}")
    
    def generate_report(self, import_id: int, output_file: str = None) -> None:
        """Generate detailed performance report"""
        try:
            if output_file is None:
                output_file = f"performance_report_import_{import_id}.txt"
            
            self.evaluator.save_performance_report(import_id, output_file)
            print(f"Performance report saved to {output_file}")
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
    
    def launch_dashboard(self, port: int = 8050) -> None:
        """Launch interactive dashboard"""
        try:
            print(f"Launching dashboard on http://localhost:{port}")
            dashboard = NBADashboard(self.db.db_path)
            dashboard.run(port=port)
            
        except Exception as e:
            print(f"Error launching dashboard: {str(e)}")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description="NBA Game Prediction Application")
    parser.add_argument("--db", default="nba_predictions.db", help="Database file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import historical data")
    import_parser.add_argument("file", help="Path to CSV or Excel file")
    import_parser.add_argument("--description", help="Description of the dataset")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train predictive models")
    train_parser.add_argument("import_id", type=int, help="Import ID to train on")
    train_parser.add_argument("--models", nargs="+",
                             choices=["decision_tree", "random_forest", "xgboost", "baseline"],
                             default=["decision_tree", "random_forest", "xgboost", "baseline"],
                             help="Models to train")
    train_parser.add_argument("--temporal", action="store_true",
                             help="Use temporal train/test split instead of random split")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare model performance")
    compare_parser.add_argument("--import_id", type=int, help="Specific import ID to compare")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List imported datasets")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument("import_id", type=int, help="Import ID for report")
    report_parser.add_argument("--output", help="Output file path")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch interactive dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    
    # Auto command (import + train + compare)
    auto_parser = subparsers.add_parser("auto", help="Import data and train all models")
    auto_parser.add_argument("file", help="Path to CSV or Excel file")
    auto_parser.add_argument("--description", help="Description of the dataset")
    auto_parser.add_argument("--temporal", action="store_true",
                            help="Use temporal train/test split instead of random split")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize predictor
    predictor = NBAPredictor(args.db)
    
    # Execute commands
    if args.command == "import":
        predictor.import_data(args.file, args.description)
    
    elif args.command == "train":
        predictor.train_models(args.import_id, args.models, use_temporal_split=args.temporal)
    
    elif args.command == "compare":
        predictor.compare_models(args.import_id)
    
    elif args.command == "list":
        predictor.list_imports()
    
    elif args.command == "report":
        predictor.generate_report(args.import_id, args.output)
    
    elif args.command == "dashboard":
        predictor.launch_dashboard(args.port)
    
    elif args.command == "auto":
        # Import data
        result = predictor.import_data(args.file, args.description)
        if result["success"]:
            import_id = result["import_id"]
            # Train models
            predictor.train_models(import_id, use_temporal_split=args.temporal)
            # Compare results
            predictor.compare_models(import_id)

if __name__ == "__main__":
    main()