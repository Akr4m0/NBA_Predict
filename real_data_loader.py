import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json
from database import NBADatabase
from data_importer import DataImporter

class RealDataLoader:
    """
    Load and compare real game results with predictions for model verification
    """
    
    def __init__(self, db: NBADatabase):
        self.db = db
        self.importer = DataImporter(db)
    
    def load_real_results(self, file_path: str, description: str = "Real game results") -> int:
        """
        Load actual game results from CSV/Excel file for verification
        Returns: import_record_id
        """
        try:
            import_id, df = self.importer.import_historical_data(file_path, description)
            print(f"Loaded {len(df)} real game results")
            return import_id
        except Exception as e:
            print(f"Error loading real results: {str(e)}")
            raise
    
    def compare_predictions_with_reality(self, prediction_import_id: int, 
                                       real_results_import_id: int) -> Dict[str, Any]:
        """
        Compare model predictions with actual game results
        """
        # Get prediction data and results
        prediction_results = self.db.get_prediction_results(import_record_id=prediction_import_id)
        
        if len(prediction_results) == 0:
            raise ValueError("No prediction results found for the specified import")
        
        # Get real game data
        real_games = self.db.get_game_data(import_record_id=real_results_import_id)
        prediction_games = self.db.get_game_data(import_record_id=prediction_import_id)
        
        verification_results = {}
        
        for _, pred_result in prediction_results.iterrows():
            model_id = pred_result['model_id']
            model_info = self.db.get_models()
            model_name = model_info[model_info['id'] == model_id].iloc[0]['name']
            
            # Parse predictions
            predictions = json.loads(pred_result['predictions']) if pred_result['predictions'] else []
            
            if not predictions:
                continue
            
            # Match games between prediction and real data
            matched_games, verification_metrics = self._match_and_verify_games(
                prediction_games, real_games, predictions
            )
            
            verification_results[model_name] = {
                'model_id': model_id,
                'matched_games': len(matched_games),
                'verification_metrics': verification_metrics,
                'detailed_results': matched_games
            }
        
        return verification_results
    
    def _match_and_verify_games(self, prediction_games: pd.DataFrame, 
                               real_games: pd.DataFrame, 
                               predictions: List) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Match prediction games with real games and calculate verification metrics
        """
        matched_games = []
        correct_predictions = 0
        total_matched = 0
        
        # Convert predictions to results (assuming predictions are encoded)
        prediction_games = prediction_games.copy()
        if len(predictions) == len(prediction_games):
            prediction_games['predicted_result_encoded'] = predictions
        
        # Try to match games based on teams and date
        for pred_idx, pred_game in prediction_games.iterrows():
            # Find matching real game
            matching_real_games = real_games[
                (real_games['home_team'] == pred_game['home_team']) & 
                (real_games['away_team'] == pred_game['away_team'])
            ]
            
            # If multiple matches, try to match by date
            if len(matching_real_games) > 1 and 'game_date' in pred_game and pd.notna(pred_game['game_date']):
                matching_real_games = matching_real_games[
                    pd.to_datetime(matching_real_games['game_date']).dt.date == 
                    pd.to_datetime(pred_game['game_date']).date()
                ]
            
            if len(matching_real_games) > 0:
                real_game = matching_real_games.iloc[0]
                
                # Determine actual result
                actual_result = self._determine_game_result(real_game)
                predicted_result = self._determine_game_result(pred_game)
                
                match_info = {
                    'home_team': pred_game['home_team'],
                    'away_team': pred_game['away_team'],
                    'predicted_result': predicted_result,
                    'actual_result': actual_result,
                    'correct': predicted_result == actual_result,
                    'game_date': pred_game.get('game_date'),
                    'prediction_confidence': getattr(pred_game, 'confidence', None)
                }
                
                matched_games.append(match_info)
                total_matched += 1
                
                if predicted_result == actual_result:
                    correct_predictions += 1
        
        # Calculate verification metrics
        verification_metrics = {
            'accuracy': correct_predictions / total_matched if total_matched > 0 else 0,
            'total_matched_games': total_matched,
            'correct_predictions': correct_predictions,
            'match_rate': total_matched / len(prediction_games) if len(prediction_games) > 0 else 0
        }
        
        return matched_games, verification_metrics
    
    def _determine_game_result(self, game: pd.Series) -> str:
        """Determine the result of a game (home_win, away_win, or tie)"""
        if 'result' in game and pd.notna(game['result']):
            return game['result']
        
        if 'home_score' in game and 'away_score' in game:
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                home_score = float(game['home_score'])
                away_score = float(game['away_score'])
                
                if home_score > away_score:
                    return 'home_win'
                elif away_score > home_score:
                    return 'away_win'
                else:
                    return 'tie'
        
        return 'unknown'
    
    def generate_verification_report(self, verification_results: Dict[str, Any], 
                                   output_file: str = None) -> None:
        """Generate a detailed verification report"""
        
        if output_file is None:
            output_file = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("NBA Prediction Verification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("VERIFICATION SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            best_model = None
            best_accuracy = 0
            
            for model_name, results in verification_results.items():
                metrics = results['verification_metrics']
                accuracy = metrics['accuracy']
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
                
                f.write(f"Model: {model_name}\n")
                f.write(f"  Verification Accuracy: {accuracy:.4f}\n")
                f.write(f"  Matched Games: {metrics['total_matched_games']}\n")
                f.write(f"  Correct Predictions: {metrics['correct_predictions']}\n")
                f.write(f"  Match Rate: {metrics['match_rate']:.4f}\n\n")
            
            f.write(f"Best Performing Model: {best_model} (Accuracy: {best_accuracy:.4f})\n\n")
            
            # Detailed results
            f.write("DETAILED VERIFICATION RESULTS\n")
            f.write("-" * 40 + "\n\n")
            
            for model_name, results in verification_results.items():
                f.write(f"Model: {model_name}\n")
                f.write("=" * (len(model_name) + 7) + "\n")
                
                detailed_results = results['detailed_results']
                correct_count = sum(1 for game in detailed_results if game['correct'])
                
                f.write(f"Correct Predictions: {correct_count}/{len(detailed_results)}\n\n")
                
                # Show some sample predictions
                f.write("Sample Predictions:\n")
                for i, game in enumerate(detailed_results[:10]):  # Show first 10
                    status = "CORRECT" if game['correct'] else "INCORRECT"
                    f.write(f"  {status} {game['home_team']} vs {game['away_team']}: "
                           f"Predicted {game['predicted_result']}, Actual {game['actual_result']}\n")
                
                if len(detailed_results) > 10:
                    f.write(f"  ... and {len(detailed_results) - 10} more games\n")
                
                f.write("\n")
        
        print(f"Verification report saved to {output_file}")
    
    def create_verification_summary_table(self, verification_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a summary table of verification results"""
        
        summary_data = []
        
        for model_name, results in verification_results.items():
            metrics = results['verification_metrics']
            summary_data.append({
                'Model': model_name,
                'Verification_Accuracy': round(metrics['accuracy'], 4),
                'Matched_Games': metrics['total_matched_games'],
                'Correct_Predictions': metrics['correct_predictions'],
                'Match_Rate': round(metrics['match_rate'], 4)
            })
        
        return pd.DataFrame(summary_data).sort_values('Verification_Accuracy', ascending=False)
    
    def find_prediction_gaps(self, prediction_import_id: int, 
                           real_results_import_id: int) -> Dict[str, List]:
        """
        Find games that were predicted but don't have real results, 
        and real games that weren't predicted
        """
        prediction_games = self.db.get_game_data(import_record_id=prediction_import_id)
        real_games = self.db.get_game_data(import_record_id=real_results_import_id)
        
        # Convert to sets for easy comparison
        pred_games_set = set()
        for _, game in prediction_games.iterrows():
            game_key = (game['home_team'], game['away_team'], 
                       pd.to_datetime(game['game_date']).date() if 'game_date' in game else None)
            pred_games_set.add(game_key)
        
        real_games_set = set()
        for _, game in real_games.iterrows():
            game_key = (game['home_team'], game['away_team'], 
                       pd.to_datetime(game['game_date']).date() if 'game_date' in game else None)
            real_games_set.add(game_key)
        
        # Find gaps
        gaps = {
            'predicted_but_no_real_data': list(pred_games_set - real_games_set),
            'real_games_not_predicted': list(real_games_set - pred_games_set),
            'coverage_stats': {
                'total_predictions': len(pred_games_set),
                'total_real_games': len(real_games_set),
                'overlap': len(pred_games_set & real_games_set),
                'coverage_rate': len(pred_games_set & real_games_set) / len(real_games_set) if len(real_games_set) > 0 else 0
            }
        }
        
        return gaps