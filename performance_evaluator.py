import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import json
from database import NBADatabase

class PerformanceEvaluator:
    def __init__(self, db: NBADatabase):
        self.db = db
        
    def compare_all_models(self, import_record_id: int = None) -> pd.DataFrame:
        """Compare performance of all models for a specific import or all imports"""
        results_df = self.db.get_prediction_results(import_record_id=import_record_id)
        models_df = self.db.get_models()
        imports_df = self.db.get_import_records()
        
        # Merge dataframes
        comparison = results_df.merge(models_df, left_on='model_id', right_on='id', suffixes=('', '_model'))
        comparison = comparison.merge(imports_df, left_on='import_record_id', right_on='id', suffixes=('', '_import'))
        
        # Select and rename columns for clarity
        comparison = comparison[[
            'filename', 'name', 'type', 'accuracy', 'precision_score', 
            'recall', 'f1_score', 'created_date'
        ]].rename(columns={
            'filename': 'Dataset',
            'name': 'Model_Name',
            'type': 'Model_Type',
            'accuracy': 'Accuracy',
            'precision_score': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1_Score',
            'created_date': 'Created_Date'
        })
        
        return comparison.round(4).sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
    
    def get_best_models_by_metric(self, metric: str = 'accuracy') -> pd.DataFrame:
        """Get the best performing models for each dataset by specified metric"""
        comparison = self.compare_all_models()
        
        if metric.lower() not in ['accuracy', 'precision', 'recall', 'f1_score']:
            raise ValueError("Metric must be one of: accuracy, precision, recall, f1_score")
        
        metric_col = metric.capitalize() if metric != 'f1_score' else 'F1_Score'
        
        # Get best model for each dataset
        best_models = comparison.loc[comparison.groupby('Dataset')[metric_col].idxmax()]
        
        return best_models.sort_values(metric_col, ascending=False)
    
    def create_performance_visualization(self, save_path: str = None) -> None:
        """Create comprehensive performance visualization"""
        comparison = self.compare_all_models()
        
        if len(comparison) == 0:
            print("No model results found for visualization")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'Precision vs Recall', 
                          'F1-Score by Model Type', 'Performance Metrics Heatmap'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "heatmap"}]]
        )
        
        # 1. Accuracy comparison bar chart
        accuracy_data = comparison.groupby(['Dataset', 'Model_Type'])['Accuracy'].max().reset_index()
        for dataset in accuracy_data['Dataset'].unique():
            dataset_data = accuracy_data[accuracy_data['Dataset'] == dataset]
            fig.add_trace(
                go.Bar(name=f'{dataset}', 
                      x=dataset_data['Model_Type'], 
                      y=dataset_data['Accuracy'],
                      showlegend=True),
                row=1, col=1
            )
        
        # 2. Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(x=comparison['Recall'], 
                      y=comparison['Precision'],
                      mode='markers',
                      text=comparison['Model_Name'],
                      marker=dict(
                          size=comparison['F1_Score'] * 20,
                          color=comparison['Accuracy'],
                          colorscale='Viridis',
                          showscale=True
                      ),
                      showlegend=False),
            row=1, col=2
        )
        
        # 3. F1-Score box plot by model type
        for model_type in comparison['Model_Type'].unique():
            type_data = comparison[comparison['Model_Type'] == model_type]
            fig.add_trace(
                go.Box(y=type_data['F1_Score'], 
                      name=model_type,
                      showlegend=False),
                row=2, col=1
            )
        
        # 4. Performance metrics heatmap
        metrics_pivot = comparison.groupby('Model_Type')[['Accuracy', 'Precision', 'Recall', 'F1_Score']].mean()
        fig.add_trace(
            go.Heatmap(z=metrics_pivot.values,
                      x=metrics_pivot.columns,
                      y=metrics_pivot.index,
                      colorscale='RdYlBu_r',
                      showscale=True),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="NBA Prediction Models Performance Analysis",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_detailed_report(self, import_record_id: int) -> Dict[str, Any]:
        """Generate a detailed performance report for a specific dataset"""
        results_df = self.db.get_prediction_results(import_record_id=import_record_id)
        
        if len(results_df) == 0:
            return {"error": "No results found for the specified import record"}
        
        models_df = self.db.get_models()
        import_info = self.db.get_import_records()
        import_info = import_info[import_info['id'] == import_record_id].iloc[0]
        
        report = {
            "dataset_info": {
                "filename": import_info['filename'],
                "import_date": import_info['import_date'],
                "record_count": import_info['record_count'],
                "description": import_info['description']
            },
            "model_performance": [],
            "summary": {}
        }
        
        # Detailed results for each model
        for _, result in results_df.iterrows():
            model_info = models_df[models_df['id'] == result['model_id']].iloc[0]
            
            model_performance = {
                "model_name": model_info['name'],
                "model_type": model_info['type'],
                "parameters": json.loads(model_info['parameters']),
                "metrics": {
                    "accuracy": round(result['accuracy'], 4),
                    "precision": round(result['precision_score'], 4),
                    "recall": round(result['recall'], 4),
                    "f1_score": round(result['f1_score'], 4)
                },
                "created_date": result['created_date']
            }
            
            # Add confusion matrix if predictions are available
            if result['predictions'] and result['actual_results']:
                predictions = json.loads(result['predictions'])
                actual = json.loads(result['actual_results'])
                model_performance["confusion_matrix"] = confusion_matrix(actual, predictions).tolist()
            
            report["model_performance"].append(model_performance)
        
        # Summary statistics
        metrics_df = pd.DataFrame([mp["metrics"] for mp in report["model_performance"]])
        report["summary"] = {
            "total_models": len(report["model_performance"]),
            "best_accuracy": {
                "value": float(metrics_df['accuracy'].max()),
                "model": report["model_performance"][metrics_df['accuracy'].idxmax()]["model_name"]
            },
            "best_f1": {
                "value": float(metrics_df['f1_score'].max()),
                "model": report["model_performance"][metrics_df['f1_score'].idxmax()]["model_name"]
            },
            "average_metrics": {
                "accuracy": round(float(metrics_df['accuracy'].mean()), 4),
                "precision": round(float(metrics_df['precision'].mean()), 4),
                "recall": round(float(metrics_df['recall'].mean()), 4),
                "f1_score": round(float(metrics_df['f1_score'].mean()), 4)
            }
        }
        
        return report
    
    def compare_with_baseline(self, import_record_id: int, baseline_accuracy: float = 0.5) -> pd.DataFrame:
        """Compare model performance against a baseline accuracy"""
        comparison = self.compare_all_models(import_record_id=import_record_id)
        
        comparison['Improvement_over_baseline'] = comparison['Accuracy'] - baseline_accuracy
        comparison['Improvement_percentage'] = (comparison['Improvement_over_baseline'] / baseline_accuracy) * 100
        
        return comparison.sort_values('Improvement_over_baseline', ascending=False)
    
    def save_performance_report(self, import_record_id: int, filename: str) -> None:
        """Save a detailed performance report to file"""
        report = self.generate_detailed_report(import_record_id)
        
        # Save as JSON
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
        
        # Save as readable text
        elif filename.endswith('.txt'):
            with open(filename, 'w') as f:
                f.write("NBA Prediction Models Performance Report\n")
                f.write("=" * 50 + "\n\n")
                
                # Dataset info
                f.write("Dataset Information:\n")
                f.write(f"Filename: {report['dataset_info']['filename']}\n")
                f.write(f"Import Date: {report['dataset_info']['import_date']}\n")
                f.write(f"Record Count: {report['dataset_info']['record_count']}\n")
                f.write(f"Description: {report['dataset_info']['description']}\n\n")
                
                # Model performance
                f.write("Model Performance:\n")
                f.write("-" * 30 + "\n")
                for model in report['model_performance']:
                    f.write(f"\nModel: {model['model_name']}\n")
                    f.write(f"Type: {model['model_type']}\n")
                    f.write(f"Accuracy: {model['metrics']['accuracy']}\n")
                    f.write(f"Precision: {model['metrics']['precision']}\n")
                    f.write(f"Recall: {model['metrics']['recall']}\n")
                    f.write(f"F1-Score: {model['metrics']['f1_score']}\n")
                
                # Summary
                f.write(f"\nSummary:\n")
                f.write(f"Total Models Tested: {report['summary']['total_models']}\n")
                f.write(f"Best Accuracy: {report['summary']['best_accuracy']['value']} ({report['summary']['best_accuracy']['model']})\n")
                f.write(f"Best F1-Score: {report['summary']['best_f1']['value']} ({report['summary']['best_f1']['model']})\n")
                
        else:
            raise ValueError("Filename must end with .json or .txt")