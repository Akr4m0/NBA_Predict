import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from database import NBADatabase
from performance_evaluator import PerformanceEvaluator
from data_importer import DataImporter

class NBADashboard:
    def __init__(self, db_path: str = "nba_predictions.db"):
        self.db = NBADatabase(db_path)
        self.evaluator = PerformanceEvaluator(self.db)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("NBA Game Prediction Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Navigation tabs
            dcc.Tabs(id="main-tabs", value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Model Comparison', value='comparison'),
                dcc.Tab(label='Dataset Analysis', value='datasets'),
                dcc.Tab(label='Detailed Results', value='detailed')
            ]),
            
            html.Div(id='tab-content', style={'padding': 20})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'value')
        )
        def render_tab_content(active_tab):
            if active_tab == 'overview':
                return self.create_overview_tab()
            elif active_tab == 'comparison':
                return self.create_comparison_tab()
            elif active_tab == 'datasets':
                return self.create_datasets_tab()
            elif active_tab == 'detailed':
                return self.create_detailed_tab()
        
        @self.app.callback(
            [Output('performance-chart', 'figure'),
             Output('metrics-table', 'data')],
            [Input('dataset-dropdown', 'value'),
             Input('metric-dropdown', 'value')]
        )
        def update_comparison_charts(selected_dataset, selected_metric):
            return self.update_performance_visualization(selected_dataset, selected_metric)
        
        @self.app.callback(
            Output('detailed-report', 'children'),
            Input('detailed-dataset-dropdown', 'value')
        )
        def update_detailed_report(selected_dataset):
            return self.create_detailed_report_content(selected_dataset)
    
    def create_overview_tab(self):
        """Create overview tab content"""
        # Get summary statistics
        imports_df = self.db.get_import_records()
        models_df = self.db.get_models()
        results_df = self.db.get_prediction_results()
        
        total_imports = len(imports_df)
        total_models = len(models_df)
        total_results = len(results_df)
        
        # Best performing model
        best_model = None
        if total_results > 0:
            best_idx = results_df['accuracy'].idxmax()
            best_model_id = results_df.loc[best_idx, 'model_id']
            best_model = models_df[models_df['id'] == best_model_id].iloc[0]['name']
            best_accuracy = results_df.loc[best_idx, 'accuracy']
        
        return html.Div([
            html.H2("Dashboard Overview"),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3(f"{total_imports}", style={'color': '#1f77b4', 'margin': 0}),
                    html.P("Datasets Imported")
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'border': '1px solid #ddd', 'margin': 10, 'borderRadius': 5}),
                
                html.Div([
                    html.H3(f"{total_models}", style={'color': '#ff7f0e', 'margin': 0}),
                    html.P("Models Trained")
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'border': '1px solid #ddd', 'margin': 10, 'borderRadius': 5}),
                
                html.Div([
                    html.H3(f"{total_results}", style={'color': '#2ca02c', 'margin': 0}),
                    html.P("Prediction Results")
                ], className='summary-card', style={'textAlign': 'center', 'padding': 20, 'border': '1px solid #ddd', 'margin': 10, 'borderRadius': 5}),
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            
            # Best model info
            html.Div([
                html.H3("Best Performing Model"),
                html.P(f"Model: {best_model}" if best_model else "No models trained yet"),
                html.P(f"Accuracy: {best_accuracy:.3f}" if best_model else "")
            ], style={'textAlign': 'center', 'marginTop': 30}) if best_model else html.Div(),
            
            # Recent activity
            html.Div([
                html.H3("Recent Imports"),
                dash_table.DataTable(
                    data=imports_df.tail(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in imports_df.columns if i != 'id'],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'}
                )
            ], style={'marginTop': 30})
        ])
    
    def create_comparison_tab(self):
        """Create model comparison tab"""
        imports_df = self.db.get_import_records()
        
        dataset_options = [{'label': 'All Datasets', 'value': 'all'}]
        dataset_options.extend([{'label': row['filename'], 'value': row['id']} 
                              for _, row in imports_df.iterrows()])
        
        return html.Div([
            html.H2("Model Performance Comparison"),
            
            html.Div([
                html.Div([
                    html.Label("Select Dataset:"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=dataset_options,
                        value='all'
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Primary Metric:"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': 'Accuracy', 'value': 'accuracy'},
                            {'label': 'Precision', 'value': 'precision'},
                            {'label': 'Recall', 'value': 'recall'},
                            {'label': 'F1-Score', 'value': 'f1_score'}
                        ],
                        value='accuracy'
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            # Performance chart
            dcc.Graph(id='performance-chart'),
            
            # Metrics table
            html.H3("Performance Metrics"),
            dash_table.DataTable(
                id='metrics-table',
                columns=[
                    {"name": "Model", "id": "Model_Name"},
                    {"name": "Type", "id": "Model_Type"},
                    {"name": "Accuracy", "id": "Accuracy", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "Precision", "id": "Precision", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "Recall", "id": "Recall", "type": "numeric", "format": {"specifier": ".4f"}},
                    {"name": "F1-Score", "id": "F1_Score", "type": "numeric", "format": {"specifier": ".4f"}}
                ],
                sort_action="native",
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )
        ])
    
    def create_datasets_tab(self):
        """Create datasets analysis tab"""
        imports_df = self.db.get_import_records()
        
        return html.Div([
            html.H2("Dataset Analysis"),
            
            # Datasets table
            dash_table.DataTable(
                data=imports_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in imports_df.columns if i != 'id'],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10
            ),
            
            # Dataset statistics
            html.Div(id='dataset-stats', style={'marginTop': 30})
        ])
    
    def create_detailed_tab(self):
        """Create detailed results tab"""
        imports_df = self.db.get_import_records()
        
        dataset_options = [{'label': row['filename'], 'value': row['id']} 
                          for _, row in imports_df.iterrows()]
        
        return html.Div([
            html.H2("Detailed Results Analysis"),
            
            html.Label("Select Dataset for Detailed Analysis:"),
            dcc.Dropdown(
                id='detailed-dataset-dropdown',
                options=dataset_options,
                value=dataset_options[0]['value'] if dataset_options else None
            ),
            
            html.Div(id='detailed-report', style={'marginTop': 30})
        ])
    
    def update_performance_visualization(self, selected_dataset, selected_metric):
        """Update performance visualization based on selections"""
        # Get comparison data
        import_record_id = None if selected_dataset == 'all' else selected_dataset
        comparison_df = self.evaluator.compare_all_models(import_record_id)
        
        if len(comparison_df) == 0:
            # Return empty figure and data
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No data available", 
                                   xref="paper", yref="paper", 
                                   x=0.5, y=0.5, showarrow=False)
            return empty_fig, []
        
        # Create performance chart
        metric_col = selected_metric.capitalize() if selected_metric != 'f1_score' else 'F1_Score'
        
        fig = px.bar(comparison_df, 
                    x='Model_Name', 
                    y=metric_col,
                    color='Model_Type',
                    title=f'Model {selected_metric.capitalize()} Comparison',
                    labels={metric_col: selected_metric.capitalize()})
        
        fig.update_layout(xaxis_tickangle=-45)
        
        # Prepare table data
        table_data = comparison_df[['Model_Name', 'Model_Type', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].to_dict('records')
        
        return fig, table_data
    
    def create_detailed_report_content(self, selected_dataset):
        """Create detailed report content for selected dataset"""
        if not selected_dataset:
            return html.Div("Please select a dataset")
        
        report = self.evaluator.generate_detailed_report(selected_dataset)
        
        if 'error' in report:
            return html.Div(f"Error: {report['error']}")
        
        # Create content
        content = [
            html.H3("Dataset Information"),
            html.P(f"Filename: {report['dataset_info']['filename']}"),
            html.P(f"Import Date: {report['dataset_info']['import_date']}"),
            html.P(f"Record Count: {report['dataset_info']['record_count']}"),
            
            html.H3("Performance Summary"),
            html.P(f"Total Models Tested: {report['summary']['total_models']}"),
            html.P(f"Best Accuracy: {report['summary']['best_accuracy']['value']:.4f} ({report['summary']['best_accuracy']['model']})"),
            html.P(f"Best F1-Score: {report['summary']['best_f1']['value']:.4f} ({report['summary']['best_f1']['model']})"),
        ]
        
        # Add model details
        content.append(html.H3("Individual Model Results"))
        for model in report['model_performance']:
            content.append(html.Div([
                html.H4(model['model_name']),
                html.P(f"Type: {model['model_type']}"),
                html.P(f"Accuracy: {model['metrics']['accuracy']:.4f}"),
                html.P(f"Precision: {model['metrics']['precision']:.4f}"),
                html.P(f"Recall: {model['metrics']['recall']:.4f}"),
                html.P(f"F1-Score: {model['metrics']['f1_score']:.4f}"),
            ], style={'border': '1px solid #ddd', 'padding': 10, 'margin': 10, 'borderRadius': 5}))
        
        return html.Div(content)
    
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = NBADashboard()
    dashboard.run()