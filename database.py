import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import json

class NBADatabase:
    def __init__(self, db_path: str = "nba_predictions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Import records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS import_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_path TEXT,
                    record_count INTEGER,
                    description TEXT
                )
            ''')
            
            # Models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    parameters TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Prediction results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    import_record_id INTEGER,
                    model_id INTEGER,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    predictions TEXT,
                    actual_results TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (import_record_id) REFERENCES import_records (id),
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            ''')
            
            # Historical game data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS game_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    import_record_id INTEGER,
                    home_team TEXT,
                    away_team TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    game_date DATE,
                    season TEXT,
                    home_stats TEXT,
                    away_stats TEXT,
                    result TEXT,
                    FOREIGN KEY (import_record_id) REFERENCES import_records (id)
                )
            ''')
            
            conn.commit()
    
    def register_import(self, filename: str, file_path: str, record_count: int, description: str = None) -> int:
        """Register a new data import and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO import_records (filename, file_path, record_count, description)
                VALUES (?, ?, ?, ?)
            ''', (filename, file_path, record_count, description))
            return cursor.lastrowid
    
    def register_model(self, name: str, model_type: str, parameters: Dict[str, Any]) -> int:
        """Register a new model and return its ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO models (name, type, parameters)
                VALUES (?, ?, ?)
            ''', (name, model_type, json.dumps(parameters)))
            return cursor.lastrowid
    
    def save_prediction_results(self, import_record_id: int, model_id: int, 
                              accuracy: float, precision: float, recall: float, 
                              f1: float, predictions: List, actual_results: List = None):
        """Save prediction results to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_results 
                (import_record_id, model_id, accuracy, precision_score, recall, f1_score, predictions, actual_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (import_record_id, model_id, accuracy, precision, recall, f1, 
                  json.dumps(predictions), json.dumps(actual_results) if actual_results else None))
            return cursor.lastrowid
    
    def save_game_data(self, import_record_id: int, games_df: pd.DataFrame):
        """Save game data to database"""
        with sqlite3.connect(self.db_path) as conn:
            for _, row in games_df.iterrows():
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO game_data 
                    (import_record_id, home_team, away_team, home_score, away_score, 
                     game_date, season, home_stats, away_stats, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    import_record_id,
                    row.get('home_team'),
                    row.get('away_team'), 
                    row.get('home_score'),
                    row.get('away_score'),
                    row.get('game_date'),
                    row.get('season'),
                    json.dumps(row.get('home_stats', {})),
                    json.dumps(row.get('away_stats', {})),
                    row.get('result')
                ))
    
    def get_import_records(self) -> pd.DataFrame:
        """Get all import records"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM import_records", conn)
    
    def get_models(self) -> pd.DataFrame:
        """Get all models"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM models", conn)
    
    def get_prediction_results(self, import_record_id: int = None, model_id: int = None) -> pd.DataFrame:
        """Get prediction results with optional filtering"""
        query = "SELECT * FROM prediction_results"
        params = []
        
        if import_record_id or model_id:
            query += " WHERE "
            conditions = []
            if import_record_id:
                conditions.append("import_record_id = ?")
                params.append(import_record_id)
            if model_id:
                conditions.append("model_id = ?")
                params.append(model_id)
            query += " AND ".join(conditions)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_game_data(self, import_record_id: int = None) -> pd.DataFrame:
        """Get game data with optional filtering by import record"""
        query = "SELECT * FROM game_data"
        params = []
        
        if import_record_id:
            query += " WHERE import_record_id = ?"
            params.append(import_record_id)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)