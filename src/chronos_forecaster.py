# =============================================================================
# src/chronos_forecaster.py - Chronos-2 Forecasting Module
# Created by: Shohruh127
# Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-11-19 11:11:39
# Current User's Login: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import pandas as pd
import torch
from chronos import Chronos2Pipeline
from typing import Optional, List
import yaml
from pathlib import Path


class ChronosForecaster:
    """Chronos-2 time series forecasting wrapper"""

    def __init__(self, config_path: str = "configs/chronos_config.yaml"):
        """
        Initialize Chronos forecaster
        
        Args:
            config_path: Path to configuration file
        """
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.pipeline = None
        self.historical_data = None
        self.predictions = None

    def _default_config(self) -> dict:
        """Default configuration if file not found"""
        return {
            'model': {
                'name': 'amazon/chronos-2',
                'device': 'cuda',
                'dtype': 'bfloat16'
            },
            'forecasting': {
                'prediction_length': 4,
                'quantile_levels': [0.1, 0.5, 0.9],
                'batch_size': 256
            },
            'seed': 42
        }

    def load_model(self):
        """Load Chronos-2 model"""
        device = self.config['model']['device']
        dtype = self.config['model']['dtype']

        print(f"📥 Loading Chronos-2 model...")
        print(f"   Device: {device}")
        print(f"   Dtype: {dtype}")

        torch_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.config['model']['name'],
            device_map=device,
            torch_dtype=torch_dtype
        )

        print(f"✅ Model loaded successfully")

    def load_data(self, data: pd.DataFrame):
        """
        Load historical data
        
        Args:
            data: DataFrame with columns ['id', 'timestamp', 'target']
        """
        required_cols = ['id', 'timestamp', 'target']
        missing = [c for c in required_cols if c not in data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.historical_data = data.copy()
        self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])

        print(f"✅ Loaded {len(data):,} historical records")
        print(f"   Time series: {data['id'].nunique()}")
        print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """
        Generate forecasts
        
        Args:
            horizon: Number of steps to forecast (default from config)
            
        Returns:
            DataFrame with predictions
        """
        if self.pipeline is None:
            self.load_model()

        if self.historical_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if horizon is None:
            horizon = self.config['forecasting']['prediction_length']

        print(f"🔮 Generating forecasts...")
        print(f"   Horizon: {horizon} steps")
        print(f"   Batch size: {self.config['forecasting']['batch_size']}")

        predictions = self.pipeline.predict_df(
            self.historical_data,
            prediction_length=horizon,
            quantile_levels=self.config['forecasting']['quantile_levels'],
            id_column='id',
            timestamp_column='timestamp',
            target='target',
            batch_size=self.config['forecasting']['batch_size']
        )

        self.predictions = predictions

        print(f"✅ Generated {len(predictions):,} predictions")
        print(f"   Time series: {predictions.index.nunique()}")

        return predictions

    def save_predictions(self, filepath: str):
        """Save predictions to CSV"""
        if self.predictions is None:
            raise ValueError("No predictions. Call predict() first.")

        self.predictions.to_csv(filepath, index=True)
        print(f"💾 Saved predictions to {filepath}")

    def get_metrics(self, test_data: Optional[pd.DataFrame] = None) -> dict:
        """
        Calculate forecast metrics
        
        Args:
            test_data: Test dataset for validation
            
        Returns:
            Dictionary of metrics
        """
        if self.predictions is None:
            raise ValueError("No predictions available")

        metrics = {
            'total_predictions': len(self.predictions),
            'time_series': self.predictions.index.nunique(),
            'date_range': (
                self.predictions['timestamp'].min(),
                self.predictions['timestamp'].max()
            ),
            'mean_prediction': self.predictions['predictions'].mean(),
            'std_prediction': self.predictions['predictions'].std()
        }

        if test_data is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np

            merged = pd.merge(
                test_data,
                self.predictions.reset_index(),
                on=['id', 'timestamp'],
                how='inner'
            )

            if len(merged) > 0:
                mae = mean_absolute_error(merged['target'], merged['predictions'])
                rmse = np.sqrt(mean_squared_error(merged['target'], merged['predictions']))
                mape = np.mean(np.abs((merged['target'] - merged['predictions']) / merged['target'])) * 100

                metrics.update({
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })

        return metrics
