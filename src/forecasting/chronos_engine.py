# =============================================================================
# src/forecasting/chronos_engine.py - Chronos-2 Forecasting Engine
# Created by: Shohruh127
# Phase 3: Challenger Protocol - Chronos Engine
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import torch and Chronos at module level
try:
    import torch
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    warnings.warn("Chronos/torch not available, will use fallback predictions")


class ChronosEngine:
    """
    Wrapper for Amazon Chronos-2 forecasting with probabilistic outputs.
    Handles short time series and provides configurable quantile predictions.
    """

    def __init__(self, model_id: str = "amazon/chronos-t5-base", quantiles: Optional[List[float]] = None):
        """
        Initialize Chronos Engine.
        
        Args:
            model_id: Chronos model identifier
            quantiles: List of quantiles to compute (default: [0.1, 0.5, 0.9])
        """
        self.model_id = model_id
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.pipeline = None
        
    def _lazy_load_pipeline(self):
        """Lazy load the Chronos pipeline when needed."""
        if self.pipeline is None and CHRONOS_AVAILABLE:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=device,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
                )
            except Exception as e:
                self.pipeline = None
                warnings.warn(f"Failed to load Chronos pipeline: {e}")
        elif not CHRONOS_AVAILABLE:
            self.pipeline = None
    
    def forecast(self, series: pd.Series, horizon: int = 2) -> Dict:
        """
        Generate probabilistic forecast for a single time series.
        
        Args:
            series: Historical time series data
            horizon: Number of steps to forecast ahead
            
        Returns:
            Dictionary with:
                - median: List[float] - point forecast (50th percentile)
                - quantile_10: List[float] - 10th percentile (low bound)
                - quantile_90: List[float] - 90th percentile (high bound)
                - horizon: int - forecast horizon
                - model_version: str - model identifier
        """
        if len(series) < 2:
            raise ValueError("Series must have at least 2 points for forecasting")
        
        # Handle short series by using simple extrapolation as fallback
        if len(series) < 4:
            return self._fallback_forecast(series, horizon)
        
        self._lazy_load_pipeline()
        
        if self.pipeline is None:
            return self._fallback_forecast(series, horizon)
        
        try:
            # Convert series to tensor format expected by Chronos
            context = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0)
            
            # Generate forecast
            forecast_result = self.pipeline.predict(
                context=context,
                prediction_length=horizon,
                num_samples=100
            )
            
            # Extract quantiles
            quantile_values = {}
            for q in self.quantiles:
                quantile_values[f"quantile_{int(q*100)}"] = forecast_result.quantile(q, dim=1).squeeze().tolist()
            
            median = quantile_values.get("quantile_50", forecast_result.median(dim=1).squeeze().tolist())
            quantile_10 = quantile_values.get("quantile_10", [])
            quantile_90 = quantile_values.get("quantile_90", [])
            
            # Ensure lists are proper format
            if not isinstance(median, list):
                median = [median]
            
            result = {
                "median": median,
                "quantile_10": quantile_10 if quantile_10 else median,
                "quantile_90": quantile_90 if quantile_90 else median,
                "horizon": horizon,
                "model_version": self.model_id
            }
                
            return result
            
        except Exception as e:
            warnings.warn(f"Chronos forecast failed: {e}. Using fallback.")
            return self._fallback_forecast(series, horizon)
    
    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        """
        Fallback forecasting using linear extrapolation for short series.
        
        Args:
            series: Historical time series
            horizon: Forecast horizon
            
        Returns:
            Dictionary with median predictions and uncertainty bounds
        """
        values = series.values
        
        # Linear trend estimation
        if len(values) >= 2:
            x = np.arange(len(values))
            y = values
            
            # Simple linear regression
            slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
            intercept = np.mean(y) - slope * np.mean(x)
            
            # Forecast
            future_x = np.arange(len(values), len(values) + horizon)
            median = [slope * xi + intercept for xi in future_x]
            
            # Add uncertainty bounds based on historical std
            std = np.std(values)
            quantile_10 = [m - 1.28 * std for m in median]  # ~10th percentile
            quantile_90 = [m + 1.28 * std for m in median]  # ~90th percentile
        else:
            # Very short series - use last value
            last_val = values[-1]
            median = [last_val] * horizon
            std = abs(last_val) * 0.1  # 10% uncertainty
            quantile_10 = [last_val - std] * horizon
            quantile_90 = [last_val + std] * horizon
        
        return {
            "median": median,
            "quantile_10": quantile_10,
            "quantile_90": quantile_90,
            "horizon": horizon,
            "model_version": f"{self.model_id}_fallback"
        }
    
    def forecast_batch(self, df: pd.DataFrame, target_cols: List[str], horizon: int = 2) -> pd.DataFrame:
        """
        Generate forecasts for multiple time series in batch.
        
        Args:
            df: DataFrame with time series (rows are time steps)
            target_cols: List of column names to forecast
            horizon: Forecast horizon
            
        Returns:
            DataFrame with forecast results for each series
        """
        results = []
        
        for col in target_cols:
            if col not in df.columns:
                warnings.warn(f"Column {col} not found in DataFrame, skipping")
                continue
            
            series = df[col].dropna()
            
            if len(series) < 2:
                warnings.warn(f"Column {col} has insufficient data, skipping")
                continue
            
            try:
                forecast_result = self.forecast(series, horizon)
                
                for i, (median_val, low_val, high_val) in enumerate(zip(
                    forecast_result["median"],
                    forecast_result["quantile_10"],
                    forecast_result["quantile_90"]
                )):
                    results.append({
                        "series_name": col,
                        "forecast_step": i + 1,
                        "median": median_val,
                        "quantile_10": low_val,
                        "quantile_90": high_val,
                        "model_version": forecast_result["model_version"]
                    })
            except Exception as e:
                warnings.warn(f"Error forecasting {col}: {e}")
                continue
        
        return pd.DataFrame(results)
