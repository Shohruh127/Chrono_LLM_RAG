# =============================================================================
# src/forecasting/ets_challenger.py - ETS Statistical Baseline
# Created by: Shohruh127
# Phase 3: Challenger Protocol - ETS Challenger
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class ETSChallenger:
    """
    Exponential Smoothing Time Series (ETS) challenger model.
    Provides statistical baseline for validating AI predictions.
    Handles edge cases gracefully with automatic fallback strategies.
    """

    def __init__(self, auto_select: bool = True, fallback_to_simple: bool = True):
        """
        Initialize ETS Challenger.
        
        Args:
            auto_select: Automatically select best ETS parameters
            fallback_to_simple: Use simple methods for edge cases
        """
        self.auto_select = auto_select
        self.fallback_to_simple = fallback_to_simple
    
    def forecast(self, series: pd.Series, horizon: int = 2) -> Dict:
        """
        Generate ETS forecast for a time series.
        
        Args:
            series: Historical time series data
            horizon: Number of steps to forecast ahead
            
        Returns:
            Dictionary with:
                - median: List[float] - point forecast
                - model_type: str - model used ('ETS', 'SimpleAverage', 'Constant')
                - parameters: dict - model parameters used
        """
        if len(series) < 2:
            raise ValueError("Series must have at least 2 points for forecasting")
        
        # Handle edge cases
        if len(series) < 4:
            return self._simple_average_forecast(series, horizon)
        
        if self._is_constant_series(series):
            return self._constant_forecast(series, horizon)
        
        # Try ETS fitting
        if self.auto_select:
            return self.auto_fit(series, horizon)
        else:
            return self._fit_ets(series, horizon)
    
    def auto_fit(self, series: pd.Series, horizon: int = 2) -> Dict:
        """
        Automatically select and fit best ETS model.
        
        Args:
            series: Historical time series
            horizon: Forecast horizon
            
        Returns:
            Dictionary with forecast and model info
        """
        # Try different ETS configurations
        configs = [
            {'trend': 'add', 'seasonal': None},
            {'trend': None, 'seasonal': None},
            {'trend': 'mul', 'seasonal': None},
        ]
        
        best_forecast = None
        best_aic = float('inf')
        best_params = None
        
        for config in configs:
            try:
                result = self._fit_ets(series, horizon, **config)
                
                # Use simple heuristic for model selection (in real scenario, use AIC/BIC)
                # For now, we'll prefer the first successful fit
                if best_forecast is None:
                    best_forecast = result['median']
                    best_params = config
                    break
                    
            except Exception:
                continue
        
        # Fallback if all ETS models fail
        if best_forecast is None:
            if self.fallback_to_simple:
                return self._simple_average_forecast(series, horizon)
            else:
                raise ValueError("Could not fit any ETS model")
        
        return {
            "median": best_forecast,
            "model_type": "ETS",
            "parameters": best_params
        }
    
    def _fit_ets(self, series: pd.Series, horizon: int, 
                 trend: Optional[str] = 'add', 
                 seasonal: Optional[str] = None) -> Dict:
        """
        Fit ETS model with specified parameters.
        
        Args:
            series: Time series data
            horizon: Forecast horizon
            trend: Trend component ('add', 'mul', None)
            seasonal: Seasonal component ('add', 'mul', None)
            
        Returns:
            Dictionary with forecast results
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Handle zeros in multiplicative models
            if trend == 'mul' or seasonal == 'mul':
                if (series <= 0).any():
                    # Switch to additive
                    trend = 'add' if trend == 'mul' else trend
                    seasonal = 'add' if seasonal == 'mul' else seasonal
            
            # Fit model
            model = ExponentialSmoothing(
                series.values,
                trend=trend,
                seasonal=seasonal,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=True)
            
            # Generate forecast
            forecast_values = fitted_model.forecast(steps=horizon)
            
            return {
                "median": forecast_values.tolist(),
                "model_type": "ETS",
                "parameters": {
                    'trend': trend,
                    'seasonal': seasonal
                }
            }
            
        except Exception as e:
            # If ETS fails, use fallback
            if self.fallback_to_simple:
                return self._simple_average_forecast(series, horizon)
            else:
                raise e
    
    def _simple_average_forecast(self, series: pd.Series, horizon: int) -> Dict:
        """
        Simple average forecast for very short series.
        
        Args:
            series: Time series (< 4 points)
            horizon: Forecast horizon
            
        Returns:
            Dictionary with constant forecast based on average
        """
        mean_value = series.mean()
        forecast = [mean_value] * horizon
        
        return {
            "median": forecast,
            "model_type": "SimpleAverage",
            "parameters": {
                'method': 'mean',
                'value': mean_value
            }
        }
    
    def _constant_forecast(self, series: pd.Series, horizon: int) -> Dict:
        """
        Constant forecast for series with no variation.
        
        Args:
            series: Constant time series
            horizon: Forecast horizon
            
        Returns:
            Dictionary with constant forecast
        """
        constant_value = series.iloc[-1]
        forecast = [constant_value] * horizon
        
        return {
            "median": forecast,
            "model_type": "Constant",
            "parameters": {
                'method': 'constant',
                'value': constant_value
            }
        }
    
    def _is_constant_series(self, series: pd.Series, tolerance: float = 1e-6) -> bool:
        """
        Check if series is effectively constant.
        
        Args:
            series: Time series to check
            tolerance: Tolerance for considering values equal
            
        Returns:
            True if series is constant
        """
        return series.std() < tolerance
