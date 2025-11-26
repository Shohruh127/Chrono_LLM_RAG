# =============================================================================
# tests/test_forecasting/test_chronos_engine.py
# Created by: Shohruh127
# Phase 3: Challenger Protocol - Chronos Engine Tests
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from forecasting.chronos_engine import ChronosEngine


class TestChronosEngine:
    """Test suite for ChronosEngine"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = ChronosEngine()
        
        # Sample series for testing
        self.normal_series = pd.Series([100, 105, 110, 108, 115, 120, 125, 130, 128])
        self.short_series = pd.Series([100, 105, 110])
        self.very_short_series = pd.Series([100, 105])
        self.constant_series = pd.Series([100] * 10)
    
    def test_initialization(self):
        """Test engine initialization with default parameters"""
        engine = ChronosEngine()
        assert engine.model_id == "amazon/chronos-t5-base"
        assert engine.quantiles == [0.1, 0.5, 0.9]
        assert engine.pipeline is None
    
    def test_initialization_custom_params(self):
        """Test engine initialization with custom parameters"""
        engine = ChronosEngine(
            model_id="custom-model",
            quantiles=[0.05, 0.5, 0.95]
        )
        assert engine.model_id == "custom-model"
        assert engine.quantiles == [0.05, 0.5, 0.95]
    
    def test_forecast_normal_series(self):
        """Test forecasting with normal time series"""
        result = self.engine.forecast(self.normal_series, horizon=2)
        
        # Check structure
        assert 'median' in result
        assert 'quantile_10' in result
        assert 'quantile_90' in result
        assert 'horizon' in result
        assert 'model_version' in result
        
        # Check data types
        assert isinstance(result['median'], list)
        assert isinstance(result['quantile_10'], list)
        assert isinstance(result['quantile_90'], list)
        assert isinstance(result['horizon'], int)
        
        # Check lengths
        assert len(result['median']) == 2
        assert len(result['quantile_10']) == 2
        assert len(result['quantile_90']) == 2
        assert result['horizon'] == 2
    
    def test_forecast_short_series(self):
        """Test forecasting with short time series (< 4 points)"""
        result = self.engine.forecast(self.short_series, horizon=2)
        
        # Should use fallback
        assert 'median' in result
        assert len(result['median']) == 2
        assert 'fallback' in result['model_version']
    
    def test_forecast_very_short_series(self):
        """Test forecasting with very short series (2 points)"""
        result = self.engine.forecast(self.very_short_series, horizon=2)
        
        assert 'median' in result
        assert len(result['median']) == 2
    
    def test_forecast_insufficient_data(self):
        """Test forecasting with insufficient data raises error"""
        single_point = pd.Series([100])
        
        with pytest.raises(ValueError, match="at least 2 points"):
            self.engine.forecast(single_point, horizon=2)
    
    def test_forecast_different_horizons(self):
        """Test forecasting with different horizon values"""
        for horizon in [1, 2, 5, 10]:
            result = self.engine.forecast(self.normal_series, horizon=horizon)
            assert len(result['median']) == horizon
    
    def test_forecast_constant_series(self):
        """Test forecasting with constant series"""
        result = self.engine.forecast(self.constant_series, horizon=3)
        
        # Forecast should be close to constant value
        assert 'median' in result
        assert len(result['median']) == 3
        # Values should be relatively constant
        for val in result['median']:
            assert 95 <= val <= 105  # Allow some variation
    
    def test_forecast_batch_single_column(self):
        """Test batch forecasting with single column"""
        df = pd.DataFrame({
            'series1': [100, 105, 110, 115, 120]
        })
        
        result = self.engine.forecast_batch(df, target_cols=['series1'], horizon=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 forecast steps
        assert 'series_name' in result.columns
        assert 'forecast_step' in result.columns
        assert 'median' in result.columns
        assert 'quantile_10' in result.columns
        assert 'quantile_90' in result.columns
    
    def test_forecast_batch_multiple_columns(self):
        """Test batch forecasting with multiple columns"""
        df = pd.DataFrame({
            'series1': [100, 105, 110, 115, 120],
            'series2': [200, 210, 220, 230, 240],
            'series3': [50, 52, 54, 56, 58]
        })
        
        result = self.engine.forecast_batch(df, target_cols=['series1', 'series2', 'series3'], horizon=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 3 series * 2 forecast steps
        assert set(result['series_name'].unique()) == {'series1', 'series2', 'series3'}
    
    def test_forecast_batch_missing_column(self):
        """Test batch forecasting with missing column (should skip)"""
        df = pd.DataFrame({
            'series1': [100, 105, 110, 115, 120]
        })
        
        # Should not raise error, just skip missing column
        result = self.engine.forecast_batch(df, target_cols=['series1', 'nonexistent'], horizon=2)
        
        assert isinstance(result, pd.DataFrame)
        # Only series1 should be in results
        assert set(result['series_name'].unique()) == {'series1'}
    
    def test_forecast_batch_with_nans(self):
        """Test batch forecasting with NaN values"""
        df = pd.DataFrame({
            'series1': [100, np.nan, 110, 115, 120]
        })
        
        result = self.engine.forecast_batch(df, target_cols=['series1'], horizon=2)
        
        # Should handle NaNs by dropping them
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_fallback_forecast_linear_trend(self):
        """Test fallback forecasting captures linear trend"""
        # Create series with clear linear trend
        series = pd.Series([10, 20, 30, 40])
        result = self.engine._fallback_forecast(series, horizon=2)
        
        # Forecast should continue the trend
        assert result['median'][0] > 40
        assert result['median'][1] > result['median'][0]
    
    def test_fallback_forecast_uncertainty_bounds(self):
        """Test fallback forecasting includes uncertainty bounds"""
        result = self.engine._fallback_forecast(self.normal_series, horizon=2)
        
        # Lower bound should be less than median
        assert all(result['quantile_10'][i] < result['median'][i] for i in range(2))
        
        # Upper bound should be greater than median
        assert all(result['quantile_90'][i] > result['median'][i] for i in range(2))
    
    def test_forecast_output_format(self):
        """Test forecast output has correct format"""
        result = self.engine.forecast(self.normal_series, horizon=3)
        
        # All lists should have same length
        assert len(result['median']) == len(result['quantile_10'])
        assert len(result['median']) == len(result['quantile_90'])
        assert len(result['median']) == result['horizon']
        
        # All values should be numeric
        for val in result['median']:
            assert isinstance(val, (int, float, np.number))
        for val in result['quantile_10']:
            assert isinstance(val, (int, float, np.number))
        for val in result['quantile_90']:
            assert isinstance(val, (int, float, np.number))
