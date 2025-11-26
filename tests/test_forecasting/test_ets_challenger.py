# =============================================================================
# tests/test_forecasting/test_ets_challenger.py
# Created by: Shohruh127
# Phase 3: Challenger Protocol - ETS Challenger Tests
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from forecasting.ets_challenger import ETSChallenger


class TestETSChallenger:
    """Test suite for ETSChallenger"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.challenger = ETSChallenger()
        
        # Sample series for testing
        self.normal_series = pd.Series([100, 105, 110, 108, 115, 120, 125, 130, 128])
        self.short_series = pd.Series([100, 105, 110])
        self.very_short_series = pd.Series([100, 105])
        self.constant_series = pd.Series([100.0] * 10)
        self.series_with_zeros = pd.Series([0, 1, 2, 3, 4, 5])
    
    def test_initialization(self):
        """Test challenger initialization with default parameters"""
        challenger = ETSChallenger()
        assert challenger.auto_select is True
        assert challenger.fallback_to_simple is True
    
    def test_initialization_custom_params(self):
        """Test challenger initialization with custom parameters"""
        challenger = ETSChallenger(auto_select=False, fallback_to_simple=False)
        assert challenger.auto_select is False
        assert challenger.fallback_to_simple is False
    
    def test_forecast_normal_series(self):
        """Test forecasting with normal time series"""
        result = self.challenger.forecast(self.normal_series, horizon=2)
        
        # Check structure
        assert 'median' in result
        assert 'model_type' in result
        assert 'parameters' in result
        
        # Check data types
        assert isinstance(result['median'], list)
        assert isinstance(result['model_type'], str)
        assert isinstance(result['parameters'], dict)
        
        # Check lengths
        assert len(result['median']) == 2
    
    def test_forecast_short_series(self):
        """Test forecasting with short time series (< 4 points)"""
        result = self.challenger.forecast(self.short_series, horizon=2)
        
        # Should use simple average
        assert 'median' in result
        assert len(result['median']) == 2
        assert result['model_type'] == 'SimpleAverage'
    
    def test_forecast_very_short_series(self):
        """Test forecasting with very short series (2 points)"""
        result = self.challenger.forecast(self.very_short_series, horizon=2)
        
        assert 'median' in result
        assert len(result['median']) == 2
        assert result['model_type'] == 'SimpleAverage'
    
    def test_forecast_insufficient_data(self):
        """Test forecasting with insufficient data raises error"""
        single_point = pd.Series([100])
        
        with pytest.raises(ValueError, match="at least 2 points"):
            self.challenger.forecast(single_point, horizon=2)
    
    def test_forecast_constant_series(self):
        """Test forecasting with constant series"""
        result = self.challenger.forecast(self.constant_series, horizon=3)
        
        # Should use constant forecast
        assert result['model_type'] == 'Constant'
        assert len(result['median']) == 3
        
        # All forecast values should be 100
        for val in result['median']:
            assert abs(val - 100.0) < 1e-6
    
    def test_forecast_series_with_zeros(self):
        """Test forecasting with series containing zeros"""
        result = self.challenger.forecast(self.series_with_zeros, horizon=2)
        
        # Should handle zeros gracefully
        assert 'median' in result
        assert len(result['median']) == 2
    
    def test_forecast_different_horizons(self):
        """Test forecasting with different horizon values"""
        for horizon in [1, 2, 5, 10]:
            result = self.challenger.forecast(self.normal_series, horizon=horizon)
            assert len(result['median']) == horizon
    
    def test_auto_fit(self):
        """Test automatic parameter selection"""
        result = self.challenger.auto_fit(self.normal_series, horizon=2)
        
        assert 'median' in result
        assert 'model_type' in result
        assert 'parameters' in result
        assert len(result['median']) == 2
    
    def test_simple_average_forecast(self):
        """Test simple average forecasting method"""
        result = self.challenger._simple_average_forecast(self.short_series, horizon=3)
        
        assert result['model_type'] == 'SimpleAverage'
        assert len(result['median']) == 3
        
        # All values should be close to the mean
        expected_mean = self.short_series.mean()
        for val in result['median']:
            assert abs(val - expected_mean) < 1e-6
    
    def test_constant_forecast(self):
        """Test constant forecasting method"""
        result = self.challenger._constant_forecast(self.constant_series, horizon=4)
        
        assert result['model_type'] == 'Constant'
        assert len(result['median']) == 4
        
        # All forecast values should equal last value
        last_value = self.constant_series.iloc[-1]
        for val in result['median']:
            assert abs(val - last_value) < 1e-6
    
    def test_is_constant_series_true(self):
        """Test constant series detection - positive case"""
        assert self.challenger._is_constant_series(self.constant_series) == True
    
    def test_is_constant_series_false(self):
        """Test constant series detection - negative case"""
        assert self.challenger._is_constant_series(self.normal_series) == False
    
    def test_is_constant_series_near_constant(self):
        """Test constant series detection with small variations"""
        near_constant = pd.Series([100.0, 100.0000001, 99.9999999, 100.0])
        assert self.challenger._is_constant_series(near_constant) == True
    
    def test_fit_ets_with_trend(self):
        """Test ETS fitting with trend component"""
        # Series with clear trend
        trending_series = pd.Series([10, 20, 30, 40, 50, 60])
        
        result = self.challenger._fit_ets(trending_series, horizon=2, trend='add', seasonal=None)
        
        assert 'median' in result
        assert len(result['median']) == 2
        # Forecast should continue upward trend
        assert result['median'][0] > 60
    
    def test_forecast_output_format(self):
        """Test forecast output has correct format"""
        result = self.challenger.forecast(self.normal_series, horizon=3)
        
        # Required keys
        assert 'median' in result
        assert 'model_type' in result
        assert 'parameters' in result
        
        # Correct types
        assert isinstance(result['median'], list)
        assert isinstance(result['model_type'], str)
        assert isinstance(result['parameters'], dict)
        
        # Correct length
        assert len(result['median']) == 3
        
        # All values should be numeric
        for val in result['median']:
            assert isinstance(val, (int, float, np.number))
    
    def test_forecast_with_negative_values(self):
        """Test forecasting with negative values"""
        negative_series = pd.Series([-10, -5, 0, 5, 10, 15])
        result = self.challenger.forecast(negative_series, horizon=2)
        
        assert 'median' in result
        assert len(result['median']) == 2
    
    def test_forecast_with_large_values(self):
        """Test forecasting with large values"""
        large_series = pd.Series([1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6])
        result = self.challenger.forecast(large_series, horizon=2)
        
        assert 'median' in result
        assert len(result['median']) == 2
    
    def test_forecast_reproducibility(self):
        """Test that forecasts are reproducible"""
        result1 = self.challenger.forecast(self.normal_series, horizon=2)
        result2 = self.challenger.forecast(self.normal_series, horizon=2)
        
        # Results should be the same
        assert result1['median'] == result2['median']
        assert result1['model_type'] == result2['model_type']
    
    def test_fallback_behavior(self):
        """Test fallback to simple methods when ETS fails"""
        challenger = ETSChallenger(auto_select=False, fallback_to_simple=True)
        
        # Very short series should trigger fallback
        result = challenger.forecast(self.short_series, horizon=2)
        
        assert result['model_type'] in ['SimpleAverage', 'Constant', 'ETS']
    
    def test_parameters_included(self):
        """Test that model parameters are included in output"""
        result = self.challenger.forecast(self.normal_series, horizon=2)
        
        assert 'parameters' in result
        params = result['parameters']
        
        # Parameters should be a non-empty dict
        assert isinstance(params, dict)
        assert len(params) > 0
