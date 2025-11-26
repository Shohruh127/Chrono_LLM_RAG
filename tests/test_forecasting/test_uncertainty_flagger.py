# =============================================================================
# tests/test_forecasting/test_uncertainty_flagger.py
# Created by: Shohruh127
# Phase 3: Challenger Protocol - Uncertainty Flagger Tests
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from forecasting.uncertainty_flagger import UncertaintyFlagger


class TestUncertaintyFlagger:
    """Test suite for UncertaintyFlagger"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.flagger = UncertaintyFlagger(high_threshold=0.20, medium_threshold=0.10)
        
        # Sample forecast results
        self.chronos_result = {
            'median': [135, 142],
            'quantile_10': [130, 135],
            'quantile_90': [140, 150]
        }
        
        self.ets_result = {
            'median': [132, 155],
            'parameters': {'trend': 'add'}
        }
    
    def test_initialization(self):
        """Test flagger initialization with default parameters"""
        flagger = UncertaintyFlagger()
        assert flagger.high_threshold == 0.20
        assert flagger.medium_threshold == 0.10
    
    def test_initialization_custom_thresholds(self):
        """Test flagger initialization with custom thresholds"""
        flagger = UncertaintyFlagger(high_threshold=0.25, medium_threshold=0.15)
        assert flagger.high_threshold == 0.25
        assert flagger.medium_threshold == 0.15
    
    def test_compare_basic(self):
        """Test basic comparison of two forecasts"""
        comparison = self.flagger.compare(self.chronos_result, self.ets_result)
        
        # Check structure
        assert 'step_1' in comparison
        assert 'step_2' in comparison
        
        # Check step_1 contents
        step1 = comparison['step_1']
        assert 'chronos' in step1
        assert 'ets' in step1
        assert 'divergence' in step1
        assert 'divergence_pct' in step1
        assert 'flag' in step1
        assert 'requires_review' in step1
    
    def test_compare_low_divergence(self):
        """Test comparison with low divergence (<10%)"""
        chronos = {'median': [100, 105]}
        ets = {'median': [102, 106]}
        
        comparison = self.flagger.compare(chronos, ets)
        
        assert comparison['step_1']['flag'] == 'LOW'
        assert comparison['step_1']['requires_review'] is False
        assert comparison['step_2']['flag'] == 'LOW'
        assert comparison['step_2']['requires_review'] is False
    
    def test_compare_medium_divergence(self):
        """Test comparison with medium divergence (10-20%)"""
        chronos = {'median': [100, 120]}
        ets = {'median': [112, 132]}  # ~11-12% divergence
        
        comparison = self.flagger.compare(chronos, ets)
        
        assert comparison['step_1']['flag'] == 'MEDIUM'
        assert comparison['step_1']['requires_review'] is False
    
    def test_compare_high_divergence(self):
        """Test comparison with high divergence (>20%)"""
        chronos = {'median': [100, 150]}
        ets = {'median': [130, 200]}  # >20% divergence
        
        comparison = self.flagger.compare(chronos, ets)
        
        assert comparison['step_1']['flag'] == 'HIGH'
        assert comparison['step_1']['requires_review'] is True
        assert comparison['step_2']['flag'] == 'HIGH'
        assert comparison['step_2']['requires_review'] is True
    
    def test_compare_different_lengths(self):
        """Test comparison with different forecast lengths"""
        chronos = {'median': [100, 105, 110]}
        ets = {'median': [102, 106]}
        
        comparison = self.flagger.compare(chronos, ets)
        
        # Should only compare overlapping steps
        assert len(comparison) == 2
        assert 'step_1' in comparison
        assert 'step_2' in comparison
        assert 'step_3' not in comparison
    
    def test_compare_empty_results(self):
        """Test comparison with empty results raises error"""
        chronos = {'median': []}
        ets = {'median': [100]}
        
        with pytest.raises(ValueError, match="must contain 'median'"):
            self.flagger.compare(chronos, ets)
    
    def test_compare_missing_median(self):
        """Test comparison with missing median key"""
        chronos = {'values': [100, 105]}
        ets = {'median': [102, 106]}
        
        with pytest.raises(ValueError, match="must contain 'median'"):
            self.flagger.compare(chronos, ets)
    
    def test_compare_divergence_calculation(self):
        """Test divergence percentage calculation"""
        chronos = {'median': [100]}
        ets = {'median': [80]}
        
        comparison = self.flagger.compare(chronos, ets)
        
        # Divergence should be |100-80|/80 = 0.25 = 25%
        assert abs(comparison['step_1']['divergence_pct'] - 25.0) < 0.01
    
    def test_compare_zero_ets_value(self):
        """Test comparison when ETS value is zero (edge case)"""
        chronos = {'median': [100]}
        ets = {'median': [0]}
        
        comparison = self.flagger.compare(chronos, ets)
        
        # Should handle division by near-zero
        assert 'divergence_pct' in comparison['step_1']
        assert comparison['step_1']['divergence_pct'] > 0
    
    def test_flag_dataframe_basic(self):
        """Test adding flags to DataFrame"""
        df = pd.DataFrame({
            'chronos_forecast': [100, 110, 120],
            'ets_forecast': [102, 130, 160]
        })
        
        flagged_df = self.flagger.flag_dataframe(df)
        
        # Check new columns exist
        assert '_uncertainty_flag' in flagged_df.columns
        assert '_divergence_pct' in flagged_df.columns
        assert '_requires_review' in flagged_df.columns
        
        # Check DataFrame length unchanged
        assert len(flagged_df) == len(df)
    
    def test_flag_dataframe_custom_columns(self):
        """Test adding flags with custom column names"""
        df = pd.DataFrame({
            'ai_pred': [100, 110],
            'stat_pred': [102, 130]
        })
        
        flagged_df = self.flagger.flag_dataframe(
            df, 
            chronos_col='ai_pred', 
            ets_col='stat_pred'
        )
        
        assert '_uncertainty_flag' in flagged_df.columns
        assert len(flagged_df) == 2
    
    def test_flag_dataframe_missing_columns(self):
        """Test flag_dataframe with missing columns raises error"""
        df = pd.DataFrame({
            'chronos_forecast': [100, 110]
        })
        
        with pytest.raises(ValueError, match="must contain both"):
            self.flagger.flag_dataframe(df)
    
    def test_flag_dataframe_flag_values(self):
        """Test that correct flags are assigned in DataFrame"""
        df = pd.DataFrame({
            'chronos_forecast': [100, 110, 150],
            'ets_forecast': [102, 130, 200]  # LOW, MEDIUM, HIGH
        })
        
        flagged_df = self.flagger.flag_dataframe(df)
        
        # Check flag assignments
        assert flagged_df.iloc[0]['_uncertainty_flag'] == 'LOW'
        assert flagged_df.iloc[1]['_uncertainty_flag'] == 'MEDIUM'
        assert flagged_df.iloc[2]['_uncertainty_flag'] == 'HIGH'
        
        # Check review requirements
        assert flagged_df.iloc[0]['_requires_review'] == False
        assert flagged_df.iloc[1]['_requires_review'] == False
        assert flagged_df.iloc[2]['_requires_review'] == True
    
    def test_flag_dataframe_original_unchanged(self):
        """Test that original DataFrame is not modified"""
        df = pd.DataFrame({
            'chronos_forecast': [100, 110],
            'ets_forecast': [102, 130]
        })
        
        original_columns = df.columns.tolist()
        flagged_df = self.flagger.flag_dataframe(df)
        
        # Original should be unchanged
        assert df.columns.tolist() == original_columns
        assert '_uncertainty_flag' not in df.columns
    
    def test_get_flagged_points(self):
        """Test extracting flagged points"""
        chronos = {'median': [100, 150]}
        ets = {'median': [130, 200]}  # Both HIGH
        
        comparison = self.flagger.compare(chronos, ets)
        flagged = self.flagger.get_flagged_points(comparison)
        
        assert len(flagged) == 2
        assert all('step' in point for point in flagged)
        assert all('chronos' in point for point in flagged)
        assert all('ets' in point for point in flagged)
        assert all('divergence_pct' in point for point in flagged)
    
    def test_get_flagged_points_no_flags(self):
        """Test extracting flagged points when none exist"""
        chronos = {'median': [100, 105]}
        ets = {'median': [102, 106]}  # Both LOW
        
        comparison = self.flagger.compare(chronos, ets)
        flagged = self.flagger.get_flagged_points(comparison)
        
        assert len(flagged) == 0
    
    def test_summary_report(self):
        """Test generating summary report"""
        chronos = {'median': [100, 110, 150, 160]}
        ets = {'median': [102, 130, 200, 162]}  # LOW, MEDIUM, HIGH, LOW
        
        comparison = self.flagger.compare(chronos, ets)
        summary = self.flagger.summary_report(comparison)
        
        # Check structure
        assert 'total_points' in summary
        assert 'high_uncertainty' in summary
        assert 'medium_uncertainty' in summary
        assert 'low_uncertainty' in summary
        assert 'requires_review' in summary
        assert 'avg_divergence_pct' in summary
        assert 'max_divergence_pct' in summary
        
        # Check values
        assert summary['total_points'] == 4
        assert summary['high_uncertainty'] == 1
        assert summary['medium_uncertainty'] == 1
        assert summary['low_uncertainty'] == 2
        assert summary['requires_review'] == 1
    
    def test_summary_report_all_low(self):
        """Test summary report with all low divergence"""
        chronos = {'median': [100, 105]}
        ets = {'median': [102, 106]}
        
        comparison = self.flagger.compare(chronos, ets)
        summary = self.flagger.summary_report(comparison)
        
        assert summary['high_uncertainty'] == 0
        assert summary['medium_uncertainty'] == 0
        assert summary['low_uncertainty'] == 2
        assert summary['requires_review'] == 0
    
    def test_threshold_boundary_conditions(self):
        """Test flag assignment at exact threshold boundaries"""
        # Test exact 10% divergence
        chronos_10 = {'median': [100]}
        ets_10 = {'median': [110]}
        comparison_10 = self.flagger.compare(chronos_10, ets_10)
        # At boundary, should be MEDIUM
        
        # Test exact 20% divergence
        chronos_20 = {'median': [100]}
        ets_20 = {'median': [120]}
        comparison_20 = self.flagger.compare(chronos_20, ets_20)
        # At boundary, should be MEDIUM (not HIGH because we use >)
        
        # Just over 20% should be HIGH (need more than 20%)
        chronos_21 = {'median': [100]}
        ets_21 = {'median': [130]}  # 30% divergence
        comparison_21 = self.flagger.compare(chronos_21, ets_21)
        assert comparison_21['step_1']['flag'] == 'HIGH'
