# =============================================================================
# src/forecasting/uncertainty_flagger.py - Uncertainty Detection
# Created by: Shohruh127
# Phase 3: Challenger Protocol - Uncertainty Flagger
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


class UncertaintyFlagger:
    """
    Compares Chronos AI predictions with ETS statistical baseline.
    Flags high-divergence predictions for analyst review.
    """

    def __init__(self, high_threshold: float = 0.20, medium_threshold: float = 0.10):
        """
        Initialize Uncertainty Flagger.
        
        Args:
            high_threshold: Threshold for HIGH uncertainty (default 20%)
            medium_threshold: Threshold for MEDIUM uncertainty (default 10%)
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def _calculate_divergence(self, chronos_val: float, ets_val: float) -> float:
        """
        Calculate percentage divergence between two values.
        
        Args:
            chronos_val: Chronos forecast value
            ets_val: ETS forecast value
            
        Returns:
            Divergence as a decimal (0.20 = 20%)
        """
        # Use max of absolute values to handle negative numbers properly
        denominator = max(abs(chronos_val), abs(ets_val), 1e-6)
        return abs(chronos_val - ets_val) / denominator
    
    def _get_flag(self, divergence: float) -> tuple:
        """
        Determine uncertainty flag and review requirement.
        
        Args:
            divergence: Divergence value (decimal)
            
        Returns:
            Tuple of (flag_str, requires_review_bool)
        """
        if divergence > self.high_threshold:
            return "HIGH", True
        elif divergence > self.medium_threshold:
            return "MEDIUM", False
        else:
            return "LOW", False
    
    def compare(self, chronos_result: Dict, ets_result: Dict) -> Dict:
        """
        Point-by-point comparison of Chronos vs ETS forecasts.
        
        Args:
            chronos_result: Chronos forecast dictionary with 'median' key
            ets_result: ETS forecast dictionary with 'median' key
            
        Returns:
            Dictionary with comparison results for each forecast step:
            {
                "step_1": {
                    "chronos": float,
                    "ets": float,
                    "divergence": float,
                    "divergence_pct": float,
                    "flag": str ("HIGH", "MEDIUM", "LOW"),
                    "requires_review": bool
                },
                ...
            }
        """
        chronos_median = chronos_result.get('median', [])
        ets_median = ets_result.get('median', [])
        
        if not chronos_median or not ets_median:
            raise ValueError("Both results must contain 'median' forecasts")
        
        # Handle different lengths by taking minimum
        min_length = min(len(chronos_median), len(ets_median))
        
        comparison = {}
        
        for i in range(min_length):
            chronos_val = chronos_median[i]
            ets_val = ets_median[i]
            
            # Calculate divergence using helper
            divergence = self._calculate_divergence(chronos_val, ets_val)
            divergence_pct = divergence * 100
            
            # Determine flag
            flag, requires_review = self._get_flag(divergence)
            
            comparison[f"step_{i+1}"] = {
                "chronos": chronos_val,
                "ets": ets_val,
                "divergence": divergence,
                "divergence_pct": divergence_pct,
                "flag": flag,
                "requires_review": requires_review
            }
        
        return comparison
    
    def flag_dataframe(self, df: pd.DataFrame, 
                       chronos_col: str = 'chronos_forecast',
                       ets_col: str = 'ets_forecast') -> pd.DataFrame:
        """
        Add uncertainty flags to a DataFrame with forecasts.
        
        Args:
            df: DataFrame with chronos and ets forecast columns
            chronos_col: Name of Chronos forecast column
            ets_col: Name of ETS forecast column
            
        Returns:
            DataFrame with added columns:
                - _uncertainty_flag: 'HIGH', 'MEDIUM', 'LOW'
                - _divergence_pct: Percentage difference
                - _requires_review: Boolean
        """
        if chronos_col not in df.columns or ets_col not in df.columns:
            raise ValueError(f"DataFrame must contain both {chronos_col} and {ets_col} columns")
        
        df = df.copy()
        
        # Calculate divergence for each row
        divergences = []
        flags = []
        requires_reviews = []
        
        for _, row in df.iterrows():
            chronos_val = row[chronos_col]
            ets_val = row[ets_col]
            
            # Calculate divergence using helper
            divergence = self._calculate_divergence(chronos_val, ets_val)
            divergence_pct = divergence * 100
            
            # Determine flag
            flag, requires_review = self._get_flag(divergence)
            
            divergences.append(divergence_pct)
            flags.append(flag)
            requires_reviews.append(requires_review)
        
        # Add columns to DataFrame
        df['_uncertainty_flag'] = flags
        df['_divergence_pct'] = divergences
        df['_requires_review'] = requires_reviews
        
        return df
    
    def get_flagged_points(self, comparison: Dict) -> List[Dict]:
        """
        Extract points that require review from comparison results.
        
        Args:
            comparison: Result from compare() method
            
        Returns:
            List of dictionaries for points requiring review
        """
        flagged = []
        
        for step, details in comparison.items():
            if details.get('requires_review', False):
                flagged.append({
                    'step': step,
                    'chronos': details['chronos'],
                    'ets': details['ets'],
                    'divergence_pct': details['divergence_pct'],
                    'flag': details['flag']
                })
        
        return flagged
    
    def summary_report(self, comparison: Dict) -> Dict:
        """
        Generate summary statistics for a comparison.
        
        Args:
            comparison: Result from compare() method
            
        Returns:
            Dictionary with summary statistics
        """
        total_points = len(comparison)
        high_flags = sum(1 for v in comparison.values() if v['flag'] == 'HIGH')
        medium_flags = sum(1 for v in comparison.values() if v['flag'] == 'MEDIUM')
        low_flags = sum(1 for v in comparison.values() if v['flag'] == 'LOW')
        requires_review = sum(1 for v in comparison.values() if v['requires_review'])
        
        avg_divergence = np.mean([v['divergence_pct'] for v in comparison.values()])
        max_divergence = max([v['divergence_pct'] for v in comparison.values()])
        
        return {
            'total_points': total_points,
            'high_uncertainty': high_flags,
            'medium_uncertainty': medium_flags,
            'low_uncertainty': low_flags,
            'requires_review': requires_review,
            'avg_divergence_pct': avg_divergence,
            'max_divergence_pct': max_divergence
        }
