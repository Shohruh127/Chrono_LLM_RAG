# =============================================================================
# src/forecasting/__init__.py - Forecasting Module Exports
# Created by: Shohruh127
# Phase 3: Challenger Protocol - Module Interface
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

from .chronos_engine import ChronosEngine
from .ets_challenger import ETSChallenger
from .uncertainty_flagger import UncertaintyFlagger

import pandas as pd
from typing import Dict


def forecast_with_validation(
    series: pd.Series,
    horizon: int = 2,
    uncertainty_threshold: float = 0.20
) -> Dict:
    """
    Run dual-model forecast with automatic uncertainty flagging.
    
    This is the main convenience function for the Challenger Protocol.
    It runs both Chronos-2 (AI) and ETS (statistical baseline) forecasts,
    then compares them to identify high-uncertainty predictions.
    
    Args:
        series: Historical time series data
        horizon: Number of steps to forecast ahead (default: 2)
        uncertainty_threshold: Threshold for HIGH uncertainty flag (default: 0.20)
        
    Returns:
        Dictionary containing:
            - chronos_forecast: Chronos prediction results
            - ets_forecast: ETS prediction results
            - comparison: Point-by-point comparison with flags
            - summary: Summary statistics
            - requires_review: Boolean - True if any point needs review
            
    Example:
        >>> series = pd.Series([100, 105, 110, 108, 115, 120, 125, 130, 128])
        >>> result = forecast_with_validation(series, horizon=2)
        >>> print(result['comparison'])
        >>> if result['requires_review']:
        >>>     print("⚠️ High uncertainty detected - analyst review required")
    """
    # Initialize models
    chronos = ChronosEngine()
    ets = ETSChallenger()
    flagger = UncertaintyFlagger(high_threshold=uncertainty_threshold)
    
    # Run dual forecasts
    chronos_forecast = chronos.forecast(series, horizon)
    ets_forecast = ets.forecast(series, horizon)
    
    # Compare and flag
    comparison = flagger.compare(chronos_forecast, ets_forecast)
    summary = flagger.summary_report(comparison)
    
    # Check if any point requires review
    requires_review = any(v['requires_review'] for v in comparison.values())
    
    return {
        'chronos_forecast': chronos_forecast,
        'ets_forecast': ets_forecast,
        'comparison': comparison,
        'summary': summary,
        'requires_review': requires_review
    }


__all__ = [
    'ChronosEngine',
    'ETSChallenger',
    'UncertaintyFlagger',
    'forecast_with_validation'
]
