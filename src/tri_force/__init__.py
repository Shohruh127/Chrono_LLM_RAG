# =============================================================================
# src/tri_force/__init__.py - Tri-Force Module Exports
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

"""
Tri-Force Model Stack: High-performance inference with three specialist models.

This module provides hot-path inference capabilities for:
1. Forecaster (Chronos): Zero-shot time series forecasting
2. Logic Engineer (Qwen Coder): Python code generation
3. Cultural Analyst (Llama-Uz): Uzbek linguistic analysis

Usage:
    from src.tri_force import TriForceStack, HardwareOptimizer
    
    stack = TriForceStack(config_path="configs/models_config.yaml")
    stack.load_all()  # Load all models for hot-path inference
    
    result = stack.route_query("Forecast GDP for next year")
"""

__version__ = "1.0.0"
__author__ = "Shohruh127"

from .hardware_optimizer import (
    HardwareOptimizer,
    VRAMReport,
    check_gpu,
    get_device
)

from .model_stack import (
    TriForceStack,
    QueryType
)

__all__ = [
    # Hardware optimization
    "HardwareOptimizer",
    "VRAMReport",
    "check_gpu",
    "get_device",
    # Model stack
    "TriForceStack",
    "QueryType",
]
