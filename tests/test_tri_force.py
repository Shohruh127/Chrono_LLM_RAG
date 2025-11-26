# =============================================================================
# tests/test_tri_force.py - Tests for Tri-Force Model Stack
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

"""
Unit tests for the Tri-Force model stack components.
Tests hardware optimizer, model stack, and query routing.

Note: Uses direct imports to avoid issues with src/__init__.py
"""

import pytest
import sys
import os

# Add tri_force module directly to path to avoid src/__init__.py issues
_tri_force_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'tri_force')
if _tri_force_path not in sys.path:
    sys.path.insert(0, _tri_force_path)

import torch
from unittest.mock import Mock, patch, MagicMock
import yaml

# Direct imports from tri_force module
from hardware_optimizer import HardwareOptimizer, VRAMReport, check_gpu, get_device
from model_stack import TriForceStack, QueryType


class TestHardwareOptimizer:
    """Tests for HardwareOptimizer class."""
    
    def test_import(self):
        """Test that HardwareOptimizer can be imported."""
        assert HardwareOptimizer is not None
    
    def test_initialization(self):
        """Test HardwareOptimizer initialization."""
        optimizer = HardwareOptimizer(vram_budget_gb=30.0)
        assert optimizer.vram_budget_gb == 30.0
    
    def test_device_detection(self):
        """Test device detection returns valid device."""
        optimizer = HardwareOptimizer()
        device = optimizer.device
        assert device in ["cuda", "cpu"]
    
    def test_device_name(self):
        """Test device name property."""
        optimizer = HardwareOptimizer()
        name = optimizer.device_name
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_quantization_config(self):
        """Test NF4 quantization config generation."""
        optimizer = HardwareOptimizer()
        config = optimizer.get_quantization_config()
        
        assert "load_in_4bit" in config
        assert config["load_in_4bit"] is True
        assert config["bnb_4bit_quant_type"] == "nf4"
        assert "bnb_4bit_compute_dtype" in config
        assert "bnb_4bit_use_double_quant" in config
    
    def test_vram_report_structure(self):
        """Test VRAM report structure."""
        optimizer = HardwareOptimizer()
        report = optimizer.get_vram_usage()
        
        assert isinstance(report, VRAMReport)
        assert hasattr(report, "total_gb")
        assert hasattr(report, "used_gb")
        assert hasattr(report, "free_gb")
        assert hasattr(report, "utilization_percent")
        assert hasattr(report, "device_name")
    
    def test_vram_budget_check(self):
        """Test VRAM budget checking."""
        optimizer = HardwareOptimizer(vram_budget_gb=100.0)  # High budget
        result = optimizer.check_vram_budget()
        assert isinstance(result, bool)
    
    def test_memory_estimate(self):
        """Test model memory estimation."""
        optimizer = HardwareOptimizer()
        
        # Test NF4 estimate
        nf4_estimate = optimizer.get_model_memory_estimate(7.0, "nf4")
        assert nf4_estimate > 0
        assert nf4_estimate < 10  # Should be less than 10GB for 7B params with NF4
        
        # Test FP16 estimate
        fp16_estimate = optimizer.get_model_memory_estimate(7.0, "fp16")
        assert fp16_estimate > nf4_estimate  # FP16 should use more memory
    
    def test_check_gpu_function(self):
        """Test check_gpu utility function."""
        result = check_gpu()
        assert isinstance(result, bool)
    
    def test_get_device_function(self):
        """Test get_device utility function."""
        device = get_device()
        assert device in ["cuda", "cpu"]


class TestQueryType:
    """Tests for QueryType enum."""
    
    def test_query_type_values(self):
        """Test QueryType enum values."""
        assert QueryType.FORECAST.value == "forecast"
        assert QueryType.CODE.value == "code"
        assert QueryType.CULTURAL.value == "cultural"
        assert QueryType.UNKNOWN.value == "unknown"


class TestTriForceStack:
    """Tests for TriForceStack class."""
    
    def test_import(self):
        """Test that TriForceStack can be imported."""
        assert TriForceStack is not None
    
    def test_initialization_without_config(self):
        """Test TriForceStack initialization without config file."""
        stack = TriForceStack()
        assert stack is not None
        assert stack.config is not None
    
    def test_initialization_with_config(self, tmp_path):
        """Test TriForceStack initialization with config file."""
        # Create a temporary config file
        config = {
            "vram_budget_gb": 25.0,
            "forecaster": {
                "model_id": "amazon/chronos-t5-base"
            }
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        stack = TriForceStack(config_path=str(config_path))
        assert stack.config["vram_budget_gb"] == 25.0
    
    def test_models_not_loaded_initially(self):
        """Test that models are not loaded on initialization."""
        stack = TriForceStack()
        assert stack._models_loaded["forecaster"] is False
        assert stack._models_loaded["logic_engineer"] is False
        assert stack._models_loaded["cultural_analyst"] is False
    
    def test_detect_query_type_forecast(self):
        """Test query type detection for forecasting queries."""
        stack = TriForceStack()
        
        forecast_queries = [
            "Forecast GDP for next year",
            "Predict sales trend",
            "What is the future projection?",
            "Bashorat qiling"
        ]
        
        for query in forecast_queries:
            result = stack.detect_query_type(query)
            assert result == QueryType.FORECAST, f"Failed for query: {query}"
    
    def test_detect_query_type_code(self):
        """Test query type detection for code queries."""
        stack = TriForceStack()
        
        code_queries = [
            "Write Python code to calculate mean",
            "Create a function for sorting",
            "Calculate the formula",
            "Algorithm for search"
        ]
        
        for query in code_queries:
            result = stack.detect_query_type(query)
            assert result == QueryType.CODE, f"Failed for query: {query}"
    
    def test_detect_query_type_cultural(self):
        """Test query type detection for cultural queries."""
        stack = TriForceStack()
        
        cultural_queries = [
            "Translate to Uzbek",
            "O'zbekiston haqida",
            "Toshkent viloyati",
            "Cultural analysis"
        ]
        
        for query in cultural_queries:
            result = stack.detect_query_type(query)
            assert result == QueryType.CULTURAL, f"Failed for query: {query}"
    
    def test_detect_query_type_unknown(self):
        """Test query type detection for unknown queries."""
        stack = TriForceStack()
        
        unknown_queries = [
            "Hello world",
            "What is 2+2?",
            "Random question"
        ]
        
        for query in unknown_queries:
            result = stack.detect_query_type(query)
            assert result == QueryType.UNKNOWN, f"Failed for query: {query}"
    
    def test_health_check_structure(self):
        """Test health check returns proper structure."""
        stack = TriForceStack()
        health = stack.health_check()
        
        assert "hardware" in health
        assert "models" in health
        assert "all_loaded" in health
        
        assert "device" in health["hardware"]
        assert "device_name" in health["hardware"]
        
        assert "forecaster" in health["models"]
        assert "logic_engineer" in health["models"]
        assert "cultural_analyst" in health["models"]
    
    def test_health_check_initial_state(self):
        """Test health check shows correct initial state."""
        stack = TriForceStack()
        health = stack.health_check()
        
        assert health["all_loaded"] is False
        assert health["models"]["forecaster"]["loaded"] is False
        assert health["models"]["logic_engineer"]["loaded"] is False
        assert health["models"]["cultural_analyst"]["loaded"] is False


class TestModuleExports:
    """Tests for module __init__ exports."""
    
    def test_tri_force_module_imports(self):
        """Test that main classes can be imported from module."""
        # All imports were done at module level, verify they exist
        assert HardwareOptimizer is not None
        assert VRAMReport is not None
        assert TriForceStack is not None
        assert QueryType is not None
        assert check_gpu is not None
        assert get_device is not None


class TestConfigFiles:
    """Tests for configuration file loading."""
    
    def test_models_config_exists(self):
        """Test that models_config.yaml exists and is valid."""
        import os
        config_path = "configs/models_config.yaml"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "vram_budget_gb" in config
        assert "forecaster" in config
        assert "logic_engineer" in config
        assert "cultural_analyst" in config
    
    def test_security_config_exists(self):
        """Test that security_config.yaml exists and is valid."""
        import os
        config_path = "configs/security_config.yaml"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "blocked_imports" in config
        assert "allowed_pandas_operations" in config
        assert "allowed_numpy_operations" in config
        assert "execution" in config
    
    def test_blocked_imports_contains_dangerous_modules(self):
        """Test that security config blocks dangerous imports."""
        config_path = "configs/security_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        blocked = config["blocked_imports"]
        
        # Check for critical dangerous modules
        dangerous = ["os", "sys", "subprocess", "socket"]
        for module in dangerous:
            assert module in blocked, f"Dangerous module '{module}' not in blocked imports"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
