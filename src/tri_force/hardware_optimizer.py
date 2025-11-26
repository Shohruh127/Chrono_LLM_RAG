# =============================================================================
# src/tri_force/hardware_optimizer.py - Hardware Optimization for Tri-Force Stack
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

"""
Hardware optimization utilities for Tri-Force model stack.
Provides GPU memory management, NF4 quantization configs, and VRAM monitoring.
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VRAMReport:
    """VRAM usage report."""
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_percent: float
    device_name: str


class HardwareOptimizer:
    """
    Hardware optimization utilities for running Tri-Force model stack.
    
    Provides:
    - Automatic device detection (CUDA/CPU fallback)
    - NF4 quantization configuration for bitsandbytes
    - GPU memory monitoring and management
    - VRAM usage reporting
    """
    
    def __init__(self, vram_budget_gb: float = 30.0):
        """
        Initialize hardware optimizer.
        
        Args:
            vram_budget_gb: Maximum VRAM budget in GB (default 30GB for A100 headroom)
        """
        self.vram_budget_gb = vram_budget_gb
        self._device = None
        self._device_name = None
    
    @property
    def device(self) -> str:
        """Get the compute device (cuda or cpu)."""
        if self._device is None:
            self._device = self._detect_device()
        return self._device
    
    @property
    def device_name(self) -> str:
        """Get the device name."""
        if self._device_name is None:
            if torch.cuda.is_available():
                self._device_name = torch.cuda.get_device_name(0)
            else:
                self._device_name = "CPU"
        return self._device_name
    
    def _detect_device(self) -> str:
        """Detect available compute device."""
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return "cuda"
        else:
            print("⚠️ No GPU detected, using CPU (performance will be limited)")
            return "cpu"
    
    def get_quantization_config(self, compute_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
        """
        Get NF4 quantization configuration for bitsandbytes.
        
        Args:
            compute_dtype: Compute dtype for quantization (default: float16)
            
        Returns:
            Dictionary with quantization configuration
        """
        if compute_dtype is None:
            compute_dtype = torch.float16
        
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": True,
        }
    
    def get_bnb_config(self, compute_dtype: Optional[torch.dtype] = None):
        """
        Get BitsAndBytesConfig for model loading.
        
        Args:
            compute_dtype: Compute dtype for quantization (default: float16)
            
        Returns:
            BitsAndBytesConfig object
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers with bitsandbytes support required. "
                "Install with: pip install transformers bitsandbytes"
            )
        
        if compute_dtype is None:
            compute_dtype = torch.float16
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    def get_vram_usage(self) -> VRAMReport:
        """
        Get current VRAM usage statistics.
        
        Returns:
            VRAMReport with memory statistics
        """
        if not torch.cuda.is_available():
            return VRAMReport(
                total_gb=0.0,
                used_gb=0.0,
                free_gb=0.0,
                utilization_percent=0.0,
                device_name="CPU"
            )
        
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        
        total_gb = total / (1024 ** 3)
        used_gb = reserved / (1024 ** 3)
        free_gb = (total - reserved) / (1024 ** 3)
        utilization = (reserved / total) * 100
        
        return VRAMReport(
            total_gb=total_gb,
            used_gb=used_gb,
            free_gb=free_gb,
            utilization_percent=utilization,
            device_name=self.device_name
        )
    
    def print_vram_report(self) -> None:
        """Print formatted VRAM usage report."""
        report = self.get_vram_usage()
        
        print("\n" + "=" * 50)
        print("📊 VRAM Usage Report")
        print("=" * 50)
        print(f"Device: {report.device_name}")
        print(f"Total:  {report.total_gb:.2f} GB")
        print(f"Used:   {report.used_gb:.2f} GB")
        print(f"Free:   {report.free_gb:.2f} GB")
        print(f"Usage:  {report.utilization_percent:.1f}%")
        
        if report.used_gb > self.vram_budget_gb:
            print(f"⚠️ WARNING: Exceeding VRAM budget of {self.vram_budget_gb} GB!")
        else:
            headroom = self.vram_budget_gb - report.used_gb
            print(f"✅ Within budget ({headroom:.2f} GB headroom)")
        print("=" * 50 + "\n")
    
    def check_vram_budget(self) -> bool:
        """
        Check if current VRAM usage is within budget.
        
        Returns:
            True if within budget, False otherwise
        """
        report = self.get_vram_usage()
        return report.used_gb <= self.vram_budget_gb
    
    def clear_cache(self) -> None:
        """Clear GPU cache to free up memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✅ GPU cache cleared")
    
    def get_model_memory_estimate(self, param_count_billions: float, 
                                   quantization: str = "nf4") -> float:
        """
        Estimate VRAM needed for a model.
        
        Args:
            param_count_billions: Model parameter count in billions
            quantization: Quantization type ("nf4", "int8", "fp16", "fp32")
            
        Returns:
            Estimated VRAM in GB
        """
        bytes_per_param = {
            "nf4": 0.5,      # 4-bit = 0.5 bytes
            "int8": 1.0,     # 8-bit = 1 byte  
            "fp16": 2.0,     # 16-bit = 2 bytes
            "fp32": 4.0,     # 32-bit = 4 bytes
        }
        
        bpp = bytes_per_param.get(quantization, 0.5)
        # Add ~20% overhead for activations/KV cache
        overhead = 1.2
        
        return param_count_billions * bpp * overhead


def check_gpu() -> bool:
    """Quick GPU availability check."""
    optimizer = HardwareOptimizer()
    return optimizer.device == "cuda"


def get_device() -> str:
    """Get the best available device."""
    optimizer = HardwareOptimizer()
    return optimizer.device


if __name__ == "__main__":
    # Quick test
    optimizer = HardwareOptimizer()
    print(f"Device: {optimizer.device}")
    optimizer.print_vram_report()
