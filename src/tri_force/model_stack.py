# =============================================================================
# src/tri_force/model_stack.py - Tri-Force Model Stack Manager
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

"""
Tri-Force Model Stack: Hot-path inference with three specialist models.

Models:
1. Forecaster (amazon/chronos-t5-base): Zero-shot time series forecasting
2. Logic Engineer (Qwen/Qwen2.5-Coder-7B): Python code generation and logic
3. Cultural Analyst (behbudiy/Llama-3.1-8B-Uz): Uzbek linguistic analysis
"""

import torch
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from enum import Enum
import yaml

# Support both relative and absolute imports
try:
    from .hardware_optimizer import HardwareOptimizer, VRAMReport
except ImportError:
    from hardware_optimizer import HardwareOptimizer, VRAMReport


class QueryType(Enum):
    """Types of queries for model routing."""
    FORECAST = "forecast"
    CODE = "code"
    CULTURAL = "cultural"
    UNKNOWN = "unknown"


class TriForceStack:
    """
    Manages three specialist models for hot-path inference.
    
    The stack loads all models simultaneously to eliminate load/unload latency,
    using NF4 quantization to fit within VRAM constraints.
    
    Models:
    - Forecaster: amazon/chronos-t5-base (time series)
    - Logic Engineer: Qwen/Qwen2.5-Coder-7B (code generation)
    - Cultural Analyst: behbudiy/Llama-3.1-8B-Uz (Uzbek text)
    """
    
    # Keywords for query routing
    FORECAST_KEYWORDS = [
        "forecast", "predict", "bashorat", "prognoz", "trend", "future",
        "projection", "estimate", "time series", "vaqt qatori"
    ]
    
    CODE_KEYWORDS = [
        "code", "python", "calculate", "compute", "function", "algorithm",
        "script", "program", "formula", "hisoblash", "kod"
    ]
    
    CULTURAL_KEYWORDS = [
        "uzbek", "o'zbek", "toshkent", "uzbekistan", "madaniyat", "til",
        "language", "cultural", "translate", "tarjima", "matn"
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Tri-Force stack.
        
        Args:
            config_path: Path to models configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.hardware = HardwareOptimizer(
            vram_budget_gb=self.config.get("vram_budget_gb", 30.0)
        )
        
        # Model instances (lazy loaded)
        self._forecaster = None
        self._forecaster_tokenizer = None
        self._logic_engineer = None
        self._logic_tokenizer = None
        self._cultural_analyst = None
        self._cultural_tokenizer = None
        
        # Loading status
        self._models_loaded = {
            "forecaster": False,
            "logic_engineer": False,
            "cultural_analyst": False
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "vram_budget_gb": 30.0,
            "forecaster": {
                "model_id": "amazon/chronos-t5-base",
                "use_quantization": False,  # Chronos uses different loading
                "max_context_length": 512
            },
            "logic_engineer": {
                "model_id": "Qwen/Qwen2.5-Coder-7B",
                "use_quantization": True,
                "max_new_tokens": 1024,
                "temperature": 0.3,
                "top_p": 0.9
            },
            "cultural_analyst": {
                "model_id": "behbudiy/Llama-3.1-8B-Uz",
                "use_quantization": True,
                "max_new_tokens": 800,
                "temperature": 0.5,
                "top_p": 0.85
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with defaults
                for key in loaded_config:
                    if isinstance(loaded_config[key], dict) and key in default_config:
                        default_config[key].update(loaded_config[key])
                    else:
                        default_config[key] = loaded_config[key]
        
        return default_config
    
    def load_forecaster(self) -> None:
        """Load the Chronos forecasting model."""
        if self._models_loaded["forecaster"]:
            print("✅ Forecaster already loaded")
            return
        
        print("📥 Loading Forecaster (Chronos-T5-Base)...")
        
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting required. Install with: pip install chronos-forecasting"
            )
        
        model_id = self.config["forecaster"]["model_id"]
        device = self.hardware.device
        
        self._forecaster = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        self._models_loaded["forecaster"] = True
        print(f"✅ Forecaster loaded: {model_id}")
        self.hardware.print_vram_report()
    
    def load_logic_engineer(self) -> None:
        """Load the Qwen Coder model for code generation."""
        if self._models_loaded["logic_engineer"]:
            print("✅ Logic Engineer already loaded")
            return
        
        print("📥 Loading Logic Engineer (Qwen2.5-Coder-7B)...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers required. Install with: pip install transformers")
        
        model_id = self.config["logic_engineer"]["model_id"]
        
        self._logic_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        if self._logic_tokenizer.pad_token is None:
            self._logic_tokenizer.pad_token = self._logic_tokenizer.eos_token
        
        if self.config["logic_engineer"]["use_quantization"] and self.hardware.device == "cuda":
            bnb_config = self.hardware.get_bnb_config()
            self._logic_engineer = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self._logic_engineer = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.hardware.device == "cuda" else torch.float32,
                device_map="auto" if self.hardware.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        self._models_loaded["logic_engineer"] = True
        print(f"✅ Logic Engineer loaded: {model_id}")
        self.hardware.print_vram_report()
    
    def load_cultural_analyst(self) -> None:
        """Load the Uzbek cultural analysis model."""
        if self._models_loaded["cultural_analyst"]:
            print("✅ Cultural Analyst already loaded")
            return
        
        print("📥 Loading Cultural Analyst (Llama-3.1-8B-Uz)...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers required. Install with: pip install transformers")
        
        model_id = self.config["cultural_analyst"]["model_id"]
        
        self._cultural_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        if self._cultural_tokenizer.pad_token is None:
            self._cultural_tokenizer.pad_token = self._cultural_tokenizer.eos_token
        
        if self.config["cultural_analyst"]["use_quantization"] and self.hardware.device == "cuda":
            bnb_config = self.hardware.get_bnb_config()
            self._cultural_analyst = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self._cultural_analyst = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.hardware.device == "cuda" else torch.float32,
                device_map="auto" if self.hardware.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        self._models_loaded["cultural_analyst"] = True
        print(f"✅ Cultural Analyst loaded: {model_id}")
        self.hardware.print_vram_report()
    
    def load_all(self) -> None:
        """Load all three models (hot-path inference setup)."""
        print("\n" + "=" * 60)
        print("🚀 Loading Tri-Force Model Stack (Hot-Path Inference)")
        print("=" * 60 + "\n")
        
        self.load_forecaster()
        self.load_logic_engineer()
        self.load_cultural_analyst()
        
        print("\n" + "=" * 60)
        print("✅ All models loaded successfully!")
        print("=" * 60)
        self.hardware.print_vram_report()
        
        if not self.hardware.check_vram_budget():
            print("⚠️ WARNING: VRAM usage exceeds budget!")
    
    def detect_query_type(self, query: str) -> QueryType:
        """
        Detect the type of query for routing to appropriate model.
        
        Args:
            query: User query string
            
        Returns:
            QueryType enum value
        """
        query_lower = query.lower()
        
        # Check for forecast-related keywords
        for keyword in self.FORECAST_KEYWORDS:
            if keyword in query_lower:
                return QueryType.FORECAST
        
        # Check for code-related keywords
        for keyword in self.CODE_KEYWORDS:
            if keyword in query_lower:
                return QueryType.CODE
        
        # Check for cultural/Uzbek keywords
        for keyword in self.CULTURAL_KEYWORDS:
            if keyword in query_lower:
                return QueryType.CULTURAL
        
        return QueryType.UNKNOWN
    
    def route_query(self, query: str, query_type: Optional[QueryType] = None) -> Dict[str, Any]:
        """
        Route a query to the appropriate model.
        
        Args:
            query: User query string
            query_type: Optional explicit query type (auto-detected if not provided)
            
        Returns:
            Dictionary with model name and response
        """
        if query_type is None:
            query_type = self.detect_query_type(query)
        
        result = {
            "query": query,
            "query_type": query_type.value,
            "model": None,
            "response": None,
            "error": None
        }
        
        try:
            if query_type == QueryType.FORECAST:
                result["model"] = "forecaster"
                result["response"] = self._handle_forecast_query(query)
            elif query_type == QueryType.CODE:
                result["model"] = "logic_engineer"
                result["response"] = self._generate_code(query)
            elif query_type == QueryType.CULTURAL:
                result["model"] = "cultural_analyst"
                result["response"] = self._analyze_cultural(query)
            else:
                # Default to cultural analyst for unknown queries
                result["model"] = "cultural_analyst"
                result["response"] = self._analyze_cultural(query)
                
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _handle_forecast_query(self, query: str) -> str:
        """Handle forecasting queries."""
        if not self._models_loaded["forecaster"]:
            self.load_forecaster()
        
        return (
            "Forecaster ready. To generate forecasts, use the forecast() method "
            "with time series data. Query: " + query
        )
    
    def _generate_code(self, query: str) -> str:
        """Generate code using the Logic Engineer model."""
        if not self._models_loaded["logic_engineer"]:
            self.load_logic_engineer()
        
        config = self.config["logic_engineer"]
        
        prompt = f"<|im_start|>system\nYou are a Python coding assistant. Write clean, efficient code.\n<|im_end|>\n<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self._logic_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.hardware.device)
        
        with torch.no_grad():
            outputs = self._logic_engineer.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", 1024),
                temperature=config.get("temperature", 0.3),
                top_p=config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self._logic_tokenizer.pad_token_id,
                eos_token_id=self._logic_tokenizer.eos_token_id
            )
        
        response = self._logic_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        return response
    
    def _analyze_cultural(self, query: str) -> str:
        """Analyze text using the Cultural Analyst model."""
        if not self._models_loaded["cultural_analyst"]:
            self.load_cultural_analyst()
        
        config = self.config["cultural_analyst"]
        
        prompt = f"<s>[INST] {query} [/INST]"
        
        inputs = self._cultural_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.hardware.device)
        
        with torch.no_grad():
            outputs = self._cultural_analyst.generate(
                **inputs,
                max_new_tokens=config.get("max_new_tokens", 800),
                temperature=config.get("temperature", 0.5),
                top_p=config.get("top_p", 0.85),
                do_sample=True,
                pad_token_id=self._cultural_tokenizer.pad_token_id,
                eos_token_id=self._cultural_tokenizer.eos_token_id
            )
        
        response = self._cultural_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response after instruction
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def forecast(self, context: Union[torch.Tensor, List[float]], 
                 prediction_length: int = 4,
                 num_samples: int = 20) -> torch.Tensor:
        """
        Generate time series forecasts.
        
        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            num_samples: Number of sample paths to generate
            
        Returns:
            Tensor of shape (num_samples, prediction_length)
        """
        if not self._models_loaded["forecaster"]:
            self.load_forecaster()
        
        if isinstance(context, list):
            context = torch.tensor(context)
        
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        forecasts = self._forecaster.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples
        )
        
        return forecasts
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health status of all models.
        
        Returns:
            Dictionary with health status for each model
        """
        status = {
            "hardware": {
                "device": self.hardware.device,
                "device_name": self.hardware.device_name,
                "vram_usage": None,
                "within_budget": None
            },
            "models": {
                "forecaster": {
                    "loaded": self._models_loaded["forecaster"],
                    "model_id": self.config["forecaster"]["model_id"],
                    "ready": self._forecaster is not None
                },
                "logic_engineer": {
                    "loaded": self._models_loaded["logic_engineer"],
                    "model_id": self.config["logic_engineer"]["model_id"],
                    "ready": self._logic_engineer is not None
                },
                "cultural_analyst": {
                    "loaded": self._models_loaded["cultural_analyst"],
                    "model_id": self.config["cultural_analyst"]["model_id"],
                    "ready": self._cultural_analyst is not None
                }
            },
            "all_loaded": all(self._models_loaded.values())
        }
        
        if self.hardware.device == "cuda":
            report = self.hardware.get_vram_usage()
            status["hardware"]["vram_usage"] = {
                "total_gb": report.total_gb,
                "used_gb": report.used_gb,
                "free_gb": report.free_gb,
                "utilization_percent": report.utilization_percent
            }
            status["hardware"]["within_budget"] = self.hardware.check_vram_budget()
        
        return status
    
    def unload_all(self) -> None:
        """Unload all models and clear GPU memory."""
        print("🗑️ Unloading all models...")
        
        self._forecaster = None
        self._forecaster_tokenizer = None
        self._logic_engineer = None
        self._logic_tokenizer = None
        self._cultural_analyst = None
        self._cultural_tokenizer = None
        
        for key in self._models_loaded:
            self._models_loaded[key] = False
        
        self.hardware.clear_cache()
        print("✅ All models unloaded")


if __name__ == "__main__":
    # Quick test
    stack = TriForceStack()
    print(f"Device: {stack.hardware.device}")
    
    # Test query routing
    test_queries = [
        "Forecast GDP for next 4 years",
        "Write Python code to calculate mean",
        "Translate this text to Uzbek",
        "What is the economic situation?"
    ]
    
    for query in test_queries:
        query_type = stack.detect_query_type(query)
        print(f"Query: '{query}' -> Type: {query_type.value}")
