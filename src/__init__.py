# Chrono_LLM_RAG
# Created by: Shohruh127
# Date: 2025-11-19 10:47:49
# Repository ID: 1099678425

__version__ = "1.0.0"
__author__ = "Shohruh127"
__repository__ = "Chrono_LLM_RAG"
__repo_id__ = "1099678425"


def _lazy_import():
    """Lazy import of heavy dependencies"""
    from .data_loader import DataLoader
    from .preprocessor import UzbekXLSXPreprocessor
    from .chronos_forecaster import ChronosForecaster
    from .rag_system import RAGSystem
    from .llm_analyzer import LLMAnalyzer
    
    return DataLoader, UzbekXLSXPreprocessor, ChronosForecaster, RAGSystem, LLMAnalyzer


# Import lightweight modules directly
from .data_loader import DataLoader
from .preprocessor import UzbekXLSXPreprocessor

# Heavy imports are lazy
__all__ = [
    "DataLoader",
    "UzbekXLSXPreprocessor",
    "ChronosForecaster",
    "RAGSystem",
    "LLMAnalyzer"
]


def __getattr__(name):
    """Lazy loading of heavy modules"""
    if name == "ChronosForecaster":
        from .chronos_forecaster import ChronosForecaster
        return ChronosForecaster
    elif name == "RAGSystem":
        from .rag_system import RAGSystem
        return RAGSystem
    elif name == "LLMAnalyzer":
        from .llm_analyzer import LLMAnalyzer
        return LLMAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

