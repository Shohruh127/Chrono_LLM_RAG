# Chrono_LLM_RAG
# Created by: Shohruh127
# Date: 2025-11-19 10:47:49
# Repository ID: 1099678425

__version__ = "1.0.0"
__author__ = "Shohruh127"
__repository__ = "Chrono_LLM_RAG"
__repo_id__ = "1099678425"

# All imports are lazy to avoid heavy dependencies on module load
__all__ = [
    "DataLoader",
    "UzbekXLSXPreprocessor",
    "ChronosForecaster",
    "RAGSystem",
    "LLMAnalyzer",
]


def __getattr__(name):
    """Lazy loading of all modules"""
    if name == "DataLoader":
        from .data_loader import DataLoader

        return DataLoader
    elif name == "UzbekXLSXPreprocessor":
        from .preprocessor import UzbekXLSXPreprocessor

        return UzbekXLSXPreprocessor
    elif name == "ChronosForecaster":
        from .chronos_forecaster import ChronosForecaster

        return ChronosForecaster
    elif name == "RAGSystem":
        from .rag_system import RAGSystem

        return RAGSystem
    elif name == "LLMAnalyzer":
        from .llm_analyzer import LLMAnalyzer

        return LLMAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

