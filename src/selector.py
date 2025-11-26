# =============================================================================
# src/selector.py - Context Propagator (Stub for Phase 5)
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

"""
Context Propagator - Stub implementation for Sovereign Agent integration.
This would manage context and DataFrame routing referenced in Phase 4.
"""

import pandas as pd
from typing import Optional, Dict


class ContextPropagator:
    """
    Stub implementation of context propagator for DataFrame management.
    In a full implementation, this would handle context switching and routing.
    """

    def __init__(self):
        """Initialize context propagator."""
        self.contexts = {}
        self.current_context = None

    def set_context(self, name: str, df: pd.DataFrame, description: str = ""):
        """
        Set a named context with a DataFrame.
        
        Args:
            name: Context name
            df: DataFrame to store
            description: Optional description
        """
        self.contexts[name] = {
            'df': df,
            'description': description,
            'schema': self._extract_schema(df)
        }
        self.current_context = name

    def get_context(self, name: Optional[str] = None) -> Dict:
        """
        Get a named context.
        
        Args:
            name: Context name (if None, returns current context)
            
        Returns:
            Context dictionary with df, description, and schema
        """
        if name is None:
            name = self.current_context
        
        return self.contexts.get(name, None)

    def get_dataframe(self, name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame from context.
        
        Args:
            name: Context name (if None, uses current context)
            
        Returns:
            DataFrame or None
        """
        context = self.get_context(name)
        if context:
            return context['df']
        return None

    def get_schema(self, name: Optional[str] = None) -> Optional[Dict]:
        """
        Get DataFrame schema from context.
        
        Args:
            name: Context name (if None, uses current context)
            
        Returns:
            Schema dictionary or None
        """
        context = self.get_context(name)
        if context:
            return context['schema']
        return None

    def _extract_schema(self, df: pd.DataFrame) -> Dict:
        """
        Extract schema information from DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Schema dictionary with columns, dtypes, shape
        """
        return {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'index': df.index.name
        }
