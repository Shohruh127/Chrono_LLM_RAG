# =============================================================================
# src/selector/context_propagator.py - Context Propagation for Domain-Specific Analysis
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Phase 4: Selector Architecture
# =============================================================================

import pandas as pd
from typing import Optional
from datetime import datetime


class ContextPropagator:
    """Propagates selected context across modules for domain-specific analysis."""

    def __init__(self):
        """Initialize ContextPropagator with empty context."""
        self._current_context = None

    def set_context(self, sheet_name: str, df: pd.DataFrame, domain: str):
        """
        Set the current analysis context.

        Args:
            sheet_name: Name of the selected sheet
            df: DataFrame containing the sheet data
            domain: Detected domain name (e.g., "Agriculture", "Industry")
        """
        self._current_context = {
            "sheet_name": sheet_name,
            "dataframe": df.copy(),
            "domain": domain,
            "set_at": datetime.utcnow().isoformat(),
            "rows": len(df),
            "cols": len(df.columns),
            "columns": list(df.columns),
        }

    def get_context(self) -> dict:
        """
        Get current context for other modules.

        Returns:
            Dictionary with context information (without DataFrame):
            {
                "sheet_name": "7-Agriculture",
                "domain": "Agriculture",
                "set_at": "2025-11-26T16:30:00",
                "rows": 150,
                "cols": 12,
                "columns": [...]
            }
            Returns None if no context is set
        """
        if self._current_context is None:
            return None

        # Return context without the actual DataFrame (for lightweight access)
        context = self._current_context.copy()
        context.pop("dataframe", None)
        return context

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded DataFrame.

        Returns:
            DataFrame if context is set, None otherwise
        """
        if self._current_context is None:
            return None

        return self._current_context["dataframe"].copy()

    def get_domain_prompt(self) -> str:
        """
        Generate domain-specific prompt for LLM agents.

        Returns:
            Formatted prompt string for the current domain

        Example:
            "You are analyzing Agriculture data for Uzbekistan...
             Focus on agricultural metrics, crop yields, and farming indicators.
             Available columns: ['Region', 'Crop_Yield_2020', 'Crop_Yield_2021', ...]"
        """
        if self._current_context is None:
            return "No data context is currently set. Please select a sheet first."

        domain = self._current_context["domain"]
        sheet_name = self._current_context["sheet_name"]
        columns = self._current_context["columns"]

        # Domain-specific focus areas
        domain_focus = {
            "Agriculture": (
                "agricultural metrics, crop yields, farming indicators, "
                "livestock production, and rural development"
            ),
            "Industry": "industrial output, manufacturing metrics, production capacity, and sectoral performance",
            "Demography": "population statistics, demographic trends, age distribution, and social indicators",
            "Trade": "trade volumes, import/export data, commercial activities, and market dynamics",
            "Transport": (
                "transportation metrics, logistics data, infrastructure utilization, " "and mobility patterns"
            ),
            "Construction": (
                "construction output, building permits, real estate development, " "and infrastructure projects"
            ),
            "Finance": "financial indicators, banking metrics, investment data, and economic flows",
            "Education": "educational statistics, enrollment rates, academic performance, and institutional metrics",
            "Healthcare": "health indicators, medical services data, patient statistics, and public health metrics",
            "Culture": "cultural activities, institutional performance, and creative sector indicators",
            "Sports": "sports activities, athletic performance, and recreational metrics",
            "Tourism": "tourism statistics, visitor data, hospitality metrics, and travel indicators",
            "Unknown": "general economic and statistical indicators",
        }

        focus = domain_focus.get(domain, domain_focus["Unknown"])

        prompt = f"""You are analyzing {domain} data for Uzbekistan economic regions.
Sheet: {sheet_name}

Focus on {focus}.

Available data columns: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}

Key guidelines:
- Base your analysis strictly on the provided data
- Consider regional variations across Uzbekistan
- Highlight trends, patterns, and significant changes
- Provide insights relevant to {domain.lower()} sector planning
- When discussing specific values, always cite the data source"""

        return prompt

    def clear_context(self):
        """Reset context when user selects new sheet."""
        self._current_context = None

    def has_context(self) -> bool:
        """
        Check if context is currently set.

        Returns:
            True if context exists, False otherwise
        """
        return self._current_context is not None

    def get_domain(self) -> Optional[str]:
        """
        Get the current domain.

        Returns:
            Domain name if context is set, None otherwise
        """
        if self._current_context is None:
            return None
        return self._current_context["domain"]

    def get_sheet_name(self) -> Optional[str]:
        """
        Get the current sheet name.

        Returns:
            Sheet name if context is set, None otherwise
        """
        if self._current_context is None:
            return None
        return self._current_context["sheet_name"]

    def __repr__(self) -> str:
        if self._current_context is None:
            return "ContextPropagator(context=None)"
        return f"ContextPropagator(domain='{self.get_domain()}', sheet='{self.get_sheet_name()}')"
