# =============================================================================
# src/selector/__init__.py - Selector Module Exports
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Phase 4: Selector Architecture
# =============================================================================

"""
Selector Module - Domain-Specific Sheet Selection for Multi-Sheet Excel Files

This module provides functionality to:
- List and preview sheets from Excel files
- Detect domains from sheet names (Uzbek/English)
- Propagate context to downstream modules
- Create Gradio UI components for sheet selection

Usage:
    from src.selector import SheetManager, ContextPropagator, create_selector

    manager, context = create_selector("data.xlsx")
    sheets = manager.list_sheets()
    df = manager.select_sheet("7-Agriculture")
    context.set_context("7-Agriculture", df, "Agriculture")
"""

from .sheet_manager import SheetManager
from .context_propagator import ContextPropagator
from .ui_components import (
    create_sheet_dropdown,
    create_sheet_preview,
    create_domain_badge,
    create_selector_interface,
    create_compact_selector,
)


def create_selector(filepath: str):
    """
    Initialize selector components for a file.

    Args:
        filepath: Path to Excel file

    Returns:
        Tuple of (SheetManager, ContextPropagator)

    Example:
        manager, context = create_selector("Namangan_Macro_2024.xlsx")
        sheets = manager.list_sheets()
        df = manager.select_sheet("7-Agriculture")
        context.set_context("7-Agriculture", df, "Agriculture")
    """
    manager = SheetManager(filepath)
    context = ContextPropagator()
    return manager, context


__all__ = [
    "SheetManager",
    "ContextPropagator",
    "create_selector",
    "create_sheet_dropdown",
    "create_sheet_preview",
    "create_domain_badge",
    "create_selector_interface",
    "create_compact_selector",
]
