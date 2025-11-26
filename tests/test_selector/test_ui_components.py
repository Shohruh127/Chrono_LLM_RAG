# =============================================================================
# tests/test_selector/test_ui_components.py - Tests for UI Components
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pytest
import pandas as pd
from src.selector.sheet_manager import SheetManager
from src.selector.context_propagator import ContextPropagator
from src.selector.ui_components import create_sheet_preview, create_domain_badge
from src.selector import create_selector


@pytest.fixture
def test_excel_file(tmp_path):
    """Create a test Excel file"""
    filepath = tmp_path / "test_ui.xlsx"

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df1 = pd.DataFrame({"Region": ["A", "B"], "2020": [100, 200], "2021": [110, 210]})
        df1.to_excel(writer, sheet_name="1-Agriculture", index=False)

        df2 = pd.DataFrame({"Region": ["C", "D"], "2020": [300, 400], "2021": [310, 410]})
        df2.to_excel(writer, sheet_name="2-Industry", index=False)

    return filepath


def test_create_sheet_preview(test_excel_file):
    """Test creating sheet preview"""
    manager = SheetManager(test_excel_file)

    preview = create_sheet_preview(manager, "1-Agriculture", max_rows=1)

    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 1
    assert "Region" in preview.columns


def test_create_sheet_preview_invalid_sheet(test_excel_file):
    """Test preview with invalid sheet"""
    manager = SheetManager(test_excel_file)

    preview = create_sheet_preview(manager, "Invalid", max_rows=5)

    # Should return error DataFrame
    assert isinstance(preview, pd.DataFrame)
    assert "Error" in preview.columns or len(preview) == 0


def test_create_domain_badge_no_context():
    """Test domain badge without context"""
    context = ContextPropagator()

    badge = create_domain_badge(context)

    assert isinstance(badge, str)
    assert "No domain selected" in badge


def test_create_domain_badge_with_context():
    """Test domain badge with context"""
    context = ContextPropagator()
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    context.set_context("7-Agriculture", df, "Agriculture")

    badge = create_domain_badge(context)

    assert isinstance(badge, str)
    assert "Agriculture" in badge
    assert "7-Agriculture" in badge
    assert "2 rows" in badge


def test_create_domain_badge_different_domains():
    """Test domain badge for different domains"""
    domains = ["Agriculture", "Industry", "Demography", "Trade", "Unknown"]

    for domain in domains:
        context = ContextPropagator()
        df = pd.DataFrame({"A": [1]})
        context.set_context(f"Sheet-{domain}", df, domain)

        badge = create_domain_badge(context)

        assert domain in badge
        assert f"Sheet-{domain}" in badge


def test_create_selector(test_excel_file):
    """Test create_selector convenience function"""
    manager, context = create_selector(str(test_excel_file))

    assert isinstance(manager, SheetManager)
    assert isinstance(context, ContextPropagator)
    assert not context.has_context()


def test_create_selector_workflow(test_excel_file):
    """Test complete selector workflow"""
    # Initialize
    manager, context = create_selector(str(test_excel_file))

    # List sheets
    sheets = manager.list_sheets()
    assert len(sheets) == 2

    # Select sheet
    df = manager.select_sheet("1-Agriculture")
    assert len(df) == 2

    # Set context
    domain = manager.detect_domain("1-Agriculture")
    context.set_context("1-Agriculture", df, domain)

    # Verify context
    assert context.has_context()
    assert context.get_domain() == "Agriculture"

    # Get prompt
    prompt = context.get_domain_prompt()
    assert "Agriculture" in prompt


def test_domain_badge_html_structure():
    """Test that domain badge returns valid HTML"""
    context = ContextPropagator()
    df = pd.DataFrame({"A": [1, 2, 3]})
    context.set_context("Test-Sheet", df, "Agriculture")

    badge = create_domain_badge(context)

    # Check for HTML tags
    assert "<div" in badge
    assert "</div>" in badge
    assert "style=" in badge


def test_sheet_preview_max_rows(test_excel_file):
    """Test that preview respects row limit"""
    manager = SheetManager(test_excel_file)

    # File has 2 rows, request 10
    preview = create_sheet_preview(manager, "1-Agriculture", max_rows=10)

    # Should only return available rows
    assert len(preview) <= 10
