# =============================================================================
# tests/test_selector/test_sheet_manager.py - Tests for SheetManager
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pytest
import pandas as pd
from src.selector.sheet_manager import SheetManager


@pytest.fixture
def test_excel_file(tmp_path):
    """Create a test Excel file with multiple sheets"""
    filepath = tmp_path / "test_data.xlsx"

    # Create multiple sheets with different domains
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Agriculture sheet
        df_agr = pd.DataFrame(
            {
                "Region": ["Namangan", "Uchqo'rg'on", "Pop"],
                "2020": [1250, 890, 650],
                "2021": [1300, 920, 680],
                "2022": [1280, 910, 660],
                "2023": [1350, 950, 700],
            }
        )
        df_agr.to_excel(writer, sheet_name="7-Qishloq xo'jaligi", index=False)

        # Industry sheet
        df_ind = pd.DataFrame(
            {
                "Region": ["Namangan", "Uchqo'rg'on"],
                "2020": [2500, 1800],
                "2021": [2600, 1850],
                "2022": [2700, 1900],
                "2023": [2800, 1950],
            }
        )
        df_ind.to_excel(writer, sheet_name="3-Sanoat", index=False)

        # Demography sheet
        df_demo = pd.DataFrame(
            {
                "Region": ["Namangan", "Uchqo'rg'on", "Pop", "Chust"],
                "Population_2020": [150000, 80000, 60000, 90000],
                "Population_2021": [152000, 81000, 61000, 91000],
                "Population_2022": [154000, 82000, 62000, 92000],
            }
        )
        df_demo.to_excel(writer, sheet_name="1-Demografiya", index=False)

        # English named sheet
        df_trade = pd.DataFrame(
            {
                "Region": ["Namangan"],
                "Export_2020": [500],
                "Export_2021": [520],
                "Import_2020": [300],
                "Import_2021": [310],
            }
        )
        df_trade.to_excel(writer, sheet_name="5-Trade", index=False)

    return filepath


def test_sheet_manager_initialization(test_excel_file):
    """Test SheetManager initialization"""
    manager = SheetManager(test_excel_file)
    assert manager.filepath.exists()
    assert manager.filepath.suffix == ".xlsx"


def test_sheet_manager_file_not_found():
    """Test SheetManager with non-existent file"""
    with pytest.raises(FileNotFoundError):
        SheetManager("nonexistent_file.xlsx")


def test_sheet_manager_invalid_format(tmp_path):
    """Test SheetManager with invalid file format"""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("test")

    with pytest.raises(ValueError, match="Invalid file format"):
        SheetManager(invalid_file)


def test_list_sheets(test_excel_file):
    """Test listing sheets with metadata"""
    manager = SheetManager(test_excel_file)
    sheets = manager.list_sheets()

    assert len(sheets) == 4
    assert all("name" in s for s in sheets)
    assert all("rows" in s for s in sheets)
    assert all("cols" in s for s in sheets)
    assert all("domain" in s for s in sheets)

    # Check specific sheets
    sheet_names = [s["name"] for s in sheets]
    assert "7-Qishloq xo'jaligi" in sheet_names
    assert "3-Sanoat" in sheet_names
    assert "1-Demografiya" in sheet_names
    assert "5-Trade" in sheet_names


def test_detect_domain_uzbek(test_excel_file):
    """Test domain detection for Uzbek sheet names"""
    manager = SheetManager(test_excel_file)

    assert manager.detect_domain("7-Qishloq xo'jaligi") == "Agriculture"
    assert manager.detect_domain("3-Sanoat") == "Industry"
    assert manager.detect_domain("1-Demografiya") == "Demography"


def test_detect_domain_english(test_excel_file):
    """Test domain detection for English sheet names"""
    manager = SheetManager(test_excel_file)

    assert manager.detect_domain("5-Trade") == "Trade"
    assert manager.detect_domain("Agriculture") == "Agriculture"
    assert manager.detect_domain("Industry") == "Industry"


def test_detect_domain_unknown(test_excel_file):
    """Test domain detection for unknown domains"""
    manager = SheetManager(test_excel_file)

    assert manager.detect_domain("Unknown Domain") == "Unknown"
    assert manager.detect_domain("Random Sheet") == "Unknown"


def test_get_sheet_preview(test_excel_file):
    """Test getting sheet preview"""
    manager = SheetManager(test_excel_file)

    preview = manager.get_sheet_preview("7-Qishloq xo'jaligi", max_rows=2)
    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 2
    assert "Region" in preview.columns


def test_get_sheet_preview_invalid_sheet(test_excel_file):
    """Test preview with invalid sheet name"""
    manager = SheetManager(test_excel_file)

    with pytest.raises(ValueError, match="Sheet .* not found"):
        manager.get_sheet_preview("Invalid Sheet")


def test_select_sheet(test_excel_file):
    """Test selecting and loading a sheet"""
    manager = SheetManager(test_excel_file)

    df = manager.select_sheet("3-Sanoat")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Region" in df.columns


def test_select_sheet_invalid(test_excel_file):
    """Test selecting invalid sheet"""
    manager = SheetManager(test_excel_file)

    with pytest.raises(ValueError, match="Sheet .* not found"):
        manager.select_sheet("Invalid Sheet")


def test_get_current_context(test_excel_file):
    """Test getting current context"""
    manager = SheetManager(test_excel_file)

    # Before selection
    context = manager.get_current_context()
    assert context["sheet"] is None
    assert context["domain"] is None

    # After selection
    manager.select_sheet("1-Demografiya")
    context = manager.get_current_context()
    assert context["sheet"] == "1-Demografiya"
    assert context["domain"] == "Demography"
    assert context["rows"] == 4
    assert context["cols"] == 4


def test_get_available_domains(test_excel_file):
    """Test getting available domains"""
    manager = SheetManager(test_excel_file)

    domains = manager.get_available_domains()
    assert "Agriculture" in domains
    assert "Industry" in domains
    assert "Demography" in domains
    assert "Trade" in domains


def test_sheet_manager_repr(test_excel_file):
    """Test string representation"""
    manager = SheetManager(test_excel_file)
    repr_str = repr(manager)

    assert "SheetManager" in repr_str
    assert "sheets=4" in repr_str
