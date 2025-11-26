# Test Excel Parser
# Created by: Shohruh127

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import directly without loading main package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ingestion.excel_parser import ExcelParser


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestExcelParser:
    """Tests for ExcelParser class."""
    
    def test_init_without_file(self):
        """Test initialization without a file path."""
        parser = ExcelParser()
        assert parser.filepath is None
        
    def test_init_with_file(self):
        """Test initialization with a file path."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        assert parser.filepath == filepath
        
    def test_set_file(self):
        """Test setting file after initialization."""
        parser = ExcelParser()
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser.set_file(filepath)
        assert parser.filepath == filepath
        
    def test_list_sheets(self):
        """Test listing sheet names."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        sheets = parser.list_sheets()
        assert isinstance(sheets, list)
        assert 'Demography' in sheets
        assert 'Industry' in sheets
        
    def test_list_sheets_no_file_raises(self):
        """Test that list_sheets raises error without file."""
        parser = ExcelParser()
        with pytest.raises(ValueError):
            parser.list_sheets()
            
    def test_list_sheets_file_not_found_raises(self):
        """Test that list_sheets raises error for missing file."""
        parser = ExcelParser("/nonexistent/file.xlsx")
        with pytest.raises(FileNotFoundError):
            parser.list_sheets()
            
    def test_parse_sheet_by_name(self):
        """Test parsing a sheet by name."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        df = parser.parse_sheet("Demography", detect_header=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
    def test_parse_sheet_by_index(self):
        """Test parsing a sheet by index."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        df = parser.parse_sheet(0, detect_header=True)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
    def test_header_detection(self):
        """Test that header row is correctly detected."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        df = parser.parse_sheet("Industry", detect_header=True)
        
        # The header should be detected correctly
        assert isinstance(df, pd.DataFrame)
        metadata = parser.get_sheet_metadata("Industry")
        assert 'header_row' in metadata
        
    def test_get_sheet_metadata(self):
        """Test getting sheet metadata."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        parser.parse_sheet("Demography")
        metadata = parser.get_sheet_metadata("Demography")
        
        assert 'row_count' in metadata
        assert 'column_count' in metadata
        assert 'detected_encoding' in metadata
        
    def test_parse_all_sheets(self):
        """Test parsing all sheets at once."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        parser = ExcelParser(filepath)
        result = parser.parse_all_sheets()
        
        assert isinstance(result, dict)
        assert 'Demography' in result
        assert 'Industry' in result
        assert isinstance(result['Demography'], pd.DataFrame)
        
    def test_forward_fill_merged_cells(self):
        """Test forward filling of merged cells."""
        # Create a DataFrame with NaN gaps (simulating merged cells)
        df = pd.DataFrame({
            'A': ['Header', None, None],
            'B': [1, 2, 3]
        })
        
        parser = ExcelParser()
        filled_df = parser._forward_fill_merged_cells(df)
        
        # Should forward fill the NaN values
        assert filled_df.iloc[1, 0] == 'Header'
        assert filled_df.iloc[2, 0] == 'Header'


class TestEncodingDetection:
    """Tests for encoding detection functionality."""
    
    def test_encoding_fallback_order(self):
        """Test that encoding fallback order is defined."""
        assert ExcelParser.ENCODING_FALLBACK_ORDER == [
            'utf-8', 'cp1251', 'windows-1252', 'iso-8859-1'
        ]
        
    def test_detect_encoding_with_bytes(self):
        """Test encoding detection with byte data."""
        parser = ExcelParser()
        # UTF-8 encoded text
        utf8_bytes = "Hello World".encode('utf-8')
        encoding = parser._detect_encoding(utf8_bytes)
        assert encoding is not None
        
    def test_detect_encoding_cyrillic(self):
        """Test encoding detection with Cyrillic text."""
        parser = ExcelParser()
        # CP1251 encoded Cyrillic text
        cyrillic_text = "Привет мир"
        cp1251_bytes = cyrillic_text.encode('cp1251')
        encoding = parser._detect_encoding(cp1251_bytes)
        assert encoding is not None


class TestHeaderDetection:
    """Tests for heuristic header detection."""
    
    def test_find_header_row_simple(self):
        """Test finding header row in simple DataFrame."""
        df = pd.DataFrame([
            ['Title', None, None],
            ['Name', 'Value', 'Date'],
            ['Item1', 100, '2020-01-01'],
            ['Item2', 200, '2020-01-02']
        ])
        
        parser = ExcelParser()
        header_row = parser._find_header_row(df)
        
        # Header should be row 1 (with 'Name', 'Value', 'Date')
        assert header_row == 1
        
    def test_find_header_row_with_empty_rows(self):
        """Test finding header with empty rows at top."""
        df = pd.DataFrame([
            [None, None, None],
            [None, None, None],
            ['Col1', 'Col2', 'Col3'],
            [1, 2, 3]
        ])
        
        parser = ExcelParser()
        header_row = parser._find_header_row(df)
        
        # Header should be row 2
        assert header_row == 2
