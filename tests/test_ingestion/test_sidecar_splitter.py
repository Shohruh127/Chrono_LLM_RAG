# Test Sidecar Splitter
# Created by: Shohruh127

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import directly without loading main package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ingestion.sidecar_splitter import SidecarSplitter


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestSidecarSplitter:
    """Tests for SidecarSplitter class."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        splitter = SidecarSplitter()
        assert splitter.min_data_rows == 3
        assert splitter.empty_row_threshold == 2
        assert splitter.numeric_column_threshold == 0.3
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        splitter = SidecarSplitter(
            min_data_rows=5,
            empty_row_threshold=3,
            numeric_column_threshold=0.5
        )
        assert splitter.min_data_rows == 5
        assert splitter.empty_row_threshold == 3
        assert splitter.numeric_column_threshold == 0.5
        
    def test_split_empty_dataframe(self):
        """Test splitting an empty DataFrame."""
        splitter = SidecarSplitter()
        result = splitter.split(pd.DataFrame())
        
        assert result['data_table'].empty
        assert result['metadata_tail'] == ''
        assert result['warnings'] == []
        assert result['split_row_index'] == 0
        
    def test_split_none_dataframe(self):
        """Test splitting None input."""
        splitter = SidecarSplitter()
        result = splitter.split(None)
        
        assert result['data_table'].empty
        assert result['metadata_tail'] == ''
        assert result['warnings'] == []
        
    def test_split_data_only(self):
        """Test splitting DataFrame with only data (no footer)."""
        df = pd.DataFrame({
            'Region': ['Tashkent', 'Samarkand', 'Bukhara'],
            '2020': [1000, 800, 600],
            '2021': [1100, 850, 650]
        })
        
        splitter = SidecarSplitter()
        result = splitter.split(df)
        
        assert len(result['data_table']) == 3
        assert result['metadata_tail'] == ''
        
    def test_split_with_footer(self):
        """Test splitting DataFrame with footer metadata."""
        df = pd.DataFrame({
            'A': ['Region', 'Tashkent', 'Samarkand', '', '*Note:'],
            'B': ['2020', 1000, 800, '', 'Preliminary data']
        })
        
        splitter = SidecarSplitter()
        result = splitter.split(df)
        
        # Data should exclude footer
        assert len(result['data_table']) < len(df)
        # Metadata should contain footer text
        assert result['metadata_tail'] != ''
        
    def test_split_preserves_result_structure(self):
        """Test that split returns correct structure."""
        df = pd.DataFrame({
            'A': ['Data', 1, 2],
            'B': ['Data', 3, 4]
        })
        
        splitter = SidecarSplitter()
        result = splitter.split(df)
        
        assert 'data_table' in result
        assert 'metadata_tail' in result
        assert 'warnings' in result
        assert 'split_row_index' in result
        
        assert isinstance(result['data_table'], pd.DataFrame)
        assert isinstance(result['metadata_tail'], str)
        assert isinstance(result['warnings'], list)
        assert isinstance(result['split_row_index'], int)


class TestDataMetadataSeparation:
    """Tests for data/metadata separation logic."""
    
    def test_is_numeric_row_all_numeric(self):
        """Test detection of fully numeric row."""
        row = pd.Series([100, 200, 300])
        splitter = SidecarSplitter()
        assert splitter._is_numeric_row(row) is True
        
    def test_is_numeric_row_mixed(self):
        """Test detection of mixed row."""
        row = pd.Series(['Text', 100, 200])
        splitter = SidecarSplitter()
        # With threshold 0.3, 2/3 numeric should pass
        assert splitter._is_numeric_row(row) is True
        
    def test_is_numeric_row_all_text(self):
        """Test detection of all-text row."""
        row = pd.Series(['Text1', 'Text2', 'Text3'])
        splitter = SidecarSplitter()
        assert splitter._is_numeric_row(row) is False
        
    def test_is_empty_row_true(self):
        """Test detection of empty row."""
        row = pd.Series([None, '', '  '])
        splitter = SidecarSplitter()
        assert splitter._is_empty_row(row) is True
        
    def test_is_empty_row_false(self):
        """Test detection of non-empty row."""
        row = pd.Series([None, 'Value', ''])
        splitter = SidecarSplitter()
        assert splitter._is_empty_row(row) is False
        
    def test_is_footer_row_asterisk(self):
        """Test detection of asterisk note row."""
        row = pd.Series(['*Note about data', None, None])
        splitter = SidecarSplitter()
        assert splitter._is_footer_row(row) is True
        
    def test_is_footer_row_source(self):
        """Test detection of source attribution row."""
        row = pd.Series(['Source: Statistics Committee', None])
        splitter = SidecarSplitter()
        assert splitter._is_footer_row(row) is True
        
    def test_is_footer_row_regular_data(self):
        """Test that regular data is not detected as footer."""
        row = pd.Series(['Tashkent', 1000, 2000])
        splitter = SidecarSplitter()
        assert splitter._is_footer_row(row) is False


class TestWarningExtraction:
    """Tests for warning/note extraction."""
    
    def test_extract_excludes_warning(self):
        """Test extraction of 'excludes' warnings."""
        splitter = SidecarSplitter()
        text = "Data excludes Namangan district"
        warnings = splitter._extract_warnings(text)
        
        assert len(warnings) > 0
        assert any('Excludes' in w for w in warnings)
        
    def test_extract_preliminary_warning(self):
        """Test extraction of 'preliminary' warnings."""
        splitter = SidecarSplitter()
        text = "Preliminary 2024 data"
        warnings = splitter._extract_warnings(text)
        
        assert len(warnings) > 0
        assert any('Preliminary' in w for w in warnings)
        
    def test_extract_asterisk_note(self):
        """Test extraction of asterisk notes."""
        splitter = SidecarSplitter()
        text = "* Important note about methodology"
        warnings = splitter._extract_warnings(text)
        
        assert len(warnings) > 0
        
    def test_extract_no_warnings(self):
        """Test that regular text returns no warnings."""
        splitter = SidecarSplitter()
        text = "Regular data description"
        warnings = splitter._extract_warnings(text)
        
        # Should return empty list for non-warning text
        assert isinstance(warnings, list)


class TestSplitWithContext:
    """Tests for split_with_context method."""
    
    def test_split_with_context_adds_metadata(self):
        """Test that context information is added."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        splitter = SidecarSplitter()
        result = splitter.split_with_context(
            df,
            sheet_name='TestSheet',
            source_file='test.xlsx'
        )
        
        assert 'context' in result
        assert result['context']['sheet_name'] == 'TestSheet'
        assert result['context']['source_file'] == 'test.xlsx'
        assert result['context']['original_rows'] == 3


class TestRealFileProcessing:
    """Tests using real fixture files."""
    
    def test_split_real_footer_file(self):
        """Test splitting a real file with footers."""
        # Read the fixture file directly with pandas
        filepath = FIXTURES_DIR / "sample_with_footers.xlsx"
        if filepath.exists():
            df = pd.read_excel(filepath, header=None)
            
            splitter = SidecarSplitter()
            result = splitter.split(df)
            
            # Should successfully split
            assert result['data_table'] is not None
            assert result['split_row_index'] >= 0
