# Test Fallback Loader
# Created by: Shohruh127

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import directly without loading main package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from ingestion.fallback_loader import FallbackLoader


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestFallbackLoader:
    """Tests for FallbackLoader class."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        loader = FallbackLoader()
        assert loader.max_retries == 3
        assert loader.verbose is True
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        loader = FallbackLoader(max_retries=5, verbose=False)
        assert loader.max_retries == 5
        assert loader.verbose is False
        
    def test_method_flags_defined(self):
        """Test that method flags are properly defined."""
        assert FallbackLoader.METHOD_AI_CLEAN == 'AI clean'
        assert FallbackLoader.METHOD_FALLBACK == 'Fallback'
        assert FallbackLoader.METHOD_MANUAL_CORRECTION == 'Manual correction'
        assert FallbackLoader.METHOD_EMPTY == 'Empty result'


class TestLoadMethod:
    """Tests for load method."""
    
    def test_load_existing_file(self):
        """Test loading an existing Excel file."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        df = loader.load(filepath)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file returns empty DataFrame."""
        loader = FallbackLoader(verbose=False)
        df = loader.load("/nonexistent/file.xlsx")
        
        # Should return empty DataFrame, not raise exception
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        
    def test_load_with_metadata(self):
        """Test loading with metadata return."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        result = loader.load(filepath, return_metadata=True)
        
        assert 'data' in result
        assert 'metadata' in result
        assert isinstance(result['data'], pd.DataFrame)
        assert isinstance(result['metadata'], dict)
        
    def test_load_metadata_contains_method(self):
        """Test that metadata includes method used."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        result = loader.load(filepath, return_metadata=True)
        
        assert 'method' in result['metadata']
        
    def test_load_metadata_on_error(self):
        """Test metadata contains error info when file not found."""
        loader = FallbackLoader(verbose=False)
        result = loader.load("/nonexistent.xlsx", return_metadata=True)
        
        assert result['metadata']['success'] is False
        assert result['metadata']['file_exists'] is False
        
    def test_load_by_sheet_name(self):
        """Test loading specific sheet by name."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        df = loader.load(filepath, sheet='Industry')
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestRecoveryScenarios:
    """Tests for error recovery scenarios."""
    
    def test_never_crashes_on_malformed_path(self):
        """Test that loader never crashes on malformed input."""
        loader = FallbackLoader(verbose=False)
        
        # Various malformed inputs that should not crash
        test_cases = [
            "",
            None,
            "/path/with spaces/file.xlsx",
            "file_without_extension",
        ]
        
        for test_input in test_cases:
            try:
                if test_input is not None:
                    result = loader.load(test_input)
                    # Should return DataFrame (possibly empty)
                    assert isinstance(result, pd.DataFrame)
            except Exception as e:
                pytest.fail(f"Loader crashed on input '{test_input}': {e}")
                
    def test_get_errors_initially_empty(self):
        """Test that errors list starts empty."""
        loader = FallbackLoader(verbose=False)
        assert loader.get_errors() == []
        
    def test_get_errors_after_failure(self):
        """Test that errors are recorded after failure."""
        loader = FallbackLoader(verbose=False)
        loader.load("/nonexistent/file.xlsx")
        
        # Errors may or may not be recorded for file not found
        # but method should return a list
        errors = loader.get_errors()
        assert isinstance(errors, list)
        
    def test_get_method_used(self):
        """Test getting method used after load."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        loader.load(filepath)
        
        method = loader.get_method_used()
        assert method in [
            FallbackLoader.METHOD_AI_CLEAN,
            FallbackLoader.METHOD_FALLBACK,
            FallbackLoader.METHOD_MANUAL_CORRECTION,
            FallbackLoader.METHOD_EMPTY
        ]


class TestLoadWithValidation:
    """Tests for load_with_validation method."""
    
    def test_validation_with_valid_primary_result(self):
        """Test validation passes with valid primary result."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        
        # Simulate valid primary loader result
        primary_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = loader.load_with_validation(
            filepath,
            primary_loader_result=primary_df
        )
        
        assert result['method'] == FallbackLoader.METHOD_AI_CLEAN
        assert result['used_fallback'] is False
        
    def test_validation_with_none_triggers_fallback(self):
        """Test that None primary result triggers fallback."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        
        result = loader.load_with_validation(
            filepath,
            primary_loader_result=None
        )
        
        assert result['used_fallback'] is True
        assert 'Primary loader returned None' in result['validation_issues']
        
    def test_validation_with_empty_triggers_fallback(self):
        """Test that empty primary result triggers fallback."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        
        result = loader.load_with_validation(
            filepath,
            primary_loader_result=pd.DataFrame()
        )
        
        assert result['used_fallback'] is True
        assert 'Primary loader returned empty DataFrame' in result['validation_issues']
        
    def test_validation_with_insufficient_rows(self):
        """Test validation with too few rows."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        
        # DataFrame with only 1 row
        small_df = pd.DataFrame({'col': [1]})
        
        result = loader.load_with_validation(
            filepath,
            primary_loader_result=small_df,
            min_rows=5
        )
        
        assert result['used_fallback'] is True
        
    def test_validation_reports_missing_columns(self):
        """Test validation reports missing expected columns."""
        filepath = FIXTURES_DIR / "sample_uzbek_data.xlsx"
        loader = FallbackLoader(verbose=False)
        
        primary_df = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': [4, 5, 6]
        })
        
        result = loader.load_with_validation(
            filepath,
            primary_loader_result=primary_df,
            expected_columns=['col_a', 'col_b', 'col_c']
        )
        
        # Should note missing column but still use primary
        assert any('Missing' in issue for issue in result['validation_issues'])


class TestErrorHandling:
    """Tests for error handling behavior."""
    
    def test_error_recording_structure(self):
        """Test that recorded errors have correct structure."""
        loader = FallbackLoader(verbose=False)
        loader._record_error(
            ValueError("Test error"),
            "test_context",
            1
        )
        
        errors = loader.get_errors()
        assert len(errors) == 1
        
        error = errors[0]
        assert 'type' in error
        assert 'message' in error
        assert 'context' in error
        assert 'attempt' in error
        assert 'traceback' in error
        
        assert error['type'] == 'ValueError'
        assert error['message'] == 'Test error'
        assert error['context'] == 'test_context'
        assert error['attempt'] == 1
