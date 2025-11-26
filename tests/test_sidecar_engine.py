# Test Sidecar Engine
# Created by: Shohruh127
# Date: 2025-11-26

import pytest
import pandas as pd
import tempfile
import os
import sys
import importlib.util

# Load module directly without triggering src/__init__.py
spec = importlib.util.spec_from_file_location(
    "sidecar_engine", 
    os.path.join(os.path.dirname(__file__), '..', 'src', 'sidecar_engine.py')
)
sidecar_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidecar_module)
DictionaryIngestionEngine = sidecar_module.DictionaryIngestionEngine


def test_sidecar_engine_single_sheet():
    """Test sidecar engine with single sheet Excel file"""
    engine = DictionaryIngestionEngine()
    
    # Create test data
    df = pd.DataFrame({
        'Location': ['Tashkent', 'Samarkand', 'Bukhara'],
        '2020': [100, 200, 150],
        '2021': [110, 210, 160],
        '2022': [120, 220, 170]
    })
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as tmp:
        temp_path = tmp.name
    
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    
    try:
        # Load with sidecar engine
        status = engine.load_excel_file(temp_path)
        
        assert status['sheets_loaded'] == 1
        assert 'Data' in status['sheet_names']
        
        # Get sheet data
        sheet_data = engine.get_sheet_data('Data')
        assert sheet_data is not None
        assert len(sheet_data) == 3  # 3 rows of data
        
    finally:
        os.unlink(temp_path)


def test_sidecar_engine_multi_sheet():
    """Test sidecar engine with multi-sheet Excel file"""
    engine = DictionaryIngestionEngine()
    
    # Create test data for multiple sheets
    df1 = pd.DataFrame({
        'Region': ['A', 'B'],
        '2020': [10, 20],
        '2021': [15, 25]
    })
    
    df2 = pd.DataFrame({
        'Region': ['C', 'D'],
        '2020': [30, 40],
        '2021': [35, 45]
    })
    
    # Save to temporary Excel file with multiple sheets
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as tmp:
        temp_path = tmp.name
    
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Industry', index=False)
        df2.to_excel(writer, sheet_name='Agriculture', index=False)
    
    try:
        # Load with sidecar engine
        status = engine.load_excel_file(temp_path)
        
        assert status['sheets_loaded'] == 2
        assert 'Industry' in status['sheet_names']
        assert 'Agriculture' in status['sheet_names']
        
        # Get sheet list
        sheets = engine.get_sheet_list()
        assert len(sheets) == 2
        
        # Get individual sheet data
        ind_data = engine.get_sheet_data('Industry')
        agr_data = engine.get_sheet_data('Agriculture')
        
        assert ind_data is not None
        assert agr_data is not None
        assert len(ind_data) == 2
        assert len(agr_data) == 2
        
    finally:
        os.unlink(temp_path)


def test_column_sanitization():
    """Test column name sanitization for duplicates"""
    engine = DictionaryIngestionEngine()
    
    # Create DataFrame with duplicate columns
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['Value', 'Value', 'Year'])
    
    # Sanitize
    df_clean = engine._sanitize_columns(df)
    
    assert 'Value' in df_clean.columns
    assert 'Value.1' in df_clean.columns
    assert 'Year' in df_clean.columns
    assert len(df_clean.columns) == 3
    assert len(set(df_clean.columns)) == 3  # All unique


def test_year_column_extraction():
    """Test extraction of year columns"""
    engine = DictionaryIngestionEngine()
    
    # Create DataFrame with year columns
    df = pd.DataFrame({
        'Location': ['A', 'B'],
        2020: [10, 20],
        2021: [15, 25],
        '2022': [20, 30],
        'Other': ['X', 'Y']
    })
    
    year_cols = engine._extract_year_columns(df)
    
    assert 2020 in year_cols
    assert 2021 in year_cols
    assert len(year_cols) >= 2  # At least 2020 and 2021


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
