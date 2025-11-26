# Test Report Generator
# Created by: Shohruh127
# Date: 2025-11-26

import pytest
import pandas as pd
import tempfile
import os
import importlib.util

# Load module directly without triggering src/__init__.py
spec = importlib.util.spec_from_file_location(
    "report_generator", 
    os.path.join(os.path.dirname(__file__), '..', 'src', 'report_generator.py')
)
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)
ReportGenerator = report_module.ReportGenerator


def test_report_generator_basic():
    """Test basic report generation"""
    generator = ReportGenerator()
    
    # Create test data
    historical = pd.DataFrame({
        'timestamp': pd.date_range('2020', periods=5, freq='Y'),
        'target': [100, 110, 120, 130, 140],
        'id': ['LOC_001'] * 5
    })
    
    predictions = pd.DataFrame({
        'timestamp': pd.date_range('2025', periods=3, freq='Y'),
        'predictions': [150, 160, 170]
    })
    
    # Generate report
    html = generator.generate_html_report(
        sheet_name="Test Sheet",
        historical_data=historical,
        predictions=predictions
    )
    
    # Check HTML content
    assert 'Test Sheet' in html
    assert 'Historical Records' in html
    assert '<!DOCTYPE html>' in html
    assert '</html>' in html


def test_report_generator_no_predictions():
    """Test report generation without predictions"""
    generator = ReportGenerator()
    
    historical = pd.DataFrame({
        'timestamp': pd.date_range('2020', periods=5, freq='Y'),
        'target': [100, 110, 120, 130, 140]
    })
    
    html = generator.generate_html_report(
        sheet_name="Test",
        historical_data=historical,
        predictions=None
    )
    
    assert 'Test' in html
    assert 'Historical Records' in html


def test_report_generator_with_metadata():
    """Test report generation with metadata"""
    generator = ReportGenerator()
    
    historical = pd.DataFrame({
        'timestamp': pd.date_range('2020', periods=3, freq='Y'),
        'target': [100, 110, 120]
    })
    
    metadata = {
        'sheet_info': 'Agriculture data',
        'data_rows': 3,
        'year_columns': {2020: ['col1'], 2021: ['col2']}
    }
    
    html = generator.generate_html_report(
        sheet_name="Test",
        historical_data=historical,
        metadata=metadata
    )
    
    assert 'Metadata' in html
    assert 'sheet_info' in html


def test_report_generator_with_warnings():
    """Test report generation with warnings"""
    generator = ReportGenerator()
    
    historical = pd.DataFrame({
        'timestamp': pd.date_range('2020', periods=3, freq='Y'),
        'target': [100, 110, 120]
    })
    
    warnings = [
        "Missing data for year 2019",
        "Column name duplicates detected"
    ]
    
    html = generator.generate_html_report(
        sheet_name="Test",
        historical_data=historical,
        warnings=warnings
    )
    
    assert 'Warnings' in html
    assert 'Missing data' in html


def test_save_html_report():
    """Test saving HTML report to file"""
    generator = ReportGenerator()
    
    html = "<html><body>Test Report</body></html>"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        temp_path = f.name
    
    try:
        saved_path = generator.save_html_report(html, temp_path)
        
        assert saved_path == temp_path
        assert os.path.exists(temp_path)
        
        with open(temp_path, 'r') as f:
            content = f.read()
        
        assert content == html
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
