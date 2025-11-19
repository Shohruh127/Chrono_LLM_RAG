# Test Data Loader
# Created by: Shohruh127
# Date: 2025-11-19 10:47:49

import pytest
import pandas as pd
from src.data_loader import DataLoader


def test_data_loader():
    loader = DataLoader()
    
    df = pd.DataFrame({
        'id': ['LOC_001'] * 5,
        'timestamp': pd.date_range('2020', periods=5, freq='Y'),
        'target': [10, 12, 14, 16, 18]
    })
    
    df.to_csv('test_data.csv', index=False)
    loaded = loader.load('test_data.csv')
    
    assert len(loaded) == 5
    assert 'id' in loaded.columns
    assert 'timestamp' in loaded.columns
    assert 'target' in loaded.columns
