# Data Loader Module
# Created by: Shohruh127
# Date: 2025-11-19 10:47:49

import pandas as pd
from pathlib import Path
from typing import Union


class DataLoader:
    def __init__(self):
        self.data = None
        self.filepath = None
    
    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
        
        df.columns = [c.strip().lower() for c in df.columns]
        
        required = ['id', 'timestamp', 'target']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
        
        self.data = df
        self.filepath = filepath
        
        print(f"✅ Loaded {len(df):,} records")
        print(f"   Time series: {df['id'].nunique()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def validate(self) -> dict:
        if self.data is None:
            raise ValueError("No data loaded")
        
        df = self.data
        
        return {
            'total_records': len(df),
            'time_series': df['id'].nunique(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated(subset=['id', 'timestamp']).sum(),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'target_stats': {
                'mean': df['target'].mean(),
                'std': df['target'].std(),
                'min': df['target'].min(),
                'max': df['target'].max()
            }
        }
