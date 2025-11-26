# =============================================================================
# src/sidecar_engine.py - Sidecar Ingestion Engine for Multi-Sheet Excel
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re
from datetime import datetime, timezone


class DictionaryIngestionEngine:
    """
    Multi-sheet Excel ingestion engine that keeps sheets separate.
    Implements the "Selector Pattern" to avoid "Frankenstein" datasets.
    
    This engine:
    1. Loads all sheets from Excel file
    2. Splits data (head) from metadata (tail/footers)
    3. Sanitizes column names (handles duplicates)
    4. Keeps sheets separate in a dictionary
    5. Allows user to select one sheet at a time for analysis
    """
    
    def __init__(self):
        self.sheets_data = {}  # Dictionary: {sheet_name: DataFrame}
        self.sheets_metadata = {}  # Dictionary: {sheet_name: metadata_dict}
        self.file_path = None
        self.status = ""
        self.warnings = []
        
    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize column names to handle duplicates.
        Converts duplicate columns like ['Value', 'Value'] to ['Value', 'Value.1']
        """
        cols = df.columns.tolist()
        new_cols = []
        seen = {}
        
        for col in cols:
            col_str = str(col).strip()
            
            if col_str in seen:
                seen[col_str] += 1
                new_col = f"{col_str}.{seen[col_str]}"
            else:
                seen[col_str] = 0
                new_col = col_str
                
            new_cols.append(new_col)
        
        df.columns = new_cols
        return df
    
    def _detect_metadata_rows(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """
        Detect metadata/footer rows at the end of the sheet.
        Returns the index where data ends and list of metadata lines.
        """
        metadata_lines = []
        data_end_idx = len(df)
        
        # Look for common footer indicators
        footer_keywords = ['source:', 'note:', 'total:', 'sum:', 'итого:', 'манба:', 'изох:']
        
        # Scan from bottom up
        for idx in range(len(df) - 1, max(0, len(df) - 20), -1):
            row = df.iloc[idx]
            row_str = ' '.join([str(val).lower() for val in row if pd.notna(val)])
            
            if any(keyword in row_str for keyword in footer_keywords):
                data_end_idx = idx
                metadata_lines.insert(0, row_str)
            elif pd.isna(row).all():
                # Empty row - might indicate start of metadata section
                if metadata_lines:  # If we already found some metadata
                    data_end_idx = idx
            elif data_end_idx < len(df):
                # We're in metadata section
                metadata_lines.insert(0, row_str)
        
        return data_end_idx, metadata_lines
    
    def _extract_year_columns(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Extract columns that represent years.
        Returns mapping of year -> list of column names for that year.
        """
        year_columns = {}
        
        for col in df.columns:
            # Try to parse year from column name or first row value
            try:
                # Check column name
                if isinstance(col, (int, float)):
                    year = int(col)
                    if 2000 <= year <= 2030:
                        if year not in year_columns:
                            year_columns[year] = []
                        year_columns[year].append(col)
                        continue
                
                # Check for year pattern in string
                col_str = str(col)
                year_match = re.search(r'\b(20[0-2][0-9])\b', col_str)
                if year_match:
                    year = int(year_match.group(1))
                    if year not in year_columns:
                        year_columns[year] = []
                    year_columns[year].append(col)
                    continue
                    
                # Check first non-null value in column
                first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if first_val:
                    try:
                        year = int(float(first_val))
                        if 2000 <= year <= 2030:
                            if year not in year_columns:
                                year_columns[year] = []
                            year_columns[year].append(col)
                    except:
                        pass
            except:
                pass
        
        return year_columns
    
    def load_excel_file(self, file_path: str) -> Dict[str, str]:
        """
        Load multi-sheet Excel file and process each sheet.
        
        Returns:
            Dictionary with status and details for each sheet
        """
        self.file_path = file_path
        self.sheets_data = {}
        self.sheets_metadata = {}
        self.warnings = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            self.status = f"✅ Loaded {len(sheet_names)} sheets from Excel file"
            
            for sheet_name in sheet_names:
                try:
                    # Read sheet
                    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                    
                    # Sanitize columns
                    df_raw = self._sanitize_columns(df_raw)
                    
                    # Detect and separate metadata
                    data_end_idx, metadata_lines = self._detect_metadata_rows(df_raw)
                    
                    # Split data and metadata
                    df_data = df_raw.iloc[:data_end_idx].copy()
                    
                    # Store metadata
                    self.sheets_metadata[sheet_name] = {
                        'metadata_lines': metadata_lines,
                        'original_rows': len(df_raw),
                        'data_rows': len(df_data),
                        'metadata_rows': len(df_raw) - data_end_idx
                    }
                    
                    # Try to identify header row(s)
                    # For now, use first row as header
                    if len(df_data) > 1:
                        df_data.columns = df_data.iloc[0]
                        df_data = df_data.iloc[1:].reset_index(drop=True)
                        df_data = self._sanitize_columns(df_data)
                    
                    self.sheets_data[sheet_name] = df_data
                    
                    # Extract year columns for this sheet
                    year_cols = self._extract_year_columns(df_data)
                    self.sheets_metadata[sheet_name]['year_columns'] = year_cols
                    
                except Exception as e:
                    warning = f"⚠️ Error processing sheet '{sheet_name}': {str(e)}"
                    self.warnings.append(warning)
                    continue
            
            # Build status report
            status_report = {
                'status': self.status,
                'file': Path(file_path).name,
                'sheets_loaded': len(self.sheets_data),
                'sheet_names': list(self.sheets_data.keys()),
                'warnings': self.warnings
            }
            
            return status_report
            
        except Exception as e:
            error_msg = f"❌ Error loading Excel file: {str(e)}"
            self.status = error_msg
            return {
                'status': error_msg,
                'error': str(e),
                'sheets_loaded': 0
            }
    
    def get_sheet_list(self) -> List[str]:
        """Get list of available sheet names"""
        return list(self.sheets_data.keys())
    
    def get_sheet_data(self, sheet_name: str) -> Optional[pd.DataFrame]:
        """Get data for a specific sheet"""
        return self.sheets_data.get(sheet_name)
    
    def get_sheet_metadata(self, sheet_name: str) -> Optional[Dict]:
        """Get metadata for a specific sheet"""
        return self.sheets_metadata.get(sheet_name)
    
    def get_sheet_summary(self, sheet_name: str) -> str:
        """
        Get a summary of a specific sheet.
        Useful for display in UI.
        """
        if sheet_name not in self.sheets_data:
            return f"❌ Sheet '{sheet_name}' not found"
        
        df = self.sheets_data[sheet_name]
        meta = self.sheets_metadata.get(sheet_name, {})
        
        summary = f"### 📊 Sheet: {sheet_name}\n\n"
        summary += f"**Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns\n"
        summary += f"**Data Rows:** {meta.get('data_rows', 'N/A')}\n"
        summary += f"**Metadata Rows:** {meta.get('metadata_rows', 0)}\n\n"
        
        # Year columns
        year_cols = meta.get('year_columns', {})
        if year_cols:
            years = sorted(year_cols.keys())
            summary += f"**Years Detected:** {years[0]}-{years[-1]} ({len(years)} years)\n\n"
        
        # Metadata lines
        metadata_lines = meta.get('metadata_lines', [])
        if metadata_lines:
            summary += f"**Metadata/Notes:**\n"
            for line in metadata_lines[:3]:  # Show first 3 lines
                summary += f"- {line[:100]}\n"
            if len(metadata_lines) > 3:
                summary += f"- ... and {len(metadata_lines) - 3} more lines\n"
        
        return summary
    
    def get_status(self) -> str:
        """Get current status message"""
        return self.status


# Initialize engine
sidecar_engine = DictionaryIngestionEngine()

print("✅ Sidecar Ingestion Engine ready!")
print(f"Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: Shohruh127")
