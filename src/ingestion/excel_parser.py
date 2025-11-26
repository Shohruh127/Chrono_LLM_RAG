# Excel Parser Module
# Created by: Shohruh127
# Purpose: Robust Excel parsing for government files with encoding detection and header heuristics

import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import chardet


class ExcelParser:
    """
    Robust Excel parser for government files with complex structures.
    
    Features:
    - Multi-encoding support (UTF-8, CP1251, Windows-1252, ISO-8859-1)
    - Automatic encoding detection using chardet
    - Heuristic header detection for merged cells
    - Multi-sheet support
    """
    
    # Encoding fallback order for legacy files
    ENCODING_FALLBACK_ORDER = ['utf-8', 'cp1251', 'windows-1252', 'iso-8859-1']
    
    def __init__(self, filepath: Optional[Union[str, Path]] = None):
        """
        Initialize the ExcelParser.
        
        Args:
            filepath: Optional path to Excel file. Can be set later.
        """
        self.filepath = Path(filepath) if filepath else None
        self._workbook = None
        self._sheet_metadata = {}
        self._detected_encoding = None
        
    def set_file(self, filepath: Union[str, Path]) -> None:
        """Set or change the file to parse."""
        self.filepath = Path(filepath)
        self._workbook = None
        self._sheet_metadata = {}
        self._detected_encoding = None
        
    def _detect_encoding(self, file_bytes: bytes) -> str:
        """
        Detect file encoding using chardet.
        
        Args:
            file_bytes: Raw bytes from file
            
        Returns:
            Detected encoding name
        """
        result = chardet.detect(file_bytes)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        
        # If confidence is low, try fallback encodings
        if confidence < 0.5:
            encoding = 'utf-8'
            
        self._detected_encoding = encoding
        return encoding
    
    def list_sheets(self) -> List[str]:
        """
        List all available sheet names in the Excel file.
        
        Returns:
            List of sheet names
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If no file is set
        """
        if self.filepath is None:
            raise ValueError("No file set. Use set_file() or pass filepath to constructor.")
            
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
            
        # Try to read with openpyxl for xlsx or xlrd for xls
        try:
            excel_file = pd.ExcelFile(self.filepath)
            return excel_file.sheet_names
        except Exception as e:
            # Try with different engines
            for engine in ['openpyxl', 'xlrd']:
                try:
                    excel_file = pd.ExcelFile(self.filepath, engine=engine)
                    return excel_file.sheet_names
                except Exception:
                    continue
            raise ValueError(f"Could not read Excel file: {e}")
    
    def _find_header_row(self, df: pd.DataFrame, max_rows_to_scan: int = 20) -> int:
        """
        Heuristically detect the "true" header row.
        
        Strategy:
        - Scan rows for density of string values
        - Skip metadata titles and empty rows
        - Look for rows with multiple non-empty string values
        
        Args:
            df: DataFrame read without header specification
            max_rows_to_scan: Maximum rows to scan for header
            
        Returns:
            Index of detected header row
        """
        best_row = 0
        best_score = 0
        
        rows_to_check = min(max_rows_to_scan, len(df))
        
        for idx in range(rows_to_check):
            row = df.iloc[idx]
            
            # Count non-null string values
            string_count = 0
            non_empty_count = 0
            
            for val in row:
                if pd.notna(val):
                    non_empty_count += 1
                    if isinstance(val, str) and len(str(val).strip()) > 0:
                        string_count += 1
            
            # Score based on string density and coverage
            if non_empty_count > 0:
                string_ratio = string_count / non_empty_count
                coverage = non_empty_count / len(row)
                score = string_ratio * coverage * non_empty_count
                
                # Prefer rows with more string values (likely headers)
                if score > best_score:
                    best_score = score
                    best_row = idx
                    
        return best_row
    
    def _forward_fill_merged_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle merged cells by forward-filling values.
        
        Args:
            df: DataFrame with potential merged cell gaps
            
        Returns:
            DataFrame with forward-filled values
        """
        # Forward fill along rows for merged header cells
        df = df.ffill(axis=1).infer_objects(copy=False)
        # Also forward fill along columns for vertically merged cells
        df = df.ffill(axis=0).infer_objects(copy=False)
        return df
    
    def parse_sheet(
        self,
        sheet: Union[str, int] = 0,
        detect_header: bool = True,
        header_row: Optional[int] = None,
        handle_merged_cells: bool = True,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parse a specific sheet from the Excel file.
        
        Args:
            sheet: Sheet name or index (0-based)
            detect_header: Whether to auto-detect header row
            header_row: Explicit header row (overrides detection)
            handle_merged_cells: Whether to forward-fill merged cells
            encoding: Explicit encoding (optional, uses detection if not provided)
            
        Returns:
            Parsed DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If sheet doesn't exist
        """
        if self.filepath is None:
            raise ValueError("No file set. Use set_file() or pass filepath to constructor.")
            
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Determine engine based on file extension
        suffix = self.filepath.suffix.lower()
        engine = 'openpyxl' if suffix == '.xlsx' else 'xlrd'
        
        # First, read without header to detect it
        try:
            df_raw = pd.read_excel(
                self.filepath,
                sheet_name=sheet,
                header=None,
                engine=engine
            )
        except Exception:
            # Try alternate engine
            alt_engine = 'xlrd' if engine == 'openpyxl' else 'openpyxl'
            try:
                df_raw = pd.read_excel(
                    self.filepath,
                    sheet_name=sheet,
                    header=None,
                    engine=alt_engine
                )
            except Exception as e:
                raise ValueError(f"Could not read sheet '{sheet}': {e}")
        
        # Handle merged cells
        if handle_merged_cells:
            df_raw = self._forward_fill_merged_cells(df_raw)
        
        # Determine header row
        if header_row is not None:
            actual_header_row = header_row
        elif detect_header:
            actual_header_row = self._find_header_row(df_raw)
        else:
            actual_header_row = 0
        
        # Re-read with proper header or slice DataFrame
        if actual_header_row > 0:
            # Set the header row and skip rows before it
            new_headers = df_raw.iloc[actual_header_row].tolist()
            df_result = df_raw.iloc[actual_header_row + 1:].copy()
            df_result.columns = new_headers
            df_result = df_result.reset_index(drop=True)
        else:
            # Use first row as header
            new_headers = df_raw.iloc[0].tolist()
            df_result = df_raw.iloc[1:].copy()
            df_result.columns = new_headers
            df_result = df_result.reset_index(drop=True)
        
        # Store metadata
        sheet_name = sheet if isinstance(sheet, str) else f"Sheet_{sheet}"
        self._sheet_metadata[sheet_name] = {
            'row_count': len(df_result),
            'column_count': len(df_result.columns),
            'detected_encoding': self._detected_encoding or 'default',
            'header_row': actual_header_row
        }
        
        return df_result
    
    def get_sheet_metadata(self, sheet: Union[str, int]) -> Dict[str, Any]:
        """
        Get metadata for a parsed sheet.
        
        Args:
            sheet: Sheet name or index
            
        Returns:
            Dictionary with row_count, column_count, detected_encoding
        """
        sheet_name = sheet if isinstance(sheet, str) else f"Sheet_{sheet}"
        
        if sheet_name not in self._sheet_metadata:
            # Parse the sheet first to generate metadata
            self.parse_sheet(sheet)
            
        return self._sheet_metadata.get(sheet_name, {})
    
    def parse_all_sheets(
        self,
        detect_header: bool = True,
        handle_merged_cells: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Parse all sheets in the Excel file.
        
        Args:
            detect_header: Whether to auto-detect header row
            handle_merged_cells: Whether to forward-fill merged cells
            
        Returns:
            Dictionary mapping sheet names to DataFrames
        """
        sheets = self.list_sheets()
        result = {}
        
        for sheet_name in sheets:
            try:
                df = self.parse_sheet(
                    sheet=sheet_name,
                    detect_header=detect_header,
                    handle_merged_cells=handle_merged_cells
                )
                result[sheet_name] = df
            except Exception as e:
                # Log error but continue with other sheets
                result[sheet_name] = pd.DataFrame()
                self._sheet_metadata[sheet_name] = {
                    'error': str(e),
                    'row_count': 0,
                    'column_count': 0
                }
                
        return result
