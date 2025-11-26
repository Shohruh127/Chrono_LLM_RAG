# Sidecar Splitter Module
# Created by: Shohruh127
# Purpose: Separate clean data tables from metadata footers for the "Sidecar" strategy

import pandas as pd
from typing import Dict, List, Any, Optional
import re


class SidecarSplitter:
    """
    Separates clean numeric data from metadata footers.
    
    The "Sidecar" strategy:
    - Data Table: Clean numeric data for the Forecaster (Chronos)
    - Metadata Tail: Unstructured footers, notes, exclusions for LLM (Llama-Uz)
    
    This separation ensures the forecaster receives clean data while
    preserving contextual metadata for the language model.
    """
    
    # Common patterns indicating footer/notes content
    FOOTER_PATTERNS = [
        r'^\s*\*',           # Lines starting with asterisk
        r'^\s*примечани',    # "Note" in Russian
        r'^\s*изоҳ',         # "Note" in Uzbek Cyrillic
        r'^\s*исключ',       # "Excludes" in Russian
        r'^\s*предварител',  # "Preliminary" in Russian
        r'^\s*source:',      # Source attribution
        r'^\s*манба',        # "Source" in Uzbek
        r'^\s*eslatma',      # "Note" in Uzbek Latin
        r'^\s*без\s+учета',  # "Excludes" in Russian
    ]
    
    # Patterns for extracting warnings/notes
    WARNING_PATTERNS = [
        (r'исключ[а-я]*\s+(.+)', 'Excludes'),      # Russian "excludes"
        (r'без\s+учета\s+(.+)', 'Excludes'),        # Russian "without accounting for"
        (r'предварител[а-я]*\s*(.+)', 'Preliminary'), # Russian "preliminary"
        (r'exclud(es|ing)?\s+(.+)', 'Excludes'),    # English excludes
        (r'preliminary\s+(.+)', 'Preliminary'),     # English preliminary
        (r'\*\s*(.+)', 'Note'),                     # Asterisk notes
    ]
    
    def __init__(
        self,
        min_data_rows: int = 3,
        empty_row_threshold: int = 2,
        numeric_column_threshold: float = 0.3
    ):
        """
        Initialize the SidecarSplitter.
        
        Args:
            min_data_rows: Minimum rows to consider as data table
            empty_row_threshold: Number of consecutive empty rows indicating split
            numeric_column_threshold: Minimum ratio of numeric columns for data rows
        """
        self.min_data_rows = min_data_rows
        self.empty_row_threshold = empty_row_threshold
        self.numeric_column_threshold = numeric_column_threshold
        
    def _is_numeric_row(self, row: pd.Series) -> bool:
        """
        Check if a row contains primarily numeric data.
        
        Args:
            row: DataFrame row
            
        Returns:
            True if row is primarily numeric
        """
        numeric_count = 0
        non_null_count = 0
        
        for val in row:
            if pd.notna(val):
                non_null_count += 1
                try:
                    # Try to convert to float
                    float(str(val).replace(',', '.').replace(' ', ''))
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
                    
        if non_null_count == 0:
            return False
            
        return (numeric_count / non_null_count) >= self.numeric_column_threshold
    
    def _is_empty_row(self, row: pd.Series) -> bool:
        """
        Check if a row is effectively empty.
        
        Args:
            row: DataFrame row
            
        Returns:
            True if row is empty or contains only whitespace
        """
        for val in row:
            if pd.notna(val):
                if isinstance(val, str):
                    if val.strip():
                        return False
                else:
                    return False
        return True
    
    def _is_footer_row(self, row: pd.Series) -> bool:
        """
        Check if a row appears to be footer/notes content.
        
        Args:
            row: DataFrame row
            
        Returns:
            True if row matches footer patterns
        """
        # Get the first non-null value
        first_val = None
        for val in row:
            if pd.notna(val) and str(val).strip():
                first_val = str(val).strip().lower()
                break
                
        if first_val is None:
            return False
            
        # Check against footer patterns
        for pattern in self.FOOTER_PATTERNS:
            if re.search(pattern, first_val, re.IGNORECASE):
                return True
                
        return False
    
    def _find_split_point(self, df: pd.DataFrame) -> int:
        """
        Find the row index where data ends and metadata begins.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Index of the split point (first row of metadata)
        """
        consecutive_empty = 0
        last_data_row = len(df)
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Check for footer patterns
            if self._is_footer_row(row):
                return idx
                
            # Check for empty rows
            if self._is_empty_row(row):
                consecutive_empty += 1
                if consecutive_empty >= self.empty_row_threshold:
                    # Found enough empty rows - this might be the split
                    return idx - consecutive_empty + 1
            else:
                consecutive_empty = 0
                
                # Check if this row is primarily text (not numeric)
                if idx >= self.min_data_rows and not self._is_numeric_row(row):
                    # This row is text-only after data rows
                    return idx
                    
                last_data_row = idx + 1
                
        return last_data_row
    
    def _extract_warnings(self, text: str) -> List[str]:
        """
        Extract warning/note phrases from footer text.
        
        Args:
            text: Footer text to analyze
            
        Returns:
            List of extracted warnings/notes
        """
        warnings = []
        
        for pattern, prefix in self.WARNING_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle groups
                    match = match[-1]  # Get last group
                if match and match.strip():
                    warning_text = f"{prefix}: {match.strip()}"
                    if warning_text not in warnings:
                        warnings.append(warning_text)
                        
        return warnings
    
    def _extract_metadata_text(self, df: pd.DataFrame, start_row: int) -> str:
        """
        Extract and format metadata text from footer rows.
        
        Args:
            df: Full DataFrame
            start_row: Index where metadata begins
            
        Returns:
            Formatted metadata text
        """
        if start_row >= len(df):
            return ""
            
        metadata_rows = df.iloc[start_row:]
        lines = []
        
        for _, row in metadata_rows.iterrows():
            row_text = []
            for val in row:
                if pd.notna(val):
                    text = str(val).strip()
                    if text:
                        row_text.append(text)
            if row_text:
                lines.append(' '.join(row_text))
                
        return '\n'.join(lines)
    
    def _clean_data_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data table for forecasting use.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def split(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Split DataFrame into clean data and metadata components.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Dictionary containing:
                - data_table: Clean DataFrame for forecasting
                - metadata_tail: Raw text for LLM
                - warnings: List of extracted warnings/notes
                - split_row_index: Where the split occurred
        """
        if df is None or df.empty:
            return {
                'data_table': pd.DataFrame(),
                'metadata_tail': '',
                'warnings': [],
                'split_row_index': 0
            }
            
        # Find split point
        split_idx = self._find_split_point(df)
        
        # Extract data table
        data_table = df.iloc[:split_idx].copy()
        data_table = self._clean_data_table(data_table)
        
        # Extract metadata text
        metadata_text = self._extract_metadata_text(df, split_idx)
        
        # Extract warnings from metadata
        warnings = self._extract_warnings(metadata_text)
        
        return {
            'data_table': data_table,
            'metadata_tail': metadata_text,
            'warnings': warnings,
            'split_row_index': split_idx
        }
    
    def split_with_context(
        self,
        df: pd.DataFrame,
        sheet_name: Optional[str] = None,
        source_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Split DataFrame with additional context information.
        
        Args:
            df: DataFrame to split
            sheet_name: Optional sheet name for context
            source_file: Optional source filename for context
            
        Returns:
            Split result with additional context fields
        """
        result = self.split(df)
        
        # Add context
        result['context'] = {
            'sheet_name': sheet_name,
            'source_file': source_file,
            'original_rows': len(df) if df is not None else 0,
            'data_rows': len(result['data_table']),
            'metadata_rows': (len(df) - result['split_row_index']) if df is not None else 0
        }
        
        return result
