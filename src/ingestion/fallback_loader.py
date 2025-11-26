# Fallback Loader Module
# Created by: Shohruh127
# Purpose: Deterministic safety net for when AI-based cleaning fails

import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import logging
import traceback


# Set up logging
logger = logging.getLogger(__name__)


class FallbackLoader:
    """
    Deterministic safety net for Excel file loading.
    
    This loader provides a robust fallback mechanism when more sophisticated
    parsing methods fail or produce unreliable results. It prioritizes
    reliability over feature completeness.
    
    Key principles:
    - Never hard-crash on malformed input
    - Always return something usable
    - Keep all columns with minimal transformation
    - Log detailed error information for debugging
    """
    
    # Method flags for tracking which loader produced the data
    METHOD_AI_CLEAN = 'AI clean'
    METHOD_FALLBACK = 'Fallback'
    METHOD_MANUAL_CORRECTION = 'Manual correction'
    METHOD_EMPTY = 'Empty result'
    
    def __init__(self, max_retries: int = 3, verbose: bool = True):
        """
        Initialize the FallbackLoader.
        
        Args:
            max_retries: Maximum number of loading attempts
            verbose: Whether to log detailed information
        """
        self.max_retries = max_retries
        self.verbose = verbose
        self._errors: List[Dict[str, Any]] = []
        self._method_used: str = ''
        
    def _log(self, message: str, level: str = 'info') -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            if level == 'error':
                logger.error(message)
            elif level == 'warning':
                logger.warning(message)
            else:
                logger.info(message)
    
    def _record_error(
        self,
        error: Exception,
        context: str,
        attempt: int
    ) -> None:
        """Record an error for later debugging."""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'attempt': attempt,
            'traceback': traceback.format_exc()
        }
        self._errors.append(error_info)
        self._log(f"Error in {context} (attempt {attempt}): {error}", 'error')
    
    def _try_basic_load(
        self,
        filepath: Path,
        sheet: Union[str, int] = 0
    ) -> Optional[pd.DataFrame]:
        """
        Attempt basic pandas read with minimal assumptions.
        
        Args:
            filepath: Path to Excel file
            sheet: Sheet name or index
            
        Returns:
            DataFrame if successful, None otherwise
        """
        suffix = filepath.suffix.lower()
        
        # Try different engines
        engines = ['openpyxl', 'xlrd'] if suffix == '.xlsx' else ['xlrd', 'openpyxl']
        
        for engine in engines:
            try:
                df = pd.read_excel(
                    filepath,
                    sheet_name=sheet,
                    header=None,  # Don't assume header
                    engine=engine
                )
                return df
            except Exception as e:
                self._record_error(e, f"basic_load with {engine}", 1)
                continue
                
        return None
    
    def _try_headerless_load(
        self,
        filepath: Path,
        sheet: Union[str, int] = 0
    ) -> Optional[pd.DataFrame]:
        """
        Load file without any header processing.
        
        Args:
            filepath: Path to Excel file
            sheet: Sheet name or index
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            # Read all data as strings to avoid type conversion errors
            df = pd.read_excel(
                filepath,
                sheet_name=sheet,
                header=None,
                dtype=str
            )
            return df
        except Exception as e:
            self._record_error(e, "headerless_load", 2)
            return None
    
    def _try_csv_fallback(
        self,
        filepath: Path,
        sheet: Union[str, int] = 0
    ) -> Optional[pd.DataFrame]:
        """
        Last resort: try to read CSV files with multiple encodings.
        
        This method only handles CSV files as a fallback for when
        a CSV file is passed to the loader. For Excel files (.xlsx/.xls),
        this method returns None.
        
        Args:
            filepath: Path to file
            sheet: Ignored for CSV files
            
        Returns:
            DataFrame if successful (only for CSV files), None otherwise
        """
        if filepath.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(filepath, header=None, encoding='utf-8')
                return df
            except Exception:
                pass
                
            try:
                df = pd.read_csv(filepath, header=None, encoding='cp1251')
                return df
            except Exception as e:
                self._record_error(e, "csv_fallback", 3)
                
        return None
    
    def load(
        self,
        filepath: Union[str, Path],
        sheet: Union[str, int] = 0,
        return_metadata: bool = False
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Load an Excel file with multiple fallback strategies.
        
        This method never raises exceptions - it always returns something.
        
        Args:
            filepath: Path to Excel file
            sheet: Sheet name or index
            return_metadata: If True, return dict with metadata
            
        Returns:
            DataFrame, or dict with 'data' and 'metadata' if return_metadata=True
        """
        filepath = Path(filepath)
        self._errors = []
        self._method_used = ''
        
        # Check if file exists
        if not filepath.exists():
            self._log(f"File not found: {filepath}", 'error')
            self._method_used = self.METHOD_EMPTY
            result_df = pd.DataFrame()
            
            if return_metadata:
                return {
                    'data': result_df,
                    'metadata': {
                        'method': self._method_used,
                        'errors': self._errors,
                        'success': False,
                        'file_exists': False
                    }
                }
            return result_df
        
        # Try loading strategies in order
        df = None
        
        # Strategy 1: Basic load
        df = self._try_basic_load(filepath, sheet)
        if df is not None and not df.empty:
            self._method_used = self.METHOD_FALLBACK
            self._log(f"Loaded {len(df)} rows using basic loader")
        else:
            # Strategy 2: Headerless load
            df = self._try_headerless_load(filepath, sheet)
            if df is not None and not df.empty:
                self._method_used = self.METHOD_FALLBACK
                self._log(f"Loaded {len(df)} rows using headerless loader")
            else:
                # Strategy 3: CSV fallback (for debugging)
                df = self._try_csv_fallback(filepath, sheet)
                if df is not None and not df.empty:
                    self._method_used = self.METHOD_FALLBACK
                    self._log(f"Loaded {len(df)} rows using CSV fallback")
                else:
                    # All strategies failed - return empty DataFrame
                    self._method_used = self.METHOD_EMPTY
                    df = pd.DataFrame()
                    self._log("All loading strategies failed", 'error')
        
        if return_metadata:
            return {
                'data': df,
                'metadata': {
                    'method': self._method_used,
                    'errors': self._errors,
                    'success': not df.empty,
                    'row_count': len(df),
                    'column_count': len(df.columns) if not df.empty else 0,
                    'file_exists': True
                }
            }
            
        return df
    
    def load_with_validation(
        self,
        filepath: Union[str, Path],
        primary_loader_result: Optional[pd.DataFrame] = None,
        expected_columns: Optional[List[str]] = None,
        min_rows: int = 1
    ) -> Dict[str, Any]:
        """
        Load with validation against expected schema.
        
        Used to verify AI-cleaned results or trigger fallback.
        
        Args:
            filepath: Path to Excel file
            primary_loader_result: Result from AI/primary loader to validate
            expected_columns: Expected column names
            min_rows: Minimum expected rows
            
        Returns:
            Dict with validated data and metadata
        """
        # Check if primary result is valid
        use_primary = True
        validation_issues = []
        
        if primary_loader_result is None:
            use_primary = False
            validation_issues.append("Primary loader returned None")
        elif primary_loader_result.empty:
            use_primary = False
            validation_issues.append("Primary loader returned empty DataFrame")
        elif len(primary_loader_result) < min_rows:
            use_primary = False
            validation_issues.append(f"Primary loader returned fewer than {min_rows} rows")
        elif expected_columns:
            # Check for hallucinated columns
            actual_cols = set(str(c).lower() for c in primary_loader_result.columns)
            expected_cols = set(c.lower() for c in expected_columns)
            
            missing = expected_cols - actual_cols
            if missing:
                validation_issues.append(f"Missing expected columns: {missing}")
                # Don't necessarily fail, but note it
        
        if use_primary:
            self._method_used = self.METHOD_AI_CLEAN
            return {
                'data': primary_loader_result,
                'method': self.METHOD_AI_CLEAN,
                'validation_issues': validation_issues,
                'used_fallback': False
            }
        else:
            # Use fallback
            self._log("Primary loader failed validation, using fallback", 'warning')
            result = self.load(filepath, return_metadata=True)
            
            return {
                'data': result['data'],
                'method': self.METHOD_FALLBACK,
                'validation_issues': validation_issues,
                'used_fallback': True,
                'fallback_errors': result['metadata'].get('errors', [])
            }
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get list of errors from last load operation."""
        return self._errors
    
    def get_method_used(self) -> str:
        """Get the method flag indicating how data was loaded."""
        return self._method_used
