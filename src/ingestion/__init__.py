# Ingestion Module
# Created by: Shohruh127
# Purpose: Data ingestion pipeline for government Excel files

"""
Chrono_LLM_RAG Ingestion Module

This module provides robust data ingestion for government Excel files
with complex structures including:
- Legacy encodings (CP1251, UTF-8, Windows-1252)
- Merged headers spanning multiple rows
- Multi-sheet structures
- Mixed Cyrillic/Latin scripts
- Unstructured footers with metadata

The "Sidecar" strategy separates:
- Clean data (for math/forecasting via Chronos)
- Metadata footers (for LLM context extraction via Llama-Uz)

Example usage:
    from src.ingestion import ExcelParser, SidecarSplitter, FallbackLoader
    
    # Parse Excel file
    parser = ExcelParser("Namangan_Macro_2024.xlsx")
    sheets = parser.list_sheets()
    
    # Load specific sheet
    df_raw = parser.parse_sheet("Agriculture", detect_header=True)
    
    # Split data from metadata
    splitter = SidecarSplitter()
    result = splitter.split(df_raw)
    
    data_for_chronos = result["data_table"]
    context_for_llama = result["metadata_tail"]
    warnings = result["warnings"]
"""

from .excel_parser import ExcelParser
from .sidecar_splitter import SidecarSplitter
from .fallback_loader import FallbackLoader

from pathlib import Path
from typing import Union, Dict, Any, Optional
import pandas as pd


__all__ = [
    'ExcelParser',
    'SidecarSplitter', 
    'FallbackLoader',
    'ingest_file'
]


def ingest_file(
    filepath: Union[str, Path],
    sheet: Union[str, int] = 0,
    detect_header: bool = True,
    split_data: bool = True,
    use_fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Orchestrate the full ingestion pipeline for an Excel file.
    
    This convenience function combines ExcelParser, SidecarSplitter, and
    FallbackLoader to provide a complete, robust ingestion process.
    
    Args:
        filepath: Path to Excel file
        sheet: Sheet name or index to process
        detect_header: Whether to auto-detect header row
        split_data: Whether to split data from metadata footer
        use_fallback_on_error: Whether to use FallbackLoader on errors
        
    Returns:
        Dictionary containing:
            - data: Clean DataFrame (or split data_table if split_data=True)
            - raw_data: Original parsed DataFrame
            - metadata_tail: Footer text (if split_data=True)
            - warnings: Extracted warnings (if split_data=True)
            - sheet_metadata: Parsing metadata
            - method: Loading method used
            - success: Whether loading succeeded
            
    Example:
        result = ingest_file("macro_data.xlsx", sheet="Agriculture")
        df = result["data"]
        notes = result["metadata_tail"]
    """
    filepath = Path(filepath)
    result = {
        'data': None,
        'raw_data': None,
        'metadata_tail': '',
        'warnings': [],
        'sheet_metadata': {},
        'method': '',
        'success': False,
        'errors': []
    }
    
    # Initialize components
    parser = ExcelParser(filepath)
    splitter = SidecarSplitter()
    fallback = FallbackLoader(verbose=True)
    
    try:
        # Attempt to parse with ExcelParser
        df_raw = parser.parse_sheet(
            sheet=sheet,
            detect_header=detect_header,
            handle_merged_cells=True
        )
        
        result['raw_data'] = df_raw
        result['sheet_metadata'] = parser.get_sheet_metadata(sheet)
        result['method'] = 'ExcelParser'
        
        if split_data:
            # Split data from metadata
            split_result = splitter.split(df_raw)
            result['data'] = split_result['data_table']
            result['metadata_tail'] = split_result['metadata_tail']
            result['warnings'] = split_result['warnings']
            result['split_row_index'] = split_result['split_row_index']
        else:
            result['data'] = df_raw
            
        result['success'] = True
        
    except Exception as e:
        result['errors'].append({
            'type': type(e).__name__,
            'message': str(e),
            'stage': 'ExcelParser'
        })
        
        if use_fallback_on_error:
            # Use fallback loader
            try:
                fallback_result = fallback.load(filepath, sheet=sheet, return_metadata=True)
                
                df_raw = fallback_result['data']
                result['raw_data'] = df_raw
                result['method'] = fallback.get_method_used()
                result['errors'].extend(fallback.get_errors())
                
                if not df_raw.empty:
                    if split_data:
                        split_result = splitter.split(df_raw)
                        result['data'] = split_result['data_table']
                        result['metadata_tail'] = split_result['metadata_tail']
                        result['warnings'] = split_result['warnings']
                        result['split_row_index'] = split_result['split_row_index']
                    else:
                        result['data'] = df_raw
                        
                    result['success'] = True
                else:
                    result['data'] = pd.DataFrame()
                    
            except Exception as fallback_error:
                result['errors'].append({
                    'type': type(fallback_error).__name__,
                    'message': str(fallback_error),
                    'stage': 'FallbackLoader'
                })
                result['data'] = pd.DataFrame()
        else:
            result['data'] = pd.DataFrame()
            
    return result


def ingest_all_sheets(
    filepath: Union[str, Path],
    detect_header: bool = True,
    split_data: bool = True,
    use_fallback_on_error: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Ingest all sheets from an Excel file.
    
    Args:
        filepath: Path to Excel file
        detect_header: Whether to auto-detect header row
        split_data: Whether to split data from metadata footer
        use_fallback_on_error: Whether to use FallbackLoader on errors
        
    Returns:
        Dictionary mapping sheet names to ingest_file results
    """
    filepath = Path(filepath)
    parser = ExcelParser(filepath)
    
    try:
        sheets = parser.list_sheets()
    except Exception:
        return {}
        
    results = {}
    for sheet_name in sheets:
        results[sheet_name] = ingest_file(
            filepath,
            sheet=sheet_name,
            detect_header=detect_header,
            split_data=split_data,
            use_fallback_on_error=use_fallback_on_error
        )
        
    return results
