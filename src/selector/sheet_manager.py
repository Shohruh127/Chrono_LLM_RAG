# =============================================================================
# src/selector/sheet_manager.py - Sheet Manager for Multi-Sheet Excel Files
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Phase 4: Selector Architecture
# =============================================================================

import pandas as pd
from typing import List, Optional
from pathlib import Path


class SheetManager:
    """Manages Excel sheet detection, preview, and selection for domain-specific analysis."""
    
    # Domain detection mapping (Uzbek/English)
    DOMAIN_MAPPINGS = {
        # Uzbek (lowercase for case-insensitive matching)
        "qishloq": "Agriculture",
        "demografiya": "Demography",
        "sanoat": "Industry",
        "savdo": "Trade",
        "transport": "Transport",
        "qurilish": "Construction",
        "moliya": "Finance",
        "ta'lim": "Education",
        "talim": "Education",
        "sog'liqni saqlash": "Healthcare",
        "sogʻliqni saqlash": "Healthcare",
        "madaniyat": "Culture",
        "sport": "Sports",
        "turizm": "Tourism",
        # English (lowercase)
        "agriculture": "Agriculture",
        "demography": "Demography",
        "industry": "Industry",
        "trade": "Trade",
        "transport": "Transport",
        "construction": "Construction",
        "finance": "Finance",
        "education": "Education",
        "healthcare": "Healthcare",
        "culture": "Culture",
        "sports": "Sports",
        "tourism": "Tourism",
    }
    
    def __init__(self, filepath: str):
        """
        Initialize SheetManager with Excel file path.
        
        Args:
            filepath: Path to Excel file
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.filepath.suffix not in ['.xlsx', '.xls']:
            raise ValueError(f"Invalid file format. Expected Excel file (.xlsx/.xls), got: {self.filepath.suffix}")
        
        self._excel_file = pd.ExcelFile(self.filepath)
        self._current_sheet = None
        self._current_df = None
    
    def list_sheets(self) -> List[dict]:
        """
        Return list of sheets with metadata.
        
        Returns:
            List of dictionaries with sheet information:
            [{"name": "7-Agriculture", "rows": 150, "cols": 12, "domain": "Agriculture"}, ...]
        """
        sheets = []
        
        for sheet_name in self._excel_file.sheet_names:
            try:
                # Read sheet to get dimensions
                df = pd.read_excel(self.filepath, sheet_name=sheet_name)
                
                # Detect domain
                domain = self.detect_domain(sheet_name)
                
                sheets.append({
                    "name": sheet_name,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "domain": domain
                })
            except Exception as e:
                # Skip problematic sheets
                print(f"⚠️  Warning: Could not read sheet '{sheet_name}': {e}")
                continue
        
        return sheets
    
    def get_sheet_preview(self, sheet_name: str, rows: int = 5) -> pd.DataFrame:
        """
        Return preview of sheet data.
        
        Args:
            sheet_name: Name of the sheet to preview
            rows: Number of rows to preview (default: 5)
            
        Returns:
            DataFrame with preview data
        """
        if sheet_name not in self._excel_file.sheet_names:
            raise ValueError(f"Sheet '{sheet_name}' not found in file. Available sheets: {self._excel_file.sheet_names}")
        
        df = pd.read_excel(self.filepath, sheet_name=sheet_name, nrows=rows)
        return df
    
    def select_sheet(self, sheet_name: str) -> pd.DataFrame:
        """
        Load and return the selected sheet as DataFrame.
        
        Args:
            sheet_name: Name of the sheet to select
            
        Returns:
            Full DataFrame of the selected sheet
        """
        if sheet_name not in self._excel_file.sheet_names:
            raise ValueError(f"Sheet '{sheet_name}' not found in file. Available sheets: {self._excel_file.sheet_names}")
        
        self._current_sheet = sheet_name
        self._current_df = pd.read_excel(self.filepath, sheet_name=sheet_name)
        
        return self._current_df.copy()
    
    def get_current_context(self) -> dict:
        """
        Return current selection context.
        
        Returns:
            Dictionary with current context:
            {"sheet": "7-Agriculture", "domain": "Agriculture", "loaded_at": ..., "rows": 150, "cols": 12}
        """
        if self._current_sheet is None:
            return {
                "sheet": None,
                "domain": None,
                "loaded_at": None,
                "rows": 0,
                "cols": 0
            }
        
        from datetime import datetime
        
        return {
            "sheet": self._current_sheet,
            "domain": self.detect_domain(self._current_sheet),
            "loaded_at": datetime.utcnow().isoformat(),
            "rows": len(self._current_df) if self._current_df is not None else 0,
            "cols": len(self._current_df.columns) if self._current_df is not None else 0
        }
    
    def detect_domain(self, sheet_name: str) -> str:
        """
        Infer domain from sheet name.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            Detected domain name (e.g., "Agriculture", "Industry")
            Returns "Unknown" if domain cannot be detected
            
        Examples:
            "7-Agriculture" -> "Agriculture"
            "3-Sanoat" -> "Industry" (Uzbek translation)
            "Qishloq xo'jaligi" -> "Agriculture"
        """
        # Normalize sheet name for matching
        normalized_name = sheet_name.lower().strip()
        
        # Remove common prefixes (numbers, dashes)
        # e.g., "7-Agriculture" -> "agriculture"
        import re
        normalized_name = re.sub(r'^\d+[-.\s]*', '', normalized_name)
        
        # Try exact match first
        if normalized_name in self.DOMAIN_MAPPINGS:
            return self.DOMAIN_MAPPINGS[normalized_name]
        
        # Try partial match (check if any mapping key is in the sheet name)
        for key, domain in self.DOMAIN_MAPPINGS.items():
            if key in normalized_name:
                return domain
        
        # Try to extract domain from common patterns
        # e.g., "7-Qishloq xo'jaligi" -> look for "qishloq"
        for key, domain in self.DOMAIN_MAPPINGS.items():
            if key in normalized_name.split():
                return domain
        
        return "Unknown"
    
    def get_available_domains(self) -> List[str]:
        """
        Get list of unique domains available in the file.
        
        Returns:
            List of domain names found in the file
        """
        sheets = self.list_sheets()
        domains = list(set([s["domain"] for s in sheets]))
        domains.sort()
        return domains
    
    def __repr__(self) -> str:
        return f"SheetManager(filepath='{self.filepath}', sheets={len(self._excel_file.sheet_names)})"
