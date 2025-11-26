# =============================================================================
# src/agents/query_translator.py - Query Translation and Code Generation
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import re
from typing import Dict, Optional


class QueryTranslator:
    """
    Translates Uzbek queries to English and extracts intent for code generation.
    Supports common Uzbek query patterns for numerical/statistical operations.
    """

    # Uzbek to English pattern mappings
    UZBEK_PATTERNS = {
        # Question words
        "qancha": "how much",
        "necha": "how many",
        "qachon": "when",
        "qayerda": "where",
        "kimlar": "who",
        
        # Operations
        "jami": "total",
        "yig'indi": "sum",
        "o'rtacha": "average",
        "eng katta": "maximum",
        "eng kichik": "minimum",
        "eng yuqori": "highest",
        "eng past": "lowest",
        
        # Filters
        "yil bo'yicha": "by year",
        "yilda": "in year",
        "viloyat bo'yicha": "by region",
        "viloyatda": "in region",
        "bo'yicha": "by",
        "da": "in",
        
        # Data types
        "hosildorlik": "yield",
        "ishlab chiqarish": "production",
        "qishloq xo'jaligi": "agriculture",
        "sanoat": "industry",
        "savdo": "trade",
        
        # Common terms
        "bo'lgan": "was",
        "edi": "was",
        "nima": "what",
    }

    def __init__(self, llama_uz_model=None):
        """
        Initialize query translator.
        
        Args:
            llama_uz_model: Optional Llama-3.1-8B-Uz model for advanced translation
                           (if None, uses pattern-based translation)
        """
        self.llama_model = llama_uz_model

    def translate_to_english(self, uzbek_query: str) -> str:
        """
        Translate Uzbek query to precise English.
        
        Args:
            uzbek_query: Query in Uzbek language
            
        Returns:
            Translated English query
        """
        # If LLM model is available, use it for better translation
        if self.llama_model is not None:
            return self._llm_translate(uzbek_query)
        
        # Otherwise, use pattern-based translation
        return self._pattern_translate(uzbek_query)

    def _pattern_translate(self, uzbek_query: str) -> str:
        """
        Pattern-based translation using dictionary lookup.
        
        Args:
            uzbek_query: Query in Uzbek
            
        Returns:
            English translation
        """
        english_query = uzbek_query.lower()
        
        # Replace Uzbek patterns with English equivalents
        for uzbek, english in self.UZBEK_PATTERNS.items():
            english_query = english_query.replace(uzbek, english)
        
        # Clean up extra spaces
        english_query = re.sub(r'\s+', ' ', english_query).strip()
        
        return english_query

    def _llm_translate(self, uzbek_query: str) -> str:
        """
        LLM-based translation using Llama model.
        
        Args:
            uzbek_query: Query in Uzbek
            
        Returns:
            English translation
        """
        # This would call the actual LLM model
        # For now, fallback to pattern translation
        return self._pattern_translate(uzbek_query)

    def extract_intent(self, query: str) -> Dict:
        """
        Extract query intent and parameters.
        
        Args:
            query: English query (translated if needed)
            
        Returns:
            dict: {
                "operation": str (sum, average, max, min, etc.),
                "column": str or None,
                "filter": dict or None,
                "groupby": str or None
            }
        """
        query_lower = query.lower()
        intent = {
            "operation": None,
            "column": None,
            "filter": None,
            "groupby": None
        }
        
        # Detect operation
        if any(word in query_lower for word in ['total', 'sum', 'jami']):
            intent['operation'] = 'sum'
        elif any(word in query_lower for word in ['average', 'mean', "o'rtacha"]):
            intent['operation'] = 'mean'
        elif any(word in query_lower for word in ['maximum', 'max', 'highest', 'eng katta']):
            intent['operation'] = 'max'
        elif any(word in query_lower for word in ['minimum', 'min', 'lowest', 'eng kichik']):
            intent['operation'] = 'min'
        elif any(word in query_lower for word in ['count', 'number of']):
            intent['operation'] = 'count'
        
        # Extract year filter
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            intent['filter'] = {'year': year}
        
        # Extract column name (simple heuristic)
        data_keywords = ['production', 'yield', 'output', 'agriculture', 'industry', 'trade']
        for keyword in data_keywords:
            if keyword in query_lower:
                intent['column'] = keyword
                break
        
        return intent

    def generate_code_prompt(self, english_query: str, df_schema: Dict) -> str:
        """
        Create prompt for Qwen-Coder to generate pandas code.
        
        Args:
            english_query: Translated English query
            df_schema: DataFrame schema with column names and types
            
        Returns:
            Formatted prompt for code generation
        """
        columns_info = df_schema.get('columns', [])
        dtypes_info = df_schema.get('dtypes', {})
        
        columns_str = ', '.join(columns_info)
        
        prompt = f"""Generate Python pandas code to answer this query using the DataFrame 'df'.

Query: {english_query}

DataFrame schema:
- Columns: {columns_str}
- Data types: {dtypes_info}

Requirements:
1. Use only pandas operations on the 'df' DataFrame
2. Store the final result in a variable named 'result'
3. Do NOT print anything or use display functions
4. Keep the code simple and efficient
5. Use proper filtering and aggregation

Example:
result = df[df['Year'] == 2023]['Production'].sum()

Generate the code:"""
        
        return prompt
