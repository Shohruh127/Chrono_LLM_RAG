# =============================================================================
# src/agents/sovereign_agent.py - Main PAL Orchestrator
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import re
import time
from typing import Dict, Any
import pandas as pd

from .query_translator import QueryTranslator
from .ast_guardrails import ASTGuardrails
from .safe_executor import SafeExecutor


class SovereignAgent:
    """
    Main PAL (Program-Aided Language Model) orchestrator.
    Ensures no hallucinated numbers by generating and executing Python code.
    
    Pipeline:
    1. Detect language (Uzbek/English)
    2. Translate query if needed
    3. Generate Python code
    4. Validate with AST guardrails
    5. Execute safely
    6. Return answer with code citation
    """

    def __init__(self, model_stack, context_propagator, guardrails=None):
        """
        Initialize Sovereign Agent.
        
        Args:
            model_stack: TriForceStack with LLM models
            context_propagator: ContextPropagator for DataFrame management
            guardrails: Optional ASTGuardrails instance (creates if None)
        """
        self.model_stack = model_stack
        self.context = context_propagator
        self.guardrails = guardrails or ASTGuardrails()
        self.translator = QueryTranslator(model_stack.get_translator_model())
        self.executor = SafeExecutor(timeout=self.guardrails.get_timeout())

    def answer(self, query: str, df: pd.DataFrame) -> Dict:
        """
        Full PAL pipeline to answer a query.
        
        Args:
            query: User query (Uzbek or English)
            df: DataFrame to query
            
        Returns:
            dict: {
                "answer": Any,
                "answer_text": str,
                "code": str,
                "cell_reference": str,
                "confidence": str,
                "warnings": List[str],
                "execution_time_ms": float,
                "error": str or None
            }
        """
        start_time = time.time()
        warnings = []
        
        # Step 1: Detect language
        language = self._detect_language(query)
        
        # Step 2: Translate if Uzbek
        if language == 'uzbek':
            english_query = self.translator.translate_to_english(query)
            warnings.append(f"Translated from Uzbek: {english_query}")
        else:
            english_query = query
        
        # Step 3: Extract DataFrame schema
        df_schema = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape
        }
        
        # Step 4: Generate Python code
        try:
            code = self._generate_code(english_query, df_schema)
        except Exception as e:
            return self._error_response(
                f"Code generation failed: {str(e)}",
                execution_time=(time.time() - start_time) * 1000
            )
        
        # Step 5: Validate with AST guardrails
        validation = self.guardrails.validate(code)
        if not validation['safe']:
            return self._error_response(
                "Query rejected for security reasons",
                violations=validation['violations'],
                execution_time=(time.time() - start_time) * 1000
            )
        
        # Step 6: Execute safely
        exec_result = self._execute_safely(code, df)
        
        if not exec_result['success']:
            return self._error_response(
                f"Execution failed: {exec_result['error']}",
                code=code,
                execution_time=exec_result['execution_time_ms']
            )
        
        # Step 7: Format answer
        answer = exec_result['result']
        total_time = (time.time() - start_time) * 1000
        
        # Generate cell reference
        cell_ref = self._generate_cell_reference(code, df)
        
        # Generate answer text
        answer_text = self._generate_answer_text(answer, english_query, language)
        
        return {
            "answer": answer,
            "answer_text": answer_text,
            "code": code,
            "cell_reference": cell_ref,
            "confidence": "HIGH",  # Based on code execution success
            "warnings": warnings,
            "execution_time_ms": round(total_time, 2),
            "error": None
        }

    def _detect_language(self, query: str) -> str:
        """
        Detect if query is Uzbek or English.
        
        Args:
            query: Input query
            
        Returns:
            'uzbek' or 'english'
        """
        # Simple heuristic: check for common Uzbek words
        uzbek_indicators = [
            'qancha', 'necha', 'jami', "o'rtacha", 'yil', 'yilda',
            'viloyat', 'hosildorlik', 'ishlab chiqarish', 
            "qishloq xo'jaligi", "bo'yicha", "bo'lgan"
        ]
        
        query_lower = query.lower()
        for indicator in uzbek_indicators:
            if indicator in query_lower:
                return 'uzbek'
        
        return 'english'

    def _generate_code(self, english_query: str, df_schema: Dict) -> str:
        """
        Use Qwen-Coder to generate pandas code.
        
        Args:
            english_query: Query in English
            df_schema: DataFrame schema
            
        Returns:
            Generated Python code
        """
        # Generate prompt for code generation
        prompt = self.translator.generate_code_prompt(english_query, df_schema)
        
        # Use model stack to generate code
        code = self.model_stack.generate_code(prompt)
        
        # Clean up code (remove markdown, etc.)
        code = self._clean_code(code)
        
        return code

    def _clean_code(self, code: str) -> str:
        """
        Clean generated code (remove markdown, comments, etc.).
        
        Args:
            code: Raw generated code
            
        Returns:
            Cleaned code
        """
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code

    def _execute_safely(self, code: str, df: pd.DataFrame) -> Dict:
        """
        Execute validated code in sandbox.
        
        Args:
            code: Validated Python code
            df: DataFrame to use
            
        Returns:
            Execution result dictionary
        """
        context = {'df': df}
        return self.executor.execute(code, context)

    def _generate_cell_reference(self, code: str, df: pd.DataFrame) -> str:
        """
        Generate cell reference for auditability.
        
        Args:
            code: Executed code
            df: DataFrame
            
        Returns:
            Human-readable cell reference
        """
        # Extract column name from code - look for patterns like ['column_name']
        col_matches = re.findall(r'\[[\'"](\w+)[\'"]\]', code)
        
        # Get the last column mentioned (usually the one being aggregated)
        if len(col_matches) >= 2:
            column = col_matches[-1]  # Last column is usually the data column
        elif col_matches:
            column = col_matches[0]
        else:
            column = "target"
        
        # Extract filter conditions
        year_match = re.search(r'==\s*(\d{4})', code)
        if year_match:
            year = year_match.group(1)
            return f"Column '{column}', Year={year}"
        
        return f"Column '{column}', All rows"

    def _generate_answer_text(self, answer: Any, query: str, language: str) -> str:
        """
        Generate human-readable answer text.
        
        Args:
            answer: Computed answer
            query: Original query
            language: 'uzbek' or 'english'
            
        Returns:
            Formatted answer text
        """
        # Format number if it's numeric
        if isinstance(answer, (int, float)):
            formatted = f"{answer:,.2f}" if isinstance(answer, float) else f"{answer:,}"
        else:
            formatted = str(answer)
        
        # Simple template
        if language == 'uzbek':
            return f"Natija: {formatted}"
        else:
            return f"The result is {formatted}."

    def _error_response(self, error: str, code: str = None, 
                       violations: list = None, execution_time: float = 0) -> Dict:
        """
        Generate error response.
        
        Args:
            error: Error message
            code: Optional code that caused error
            violations: Optional security violations
            execution_time: Execution time in ms
            
        Returns:
            Error response dictionary
        """
        return {
            "answer": None,
            "answer_text": None,
            "code": code,
            "cell_reference": None,
            "confidence": "NONE",
            "warnings": [],
            "execution_time_ms": round(execution_time, 2),
            "error": error,
            "violations": violations or []
        }
