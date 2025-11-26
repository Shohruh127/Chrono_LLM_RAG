# =============================================================================
# src/agents/safe_executor.py - Sandboxed Code Execution
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import signal
import sys
from typing import Any, Dict
import numpy as np
import pandas as pd
import math
import statistics
from datetime import datetime


class TimeoutError(Exception):
    """Custom exception for execution timeout."""
    pass


class SafeExecutor:
    """
    Safe code executor with restricted namespace and timeout.
    Executes validated Python code in a sandboxed environment.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize safe executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(self, code: str, context: Dict) -> Dict:
        """
        Execute validated code in restricted namespace.
        
        Args:
            code: Python code to execute (must be validated first)
            context: Dictionary with available variables (e.g., {"df": DataFrame})
            
        Returns:
            dict: {
                "success": bool,
                "result": Any,
                "error": str or None,
                "execution_time_ms": float
            }
        """
        import time
        
        start_time = time.time()
        
        # Create sandboxed environment
        sandbox = self._create_sandbox(context)
        
        # Set up timeout handler (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Execution exceeded {self.timeout} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        
        try:
            # Execute the code
            exec(code, sandbox['globals'], sandbox['locals'])
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            # Extract result
            result = self._capture_result(sandbox['locals'])
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "result": result,
                "error": None,
                "execution_time_ms": round(execution_time, 2)
            }
            
        except TimeoutError as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return {
                "success": False,
                "result": None,
                "error": f"Timeout: {str(e)}",
                "execution_time_ms": self.timeout * 1000
            }
            
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
                "execution_time_ms": round(execution_time, 2)
            }

    def _create_sandbox(self, context: Dict) -> Dict:
        """
        Create restricted globals/locals for exec().
        
        Args:
            context: User-provided context (e.g., DataFrame)
            
        Returns:
            dict with 'globals' and 'locals' keys
        """
        # Safe built-ins (very restricted)
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Safe modules
        safe_globals = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'math': math,
            'statistics': statistics,
            'datetime': datetime,
        }
        
        # Add user context
        safe_globals.update(context)
        
        # Separate locals for result capture
        safe_locals = {}
        
        return {
            'globals': safe_globals,
            'locals': safe_locals
        }

    def _capture_result(self, local_vars: Dict) -> Any:
        """
        Extract 'result' variable from execution.
        
        Args:
            local_vars: Local variables after execution
            
        Returns:
            The value of 'result' variable, or None
        """
        # Look for 'result' variable
        if 'result' in local_vars:
            return local_vars['result']
        
        # If no 'result', return None
        return None
