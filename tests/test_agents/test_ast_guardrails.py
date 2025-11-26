# =============================================================================
# tests/test_agents/test_ast_guardrails.py - AST Guardrails Tests
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import pytest
from src.agents.ast_guardrails import ASTGuardrails


class TestASTGuardrails:
    """Test suite for AST-based security validation."""

    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        return ASTGuardrails()

    def test_safe_pandas_code(self, guardrails):
        """Test that safe pandas code passes validation."""
        code = "result = df[df['Year'] == 2023]['Production'].sum()"
        validation = guardrails.validate(code)
        
        assert validation['safe'] is True
        assert len(validation['violations']) == 0
        assert validation['ast_tree'] is not None

    def test_blocked_import_os(self, guardrails):
        """Test that 'os' import is blocked."""
        code = """
import os
result = os.listdir('/')
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('os' in v for v in validation['violations'])

    def test_blocked_import_sys(self, guardrails):
        """Test that 'sys' import is blocked."""
        code = """
import sys
result = sys.exit()
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('sys' in v for v in validation['violations'])

    def test_blocked_import_subprocess(self, guardrails):
        """Test that 'subprocess' import is blocked."""
        code = """
import subprocess
result = subprocess.run(['ls'])
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('subprocess' in v for v in validation['violations'])

    def test_blocked_from_import(self, guardrails):
        """Test that blocked 'from' imports are caught."""
        code = """
from os import system
result = system('ls')
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('os' in v for v in validation['violations'])

    def test_blocked_call_eval(self, guardrails):
        """Test that 'eval' call is blocked."""
        code = """
result = eval('1 + 1')
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('eval' in v for v in validation['violations'])

    def test_blocked_call_exec(self, guardrails):
        """Test that 'exec' call is blocked."""
        code = """
exec('print("hello")')
result = None
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('exec' in v for v in validation['violations'])

    def test_blocked_call_open(self, guardrails):
        """Test that 'open' call is blocked."""
        code = """
result = open('/etc/passwd').read()
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('open' in v for v in validation['violations'])

    def test_blocked_attribute_class(self, guardrails):
        """Test that '__class__' attribute access is blocked."""
        code = """
result = df.__class__
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('__class__' in v for v in validation['violations'])

    def test_blocked_attribute_globals(self, guardrails):
        """Test that '__globals__' attribute access is blocked."""
        code = """
def func():
    pass
result = func.__globals__
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('__globals__' in v for v in validation['violations'])

    def test_syntax_error(self, guardrails):
        """Test that syntax errors are caught."""
        code = "result = df[df['Year'] == 2023"  # Missing closing bracket
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert any('Syntax error' in v for v in validation['violations'])
        assert validation['ast_tree'] is None

    def test_safe_numpy_operations(self, guardrails):
        """Test that safe numpy operations pass."""
        code = """
import numpy as np
result = np.mean(df['target'].values)
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is True
        assert len(validation['violations']) == 0

    def test_safe_math_operations(self, guardrails):
        """Test that safe math operations pass."""
        code = """
import math
result = math.sqrt(df['target'].sum())
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is True
        assert len(validation['violations']) == 0

    def test_multiple_violations(self, guardrails):
        """Test that multiple violations are all reported."""
        code = """
import os
import sys
result = eval('open("/etc/passwd")')
"""
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False
        assert len(validation['violations']) >= 3  # os, sys, eval

    def test_get_timeout(self, guardrails):
        """Test timeout getter."""
        timeout = guardrails.get_timeout()
        assert timeout == 30

    def test_get_max_output_size(self, guardrails):
        """Test max output size getter."""
        max_size = guardrails.get_max_output_size()
        assert max_size == 1048576
