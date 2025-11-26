# =============================================================================
# tests/test_agents/test_safe_executor.py - Safe Executor Tests
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from src.agents.safe_executor import SafeExecutor


class TestSafeExecutor:
    """Test suite for safe code execution."""

    @pytest.fixture
    def executor(self):
        """Create executor instance for testing."""
        return SafeExecutor(timeout=5)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'Production': [100, 120, 140, 160],
            'target': [10, 20, 30, 40]
        })

    def test_simple_sum(self, executor, sample_df):
        """Test simple sum operation."""
        code = "result = df['target'].sum()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 100
        assert result['error'] is None

    def test_filtered_sum(self, executor, sample_df):
        """Test filtered sum operation."""
        code = "result = df[df['Year'] == 2023]['Production'].sum()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 160
        assert result['error'] is None

    def test_mean_calculation(self, executor, sample_df):
        """Test mean calculation."""
        code = "result = df['target'].mean()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 25.0
        assert result['error'] is None

    def test_max_value(self, executor, sample_df):
        """Test max value."""
        code = "result = df['Production'].max()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 160
        assert result['error'] is None

    def test_min_value(self, executor, sample_df):
        """Test min value."""
        code = "result = df['Production'].min()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 100
        assert result['error'] is None

    def test_numpy_operations(self, executor, sample_df):
        """Test numpy operations."""
        code = "result = np.mean(df['target'].values)"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 25.0
        assert result['error'] is None

    def test_math_operations(self, executor, sample_df):
        """Test math module operations."""
        code = "result = math.sqrt(df['target'].sum())"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 10.0
        assert result['error'] is None

    def test_execution_error(self, executor, sample_df):
        """Test handling of execution errors."""
        code = "result = df['NonExistentColumn'].sum()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is False
        assert result['result'] is None
        assert result['error'] is not None
        assert 'KeyError' in result['error']

    def test_no_result_variable(self, executor, sample_df):
        """Test code without result variable."""
        code = "x = df['target'].sum()"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] is None  # No 'result' variable

    def test_execution_time_tracking(self, executor, sample_df):
        """Test that execution time is tracked."""
        code = "result = df['target'].sum()"
        result = executor.execute(code, {'df': sample_df})
        
        assert 'execution_time_ms' in result
        assert result['execution_time_ms'] > 0

    def test_safe_builtins(self, executor, sample_df):
        """Test that safe built-ins are available."""
        code = """
values = list(df['target'])
result = sum(values)
"""
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 100

    def test_complex_filtering(self, executor, sample_df):
        """Test complex filtering operations."""
        code = """
filtered = df[(df['Year'] >= 2021) & (df['Year'] <= 2023)]
result = filtered['Production'].mean()
"""
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 140.0

    def test_groupby_operations(self, executor):
        """Test groupby operations."""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'],
            'Value': [10, 20, 30, 40]
        })
        code = "result = df.groupby('Category')['Value'].sum().max()"
        result = executor.execute(code, {'df': df})
        
        assert result['success'] is True
        assert result['result'] == 60  # Category B: 20 + 40

    def test_multiple_operations(self, executor, sample_df):
        """Test multiple operations in one code block."""
        code = """
total = df['target'].sum()
count = len(df)
result = total / count
"""
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is True
        assert result['result'] == 25.0

    def test_syntax_error_handling(self, executor, sample_df):
        """Test handling of syntax errors."""
        code = "result = df['target'.sum()"  # Missing closing bracket
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is False
        assert result['result'] is None
        assert result['error'] is not None

    def test_restricted_builtins(self, executor, sample_df):
        """Test that dangerous built-ins are not available."""
        code = "result = open('/etc/passwd')"
        result = executor.execute(code, {'df': sample_df})
        
        assert result['success'] is False
        assert 'NameError' in result['error'] or 'not defined' in result['error']
