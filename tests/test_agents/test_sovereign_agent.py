# =============================================================================
# tests/test_agents/test_sovereign_agent.py - Sovereign Agent Tests
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.sovereign_agent import SovereignAgent
from agents.ast_guardrails import ASTGuardrails
from tri_force import TriForceStack
from selector import ContextPropagator


class TestSovereignAgent:
    """Test suite for Sovereign Agent PAL orchestration."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        model_stack = TriForceStack()
        context = ContextPropagator()
        guardrails = ASTGuardrails()
        return SovereignAgent(model_stack, context, guardrails)

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'Production': [100, 120, 140, 160],
            'target': [10, 20, 30, 40]
        })

    def test_english_query_sum(self, agent, sample_df):
        """Test English query for sum operation."""
        query = "What is the total target?"
        result = agent.answer(query, sample_df)
        
        assert result['error'] is None
        assert result['answer'] == 100
        assert result['code'] is not None
        assert result['confidence'] == 'HIGH'

    def test_english_query_with_year_filter(self, agent, sample_df):
        """Test English query with year filter."""
        query = "What is the total production in 2023?"
        result = agent.answer(query, sample_df)
        
        assert result['error'] is None
        assert result['answer'] == 160
        assert '2023' in result['code']

    def test_uzbek_query_translation(self, agent, sample_df):
        """Test Uzbek query translation and execution."""
        query = "2023 yilda jami target qancha?"
        result = agent.answer(query, sample_df)
        
        assert result['error'] is None
        assert result['answer'] is not None
        assert len(result['warnings']) > 0  # Translation warning

    def test_language_detection_english(self, agent):
        """Test language detection for English."""
        query = "What is the total?"
        lang = agent._detect_language(query)
        assert lang == 'english'

    def test_language_detection_uzbek(self, agent):
        """Test language detection for Uzbek."""
        query = "2023 yilda qancha bo'lgan?"
        lang = agent._detect_language(query)
        assert lang == 'uzbek'

    def test_security_rejection_os_import(self, agent, sample_df):
        """Test that malicious queries are rejected."""
        # This would need the code generator to produce malicious code,
        # which our stub won't do. So we test the guardrails directly.
        guardrails = agent.guardrails
        code = "import os\nresult = os.listdir('/')"
        validation = guardrails.validate(code)
        
        assert validation['safe'] is False

    def test_cell_reference_generation(self, agent, sample_df):
        """Test cell reference generation."""
        query = "What is the total production in 2023?"
        result = agent.answer(query, sample_df)
        
        assert result['cell_reference'] is not None
        assert 'Year=2023' in result['cell_reference'] or 'target' in result['cell_reference']

    def test_answer_text_generation(self, agent, sample_df):
        """Test answer text generation."""
        query = "What is the total target?"
        result = agent.answer(query, sample_df)
        
        assert result['answer_text'] is not None
        assert '100' in result['answer_text'] or result['answer_text'] != ""

    def test_execution_time_tracking(self, agent, sample_df):
        """Test that execution time is tracked."""
        query = "What is the total target?"
        result = agent.answer(query, sample_df)
        
        assert 'execution_time_ms' in result
        assert result['execution_time_ms'] > 0

    def test_code_cleaning(self, agent):
        """Test code cleaning (markdown removal)."""
        code_with_markdown = "```python\nresult = df['target'].sum()\n```"
        cleaned = agent._clean_code(code_with_markdown)
        
        assert '```' not in cleaned
        assert 'result = df' in cleaned

    def test_average_operation(self, agent, sample_df):
        """Test average operation."""
        query = "What is the average target?"
        result = agent.answer(query, sample_df)
        
        assert result['error'] is None
        assert result['answer'] == 25.0

    def test_max_operation(self, agent, sample_df):
        """Test maximum operation."""
        query = "What is the maximum target?"
        result = agent.answer(query, sample_df)
        
        assert result['error'] is None
        assert result['answer'] == 40

    def test_min_operation(self, agent, sample_df):
        """Test minimum operation."""
        query = "What is the minimum target?"
        result = agent.answer(query, sample_df)
        
        assert result['error'] is None
        assert result['answer'] == 10

    def test_error_handling_invalid_column(self, agent, sample_df):
        """Test error handling for invalid column."""
        # Generate code that references invalid column
        code = "result = df['NonExistent'].sum()"
        exec_result = agent._execute_safely(code, sample_df)
        
        assert exec_result['success'] is False
        assert 'KeyError' in exec_result['error']

    def test_uzbek_answer_text(self, agent, sample_df):
        """Test that Uzbek queries get Uzbek answer text."""
        query = "2023 yilda jami target qancha?"
        result = agent.answer(query, sample_df)
        
        # Answer text should contain "Natija" for Uzbek
        assert result['answer_text'] is not None

    def test_confidence_high_on_success(self, agent, sample_df):
        """Test that successful execution gives HIGH confidence."""
        query = "What is the total target?"
        result = agent.answer(query, sample_df)
        
        assert result['confidence'] == 'HIGH'

    def test_confidence_none_on_error(self, agent, sample_df):
        """Test that errors give NONE confidence."""
        # Create an agent with broken guardrails to simulate error
        result = agent._error_response("Test error")
        assert result['confidence'] == 'NONE'

    def test_empty_warnings_on_english(self, agent, sample_df):
        """Test that English queries have no translation warnings."""
        query = "What is the total target?"
        result = agent.answer(query, sample_df)
        
        # Warnings should be empty or not contain translation
        if len(result['warnings']) > 0:
            assert not any('Translated' in w for w in result['warnings'])

    def test_generate_cell_reference_with_column(self, agent, sample_df):
        """Test cell reference generation with column name."""
        code = "result = df[df['Year'] == 2023]['Production'].sum()"
        cell_ref = agent._generate_cell_reference(code, sample_df)
        
        assert 'Production' in cell_ref
        assert 'Year=2023' in cell_ref

    def test_generate_cell_reference_no_filter(self, agent, sample_df):
        """Test cell reference generation without filter."""
        code = "result = df['target'].sum()"
        cell_ref = agent._generate_cell_reference(code, sample_df)
        
        assert 'target' in cell_ref
        assert 'All rows' in cell_ref
