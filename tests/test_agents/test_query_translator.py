# =============================================================================
# tests/test_agents/test_query_translator.py - Query Translator Tests
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import pytest
import sys
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.query_translator import QueryTranslator


class TestQueryTranslator:
    """Test suite for query translation and intent extraction."""

    @pytest.fixture
    def translator(self):
        """Create translator instance for testing."""
        return QueryTranslator()

    def test_uzbek_to_english_simple(self, translator):
        """Test simple Uzbek to English translation."""
        uzbek = "2023 yilda jami ishlab chiqarish qancha bo'lgan?"
        english = translator.translate_to_english(uzbek)
        
        assert 'total' in english.lower() or 'sum' in english.lower()
        assert '2023' in english
        assert 'production' in english.lower()

    def test_uzbek_pattern_qancha(self, translator):
        """Test 'qancha' (how much) pattern."""
        uzbek = "qancha"
        english = translator.translate_to_english(uzbek)
        assert 'how much' in english.lower()

    def test_uzbek_pattern_jami(self, translator):
        """Test 'jami' (total) pattern."""
        uzbek = "jami"
        english = translator.translate_to_english(uzbek)
        assert 'total' in english.lower()

    def test_uzbek_pattern_ortacha(self, translator):
        """Test 'o'rtacha' (average) pattern."""
        uzbek = "o'rtacha"
        english = translator.translate_to_english(uzbek)
        assert 'average' in english.lower()

    def test_uzbek_pattern_eng_katta(self, translator):
        """Test 'eng katta' (maximum) pattern."""
        uzbek = "eng katta"
        english = translator.translate_to_english(uzbek)
        assert 'maximum' in english.lower()

    def test_uzbek_pattern_eng_kichik(self, translator):
        """Test 'eng kichik' (minimum) pattern."""
        uzbek = "eng kichik"
        english = translator.translate_to_english(uzbek)
        assert 'minimum' in english.lower()

    def test_extract_intent_sum(self, translator):
        """Test intent extraction for sum operation."""
        query = "What is the total production in 2023?"
        intent = translator.extract_intent(query)
        
        assert intent['operation'] == 'sum'
        assert intent['filter'] is not None
        assert intent['filter']['year'] == 2023

    def test_extract_intent_average(self, translator):
        """Test intent extraction for average operation."""
        query = "What is the average yield?"
        intent = translator.extract_intent(query)
        
        assert intent['operation'] == 'mean'

    def test_extract_intent_maximum(self, translator):
        """Test intent extraction for maximum operation."""
        query = "What is the maximum production?"
        intent = translator.extract_intent(query)
        
        assert intent['operation'] == 'max'

    def test_extract_intent_minimum(self, translator):
        """Test intent extraction for minimum operation."""
        query = "What is the minimum value?"
        intent = translator.extract_intent(query)
        
        assert intent['operation'] == 'min'

    def test_extract_intent_count(self, translator):
        """Test intent extraction for count operation."""
        query = "How many records are there?"
        intent = translator.extract_intent(query)
        
        assert intent['operation'] == 'count'

    def test_extract_intent_with_year(self, translator):
        """Test year extraction from query."""
        query = "What was the production in 2022?"
        intent = translator.extract_intent(query)
        
        assert intent['filter'] is not None
        assert intent['filter']['year'] == 2022

    def test_extract_intent_column_detection(self, translator):
        """Test column name detection."""
        query = "What is the total agriculture production?"
        intent = translator.extract_intent(query)
        
        assert intent['column'] is not None
        assert 'agriculture' in intent['column'].lower()

    def test_generate_code_prompt(self, translator):
        """Test code prompt generation."""
        query = "What is the total production in 2023?"
        schema = {
            'columns': ['Year', 'Production', 'target'],
            'dtypes': {'Year': 'int64', 'Production': 'float64', 'target': 'float64'}
        }
        
        prompt = translator.generate_code_prompt(query, schema)
        
        assert query in prompt
        assert 'Year' in prompt
        assert 'Production' in prompt
        assert 'result' in prompt
        assert 'df' in prompt

    def test_uzbek_complex_query(self, translator):
        """Test complex Uzbek query translation."""
        uzbek = "2023 yilda qishloq xo'jaligi ishlab chiqarishi qancha bo'lgan?"
        english = translator.translate_to_english(uzbek)
        
        assert '2023' in english
        assert any(word in english.lower() for word in ['agriculture', 'agricultural'])
        assert any(word in english.lower() for word in ['production', 'how much'])

    def test_english_query_passthrough(self, translator):
        """Test that English queries pass through correctly."""
        english_query = "What is the total production?"
        result = translator.translate_to_english(english_query)
        
        # Should be similar to input (case may differ)
        assert 'production' in result.lower()

    def test_extract_intent_no_operation(self, translator):
        """Test intent extraction when no clear operation."""
        query = "Tell me about the data"
        intent = translator.extract_intent(query)
        
        assert intent['operation'] is None

    def test_uzbek_year_pattern(self, translator):
        """Test 'yilda' (in year) pattern."""
        uzbek = "2023 yilda"
        english = translator.translate_to_english(uzbek)
        
        assert '2023' in english
        assert 'year' in english.lower()

    def test_multiple_uzbek_patterns(self, translator):
        """Test query with multiple Uzbek patterns."""
        uzbek = "2023 yilda jami o'rtacha hosildorlik qancha?"
        english = translator.translate_to_english(uzbek)
        
        assert '2023' in english
        assert 'total' in english.lower() or 'average' in english.lower()
        assert 'yield' in english.lower()
