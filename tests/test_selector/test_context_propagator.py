# =============================================================================
# tests/test_selector/test_context_propagator.py - Tests for ContextPropagator
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pytest
import pandas as pd
from src.selector.context_propagator import ContextPropagator


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "Region": ["Namangan", "Uchqo'rg'on", "Pop"],
            "2020": [1250, 890, 650],
            "2021": [1300, 920, 680],
            "2022": [1280, 910, 660],
            "2023": [1350, 950, 700],
        }
    )


def test_context_propagator_initialization():
    """Test ContextPropagator initialization"""
    context = ContextPropagator()
    assert not context.has_context()
    assert context.get_context() is None
    assert context.get_dataframe() is None
    assert context.get_domain() is None
    assert context.get_sheet_name() is None


def test_set_context(sample_dataframe):
    """Test setting context"""
    context = ContextPropagator()

    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    assert context.has_context()
    assert context.get_domain() == "Agriculture"
    assert context.get_sheet_name() == "7-Agriculture"


def test_get_context(sample_dataframe):
    """Test getting context information"""
    context = ContextPropagator()
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    ctx = context.get_context()

    assert ctx is not None
    assert ctx["sheet_name"] == "7-Agriculture"
    assert ctx["domain"] == "Agriculture"
    assert ctx["rows"] == 3
    assert ctx["cols"] == 5
    assert "set_at" in ctx
    assert "columns" in ctx
    assert "Region" in ctx["columns"]


def test_get_dataframe(sample_dataframe):
    """Test getting DataFrame"""
    context = ContextPropagator()
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    df = context.get_dataframe()

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == list(sample_dataframe.columns)

    # Ensure it's a copy
    assert df is not sample_dataframe


def test_get_domain_prompt_with_context(sample_dataframe):
    """Test domain-specific prompt generation"""
    context = ContextPropagator()
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    prompt = context.get_domain_prompt()

    assert "Agriculture" in prompt
    assert "agricultural metrics" in prompt
    assert "7-Agriculture" in prompt
    assert "Region" in prompt
    assert "Uzbekistan" in prompt


def test_get_domain_prompt_without_context():
    """Test prompt generation without context"""
    context = ContextPropagator()

    prompt = context.get_domain_prompt()

    assert "No data context" in prompt


def test_get_domain_prompt_for_different_domains(sample_dataframe):
    """Test domain-specific prompts for various domains"""
    domains = [
        ("Industry", "industrial output"),
        ("Demography", "population statistics"),
        ("Trade", "trade volumes"),
        ("Transport", "transportation metrics"),
        ("Finance", "financial indicators"),
        ("Education", "educational statistics"),
    ]

    for domain, expected_text in domains:
        context = ContextPropagator()
        context.set_context(f"Sheet-{domain}", sample_dataframe, domain)
        prompt = context.get_domain_prompt()

        assert domain in prompt
        assert expected_text in prompt


def test_clear_context(sample_dataframe):
    """Test clearing context"""
    context = ContextPropagator()
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    assert context.has_context()

    context.clear_context()

    assert not context.has_context()
    assert context.get_context() is None
    assert context.get_dataframe() is None
    assert context.get_domain() is None


def test_has_context(sample_dataframe):
    """Test context existence check"""
    context = ContextPropagator()

    assert not context.has_context()

    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")
    assert context.has_context()

    context.clear_context()
    assert not context.has_context()


def test_context_immutability(sample_dataframe):
    """Test that returned DataFrame is a copy"""
    context = ContextPropagator()
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    df1 = context.get_dataframe()
    df2 = context.get_dataframe()

    # Modify df1
    df1.loc[0, "Region"] = "Modified"

    # df2 and original should be unchanged
    assert df2.loc[0, "Region"] != "Modified"
    assert context.get_dataframe().loc[0, "Region"] != "Modified"


def test_context_propagator_repr_without_context():
    """Test string representation without context"""
    context = ContextPropagator()
    repr_str = repr(context)

    assert "ContextPropagator" in repr_str
    assert "None" in repr_str


def test_context_propagator_repr_with_context(sample_dataframe):
    """Test string representation with context"""
    context = ContextPropagator()
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")

    repr_str = repr(context)

    assert "ContextPropagator" in repr_str
    assert "Agriculture" in repr_str
    assert "7-Agriculture" in repr_str


def test_multiple_context_updates(sample_dataframe):
    """Test updating context multiple times"""
    context = ContextPropagator()

    # First context
    context.set_context("7-Agriculture", sample_dataframe, "Agriculture")
    assert context.get_domain() == "Agriculture"

    # Update context
    df2 = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
    context.set_context("3-Industry", df2, "Industry")

    assert context.get_domain() == "Industry"
    assert context.get_sheet_name() == "3-Industry"
    assert len(context.get_dataframe()) == 2


def test_get_domain_prompt_unknown_domain(sample_dataframe):
    """Test prompt generation for unknown domain"""
    context = ContextPropagator()
    context.set_context("Unknown-Sheet", sample_dataframe, "Unknown")

    prompt = context.get_domain_prompt()

    assert "Unknown" in prompt
    assert "general economic" in prompt
