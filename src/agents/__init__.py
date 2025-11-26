# =============================================================================
# src/agents/__init__.py - Agents Module
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

"""
Sovereign Agent module for PAL (Program-Aided Language Model) architecture.
Prevents hallucinated numbers by generating and executing Python code.
"""

from .sovereign_agent import SovereignAgent
from .query_translator import QueryTranslator
from .ast_guardrails import ASTGuardrails
from .safe_executor import SafeExecutor

__all__ = [
    'SovereignAgent',
    'QueryTranslator',
    'ASTGuardrails',
    'SafeExecutor',
    'create_sovereign_agent'
]


def create_sovereign_agent(model_stack, context_propagator):
    """
    Convenience function to initialize agent with all dependencies.
    
    Args:
        model_stack: TriForceStack instance
        context_propagator: ContextPropagator instance
        
    Returns:
        Configured SovereignAgent instance
    """
    guardrails = ASTGuardrails()
    return SovereignAgent(model_stack, context_propagator, guardrails)
