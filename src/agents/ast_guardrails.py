# =============================================================================
# src/agents/ast_guardrails.py - AST-based Code Security Validation
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import ast
from typing import List, Dict
from pathlib import Path
import yaml


class ASTGuardrails:
    """
    AST-based code security validation to prevent malicious code execution.
    Parses Python code and checks for dangerous imports, calls, and attributes.
    """

    def __init__(self, config_path: str = "configs/security_config.yaml"):
        """
        Initialize guardrails with security configuration.
        
        Args:
            config_path: Path to security configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        self.blocked_imports = set(self.config.get('blocked_imports', []))
        self.blocked_calls = set(self.config.get('blocked_calls', []))
        self.blocked_attributes = set(self.config.get('blocked_attributes', []))
        self.allowed_modules = set(self.config.get('allowed_modules', []))

    def _load_config(self) -> dict:
        """Load security configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            # Return default secure configuration
            return {
                'blocked_imports': ['os', 'sys', 'subprocess', 'socket'],
                'blocked_calls': ['open', 'eval', 'exec', 'compile', '__import__'],
                'blocked_attributes': ['__class__', '__bases__', '__code__', '__globals__'],
                'allowed_modules': ['pandas', 'numpy', 'math', 'statistics'],
                'execution_timeout_seconds': 30,
                'max_output_size_bytes': 1048576
            }
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, IOError) as e:
            # Fall back to default configuration on error
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            print("Using default security configuration.")
            return {
                'blocked_imports': ['os', 'sys', 'subprocess', 'socket'],
                'blocked_calls': ['open', 'eval', 'exec', 'compile', '__import__'],
                'blocked_attributes': ['__class__', '__bases__', '__code__', '__globals__'],
                'allowed_modules': ['pandas', 'numpy', 'math', 'statistics'],
                'execution_timeout_seconds': 30,
                'max_output_size_bytes': 1048576
            }

    def validate(self, code: str) -> Dict:
        """
        Parse and validate code safety using AST analysis.
        
        Args:
            code: Python code string to validate
            
        Returns:
            dict: {
                "safe": bool,
                "violations": List[str],
                "ast_tree": ast.AST or None
            }
        """
        violations = []
        tree = None
        
        # Try to parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"Syntax error: {str(e)}")
            return {
                "safe": False,
                "violations": violations,
                "ast_tree": None
            }
        
        # Check for security violations
        violations.extend(self._check_imports(tree))
        violations.extend(self._check_calls(tree))
        violations.extend(self._check_attributes(tree))
        
        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "ast_tree": tree
        }

    def _check_imports(self, tree: ast.AST) -> List[str]:
        """
        Check for blocked imports in the AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of violation messages
        """
        violations = []
        
        for node in ast.walk(tree):
            # Check regular imports: import os
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.blocked_imports:
                        violations.append(f"Blocked import: {alias.name}")
            
            # Check from imports: from os import system
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in self.blocked_imports:
                    violations.append(f"Blocked import: {node.module}")
        
        return violations

    def _check_calls(self, tree: ast.AST) -> List[str]:
        """
        Check for dangerous function calls in the AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of violation messages
        """
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Direct function calls: eval(), exec()
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.blocked_calls:
                        violations.append(f"Blocked call: {node.func.id}")
                
                # Attribute calls: obj.method()
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.blocked_calls:
                        violations.append(f"Blocked call: {node.func.attr}")
        
        return violations

    def _check_attributes(self, tree: ast.AST) -> List[str]:
        """
        Check for blocked attribute access (dunder methods).
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            List of violation messages
        """
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in self.blocked_attributes:
                    violations.append(f"Blocked attribute access: {node.attr}")
        
        return violations

    def get_timeout(self) -> int:
        """Get execution timeout from config."""
        return self.config.get('execution_timeout_seconds', 30)

    def get_max_output_size(self) -> int:
        """Get maximum output size from config."""
        return self.config.get('max_output_size_bytes', 1048576)
