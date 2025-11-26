# =============================================================================
# src/code_generator.py - Qwen2.5-Coder Integration for PAL Pattern
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict
import re
import traceback
from datetime import datetime, timezone


class CodeGenerator:
    """
    Qwen2.5-Coder-7B-Instruct integration for PAL (Program-Aided Language) pattern.
    
    This implements "Code-as-Reasoning" to ensure 0% arithmetic hallucinations.
    The AI writes Python code to answer queries, then executes it for accurate results.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.available = False
        
    def load_model(self, use_4bit: bool = True):
        """
        Load Qwen2.5-Coder model
        
        Args:
            use_4bit: Whether to use 4-bit quantization
        """
        print(f"📥 Loading Qwen2.5-Coder: {self.model_name}")
        print(f"   4-bit quantization: {use_4bit}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("   ✅ Padding token set")
        
        if use_4bit:
            quantization_config = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **quantization_config,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        self.available = True
        print(f"✅ Qwen2.5-Coder loaded on {next(self.model.parameters()).device}")
    
    def generate_code(self, question: str, context: str, temperature: float = 0.2) -> str:
        """
        Generate Python code to answer a question.
        
        Args:
            question: User question
            context: Data context (available variables, data structure)
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated Python code
        """
        if not self.available:
            return "# Error: Model not loaded"
        
        # Build prompt for code generation
        prompt = f"""<|im_start|>system
You are an expert Python programmer. Generate clean, correct Python code to answer questions about data.

Rules:
1. Use only standard Python libraries (pandas, numpy)
2. Code must be executable and return a clear answer
3. Add comments to explain logic
4. Handle edge cases and errors
5. Print the final answer clearly

Available context:
{context}
<|im_end|>
<|im_start|>user
Question: {question}

Generate Python code to answer this question:
<|im_end|>
<|im_start|>assistant
```python
"""
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4000
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract code between ```python and ```
            code_match = re.search(r'```python\s*(.*?)\s*```', full_response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Try to extract from assistant response
                if "<|im_start|>assistant" in full_response:
                    code = full_response.split("<|im_start|>assistant")[-1]
                    code = code.replace("<|im_end|>", "").strip()
                else:
                    code = full_response.strip()
            
            # Clean up
            code = code.replace("```python", "").replace("```", "").strip()
            
            return code
            
        except Exception as e:
            error_msg = f"# Error generating code: {str(e)}\n# {traceback.format_exc()}"
            print(error_msg)
            return error_msg
    
    def execute_code(self, code: str, data_context: Dict) -> str:
        """
        Execute generated code in a safe context.
        
        Args:
            code: Python code to execute
            data_context: Dictionary with data variables (e.g., {'df': dataframe})
            
        Returns:
            Execution result as string
        """
        try:
            # Create execution context
            exec_globals = {
                'pd': __import__('pandas'),
                'np': __import__('numpy'),
                '__builtins__': __builtins__,
            }
            exec_globals.update(data_context)
            
            exec_locals = {}
            
            # Capture print output
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Execute code
                exec(code, exec_globals, exec_locals)
                
                # Get output
                output = captured_output.getvalue()
                
            finally:
                sys.stdout = old_stdout
            
            # If no print output, try to find a result variable
            if not output.strip():
                for var_name in ['result', 'answer', 'output']:
                    if var_name in exec_locals:
                        output = str(exec_locals[var_name])
                        break
            
            return output if output.strip() else "✅ Code executed successfully (no output)"
            
        except Exception as e:
            error_msg = f"❌ Execution Error:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg
    
    def answer_with_code(self, question: str, data_context: Dict, 
                        context_description: str = "") -> Dict[str, str]:
        """
        Answer a question by generating and executing code (PAL pattern).
        
        Args:
            question: User question
            data_context: Dictionary with data variables
            context_description: Description of available data
            
        Returns:
            Dictionary with 'code', 'result', and 'status'
        """
        if not self.available:
            return {
                'code': '',
                'result': '❌ Model not loaded. Call load_model() first.',
                'status': 'error'
            }
        
        print(f"\n{'='*60}")
        print(f"🤖 PAL: Generating code for question...")
        print(f"   Question: {question[:80]}...")
        
        # Generate code
        code = self.generate_code(question, context_description)
        
        print(f"📝 Generated code ({len(code)} chars)")
        print(f"{'='*60}\n")
        print(code)
        print(f"\n{'='*60}")
        
        # Execute code
        print(f"⚡ Executing code...")
        result = self.execute_code(code, data_context)
        
        print(f"✅ Result ({len(result)} chars):")
        print(result)
        print(f"{'='*60}\n")
        
        status = 'success' if not result.startswith('❌') else 'error'
        
        return {
            'code': code,
            'result': result,
            'status': status
        }


# Initialize code generator
code_generator = CodeGenerator()

print("✅ Qwen2.5-Coder PAL Engine ready!")
print(f"Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: Shohruh127")
