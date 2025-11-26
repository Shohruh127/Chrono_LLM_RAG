# =============================================================================
# src/llm_analyzer.py - LLM Analysis Module
# Created by: Shohruh127
# Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-11-19 11:11:39
# Current User's Login: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict
from pathlib import Path
import yaml


class LLMAnalyzer:
    """LLM-based analysis with RAG integration"""

    def __init__(self,
                 model_name: str = "behbudiy/Llama-3.1-8B-Uz",
                 rag_system=None,
                 config_path: str = "configs/prompts.yaml"):
        """
        Initialize LLM analyzer
        
        Args:
            model_name: Hugging Face model name (default: behbudiy/Llama-3.1-8B-Uz for Uzbek cultural context)
            rag_system: RAG system instance
            config_path: Path to prompts configuration
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.rag = rag_system
        self.available = False
        self.conversation_history = []

        # Load prompts config if exists
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
        else:
            self.prompts_config = None

    def load_model(self, use_4bit: bool = True):
        """
        Load LLM model
        
        Args:
            use_4bit: Whether to use 4-bit quantization
        """
        print(f"📥 Loading LLM: {self.model_name}")
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
                device_map="cuda",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        self.available = True
        print(f"✅ LLM loaded on {next(self.model.parameters()).device}")

    def load_data(self, hist_df, pred_df=None):
        """Load data into RAG system"""
        if self.rag:
            self.rag.load_data(hist_df, pred_df)
            print("✅ Data loaded into RAG")

    def analyze(self, question: str, temperature: float = 0.3, max_tokens: int = 800) -> str:
        """
        Analyze question with LLM + RAG
        
        Args:
            question: User question
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        if not self.available:
            return "❌ LLM not loaded. Call load_model() first."

        if not question or not question.strip():
            return "❓ Please ask a question."

        print(f"\n{'='*60}")
        print(f"🤖 Processing: '{question[:80]}...'")

        # Get RAG context
        rag_context = ""
        if self.rag:
            rag_context = self.rag.build_context_for_llm(question)
            print(f"📊 RAG Context: {len(rag_context)} chars")

        # Detect language
        uzbek_words = ['qanday', 'nima', 'nega', 'sanoat', 'qishloq', 'taqqoslang', 'ber', 'ayt', 'haqida', 'bashorat']
        is_uzbek = any(word in question.lower() for word in uzbek_words)

        print(f"🌐 Language: {'Uzbek' if is_uzbek else 'English'}")

        # Build system prompt
        if is_uzbek:
            system_msg = """Siz Toshkent viloyati ma'lumotlari tahlilchisisiz.

QATTIQ QOIDALAR:
1. FAQAT DATABASE CONTEXT dagi raqamlardan foydalaning
2. Agar ma'lumot yo'q bo'lsa, "Ma'lumot mavjud emas" deb yozing
3. Har bir raqamni yil bilan ko'rsating
4. HECH QACHON taxmin qilmang

JAVOB FORMATI:
- Aniq raqamlar va yillar
- Hisob-kitoblarni ko'rsating"""
        else:
            system_msg = """You are a data analyst for Tashkent region.

STRICT RULES:
1. Use ONLY numbers from DATABASE CONTEXT
2. If data is missing, write "Data not available for <year/location>"
3. Always cite year with each number
4. NEVER guess or make up values

OUTPUT FORMAT:
- Specific numbers and years
- Show calculations"""

        # Build prompt
        prompt = f"""<s>[INST] {system_msg}

{rag_context}

USER QUESTION: {question}

YOUR ANSWER: [/INST]"""

        print(f"📝 Prompt: {len(prompt)} chars")

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4000
            )

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = {k: v.to(device) for k, v in inputs.items()}

            print(f"⚡ Generating...")

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.85,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract response
            if "[/INST]" in full_response:
                response = full_response.split("[/INST]")[-1].strip()
            else:
                response = full_response.strip()

            # Clean up
            for token in ["</s>", "[INST]", "[/INST]", "<s>"]:
                response = response.replace(token, "")
            response = response.strip()

            print(f"✅ Response: {len(response)} chars")
            print(f"{'='*60}\n")

            # Store in history
            self.conversation_history.append({
                'question': question,
                'answer': response
            })

            return response

        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {type(e).__name__}\n\n{str(e)}\n\n```\n{traceback.format_exc()}\n```"
            print(error_msg)
            return error_msg

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")

    def get_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history
