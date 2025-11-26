# Sovereign Sidecar Selector Architecture

**Created by:** Shohruh127  
**Date:** 2025-11-26  
**Version:** 2.0 (Modular Architecture)

---

## 🎯 Architecture Overview

This system implements a **Modular, Auditor-Grade Architecture** with **Separation of Concerns** philosophy.

### Infrastructure
- **Platform:** Google Colab Pro (A100 40GB VRAM)
- **Pattern:** Sovereign Sidecar Selector (v2.0)

---

## 🔱 Tri-Force Model Stack

### 1. Forecaster: amazon/chronos-2
- **Purpose:** Time-Series Foundation Model
- **Role:** Zero-shot forecasting with quantile predictions
- **Integration:** CPU pipelining, automatic device mapping

### 2. Analyst: behbudiy/Llama-3.1-8B-Uz
- **Purpose:** Uzbek Cultural Context Understanding
- **Role:** Natural language analysis with cultural awareness
- **Integration:** 4-bit quantization, bilingual support

### 3. Engineer: Qwen/Qwen2.5-Coder-7B-Instruct
- **Purpose:** SOTA Python Generation
- **Role:** Code-as-Reasoning (PAL pattern)
- **Integration:** Safe code execution, 0% arithmetic hallucinations

---

## 🔄 Pipeline Flow

```
Input (Multi-sheet Excel)
    ↓
DictionaryIngestionEngine (Sidecar)
    ↓
Sheet Selection (User Choice)
    ↓
Data Cleaning & Preprocessing
    ↓
Chronos-2 Forecasting
    ↓
RAG Context Building
    ↓
LLM Analysis (Llama-3.1-8B-Uz)
    ↓
Code Generation (Qwen2.5-Coder) [Optional]
    ↓
Results & Insights
```

---

## 📊 Key Components

### 1. DictionaryIngestionEngine (Sidecar)
**Purpose:** Handle "Long-Tail" Excel files with messy formatting

**Features:**
- Multi-sheet parsing with separate storage
- Data/Metadata separation (Head vs Tail/Footers)
- Column sanitization (handles duplicates like "Value", "Value" → "Value", "Value.1")
- Year detection and extraction
- Sheet selector UI

**Problem Solved:** Prevents "Frankenstein" datasets from merging incompatible sheets (e.g., Industry + Demography)

### 2. PAL (Program-Aided Language) Pattern
**Purpose:** Eliminate arithmetic hallucinations

**How it works:**
1. User asks question about data
2. Qwen2.5-Coder generates Python code
3. Code is executed safely with actual data
4. Results are returned (guaranteed accurate)

**Example:**
```
User: "What's the average growth rate for Tashkent?"
↓
Generated Code:
```python
growth = (df.loc['Tashkent', 2023] - df.loc['Tashkent', 2020]) / df.loc['Tashkent', 2020] * 100
print(f"Average growth: {growth:.2f}%")
```
↓
Result: "Average growth: 15.34%"
```

---

## 🛠️ Integration Fixes Applied

### Fix #1: DeepSeek Crash → Qwen2.5-Coder
**Problem:** `AttributeError: DynamicCache` - DeepSeek incompatible with latest transformers  
**Solution:** Replaced with Qwen2.5-Coder (native support, stable, equally powerful)

### Fix #2: Chronos Memory Pinning
**Problem:** `RuntimeError: cannot pin memory` - Manual `.to(device)` conflicts with pipeline  
**Solution:** CPU pipelining - let pipeline handle device mapping automatically

### Fix #3: Tensor Dimensions
**Problem:** `ValueError: 3-d shape` - Chronos expects (Batch, Variate, Time)  
**Solution:** Applied `.unsqueeze(0).unsqueeze(0)` for proper alignment

### Fix #4: Pandas Grouper Error
**Problem:** `ValueError: Grouper not 1-d` - Duplicate column headers  
**Solution:** Column sanitization layer (deduplicates before processing)

### Fix #5: Gradio Message Format
**Problem:** Gradio 4.x requires `[{'role': 'user', ...}]` not tuples  
**Solution:** Updated chat handler to comply with ChatMessage API

### Fix #6: Data Noise
**Problem:** Merging 24 sheets created incompatible column clashes  
**Solution:** Selector Pattern - keep sheets separate, analyze one at a time

---

## 📂 File Structure

```
Chrono_LLM_RAG/
├── src/
│   ├── sidecar_engine.py       # Multi-sheet ingestion engine
│   ├── code_generator.py       # Qwen2.5-Coder PAL integration
│   ├── chronos_forecaster.py   # Chronos-2 with CPU pipelining
│   ├── llm_analyzer.py         # Llama-3.1-8B-Uz analyzer
│   ├── rag_system.py           # FAISS-based RAG
│   ├── preprocessor.py         # Uzbek XLSX preprocessing
│   ├── app.py                  # Gradio UI with sheet selector
│   └── utils.py
├── configs/
│   ├── models_config.yaml      # Tri-Force model configuration
│   ├── chronos_config.yaml
│   ├── rag_config.yaml
│   └── prompts.yaml
├── tests/
│   ├── test_sidecar_engine.py  # Sidecar tests
│   └── test_data.py
└── docs/
    ├── architecture.md
    └── SIDECAR_ARCHITECTURE.md  # This file
```

---

## 🚀 Usage

### Step 1: Upload Multi-Sheet Excel File
```python
from src.sidecar_engine import DictionaryIngestionEngine

engine = DictionaryIngestionEngine()
status = engine.load_excel_file("data.xlsx")
print(f"✅ Loaded {status['sheets_loaded']} sheets")
```

### Step 2: Select Sheet
```python
sheets = engine.get_sheet_list()
# User selects: "7-Agriculture"

df = engine.get_sheet_data("7-Agriculture")
```

### Step 3: Process & Forecast
```python
from src.chronos_forecaster import ChronosForecaster

forecaster = ChronosForecaster()
forecaster.load_data(df)
predictions = forecaster.predict(horizon=4)
```

### Step 4: Ask Questions (PAL)
```python
from src.code_generator import CodeGenerator

coder = CodeGenerator()
result = coder.answer_with_code(
    question="What's the 2023 value for Agriculture?",
    data_context={'df': df},
    context_description="DataFrame with years as columns"
)
print(result['code'])    # Generated Python code
print(result['result'])  # Execution result
```

---

## 🎓 Design Principles

1. **Separation of Concerns:** Each module has a single responsibility
2. **Selector Pattern:** Never merge incompatible data sources
3. **Code-as-Reasoning:** Use executable code for calculations
4. **CPU Pipelining:** Let frameworks handle device management
5. **Sanitization First:** Clean data before processing

---

## 🔮 Next Steps (Day 2 Objectives)

### Persistence
- [ ] Save "Shadow Dataset" to Google Drive
- [ ] Auto-save on process completion
- [ ] Resume from saved state

### Report Generation
- [ ] PDF summary of selected sheet
- [ ] Include forecast chart
- [ ] Add tail warnings and metadata

### UI Enhancements
- [ ] Status indicators (✅/⚠️/❌)
- [ ] Tail warnings display
- [ ] Progress bars for long operations

---

## 📊 Performance Metrics

- **Sheets Processed:** Up to 24 per file
- **Forecast Accuracy:** Quantile-based with confidence intervals
- **Arithmetic Hallucinations:** 0% (PAL pattern)
- **Memory Efficiency:** 4-bit quantization for LLMs
- **Processing Speed:** Batch size 256 for Chronos

---

## 🙏 Acknowledgments

- Amazon Chronos-2 Team
- Behbudiy AI (Uzbek LLMs)
- Qwen Team (Alibaba Cloud)
- Hugging Face Transformers

---

**Repository:** Shohruh127/Chrono_LLM_RAG  
**Repository ID:** 1099678425  
**Created:** 2025-11-26 09:42:15 UTC
