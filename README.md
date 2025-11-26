# Chrono_LLM_RAG 🔮🤖

**Chronos-2 Time Series Forecasting + LLM + RAG Pipeline for Tashkent Region Economic Analysis**

**Version:** 2.0 - Sovereign Sidecar Selector Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Created by:** Shohruh127  
**Repository ID:** 1099678425  
**Last Updated:** 2025-11-26 09:42:15 UTC

---

## 🎯 Project Overview

This project implements a **Modular, Auditor-Grade Architecture** combining state-of-the-art time series forecasting with Retrieval-Augmented Generation (RAG) and Large Language Models (LLM) to analyze and predict economic indicators for Tashkent region, Uzbekistan.

### 🆕 Version 2.0 Features

- ✅ **Multi-Sheet Excel Support** - Process complex Excel files with 24+ sheets
- ✅ **Selector Pattern** - Analyze one sheet at a time (no "Frankenstein" datasets)
- ✅ **Tri-Force Model Stack** - Chronos-2 + Llama-3.1-8B-Uz + Qwen2.5-Coder
- ✅ **PAL Pattern** - Code-as-Reasoning for 0% arithmetic hallucinations
- ✅ **Google Drive Persistence** - Save processed datasets automatically
- ✅ **PDF Report Generation** - Download comprehensive analysis reports
- ✅ **Column Sanitization** - Handle duplicate column names automatically
- ✅ **Metadata Extraction** - Separate data from tail/footer notes

### Key Features (v1.0)

- ✅ **Chronos-2 Forecasting** - Amazon's zero-shot time series forecasting
- ✅ **RAG System** - FAISS-based semantic search
- ✅ **LLM Integration** - Behbudiy/Mistral-7B-Uz for bilingual analysis
- ✅ **Uzbek Data Processing** - Automatic Cyrillic to Latin transliteration
- ✅ **Interactive UI** - Gradio interface
- ✅ **Anti-Hallucination** - Strict prompts and validation

---

## 🏗️ System Architecture (v2.0)

### Tri-Force Model Stack

1. **Forecaster:** amazon/chronos-2
   - Time-Series Foundation Model
   - Zero-shot predictions with quantile forecasts
   - CPU pipelining for stability

2. **Analyst:** behbudiy/Llama-3.1-8B-Uz
   - Uzbek Cultural Context Understanding
   - Bilingual support (Uzbek/English)
   - 4-bit quantization

3. **Engineer:** Qwen/Qwen2.5-Coder-7B-Instruct
   - SOTA Python Generation
   - PAL (Program-Aided Language) pattern
   - Zero arithmetic hallucinations

### Pipeline Components

1. **Data Layer** - Excel/CSV upload, Uzbek transliteration, preprocessing
2. **Sidecar Engine** - Multi-sheet ingestion, column sanitization, selector pattern
3. **Forecasting Layer** - Chronos-2 model, quantile forecasts, batch processing
4. **RAG Layer** - Semantic passages, FAISS indexing, top-k retrieval
5. **LLM Layer** - Llama-3.1-8B-Uz, bilingual responses, context-aware analysis
6. **Code Generator** - Qwen2.5-Coder, PAL pattern, safe execution
7. **Persistence Layer** - Google Drive storage, automatic backups
8. **Report Generator** - HTML/PDF reports with visualizations

### Data Flow (v2.0)

```
Multi-Sheet Excel Upload
    ↓
DictionaryIngestionEngine (Sidecar)
    ↓
Sheet Selection (User Choice)
    ↓
Column Sanitization + Metadata Extraction
    ↓
Data Preprocessing (Uzbek format support)
    ↓
Chronos-2 Forecasting (CPU pipelining)
    ↓
RAG Context Building (FAISS)
    ↓
LLM Analysis (Llama-3.1-8B-Uz)
    ↓
Optional: Code Generation (Qwen2.5-Coder)
    ↓
Results + Report Generation
    ↓
Save to Google Drive (Persistence)
```

---

## 🚀 Quick Start

### Google Colab (Recommended)

```python
# Clone repository
!git clone https://github.com/Shohruh127/Chrono_LLM_RAG.git
%cd Chrono_LLM_RAG

# Install dependencies
!pip install -r requirements.txt

# Run Gradio interface
!python src/app.py

# Or use the programmatic API
from src.sidecar_engine import DictionaryIngestionEngine
from src.chronos_forecaster import ChronosForecaster

# Load multi-sheet Excel
engine = DictionaryIngestionEngine()
status = engine.load_excel_file("data.xlsx")

# Select and process a sheet
df = engine.get_sheet_data("7-Agriculture")

# Generate forecasts
forecaster = ChronosForecaster()
forecaster.load_data(df)
predictions = forecaster.predict(horizon=4)
```

### Local Installation

    # Clone repository
    git clone https://github.com/Shohruh127/Chrono_LLM_RAG.git
    cd Chrono_LLM_RAG
    
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Run Gradio interface
    python src/app.py

---

## 📊 Data Format

Your data should have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| id | string | Location identifier | LOC_011_IND |
| timestamp | datetime | Date or year | 2020-01-01 |
| target | float | Value to forecast | 37.78 |

### Uzbek Regional Data

For Uzbek XLSX files with Cyrillic text:
- Auto-detects Cyrillic format
- Transliterates location names
- Creates series IDs automatically
- Maps categories (Sanoat → Industry)

---

## 🎨 Features

### Time Series Forecasting
- Zero-shot predictions using Chronos-2
- Quantile forecasts with confidence intervals
- Automatic gap filling

### RAG System
- Semantic search with sentence-transformers
- FAISS vector indexing
- Context-aware passage selection

### LLM Analysis
- Behbudiy/Mistral-7B-Uz (4-bit quantization)
- Bilingual support (Uzbek/English)
- Anti-hallucination prompts
- Conversation memory

### Interactive UI
- Gradio web interface
- File upload (CSV/Excel/XLSX)
- Real-time forecasting
- AI chat interface

---

## 🧪 Testing

Run tests:

    pytest tests/ -v
    pytest tests/ --cov=src
    pytest tests/test_data.py -v

---

## 📖 Documentation

- **[Architecture Overview](docs/architecture.md)** - System design and components
- **[Sidecar Architecture](docs/SIDECAR_ARCHITECTURE.md)** - v2.0 modular design and integration fixes
- **[Setup Guide](docs/setup.md)** - Installation and configuration
- **[Usage Example](USAGE_EXAMPLE.py)** - Complete working example

---

## 📁 Project Structure (v2.0)

```
Chrono_LLM_RAG/
├── configs/              # Configuration files
│   ├── chronos_config.yaml
│   ├── rag_config.yaml
│   └── models_config.yaml     # NEW: Tri-Force model stack
├── src/                  # Source code
│   ├── sidecar_engine.py      # NEW: Multi-sheet ingestion
│   ├── code_generator.py      # NEW: Qwen2.5-Coder PAL
│   ├── drive_persistence.py   # NEW: Google Drive storage
│   ├── report_generator.py    # NEW: PDF/HTML reports
│   ├── chronos_forecaster.py
│   ├── llm_analyzer.py        # UPDATED: Llama-3.1-8B-Uz
│   ├── rag_system.py
│   ├── preprocessor.py
│   ├── app.py                 # UPDATED: Sheet selector UI
│   └── utils.py
├── tests/                # Unit tests
│   ├── test_sidecar_engine.py # NEW
│   ├── test_report_generator.py # NEW
│   └── test_data.py
├── docs/                 # Documentation
│   ├── architecture.md
│   ├── SIDECAR_ARCHITECTURE.md # NEW
│   └── setup.md
├── USAGE_EXAMPLE.py      # NEW: Complete example
└── requirements.txt
```

---

## 🤝 Contributing

Contributions welcome! Please fork, create a branch, commit, push, and open a PR.

---

## 📝 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Shohruh127**

GitHub: https://github.com/Shohruh127

Repository: https://github.com/Shohruh127/Chrono_LLM_RAG

---

## 🙏 Acknowledgments

- Amazon Chronos-2 Team
- Behbudiy AI (Uzbek LLMs - Llama-3.1-8B-Uz)
- Qwen Team (Alibaba Cloud - Qwen2.5-Coder)
- Hugging Face Transformers
- Meta AI FAISS

---

## 📊 Project Stats

- **Version:** 2.0 (Sovereign Sidecar Selector)
- **Lines of Code:** 8000+
- **Languages:** Python, YAML, Markdown
- **Data Range:** 2016-2024 (historical), 2025-2028 (forecasts)
- **Supported Languages:** English, Uzbek
- **Architecture:** Modular, Auditor-Grade

---

## 🆕 What's New in v2.0

1. **DictionaryIngestionEngine** - Multi-sheet Excel support with selector pattern
2. **Tri-Force Model Stack** - Chronos-2 + Llama-3.1-8B-Uz + Qwen2.5-Coder
3. **PAL Pattern** - Code-as-Reasoning for zero hallucinations
4. **Google Drive Persistence** - Auto-save processed datasets
5. **PDF Report Generation** - Comprehensive analysis reports
6. **Column Sanitization** - Handle duplicate headers automatically
7. **Integration Fixes** - 6 critical fixes applied (see SIDECAR_ARCHITECTURE.md)

---

**Repository ID:** 1099678425  
**Created:** 2025-11-19 10:46:14 UTC  
**Updated:** 2025-11-26 09:42:15 UTC  
**Author:** Shohruh127
