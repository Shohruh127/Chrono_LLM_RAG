# Chrono_LLM_RAG 🔮🤖

**Chronos-2 Time Series Forecasting + LLM + RAG Pipeline for Tashkent Region Economic Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Created by:** Shohruh127  
**Repository ID:** 1099678425  
**Last Updated:** 2025-11-19 10:46:14 UTC

---

## 🎯 Project Overview

This project combines state-of-the-art time series forecasting with Retrieval-Augmented Generation (RAG) and Large Language Models (LLM) to analyze and predict economic indicators for Tashkent region, Uzbekistan.

### Key Features

- ✅ **Chronos-2 Forecasting** - Amazon's zero-shot time series forecasting
- ✅ **RAG System** - FAISS-based semantic search
- ✅ **LLM Integration** - Behbudiy/Mistral-7B-Uz for bilingual analysis
- ✅ **Uzbek Data Processing** - Automatic Cyrillic to Latin transliteration
- ✅ **Interactive UI** - Gradio interface
- ✅ **Anti-Hallucination** - Strict prompts and validation
- ✅ **Sovereign Agent (Phase 5)** - PAL architecture with <1% hallucination rate
- ✅ **Security Guardrails** - AST-based code validation and sandboxed execution

---

## 🏗️ System Architecture

### Pipeline Components

1. **Data Layer** - Excel/CSV upload, Uzbek transliteration, preprocessing
2. **Forecasting Layer** - Chronos-2 model, quantile forecasts, batch processing
3. **RAG Layer** - Semantic passages, FAISS indexing, top-k retrieval
4. **LLM Layer** - Mistral-7B-Uz, bilingual responses, context-aware analysis

### Data Flow

Data Upload → Preprocessing → Chronos Forecasting → RAG Indexing → LLM Analysis → User Response

---

## 🚀 Quick Start

### Google Colab

    # Clone repository
    !git clone https://github.com/Shohruh127/Chrono_LLM_RAG.git
    %cd Chrono_LLM_RAG
    
    # Install dependencies
    !pip install -r requirements.txt

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

- Architecture Overview - docs/architecture.md
- Setup Guide - docs/setup.md
- API Reference - docs/api_reference.md

---

## 📁 Project Structure

    Chrono_LLM_RAG/
    ├── configs/              # Configuration files
    ├── src/                  # Source code
    ├── notebooks/            # Jupyter notebooks
    ├── tests/                # Unit tests
    ├── data/                 # Data directory
    ├── models/               # Model storage
    └── docs/                 # Documentation

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
- Behbudiy (Uzbek LLM)
- Hugging Face Transformers
- Meta AI FAISS

---

## 📊 Project Stats

- Lines of Code: 5000+
- Languages: Python, YAML, Markdown
- Data Range: 2016-2024 (historical), 2025-2028 (forecasts)
- Locations: 45 time series
- Supported Languages: English, Uzbek

---

**Repository ID:** 1099678425  
**Created:** 2025-11-19 10:46:14 UTC  
**Author:** Shohruh127
