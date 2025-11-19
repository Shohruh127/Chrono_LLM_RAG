# Architecture

**Created by:** Shohruh127  
**Date:** 2025-11-19 10:47:49  
**Repository ID:** 1099678425

## System Components

1. **Data Layer** - File upload, validation, preprocessing
2. **Chronos-2 Forecaster** - Zero-shot time series predictions
3. **RAG System** - FAISS-based semantic search
4. **LLM Analyzer** - Behbudiy/Mistral-7B-Uz bilingual analysis

## Data Flow

    Input Data → Preprocessing → Chronos Forecasting → RAG Indexing → LLM Analysis → User Response

## Technologies

- **Chronos-2** - Amazon time series model
- **FAISS** - Meta AI vector search
- **Transformers** - Hugging Face library
- **Gradio** - Interactive UI
- **Behbudiy/Mistral-7B-Uz** - Uzbek LLM

## Key Features

- Zero-shot forecasting
- Bilingual support (Uzbek/English)
- Anti-hallucination prompts
- Conversation memory
- Uzbek Cyrillic transliteration
