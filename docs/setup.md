# Setup Guide

**Created by:** Shohruh127  
**Date:** 2025-11-19 10:47:49

## Google Colab

    !git clone https://github.com/Shohruh127/Chrono_LLM_RAG.git
    %cd Chrono_LLM_RAG
    !pip install -r requirements.txt

## Local Setup

    git clone https://github.com/Shohruh127/Chrono_LLM_RAG.git
    cd Chrono_LLM_RAG
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Configuration

Edit files in configs/ directory:
- chronos_config.yaml - Forecasting settings
- rag_config.yaml - RAG retrieval settings
- prompts.yaml - LLM prompts

## Running

    # Gradio interface
    python src/app.py
    
    # Jupyter notebook
    jupyter notebook notebooks/03_full_pipeline_demo.ipynb
