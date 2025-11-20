# =============================================================================
# src/app.py - Gradio Web Interface
# Created by: Shohruh127
# Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-11-19 11:11:39
# Current User's Login: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import UzbekXLSXPreprocessor
from src.chronos_forecaster import ChronosForecaster
from src.rag_system import RAGSystem
from src.llm_analyzer import LLMAnalyzer


# Initialize components
preprocessor = UzbekXLSXPreprocessor()
forecaster = ChronosForecaster()
rag_system = RAGSystem()
llm_analyzer = LLMAnalyzer(rag_system=rag_system)

# Global state
llm_loaded = False


def create_forecast_plot(context_df, pred_df, location_id):
    """Create plotly forecast visualization"""
    fig = go.Figure()

    # Historical
    hist = context_df[context_df['id'] == location_id].tail(50)
    fig.add_trace(go.Scatter(
        x=hist['timestamp'],
        y=hist['target'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))

    # Predictions
    preds = pred_df[pred_df.index == location_id]
    if len(preds) > 0:
        fig.add_trace(go.Scatter(
            x=preds['timestamp'],
            y=preds['predictions'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))

    fig.update_layout(
        title=f'Forecast for {location_id}',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def upload_and_analyze(file):
    """Analyze uploaded file"""
    if file is None:
        return "❌ Please upload a file", None, None

    try:
        # Check if Uzbek format
        df_test = pd.read_excel(file.name, header=None)

        if preprocessor.is_uzbek_regional_format(df_test):
            df, report, mapping = preprocessor.process_uzbek_xlsx(file.name)
        else:
            # Standard format
            df = pd.read_excel(file.name) if file.name.endswith('.xlsx') else pd.read_csv(file.name)
            report = f"✅ Loaded {len(df)} records\n\nColumns: {', '.join(df.columns)}"
            mapping = {}

        return report, df, mapping

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


def generate_forecast(data, horizon):
    """Generate forecast"""
    if data is None:
        return "❌ No data loaded", None, None

    try:
        forecaster.load_data(data)
        predictions = forecaster.predict(horizon=int(horizon))

        # Create plot for first location
        first_loc = data['id'].iloc[0]
        fig = create_forecast_plot(data, predictions, first_loc)

        msg = f"✅ Generated {len(predictions)} predictions for {data['id'].nunique()} locations"

        return msg, predictions, fig

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


def load_llm():
    """Load LLM model"""
    global llm_loaded

    if llm_loaded:
        return "✅ LLM already loaded"

    try:
        llm_analyzer.load_model(use_4bit=True)
        llm_loaded = True
        return "✅ LLM loaded successfully"
    except Exception as e:
        return f"❌ Error loading LLM: {str(e)}"


def chat_with_llm(message, history, hist_data, pred_data):
    """Chat with LLM"""
    if not llm_loaded:
        return history + [[message, "❌ Please load LLM first (click 'Load LLM' button)"]]

    if hist_data is None:
        return history + [[message, "❌ Please load data and generate forecast first"]]

    try:
        # Load data into RAG if not done
        if rag_system.historical_data is None:
            rag_system.load_data(hist_data, pred_data, preprocessor.location_mapping)

        # Get response
        response = llm_analyzer.analyze(message)

        return history + [[message, response]]

    except Exception as e:
        return history + [[message, f"❌ Error: {str(e)}"]]


# Build Gradio interface
with gr.Blocks(title="Chrono_LLM_RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔮 Chrono_LLM_RAG: Time Series Forecasting + AI Analysis
    
    **Created by: Shohruh127**  
    **Repository: Chrono_LLM_RAG**  
    **Date: 2025-11-19 11:11:39 UTC**
    
    Upload your data, generate forecasts with Chronos-2, and chat with AI about results!
    """)

    # State
    data_state = gr.State(None)
    predictions_state = gr.State(None)
    mapping_state = gr.State({})

    with gr.Tabs():
        # Tab 1: Upload & Forecast
        with gr.Tab("📊 Forecast"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="📁 Upload Data (CSV/XLSX)")
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")

                    horizon_slider = gr.Slider(1, 20, value=4, step=1, label="Forecast Horizon")
                    forecast_btn = gr.Button("🚀 Generate Forecast", variant="primary")

                with gr.Column():
                    analysis_output = gr.Markdown()
                    forecast_output = gr.Markdown()
                    plot_output = gr.Plot()

        # Tab 2: AI Chat
        with gr.Tab("🤖 AI Analysis"):
            gr.Markdown("""
            ### Chat with AI about your forecasts
            
            Ask questions like:
            - "What's the trend for LOC_011_IND?"
            - "Compare industry vs agriculture"
            - "Sanoat va qishloq xo'jaligini taqqoslang" (Uzbek)
            """)

            load_llm_btn = gr.Button("🔄 Load LLM (First Time Only)", variant="secondary")
            llm_status = gr.Markdown()

            chatbot = gr.Chatbot(height=500, type="messages")
            msg_input = gr.Textbox(label="Ask a question", placeholder="e.g., What happened with LOC_011_IND?")
            
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat", variant="secondary")

    # Event handlers
    analyze_btn.click(
        upload_and_analyze,
        inputs=[file_input],
        outputs=[analysis_output, data_state, mapping_state]
    )

    forecast_btn.click(
        generate_forecast,
        inputs=[data_state, horizon_slider],
        outputs=[forecast_output, predictions_state, plot_output]
    )

    load_llm_btn.click(
        load_llm,
        outputs=[llm_status]
    )

    send_btn.click(
        chat_with_llm,
        inputs=[msg_input, chatbot, data_state, predictions_state],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        msg_input
    )

    msg_input.submit(
        chat_with_llm,
        inputs=[msg_input, chatbot, data_state, predictions_state],
        outputs=[chatbot]
    ).then(
        lambda: "",
        None,
        msg_input
    )

    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting Chrono_LLM_RAG Application")
    print("="*70)
    print(f"Created by: Shohruh127")
    print(f"Date: 2025-11-19 11:11:39 UTC")
    print("="*70)
    
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
