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
from src.selector import SheetManager, ContextPropagator


# Initialize components
preprocessor = UzbekXLSXPreprocessor()
forecaster = ChronosForecaster()
rag_system = RAGSystem()
llm_analyzer = LLMAnalyzer(rag_system=rag_system)

# Global state
llm_loaded = False
sheet_manager = None
context_propagator = ContextPropagator()


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
    """Analyze uploaded file and detect if multi-sheet"""
    global sheet_manager
    
    if file is None:
        return "❌ Please upload a file", None, None, None, None
    
    try:
        # Check if it's an Excel file with multiple sheets
        if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            try:
                sheet_manager = SheetManager(file.name)
                sheets = sheet_manager.list_sheets()
                
                # If multiple sheets, prompt for selection
                if len(sheets) > 1:
                    sheets_info = "\n".join([
                        f"- **{s['name']}** ({s['domain']}) - {s['rows']} rows × {s['cols']} cols"
                        for s in sheets
                    ])
                    report = f"📋 **Multi-sheet Excel file detected!**\n\n{len(sheets)} sheets found:\n{sheets_info}\n\n👇 **Please select a domain from the dropdown below**"
                    
                    # Create dropdown choices
                    choices = [
                        f"{s['name']} ({s['domain']}) - {s['rows']} rows × {s['cols']} cols"
                        for s in sheets
                    ]
                    
                    return report, None, None, gr.update(choices=choices, visible=True, value=choices[0] if choices else None), sheets
                else:
                    # Single sheet, load directly
                    df = sheet_manager.select_sheet(sheets[0]['name'])
                    domain = sheet_manager.detect_domain(sheets[0]['name'])
                    context_propagator.set_context(sheets[0]['name'], df, domain)
                    
                    report = f"✅ Loaded single sheet: **{sheets[0]['name']}** ({domain})\n\n{len(df)} rows × {len(df.columns)} columns"
                    return report, df, None, gr.update(visible=False), sheets
                    
            except Exception as e:
                # If sheet detection fails, try old format
                print(f"Sheet detection failed: {e}, trying old format")
                pass
        
        # Old workflow for non-multi-sheet files
        df_test = pd.read_excel(file.name, header=None)
        
        if preprocessor.is_uzbek_regional_format(df_test):
            df, report, mapping = preprocessor.process_uzbek_xlsx(file.name)
        else:
            # Standard format
            df = pd.read_excel(file.name) if file.name.endswith('.xlsx') else pd.read_csv(file.name)
            report = f"✅ Loaded {len(df)} records\n\nColumns: {', '.join(df.columns)}"
            mapping = {}
        
        return report, df, mapping, gr.update(visible=False), None
    
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, gr.update(visible=False), None


def select_domain(dropdown_value, sheets_data):
    """Handle domain selection from dropdown"""
    global sheet_manager
    
    if dropdown_value is None or sheets_data is None:
        return "❌ Please select a domain", None, gr.update(value="")
    
    try:
        # Extract sheet name from dropdown value
        matching_sheet = None
        for sheet in sheets_data:
            if dropdown_value.startswith(sheet['name']):
                matching_sheet = sheet['name']
                break
        
        if matching_sheet is None:
            return "❌ Invalid selection", None, gr.update(value="")
        
        # Load selected sheet
        df = sheet_manager.select_sheet(matching_sheet)
        domain = sheet_manager.detect_domain(matching_sheet)
        
        # Set context
        context_propagator.set_context(matching_sheet, df, domain)
        
        # Create domain badge
        badge_html = f"""
        <div style="padding: 15px; border-radius: 8px; background: linear-gradient(135deg, #4CAF50 0%, #4CAF50dd 100%); box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <div style="color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;">
                🏷️ Selected Domain: {domain}
            </div>
            <div style="color: rgba(255,255,255,0.9); font-size: 14px;">
                📊 Sheet: {matching_sheet}
            </div>
            <div style="color: rgba(255,255,255,0.8); font-size: 12px; margin-top: 5px;">
                📐 {len(df)} rows × {len(df.columns)} columns
            </div>
        </div>
        """
        
        msg = f"✅ **Domain selected successfully!**\n\n- **Domain:** {domain}\n- **Sheet:** {matching_sheet}\n- **Data:** {len(df)} rows × {len(df.columns)} columns\n\n🚀 You can now proceed to forecasting!"
        
        return msg, df, gr.update(value=badge_html)
    
    except Exception as e:
        return f"❌ Error selecting domain: {str(e)}", None, gr.update(value="")


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
        
        # Add domain context to the message if available
        if context_propagator.has_context():
            domain_prompt = context_propagator.get_domain_prompt()
            enhanced_message = f"{domain_prompt}\n\nUser Question: {message}"
            response = llm_analyzer.analyze(enhanced_message)
        else:
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
    sheets_state = gr.State(None)

    with gr.Tabs():
        # Tab 1: Upload & Forecast
        with gr.Tab("📊 Forecast"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="📁 Upload Data (CSV/XLSX)")
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    # Domain selector (hidden by default)
                    domain_dropdown = gr.Dropdown(
                        label="📋 Select Domain",
                        info="Choose specific economic domain to analyze",
                        visible=False
                    )
                    select_domain_btn = gr.Button("✅ Confirm Domain Selection", variant="primary", visible=False)

                    horizon_slider = gr.Slider(1, 20, value=4, step=1, label="Forecast Horizon")
                    forecast_btn = gr.Button("🚀 Generate Forecast", variant="primary")

                with gr.Column():
                    analysis_output = gr.Markdown()
                    domain_badge = gr.HTML()
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
        outputs=[analysis_output, data_state, mapping_state, domain_dropdown, sheets_state]
    )
    
    # Show dropdown when it has choices
    domain_dropdown.change(
        lambda x: gr.update(visible=True) if x is not None else gr.update(visible=False),
        inputs=[domain_dropdown],
        outputs=[select_domain_btn]
    )
    
    select_domain_btn.click(
        select_domain,
        inputs=[domain_dropdown, sheets_state],
        outputs=[analysis_output, data_state, domain_badge]
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
