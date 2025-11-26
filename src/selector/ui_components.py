# =============================================================================
# src/selector/ui_components.py - Gradio UI Components for Sheet Selection
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Phase 4: Selector Architecture
# =============================================================================

import gradio as gr
import pandas as pd
from typing import Tuple, Optional
from .sheet_manager import SheetManager
from .context_propagator import ContextPropagator


def create_sheet_dropdown(sheet_manager: SheetManager) -> gr.Dropdown:
    """
    Create dropdown with available sheets.
    
    Args:
        sheet_manager: SheetManager instance
        
    Returns:
        Gradio Dropdown component
    """
    sheets = sheet_manager.list_sheets()
    
    # Format choices as "Sheet Name (Domain) - rows x cols"
    choices = [
        f"{s['name']} ({s['domain']}) - {s['rows']} rows × {s['cols']} cols"
        for s in sheets
    ]
    
    # Store sheet names for value retrieval
    sheet_names = [s['name'] for s in sheets]
    
    return gr.Dropdown(
        choices=choices,
        label="📋 Select Data Domain",
        info="Choose a specific economic domain to analyze",
        value=choices[0] if choices else None
    ), sheet_names


def create_sheet_preview(sheet_manager: SheetManager, sheet_name: str, rows: int = 5) -> pd.DataFrame:
    """
    Create preview table component.
    
    Args:
        sheet_manager: SheetManager instance
        sheet_name: Name of sheet to preview
        rows: Number of rows to preview (default: 5)
        
    Returns:
        DataFrame for preview
    """
    try:
        preview_df = sheet_manager.get_sheet_preview(sheet_name, rows=rows)
        return preview_df
    except Exception as e:
        print(f"Error creating preview: {e}")
        return pd.DataFrame({"Error": [str(e)]})


def create_domain_badge(context: ContextPropagator) -> str:
    """
    Show current domain as colored badge.
    
    Args:
        context: ContextPropagator instance
        
    Returns:
        HTML string for domain badge
    """
    if not context.has_context():
        return """
        <div style="padding: 10px; border-radius: 5px; background-color: #f0f0f0; text-align: center;">
            <span style="color: #666;">No domain selected</span>
        </div>
        """
    
    domain = context.get_domain()
    sheet_name = context.get_sheet_name()
    ctx = context.get_context()
    
    # Domain-specific colors
    domain_colors = {
        "Agriculture": "#4CAF50",
        "Industry": "#2196F3",
        "Demography": "#9C27B0",
        "Trade": "#FF9800",
        "Transport": "#795548",
        "Construction": "#607D8B",
        "Finance": "#009688",
        "Education": "#3F51B5",
        "Healthcare": "#E91E63",
        "Culture": "#673AB7",
        "Sports": "#FF5722",
        "Tourism": "#00BCD4",
        "Unknown": "#9E9E9E"
    }
    
    color = domain_colors.get(domain, "#9E9E9E")
    
    badge_html = f"""
    <div style="padding: 15px; border-radius: 8px; background: linear-gradient(135deg, {color} 0%, {color}dd 100%); box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <div style="color: white; font-size: 18px; font-weight: bold; margin-bottom: 5px;">
            🏷️ Current Context: {domain}
        </div>
        <div style="color: rgba(255,255,255,0.9); font-size: 14px;">
            📊 Sheet: {sheet_name}
        </div>
        <div style="color: rgba(255,255,255,0.8); font-size: 12px; margin-top: 5px;">
            📐 {ctx['rows']} rows × {ctx['cols']} columns
        </div>
    </div>
    """
    
    return badge_html


def create_selector_interface(filepath: str) -> Tuple[gr.Blocks, SheetManager, ContextPropagator]:
    """
    Full selector interface with dropdown, preview, and confirm button.
    
    Args:
        filepath: Path to Excel file
        
    Returns:
        Tuple of (Gradio Blocks interface, SheetManager, ContextPropagator)
    """
    manager = SheetManager(filepath)
    context = ContextPropagator()
    
    with gr.Blocks() as interface:
        gr.Markdown("## 📋 Sheet Selector - Domain-Specific Analysis")
        
        # Dropdown and sheet names state
        dropdown, sheet_names = create_sheet_dropdown(manager)
        sheet_names_state = gr.State(sheet_names)
        
        # Preview section
        with gr.Row():
            with gr.Column(scale=2):
                preview_output = gr.DataFrame(
                    label="👁️ Preview (first 5 rows)",
                    interactive=False,
                    wrap=True
                )
            
            with gr.Column(scale=1):
                domain_badge_output = gr.HTML(
                    value=create_domain_badge(context),
                    label="Current Context"
                )
        
        # Action buttons
        with gr.Row():
            preview_btn = gr.Button("👁️ Preview Selected Sheet", variant="secondary")
            confirm_btn = gr.Button("✅ Confirm Selection", variant="primary")
        
        # Status message
        status_output = gr.Markdown()
        
        # Event handlers
        def on_preview(dropdown_value, sheet_names_list):
            if dropdown_value is None:
                return pd.DataFrame({"Message": ["Please select a sheet"]}), ""
            
            # Extract sheet name from dropdown value
            idx = [i for i, choice in enumerate([
                f"{manager.list_sheets()[i]['name']} ({manager.list_sheets()[i]['domain']}) - {manager.list_sheets()[i]['rows']} rows × {manager.list_sheets()[i]['cols']} cols"
                for i in range(len(sheet_names_list))
            ]) if dropdown_value.startswith(manager.list_sheets()[i]['name'])][0]
            
            sheet_name = sheet_names_list[idx]
            preview_df = create_sheet_preview(manager, sheet_name)
            return preview_df, f"**Preview of:** {sheet_name}"
        
        def on_confirm(dropdown_value, sheet_names_list):
            if dropdown_value is None:
                return create_domain_badge(context), "❌ Please select a sheet first"
            
            # Extract sheet name from dropdown value
            sheets = manager.list_sheets()
            matching_sheet = None
            for sheet in sheets:
                if dropdown_value.startswith(sheet['name']):
                    matching_sheet = sheet['name']
                    break
            
            if matching_sheet is None:
                return create_domain_badge(context), "❌ Invalid sheet selection"
            
            # Load sheet and set context
            df = manager.select_sheet(matching_sheet)
            domain = manager.detect_domain(matching_sheet)
            context.set_context(matching_sheet, df, domain)
            
            badge = create_domain_badge(context)
            msg = f"✅ **Context set successfully!**\n\n- **Domain:** {domain}\n- **Sheet:** {matching_sheet}\n- **Data:** {len(df)} rows × {len(df.columns)} columns"
            
            return badge, msg
        
        preview_btn.click(
            on_preview,
            inputs=[dropdown, sheet_names_state],
            outputs=[preview_output, status_output]
        )
        
        confirm_btn.click(
            on_confirm,
            inputs=[dropdown, sheet_names_state],
            outputs=[domain_badge_output, status_output]
        )
        
        # Auto-preview first sheet on load
        interface.load(
            on_preview,
            inputs=[dropdown, sheet_names_state],
            outputs=[preview_output, status_output]
        )
    
    return interface, manager, context


def create_compact_selector(filepath: str) -> Tuple[gr.Dropdown, gr.DataFrame, gr.HTML, gr.Button, SheetManager, ContextPropagator]:
    """
    Create compact selector components for embedding in existing interface.
    
    Args:
        filepath: Path to Excel file
        
    Returns:
        Tuple of (dropdown, preview_df, badge_html, confirm_btn, manager, context)
    """
    manager = SheetManager(filepath)
    context = ContextPropagator()
    
    dropdown, sheet_names = create_sheet_dropdown(manager)
    preview_df = gr.DataFrame(label="👁️ Preview", interactive=False)
    badge_html = gr.HTML(value=create_domain_badge(context))
    confirm_btn = gr.Button("✅ Confirm Domain Selection", variant="primary")
    
    return dropdown, preview_df, badge_html, confirm_btn, manager, context
