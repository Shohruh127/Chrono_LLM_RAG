# =============================================================================
# src/report_generator.py - PDF Report Generation Module
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, List
import io
import base64
import plotly.graph_objects as go


class ReportGenerator:
    """
    Generate PDF reports for selected sheets with forecasts and metadata.
    
    Features:
    - Summary statistics
    - Forecast visualizations
    - Tail warnings and metadata
    - Downloadable PDF format
    """
    
    def __init__(self):
        self.report_data = {}
    
    def generate_html_report(self,
                           sheet_name: str,
                           historical_data: pd.DataFrame,
                           predictions: Optional[pd.DataFrame] = None,
                           metadata: Optional[Dict] = None,
                           warnings: Optional[List[str]] = None) -> str:
        """
        Generate HTML report that can be converted to PDF
        
        Args:
            sheet_name: Name of the sheet
            historical_data: Historical DataFrame
            predictions: Predictions DataFrame
            metadata: Sheet metadata
            warnings: List of warnings/notes
            
        Returns:
            HTML string
        """
        html_parts = []
        
        # Header
        html_parts.append(self._generate_header(sheet_name))
        
        # Summary Section
        html_parts.append(self._generate_summary(historical_data, predictions))
        
        # Statistics Section
        html_parts.append(self._generate_statistics(historical_data))
        
        # Forecast Chart (if available)
        if predictions is not None:
            html_parts.append(self._generate_forecast_chart(historical_data, predictions))
        
        # Metadata Section
        if metadata:
            html_parts.append(self._generate_metadata_section(metadata))
        
        # Warnings Section
        if warnings:
            html_parts.append(self._generate_warnings_section(warnings))
        
        # Footer
        html_parts.append(self._generate_footer())
        
        # Combine all parts
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{sheet_name} - Report</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    {''.join(html_parts)}
</body>
</html>
"""
        return html
    
    def _get_css(self) -> str:
        """Get CSS styles for the report"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 32px;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 16px;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .metadata {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        """
    
    def _generate_header(self, sheet_name: str) -> str:
        """Generate report header"""
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        return f"""
        <div class="header">
            <h1>📊 {sheet_name}</h1>
            <p>Chrono_LLM_RAG Analysis Report</p>
            <p>Generated: {timestamp}</p>
        </div>
        """
    
    def _generate_summary(self, historical: pd.DataFrame, predictions: Optional[pd.DataFrame]) -> str:
        """Generate summary section"""
        hist_years = historical['timestamp'].dt.year if 'timestamp' in historical.columns else []
        pred_years = predictions['timestamp'].dt.year if predictions is not None and 'timestamp' in predictions.columns else []
        
        hist_range = f"{hist_years.min()}-{hist_years.max()}" if len(hist_years) > 0 else "N/A"
        pred_range = f"{pred_years.min()}-{pred_years.max()}" if len(pred_years) > 0 else "N/A"
        
        return f"""
        <div class="section">
            <h2>📋 Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Historical Records</div>
                    <div class="stat-value">{len(historical):,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Historical Period</div>
                    <div class="stat-value">{hist_range}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Predictions</div>
                    <div class="stat-value">{len(predictions) if predictions is not None else 0:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Forecast Period</div>
                    <div class="stat-value">{pred_range}</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_statistics(self, historical: pd.DataFrame) -> str:
        """Generate statistics section"""
        if 'target' not in historical.columns:
            return ""
        
        stats = {
            'Mean': historical['target'].mean(),
            'Median': historical['target'].median(),
            'Std Dev': historical['target'].std(),
            'Min': historical['target'].min(),
            'Max': historical['target'].max(),
            'Range': historical['target'].max() - historical['target'].min()
        }
        
        stats_html = '<div class="stats-grid">'
        for label, value in stats.items():
            stats_html += f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value:.2f}</div>
            </div>
            """
        stats_html += '</div>'
        
        return f"""
        <div class="section">
            <h2>📈 Statistical Analysis</h2>
            {stats_html}
        </div>
        """
    
    def _generate_forecast_chart(self, historical: pd.DataFrame, predictions: pd.DataFrame) -> str:
        """Generate forecast chart as embedded image"""
        # Create Plotly chart
        fig = go.Figure()
        
        # Historical data
        if 'timestamp' in historical.columns and 'target' in historical.columns:
            fig.add_trace(go.Scatter(
                x=historical['timestamp'],
                y=historical['target'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f77b4', width=2)
            ))
        
        # Predictions
        if 'timestamp' in predictions.columns and 'predictions' in predictions.columns:
            fig.add_trace(go.Scatter(
                x=predictions['timestamp'],
                y=predictions['predictions'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title='Historical Data & Forecast',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        # Convert to HTML
        chart_html = fig.to_html(include_plotlyjs='cdn', div_id='forecast-chart')
        
        return f"""
        <div class="section">
            <h2>🔮 Forecast Visualization</h2>
            <div class="chart">
                {chart_html}
            </div>
        </div>
        """
    
    def _generate_metadata_section(self, metadata: Dict) -> str:
        """Generate metadata section"""
        meta_lines = []
        for key, value in metadata.items():
            if isinstance(value, dict):
                meta_lines.append(f"<strong>{key}:</strong>")
                for k, v in value.items():
                    meta_lines.append(f"&nbsp;&nbsp;• {k}: {v}")
            elif isinstance(value, list):
                meta_lines.append(f"<strong>{key}:</strong> {', '.join(map(str, value[:5]))}")
            else:
                meta_lines.append(f"<strong>{key}:</strong> {value}")
        
        return f"""
        <div class="section">
            <h2>📝 Metadata</h2>
            <div class="metadata">
                {'<br>'.join(meta_lines)}
            </div>
        </div>
        """
    
    def _generate_warnings_section(self, warnings: List[str]) -> str:
        """Generate warnings section"""
        warning_items = ''.join([f'<div class="warning">⚠️ {w}</div>' for w in warnings])
        
        return f"""
        <div class="section">
            <h2>⚠️ Warnings & Notes</h2>
            {warning_items}
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate report footer"""
        return f"""
        <div class="footer">
            <p><strong>Chrono_LLM_RAG</strong></p>
            <p>Created by: Shohruh127 | Repository: Chrono_LLM_RAG</p>
            <p>Sovereign Sidecar Selector Architecture v2.0</p>
        </div>
        """
    
    def save_html_report(self, html_content: str, filename: str) -> str:
        """
        Save HTML report to file
        
        Args:
            html_content: HTML string
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report saved: {filename}")
        return filename


# Initialize report generator
report_generator = ReportGenerator()

print("✅ PDF Report Generator ready!")
print(f"Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: Shohruh127")
