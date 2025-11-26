# =============================================================================
# USAGE_EXAMPLE.py - Comprehensive Usage Example
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

"""
Complete example of using the Sovereign Sidecar Selector Architecture.

This demonstrates:
1. Multi-sheet Excel ingestion
2. Sheet selection
3. Data preprocessing
4. Chronos-2 forecasting
5. PAL code generation (optional)
6. Google Drive persistence
7. PDF report generation
"""

import pandas as pd
from pathlib import Path

# Import components
from src.sidecar_engine import DictionaryIngestionEngine
from src.preprocessor import UzbekXLSXPreprocessor
from src.chronos_forecaster import ChronosForecaster
from src.code_generator import CodeGenerator
from src.report_generator import ReportGenerator
from src.drive_persistence import DrivePersistence


def main():
    print("="*70)
    print("🚀 Chrono_LLM_RAG - Sovereign Sidecar Selector Example")
    print("="*70)
    print()
    
    # =========================================================================
    # STEP 1: Initialize Components
    # =========================================================================
    print("📦 Initializing components...")
    
    sidecar = DictionaryIngestionEngine()
    preprocessor = UzbekXLSXPreprocessor()
    forecaster = ChronosForecaster()
    coder = CodeGenerator()  # Optional - for PAL pattern
    reporter = ReportGenerator()
    drive = DrivePersistence()
    
    print("✅ All components initialized\n")
    
    # =========================================================================
    # STEP 2: Load Multi-Sheet Excel File
    # =========================================================================
    print("📁 Loading Excel file with multiple sheets...")
    
    # Example file path (replace with your actual file)
    excel_file = "data/example_data.xlsx"
    
    if not Path(excel_file).exists():
        print(f"⚠️  Example file not found: {excel_file}")
        print("Creating a sample multi-sheet Excel file for demonstration...")
        
        # Create sample data
        df1 = pd.DataFrame({
            'Location': ['Tashkent', 'Samarkand', 'Bukhara'],
            '2020': [1000, 800, 600],
            '2021': [1100, 850, 650],
            '2022': [1200, 900, 700],
            '2023': [1300, 950, 750]
        })
        
        df2 = pd.DataFrame({
            'Region': ['Fergana', 'Namangan', 'Andijan'],
            '2020': [500, 400, 350],
            '2021': [550, 450, 400],
            '2022': [600, 500, 450],
            '2023': [650, 550, 500]
        })
        
        # Save sample file
        Path("data").mkdir(exist_ok=True)
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='Industry', index=False)
            df2.to_excel(writer, sheet_name='Agriculture', index=False)
        
        print(f"✅ Created sample file: {excel_file}\n")
    
    # Load with Sidecar Engine
    status = sidecar.load_excel_file(excel_file)
    
    print(f"✅ Status: {status['status']}")
    print(f"📊 Sheets loaded: {status['sheets_loaded']}")
    print(f"📋 Available sheets: {', '.join(status['sheet_names'])}\n")
    
    # =========================================================================
    # STEP 3: Select Sheet and Process
    # =========================================================================
    print("📑 Selecting sheet for analysis...")
    
    # Get available sheets
    available_sheets = sidecar.get_sheet_list()
    selected_sheet = available_sheets[0]  # Select first sheet
    
    print(f"Selected: {selected_sheet}\n")
    
    # Get sheet data
    sheet_data = sidecar.get_sheet_data(selected_sheet)
    
    print("📊 Sheet preview:")
    print(sheet_data.head())
    print()
    
    # Get sheet summary
    summary = sidecar.get_sheet_summary(selected_sheet)
    print("📝 Sheet summary:")
    print(summary)
    print()
    
    # =========================================================================
    # STEP 4: Preprocess Data (if Uzbek format)
    # =========================================================================
    print("🔄 Checking if data needs preprocessing...")
    
    if preprocessor.is_uzbek_regional_format(sheet_data):
        print("✅ Detected Uzbek regional format - processing...\n")
        
        # Save temporarily for processing
        temp_file = "/tmp/temp_sheet.xlsx"
        with pd.ExcelWriter(temp_file) as writer:
            sheet_data.to_excel(writer, sheet_name='Data', index=False)
        
        processed_data, report, mapping = preprocessor.process_uzbek_xlsx(temp_file)
        print(report)
    else:
        print("✅ Standard format - using as-is\n")
        processed_data = sheet_data
        mapping = {}
    
    # =========================================================================
    # STEP 5: Generate Forecasts with Chronos-2
    # =========================================================================
    print("🔮 Generating forecasts with Chronos-2...")
    
    # Note: This requires the Chronos model to be available
    # In a real environment, you would:
    # forecaster.load_model()
    # forecaster.load_data(processed_data)
    # predictions = forecaster.predict(horizon=4)
    
    print("⚠️  Skipping actual forecast (requires Chronos model)")
    print("In production, forecasts would be generated here.\n")
    
    # Create dummy predictions for demonstration
    predictions = pd.DataFrame({
        'timestamp': pd.date_range('2024', periods=4, freq='YS'),
        'predictions': [1400, 1500, 1600, 1700]
    })
    
    # =========================================================================
    # STEP 6: Optional - Use PAL Pattern for Questions
    # =========================================================================
    print("🤖 PAL Pattern Example (Code-as-Reasoning)...")
    print("Question: What's the average value in 2022?\n")
    
    # Note: This requires Qwen model
    # In production:
    # coder.load_model()
    # result = coder.answer_with_code(
    #     question="What's the average value in 2022?",
    #     data_context={'df': processed_data},
    #     context_description="DataFrame with years as columns"
    # )
    # print(f"Generated Code:\n{result['code']}\n")
    # print(f"Result:\n{result['result']}\n")
    
    print("⚠️  Skipping PAL (requires Qwen model)")
    print("In production, code would be generated and executed here.\n")
    
    # =========================================================================
    # STEP 7: Save to Google Drive (Day 2 Feature)
    # =========================================================================
    print("💾 Saving to Google Drive...")
    
    # Note: This requires Google Colab or Drive mounted
    try:
        save_path = drive.save_shadow_dataset(
            historical_data=processed_data,
            predictions=predictions,
            metadata={'mapping': mapping, 'sheet': selected_sheet},
            sheet_name=selected_sheet
        )
        
        if save_path:
            print(f"✅ Saved to: {save_path}\n")
        else:
            print("⚠️  Drive not available (requires Colab environment)\n")
    except Exception as e:
        print(f"⚠️  Could not save to Drive: {e}\n")
    
    # =========================================================================
    # STEP 8: Generate PDF Report (Day 2 Feature)
    # =========================================================================
    print("📄 Generating HTML/PDF Report...")
    
    metadata = sidecar.get_sheet_metadata(selected_sheet)
    warnings = sidecar.warnings if hasattr(sidecar, 'warnings') else []
    
    html_report = reporter.generate_html_report(
        sheet_name=selected_sheet,
        historical_data=processed_data,
        predictions=predictions,
        metadata=metadata,
        warnings=warnings
    )
    
    # Save report
    report_file = f"report_{selected_sheet}.html"
    saved_path = reporter.save_html_report(html_report, report_file)
    
    print(f"✅ Report saved: {saved_path}\n")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print()
    print("Summary:")
    print(f"  - Loaded: {status['sheets_loaded']} sheets")
    print(f"  - Processed: {selected_sheet}")
    print(f"  - Historical records: {len(processed_data)}")
    print(f"  - Predictions: {len(predictions)}")
    print(f"  - Report saved: {report_file}")
    print()
    print("Next steps:")
    print("  1. Open the HTML report in a browser")
    print("  2. Review the forecasts and statistics")
    print("  3. Use the PAL pattern for complex calculations")
    print("  4. Save to Google Drive for persistence")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
