# =============================================================================
# src/drive_persistence.py - Google Drive Persistence Module
# Created by: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# =============================================================================

import pandas as pd
import pickle
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
import json


class DrivePersistence:
    """
    Google Drive persistence for Shadow Dataset.
    Saves processed data to prevent loss on runtime restart.
    """
    
    def __init__(self, base_path: str = "/content/drive/MyDrive/Chrono_LLM_RAG"):
        """
        Initialize Drive Persistence
        
        Args:
            base_path: Base path in Google Drive for storage
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.models_path = self.base_path / "models"
        self.cache_path = self.base_path / "cache"
        
        # Create directories if using Colab
        if self._is_colab():
            self._mount_drive()
            self._create_directories()
    
    def _is_colab(self) -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except:
            return False
    
    def _mount_drive(self):
        """Mount Google Drive in Colab"""
        if self._is_colab():
            try:
                from google.colab import drive
                drive.mount('/content/drive', force_remount=False)
                print("✅ Google Drive mounted")
            except Exception as e:
                print(f"⚠️ Could not mount Drive: {e}")
    
    def _create_directories(self):
        """Create directory structure in Drive"""
        for path in [self.data_path, self.models_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory structure at {self.base_path}")
    
    def save_shadow_dataset(self, 
                          historical_data: pd.DataFrame,
                          predictions: Optional[pd.DataFrame] = None,
                          metadata: Optional[Dict] = None,
                          sheet_name: str = "default") -> str:
        """
        Save processed "Shadow Dataset" to Drive
        
        Args:
            historical_data: Processed historical DataFrame
            predictions: Forecast predictions DataFrame
            metadata: Additional metadata (location mapping, etc.)
            sheet_name: Name of the sheet/dataset
            
        Returns:
            Path where data was saved
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        save_dir = self.data_path / f"{sheet_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save historical data
            hist_path = save_dir / "historical.parquet"
            historical_data.to_parquet(hist_path, index=False)
            print(f"✅ Saved historical data: {hist_path}")
            
            # Save predictions if available
            if predictions is not None:
                pred_path = save_dir / "predictions.parquet"
                predictions.to_parquet(pred_path)
                print(f"✅ Saved predictions: {pred_path}")
            
            # Save metadata
            if metadata is not None:
                meta_path = save_dir / "metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                print(f"✅ Saved metadata: {meta_path}")
            
            # Save manifest
            manifest = {
                'sheet_name': sheet_name,
                'timestamp': timestamp,
                'historical_records': len(historical_data),
                'predictions_records': len(predictions) if predictions is not None else 0,
                'has_metadata': metadata is not None,
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            manifest_path = save_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"✅ Saved manifest: {manifest_path}")
            
            return str(save_dir)
            
        except Exception as e:
            print(f"❌ Error saving to Drive: {e}")
            return ""
    
    def load_shadow_dataset(self, dataset_path: str) -> Dict:
        """
        Load Shadow Dataset from Drive
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary with 'historical', 'predictions', 'metadata'
        """
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        result = {}
        
        try:
            # Load historical data
            hist_path = dataset_dir / "historical.parquet"
            if hist_path.exists():
                result['historical'] = pd.read_parquet(hist_path)
                print(f"✅ Loaded historical data: {len(result['historical'])} records")
            
            # Load predictions
            pred_path = dataset_dir / "predictions.parquet"
            if pred_path.exists():
                result['predictions'] = pd.read_parquet(pred_path)
                print(f"✅ Loaded predictions: {len(result['predictions'])} records")
            
            # Load metadata
            meta_path = dataset_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    result['metadata'] = json.load(f)
                print(f"✅ Loaded metadata")
            
            # Load manifest
            manifest_path = dataset_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    result['manifest'] = json.load(f)
                print(f"✅ Loaded manifest")
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading from Drive: {e}")
            return {}
    
    def list_saved_datasets(self) -> list:
        """
        List all saved datasets in Drive
        
        Returns:
            List of dataset directories with metadata
        """
        if not self.data_path.exists():
            return []
        
        datasets = []
        
        for item in self.data_path.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        manifest['path'] = str(item)
                        datasets.append(manifest)
                    except:
                        pass
        
        # Sort by timestamp (newest first)
        datasets.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return datasets
    
    def cleanup_old_datasets(self, keep_last: int = 5):
        """
        Remove old datasets, keeping only the most recent ones
        
        Args:
            keep_last: Number of recent datasets to keep per sheet
        """
        datasets = self.list_saved_datasets()
        
        # Group by sheet name
        by_sheet = {}
        for ds in datasets:
            sheet = ds.get('sheet_name', 'unknown')
            if sheet not in by_sheet:
                by_sheet[sheet] = []
            by_sheet[sheet].append(ds)
        
        # Remove old ones
        for sheet, sheet_datasets in by_sheet.items():
            if len(sheet_datasets) > keep_last:
                to_remove = sheet_datasets[keep_last:]
                for ds in to_remove:
                    try:
                        import shutil
                        shutil.rmtree(ds['path'])
                        print(f"🗑️ Removed old dataset: {ds['path']}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {ds['path']}: {e}")


# Initialize persistence
drive_persistence = DrivePersistence()

print("✅ Google Drive Persistence ready!")
print(f"Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current User's Login: Shohruh127")
