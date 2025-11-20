# =============================================================================
# src/rag_system.py - RAG System with FAISS
# Created by: Shohruh127
# Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-11-19 11:11:39
# Current User's Login: Shohruh127
# Repository: Shohruh127/Chrono_LLM_RAG
# Repository ID: 1099678425
# =============================================================================

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import yaml
from pathlib import Path
import re


class RAGSystem:
    """Retrieval-Augmented Generation system with FAISS vector store"""

    def __init__(self, config_path: str = "configs/rag_config.yaml"):
        """
        Initialize RAG system
        
        Args:
            config_path: Path to configuration file
        """
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.embed_model = None
        self.index = None
        self.passages = []
        self.historical_data = None
        self.predictions_data = None
        self.location_mapping = {}
        self.name_to_id_map = {}

    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            'embeddings': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'device': 'cuda',
                'batch_size': 32
            },
            'retrieval': {
                'top_k': 5,
                'min_score': 0.3
            }
        }

    def load_embedder(self):
        """Load embedding model"""
        print(f"📥 Loading embedding model: {self.config['embeddings']['model_name']}")

        self.embed_model = SentenceTransformer(
            self.config['embeddings']['model_name']
        )

        print(f"✅ Embedder loaded")

    def load_data(self, hist_df: pd.DataFrame, pred_df: Optional[pd.DataFrame] = None,
                  loc_mapping: Optional[Dict] = None):
        """
        Load data for RAG
        
        Args:
            hist_df: Historical data
            pred_df: Predictions data
            loc_mapping: Location mapping dictionary
        """
        self.historical_data = hist_df
        self.predictions_data = pred_df
        self.location_mapping = loc_mapping or {}

        # Build name-to-id mapping
        self.name_to_id_map = {}
        for loc_id, info in self.location_mapping.items():
            loc_name = info.get('location_name', '').lower()
            if loc_name:
                self.name_to_id_map[loc_name] = loc_id
                clean_name = loc_name.replace(' ', '').replace('.', '').replace('sh', '')
                self.name_to_id_map[clean_name] = loc_id

        print(f"✅ Data loaded into RAG")
        print(f"   Historical: {len(hist_df):,} records")
        if pred_df is not None:
            print(f"   Predictions: {len(pred_df):,} records")
        print(f"   Location mappings: {len(self.location_mapping)}")

    def create_passages(self) -> List[Dict]:
        """Create text passages from data"""
        if self.historical_data is None:
            raise ValueError("No data loaded")

        passages = []

        for loc_id, group in self.historical_data.groupby('id'):
            group = group.sort_values('timestamp')

            # Historical values
            hist_text = ", ".join([
                f"{row['timestamp'].year}:{row['target']:.2f}"
                for _, row in group.iterrows()
            ])

            # Statistics
            stats_text = (
                f"mean {group['target'].mean():.2f}, "
                f"min {group['target'].min():.2f} (year {group[group['target'] == group['target'].min()]['timestamp'].dt.year.iloc[0]}), "
                f"max {group['target'].max():.2f} (year {group[group['target'] == group['target'].max()]['timestamp'].dt.year.iloc[0]})"
            )

            # Location info
            loc_info = self.location_mapping.get(loc_id, {})
            loc_name = loc_info.get('location_name', loc_id)
            category = loc_info.get('category_full', 'Unknown')

            # Predictions if available
            pred_text = ""
            if self.predictions_data is not None:
                pred_loc = self.predictions_data[self.predictions_data.index == loc_id]
                if len(pred_loc) > 0:
                    pred_text = "\nPredictions: " + ", ".join([
                        f"{row['timestamp'].year}:{row['predictions']:.2f}"
                        for _, row in pred_loc.iterrows()
                    ])

            passage_text = (
                f"Location: {loc_id} ({loc_name}, {category})\n"
                f"Historical: {hist_text}\n"
                f"Stats: {stats_text}"
                f"{pred_text}"
            )

            passages.append({
                'id': loc_id,
                'text': passage_text,
                'meta': loc_info
            })

        self.passages = passages
        print(f"✅ Created {len(passages)} passages")

        return passages

    def build_index(self):
        """Build FAISS index"""
        if not self.passages:
            self.create_passages()

        if self.embed_model is None:
            self.load_embedder()

        print(f"🔄 Building FAISS index...")

        texts = [p['text'] for p in self.passages]
        embeddings = self.embed_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.config['embeddings']['batch_size']
        )

        # Normalize for cosine similarity
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        # Create index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ Index built with {len(embeddings)} vectors (dimension: {dimension})")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant passages
        
        Args:
            query: User question
            top_k: Number of passages to retrieve
            
        Returns:
            List of retrieved passages with scores
        """
        if self.index is None:
            self.build_index()

        if top_k is None:
            top_k = self.config['retrieval']['top_k']

        # Embed query
        query_emb = self.embed_model.encode([query])
        query_emb = query_emb.astype('float32')
        faiss.normalize_L2(query_emb)

        # Search
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        min_score = self.config['retrieval']['min_score']

        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score:
                passage = self.passages[idx].copy()
                passage['score'] = float(score)
                results.append(passage)

        return results

    def extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract location ID from query"""
        query_lower = query.lower()

        # Try LOC_XXX pattern
        patterns = [
            r'LOC[_\s]?(\d+)(?:[_\s]?([A-Z]{3}))?',
            r'location\s+(\d+)',
            r'loc\s+(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                num = match.group(1).zfill(3)
                base_id = f"LOC_{num}"

                if len(match.groups()) > 1 and match.group(2):
                    return f"{base_id}_{match.group(2).upper()}"

                if self.historical_data is not None:
                    matching = [loc for loc in self.historical_data['id'].unique()
                               if loc.startswith(base_id)]
                    if matching:
                        return matching[0]

                return base_id

        # Try location name matching
        for name, loc_id in self.name_to_id_map.items():
            if name in query_lower:
                return loc_id

        return None

    def build_context_for_llm(self, question: str) -> str:
        """
        Build comprehensive context for LLM
        
        Args:
            question: User question
            
        Returns:
            Formatted context string
        """
        if self.historical_data is None:
            return "No data available."

        hist = self.historical_data
        pred = self.predictions_data

        context_parts = []

        # Extract location
        location_id = self.extract_location_from_query(question)

        context_parts.append("="*70)
        context_parts.append("DATABASE CONTEXT - USE THIS DATA FOR YOUR ANSWER")
        context_parts.append("="*70)

        # Dataset overview
        context_parts.append(f"\nDATASET OVERVIEW:")
        context_parts.append(f"  Total locations: {hist['id'].nunique()}")
        context_parts.append(f"  Historical period: {hist['timestamp'].dt.year.min()}-{hist['timestamp'].dt.year.max()}")
        context_parts.append(f"  Total records: {len(hist):,}")
        context_parts.append(f"  Overall mean: {hist['target'].mean():.2f}")

        # Location-specific data
        if location_id:
            loc_data = hist[hist['id'] == location_id].sort_values('timestamp')
            if len(loc_data) > 0:
                loc_info = self.location_mapping.get(location_id, {})
                loc_name = loc_info.get('location_name', 'Unknown')
                category = loc_info.get('category_full', 'Unknown')

                context_parts.append(f"\nLOCATION: {location_id}")
                context_parts.append(f"  Name: {loc_name}")
                context_parts.append(f"  Category: {category}")

                context_parts.append(f"\nHISTORICAL STATISTICS:")
                context_parts.append(f"  Mean: {loc_data['target'].mean():.2f}")
                context_parts.append(f"  Min: {loc_data['target'].min():.2f} (year {loc_data[loc_data['target'] == loc_data['target'].min()]['timestamp'].dt.year.iloc[0]})")
                context_parts.append(f"  Max: {loc_data['target'].max():.2f} (year {loc_data[loc_data['target'] == loc_data['target'].max()]['timestamp'].dt.year.iloc[0]})")

                context_parts.append(f"\nYEAR-BY-YEAR VALUES:")
                for _, row in loc_data.iterrows():
                    context_parts.append(f"  {row['timestamp'].year}: {row['target']:.2f}")

                # Trend
                first_val = loc_data.iloc[0]['target']
                last_val = loc_data.iloc[-1]['target']
                change = last_val - first_val
                pct = (change / first_val * 100) if first_val != 0 else 0

                context_parts.append(f"\nTREND:")
                context_parts.append(f"  Start ({loc_data.iloc[0]['timestamp'].year}): {first_val:.2f}")
                context_parts.append(f"  End ({loc_data.iloc[-1]['timestamp'].year}): {last_val:.2f}")
                context_parts.append(f"  Change: {change:+.2f} ({pct:+.1f}%)")

                # Predictions
                if pred is not None and len(pred) > 0:
                    pred_loc = pred[pred.index == location_id].sort_values('timestamp')
                    if len(pred_loc) > 0:
                        context_parts.append(f"\nPREDICTIONS:")
                        context_parts.append(f"  Period: {pred_loc['timestamp'].dt.year.min()}-{pred_loc['timestamp'].dt.year.max()}")

                        context_parts.append(f"\nYEAR-BY-YEAR PREDICTIONS:")
                        for _, row in pred_loc.iterrows():
                            context_parts.append(f"  {row['timestamp'].year}: {row['predictions']:.2f}")

        # Category comparisons
        question_lower = question.lower()
        if any(word in question_lower for word in ['compare', 'taqqos', 'industry', 'agriculture', 'sanoat', 'qishloq']):
            ind_data = hist[hist['id'].str.contains('_IND', na=False)]
            agr_data = hist[hist['id'].str.contains('_AGR', na=False)]

            if len(ind_data) > 0:
                context_parts.append(f"\nINDUSTRY STATISTICS:")
                context_parts.append(f"  Locations: {ind_data['id'].nunique()}")
                context_parts.append(f"  Mean: {ind_data['target'].mean():.2f}")
                context_parts.append(f"  Total: {ind_data['target'].sum():.2f}")

            if len(agr_data) > 0:
                context_parts.append(f"\nAGRICULTURE STATISTICS:")
                context_parts.append(f"  Locations: {agr_data['id'].nunique()}")
                context_parts.append(f"  Mean: {agr_data['target'].mean():.2f}")
                context_parts.append(f"  Total: {agr_data['target'].sum():.2f}")

        context_parts.append("\n" + "="*70)
        context_parts.append("END OF DATABASE CONTEXT")
        context_parts.append("="*70)

        return "\n".join(context_parts)

    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index built yet")

        faiss.write_index(self.index, filepath)
        print(f"💾 Saved FAISS index to {filepath}")

    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(filepath)
        print(f"📥 Loaded FAISS index from {filepath}")
