# rag/vector_store.py
"""
Vector Store Database Manager.
Handles indexing, matching, and physical storage.
"""

import os
import pickle
from typing import List, Dict, Any
import numpy as np
from config import settings

class MonsoonVectorDB:
    """Manages local FAISS vector stores, persisting indexing files and source metadata structures."""
    
    def __init__(self):
        self.index_path = settings.VECTOR_DB_DIR / "faiss.index"
        self.meta_path = settings.VECTOR_DB_DIR / "metadata.pkl"
        self.dimension = 384
        self.metadata_store: List[Dict[str, Any]] = []
        
        # Initialize raw FAISS object
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            # Simple matrix similarity search fallback structure
            self.index = None
            self.fallback_vectors: List[np.ndarray] = []

        # Autoload indices if they exist
        self.load_index()

    def populate_index(self, file_path: str):
        """Reads, chunks, embeds, and loads a file into the vector workspace."""
        from rag.embed_pipeline import EmbedPipeline
        pipeline = EmbedPipeline()
        
        text_content = pipeline.read_text(file_path)
        chunks = pipeline.chunk_text(text_content)
        embeddings = np.array(pipeline.generate_embeddings(chunks), dtype=np.float32)
        
        filename = os.path.basename(file_path)
        
        for idx, chunk in enumerate(chunks):
            self.metadata_store.append({
                "source": filename,
                "chunk_id": idx,
                "context": chunk
            })
            
        if self.index is not None:
            self.index.add(embeddings)
        else:
            for emb in embeddings:
                self.fallback_vectors.append(emb)
                
        self.save_index()

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Performs cosine-similarity database retrieval on query string inputs."""
        from rag.embed_pipeline import EmbedPipeline
        pipeline = EmbedPipeline()
        
        query_emb = np.array(pipeline.generate_embeddings([query])[0], dtype=np.float32).reshape(1, -1)
        
        if len(self.metadata_store) == 0:
            return []
            
        if self.index is not None:
            distances, indices = self.index.search(query_emb, min(top_k, len(self.metadata_store)))
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata_store):
                    results.append(self.metadata_store[idx])
            return results
        else:
            # Execute manual matrix dot-product similarity comparison
            similarities = [np.dot(query_emb[0], v) for v in self.fallback_vectors]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.metadata_store[idx] for idx in top_indices if idx < len(self.metadata_store)]

    def save_index(self):
        """Saves current database binary files to the storage volume."""
        settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(self.index_path))
        else:
            with open(settings.VECTOR_DB_DIR / "fallback.npy", "wb") as f:
                np.save(f, np.array(self.fallback_vectors))
                
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)

    def load_index(self):
        """Loads saved database files from disk."""
        if not self.meta_path.exists():
            return
            
        with open(self.meta_path, "rb") as f:
            self.metadata_store = pickle.load(f)
            
        if self.index_path.exists() and self.index is not None:
            import faiss
            self.index = faiss.read_index(str(self.index_path))
        elif (settings.VECTOR_DB_DIR / "fallback.npy").exists():
            with open(settings.VECTOR_DB_DIR / "fallback.npy", "rb") as f:
                self.fallback_vectors = list(np.load(f))