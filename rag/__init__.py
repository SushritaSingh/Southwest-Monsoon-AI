# rag/__init__.py
"""
RAG package initializer. Exposes vector database, ingestion pipelines, 
and context retrievers for the generative climate assistant.
"""

from rag.vector_store import MonsoonVectorDB
from rag.embed_pipeline import EmbedPipeline, IngestionPipeline
from rag.retriever import MonsoonRetriever

__all__ = [
    "MonsoonVectorDB",
    "EmbedPipeline",
    "IngestionPipeline",
    "MonsoonRetriever"
]