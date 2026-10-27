"""
MultiModal RAG - A Hybrid Multi-Modal Retrieval Augmented Generation Pipeline

Converts PDF, PPTX, and other documents into XML via multi-modal LLMs,
semantically chunks them, indexes into a vector database, and provides
a RAG pipeline for querying.

Pipeline Flow:
    Document (PDF/PPTX) → Page Images → LLM Transcription (XML) → 
    Semantic Chunking → Vector Indexing → RAG Query

Usage:
    from multimodal_rag import MultiModalRAGPipeline
    
    pipeline = MultiModalRAGPipeline()
    pipeline.process_document("document.pdf")
    answer = pipeline.query("Your question here")
"""

__version__ = "1.0.0"

from multimodal_rag.chunking import SemanticChunk, SemanticChunker
from multimodal_rag.ingestion import DocumentIngestor, PageImage
from multimodal_rag.indexing import SearchResult, VectorIndexer
from multimodal_rag.pipeline import MultiModalRAGPipeline
from multimodal_rag.rag import RAGEngine, RAGResult
from multimodal_rag.transcription import TranscribedDocument, XMLTranscriber
from multimodal_rag.utils.config import Config, get_config
from multimodal_rag.utils.logger import get_logger

__all__ = [
    # Main pipeline
    "MultiModalRAGPipeline",
    # Core classes
    "DocumentIngestor",
    "PageImage",
    "XMLTranscriber",
    "TranscribedDocument",
    "SemanticChunk",
    "SemanticChunker",
    "VectorIndexer",
    "SearchResult",
    "RAGEngine",
    "RAGResult",
    # Utilities
    "Config",
    "get_config",
    "get_logger",
]
