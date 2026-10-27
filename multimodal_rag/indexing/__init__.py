"""
VectorIndexer - Vector database indexing for semantic chunks.

Stores semantically chunked documents into a vector database (ChromaDB,
FAISS, or Pinecone) with configurable embedding models for efficient
semantic search and retrieval.
"""

from multimodal_rag.indexing.indexing import SearchResult, VectorIndexer

__all__ = ["SearchResult", "VectorIndexer"]
