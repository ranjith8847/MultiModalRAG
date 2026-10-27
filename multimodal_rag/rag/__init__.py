"""
RAGEngine - Retrieval Augmented Generation engine.

Takes user queries, retrieves relevant chunks from the vector database,
and generates answers using a language model with retrieved context.
"""

from multimodal_rag.rag.rag import RAGEngine, RAGResult

__all__ = ["RAGEngine", "RAGResult"]
