"""
SemanticChunker - Semantically-aware document chunking.

Splits XML documents into semantically meaningful chunks rather than
fixed-length splits, preserving the semantic integrity of content
for better retrieval quality.
"""

from multimodal_rag.chunking.chunking import SemanticChunk, SemanticChunker

__all__ = ["SemanticChunk", "SemanticChunker"]
