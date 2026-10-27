"""
MultiModalRAGPipeline - The main orchestrator for the entire pipeline.

This module ties together document ingestion, LLM-based XML transcription,
semantic chunking, vector indexing, and RAG querying into a single
cohesive pipeline.
"""

from multimodal_rag.pipeline.pipeline import MultiModalRAGPipeline

__all__ = ["MultiModalRAGPipeline"]
