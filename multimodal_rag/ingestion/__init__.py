"""
DocumentIngestor - Handles document ingestion and page-to-image conversion.

Supports PDF and PPTX formats, converting each page/slide into high-quality
images for subsequent multi-modal LLM processing.
"""

from multimodal_rag.ingestion.ingestion import DocumentIngestor, PageImage

__all__ = ["DocumentIngestor", "PageImage"]
