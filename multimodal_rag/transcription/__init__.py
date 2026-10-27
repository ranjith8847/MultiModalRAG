"""
XMLTranscriber - Multi-modal LLM transcription of page images to XML.

Uses multi-modal LLMs (GPT-4o, Claude, etc.) to transcribe page images
into structured XML format, preserving document order, tables, flowcharts,
and other visual elements in a machine-readable format.
"""

from multimodal_rag.transcription.transcription import TranscribedDocument, XMLTranscriber

__all__ = ["TranscribedDocument", "XMLTranscriber"]
