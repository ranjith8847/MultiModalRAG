"""
MultiModalRAGPipeline - The main orchestrator for the entire pipeline.

This module ties together document ingestion, LLM-based XML transcription,
semantic chunking, vector indexing, and RAG querying into a single
cohesive pipeline.

Pipeline Flow:
    Document (PDF/PPTX)
        ↓
    [Ingestion] → Page Images
        ↓
    [Transcription] → XML Documents (parallel processing)
        ↓
    [Semantic Chunking] → Semantic Chunks
        ↓
    [Vector Indexing] → Vector Database
        ↓
    [RAG Query] → Answer with Sources

Usage:
    pipeline = MultiModalRAGPipeline()
    
    # Process a document
    pipeline.process_document("document.pdf")
    
    # Query the processed documents
    result = pipeline.query("What are the key findings?")
    print(result.answer)
    
    # Or process multiple documents
    pipeline.process_documents(["doc1.pdf", "doc2.pptx"])
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
from collections import Counter

from multimodal_rag.chunking import SemanticChunk, SemanticChunker
from multimodal_rag.ingestion import DocumentIngestor, PageImage
from multimodal_rag.indexing import VectorIndexer
from multimodal_rag.rag import RAGEngine, RAGResult
from multimodal_rag.transcription import TranscribedDocument, XMLTranscriber
from multimodal_rag.utils.config import (
    ChunkingConfig,
    Config,
    EmbeddingConfig,
    IndexingConfig,
    IngestionConfig,
    LLMConfig,
    RAGConfig,
    TranscriptionConfig,
    get_config,
)
from multimodal_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStatus:
    """Tracks the current status of the pipeline."""

    documents_processed: int = 0
    total_pages: int = 0
    total_chunks: int = 0
    is_indexed: bool = False
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "documents_processed": self.documents_processed,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "is_indexed": self.is_indexed,
            "last_error": self.last_error,
        }


class MultiModalRAGPipeline:
    """End-to-end MultiModal RAG pipeline orchestrator.

    This class provides a simple interface for the complete pipeline:
    1. Ingest documents (PDF, PPTX)
    2. Transcribe pages to XML using multi-modal LLMs (in parallel)
    3. Semantically chunk the XML content
    4. Index chunks into a vector database
    5. Query and generate answers with source citations

    The pipeline is designed to be:
    - **Modular**: Each stage can be used independently
    - **Configurable**: All parameters are configurable
    - **Extensible**: Easy to add new document formats or backends

    Usage:
        # Simple usage
        pipeline = MultiModalRAGPipeline()
        pipeline.process_document("report.pdf")
        answer = pipeline.query("What are the main conclusions?")

        # Custom configuration
        pipeline = MultiModalRAGPipeline(
            llm_model="claude-3-opus",
            vector_store="chroma",
            max_concurrent=5,
        )
        pipeline.process_document("presentation.pptx")
        result = pipeline.query("Explain the methodology")

        # Access intermediate results
        pipeline.process_document("document.pdf")
        print(pipeline.get_transcribed_documents())  # XML content
        print(pipeline.get_chunks())  # Semantic chunks
        print(pipeline.get_index_info())  # Index statistics
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the MultiModal RAG pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or get_config()

        # Initialize pipeline components
        self._ingestor = DocumentIngestor(self.config.ingestion)
        self._transcriber = XMLTranscriber(self.config.transcription)
        self._chunker = SemanticChunker(self.config.chunking)
        self._indexer = VectorIndexer(self.config)
        self._rag_engine = RAGEngine(self._indexer, self.config.rag)

        # Pipeline state
        self._page_images: list[PageImage] = []
        self._transcribed_documents: list[TranscribedDocument] = []
        self._chunks: list[SemanticChunk] = []
        self._status = PipelineStatus()

        logger.info("MultiModalRAGPipeline initialized")
        logger.info(
            f"  LLM Model: {self.config.llm.model}"
            f"\n  Embedding: {self.config.embedding.model}"
            f"  Vector Store: {self.config.indexing.vector_store}"
            f"\n  Max Concurrency: {self.config.transcription.max_concurrent}"
        )

    def process_document(self, file_path: str) -> dict:
        """Process a single document through the full pipeline."""

        logger.info(f"Processing document: {file_path}")

        try:
            # Step 1: Ingest pages
            logger.info("Step 1/4: Ingesting pages...")
            page_images = self._ingestor.ingest(file_path)

            # Step 2: Transcribe only the current document's pages
            logger.info("Step 2/4: Transcribing pages to XML...")
            transcribed_docs = self._transcriber.transcribe(page_images)

            # Step 3: Chunk only the newly transcribed documents
            logger.info("Step 3/4: Creating semantic chunks...")
            new_chunks = self._chunk_documents(transcribed_docs)

            chunk_ids = [str(chunk.chunk_id) for chunk in new_chunks]
            id_counts = Counter(chunk_ids)

            duplicates = {
                chunk_id: count
                for chunk_id, count in id_counts.items()
                if count > 1
            }

            if duplicates:
                raise ValueError(
                    f"Generated chunks contain duplicate IDs: {duplicates}"
                )

            if not new_chunks:
                raise RuntimeError(
                    f"No chunks were generated for {file_path}"
                )

            # Step 4: Index only the newly generated chunks
            logger.info(
                "Step 4/4: Indexing %d new chunks...",
                len(new_chunks),
            )

            index_result = self._indexer.index(new_chunks)

            # The indexer must report its actual indexing outcome.
            if isinstance(index_result, dict):
                indexed_count = None

                for key in (
                    "indexed_chunks",
                    "indexed",
                    "indexed_count",
                    "total_indexed",
                ):
                    if key in index_result:
                        indexed_count = index_result[key]
                        break

                if indexed_count is None:
                    raise RuntimeError(
                        "Indexer did not return a recognized indexed-count "
                        f"field. Result: {index_result}"
                    )

                if indexed_count != len(new_chunks):
                    raise RuntimeError(
                        f"Indexing incomplete: expected {len(new_chunks)} "
                        f"chunks, but indexer reported {indexed_count}. "
                        f"Details: {index_result}"
                    )

            elif index_result is None:
                raise RuntimeError(
                    "Indexer returned None; indexing success cannot be verified."
                )

            else:
                raise RuntimeError(
                    "Unexpected indexer result type: "
                    f"{type(index_result).__name__}. "
                    "Expected a dictionary with an indexed-count field."
                )

            # Update in-memory state after successful indexing.
            self._page_images.extend(page_images)
            self._transcribed_documents.extend(transcribed_docs)
            self._chunks.extend(new_chunks)

            self._status.documents_processed += 1
            self._status.total_pages += len(page_images)
            self._status.total_chunks = len(self._chunks)
            self._status.is_indexed = True
            self._status.last_error = None

            logger.info(
                "Document processed successfully: %d pages, "
                "%d new chunks, %d total in-memory chunks",
                len(page_images),
                len(new_chunks),
                len(self._chunks),
            )

            return {
                "status": "success",
                "pages": len(page_images),
                "new_chunks": len(new_chunks),
                "total_chunks": len(self._chunks),
                "indexed_chunks": indexed_count,
            }

        except Exception as e:
            self._status.last_error = str(e)

            logger.exception("Document processing failed")

            return {
                "status": "error",
                "error": str(e),
                "pages": 0,
                "new_chunks": 0,
                "total_chunks": len(self._chunks),
            }

    def process_documents(
        self, file_paths: list[str], skip_existing: bool = True
    ) -> dict:
        """Process multiple documents through the pipeline.

        Args:
            file_paths: List of document file paths.
            skip_existing: If True, skip documents already processed.

        Returns:
            Dict with overall processing statistics.
        """
        logger.info(f"Processing {len(file_paths)} documents")

        results = []
        for file_path in file_paths:
            result = self.process_document(file_path)
            results.append({"file": file_path, **result})

        return {
            "total_documents": len(file_paths),
            "successful": sum(
                1 for r in results if r.get("status") == "success"
            ),
            "failed": sum(
                1 for r in results if r.get("status") == "error"
            ),
            "details": results,
            "pipeline_status": self._status.to_dict(),
        }

    def _chunk_documents(
    self,
    documents: list[TranscribedDocument],
    ) -> list[SemanticChunk]:
        """Chunk only the documents passed to this method."""

        chunks_for_documents = []

        for doc in documents:
            for page in doc.pages:
                page_chunks = self._chunker.chunk(
                    xml_content=page["content"],
                    source_file=doc.source_file,
                    page_number=page["page_number"],
                )

                chunks_for_documents.extend(page_chunks)

        logger.info(
            "Created %d chunks from %d newly transcribed documents",
            len(chunks_for_documents),
            len(documents),
        )

        return chunks_for_documents

    def query(self, query: str) -> RAGResult:
        """Query the processed documents.

        Retrieves relevant chunks and generates an answer.

        Args:
            query: The user's question.

        Returns:
            RAGResult with answer and source citations.
        """
        return self._rag_engine.query(query)

    def query_with_sources(self, query: str) -> RAGResult:
        """Query with explicit source tracking.

        Args:
            query: The user's question.

        Returns:
            RAGResult with detailed source information.
        """
        return self._rag_engine.query_with_sources(query)

    # --- Getters for intermediate results ---

    def get_transcribed_documents(self) -> list[TranscribedDocument]:
        """Get all transcribed documents (XML content).

        Returns:
            List of TranscribedDocument objects.
        """
        return self._transcribed_documents.copy()

    def get_chunks(self) -> list[SemanticChunk]:
        """Get all semantic chunks.

        Returns:
            List of SemanticChunk objects.
        """
        return self._chunks.copy()

    def get_page_images(self) -> list[PageImage]:
        """Get all page images.

        Returns:
            List of PageImage objects.
        """
        return self._page_images.copy()

    def get_index_info(self) -> dict:
        """Get information about the vector index.

        Returns:
            Dict with index statistics.
        """
        return self._indexer.get_collection_info()

    
    
    def get_status(self) -> PipelineStatus:
        """Get the current pipeline status.

        Returns:
            PipelineStatus object.
        """
        return replace(self._status)

    def save_index(self, path: str):
        """Save the vector index to disk.

        Args:
            path: Directory path to save the index.
        """
        self._indexer.save(path)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: Optional[str] = None):
        """Load a previously saved vector index.

        Args:
            path: Path to the saved index. Uses config default if None.
        """
        self._indexer.load(path)
        self._status.is_indexed = True
        logger.info("Index loaded successfully")

    def export_xml(self, output_dir: str):
        """Export all transcribed XML documents to files.

        Args:
            output_dir: Directory to save XML files.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for doc in self._transcribed_documents:
            filename = Path(doc.source_file).stem
            output_path = Path(output_dir) / f"{filename}.xml"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.full_xml)
            logger.info(f"Exported XML to {output_path}")

    def export_chunks(self, output_dir: str):
        """Export all semantic chunks to a JSON file.

        Args:
            output_dir: Directory to save the chunks file.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        import json

        output_path = Path(output_dir) / "chunks.json"
        chunks_data = [chunk.to_dict() for chunk in self._chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self._chunks)} chunks to {output_path}")

    def reset(self):
        """Reset the pipeline state (keeps configuration)."""
        self._page_images.clear()
        self._transcribed_documents.clear()
        self._chunks.clear()
        self._status = PipelineStatus()
        logger.info("Pipeline state reset")

    @property
    def ingestor(self) -> DocumentIngestor:
        """Access the document ingestor directly."""
        return self._ingestor

    @property
    def transcriber(self) -> XMLTranscriber:
        """Access the XML transcriber directly."""
        return self._transcriber

    @property
    def chunker(self) -> SemanticChunker:
        """Access the semantic chunker directly."""
        return self._chunker

    @property
    def indexer(self) -> VectorIndexer:
        """Access the vector indexer directly."""
        return self._indexer

    @property
    def rag_engine(self) -> RAGEngine:
        """Access the RAG engine directly."""
        return self._rag_engine
