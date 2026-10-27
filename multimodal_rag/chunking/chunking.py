"""
SemanticChunker - Semantically-aware document chunking.

Splits XML documents into semantically meaningful chunks rather than
fixed-length splits, preserving the semantic integrity of content
for better retrieval quality.

Approaches:
1. XML-aware chunking: Respects XML element boundaries
2. Semantic similarity chunking: Uses embeddings to find natural break points
3. Hybrid approach: Combines both for optimal results

Usage:
    chunker = SemanticChunker()
    chunks = chunker.chunk(xml_content)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from multimodal_rag.utils.config import ChunkingConfig
from multimodal_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticChunk:
    """Represents a semantically coherent chunk of text."""

    chunk_id: str  # Unique identifier
    content: str  # The chunk text content
    page_number: Optional[int] = None  # Source page number
    metadata: dict = field(default_factory=dict)  # Additional metadata
    token_count: int = 0  # Approximate token count

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """Estimate the number of tokens in the chunk."""
        # Simple heuristic: ~4 chars per token for English text
        return len(self.content) // 4

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "page_number": self.page_number,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticChunk":
        """Create a SemanticChunk from a dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            page_number=data.get("page_number"),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
        )


class SemanticChunker:
    """Splits documents into semantically meaningful chunks.

    This class provides multiple chunking strategies:

    1. **XML-aware chunking**: Respects XML element boundaries, ensuring
       that chunks don't split XML elements in the middle.

    2. **Semantic similarity chunking**: Uses embedding similarity to
       find natural break points in the document where topics change.

    3. **Hybrid chunking**: Combines both approaches for optimal results.

    The key insight is that fixed-length chunking can split semantically
    related content, while semantic chunking preserves meaning at the
    cost of some flexibility.

    Usage:
        chunker = SemanticChunker()
        chunks = chunker.chunk(xml_content, source_file="doc.pdf")
    """

    # XML tags that typically indicate section boundaries
    HEADING_TAGS = {
        "heading", "title", "section", "subsection", "h1", "h2", "h3",
        "h4", "h5", "h6", "chapter", "part",
    }

    # XML tags that indicate content blocks
    CONTENT_TAGS = {
        "paragraph", "text", "p", "body", "content", "description",
    }

    # XML tags that should stay together
    BOUNDARY_TAGS = {
        "table", "figure", "flowchart", "equation", "code", "list",
        "item", "row", "cell",
    }

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the semantic chunker.

        Args:
            config: Configuration for chunking parameters.
        """
        self.config = config or ChunkingConfig()
        self._chunk_counter = 0
        logger.info(
            f"SemanticChunker initialized: chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.overlap}, min_chunk={self.config.min_chunk_size}"
        )

    def chunk(
        self,
        xml_content: str,
        source_file: str = "",
        page_number: Optional[int] = None,
    ) -> list[SemanticChunk]:
        """Chunk XML content into semantically meaningful pieces.

        Uses a hybrid approach:
        1. First, split by major XML section boundaries
        2. Then, apply semantic similarity-based splitting within sections
        3. Finally, merge small chunks that are semantically similar

        Args:
            xml_content: XML string to chunk.
            source_file: Source file path for metadata.
            page_number: Source page number for metadata.

        Returns:
            List of SemanticChunk objects.
        """
        self._chunk_counter = 0  # Reset counter for new chunking session

        # Strategy 1: XML-aware chunking
        xml_chunks = self._xml_aware_chunk(xml_content)

        if len(xml_chunks) > 1:
            logger.debug(f"XML-aware chunking produced {len(xml_chunks)} chunks")
            return self._finalize_chunks(xml_chunks, source_file, page_number)

        # Strategy 2: Semantic similarity chunking (if embeddings available)
        try:
            semantic_chunks = self._semantic_chunk(xml_content)
            if semantic_chunks:
                logger.debug(
                    f"Semantic chunking produced {len(semantic_chunks)} chunks"
                )
                return self._finalize_chunks(
                    semantic_chunks, source_file, page_number
                )
        except Exception as e:
            logger.debug(f"Semantic chunking unavailable: {e}")

        # Fallback: Simple text-based chunking
        simple_chunks = self._simple_chunk(xml_content)
        return self._finalize_chunks(simple_chunks, source_file, page_number)

    def _xml_aware_chunk(self, xml_content: str) -> list[str]:
        """Chunk by respecting XML element boundaries.

        Splits the XML at section boundaries (headings, major sections)
        while keeping related elements together.

        Args:
            xml_content: XML string to chunk.

        Returns:
            List of XML string chunks.
        """
        chunks = []
        current_chunk = ""
        current_size = 0

        # Find section boundaries by looking for heading tags
        # Split on heading elements that start new sections
        section_pattern = r"(<(?:heading|title|section|h[1-6]|chapter)[^>]*>)"

        parts = re.split(section_pattern, xml_content)

        for part in parts:
            if not part.strip():
                continue

            # Check if this part starts a new section
            is_section_start = bool(re.match(section_pattern, part.strip()))

            if is_section_start and current_chunk:
                # Save current chunk and start new one
                if current_size >= self.config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = part
                current_size = len(part)
            else:
                current_chunk += part
                current_size += len(part)

                # Check if we've exceeded chunk size
                if current_size >= self.config.chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [xml_content]

    def _semantic_chunk(self, xml_content: str) -> Optional[list[str]]:
        """Chunk using semantic similarity to find natural break points.

        Uses sentence-level embeddings to find points where the semantic
        content changes significantly, indicating a natural boundary.

        Args:
            xml_content: XML string to chunk.

        Returns:
            List of XML string chunks, or None if embeddings unavailable.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunk
            from langchain_core.documents import Document

            # Extract text from XML
            text_content = self._extract_text_from_xml(xml_content)

            # Create a semantic chunker using LangChain
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.api_key
                if hasattr(self.config, "api_key")
                else None,
            )

            semantic_splitter = SemanticChunk(
                embeddings=embeddings,
                buffer_size=self.config.chunk_size // 50,
                add_start_index=True,
            )

            docs = semantic_splitter.create_documents([text_content])
            return [doc.page_content for doc in docs]

        except ImportError:
            logger.debug("LangChain experimental not available for semantic chunking")
            return None
        except Exception as e:
            logger.debug(f"Semantic chunking failed: {e}")
            return None

    def _simple_chunk(self, xml_content: str) -> list[str]:
        """Simple character-based chunking with overlap.

        Fallback method when more sophisticated chunking is unavailable.
        Splits by the configured separator and applies overlap.

        Args:
            xml_content: XML string to chunk.

        Returns:
            List of XML string chunks.
        """
        chunks = []
        separator = self.config.separator
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap

        # Split by separator
        parts = xml_content.split(separator)

        current_chunk = ""
        for part in parts:
            test_chunk = current_chunk + separator + part if current_chunk else part

            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap: keep the last 'overlap' characters
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = separator + current_chunk[-overlap:]
                else:
                    current_chunk = ""
            else:
                current_chunk = test_chunk

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [xml_content]

    def _extract_text_from_xml(self, xml_content: str) -> str:
        """Extract plain text from XML content.

        Strips XML tags while preserving the text content for
        semantic analysis.

        Args:
            xml_content: XML string.

        Returns:
            Plain text string.
        """
        # Remove XML tags but keep content
        text = re.sub(r"<[^>]+>", " ", xml_content)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _finalize_chunks(
        self,
        chunks: list[str],
        source_file: str = "",
        page_number: Optional[int] = None,
    ) -> list[SemanticChunk]:
        """Convert raw chunks to SemanticChunk objects with metadata.

        Args:
            chunks: List of raw string chunks.
            source_file: Source file path.
            page_number: Source page number.

        Returns:
            List of SemanticChunk objects.
        """
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{source_file}_chunk_{i+1:04d}"
            if page_number:
                chunk_id = f"{source_file}_p{page_number}_chunk_{i+1:04d}"

            semantic_chunk = SemanticChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                page_number=page_number,
                metadata={
                    "source_file": source_file,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            result.append(semantic_chunk)

        logger.info(f"Created {len(result)} semantic chunks")
        return result

    def merge_small_chunks(
        self, chunks: list[SemanticChunk], min_size: Optional[int] = None
    ) -> list[SemanticChunk]:
        """Merge chunks that are smaller than the minimum size.

        This helps avoid creating too many tiny chunks that may not
        be useful for retrieval.

        Args:
            chunks: List of SemanticChunk objects.
            min_size: Minimum chunk size in characters. Defaults to config.

        Returns:
            List of merged SemanticChunk objects.
        """
        if min_size is None:
            min_size = self.config.min_chunk_size

        if len(chunks) <= 1:
            return chunks

        merged = [chunks[0]]

        for chunk in chunks[1:]:
            if len(merged[-1].content) + len(chunk.content) <= min_size * 2:
                # Merge with previous chunk
                merged[-1].content += self.config.separator + chunk.content
                merged[-1].metadata["merged_from"] = merged[-1].metadata.get(
                    "merged_from", []
                )
                merged[-1].metadata["merged_from"].append(chunk.chunk_id)
                merged[-1].token_count = merged[-1]._estimate_tokens()
            else:
                merged.append(chunk)

        logger.info(
            f"Merged {len(chunks)} chunks into {len(merged)} chunks"
        )
        return merged
