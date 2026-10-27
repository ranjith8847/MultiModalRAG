"""
RAGEngine - Retrieval Augmented Generation engine.

Takes user queries, retrieves relevant chunks from the vector database,
and generates answers using a language model with retrieved context.

Features:
- Context-aware retrieval with configurable top_k
- Custom system prompts for answer generation
- Source citation tracking
- Query rephrasing for better retrieval

Usage:
    engine = RAGEngine(indexer)
    answer = engine.query("What are the key findings?")
    print(answer.answer)
    print(answer.sources)
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from multimodal_rag.indexing import SearchResult, VectorIndexer
from multimodal_rag.utils.config import RAGConfig
from multimodal_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """Represents the result of a RAG query."""

    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)  # Source chunks
    context: str = ""  # Full retrieved context
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata,
        }


class RAGEngine:
    """Retrieval Augmented Generation engine for document queries.

    This class handles:
    - Retrieving relevant chunks from the vector database
    - Generating answers using a language model with retrieved context
    - Tracking source citations
    - Query rephrasing for improved retrieval

    The engine works with any VectorIndexer that supports search.

    Usage:
        engine = RAGEngine(indexer)
        result = engine.query("What is the main topic?")
        print(result.answer)
        for source in result.sources:
            print(f"  - Page {source.get('page_number')}: {source['content'][:100]}...")
    """

    def __init__(
        self,
        indexer: Optional[VectorIndexer] = None,
        config: Optional[RAGConfig] = None,
    ):
        """Initialize the RAG engine.

        Args:
            indexer: VectorIndexer instance for document retrieval.
            config: Configuration for RAG parameters.
        """
        self.config = config or RAGConfig()
        self._indexer = indexer
        self._check_llm_dependencies()
        logger.info(
            f"RAGEngine initialized: model={self.config.llm_model}, "
            f"top_k={self.config.top_k}"
        )

    def _check_llm_dependencies(self):
        """Verify that required LLM libraries are installed."""
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai is required for RAG generation. "
                "Install with: pip install openai"
            )

    def query(self, query: str) -> RAGResult:
        """Execute a RAG query.

        Retrieves relevant chunks and generates an answer using the
        retrieved context.

        Args:
            query: The user's question.

        Returns:
            RAGResult with answer, sources, and metadata.

        Raises:
            RuntimeError: If no indexer is configured.
        """
        if self._indexer is None:
            raise RuntimeError(
                "No vector indexer configured. Provide an indexer to RAGEngine."
            )

        logger.info(f"Processing query: '{query}'")

        # Step 1: Retrieve relevant chunks
        search_results = self._indexer.search(query, top_k=self.config.top_k)

        if not search_results:
            logger.warning("No relevant chunks found for query")
            return RAGResult(
                query=query,
                answer="I couldn't find relevant information in the documents for your query.",
                sources=[],
                metadata={"status": "no_results"},
            )

        # Step 2: Build context from retrieved chunks
        context = self._build_context(search_results)

        # Step 3: Generate answer using LLM
        answer = self._generate_answer(query, context)

        # Step 4: Format sources
        sources = [result.to_dict() for result in search_results]

        result = RAGResult(
            query=query,
            answer=answer,
            sources=sources,
            context=context,
            metadata={
                "num_sources": len(search_results),
                "model": self.config.llm_model,
            },
        )

        logger.info(
            f"Query answered with {len(search_results)} sources"
        )
        return result

    def _build_context(self, search_results: list[SearchResult]) -> str:
        """Build a context string from search results.

        Args:
            search_results: List of search results.

        Returns:
            Formatted context string with source citations.
        """
        context_parts = []

        for i, result in enumerate(search_results, start=1):
            page_info = ""
            if result.chunk.page_number:
                page_info = f" [Page {result.chunk.page_number}]"

            source_info = ""
            source_file = result.chunk.metadata.get("source_file", "")
            if source_file:
                source_info = f" (Source: {os.path.basename(source_file)})"

            context_parts.append(
                f"<context {i}>{result.chunk.content}</context {i}>{page_info}{source_info}"
            )

        return "\n\n".join(context_parts)

    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using the LLM with retrieved context."""

        import openai

        client = openai.OpenAI(
            api_key=self.config.api_key or "lm-studio",
            base_url=self.config.base_url or "http://localhost:1234/v1",
        )

        response = client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": self.config.system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Answer the question using the supplied document context. "
                        "Give a direct, informative answer. If the context does "
                        "not contain enough information, say so.\n\n"
                        f"Document context:\n{context}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
            max_tokens=2048,
            temperature=self.config.temperature,
        )

        # Log the raw response so an empty answer is not silently hidden.
        choice = response.choices[0]
        message = choice.message
        content = message.content

        logger.info(
            "LLM response: model=%s, finish_reason=%s, content=%r",
            response.model,
            choice.finish_reason,
            content,
        )

        if isinstance(content, str) and content.strip():
            return content.strip()

        # Do not silently return an empty answer.
        logger.error(
            "LLM returned empty message content. "
            "Finish reason: %s; response: %r",
            choice.finish_reason,
            response,
        )

        raise RuntimeError(
            "The LLM returned an empty answer. Check the preceding log "
            "for the raw response and finish reason."
        )


    def query_with_sources(self, query: str) -> RAGResult:
        """Query with explicit source tracking and formatting.

        Same as query() but ensures source information is always
        included in the response.

        Args:
            query: The user's question.

        Returns:
            RAGResult with detailed source information.
        """
        result = self.query(query)

        # Add formatted source references to the answer
        if result.sources:
            source_refs = []
            for i, source in enumerate(result.sources, start=1):
                page = source.get("page_number", "N/A")
                source_file = os.path.basename(
                    source.get("metadata", {}).get("source_file", "Unknown")
                )
                source_refs.append(
                    f"[{i}] Page {page} from {source_file}"
                )
            result.metadata["source_references"] = source_refs

        return result

    def set_indexer(self, indexer: VectorIndexer):
        """Set or update the vector indexer.

        Args:
            indexer: VectorIndexer instance.
        """
        self._indexer = indexer
        logger.info("Vector indexer updated")

    def get_indexer(self) -> Optional[VectorIndexer]:
        """Get the current vector indexer.

        Returns:
            Current VectorIndexer instance, or None.
        """
        return self._indexer
