"""
VectorIndexer - Vector database indexing for semantic chunks.

Stores semantically chunked documents into a vector database (ChromaDB,
FAISS, or Pinecone) with configurable embedding models for efficient
semantic search and retrieval.

Supported backends:
- ChromaDB (default, persistent, easy setup)
- FAISS (fast, in-memory or persistent)
- Pinecone (cloud, scalable)
- Qdrant (feature-rich, self-hosted or cloud)

Usage:
    indexer = VectorIndexer()
    indexer.index(chunks)
    results = indexer.search("your query", top_k=5)
"""

import http
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from multimodal_rag.chunking import SemanticChunk
from multimodal_rag.utils.config import Config, IndexingConfig
from multimodal_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector database."""

    chunk: SemanticChunk
    score: float  # Similarity score (higher = more similar)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "content": self.chunk.content,
            "page_number": self.chunk.page_number,
            "score": self.score,
            "metadata": self.chunk.metadata,
        }


class VectorIndexer:
    """Indexes semantic chunks into a vector database for retrieval.

    This class handles:
    - Converting chunks to vector embeddings
    - Storing embeddings in a vector database
    - Semantic search and retrieval
    - Multiple backend support (ChromaDB, FAISS, Pinecone, Qdrant)

    The indexer is designed to be flexible, allowing easy switching
    between different vector database backends.

    Usage:
        indexer = VectorIndexer()
        indexer.index(chunks)
        results = indexer.search("What is the main topic?", top_k=5)
    """

    SUPPORTED_BACKENDS = {"chroma", "faiss", "pinecone", "qdrant"}

    def __init__(self, config: Optional[Config] = None):
        """Initialize the vector indexer.

        Args:
            config: Full Config object (or IndexingConfig for backward compat).
        """
        # Accept either full Config or IndexingConfig
        if isinstance(config, Config):
            self._full_config = config
            self.config = config.indexing
        elif isinstance(config, IndexingConfig):
            self.config = config
            self._full_config = None
        else:
            self.config = IndexingConfig()
            self._full_config = None
        
        self._check_backend_dependencies()
        self._db = None
        self._embeddings = None
        logger.info(
            f"VectorIndexer initialized: backend={self.config.vector_store}, "
            f"collection={self.config.collection_name}"
        )

    def _check_backend_dependencies(self):
        """Verify that required backend libraries are installed."""
        backend = self.config.vector_store

        if backend == "chroma":
            try:
                import chromadb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "chromadb is required for ChromaDB backend. "
                    "Install with: pip install chromadb"
                )
        elif backend == "faiss":
            try:
                import faiss  # noqa: F401
            except ImportError:
                raise ImportError(
                    "faiss is required for FAISS backend. "
                    "Install with: pip install faiss-cpu"
                )
        elif backend == "pinecone":
            try:
                import pinecone  # noqa: F401
            except ImportError:
                raise ImportError(
                    "pinecone is required for Pinecone backend. "
                    "Install with: pip install pinecone"
                )
        elif backend == "qdrant":
            try:
                import qdrant_client  # noqa: F401
            except ImportError:
                raise ImportError(
                    "qdrant-client is required for Qdrant backend. "
                    "Install with: pip install qdrant-client"
                )

        # Embedding library check
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: F401
        except ImportError:
            raise ImportError(
                "langchain-openai is required for embeddings. "
                "Install with: pip install langchain-openai"
            )
    # def _initialize_embeddings(self):
    #     from langchain_openai import OpenAIEmbeddings

    #     embedding_config = (
    #         self._full_config.embedding
    #         if self._full_config is not None
    #         else None
    #     )

    #     model = (
    #         embedding_config.model
    #         if embedding_config and embedding_config.model
    #         else "text-embedding-3-small"
    #     )

    #     api_key = (
    #         embedding_config.api_key
    #         if embedding_config and embedding_config.api_key
    #         else "lm-studio"
    #     )

    #     base_url = (
    #         embedding_config.base_url
    #         if embedding_config and embedding_config.base_url
    #         else None
    #     )

    #     self._embeddings = OpenAIEmbeddings(
    #         model=model or "text-embedding-bge-large-en-v1.5",
    #         api_key=api_key or "lm-studio",
    #         base_url="http://localhost:1234/v1",
    #         tiktoken_enabled=False,
    #     )

    #     return self._embeddings
    def _initialize_embeddings(self):
        """Initialize embeddings for an OpenAI-compatible embedding server."""
        from langchain_openai import OpenAIEmbeddings

        embedding_config = (
            self._full_config.embedding
            if self._full_config is not None
            else None
        )

        model = (
            embedding_config.model
            if embedding_config and embedding_config.model
            else "text-embedding-bge-large-en-v1.5"
        )

        api_key = (
            embedding_config.api_key
            if embedding_config and embedding_config.api_key
            else "lm-studio"
        )

        base_url = (
            embedding_config.base_url
            if embedding_config and embedding_config.base_url
            else "http://localhost:1234/v1"
        )

        self._embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
        )

        logger.info(
            "Initialized embedding client: model=%s, endpoint=%s",
            model,
            base_url,
        )

        return self._embeddings
    def index(
        self,
        chunks: list[SemanticChunk],
        batch_size: int = 100,
    ) -> dict:
        """Index a list of semantic chunks into the vector database.

        Processes chunks in batches for efficient embedding generation
        and storage.

        Args:
            chunks: List of SemanticChunk objects to index.
            batch_size: Number of chunks to process per batch.

        Returns:
            Dict with indexing statistics.

        Raises:
            ValueError: If no chunks are provided.
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        logger.info(f"Indexing {len(chunks)} chunks into {self.config.vector_store}")

        # Initialize embeddings
        from langchain_openai import OpenAIEmbeddings

        # Get embedding config from full config or fallback
        embedding_model = None
        embedding_api_key = None
        embedding_base_url = None
        
        if self._full_config and hasattr(self._full_config, 'embedding'):
            embedding_model = self._full_config.embedding.model
            embedding_api_key = self._full_config.embedding.api_key
            embedding_base_url = self._full_config.embedding.base_url
        
        # self._embeddings = OpenAIEmbeddings(
        #     model=embedding_model or "text-embedding-3-small",
        #     api_key=embedding_api_key or "lm-studio",
        #     base_url="http://localhost:1234/v1" or None,
        #     tiktoken_enabled=False,
        # )
        self._initialize_embeddings()

        # Initialize the vector database
        self._initialize_db()

        # Process in batches
        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                self._index_batch(batch)
                total_indexed += len(batch)
                logger.debug(f"Indexed {total_indexed}/{len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to index batch {i // batch_size + 1}: {e}")
                logger.warning(
                    "This may be because your local LM Studio model doesn't support embeddings. "
                    "Try loading an embedding model in LM Studio (e.g., 'BAAI/bge-large-en-v1.5') "
                    "and update LM_STUDIO_EMBEDDING_MODEL in your .env file."
                )
                # Continue with other batches even if one fails

        logger.info(f"Successfully indexed {total_indexed} chunks")
        return {
            "total_chunks": len(chunks),
            "indexed": total_indexed,
            "backend": self.config.vector_store,
            "collection": self.config.collection_name,
        }

    def _initialize_db(self):
        """Initialize the vector database connection."""
        backend = self.config.vector_store

        if backend == "chroma":
            self._init_chroma()
        elif backend == "faiss":
            self._init_faiss()
        elif backend == "pinecone":
            self._init_pinecone()
        elif backend == "qdrant":
            self._init_qdrant()
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

    def _init_chroma(self):
        """Initialize ChromaDB."""
        import chromadb
        from langchain_chroma import Chroma

        persist_dir = self.config.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._db = Chroma(
            client=chromadb.PersistentClient(path=persist_dir),
            collection_name=self.config.collection_name,
            embedding_function=self._embeddings,
        )
        logger.info(f"ChromaDB initialized at {persist_dir}")

    def _init_faiss(self):
        """Initialize FAISS."""
        import faiss
        from langchain_community.vectorstores import FAISS

        self._db = FAISS(
            embedding_function=self._embeddings,
            index=faiss.IndexFlatL2(
                self._embeddings.dimensions
                if hasattr(self._embeddings, "dimensions")
                else 1536
            ),
            collection_name=self.config.collection_name,
        )
        logger.info("FAISS initialized")

    def _init_pinecone(self):
        """Initialize Pinecone."""
        import pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "us-east1-aws")

        pc = pinecone.Pinecone(api_key=api_key)
        index = pc.create_index(
            name=self.config.collection_name,
            dimension=self._embeddings.dimensions or 1536,
            metric=self.config.distance_metric,
        )

        from langchain_pinecone import PineconeVectorStore

        self._db = PineconeVectorStore(
            index=index,
            embedding_function=self._embeddings,
        )
        logger.info(f"Pinecone index '{self.config.collection_name}' initialized")

    def _init_qdrant(self):
        """Initialize Qdrant."""
        from langchain_qdrant import Qdrant

        client_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        self._db = Qdrant.from_documents(
            documents=[],
            embedding=self._embeddings,
            url=client_url,
            api_key=api_key,
            collection_name=self.config.collection_name,
        )
        logger.info(f"Qdrant collection '{self.config.collection_name}' initialized")

    def _index_batch(self, chunks: list[SemanticChunk]):
        """Index a batch of chunks into the vector database.

        Args:
            chunks: List of SemanticChunk objects.
        """
        # Store chunks for retrieval
        if not hasattr(self, '_indexed_chunks'):
            self._indexed_chunks = []
        self._indexed_chunks.extend(chunks)
        
        # Convert chunks to documents for the vector store
        documents = []
        metadatas = []

        for chunk in chunks:
            documents.append(chunk.content)
            metadatas.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "source_file": chunk.metadata.get("source_file", ""),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                }
            )

        # Add to vector database
        if self.config.vector_store == "chroma":
            # self._db.add_documents(documents=documents, metadatas=metadatas)
            self._db.add_texts(
                texts=documents,
                metadatas=metadatas,
                ids=[str(chunk.chunk_id) for chunk in chunks],
            )
        elif self.config.vector_store == "faiss":
            self._db.add_texts(texts=documents, metadatas=metadatas)
        elif self.config.vector_store == "pinecone":
            self._db.add_texts(texts=documents, metadatas=metadatas)
        elif self.config.vector_store == "qdrant":
            self._db.add_texts(texts=documents, metadatas=metadatas)

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> list[SearchResult]:
        """Search the vector database for relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of results to return. Defaults to config.

        Returns:
            List of SearchResult objects sorted by relevance.

        Raises:
            RuntimeError: If the database hasn't been initialized.
        """
        if self._db is None:
            raise RuntimeError(
                "Vector database not initialized. Call index() first."
            )

        top_k = top_k or self.config.top_k

        # Ensure query is a string before embedding
        query_str = str(query)

        # Perform vector search
        if self.config.vector_store == "chroma":
            results = self._db.similarity_search_with_score(
                query_str, k=top_k
            )
        elif self.config.vector_store == "faiss":
            results = self._db.similarity_search_with_score(
                query_str, k=top_k
            )
        elif self.config.vector_store == "pinecone":
            results = self._db.similarity_search_with_score(
                query_str, k=top_k
            )
        elif self.config.vector_store == "qdrant":
            results = self._db.similarity_search_with_score(
                query_str, k=top_k
            )
        else:
            raise ValueError(f"Unsupported backend: {self.config.vector_store}")

        # Convert to SearchResult objects
        search_results = []
        for doc, score in results:
            # Find the matching chunk
            matching_chunk = None
            for chunk in getattr(self, "_indexed_chunks", []):
                if chunk.chunk_id == doc.metadata.get("chunk_id", ""):
                    matching_chunk = chunk
                    break

            if matching_chunk is None:
                # Create a temporary chunk from the document
                from multimodal_rag.chunking import SemanticChunk

                matching_chunk = SemanticChunk(
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    content=doc.page_content,
                    page_number=doc.metadata.get("page_number"),
                    metadata=doc.metadata,
                )

            search_results.append(
                SearchResult(
                    chunk=matching_chunk,
                    score=score,
                )
            )

        return search_results

    def save(self, path: str):
        """Save the vector database to disk.

        Args:
            path: Directory path to save the database.
        """
        if self._db is None:
            raise RuntimeError("Nothing to save - database not initialized")

        if self.config.vector_store == "chroma":
            # ChromaDB persists automatically
            logger.info(f"ChromaDB persisted at {self.config.persist_directory}")
        elif self.config.vector_store == "faiss":
            self._db.save_local(path)
            logger.info(f"FAISS saved to {path}")
        else:
            logger.warning(
                f"Saving not fully supported for {self.config.vector_store}"
            )

    def load(self, path: Optional[str] = None):
        """Load a previously saved vector database.

        Args:
            path: Path to the saved database. Defaults to config.
        """
        if path is None:
            path = self.config.persist_directory

        logger.info(f"Loading vector database from {path}")

        # Re-initialize embeddings
        from langchain_openai import OpenAIEmbeddings

        # Get embedding config from full config or fallback
        embedding_model = None
        embedding_api_key = None
        
        if self._full_config and hasattr(self._full_config, 'embedding'):
            embedding_model = self._full_config.embedding.model
            embedding_api_key = self._full_config.embedding.api_key
        
        # self._embeddings = OpenAIEmbeddings(
        #     model=embedding_model if embedding_model else "text-embedding-3-small",
        #     api_key=embedding_api_key if embedding_api_key else None,
        # )

        # self._embeddings = OpenAIEmbeddings(
        #     model=embedding_model or "text-embedding-3-small",
        #     api_key=embedding_api_key or "lm-studio",
        #     base_url='http://localhost:1234/v1' or None,
        #     tiktoken_enabled=False,
        # )
        self._initialize_embeddings()

        # Re-initialize the database
        self._initialize_db()

    def get_collection_info(self) -> dict:
        """Get information about the current collection.

        Returns:
            Dict with collection statistics.
        """
        if self._db is None:
            return {"error": "Database not initialized"}

        if self.config.vector_store == "chroma":
            collection = self._db._collection
            return {
                "collection_name": self.config.collection_name,
                "total_vectors": collection.count(),
                "backend": "chroma",
            }

        return {"backend": self.config.vector_store}
