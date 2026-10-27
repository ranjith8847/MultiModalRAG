"""
Configuration management for the MultiModal RAG pipeline.

Centralizes all configurable parameters including API keys, model choices,
embedding settings, chunking parameters, and vector database options.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for the multi-modal LLM used in transcription and generation."""

    # Model selection
    model: str = "gpt-4o"  # gpt-4o, claude-3-opus, or any OpenAI-compatible model
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For OpenAI-compatible endpoints (e.g., LM Studio, FreeLLMAPI)
    temperature: float = 0.0
    max_tokens: int = 4096
    vision_enabled: bool = True  # Multi-modal LLM must support vision

    def __post_init__(self):
        # Priority: env var > hardcoded default
        if self.api_key is None:
            self.api_key = (
                os.getenv("LM_STUDIO_API_KEY", os.getenv("FREELLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
            )
        if self.base_url is None:
            self.base_url = (
                os.getenv("LM_STUDIO_API_BASE", os.getenv("FREELLM_API_BASE", os.getenv("OPENAI_API_BASE")))
            )
        # Override model name from env if set (e.g., LM_STUDIO_MODEL)
        lm_studio_model = os.getenv("LM_STUDIO_MODEL")
        if lm_studio_model and self.model == "gpt-4o":
            self.model = lm_studio_model


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model used for vector indexing."""

    model: str = "text-embedding-3-small"  # text-embedding-3-small, text-embedding-3-large, or local embedding model
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimensions: Optional[int] = None  # None = use model default

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = (
                os.getenv("LM_STUDIO_API_KEY", os.getenv("FREELLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
            )
        if self.base_url is None:
            self.base_url = (
                os.getenv("LM_STUDIO_API_BASE", os.getenv("FREELLM_API_BASE", os.getenv("OPENAI_API_BASE")))
            )
        # Override model name from env if set (e.g., LM_STUDIO_EMBEDDING_MODEL)
        embedding_model = os.getenv("LM_STUDIO_EMBEDDING_MODEL")
        if embedding_model and self.model == "text-embedding-3-small":
            self.model = embedding_model


@dataclass
class IngestionConfig:
    """Configuration for document ingestion and page-to-image conversion."""

    dpi: int = 300  # Resolution for page images (higher = better OCR)
    image_format: str = "png"  # png, jpeg
    max_pages: Optional[int] = None  # Limit pages for testing
    slide_width_inches: float = 13.33  # Standard 16:9 slide width
    slide_height_inches: float = 7.5  # Standard 16:9 slide height


@dataclass
class TranscriptionConfig:
    """Configuration for LLM-based XML transcription."""

    # LLM settings for transcription
    model: str = "gpt-4o"  # gpt-4o, claude-3-opus, or any OpenAI-compatible model
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For OpenAI-compatible endpoints (e.g., LM Studio, FreeLLMAPI)
    temperature: float = 0.0
    max_tokens: int = 4096

    # Transcription-specific settings
    max_concurrent: int = 10  # Number of parallel transcription workers
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    system_prompt: str = """You are an expert document analyst. Your task is to transcribe 
document pages into structured XML format that preserves the document's structure, 
content order, and semantic meaning.

Guidelines:
1. Preserve the exact order of content as it appears on the page.
2. For text content, transcribe accurately including headings, paragraphs, lists.
3. For tables, use <table> elements with <row> and <cell> tags.
4. For images/figures, describe them in <figure> tags with captions.
5. For flowcharts/diagrams, describe the structure in <flowchart> tags.
6. For equations, use LaTeX format within <equation> tags.
7. Preserve hierarchical structure (headings, subheadings).
8. Do NOT skip any content - transcribe everything visible on the page.

Output ONLY valid XML. Do not include any markdown code fences or explanations."""

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = (
                os.getenv("LM_STUDIO_API_KEY", os.getenv("FREELLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
            )
        if self.base_url is None:
            self.base_url = (
                os.getenv("LM_STUDIO_API_BASE", os.getenv("FREELLM_API_BASE", os.getenv("OPENAI_API_BASE")))
            )
        # Override model name from env if set (e.g., LM_STUDIO_VISION_MODEL)
        vision_model = os.getenv("LM_STUDIO_VISION_MODEL")
        if vision_model and self.model == "gpt-4o":
            self.model = vision_model


@dataclass
class ChunkingConfig:
    """Configuration for semantic chunking."""

    chunk_size: int = 512  # Target chunk size in tokens
    overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum tokens per chunk
    separator: str = "\n\n"  # Primary separator for splitting
    embedding_model: str = "text-embedding-3-small"


@dataclass
class IndexingConfig:
    """Configuration for vector database indexing."""

    vector_store: str = "chroma"  # chroma, faiss, pinecone, qdrant
    collection_name: str = "multimodal_rag_collection"
    persist_directory: str = "./chroma_db"  # For ChromaDB
    distance_metric: str = "cosine"  # cosine, l2, ip
    top_k: int = 5  # Number of results to retrieve


@dataclass
class RAGConfig:
    """Configuration for the RAG query pipeline."""

    llm_model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    top_k: int = 5
    system_prompt: str = """You are a helpful assistant that answers questions based on 
the provided context from a document collection. 

Guidelines:
1. Use ONLY the provided context to answer the question.
2. If the context doesn't contain enough information, say so clearly.
3. Cite the source page/document when possible.
4. Be concise but thorough in your answers.
5. If the question is unrelated to the documents, politely redirect."""

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = (
                os.getenv("LM_STUDIO_API_KEY", os.getenv("FREELLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
            )
        if self.base_url is None:
            self.base_url = (
                os.getenv("LM_STUDIO_API_BASE", os.getenv("FREELLM_API_BASE", os.getenv("OPENAI_API_BASE")))
            )
        # Override model name from env if set (e.g., LM_STUDIO_MODEL)
        rag_model = os.getenv("LM_STUDIO_MODEL")
        if rag_model and self.llm_model == "gpt-4o":
            self.llm_model = rag_model


@dataclass
class Config:
    """Main configuration class that aggregates all sub-configurations."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Create Config from environment variables."""
        return cls()


def get_config() -> Config:
    """Get the default configuration."""
    return Config.from_env()
