# MultiModal RAG — Hybrid Multi-Modal Retrieval Augmented Generation

A Python pipeline for processing enterprise documents (PDF, PPTX) through
multi-modal LLM transcription, semantic chunking, vector indexing, and
retrieval-augmented generation.

## 🌟 Key Innovation

Instead of extracting text, images, and tables separately and indexing them independently,
**MultiModal RAG converts entire PDF pages into images and passes them to a multi-modal LLM**
which transcribes them into structured XML. This approach:

- **Preserves document order** — content appears in the exact sequence as the original document
- **Maintains visual context** — tables, flowcharts, and figures are described in context
- **Better LLM understanding** — structured XML is easier for LLMs to reason about than raw text
- **Parallel processing** — pages are transcribed concurrently for speed
- **Semantic chunking** — chunks preserve semantic meaning rather than arbitrary fixed-length splits

## 📊 Pipeline Architecture

```
┌─────────────┐      ┌──────────────┐     ┌─────────────────┐
│  Documents   │────▶│   Ingestion  │────▶│  Page Images    │
│  (PDF/PPTX)  │     │ (pdf2image,  │     │  (high-res PNG) │
│              │     │  python-pptx)│     │                 │
└─────────────┘      └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐      ┌─────────────────┐
│    RAG      │◀────│  Vector DB   │◀──── │  Semantic Chunks│
│  Query/Answer│    │  (Chroma/    │      │  (XML-aware)    │
│              │    │   FAISS)     │      │                 │
└─────────────┘     └──────────────┘      └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM Transcription (Parallel)              │
│                                                             │
│  Page 1 Image ──┐                                           │
│  Page 2 Image ──┤──▶ GPT-4o / Claude ──▶ Structured XML     │
│  Page 3 Image ──┤      (Multi-modal)    (tables, figures,   │
│     ...        ─┘                        flowcharts, text)  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag

# Install dependencies
pip install -e "."

# For PDF processing, install poppler (system dependency):
#   macOS: brew install poppler
#   Ubuntu: sudo apt-get install poppler-utils

# Set up your API key
cp .env.example .env
# Edit .env and configure your preferred LLM provider
```

### Basic Usage

```python
from multimodal_rag import MultiModalRAGPipeline

# Initialize the pipeline (reads config from .env)
pipeline = MultiModalRAGPipeline()

# Process a document (PDF or PPTX)
pipeline.process_document("company_report.pdf")

# Query the processed documents
result = pipeline.query("What are the key financial findings?")
print(result.answer)

# See source citations
for source in result.sources:
    print(f"  Page {source.get('page_number')}: {source['content'][:100]}...")
```

### Custom Configuration

```python
from multimodal_rag import MultiModalRAGPipeline
from multimodal_rag.utils.config import (
    Config, LLMConfig, EmbeddingConfig, IndexingConfig,
    IngestionConfig, TranscriptionConfig, ChunkingConfig, RAGConfig,
)

# Custom configuration
config = Config()
config.llm = LLMConfig(model="qwen/qwen3.6-35b-a3b")  # or "claude-3-opus"
config.transcription = TranscriptionConfig(max_concurrent=10)
config.indexing = IndexingConfig(vector_store="chroma", top_k=5)

pipeline = MultiModalRAGPipeline(config)
pipeline.process_document("presentation.pptx")
```

### Processing Multiple Documents

```python
# Process multiple documents at once
pipeline.process_documents([
    "report_q1.pdf",
    "report_q2.pptx",
])

# Query across all documents
result = pipeline.query("Compare Q1 and Q2 performance")
```

## 📁 Project Structure

```
multimodal_rag/
├── __init__.py              # Package entry point (exports all public classes)
├── pipeline/                # Main pipeline orchestrator
│   └── pipeline.py          # MultiModalRAGPipeline class
├── ingestion/               # Document ingestion
│   └── ingestion.py         # DocumentIngestor (PDF/PPTX → images)
├── transcription/           # LLM-based transcription
│   └── transcription.py     # XMLTranscriber (images → XML)
├── chunking/                # Semantic chunking
│   └── chunking.py          # SemanticChunker (XML → chunks)
├── indexing/                # Vector indexing
│   └── indexing.py          # VectorIndexer (chunks → vector DB)
├── rag/                     # RAG query engine
│   └── rag.py               # RAGEngine (query → answer)
└── utils/                   # Utilities
    ├── config.py            # Configuration management
    └── logger.py            # Logging utilities

examples/
├── basic_usage.py           # Simple usage example
├── advanced_usage.py        # Advanced configuration
└── demo_notebook.ipynb      # Interactive demo notebook
```

## 🔧 Configuration

All pipeline parameters are configurable via the `Config` class and its
sub-configurations (`LLMConfig`, `EmbeddingConfig`, etc.). Most settings
also support environment-variable overrides.

```python
from multimodal_rag.utils.config import Config

config = Config()

# LLM settings (overridden by LM_STUDIO_MODEL env var)
config.llm.model = "qwen/qwen3.6-35b-a3b"  # or use OpenAI: "gpt-4o"
config.llm.temperature = 0.0           # Deterministic transcription
config.llm.max_tokens = 4096          # Max output tokens per page

# Embedding settings
config.embedding.model = "BAAI/bge-large-en-v1.5"  # or use OpenAI: "text-embedding-3-small"
config.embedding.dimensions = None  # Use model default (bge-large-en-v1.5 = 1024)

# Ingestion settings
config.ingestion.dpi = 300             # Image resolution
config.ingestion.max_pages = None      # Limit pages (for testing)

# Transcription settings
config.transcription.max_concurrent = 10  # Parallel workers
config.transcription.retry_attempts = 3

# Chunking settings
config.chunking.chunk_size = 512       # Target chunk size (tokens)
config.chunking.overlap = 50           # Chunk overlap

# Indexing settings
config.indexing.vector_store = "chroma"  # chroma, faiss, pinecone, qdrant
config.indexing.collection_name = "my_collection"
config.indexing.top_k = 5              # Results per query

# RAG settings
config.rag.llm_model = "qwen/qwen3.6-35b-a3b"  # or use OpenAI: "gpt-4o"
config.rag.top_k = 5
```

## 📤 Export & Inspection

```python
# Export transcribed XML
pipeline.export_xml("./output/xml/")

# Export semantic chunks as JSON
pipeline.export_chunks("./output/chunks/")

# Inspect pipeline state
print(pipeline.get_status())
print(pipeline.get_index_info())

# Access intermediate results
docs = pipeline.get_transcribed_documents()  # XML content
chunks = pipeline.get_chunks()               # Semantic chunks
```

## 🧪 Running Examples

```bash
# Basic usage example (reads from .env)
python examples/basic_usage.py

# Advanced usage with custom config
python examples/advanced_usage.py

# Interactive demo (requires Jupyter)
jupyter notebook examples/demo_notebook.ipynb
```

## 🏗️ Supported Formats

| Format | Status | Notes |
|--------|--------|-------|
| PDF    | ✅     | All PDF versions (requires Poppler system package) |
| PPTX   | ✅     | PowerPoint 2007+ |
| PPT    | ⚠️     | Legacy format (listed but not tested) |

## 🔌 Supported LLMs

| Provider | Models | Status |
|----------|--------|--------|
| OpenAI   | gpt-4o, gpt-4-turbo | ✅ |
| Anthropic | claude-3-opus, claude-3-sonnet | ✅ |
| LM Studio | Any OpenAI-compatible local model | ✅ (default) |
| FreeLLMAPI | Any OpenAI-compatible local model | ✅ |
| Azure    | Any OpenAI-compatible endpoint | ✅ |

## 🔌 Supported Vector Databases

| Backend | Status | Notes |
|---------|--------|-------|
| ChromaDB | ✅ | Default, persistent, local |
| FAISS | ✅ | Fast, in-memory or persistent |
| Pinecone | ⚠️ | Cloud, scalable (requires API key) |
| Qdrant | ⚠️ | Self-hosted or cloud (requires API key) |

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
