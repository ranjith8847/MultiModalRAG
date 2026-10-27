"""
Advanced Usage Example for MultiModal RAG Pipeline

Demonstrates advanced features:
1. Custom configuration (LLM model, vector store, chunking params)
2. Processing multiple documents
3. Exporting intermediate results (XML, chunks)
4. Querying with source tracking
5. Saving and loading indexes

Usage:
    python examples/advanced_usage.py

Requirements:
    - A PDF or PPTX file to process
    - OPENAI_API_KEY environment variable set
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multimodal_rag import MultiModalRAGPipeline
from multimodal_rag.utils.config import (
    Config,
    ChunkingConfig,
    EmbeddingConfig,
    IndexingConfig,
    IngestionConfig,
    LLMConfig,
    RAGConfig,
    TranscriptionConfig,
)


def main():
    """Run the advanced usage example."""
    print("=" * 70)
    print("  MultiModal RAG — Advanced Usage Example")
    print("=" * 70)

    # Detect which API is configured
    freellm_base = os.getenv("FREELLM_API_BASE")
    openai_key = os.getenv("OPENAI_API_KEY")

    if freellm_base:
        print(f"\n📡 Using local LLM: {freellm_base}")
    elif openai_key:
        print("\n☁️  Using OpenAI Cloud API")
    else:
        print("\n⚠️  No API configured.")
        print("   Set FREELLM_API_BASE for local FreeLLMAPI,")
        print("   or OPENAI_API_KEY for OpenAI cloud.\n")
        return

    # --- Step 1: Custom Configuration ---
    print("\n⚙️  Step 1: Custom Configuration")
    print("-" * 40)

    config = Config()

    # Customize LLM settings
    config.llm = LLMConfig(
        model="qwen/qwen3.6-35b-a3b",  # or "gpt-4-turbo", "claude-3-opus"
        temperature=0.0,  # Deterministic for accurate transcription
        max_tokens=4096,
    )

    # Customize embedding settings
    config.embedding = EmbeddingConfig(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        dimensions=1536,
    )

    # Customize ingestion settings
    config.ingestion = IngestionConfig(
        dpi=300,  # Higher DPI = better transcription quality
        image_format="png",
        max_pages=None,  # Set to limit pages for testing (e.g., 5)
    )

    # Customize transcription settings
    config.transcription = TranscriptionConfig(
        max_concurrent=10,  # Number of parallel transcription workers
        retry_attempts=3,
    )

    # Customize chunking settings
    config.chunking = ChunkingConfig(
        chunk_size=512,  # Target chunk size in tokens
        overlap=50,  # Overlap between chunks
        min_chunk_size=100,  # Minimum tokens per chunk
    )

    # Customize indexing settings
    config.indexing = IndexingConfig(
        vector_store="chroma",  # Options: chroma, faiss, pinecone, qdrant
        collection_name="my_documents",
        persist_directory="./chroma_db",
        top_k=5,  # Number of results per query
    )

    # Customize RAG settings
    config.rag = RAGConfig(
        llm_model="gpt-4o",
        temperature=0.1,
        top_k=5,
    )

    print("   ✅ Custom configuration applied")
    print(f"   LLM: {config.llm.model}")
    print(f"   Embedding: {config.embedding.model}")
    print(f"   Vector Store: {config.indexing.vector_store}")
    print(f"   Concurrency: {config.transcription.max_concurrent}")

    # --- Step 2: Initialize Pipeline ---
    print("\n📦 Step 2: Initialize Pipeline")
    print("-" * 40)

    pipeline = MultiModalRAGPipeline(config)
    print("   ✅ Pipeline ready\n")

    # --- Step 3: Process Documents ---
    print("📄 Step 3: Process Documents")
    print("-" * 40)

    sample_dir = Path(__file__).parent / "sample_documents"
    output_dir = Path(__file__).parent.parent / "output"

    if sample_dir.exists():
        documents = list(sample_dir.glob("*.pdf")) + list(
            sample_dir.glob("*.pptx")
        )

        if documents:
            doc_paths = [str(d) for d in documents]
            print(f"   Found {len(doc_paths)} document(s)")

            result = pipeline.process_documents(doc_paths)
            print(f"   Status: {result['status']}")
            print(f"   Successful: {result['successful']}")
            print(f"   Failed: {result['failed']}\n")
        else:
            print("   ⚠️  No documents found in sample_documents/\n")
    else:
        print(f"   ⚠️  sample_documents/ directory not found\n")

    # --- Step 4: Export Results ---
    print("📤 Step 4: Export Results")
    print("-" * 40)

    if output_dir.exists() or True:  # Create if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export XML
        xml_dir = output_dir / "xml"
        pipeline.export_xml(str(xml_dir))
        print(f"   ✅ XML exported to {xml_dir}")

        # Export chunks
        chunks_dir = output_dir / "chunks"
        pipeline.export_chunks(str(chunks_dir))
        print(f"   ✅ Chunks exported to {chunks_dir}")

        # Save index
        index_dir = output_dir / "index"
        pipeline.save_index(str(index_dir))
        print(f"   ✅ Index saved to {index_dir}\n")

    # --- Step 5: Query with Sources ---
    print("🔍 Step 5: Query with Source Tracking")
    print("-" * 40)

    queries = [
        "What are the main conclusions?",
        "Summarize the key findings.",
        "What methodology was used?",
    ]

    for query in queries:
        print(f"\n   ❓ '{query}'")
        result = pipeline.query_with_sources(query)
        print(f"   📝 {result.answer[:200]}...")

        if result.metadata.get("source_references"):
            print("   📚 Sources:")
            for ref in result.metadata["source_references"][:3]:
                print(f"     - {ref}")

    # --- Step 6: Inspect Pipeline State ---
    print("\n📊 Step 6: Pipeline State")
    print("-" * 40)

    status = pipeline.get_status()
    print(f"   Documents: {status.documents_processed}")
    print(f"   Pages: {status.total_pages}")
    print(f"   Chunks: {status.total_chunks}")
    print(f"   Indexed: {status.is_indexed}")

    index_info = pipeline.get_index_info()
    print(f"   Index info: {json.dumps(index_info, indent=2)}")

    # --- Step 7: Access Intermediate Results ---
    print("\n🔬 Step 7: Access Intermediate Results")
    print("-" * 40)

    # Access transcribed documents
    docs = pipeline.get_transcribed_documents()
    print(f"   Transcribed documents: {len(docs)}")
    if docs:
        print(f"   First doc: {docs[0].source_file}")
        print(f"   Pages: {len(docs[0].pages)}")

    # Access chunks
    chunks = pipeline.get_chunks()
    print(f"   Total chunks: {len(chunks)}")
    if chunks:
        print(f"   First chunk ID: {chunks[0].chunk_id}")
        print(f"   First chunk size: {len(chunks[0].content)} chars")

    print("\n" + "=" * 70)
    print("  Advanced usage example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
