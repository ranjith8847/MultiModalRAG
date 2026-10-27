"""
Basic Usage Example for MultiModal RAG Pipeline

Demonstrates the simplest way to use the MultiModal RAG pipeline:
1. Process a document (PDF or PPTX)
2. Query the processed documents

By default, this uses your local LM Studio endpoint.
To use cloud APIs, set OPENAI_API_KEY instead.

Usage:
    python examples/basic_usage.py

Requirements:
    - A PDF or PPTX file to process
    - LM Studio running on localhost:1234 (or set OPENAI_API_KEY for cloud)
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from multimodal_rag import MultiModalRAGPipeline


def main():
    """Run the basic usage example."""
    print("=" * 70)
    print("  MultiModal RAG — Basic Usage Example")
    print("=" * 70)

    # Detect which API is configured
    lm_studio_base = os.getenv("LM_STUDIO_API_BASE")
    freellm_base = os.getenv("FREELLM_API_BASE")
    openai_key = os.getenv("OPENAI_API_KEY")

    if lm_studio_base:
        print(f"\n🎓 Using LM Studio: {lm_studio_base}")
        model = os.getenv("LM_STUDIO_MODEL", "not set")
        print(f"   Model: {model}")
    elif freellm_base:
        print(f"\n📡 Using FreeLLMAPI: {freellm_base}")
    elif openai_key:
        print("\n☁️  Using OpenAI Cloud API")
    else:
        print("\n⚠️  No API configured.")
        print("   Set LM_STUDIO_API_BASE for local LM Studio,")
        print("   FREELLM_API_BASE for FreeLLMAPI,")
        print("   or OPENAI_API_KEY for OpenAI cloud.\n")
        return

    # Initialize the pipeline with default configuration
    print("\n📦 Initializing pipeline...")
    pipeline = MultiModalRAGPipeline()
    print("   ✅ Pipeline ready\n")

    # Find a document to process
    sample_dir = Path(__file__).parent / "sample_documents"
    if not sample_dir.exists():
        print(f"⚠️  No sample documents found in {sample_dir}")
        print("   Place a PDF or PPTX file in examples/sample_documents/\n")
        return

    # Get the first document
    documents = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.pptx"))
    if not documents:
        print(f"⚠️  No PDF or PPTX files found in {sample_dir}")
        print("   Place a document there and try again.\n")
        return

    document_path = str(documents[0])
    print(f"📄 Processing: {document_path}\n")

    # Process the document through the full pipeline
    result = pipeline.process_document(document_path)

    if result.get("status") == "success":
        print(f"✅ Document processed successfully!")
        print(f"   Pages: {result['pages']}")
        print(f"   Chunks: {result['total_chunks']}\n")

        # Query the processed documents
        print("=" * 70)
        print("  Querying the processed documents...")
        print("=" * 70 + "\n")

        # Example queries
        queries = [
            "What is the main topic of this document?",
            "Summarize the key points.",
        ]

        for query in queries:
            print(f"❓ Query: '{query}'")
            rag_result = pipeline.query(query)
            print(f"📝 Answer: {rag_result.answer}")
            print(f"📊 Sources: {rag_result.metadata.get('num_sources', 0)}")

            if rag_result.sources:
                print("   Sources:")
                for i, source in enumerate(rag_result.sources[:3], 1):
                    page = source.get("page_number", "N/A")
                    content_preview = source.get("content", "")[:150]
                    print(f"     [{i}] Page {page}: {content_preview}...")
            print()

        # Show pipeline status
        status = pipeline.get_status()
        print("=" * 70)
        print("  Pipeline Status")
        print("=" * 70)
        print(f"   Documents processed: {status.documents_processed}")
        print(f"   Total pages: {status.total_pages}")
        print(f"   Total chunks: {status.total_chunks}")
        print(f"   Indexed: {status.is_indexed}")

    else:
        print(f"❌ Processing failed: {result.get('error')}")


if __name__ == "__main__":
    main()
