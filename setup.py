"""Setup script for multimodal_rag package."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="multimodal-rag",
    version="1.0.0",
    description=(
        "A Hybrid Multi-Modal Retrieval Augmented Generation Pipeline. "
        "Converts PDF, PPTX documents into XML via multi-modal LLMs, "
        "semantically chunks them, indexes into a vector database, "
        "and provides a RAG pipeline for querying."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MultiModal RAG Contributors",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.0",
        "anthropic>=0.20.0",
        "pdf2image>=1.16.0",
        "python-pptx>=0.6.23",
        "Pillow>=10.0.0",
        "langchain>=0.2.0",
        "langchain-openai>=0.1.0",
        "langchain-community>=0.2.0",
        "langchain-chroma>=0.1.0",
        "chromadb>=0.4.0",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "pinecone": [
            "pinecone>=4.0.0",
            "langchain-pinecone>=0.2.0",
        ],
        "qdrant": [
            "qdrant-client>=1.6.0",
            "langchain-qdrant>=0.2.0",
        ],
        "semantic-chunking": ["langchain-experimental>=0.0.30"],
        "ocr": ["pytesseract>=0.3.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
