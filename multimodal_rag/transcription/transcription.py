"""
XMLTranscriber - Multi-modal LLM transcription of page images to XML.

Uses multi-modal LLMs (GPT-4o, Claude, etc.) to transcribe page images
into structured XML format, preserving document order, tables, flowcharts,
and other visual elements in a machine-readable format.

Features:
- Parallel processing for fast transcription
- Retry logic for failed requests
- Configurable system prompts
- Support for multiple LLM providers (OpenAI, Anthropic, Azure, etc.)

Usage:
    transcriber = XMLTranscriber()
    xml_docs = transcriber.transcribe(page_images)
"""

import asyncio
import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from multimodal_rag.ingestion import PageImage
from multimodal_rag.utils.config import TranscriptionConfig
from multimodal_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscribedDocument:
    """Represents a fully transcribed document in XML format."""

    source_file: str
    pages: list[dict] = field(default_factory=list)  # page_number -> xml_content
    full_xml: str = ""

    def get_page_xml(self, page_number: int) -> Optional[str]:
        """Get XML content for a specific page."""
        for page in self.pages:
            if page["page_number"] == page_number:
                return page["content"]
        return None

    def get_all_xml(self) -> str:
        """Get all page XML concatenated."""
        return self.full_xml


class XMLTranscriber:
    """Transcribes page images into structured XML using multi-modal LLMs.

    This class handles:
    - Sending page images to multi-modal LLMs for transcription
    - Parallel processing with configurable concurrency
    - Retry logic for failed requests
    - XML extraction and validation
    - Support for multiple LLM providers

    The LLM is prompted to output structured XML that preserves:
    - Document structure (headings, paragraphs, lists)
    - Tables with row/cell structure
    - Figures and images with descriptions
    - Flowcharts and diagrams
    - Equations in LaTeX format
    - Exact content order

    Usage:
        transcriber = XMLTranscriber()
        documents = transcriber.transcribe(page_images)
    """

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """Initialize the XML transcriber.

        Args:
            config: Configuration for transcription parameters.
        """
        self.config = config or TranscriptionConfig()
        self._check_llm_dependencies()
        logger.info(
            f"XMLTranscriber initialized: model={self.config.model}, "
            f"concurrency={self.config.max_concurrent}"
        )

    def _check_llm_dependencies(self):
        """Verify that required LLM libraries are installed."""
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai is required for LLM transcription. "
                "Install with: pip install openai"
            )

    def transcribe(self, page_images: list[PageImage]) -> list[TranscribedDocument]:
        """Transcribe a list of page images into XML documents.

        Uses parallel processing to transcribe multiple pages simultaneously.

        Args:
            page_images: List of PageImage objects to transcribe.

        Returns:
            List of TranscribedDocument objects, one per source file.
        """
        if not page_images:
            logger.warning("No page images provided for transcription")
            return []

        logger.info(
            f"Starting transcription of {len(page_images)} pages "
            f"(concurrency={self.config.max_concurrent})"
        )

        # Group pages by source file
        pages_by_file: dict[str, list[PageImage]] = {}
        for page in page_images:
            if page.source_file not in pages_by_file:
                pages_by_file[page.source_file] = []
            pages_by_file[page.source_file].append(page)

        # Transcribe each file's pages in parallel
        documents = []
        for file_path, pages in pages_by_file.items():
            logger.info(f"Transcribing {len(pages)} pages from {file_path}")
            transcribed_pages = self._transcribe_pages_parallel(pages)
            documents.append(
                self._build_document(file_path, transcribed_pages)
            )

        logger.info(f"Transcription complete: {len(documents)} documents")
        return documents

    def _build_document(self, file_path: str, pages: list[dict]) -> TranscribedDocument:
        """Build a TranscribedDocument from transcribed pages.

        Args:
            file_path: Source file path.
            pages: List of dicts with page_number and content keys.

        Returns:
            TranscribedDocument with all pages concatenated.
        """
        full_xml = ""
        for page in sorted(pages, key=lambda x: x["page_number"]):
            full_xml += page["content"] + "\n\n"

        return TranscribedDocument(
            source_file=file_path,
            pages=pages,
            full_xml=full_xml,
        )

    def _transcribe_pages_parallel(
        self, pages: list[PageImage]
    ) -> list[dict]:
        """Transcribe pages in parallel using thread pool.

        Args:
            pages: List of pages from the same source file.

        Returns:
            List of dicts with page_number and content keys.
        """
        results = []

        with ThreadPoolExecutor(
            max_workers=self.config.max_concurrent
        ) as executor:
            futures = {
                executor.submit(self._transcribe_single_page, page): page
                for page in pages
            }

            for future in as_completed(futures):
                page = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Failed to transcribe page {page.page_number}: {e}"
                    )
                    # Add empty content as fallback
                    results.append(
                        {
                            "page_number": page.page_number,
                            "content": f"<page number='{page.page_number}'><error>Transcription failed: {str(e)}</error></page>",
                        }
                    )

        # Sort by page number
        results.sort(key=lambda x: x["page_number"])
        return results

    def _transcribe_single_page(self, page: PageImage) -> dict:
        """Transcribe a single page image to XML.

        Args:
            page: A single page image.

        Returns:
            Dict with page_number and XML content.

        Raises:
            RuntimeError: If LLM API call fails after all retries.
        """
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                xml_content = self._call_llm(page.image_data)
                return {
                    "page_number": page.page_number,
                    "content": xml_content,
                }
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.config.retry_attempts} "
                    f"failed for page {page.page_number}: {e}"
                )
                if attempt < self.config.retry_attempts:
                    import time

                    time.sleep(self.config.retry_delay * attempt)

        raise RuntimeError(
            f"Failed to transcribe page {page.page_number} after "
            f"{self.config.retry_attempts} attempts"
        )

    def _call_llm(self, image_bytes: bytes) -> str:
        """Call the multi-modal LLM to transcribe an image to XML.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            XML string transcribed by the LLM.
        """
        # Convert image bytes to base64 for API call
        import base64

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine the API client based on model
        model = self.config.model.lower()

        if "gpt" in model or "4o" in model:
            return self._call_openai(base64_image)
        elif "claude" in model:
            return self._call_anthropic(base64_image)
        else:
            return self._call_openai(base64_image)  # Default to OpenAI

    def _call_openai(self, base64_image: str) -> str:
        """Call OpenAI-compatible API for XML transcription.

        Strategy for local models (LM Studio, FreeLLMAPI, etc.):
        1. Try vision mode first (for models like Qwen2-VL, GPT-4o)
        2. If vision fails, fall back to text-only mode with OCR pre-processing
        3. If all else fails, return an error placeholder

        Args:
            base64_image: Base64-encoded image string.

        Returns:
            XML string from the LLM.
        """
        import openai

        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

        # Try vision mode first (multi-modal LLM)
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.config.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Transcribe this document page into structured XML. "
                                "Preserve all text, tables, figures, and structure. "
                                "Maintain the exact order of content.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            content = response.choices[0].message.content
            return self._extract_xml(content)

        except Exception as vision_error:
            logger.warning(
                f"Vision mode failed (model may not support images): {type(vision_error).__name__}: {str(vision_error)[:200]}. "
                f"Falling back to text-only mode."
            )
            return self._call_text_only(client, base64_image)

    def _call_text_only(self, client, base64_image: str) -> str:
        """Fallback: transcribe using text-only mode.

        When the model doesn't support vision, we use one of these strategies:
        1. If Tesseract OCR is available, extract text from the image first
        2. Otherwise, send a text-only prompt with image metadata

        This handles LM Studio's local text-only models (e.g., Qwen 32B).

        Args:
            client: OpenAI client instance.
            base64_image: Base64-encoded image string.

        Returns:
            XML string from the LLM.
        """
        # Try to extract text from image using Tesseract OCR
        extracted_text = ""
        try:
            from PIL import Image
            import io
            import pytesseract

            image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(image)
            logger.info("Tesseract OCR extracted text successfully")
        except ImportError:
            logger.debug("Tesseract not available, using text-only fallback")
        except Exception as e:
            logger.debug(f"Tesseract OCR failed: {e}, using text-only fallback")

        # Build the user message
        if extracted_text.strip():
            # Use OCR-extracted text as a starting point
            user_content = (
                f"Here is the text extracted from this document page using OCR: "
                f"{extracted_text}\n\n"
                f"Please reorganize this text into structured XML format. "
                f"Preserve all text, tables, figures, and structure. "
                f"Maintain the exact order of content."
            )
        else:
            # Fallback: send image dimensions as context
            try:
                from PIL import Image
                import io

                image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
                text_hint = (
                    f"Note: This is a page image of size {image.width}x{image.height} pixels. "
                    "Please transcribe all visible text content into structured XML."
                )
            except Exception:
                text_hint = "This is a document page image. Please transcribe all visible text into structured XML."

            user_content = f"{text_hint}\n\nTranscribe this document page into structured XML. "
            "Preserve all text, tables, figures, and structure. "
            "Maintain the exact order of content."

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": self.config.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        content = response.choices[0].message.content
        return self._extract_xml(content)

    def _call_anthropic(self, base64_image: str) -> str:
        """Call Anthropic's Claude for XML transcription.

        Args:
            base64_image: Base64-encoded image string.

        Returns:
            XML string from the LLM.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self.config.api_key)

        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this document page into structured XML. "
                            "Preserve all text, tables, figures, and structure. "
                            "Maintain the exact order of content.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                    ],
                },
            ],
        )

        content = response.content[0].text
        return self._extract_xml(content)

    @staticmethod
    def _extract_xml(content: str) -> str:
        """Extract XML content from LLM response.

        Handles cases where the LLM wraps XML in markdown code blocks.

        Args:
            content: Raw LLM response.

        Returns:
            Cleaned XML string.
        """
        # Try to extract XML from markdown code blocks
        xml_pattern = r"```xml\s*(.*?)\s*```"
        match = re.search(xml_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract from generic code blocks
        code_pattern = r"```\s*(.*?)\s*```"
        match = re.search(code_pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return as-is if no code blocks found
        return content.strip()

    def transcribe_from_file(
        self, file_path: str, output_dir: Optional[str] = None
    ) -> TranscribedDocument:
        """Convenience method: ingest and transcribe a single file.

        Args:
            file_path: Path to the document file.
            output_dir: Optional directory to save transcribed XML files.

        Returns:
            TranscribedDocument with XML content.
        """
        from multimodal_rag.ingestion import DocumentIngestor

        # Step 1: Ingest pages
        ingestor = DocumentIngestor()
        page_images = ingestor.ingest(file_path)

        # Step 2: Transcribe to XML
        documents = self.transcribe(page_images)

        if not documents:
            raise RuntimeError("No documents were transcribed")

        doc = documents[0]

        # Save to file if output_dir specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            filename = Path(file_path).stem
            output_path = os.path.join(output_dir, f"{filename}.xml")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc.full_xml)
            logger.info(f"Saved transcribed XML to {output_path}")

        return doc
