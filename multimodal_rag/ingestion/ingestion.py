"""
DocumentIngestor - Handles document ingestion and page-to-image conversion.

Supports PDF and PPTX formats, converting each page/slide into high-quality
images for subsequent multi-modal LLM processing.

Uses pdf2image for PDF rendering and python-pptx for PPTX processing.
"""

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from multimodal_rag.utils.config import IngestionConfig
from multimodal_rag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageImage:
    """Represents a single page/slide as an image."""

    page_number: int
    image_data: bytes  # Raw image bytes
    source_file: str  # Original file path
    image_format: str = "png"

    def to_bytes(self) -> bytes:
        """Return the raw image bytes."""
        return self.image_data

    def save(self, output_dir: str) -> str:
        """Save the image to a file.

        Args:
            output_dir: Directory to save the image.

        Returns:
            Path to the saved image file.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"page_{self.page_number:04d}.{self.image_format}"
        )
        with open(output_path, "wb") as f:
            f.write(self.image_data)
        logger.debug(f"Saved page image to {output_path}")
        return output_path


class DocumentIngestor:
    """Converts documents (PDF, PPTX) into page images for processing.

    This class handles:
    - PDF page rendering to images at configurable DPI
    - PPTX slide rendering to images
    - Image quality and format configuration
    - Page count limiting for testing

    Usage:
        ingestor = DocumentIngestor()
        images = ingestor.ingest("document.pdf")
    """

    SUPPORTED_FORMATS = {".pdf", ".pptx", ".ppt"}

    def __init__(self, config: Optional[IngestionConfig] = None):
        """Initialize the document ingestor.

        Args:
            config: Configuration for ingestion parameters.
        """
        self.config = config or IngestionConfig()
        logger.info(
            f"DocumentIngestor initialized: DPI={self.config.dpi}, "
            f"format={self.config.image_format}, max_pages={self.config.max_pages}"
        )

    def ingest(self, file_path: str) -> list[PageImage]:
        """Ingest a document and convert all pages to images.

        Args:
            file_path: Path to the document file (PDF or PPTX).

        Returns:
            List of PageImage objects, one per page/slide.

        Raises:
            ValueError: If the file format is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{extension}'. Supported: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Ingesting document: {file_path}")

        if extension == ".pdf":
            images = self._ingest_pdf(str(path))
        elif extension in (".pptx", ".ppt"):
            images = self._ingest_pptx(str(path))

        # Apply page limit if set
        if self.config.max_pages and len(images) > self.config.max_pages:
            images = images[: self.config.max_pages]
            logger.info(f"Limited to first {self.config.max_pages} pages")

        logger.info(f"Successfully ingested {len(images)} pages from {file_path}")
        return images

    def _ingest_pdf(self, file_path: str) -> list[PageImage]:
        """Convert PDF pages to images using pdf2image.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of PageImage objects.
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image is required for PDF ingestion. "
                "Install with: pip install pdf2image\n"
                "Also install poppler on your system:\n"
                "  macOS: brew install poppler\n"
                "  Ubuntu: sudo apt-get install poppler-utils"
            )

        try:
            images = convert_from_path(
                file_path,
                dpi=self.config.dpi,
                fmt=self.config.image_format,
                jpegopt={"quality": 95},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}") from e

        page_images = []
        for page_num, image in enumerate(images, start=1):
            buffer = io.BytesIO()
            image.save(buffer, format=self.config.image_format.upper())
            page_images.append(
                PageImage(
                    page_number=page_num,
                    image_data=buffer.getvalue(),
                    source_file=file_path,
                    image_format=self.config.image_format,
                )
            )

        logger.info(f"Converted {len(images)} PDF pages to images")
        return page_images

    def _ingest_pptx(self, file_path: str) -> list[PageImage]:
        """Convert PPTX slides to images.

        Uses python-pptx to extract slide content and renders each slide
        to an image using PIL.

        Args:
            file_path: Path to the PPTX file.

        Returns:
            List of PageImage objects.
        """
        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ImportError:
            raise ImportError(
                "python-pptx is required for PPTX ingestion. "
                "Install with: pip install python-pptx"
            )

        try:
            prs = Presentation(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PPTX file: {e}") from e

        page_images = []
        slide_width = Inches(self.config.slide_width_inches)
        slide_height = Inches(self.config.slide_height_inches)

        for slide_num, slide in enumerate(prs.slides, start=1):
            # Create a blank slide with the same dimensions
            from pptx.util import Inches as InchesUtil

            blank_slide_layout = prs.slide_layouts[6]  # Blank layout
            blank_slide = prs.slides.add_slide(blank_slide_layout)
            blank_slide.width = slide.width
            blank_slide.height = slide.height

            # Copy all elements from original slide to blank slide
            for shape in slide.shapes:
                new_shape = blank_slide.shapes.copy_shape(shape)
                new_shape.left = shape.left
                new_shape.top = shape.top
                new_shape.width = shape.width
                new_shape.height = shape.height
                new_shape.rotation = shape.rotation

            # Render the blank slide to an image
            image_path = f"/tmp/slide_{slide_num}.png"
            blank_slide.save_as(image_path)

            with open(image_path, "rb") as f:
                image_data = f.read()

            page_images.append(
                PageImage(
                    page_number=slide_num,
                    image_data=image_data,
                    source_file=file_path,
                    image_format="png",
                )
            )

            # Clean up temp file
            if os.path.exists(image_path):
                os.remove(image_path)

        logger.info(f"Converted {len(page_images)} PPTX slides to images")
        return page_images

    def get_supported_formats(self) -> set[str]:
        """Return the set of supported file extensions."""
        return self.SUPPORTED_FORMATS.copy()
