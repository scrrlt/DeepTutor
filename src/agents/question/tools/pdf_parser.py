import pdfplumber
import logging
import gc
import os
import sys
from typing import Generator, Tuple

# src/agents/question/tools/pdf_parser.py

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Memory-efficient PDF parser that yields content page-by-page.
    Designed for low-resource environments (APUs, shared RAM).

    Guarantees:
    - At most one live pdfplumber Page object at a time.
    - Explicitly breaks pdfplumber's internal page cache to prevent linear growth.
    - Forces cleanup after file close.
    """

    def parse_generator(self, file_path: str) -> Generator[Tuple[str, int, int], None, None]:
        """
        Yields (text, page_number, byte_size) tuples.

        byte_size is an approximate Python-side footprint (sys.getsizeof).
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        try:
            # pdfplumber.open returns a ContextManager
            with pdfplumber.open(file_path) as pdf:
                # Use len() to get count without loading all pages
                total_pages = len(pdf.pages)
                logger.info(f"Opened PDF with {total_pages} pages: {file_path}")
                
                # Iterate by index to avoid loading all page objects into a list
                for i in range(total_pages):
                    page = None
                    try:
                        # Lazy load single page
                        page = pdf.pages[i]
                        
                        # Extract text (layout=False is faster and lighter)
                        text = page.extract_text(layout=False) or ""

                        # Approximate Python memory footprint
                        byte_size = sys.getsizeof(text)
                        
                        # Yield result
                        yield text, (i + 1), byte_size
                        
                    except Exception as e:
                        logger.error(f"Error parsing page {i + 1} of {file_path}: {e}")
                        yield "", (i + 1), 0
                    finally:
                        # Flush page-level caches if present
                        if page is not None:
                            try:
                                page.flush_cache()
                            except Exception:
                                pass

                        # CRITICAL FIX:
                        # pdfplumber caches Page objects in pdf.pages.
                        # Deleting the local variable is NOT sufficient.
                        # We must explicitly break the reference in the list.
                        try:
                            pdf.pages[i] = None
                        except Exception:
                            pass

                        del page

        except Exception as e:
            logger.error(f"Critical error opening PDF {file_path}: {e}")
            raise
        finally:
            # Force cleanup after file close to reclaim buffers and file handles
            gc.collect()

    def parse(self, file_path: str) -> str:
        """
        Legacy method.

        WARNING:
        Loads the entire document into memory.
        Use parse_generator() for large files.
        """
        logger.warning(f"Using non-streaming parse for {file_path}. High memory usage risk.")
        full_text = []
        for text, _, _ in self.parse_generator(file_path):
            full_text.append(text)
        return "\n".join(full_text)