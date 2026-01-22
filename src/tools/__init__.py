#!/usr/bin/env python
"""
Tools Package - Unified tool collection

Includes:
- rag_tool: RAG retrieval tool
- web_search: Web search tool
- query_item_tool: Query item tool
- paper_search_tool: Paper search tool
- tex_downloader: LaTeX source download tool
- tex_chunker: LaTeX text chunking tool
- question: Question generation tools (pdf_parser, question_extractor, exam_mimic)
"""

# Patch lightrag.utils BEFORE any imports that use lightrag
import importlib.util
import sys

from src.logging import get_logger

# Module logger
tools_logger = get_logger("Tools")

try:
    # Directly load lightrag.utils module without triggering lightrag/__init__.py
    _spec = importlib.util.find_spec("lightrag.utils")
    if _spec and _spec.origin:
        _utils = importlib.util.module_from_spec(_spec)
        sys.modules["lightrag.utils"] = _utils
        _spec.loader.exec_module(_utils)

        # Apply patches
        for _k, _v in {
            "verbose_debug": lambda *args, **kwargs: None,
            "VERBOSE_DEBUG": False,
            "get_env_value": lambda key, default=None: default,
            "safe_unicode_decode": lambda t: (
                t.decode("utf-8", errors="ignore") if isinstance(t, bytes) else t
            ),
        }.items():
            if not hasattr(_utils, _k):
                setattr(_utils, _k, _v)

        if not hasattr(_utils, "wrap_embedding_func_with_attrs"):

            def _wrap(**attrs):
                def dec(f):
                    for k, v in attrs.items():
                        setattr(f, k, v)
                    return f

                return dec

            _utils.wrap_embedding_func_with_attrs = _wrap
except Exception as e:
    import traceback

    tools_logger.warning("Failed to patch lightrag.utils: %s", e)
    tools_logger.debug(traceback.format_exc())

__all__: list[str] = []

# Attempt to import lightweight tools lazily; if optional dependencies
# are missing, skip them and allow importing test utilities without
# requiring the full dependency stack.

try:
    from .query_item_tool import query_numbered_item

    __all__.append("query_numbered_item")
except Exception as e:  # pragma: no cover - optional dependency
    tools_logger.debug("Optional import query_item_tool unavailable: %s", e)

try:
    from .web_search import web_search

    __all__.append("web_search")
except Exception as e:  # pragma: no cover - optional dependency
    tools_logger.debug("Optional import web_search unavailable: %s", e)

# RAG and run_code are more heavy; import them only if available
try:
    from .code_executor import run_code, run_code_sync

    __all__.extend(["run_code", "run_code_sync"])
except Exception as e:  # pragma: no cover - optional dependency
    tools_logger.debug("Optional import code_executor unavailable: %s", e)

try:
    from .rag_tool import rag_search

    __all__.append("rag_search")
except Exception as e:  # pragma: no cover - optional dependency
    tools_logger.debug("Optional import rag_tool unavailable: %s", e)

# Paper research related tools (very optional)
try:
    from .paper_search_tool import PaperSearchTool
    from .tex_chunker import TexChunker
    from .tex_downloader import TexDownloader, read_tex_file

    __all__.extend(
        [
            "PaperSearchTool",
            "TexChunker",
            "TexDownloader",
            "read_tex_file",
        ]
    )
except Exception as e:  # pragma: no cover - optional dependency
    tools_logger.debug("Optional paper tools unavailable: %s", e)

# Question generation tools (lazy import to avoid circular dependencies)
# Access via: from src.tools.question import parse_pdf_with_mineru, etc.
