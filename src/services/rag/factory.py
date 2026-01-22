"""
Pipeline Factory
================

Factory for creating and managing RAG pipelines.

Note: Pipeline imports are lazy to avoid importing heavy dependencies (lightrag, llama_index, etc.)
at module load time. This allows the core services to be imported without RAG dependencies.
"""

from collections.abc import Callable
import warnings

# Pipeline registry - populated lazily
_PIPELINES: dict[str, Callable] = {}
_PIPELINES_INITIALIZED = False


def _init_pipelines():
    """Lazily initialize pipeline registry.

    Important:
    - Do NOT import optional heavy dependencies (e.g. llama_index) here.
    - Pipelines must be imported inside their factory callables, so users can
      use other providers without installing every optional dependency.
    """
    global _PIPELINES, _PIPELINES_INITIALIZED
    if _PIPELINES_INITIALIZED:
        return

    from .pipelines import AcademicPipeline, lightrag, llamaindex
    from .pipelines.raganything import RAGAnythingPipeline
    from .pipelines.raganything_docling import RAGAnythingDoclingPipeline

    _PIPELINES.update(
        {
            "raganything": RAGAnythingPipeline,  # Full multimodal: MinerU parser, deep analysis (slow, thorough)
            "raganything_docling": RAGAnythingDoclingPipeline,  # Docling parser: Office/HTML friendly, easier setup
            "lightrag": lightrag.LightRAGPipeline,  # Knowledge graph: PDFParser, fast text-only (medium speed)
            "llamaindex": llamaindex.LlamaIndexPipeline,  # Vector-only: Simple chunking, fast (fastest)
            "academic": AcademicPipeline,  # Academic pipeline: Semantic chunking + numbered item extraction
        }
    )
    _PIPELINES_INITIALIZED = True


def get_pipeline(name: str = "raganything", kb_base_dir: str | None = None, **kwargs):
    """
    Get a pre-configured pipeline by name.

    Args:
        name: Pipeline name (raganything, raganything_docling, lightrag, llamaindex)
        kb_base_dir: Base directory for knowledge bases (passed to all pipelines)
        **kwargs: Additional arguments passed to pipeline constructor

    Returns:
        Pipeline instance

    Raises:
        ValueError: If pipeline name is not found or is None
    """
    if not name:
        raise ValueError("Pipeline name must be specified.")

    _init_pipelines()
    if name not in _PIPELINES:
        available = list(_PIPELINES.keys())
        raise ValueError(f"Pipeline '{name}' not found. Available: {available}")

    factory = _PIPELINES[name]

    try:
        # Handle different pipeline types:
        # - lightrag: callable that accepts kb_base_dir and returns a composed RAGPipeline
        # - llamaindex, raganything, raganything_docling: callables that instantiate class-based pipelines
        if name in ("lightrag",):
            return factory(kb_base_dir=kb_base_dir, **kwargs)

        if kb_base_dir:
            kwargs["kb_base_dir"] = kb_base_dir
        return factory(**kwargs)
    except ImportError as e:
        # Common case: user didn't install optional RAG backend deps (e.g. llama_index).
        raise ValueError(
            f"Pipeline '{name}' is not available because an optional dependency is missing: {e}. "
            f"Please install the required dependency for '{name}', or switch provider to 'raganything'/'lightrag'."
        ) from e


def list_pipelines() -> list[dict[str, str]]:
    """
    List available pipelines.

    Returns:
        List of pipeline info dictionaries
    """
    return [
        {
            "id": "llamaindex",
            "name": "LlamaIndex",
            "description": "Pure vector retrieval, fastest processing speed.",
        },
        {
            "id": "lightrag",
            "name": "LightRAG",
            "description": "Lightweight knowledge graph retrieval, fast processing of text documents.",
        },
        {
            "id": "raganything",
            "name": "RAG-Anything (MinerU)",
            "description": "Multimodal document processing with MinerU parser. Best for academic PDFs with complex equations and formulas.",
        },
        {
            "id": "raganything_docling",
            "name": "RAG-Anything (Docling)",
            "description": "Multimodal document processing with Docling parser. Better for Office documents (.docx, .pptx) and HTML. Easier to install.",
        },
        {
            "id": "academic",
            "name": "Academic",
            "description": "Academic document pipeline: semantic chunking and numbered item extraction for papers and textbooks.",
        },
    ]


def register_pipeline(name: str, factory: Callable):
    """
    Register a custom pipeline.

    Args:
        name: Pipeline name
        factory: Factory function or class that creates the pipeline
    """
    _init_pipelines()
    _PIPELINES[name] = factory


def has_pipeline(name: str) -> bool:
    """
    Check if a pipeline exists.

    Args:
        name: Pipeline name

    Returns:
        True if pipeline exists
    """
    if not name:
        return False

    _init_pipelines()
    return name in _PIPELINES


# Backward compatibility with old plugin API
def get_plugin(name: str) -> dict[str, Callable]:
    """
    DEPRECATED: Use get_pipeline() instead.

    Get a plugin by name (maps to pipeline API).
    """
    warnings.warn(
        "get_plugin() is deprecated, use get_pipeline() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    pipeline = get_pipeline(name)
    return {
        "initialize": pipeline.initialize,
        "search": pipeline.search,
        "delete": getattr(pipeline, "delete", lambda kb: True),
    }


def list_plugins() -> list[dict[str, str]]:
    """
    DEPRECATED: Use list_pipelines() instead.
    """
    warnings.warn(
        "list_plugins() is deprecated, use list_pipelines() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return list_pipelines()


def has_plugin(name: str) -> bool:
    """
    DEPRECATED: Use has_pipeline() instead.
    """
    warnings.warn(
        "has_plugin() is deprecated, use has_pipeline() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return has_pipeline(name)
