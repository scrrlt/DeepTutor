"""
Pipeline Factory
================

Factory for creating and managing RAG pipelines.
"""

from typing import Callable, Dict, List, Optional

# Pipeline registry (deprecated, using lazy loading in get_pipeline)
_PIPELINES: Dict[str, Callable] = {}


def get_pipeline(name: str = "raganything", kb_base_dir: Optional[str] = None, **kwargs):
    """
    Get a pre-configured pipeline by name.

    Args:
        name: Pipeline name (raganything, lightrag, llamaindex, academic)
        kb_base_dir: Base directory for knowledge bases (passed to all pipelines)
        **kwargs: Additional arguments passed to pipeline constructor

    Returns:
        Pipeline instance

    Raises:
        ValueError: If pipeline name is not found
    """
    # Lazy imports to prevent eager failures if dependencies are missing
    if name == "lightrag":
        from .pipelines.lightrag import LightRAGPipeline
        return LightRAGPipeline(kb_base_dir=kb_base_dir)
    
    elif name == "academic":
        from .pipelines.academic import AcademicPipeline
        return AcademicPipeline(kb_base_dir=kb_base_dir)

    elif name == "llamaindex":
        from .pipelines.llamaindex import LlamaIndexPipeline
        return LlamaIndexPipeline(**(kwargs | {"kb_base_dir": kb_base_dir} if kb_base_dir else kwargs))

    elif name == "raganything":
        from .pipelines.raganything import RAGAnythingPipeline
        return RAGAnythingPipeline(**(kwargs | {"kb_base_dir": kb_base_dir} if kb_base_dir else kwargs))

    available = ["lightrag", "academic", "llamaindex", "raganything"]
    raise ValueError(f"Unknown pipeline: {name}. Available: {available}")


def list_pipelines() -> List[Dict[str, str]]:
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
            "name": "RAG-Anything",
            "description": "Multimodal document processing with chart and formula extraction, builds knowledge graphs.",
        },
    ]


def register_pipeline(name: str, factory: Callable):
    """
    Register a custom pipeline.

    Args:
        name: Pipeline name
        factory: Factory function or class that creates the pipeline
    """
    _PIPELINES[name] = factory


def has_pipeline(name: str) -> bool:
    """
    Check if a pipeline exists.

    Args:
        name: Pipeline name

    Returns:
        True if pipeline exists
    """
    return name in _PIPELINES


# Backward compatibility with old plugin API
def get_plugin(name: str) -> Dict[str, Callable]:
    """
    DEPRECATED: Use get_pipeline() instead.

    Get a plugin by name (maps to pipeline API).
    """
    import warnings

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


def list_plugins() -> List[Dict[str, str]]:
    """
    DEPRECATED: Use list_pipelines() instead.
    """
    import warnings

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
    import warnings

    warnings.warn(
        "has_plugin() is deprecated, use has_pipeline() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return has_pipeline(name)
