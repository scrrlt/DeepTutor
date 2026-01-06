"""
Pipeline Factory
================

Factory for creating and managing RAG pipelines.
"""

from typing import Callable, Dict, List, Union

from .pipeline import RAGPipeline
from .pipelines.raganything import RAGAnythingPipeline
from .pipelines import lightrag, llamaindex, academic


# Pipeline registry
_PIPELINES: Dict[str, Callable] = {
    "raganything": RAGAnythingPipeline,
    "lightrag": lightrag.LightRAGPipeline,
    "llamaindex": llamaindex.LlamaIndexPipeline,
    "academic": academic.AcademicPipeline,
}


def get_pipeline(
    name: str = "raganything",
    **kwargs
) -> Union[RAGPipeline, RAGAnythingPipeline]:
    """
    Get a pre-configured pipeline by name.
    
    Args:
        name: Pipeline name (raganything, lightrag, llamaindex, academic)
        **kwargs: Additional arguments passed to pipeline constructor
        
    Returns:
        Pipeline instance
        
    Raises:
        ValueError: If pipeline name is not found
    """
    if name not in _PIPELINES:
        available = list(_PIPELINES.keys())
        raise ValueError(f"Unknown pipeline: {name}. Available: {available}")
    
    factory = _PIPELINES[name]
    
    # RAGAnythingPipeline takes kwargs, others are factory functions
    if name == "raganything":
        return factory(**kwargs) if kwargs else factory()
    else:
        # Factory functions don't take kwargs
        return factory()


def list_pipelines() -> List[Dict[str, str]]:
    """
    List available pipelines.
    
    Returns:
        List of pipeline info dictionaries
    """
    return [
        {
            "id": "raganything",
            "name": "RAG-Anything",
            "description": "End-to-end academic document processing (MinerU + LightRAG)",
        },
        {
            "id": "lightrag",
            "name": "LightRAG",
            "description": "Component-based pipeline with knowledge graph",
        },
        {
            "id": "llamaindex",
            "name": "LlamaIndex",
            "description": "Fast vector-based retrieval",
        },
        {
            "id": "academic",
            "name": "Academic",
            "description": "Academic documents with numbered item extraction",
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

