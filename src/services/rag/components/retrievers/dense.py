"""
Dense Retriever
===============

Dense vector-based retriever using FAISS or cosine similarity.
"""

import json
from pathlib import Path
import pickle
from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseComponent


class DenseRetriever(BaseComponent):
    """
    Dense vector retriever.

    Uses FAISS for fast similarity search or falls back to
    cosine similarity if FAISS is unavailable.
    """

    name = "dense_retriever"

    def __init__(self, kb_base_dir: Optional[str] = None, top_k: int = 5):
        """
        Initialize dense retriever.

        Args:
            kb_base_dir: Base directory for knowledge bases
            top_k: Number of results to return
        """
        super().__init__()
        self.kb_base_dir = kb_base_dir or str(
            Path(__file__).resolve().parent.parent.parent.parent.parent.parent
            / "data"
            / "knowledge_bases"
        )
        self.top_k = top_k

        # Try to import FAISS
        self.use_faiss = False
        try:
            import faiss

            self.faiss = faiss
            self.use_faiss = True
        except ImportError:
            self.logger.warning("FAISS not available, using simple cosine similarity")

    async def process(self, query: str, kb_name: str, **kwargs) -> Dict[str, Any]:
        """
        Search using dense embeddings with FAISS or cosine similarity.

        Args:
            query: Search query
            kb_name: Knowledge base name
            **kwargs: Additional arguments (mode, top_k, etc.)

        Returns:
            Search results dictionary with answer and sources
        """
        top_k = kwargs.get("top_k", self.top_k)
        self.logger.info(f"Dense search in {kb_name}: {query[:50]}... (top_k={top_k})")

        from src.services.embedding import get_embedding_client

        # Get query embedding
        client = get_embedding_client()
        query_embedding = np.array((await client.embed([query]))[0], dtype=np.float32)

        # Load index
        kb_dir = Path(self.kb_base_dir) / kb_name / "vector_store"
        metadata_file = kb_dir / "metadata.json"
        info_file = kb_dir / "info.json"

        if not metadata_file.exists():
            self.logger.warning(f"No vector index found at {kb_dir}")
            return {
                "query": query,
                "answer": "No documents indexed. Please upload documents first.",
                "content": "",
                "mode": "dense",
                "provider": "llamaindex",
                "results": [],
            }

        # Load metadata and info
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        use_faiss = info.get("use_faiss", False)

        if use_faiss and self.use_faiss:
            # Use FAISS for fast search
            index_file = kb_dir / "index.faiss"
            if not index_file.exists():
                self.logger.error(f"FAISS index file not found: {index_file}")
                return self._empty_response(query)

            # Load FAISS index
            index = self.faiss.read_index(str(index_file))

            # Normalize query vector for cosine similarity
            query_vec = query_embedding.reshape(1, -1)
            self.faiss.normalize_L2(query_vec)

            # Search
            distances, indices = index.search(query_vec, min(top_k, len(metadata)))

            # Build results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(metadata):  # Valid index
                    # NOTE: This conversion is specific to a FAISS IndexFlatL2 index where BOTH
                    # the stored vectors and the query vector have been L2-normalized to unit
                    # length. The stored vectors are normalized during indexing (see vector.py
                    # line 121), and the query vector is normalized just before search above.
                    #
                    # FAISS IndexFlatL2 returns squared L2 distances for these unit-normalized
                    # vectors. For unit vectors:
                    #   L2_distance² = 2 - 2 * cosine_similarity
                    #   ⇒ cosine_similarity = 1 - L2_distance² / 2
                    #
                    similarity = (
                        1.0 - dist / 2
                    )  # Convert squared L2 distance to cosine-like similarity for unit vectors
                    similarity = max(0.0, min(1.0, similarity))  # Clamp to [0,1]
                    results.append((similarity, metadata[idx]))
        else:
            # Fallback: Load embeddings and use cosine similarity
            embeddings_file = kb_dir / "embeddings.pkl"
            if not embeddings_file.exists():
                self.logger.error(f"Embeddings file not found: {embeddings_file}")
                return self._empty_response(query)

            with open(embeddings_file, "rb") as f:
                embeddings = pickle.load(f)

            # Normalize for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 1e-8:
                query_vec = query_embedding / query_norm
            else:
                # If the query embedding is (near) zero, use a zero vector to avoid
                # numerical instability and produce zero similarity scores.
                query_vec = np.zeros_like(query_embedding)

            doc_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Safely normalize document embeddings; documents with (near) zero norm
            # become zero vectors instead of being scaled by a large factor.
            doc_vecs = np.divide(
                embeddings,
                doc_norms,
                out=np.zeros_like(embeddings),
                where=doc_norms > 1e-8,
            )

            # Compute similarities
            similarities = np.dot(doc_vecs, query_vec)

            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                results.append((score, metadata[idx]))

        # Build response content
        # Format chunks cleanly for LLM context (without score annotations)
        content_parts = []
        sources = []
        for score, item in results:
            content = item.get("content", "").strip()
            if content:  # Only include non-empty chunks
                # Add chunk without score prefix for clean LLM input
                content_parts.append(content)
                sources.append(
                    {
                        "content": content,
                        "score": score,
                        "metadata": item.get("metadata", {}),
                    }
                )

        # Join chunks with clear separation
        content = "\n\n".join(content_parts)

        return {
            "query": query,
            "answer": content,  # Return clean context for LLM to use
            "content": content,
            "mode": "dense",
            "provider": "llamaindex",
            "results": sources,
        }

    def _empty_response(self, query: str) -> Dict[str, Any]:
        """Return empty response when no results found."""
        return {
            "query": query,
            "answer": "No relevant documents found.",
            "content": "",
            "mode": "dense",
            "provider": "llamaindex",
            "results": [],
        }
