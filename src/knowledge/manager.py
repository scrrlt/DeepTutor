# src/knowledge/manager.py
"""
KnowledgeBaseManager

Streaming ingestion manager with adaptive batching, memory safety, and pluggable vector-store adapter.
Designed for use in CI and production with APUs and constrained hosts.
"""

import os
import gc
import time
import logging
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple

# Check if torch is available for VRAM cleanup
try:
    import torch
except Exception:
    torch = None

# Try importing psutil for adaptive tuning
try:
    import psutil
except Exception:
    psutil = None

from src.core.logging import get_logger
from src.agents.question.tools.pdf_parser import PDFParser

logger = get_logger("KnowledgeManager")

# --- Tunable defaults ---
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_BYTES = 5_000_000
DEFAULT_MAX_WAIT_MS = 5_000
SAFETY_THRESHOLD_PERCENT = 75.0
CRITICAL_THRESHOLD_PERCENT = 90.0
ADAPTIVE_RESTORE_CYCLES = 4
MEMORY_ESTIMATE_MULTIPLIER = 2.5  # heuristic multiplier to convert utf-8 bytes -> Python memory footprint
ADAPTER_TIMEOUT_S = 30.0
ADAPTER_RETRIES = 2
ADAPTER_BACKOFF_BASE = 0.5  # seconds

from src.logging import get_logger


class KnowledgeBaseManager:
    def __init__(self, kb_name: str, base_dir: str = "data/knowledge_bases"):
        self.kb_name = kb_name
        self.base_dir = Path(base_dir) / kb_name
        self.docs_dir = self.base_dir / "docs"
        self.index_dir = self.base_dir / "index"

        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Parser (generator contract)
        self.parser = PDFParser()

        # Vector Store Adapter (Pluggable)
        # Default is a mock/stub. In production, inject your LightRAG/FAISS client here.
        # Adapter must be an async callable: async def adapter(chunks) -> dict
        self.vector_store_adapter: Callable[[List[Dict[str, Any]]], Awaitable[Dict[str, Any]]] = self._default_vector_adapter

        # Adaptive state
        self._default_batch_size = DEFAULT_BATCH_SIZE
        self._default_max_bytes = DEFAULT_MAX_BYTES
        self._adaptive_restore_counter = 0

    def set_adapter(self, adapter_fn: Callable[[List[Dict[str, Any]]], Awaitable[Dict[str, Any]]]):
        """Inject a custom vector store adapter (async callable)."""
        self.vector_store_adapter = adapter_fn

    def _cleanup_memory(self):
        """
        Aggressively release memory.
        Critical for APUs (Ryzen AI, Apple Silicon) where RAM is VRAM.
        """
        # 1. Python GC
        gc.collect()

        # 2. PyTorch CUDA/MPS Cache
        if torch:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    # Explicitly release IPC handles if multiprocessing is involved
                    if hasattr(torch.cuda, "ipc_collect"):
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    def _get_rss_mb(self) -> float:
        if psutil:
            try:
                return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def _approx_bytes(self, text: str) -> int:
        """
        Approximate the memory footprint of a text chunk in bytes.
        Uses utf-8 length and a multiplier to account for Python object overhead.
        """
        try:
            utf8_len = len(text.encode("utf-8"))
            approx = int(utf8_len * MEMORY_ESTIMATE_MULTIPLIER)
            # Ensure a minimum
            return max(256, approx)
        except Exception:
            return 1024

    def _get_adaptive_batch_limits(
        self, current_batch_size: int, current_max_bytes: int, safe_threshold_percent: float = SAFETY_THRESHOLD_PERCENT
    ) -> Tuple[int, int]:
        """
        Reduces or restores batch limits based on system memory pressure.
        Returns (new_batch_size, new_max_bytes).
        Implements hysteresis: shrink quickly, restore slowly after several stable cycles.
        """
        if not psutil:
            return current_batch_size, current_max_bytes

        try:
            mem = psutil.virtual_memory()
            if mem.percent >= safe_threshold_percent:
                # Shrink aggressively
                new_size = max(1, current_batch_size // 2)
                new_bytes = max(500_000, current_max_bytes // 2)
                # Reset restore counter
                self._adaptive_restore_counter = 0
                if new_size < current_batch_size or new_bytes < current_max_bytes:
                    logger.warning(
                        f"High Memory ({mem.percent}%). Throttling: Batch {current_batch_size}->{new_size}, Bytes {current_max_bytes}->{new_bytes}"
                    )
                return new_size, new_bytes

            # If memory is comfortable, increment restore counter and slowly restore
            self._adaptive_restore_counter += 1
            if self._adaptive_restore_counter >= ADAPTIVE_RESTORE_CYCLES:
                # Restore toward defaults but do not exceed defaults
                restored_size = min(self._default_batch_size, current_batch_size * 2)
                restored_bytes = min(self._default_max_bytes, current_max_bytes * 2)
                if restored_size != current_batch_size or restored_bytes != current_max_bytes:
                    logger.info(f"Restoring batch limits: {current_batch_size}->{restored_size}, {current_max_bytes}->{restored_bytes}")
                self._adaptive_restore_counter = 0
                return restored_size, restored_bytes

            return current_batch_size, current_max_bytes
        except Exception:
            return current_batch_size, current_max_bytes

    async def _default_vector_adapter(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Default stub for vector store insertion (Mock).
        Replace this with actual LightRAG/FAISS call.
        """
        # Simulate processing time/overhead
        await asyncio.sleep(0.05)
        return {"success": True, "persisted": len(chunks), "latency_ms": 50}

    async def process_documents(
        self,
        file_paths: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_wait_ms: int = DEFAULT_MAX_WAIT_MS,
    ):
        """
        Process documents with strict memory controls and adaptive buffering.

        Args:
            file_paths: list of file paths
            batch_size: initial max chunks per flush
            max_bytes: initial max bytes per flush (heuristic)
            max_wait_ms: max time to wait before flushing partial buffer
        """
        logger.info(f"Starting ingestion. Triggers: Count={batch_size}, Bytes={max_bytes}, Time={max_wait_ms}ms")

        total_start = time.time()

        # Runtime adaptive limits
        active_batch_size = batch_size
        active_max_bytes = max_bytes

        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    logger.error(f"File not found: {path}")
                    continue

                if path.suffix.lower() == ".pdf":
                    # Update limits before starting file
                    active_batch_size, active_max_bytes = self._get_adaptive_batch_limits(batch_size, max_bytes)
                    await self._process_pdf_stream(path, active_batch_size, active_max_bytes, max_wait_ms)
                else:
                    self._process_text_file(path)

                # Cleanup after every file
                self._cleanup_memory()

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                import traceback

                traceback.print_exc()

        logger.info(f"All processing complete. Duration: {time.time() - total_start:.2f}s")

    async def _process_pdf_stream(self, file_path: Path, batch_size: int, max_bytes: int, max_wait_ms: int):
        """
        Streams PDF content with multi-dimensional backpressure.
        Expects parser.parse_generator to yield (text, page_num, byte_size) and to be truly streaming.
        """
        logger.info(f"Streaming processing for PDF: {file_path.name}")

        chunk_buffer: List[Dict[str, Any]] = []
        current_buffer_bytes = 0
        last_flush_time = time.time() * 1000
        page_count = 0

        # Start with the configured limits; we'll adapt mid-stream
        active_batch_size = batch_size
        active_max_bytes = max_bytes

        content_generator = self.parser.parse_generator(str(file_path))

        for page_text, page_num, text_bytes in content_generator:
            # Recompute adaptive limits mid-stream
            active_batch_size, active_max_bytes = self._get_adaptive_batch_limits(active_batch_size, active_max_bytes)

            if not page_text or not page_text.strip():
                # still count page for logging but skip empty content
                page_count += 1
                continue

            # Create chunks and estimate memory impact using a calibrated heuristic
            new_chunks = self._chunk_text(page_text, source=file_path.name, page=page_num)
            approx_bytes = sum(self._approx_bytes(c["content"]) for c in new_chunks)

            # Append to buffer (we will replace the list on flush to break references)
            chunk_buffer.extend(new_chunks)
            current_buffer_bytes += approx_bytes
            page_count += 1

            # Check Triggers
            now = time.time() * 1000
            time_trigger = (now - last_flush_time) >= max_wait_ms
            size_trigger = current_buffer_bytes >= active_max_bytes
            count_trigger = len(chunk_buffer) >= active_batch_size

            if count_trigger or size_trigger or time_trigger:
                reason = "Count" if count_trigger else ("Size" if size_trigger else "Time")
                logger.debug(
                    f"Flush triggered by {reason}. Buffer: {len(chunk_buffer)} items, approx {current_buffer_bytes} bytes (active limits: {active_batch_size} chunks, {active_max_bytes} bytes)"
                )

                # If system is critically pressured, do a deep cleanup and reduce limits
                if psutil:
                    try:
                        vm = psutil.virtual_memory()
                        if vm.percent >= CRITICAL_THRESHOLD_PERCENT:
                            logger.critical(f"Critical Memory Pressure ({vm.percent}%). Forcing deep cleanup and throttling.")
                            self._cleanup_memory()
                            active_batch_size = max(1, active_batch_size // 2)
                            active_max_bytes = max(500_000, active_max_bytes // 2)
                    except Exception:
                        pass

                # Flush and handle result
                flush_result = await self._flush_batch_with_retries(chunk_buffer)
                # Replace buffer with a new list to break references (caller responsibility)
                chunk_buffer = []
                current_buffer_bytes = 0
                last_flush_time = time.time() * 1000

                # Deep Cleanup after flush
                self._cleanup_memory()

                if page_count % 50 == 0:
                    logger.info(f"Processed {page_count} pages of {file_path.name}...")

        # Flush remaining chunks
        if chunk_buffer:
            await self._flush_batch_with_retries(chunk_buffer)

        logger.info(f"Completed processing {file_path.name} ({page_count} pages).")

    def _chunk_text(self, text: str, source: str, page: int) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_size = 1000
        overlap = 100
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            segment = text[start:end]
            if segment.strip():
                chunks.append(
                    {
                        "content": segment,
                        "metadata": {"source": source, "page": page, "chunk_len": len(segment)},
                    }
                )
            if end == text_len:
                break
            start += (chunk_size - overlap)
        return chunks

    async def _flush_batch_with_retries(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Wraps _flush_batch with retries and exponential backoff.
        Returns the last adapter result or an error dict.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, ADAPTER_RETRIES + 2):  # retries + initial attempt
            try:
                # Use asyncio.wait_for to bound adapter latency
                result = await asyncio.wait_for(self._flush_batch(chunks), timeout=ADAPTER_TIMEOUT_S)
                # If adapter returned a failure payload, decide whether to retry
                if isinstance(result, dict) and not result.get("success", False):
                    last_exc = Exception(result.get("error", "adapter-failure"))
                    logger.warning(f"Adapter returned failure on attempt {attempt}: {result.get('error')}")
                    # fall through to retry if attempts remain
                else:
                    return result or {"success": True, "persisted": len(chunks)}
            except asyncio.TimeoutError:
                last_exc = Exception("adapter-timeout")
                logger.error(f"Adapter timeout on attempt {attempt}")
            except Exception as e:
                last_exc = e
                logger.error(f"Adapter exception on attempt {attempt}: {e}")

            # Backoff before next attempt
            if attempt <= ADAPTER_RETRIES + 1:
                backoff = ADAPTER_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(f"Backing off for {backoff:.2f}s before retry (attempt {attempt})")
                await asyncio.sleep(backoff)

        # All attempts failed
        logger.error(f"All adapter attempts failed: {last_exc}")
        return {"success": False, "persisted": 0, "error": str(last_exc)}

    async def _flush_batch(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send a batch of chunks to the storage engine via adapter.
        This function calls the adapter and returns its result.
        It does not attempt to delete the caller's list; callers must replace their buffer.
        """
        if not chunks:
            return {"success": True, "persisted": 0}

        start_time = time.time()
        try:
            # Call adapter (expected to be async)
            result = await self.vector_store_adapter(chunks)

            if not isinstance(result, dict):
                # Normalize unexpected adapter responses
                result = {"success": False, "persisted": 0, "error": "invalid-adapter-response"}

            if not result.get("success", False):
                logger.warning(f"Batch flush failed: {result.get('error', 'Unknown error')}")
                # Adapter-level failure is returned to caller for retry/backoff handling
            elapsed = time.time() - start_time
            logger.info(f"Flushed {len(chunks)} chunks in {elapsed:.3f}s (adapter reported: {result.get('latency_ms')})")
            return result
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")
            return {"success": False, "persisted": 0, "error": str(e)}
        finally:
            # Local cleanup only: drop any local temporaries if present
            # Do not attempt to delete the caller's 'chunks' variable here.
            self._cleanup_memory()

    def _process_text_file(self, file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = self._chunk_text(content, file_path.name, 1)
                logger.info(f"Processed text file {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
