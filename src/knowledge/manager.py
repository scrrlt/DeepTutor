#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Base Manager

<<<<<<< HEAD
Manages multiple knowledge bases and provides utilities for accessing them.
=======
Streaming ingestion manager with adaptive batching, memory safety, and pluggable vector-store adapter.
Designed for use in CI and production with APUs and constrained hosts.

Patch summary:
- FIX (blocker): adaptive batching now uses container-aware cgroup memory metrics when available,
  falling back to psutil host memory only if cgroups are unavailable.
>>>>>>> e8e972c (lock down runtime deps to preserve deterministic OOM behavior)
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from src.logging import get_logger


class KnowledgeBaseManager:
    """Manager for knowledge bases"""

    def __init__(self, base_dir="./data/knowledge_bases"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Config file to track knowledge bases
        self.config_file = self.base_dir / "kb_config.json"
        self.config = self._load_config()
        self.logger = get_logger("Knowledge")

    def _load_config(self) -> dict:
        """Load knowledge base configuration (kb_config.json only stores KB list)"""
        if self.config_file.exists():
            with open(self.config_file, encoding="utf-8") as f:
                config = json.load(f)
                # Migration: remove old "default" field if present
                if "default" in config:
                    del config["default"]
                    # Save cleaned config
                    try:
                        with open(self.config_file, "w", encoding="utf-8") as wf:
                            json.dump(config, wf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                return config
        return {"knowledge_bases": {}}

    def _save_config(self):
        """Save knowledge base configuration"""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def list_knowledge_bases(self) -> list[str]:
        """List all available knowledge bases from kb_config.json"""
        kb_list = []

        # Read knowledge base list from config file (this is the authoritative source)
        config_kbs = self.config.get("knowledge_bases", {})

<<<<<<< HEAD
        for kb_name in config_kbs.keys():
            # Verify knowledge base directory exists
            kb_dir = self.base_dir / kb_name
            if kb_dir.exists() and kb_dir.is_dir():
                kb_list.append(kb_name)
            else:
                # If in config but directory doesn't exist, log warning but don't add
                self.logger.warning(
                    f"Knowledge base '{kb_name}' is in config but directory does not exist: {kb_dir}"
                )

        # If no config file or config is empty, fallback to scanning directory (backward compatibility)
        if not kb_list and self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and item.name != "__pycache__":
                    metadata_file = item / "metadata.json"
                    if metadata_file.exists():
                        kb_list.append(item.name)

        return sorted(kb_list)

    def register_knowledge_base(self, name: str, description: str = "", set_default: bool = False):
        """Register a knowledge base"""
        kb_dir = self.base_dir / name
        if not kb_dir.exists():
            raise ValueError(f"Knowledge base directory does not exist: {kb_dir}")

        if "knowledge_bases" not in self.config:
            self.config["knowledge_bases"] = {}

        self.config["knowledge_bases"][name] = {"path": name, "description": description}

        # Only set default if explicitly requested
        if set_default:
            self.set_default(name)

        self._save_config()

    def get_knowledge_base_path(self, name: str | None = None) -> Path:
        """Get path to a knowledge base"""
        if name is None:
            name = self.config.get("default")
            if name is None:
                raise ValueError("No default knowledge base set")

        kb_dir = self.base_dir / name
        if not kb_dir.exists():
            raise ValueError(f"Knowledge base not found: {name}")

        return kb_dir

    def get_rag_storage_path(self, name: str | None = None) -> Path:
        """Get RAG storage path for a knowledge base"""
        kb_dir = self.get_knowledge_base_path(name)
        rag_storage = kb_dir / "rag_storage"
        if not rag_storage.exists():
            raise ValueError(f"RAG storage not found for knowledge base: {name or 'default'}")
        return rag_storage

    def get_images_path(self, name: str | None = None) -> Path:
        """Get images path for a knowledge base"""
        kb_dir = self.get_knowledge_base_path(name)
        return kb_dir / "images"

    def get_content_list_path(self, name: str | None = None) -> Path:
        """Get content list path for a knowledge base"""
        kb_dir = self.get_knowledge_base_path(name)
        return kb_dir / "content_list"

    def get_raw_path(self, name: str | None = None) -> Path:
        """Get raw documents path for a knowledge base"""
        kb_dir = self.get_knowledge_base_path(name)
        return kb_dir / "raw"

    def set_default(self, name: str):
        """Set default knowledge base using centralized config service."""
        if name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {name}")

        # Use centralized config service only (no longer stored in kb_config.json)
        try:
            from src.services.config import get_kb_config_service

            kb_config_service = get_kb_config_service()
            kb_config_service.set_default_kb(name)
        except Exception as e:
            print(f"Warning: Failed to save default to centralized config: {e}")

    def get_default(self) -> str | None:
        """
        Get default knowledge base name.

        Priority:
        1. Centralized config service (knowledge_base_configs.json)
        2. First knowledge base in the list (auto-fallback)
        """
        # Try centralized config first
        try:
            from src.services.config import get_kb_config_service

            kb_config_service = get_kb_config_service()
            default_kb = kb_config_service.get_default_kb()
            if default_kb and default_kb in self.list_knowledge_bases():
                return default_kb
        except Exception:
            pass

        # Fallback to first knowledge base in sorted list
        kb_list = self.list_knowledge_bases()
        if kb_list:
            return kb_list[0]

        return None

    def get_metadata(self, name: str | None = None) -> dict:
        """Get knowledge base metadata"""
        kb_dir = self.get_knowledge_base_path(name)
        metadata_file = kb_dir / "metadata.json"

        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                return json.load(f)

        return {}

    def get_info(self, name: str | None = None) -> dict:
        """Get detailed information about a knowledge base.

        This method:
        1. Gets the KB name (from parameter or default)
        2. Reads metadata.json from the KB directory
        3. Collects statistics about files and RAG status
        """
        kb_name = name or self.get_default()
        if kb_name is None:
            raise ValueError("No knowledge base name provided and no default set")

        # Get knowledge base path
        kb_dir = self.base_dir / kb_name
        if not kb_dir.exists():
            raise ValueError(f"Knowledge base directory does not exist: {kb_dir}")

        # Verify knowledge base is in config (if not, give warning but don't block)
        if kb_name not in self.config.get("knowledge_bases", {}):
            self.logger.warning(
                f"Knowledge base '{kb_name}' is not in kb_config.json, but directory exists"
            )

        info = {
            "name": kb_name,
            "path": str(kb_dir),
            "is_default": kb_name == self.get_default(),
            "metadata": {},
        }

        # Read metadata.json (if exists)
        metadata_file = kb_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    info["metadata"] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read metadata.json for KB '{kb_name}': {e}")
                info["metadata"] = {}
        else:
            # metadata.json doesn't exist, use empty dict
            info["metadata"] = {}

        # Count files - handle errors gracefully
        raw_dir = kb_dir / "raw"
        images_dir = kb_dir / "images"
        content_list_dir = kb_dir / "content_list"
        rag_storage_dir = kb_dir / "rag_storage"

        try:
            raw_count = (
                len([f for f in raw_dir.iterdir() if f.is_file()]) if raw_dir.exists() else 0
            )
        except Exception:
            raw_count = 0

        try:
            images_count = (
                len([f for f in images_dir.iterdir() if f.is_file()]) if images_dir.exists() else 0
            )
        except Exception:
            images_count = 0

        try:
            content_lists_count = (
                len(list(content_list_dir.glob("*.json"))) if content_list_dir.exists() else 0
            )
        except Exception:
            content_lists_count = 0

        metadata = info["metadata"]
        rag_provider = metadata.get("rag_provider") if isinstance(metadata, dict) else None
        info["statistics"] = {
            "raw_documents": raw_count,
            "images": images_count,
            "content_lists": content_lists_count,
            "rag_initialized": rag_storage_dir.exists() and rag_storage_dir.is_dir(),
            "rag_provider": rag_provider,  # Add RAG provider info
        }

        # Try to get RAG statistics
        if rag_storage_dir.exists() and rag_storage_dir.is_dir():
            try:
                entities_file = rag_storage_dir / "kv_store_full_entities.json"
                relations_file = rag_storage_dir / "kv_store_full_relations.json"
                chunks_file = rag_storage_dir / "kv_store_text_chunks.json"

                rag_stats = {}
                if entities_file.exists():
                    try:
                        with open(entities_file, encoding="utf-8") as f:
                            entities_data = json.load(f)
                            rag_stats["entities"] = (
                                len(entities_data) if isinstance(entities_data, (list, dict)) else 0
                            )
                    except Exception:
                        pass

                if relations_file.exists():
                    try:
                        with open(relations_file, encoding="utf-8") as f:
                            relations_data = json.load(f)
                            rag_stats["relations"] = (
                                len(relations_data)
                                if isinstance(relations_data, (list, dict))
                                else 0
                            )
                    except Exception:
                        pass

                if chunks_file.exists():
                    try:
                        with open(chunks_file, encoding="utf-8") as f:
                            chunks_data = json.load(f)
                            rag_stats["chunks"] = (
                                len(chunks_data) if isinstance(chunks_data, (list, dict)) else 0
                            )
                    except Exception:
                        pass
=======
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

    def _get_system_memory_percent(self) -> float:
        """
        Return memory usage percentage, preferring cgroup (container) limits over host memory.

        Supports:
        - Cgroup v2: /sys/fs/cgroup/memory.current + /sys/fs/cgroup/memory.max
        - Cgroup v1: /sys/fs/cgroup/memory/memory.usage_in_bytes + /sys/fs/cgroup/memory/memory.limit_in_bytes

        Falls back to psutil.virtual_memory().percent (host) if no cgroup limits are detectable.
        Returns 0.0 if no signal can be determined.
        """
        # 1) cgroup v2
        try:
            cur_path = "/sys/fs/cgroup/memory.current"
            max_path = "/sys/fs/cgroup/memory.max"
            if os.path.exists(cur_path) and os.path.exists(max_path):
                with open(cur_path, "r") as f:
                    usage = int(f.read().strip())

                with open(max_path, "r") as f:
                    lim_str = f.read().strip()

                if lim_str != "max":
                    limit = int(lim_str)
                    if limit > 0:
                        return (usage / limit) * 100.0
                # If "max" (no limit), fall through to other mechanisms
        except Exception:
            pass

        # 2) cgroup v1
        try:
            usage_path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
            limit_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            if os.path.exists(usage_path) and os.path.exists(limit_path):
                with open(usage_path, "r") as f:
                    usage = int(f.read().strip())

                with open(limit_path, "r") as f:
                    limit = int(f.read().strip())

                # Some environments report an absurdly high number to represent "no limit"
                if limit > 0 and limit < 10**15:
                    return (usage / limit) * 100.0
                # Otherwise fall through to host metrics
        except Exception:
            pass

        # 3) host fallback
        if psutil:
            try:
                return float(psutil.virtual_memory().percent)
            except Exception:
                return 0.0

        return 0.0

    def _get_adaptive_batch_limits(
        self, current_batch_size: int, current_max_bytes: int, safe_threshold_percent: float = SAFETY_THRESHOLD_PERCENT
    ) -> Tuple[int, int]:
        """
        Reduces or restores batch limits based on container/host memory pressure.

        Returns (new_batch_size, new_max_bytes).
        Implements hysteresis: shrink quickly, restore slowly after several stable cycles.
        """
        mem_percent = self._get_system_memory_percent()

        # If we can't determine memory pressure, don't change limits
        if mem_percent <= 0.0:
            return current_batch_size, current_max_bytes

        try:
            if mem_percent >= safe_threshold_percent:
                # Shrink aggressively
                new_size = max(1, current_batch_size // 2)
                new_bytes = max(500_000, current_max_bytes // 2)
                self._adaptive_restore_counter = 0

                if new_size < current_batch_size or new_bytes < current_max_bytes:
                    logger.warning(
                        f"High Memory Pressure ({mem_percent:.1f}%). "
                        f"Throttling: Batch {current_batch_size}->{new_size}, Bytes {current_max_bytes}->{new_bytes}"
                    )
                return new_size, new_bytes

            # Restore slowly after several stable cycles
            self._adaptive_restore_counter += 1
            if self._adaptive_restore_counter >= ADAPTIVE_RESTORE_CYCLES:
                restored_size = min(self._default_batch_size, current_batch_size * 2)
                restored_bytes = min(self._default_max_bytes, current_max_bytes * 2)

                if restored_size != current_batch_size or restored_bytes != current_max_bytes:
                    logger.info(
                        f"Restoring batch limits: {current_batch_size}->{restored_size}, {current_max_bytes}->{restored_bytes}"
                    )

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
                    f"Flush triggered by {reason}. Buffer: {len(chunk_buffer)} items, "
                    f"approx {current_buffer_bytes} bytes (active limits: {active_batch_size} chunks, {active_max_bytes} bytes)"
                )

                # Critical pressure handling (container-aware)
                mem_percent = self._get_system_memory_percent()
                if mem_percent >= CRITICAL_THRESHOLD_PERCENT:
                    logger.critical(f"Critical Memory Pressure ({mem_percent:.1f}%). Forcing deep cleanup and throttling.")
                    self._cleanup_memory()
                    active_batch_size = max(1, active_batch_size // 2)
                    active_max_bytes = max(500_000, active_max_bytes // 2)

                flush_result = await self._flush_batch_with_retries(chunk_buffer)

                # Break references
                chunk_buffer = []
                current_buffer_bytes = 0
                last_flush_time = time.time() * 1000

                self._cleanup_memory()
>>>>>>> e8e972c (lock down runtime deps to preserve deterministic OOM behavior)

                if rag_stats:
                    statistics = info["statistics"]
                    if isinstance(statistics, dict):
                        statistics["rag"] = rag_stats
            except Exception:
                pass

<<<<<<< HEAD
        return info
=======
        if chunk_buffer:
            await self._flush_batch_with_retries(chunk_buffer)
>>>>>>> e8e972c (lock down runtime deps to preserve deterministic OOM behavior)

    def delete_knowledge_base(self, name: str, confirm: bool = False) -> bool:
        """
        Delete a knowledge base

        Args:
            name: Knowledge base name
            confirm: If True, skip confirmation (use with caution!)

        Returns:
            True if deleted successfully
        """
        if name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {name}")

        kb_dir = self.get_knowledge_base_path(name)

        if not confirm:
            # Ask for confirmation in CLI
            print(f"⚠️  Warning: This will permanently delete the knowledge base '{name}'")
            print(f"   Path: {kb_dir}")
            response = input("Are you sure? Type 'yes' to confirm: ")
            if response.lower() != "yes":
                print("Deletion cancelled.")
                return False

        # Delete the directory
        shutil.rmtree(kb_dir)

        # Remove from config
        if name in self.config.get("knowledge_bases", {}):
            del self.config["knowledge_bases"][name]

        # Update default if this was the default
        if self.config.get("default") == name:
            remaining = self.list_knowledge_bases()
            self.config["default"] = remaining[0] if remaining else None

        self._save_config()
        return True

    def clean_rag_storage(self, name: str | None = None, backup: bool = True) -> bool:
        """
        Clean (delete) RAG storage for a knowledge base
        Useful when RAG data is corrupted

        Args:
            name: Knowledge base name (default if not specified)
            backup: If True, backup the RAG storage before deleting

        Returns:
            True if cleaned successfully
        """
        kb_name = name or self.get_default()
        kb_dir = self.get_knowledge_base_path(kb_name)
        rag_storage_dir = kb_dir / "rag_storage"

        if not rag_storage_dir.exists():
            self.logger.info(f"RAG storage does not exist for '{kb_name}'")
            return False

        # Backup if requested
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = kb_dir / f"rag_storage_backup_{timestamp}"
            shutil.copytree(rag_storage_dir, backup_dir)
            self.logger.success(f"Backed up to: {backup_dir}")

        # Delete RAG storage
        shutil.rmtree(rag_storage_dir)
        rag_storage_dir.mkdir(parents=True, exist_ok=True)

        self.logger.success(f"RAG storage cleaned for '{kb_name}'")
        return True

    def link_folder(self, kb_name: str, folder_path: str) -> dict:
        """
        Link a local folder to a knowledge base.

        Args:
            kb_name: Knowledge base name
            folder_path: Path to local folder (supports ~, relative paths)

        Returns:
            Dict with folder info including id, path, and file count

        Raises:
            ValueError: If KB not found or folder doesn't exist
        """
        if kb_name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {kb_name}")

        # Normalize path (cross-platform: handles ~, relative paths, etc.)
        folder = Path(folder_path).expanduser().resolve()

        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder}")

        # Get supported files in folder
        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
        files: list[Path] = []
        for ext in supported_extensions:
            files.extend(folder.glob(f"**/*{ext}"))

        # Generate folder ID
        import hashlib

        folder_id = hashlib.md5(  # noqa: S324
            str(folder).encode(), usedforsecurity=False
        ).hexdigest()[:8]

        # Load existing linked folders from metadata
        kb_dir = self.base_dir / kb_name
        metadata_file = kb_dir / "metadata.json"
        metadata: dict = {}

        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as fp:
                    metadata = json.load(fp)
            except Exception:
                metadata = {}

        if "linked_folders" not in metadata:
            metadata["linked_folders"] = []

        # Check if already linked
        existing_ids = [item["id"] for item in metadata.get("linked_folders", [])]
        if folder_id in existing_ids:
            # If already linked, treat as success (idempotent)
            # Find and return existing info
            for item in metadata.get("linked_folders", []):
                if item["id"] == folder_id:
                    return item

        # Add folder info
        folder_info = {
            "id": folder_id,
            "path": str(folder),
            "added_at": datetime.now().isoformat(),
            "file_count": len(files),
        }
        metadata["linked_folders"].append(folder_info)

        # Save metadata
        with open(metadata_file, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)

        return folder_info

    def get_linked_folders(self, kb_name: str) -> list[dict]:
        """
        Get list of linked folders for a knowledge base.

        Args:
            kb_name: Knowledge base name

        Returns:
            List of linked folder info dicts
        """
        if kb_name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {kb_name}")

        kb_dir = self.base_dir / kb_name
        metadata_file = kb_dir / "metadata.json"

        if not metadata_file.exists():
            return []

        try:
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)
                return metadata.get("linked_folders", [])
        except Exception:
            return []

    def unlink_folder(self, kb_name: str, folder_id: str) -> bool:
        """
        Unlink a folder from a knowledge base.

        Args:
            kb_name: Knowledge base name
            folder_id: Folder ID to unlink

        Returns:
            True if unlinked successfully, False if not found
        """
        if kb_name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {kb_name}")

        kb_dir = self.base_dir / kb_name
        metadata_file = kb_dir / "metadata.json"

        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return False

        linked = metadata.get("linked_folders", [])
        new_linked = [f for f in linked if f["id"] != folder_id]

        if len(new_linked) == len(linked):
            return False  # Not found

        metadata["linked_folders"] = new_linked

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return True

    def scan_linked_folder(self, folder_path: str) -> list[str]:
        """
        Scan a linked folder and return list of supported file paths.

        Args:
            folder_path: Path to folder

        Returns:
            List of file paths (as strings)
        """
        folder = Path(folder_path).expanduser().resolve()

        if not folder.exists() or not folder.is_dir():
            return []

        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
        files = []

        for ext in supported_extensions:
            for file_path in folder.glob(f"**/*{ext}"):
                files.append(str(file_path))

        return sorted(files)

    def detect_folder_changes(self, kb_name: str, folder_id: str) -> dict:
        """
        Detect new and modified files in a linked folder since last sync.

        This enables automatic sync of changes from local folders that may
        be synced with cloud services like SharePoint, Google Drive, etc.

        Args:
            kb_name: Knowledge base name
            folder_id: Folder ID to check for changes

        Returns:
            Dict with 'new_files', 'modified_files', and 'has_changes' keys
        """
        if kb_name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {kb_name}")

        # Get folder info
        folders = self.get_linked_folders(kb_name)
        folder_info = next((f for f in folders if f["id"] == folder_id), None)

        if not folder_info:
            raise ValueError(f"Linked folder not found: {folder_id}")

        folder_path = Path(folder_info["path"]).expanduser().resolve()
        last_sync = folder_info.get("last_sync")
        synced_files = folder_info.get("synced_files", {})

        # Parse last sync timestamp
        last_sync_time = None
        if last_sync:
            try:
                last_sync_time = datetime.fromisoformat(last_sync)
            except Exception:
                pass

        # Scan current files
        supported_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
        new_files = []
        modified_files = []

        for ext in supported_extensions:
            for file_path in folder_path.glob(f"**/*{ext}"):
                file_str = str(file_path)
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_str in synced_files:
                    # Check if modified since last sync
                    prev_mtime_str = synced_files[file_str]
                    try:
                        prev_mtime = datetime.fromisoformat(prev_mtime_str)
                        if file_mtime > prev_mtime:
                            modified_files.append(file_str)
                    except Exception:
                        modified_files.append(file_str)
                else:
                    # New file (not in synced files)
                    new_files.append(file_str)

        return {
            "new_files": sorted(new_files),
            "modified_files": sorted(modified_files),
            "has_changes": len(new_files) > 0 or len(modified_files) > 0,
            "new_count": len(new_files),
            "modified_count": len(modified_files),
        }

    def update_folder_sync_state(self, kb_name: str, folder_id: str, synced_files: list[str]):
        """
        Update the sync state for a linked folder after successful sync.

        Records which files were synced and their modification times,
        enabling future change detection.

        Args:
            kb_name: Knowledge base name
            folder_id: Folder ID
            synced_files: List of file paths that were successfully synced
        """
        if kb_name not in self.list_knowledge_bases():
            raise ValueError(f"Knowledge base not found: {kb_name}")

        kb_dir = self.base_dir / kb_name
        metadata_file = kb_dir / "metadata.json"

        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return

        linked = metadata.get("linked_folders", [])

        for folder in linked:
            if folder["id"] == folder_id:
                # Record sync timestamp
                folder["last_sync"] = datetime.now().isoformat()

                # Record file modification times
                file_states = folder.get("synced_files", {})
                for file_path in synced_files:
                    try:
                        p = Path(file_path)
                        if p.exists():
                            mtime = datetime.fromtimestamp(p.stat().st_mtime)
                            file_states[file_path] = mtime.isoformat()
                    except Exception:
                        pass

                folder["synced_files"] = file_states
                folder["file_count"] = len(file_states)
                break


def main():
    """Command-line interface for knowledge base manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Manager")
    parser.add_argument(
        "--base-dir", default="./knowledge_bases", help="Base directory for knowledge bases"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all knowledge bases")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show knowledge base information")
    info_parser.add_argument(
        "name", nargs="?", help="Knowledge base name (default if not specified)"
    )

    # Set default command
    default_parser = subparsers.add_parser("set-default", help="Set default knowledge base")
    default_parser.add_argument("name", help="Knowledge base name")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a knowledge base")
    delete_parser.add_argument("name", help="Knowledge base name")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Clean RAG command
    clean_parser = subparsers.add_parser(
        "clean-rag", help="Clean RAG storage (useful for corrupted data)"
    )
    clean_parser.add_argument(
        "name", nargs="?", help="Knowledge base name (default if not specified)"
    )
    clean_parser.add_argument(
        "--no-backup", action="store_true", help="Don't backup before cleaning"
    )

    args = parser.parse_args()

    manager = KnowledgeBaseManager(args.base_dir)

    if args.command == "list":
        kb_list = manager.list_knowledge_bases()
        default_kb = manager.get_default()

        print("\nAvailable Knowledge Bases:")
        print("=" * 60)
        if not kb_list:
            print("No knowledge bases found")
        else:
            for kb_name in kb_list:
                default_marker = " (default)" if kb_name == default_kb else ""
                print(f"  • {kb_name}{default_marker}")
        print()

    elif args.command == "info":
        try:
            info = manager.get_info(args.name)

            print("\nKnowledge Base Information:")
            print("=" * 60)
            print(f"Name: {info['name']}")
            print(f"Path: {info['path']}")
            print(f"Default: {'Yes' if info['is_default'] else 'No'}")

            if info.get("metadata"):
                print("\nMetadata:")
                for key, value in info["metadata"].items():
                    print(f"  {key}: {value}")

            print("\nStatistics:")
            stats = info["statistics"]
            print(f"  Raw documents: {stats['raw_documents']}")
            print(f"  Images: {stats['images']}")
            print(f"  Content lists: {stats['content_lists']}")
            print(f"  RAG initialized: {'Yes' if stats['rag_initialized'] else 'No'}")

            if "rag" in stats:
                print("\n  RAG Statistics:")
                for key, value in stats["rag"].items():
                    print(f"    {key}: {value}")

            print()
        except Exception as e:
            print(f"Error: {e!s}")

    elif args.command == "set-default":
        try:
            manager.set_default(args.name)
            print(f"✓ Set '{args.name}' as default knowledge base")
        except Exception as e:
            print(f"Error: {e!s}")

    elif args.command == "delete":
        try:
            success = manager.delete_knowledge_base(args.name, confirm=args.force)
            if success:
                print(f"✓ Deleted knowledge base '{args.name}'")
        except Exception as e:
            print(f"Error: {e!s}")

    elif args.command == "clean-rag":
        try:
            manager.clean_rag_storage(args.name, backup=not args.no_backup)
        except Exception as e:
            print(f"Error: {e!s}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
