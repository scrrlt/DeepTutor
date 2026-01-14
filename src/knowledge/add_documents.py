#!/usr/bin/env python3
"""
src/knowledge/add_documents.py

CLI entrypoint to ingest documents into a KnowledgeBase with memory-safe defaults.

Features:
- Optional CI-friendly mock flush wiring via --use-mock-flush.
- Optional subprocess isolation mode (--isolate-workers) that delegates to the
  ingest_isolated harness for safer runs on developer machines or CI.
- Clear mapping of CLI args to KnowledgeBaseManager parameters.
- Deterministic logging and graceful shutdown behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import List

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.knowledge.manager import KnowledgeBaseManager
from src.core.logging import get_logger

logger = get_logger("AddDocuments")


# ---------------------------------------------------------------------------
# Isolated subprocess execution
# ---------------------------------------------------------------------------
def _run_isolated_subprocess(
    file_paths: List[str],
    kb_name: str,
    batch_size: int,
    max_bytes: int,
    timeout_ms: int,
    output_dir: str,
    use_mock_flush: bool,
) -> None:
    """
    Run the ingest_isolated harness in subprocess mode.

    Each file is processed in a separate invocation to ensure:
    - Native memory is reclaimed on process exit.
    - Logs and artifacts remain deterministic and attributable.
    """
    base_cmd = [
        sys.executable,
        "tests/performance/oom_repro/ingest_isolated.py",
        "--kb_name",
        kb_name,
        "--batch-size",
        str(batch_size),
        "--max-bytes",
        str(max_bytes),
        "--output-dir",
        output_dir,
    ]
    if use_mock_flush:
        base_cmd.append("--use-mock-flush")

    for f in file_paths:
        cmd = base_cmd + ["--file", f]
        logger.info(f"Spawning isolated ingest subprocess for {f}")
        logger.debug(f"Command: {' '.join(shlex.quote(p) for p in cmd)}")
        subprocess.run(cmd, check=False)


# ---------------------------------------------------------------------------
# In-process execution
# ---------------------------------------------------------------------------
async def _run_in_process(
    file_paths: List[str],
    kb_name: str,
    batch_size: int,
    max_bytes: int,
    timeout_ms: int,
    use_mock_flush: bool,
) -> None:
    """
    Run ingestion in-process using KnowledgeBaseManager.
    """
    kb = KnowledgeBaseManager(kb_name)

    # Wire mock adapter if requested (CI)
    if use_mock_flush:
        try:
            from tests.performance.oom_repro.mock_flush_adapter import MockFlushAdapter  # type: ignore

            mock_adapter = MockFlushAdapter(latency_ms=50, mem_delta_mb=0, fail_rate=0.0)
            kb.set_adapter(mock_adapter.flush)
            logger.info("MockFlushAdapter wired (CI mode).")
        except Exception:
            logger.warning("MockFlushAdapter not available; continuing without it.")

    # Validate files
    valid_files: List[str] = []
    for f in file_paths:
        p = Path(f)
        if not p.exists():
            logger.error(f"File not found: {f}")
        else:
            valid_files.append(str(p.resolve()))

    if not valid_files:
        logger.error("No valid files to ingest. Exiting.")
        return

    try:
        await kb.process_documents(
            valid_files,
            batch_size=batch_size,
            max_bytes=max_bytes,
            max_wait_ms=timeout_ms,
        )
    except asyncio.CancelledError:
        logger.warning("Ingestion cancelled.")
    except Exception:
        logger.exception("Fatal error during ingestion.")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """
    Install signal handlers to gracefully cancel the asyncio loop.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(loop)))
        except NotImplementedError:
            # Windows / alternative event loop policies
            pass


async def _shutdown(loop: asyncio.AbstractEventLoop) -> None:
    logger.info("Shutdown requested. Cancelling tasks...")
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Knowledge Base with memory safety."
    )
    parser.add_argument("kb_name", help="Name of the knowledge base")
    parser.add_argument("files", nargs="+", help="List of file paths to ingest")
    parser.add_argument("--batch-size", type=int, default=10, help="Chunks per flush (default: 10)")
    parser.add_argument("--max-bytes", type=int, default=5_000_000, help="Max bytes per flush (default: 5MB)")
    parser.add_argument("--timeout", type=int, default=5000, help="Max wait time in ms per flush")
    parser.add_argument("--use-mock-flush", action="store_true", help="Use deterministic mock flush adapter (CI)")
    parser.add_argument(
        "--isolate-workers",
        action="store_true",
        help="Run ingestion in isolated subprocesses (safer for native leaks)",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/performance/oom_repro/logs",
        help="Output directory for harness artifacts (isolated mode)",
    )
    args = parser.parse_args()

    logger.info(f"Initializing Knowledge Base: {args.kb_name}")
    logger.info(f"Processing {len(args.files)} file(s).")

    if args.isolate_workers:
        logger.info("Running ingestion in isolated subprocess mode.")
        _run_isolated_subprocess(
            file_paths=args.files,
            kb_name=args.kb_name,
            batch_size=args.batch_size,
            max_bytes=args.max_bytes,
            timeout_ms=args.timeout,
            output_dir=args.output_dir,
            use_mock_flush=args.use_mock_flush,
        )
        logger.info("Isolated ingestion runs complete.")
        return

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    await _run_in_process(
        file_paths=args.files,
        kb_name=args.kb_name,
        batch_size=args.batch_size,
        max_bytes=args.max_bytes,
        timeout_ms=args.timeout,
        use_mock_flush=args.use_mock_flush,
    )

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
