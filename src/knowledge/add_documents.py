#!/usr/bin/env python3
"""
src/knowledge/add_documents.py

CLI entrypoint to ingest documents into a KnowledgeBase with memory-safe defaults.

Improvements:
- Optional CI-friendly mock flush wiring via --use-mock-flush.
- Optional subprocess isolation mode (--isolate-workers) that delegates to the
  ingest_isolated harness for safer runs on developer machines or CI.
- Better argument validation, logging, and graceful shutdown.
- Clear mapping of CLI args to KnowledgeBaseManager parameters.
"""

import asyncio
import sys
import argparse
import shlex
import subprocess
import signal
from pathlib import Path
from typing import List

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.knowledge.manager import KnowledgeBaseManager
from src.core.logging import get_logger

logger = get_logger("AddDocuments")


def _run_isolated_subprocess(file_paths: List[str], kb_name: str, batch_size: int, max_bytes: int, timeout_ms: int, output_dir: str, use_mock_flush: bool):
    """
    Run the ingest_isolated harness in a subprocess. This is useful when you want
    to isolate native allocations and ensure the OS reclaims memory between workers.
    """
    cmd = [
        sys.executable,
        "tests/performance/oom_repro/ingest_isolated.py",
        "--file",
        file_paths[0] if len(file_paths) == 1 else file_paths[0],  # ingest_isolated supports one file per invocation in this harness
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
        cmd.append("--use-mock-flush")

    # If multiple files are provided, run them sequentially in separate subprocesses
    if len(file_paths) > 1:
        for f in file_paths:
            cmd_local = cmd.copy()
            cmd_local[3] = f  # replace the file path position
            logger.info(f"Spawning isolated ingest subprocess for {f}")
            subprocess.run(cmd_local, check=False)
    else:
        logger.info(f"Spawning isolated ingest subprocess: {' '.join(shlex.quote(p) for p in cmd)}")
        subprocess.run(cmd, check=False)


async def _run_in_process(file_paths: List[str], kb_name: str, batch_size: int, max_bytes: int, timeout_ms: int, use_mock_flush: bool):
    """
    Run ingestion in-process using KnowledgeBaseManager.
    Optionally wires a mock adapter for CI determinism.
    """
    kb = KnowledgeBaseManager(kb_name)

    # Wire mock adapter if requested (CI)
    if use_mock_flush:
        try:
            from tests.performance.oom_repro.mock_flush_adapter import MockFlushAdapter  # type: ignore

            mock_adapter = MockFlushAdapter(latency_ms=50, mem_delta_mb=0, fail_rate=0.0)
            # Manager expects an async callable; MockFlushAdapter.flush is async
            kb.set_adapter(mock_adapter.flush)
            logger.info("Wired MockFlushAdapter for deterministic flushes (CI mode).")
        except Exception:
            logger.warning("MockFlushAdapter not available; continuing without it.")

    # Validate files exist before starting
    valid_files = []
    for f in file_paths:
        p = Path(f)
        if not p.exists():
            logger.error(f"File not found: {f}")
        else:
            valid_files.append(str(p.resolve()))

    if not valid_files:
        logger.error("No valid files to ingest. Exiting.")
        return

    # Run the ingestion
    try:
        await kb.process_documents(
            valid_files,
            batch_size=batch_size,
            max_bytes=max_bytes,
            max_wait_ms=timeout_ms,
        )
    except asyncio.CancelledError:
        logger.warning("Ingestion cancelled.")
    except Exception as e:
        logger.exception(f"Fatal error during ingestion: {e}")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop):
    """
    Install signal handlers to gracefully cancel the asyncio loop on SIGINT/SIGTERM.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_shutdown(loop)))
        except NotImplementedError:
            # add_signal_handler may not be implemented on Windows event loop policy
            pass


async def _shutdown(loop: asyncio.AbstractEventLoop):
    logger.info("Shutdown requested. Cancelling tasks...")
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Knowledge Base with memory safety.")
    parser.add_argument("kb_name", help="Name of the knowledge base")
    parser.add_argument("files", nargs='+', help="List of file paths to ingest")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of chunks per flush (default: 10)")
    parser.add_argument("--max-bytes", type=int, default=5_000_000, help="Max bytes per flush (default: 5MB)")
    parser.add_argument("--timeout", type=int, default=5000, help="Max wait time in ms per flush (default: 5000)")
    parser.add_argument("--use-mock-flush", action="store_true", help="Use deterministic mock flush adapter (CI)")
    parser.add_argument("--isolate-workers", action="store_true", help="Run ingestion in isolated subprocesses (safer for native leaks)")
    parser.add_argument("--output-dir", default="tests/performance/oom_repro/logs", help="Output directory for harness artifacts (used by isolated mode)")
    args = parser.parse_args()

    logger.info(f"Initializing Knowledge Base: {args.kb_name}")
    logger.info(f"Processing {len(args.files)} files...")

    # If isolate-workers is requested, delegate to the subprocess harness for each file
    if args.isolate_workers:
        # Run subprocesses synchronously to keep logs deterministic
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

    # Otherwise run in-process using the KnowledgeBaseManager
    loop = asyncio.get_event_loop()
    _install_signal_handlers(loop)
    try:
        await _run_in_process(
            file_paths=args.files,
            kb_name=args.kb_name,
            batch_size=args.batch_size,
            max_bytes=args.max_bytes,
            timeout_ms=args.timeout,
            use_mock_flush=args.use_mock_flush,
        )
    finally:
        logger.info("Ingestion complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
