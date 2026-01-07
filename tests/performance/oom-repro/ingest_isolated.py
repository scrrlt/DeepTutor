#!/usr/bin/env python3
"""
tests/performance/oom_repro/ingest_isolated.py

Improved ingestion harness for deterministic OOM reproduction and diagnostics.

Key improvements:
- Monitors each worker process (not the parent) to capture per-worker memory spikes.
- Starts a lightweight monitor per worker and writes per-batch logs.
- Uses index-based page access and clears pdf.pages[i-1] after processing to avoid retaining page objects.
- Enforces an in-worker RSS watermark and per-worker timeout.
- Produces a run manifest and per-batch metadata for post-mortem analysis.
- Optional mock flush wiring via --use-mock-flush (CI friendly).
"""

import argparse
import time
import os
import sys
import subprocess
import multiprocessing
import traceback
import json
from pathlib import Path

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.logging import get_logger

logger = get_logger("IngestHarness")


# Optional mock adapter import (CI)
try:
    from tests.performance.oom_repro.mock_flush_adapter import MockFlushAdapter  # type: ignore
except Exception:
    MockFlushAdapter = None


def worker_process_page_batch(file_path, start_page, end_page, queue, kb_name, use_mock_flush=False):
    """
    Worker process: loads ONLY the requested page range and returns results via queue.
    Designed to exit quickly so OS reclaims native memory.
    """
    try:
        # Import inside worker to ensure clean state
        import pdfplumber
        import psutil

        results = []
        process = psutil.Process(os.getpid())

        # If mock flush is requested inside worker, instantiate adapter (no network)
        mock_adapter = None
        if use_mock_flush and MockFlushAdapter is not None:
            mock_adapter = MockFlushAdapter(latency_ms=50, mem_delta_mb=0, fail_rate=0.0)

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            effective_end = min(end_page, total_pages + 1)

            for i in range(start_page, effective_end):
                # Per-page memory watermark (safety): 2GB default
                rss = process.memory_info().rss
                if rss > 2 * 1024 * 1024 * 1024:
                    queue.put({"status": "error", "message": "Worker hit RSS safety limit", "page": i})
                    return

                # Access page by index (0-based) and clear reference after processing
                page = pdf.pages[i - 1]
                try:
                    text = page.extract_text(layout=False) or ""
                except Exception as e:
                    # If page extraction fails, record and continue
                    results.append({"page": i, "text_len": 0, "error": str(e)})
                    # Ensure we still attempt to free page resources
                    try:
                        page.flush_cache()
                    except Exception:
                        pass
                    pdf.pages[i - 1] = None
                    continue

                results.append({"page": i, "text_len": len(text)})

                # Simulate a flush step if mock adapter present (keeps worker self-contained)
                if mock_adapter is not None:
                    # run the async flush synchronously in worker
                    import asyncio

                    try:
                        asyncio.run(mock_adapter.flush([{"content": text, "metadata": {"page": i}}]))
                    except Exception:
                        # Don't fail the whole worker for mock adapter issues
                        pass

                # Cleanup page object and break reference in pdf.pages list
                try:
                    page.flush_cache()
                except Exception:
                    pass
                pdf.pages[i - 1] = None
                del page

        queue.put({"status": "success", "data": results})

    except Exception as e:
        traceback.print_exc()
        queue.put({"status": "error", "message": str(e)})


def _start_monitor_for_pid(pid: int, output_dir: str, batch_id: str, interval: float = 0.2):
    """
    Start the monitor_memory.py script for a specific PID and write logs under output_dir/batch_{batch_id}/
    Returns the subprocess.Popen object.
    """
    batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)
    out_csv = os.path.join(batch_dir, "memory_log.csv")
    out_json = os.path.join(batch_dir, "summary.json")

    cmd = [
        sys.executable,
        "tests/performance/oom_repro/monitor_memory.py",
        "--pid",
        str(pid),
        "--output-dir",
        batch_dir,
        "--interval",
        str(interval),
    ]
    # Start monitor as a detached subprocess; we'll terminate it after worker finishes
    proc = subprocess.Popen(cmd)
    return proc, batch_dir


def run_ingestion(file_path, kb_name, batch_size=10, max_bytes=5000000, output_dir="logs", use_mock_flush=False):
    if not os.path.exists(file_path):
        logger.error("File not found")
        sys.exit(1)

    # 1. Validation & Setup
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        sys.exit(1)

    logger.info(f"Ingesting {total_pages} pages in batches of {batch_size}...")

    os.makedirs(output_dir, exist_ok=True)

    run_manifest = {
        "file": file_path,
        "total_pages": total_pages,
        "batch_size": batch_size,
        "batches": [],
        "status": "running",
        "kb_name": kb_name,
        "use_mock_flush": bool(use_mock_flush),
        "timestamp_start": time.time(),
    }

    failed_batches = []

    try:
        # 2. Process Loop
        for start_page in range(1, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size, total_pages + 1)
            batch_id = f"{start_page}-{end_page - 1}"
            queue = multiprocessing.Queue()

            # Spawn worker
            p = multiprocessing.Process(
                target=worker_process_page_batch,
                args=(file_path, start_page, end_page, queue, kb_name, use_mock_flush),
            )

            batch_start = time.time()
            p.start()

            # Start monitor for this worker PID (monitor the worker, not the parent)
            monitor_proc = None
            monitor_dir = None
            try:
                monitor_proc, monitor_dir = _start_monitor_for_pid(p.pid, output_dir, batch_id)
            except Exception:
                monitor_proc = None
                monitor_dir = None

            # Wait for worker with timeout
            p.join(timeout=120)

            # If worker still alive after timeout
            if p.is_alive():
                logger.error(f"Worker timed out on pages {start_page}-{end_page - 1}! Terminating...")
                p.terminate()
                p.join()
                failed_batches.append({"batch": batch_id, "reason": "timeout"})
                # Ensure monitor is terminated
                if monitor_proc:
                    monitor_proc.terminate()
                run_manifest["batches"].append(
                    {"batch_id": batch_id, "status": "timeout", "duration_s": time.time() - batch_start}
                )
                continue

            # Worker exit code classification
            exitcode = p.exitcode
            if exitcode is None:
                exitcode = -999  # unknown
            if exitcode != 0:
                # Common signal for OOM in containers is -9 (killed)
                reason = "crash"
                if exitcode < 0:
                    reason = f"signal_{abs(exitcode)}"
                logger.error(f"Worker crashed (Exit code {exitcode}) on pages {start_page}-{end_page - 1}.")
                failed_batches.append({"batch": batch_id, "reason": reason})
                # Terminate monitor
                if monitor_proc:
                    monitor_proc.terminate()
                run_manifest["batches"].append(
                    {"batch_id": batch_id, "status": "crash", "exitcode": exitcode, "duration_s": time.time() - batch_start}
                )
                continue

            # Collect worker result
            batch_result = {"batch_id": batch_id, "status": "unknown", "duration_s": time.time() - batch_start}
            if not queue.empty():
                res = queue.get()
                if res.get("status") == "success":
                    logger.info(f"Batch {batch_id} completed in {time.time() - batch_start:.2f}s")
                    batch_result["status"] = "success"
                    batch_result["items"] = len(res.get("data", []))
                else:
                    logger.error(f"Worker error: {res.get('message')}")
                    batch_result["status"] = "error"
                    batch_result["error"] = res.get("message")
                    failed_batches.append({"batch": batch_id, "reason": res.get("message")})
            else:
                logger.error("Worker returned no result.")
                batch_result["status"] = "no_result"
                failed_batches.append({"batch": batch_id, "reason": "no_result"})

            # Ensure monitor is terminated and collect its summary if present
            if monitor_proc:
                try:
                    # Give monitor a moment to flush
                    time.sleep(0.2)
                    monitor_proc.terminate()
                except Exception:
                    pass

            # Attach monitor summary if available
            if monitor_dir:
                summary_path = os.path.join(monitor_dir, "summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r") as sf:
                            batch_summary = json.load(sf)
                        batch_result["monitor_summary"] = batch_summary
                    except Exception:
                        pass

            run_manifest["batches"].append(batch_result)

            # Parent cleanup
            import gc

            gc.collect()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        run_manifest["status"] = "interrupted"
    finally:
        run_manifest["timestamp_end"] = time.time()
        run_manifest["failed_batches"] = failed_batches
        run_manifest["status"] = "completed" if not failed_batches else "partial_failure"

        manifest_path = os.path.join(output_dir, "ingest_manifest.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(run_manifest, f, indent=2)
            logger.info(f"Wrote run manifest to {manifest_path}")
        except Exception as e:
            logger.error(f"Failed to write manifest: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents in isolated worker processes for OOM repro.")
    parser.add_argument("--file", required=True, help="Path to PDF file to ingest")
    parser.add_argument("--kb_name", default="test", help="Knowledge base name (for metadata)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of pages per worker batch (default: 1 for determinism)")
    parser.add_argument("--max-bytes", type=int, default=5000000, help="Soft max bytes per flush (heuristic)")
    parser.add_argument("--output-dir", default="tests/performance/oom_repro/logs", help="Directory for logs and artifacts")
    parser.add_argument("--use-mock-flush", action="store_true", help="Use deterministic mock flush adapter (CI)")
    args = parser.parse_args()

    # Ensure batch-size is at least 1
    batch_size = max(1, args.batch_size)

    run_ingestion(args.file, args.kb_name, batch_size=batch_size, max_bytes=args.max_bytes, output_dir=args.output_dir, use_mock_flush=args.use_mock_flush)
