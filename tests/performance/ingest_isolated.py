#!/usr/bin/env python3
"""
tests/performance/oom_repro/ingest_isolated.py

Process‑pool based ingestion harness for deterministic OOM reproduction.

Key properties:
- Uses multiprocessing.Pool with maxtasksperchild to amortize startup cost
  while guaranteeing periodic memory reclamation.
- Each worker self‑monitors via a sidecar monitor process.
- Explicitly breaks pdfplumber internal page references (pdf.pages[i] = None).
- No IPC queues; workers return small, picklable result dicts only.
- Produces per‑batch telemetry and a run manifest suitable for CI gating.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List

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


# -----------------------------
# Monitoring sidecar
# -----------------------------
def _start_monitor(pid: int, output_dir: str, batch_id: str, interval: float = 0.2):
    batch_dir = os.path.join(output_dir, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

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
    proc = subprocess.Popen(cmd)
    return proc, batch_dir


# -----------------------------
# Worker task
# -----------------------------
def worker_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single batch inside a pool worker.
    Returns a small, picklable result dict.
    """
    file_path = args["file_path"]
    start_page = args["start_page"]
    end_page = args["end_page"]
    output_dir = args["output_dir"]
    use_mock_flush = args.get("use_mock_flush", False)

    batch_id = f"{start_page}-{end_page - 1}"
    pid = os.getpid()
    start_time = time.time()

    monitor_proc = None
    monitor_dir = None

    try:
        monitor_proc, monitor_dir = _start_monitor(pid, output_dir, batch_id)
    except Exception as e:
        logger.warning(f"Monitor start failed for batch {batch_id}: {e}")

    status = "success"
    error = None
    items = 0

    try:
        import pdfplumber
        import psutil

        process = psutil.Process(pid)

        mock_adapter = None
        if use_mock_flush and MockFlushAdapter is not None:
            mock_adapter = MockFlushAdapter(latency_ms=50, mem_delta_mb=0, fail_rate=0.0)

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            effective_end = min(end_page, total_pages + 1)

            for i in range(start_page, effective_end):
                # Hard safety watermark (2GB RSS)
                if process.memory_info().rss > 2 * 1024 * 1024 * 1024:
                    raise MemoryError(f"RSS safety limit exceeded at page {i}")

                page = pdf.pages[i - 1]
                try:
                    text = page.extract_text(layout=False) or ""
                    items += 1

                    if mock_adapter:
                        import asyncio
                        asyncio.run(
                            mock_adapter.flush(
                                [{"content": text, "metadata": {"page": i}}]
                            )
                        )
                finally:
                    try:
                        page.flush_cache()
                    except Exception:
                        pass
                    # CRITICAL: break pdfplumber internal reference
                    pdf.pages[i - 1] = None
                    del page

    except Exception as e:
        status = "error"
        error = str(e)
        traceback.print_exc()

    finally:
        if monitor_proc:
            try:
                time.sleep(0.1)
                monitor_proc.terminate()
                monitor_proc.wait(timeout=2)
            except Exception:
                pass

    monitor_summary = {}
    if monitor_dir:
        summary_path = os.path.join(monitor_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    monitor_summary = json.load(f)
            except Exception:
                pass

    return {
        "batch_id": batch_id,
        "pid": pid,
        "status": status,
        "error": error,
        "items": items,
        "duration_s": round(time.time() - start_time, 3),
        "monitor_summary": monitor_summary,
    }


# -----------------------------
# Orchestrator
# -----------------------------
def run_ingestion(
    file_path: str,
    batch_size: int,
    output_dir: str,
    use_mock_flush: bool,
):
    if not os.path.exists(file_path):
        logger.error("File not found")
        sys.exit(1)

    import pdfplumber
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

    logger.info(f"Ingesting {total_pages} pages using process pool (batch={batch_size})")
    os.makedirs(output_dir, exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size, total_pages + 1)
        tasks.append(
            {
                "file_path": file_path,
                "start_page": start,
                "end_page": end,
                "output_dir": output_dir,
                "use_mock_flush": use_mock_flush,
            }
        )

    manifest = {
        "file": file_path,
        "total_pages": total_pages,
        "batch_size": batch_size,
        "timestamp_start": time.time(),
        "batches": [],
        "failed_batches": [],
        "status": "running",
    }

    TASKS_PER_CHILD = 5
    ctx = multiprocessing.get_context("spawn")

    try:
        with ctx.Pool(
            processes=max(1, multiprocessing.cpu_count() - 1),
            maxtasksperchild=TASKS_PER_CHILD,
        ) as pool:
            for result in pool.imap_unordered(worker_task, tasks):
                manifest["batches"].append(result)
                if result["status"] != "success":
                    manifest["failed_batches"].append(
                        {"batch": result["batch_id"], "reason": result["error"]}
                    )
                    logger.error(f"Batch {result['batch_id']} failed: {result['error']}")
                else:
                    logger.info(
                        f"Batch {result['batch_id']} OK "
                        f"(PID {result['pid']}, {result['duration_s']}s)"
                    )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        manifest["status"] = "interrupted"
    except Exception as e:
        logger.error(f"Pool execution error: {e}")
        manifest["status"] = "error"
    finally:
        manifest["timestamp_end"] = time.time()
        if manifest["status"] == "running":
            manifest["status"] = (
                "completed" if not manifest["failed_batches"] else "partial_failure"
            )

        manifest_path = os.path.join(output_dir, "ingest_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Wrote run manifest to {manifest_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOM repro harness (process pool)")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--batch-size", type=int, default=1, help="Pages per worker task")
    parser.add_argument("--output-dir", default="tests/performance/oom_repro/logs")
    parser.add_argument("--use-mock-flush", action="store_true")
    args = parser.parse_args()

    run_ingestion(
        file_path=args.file,
        batch_size=max(1, args.batch_size),
        output_dir=args.output_dir,
        use_mock_flush=args.use_mock_flush,
    )
