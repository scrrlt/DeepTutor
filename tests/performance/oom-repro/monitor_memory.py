#!/usr/bin/env python3
# tests/performance/oom_repro/monitor_memory.py
"""
Lightweight process memory monitor for OOM repro runs.

Features:
- Aggregates RSS/VMS across the target PID and its children (recursive).
- Sub-second sampling (default 0.2s) to capture fast OOM spikes.
- Writes CSV time series and a JSON summary (peak RSS, duration, samples).
- Captures cgroup OOM event counters (v1/v2) when available.
- Captures cgroup memory usage/limit when available (container-aware).
- Robust to AccessDenied/NoSuchProcess and supports SIGINT/SIGTERM shutdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

import psutil

DEFAULT_INTERVAL = 0.2  # seconds
_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)


def _read_file_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _read_file_str(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return None


def _read_cgroup_memory() -> Dict[str, Any]:
    """
    Best-effort container-aware memory snapshot.

    Returns:
      {
        "mode": "cgroupv2" | "cgroupv1" | "unknown",
        "usage_bytes": int | null,
        "limit_bytes": int | null,
        "percent": float | null
      }
    """
    # cgroup v2
    try:
        cur = _read_file_int("/sys/fs/cgroup/memory.current")
        lim_s = _read_file_str("/sys/fs/cgroup/memory.max")
        if cur is not None and lim_s is not None:
            if lim_s != "max":
                lim = int(lim_s)
                pct = (cur / lim) * 100.0 if lim > 0 else None
                return {"mode": "cgroupv2", "usage_bytes": cur, "limit_bytes": lim, "percent": pct}
            return {"mode": "cgroupv2", "usage_bytes": cur, "limit_bytes": None, "percent": None}
    except Exception:
        pass

    # cgroup v1
    try:
        cur = _read_file_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        lim = _read_file_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if cur is not None and lim is not None:
            # Some envs use absurdly high lim for "no limit"
            if lim > 0 and lim < 10**15:
                pct = (cur / lim) * 100.0
                return {"mode": "cgroupv1", "usage_bytes": cur, "limit_bytes": lim, "percent": pct}
            return {"mode": "cgroupv1", "usage_bytes": cur, "limit_bytes": None, "percent": None}
    except Exception:
        pass

    return {"mode": "unknown", "usage_bytes": None, "limit_bytes": None, "percent": None}


def _read_cgroup_oom_events() -> Dict[str, Any]:
    """
    Best-effort cgroup OOM event counters (v1 + v2).
    Returns a dict with stable keys; values may be int/str.
    """
    events: Dict[str, Any] = {}

    # cgroup v2: memory.events
    try:
        path_v2 = "/sys/fs/cgroup/memory.events"
        if os.path.exists(path_v2):
            with open(path_v2, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        key, val = parts
                        try:
                            events[f"v2:{key}"] = int(val)
                        except ValueError:
                            events[f"v2:{key}"] = val
    except Exception:
        pass

    # cgroup v1: memory.oom_control (not counters, but state flags)
    try:
        path_v1 = "/sys/fs/cgroup/memory/memory.oom_control"
        if os.path.exists(path_v1):
            with open(path_v1, "r") as f:
                content = f.read()
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val = parts[-1]
                    try:
                        events[f"v1:{key}"] = int(val)
                    except ValueError:
                        events[f"v1:{key}"] = val
    except Exception:
        pass

    return events


def _aggregate_memory(proc: psutil.Process) -> Tuple[int, int, int, int]:
    """
    Aggregate RSS/VMS and threads across proc and all descendants.
    Returns (rss_bytes, vms_bytes, threads_total, procs_count).
    """
    rss = 0
    vms = 0
    threads = 0
    count = 0
    try:
        procs = [proc] + proc.children(recursive=True)
        for p in procs:
            try:
                mi = p.memory_info()
                rss += getattr(mi, "rss", 0)
                vms += getattr(mi, "vms", 0)
                threads += p.num_threads()
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0, 0, 0
    return rss, vms, threads, count


def monitor(pid: int, output_csv: str, output_json: str, interval: float = DEFAULT_INTERVAL) -> None:
    """
    Monitor the given PID and write CSV and JSON summary.

    CSV columns:
      Timestamp, PID, RSS_MB, VIRT_MB, CPU_Percent, Procs, Threads, Status, CgroupMemPercent

    JSON summary includes:
      peak_rss_mb, peak_time, samples, duration_sec, exit_reason,
      cgroup_memory_start/end, cgroup_events_start/end.
    """
    print(f"Monitoring PID {pid} (interval={interval}s)...")

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found.")
        return

    out_dir = os.path.dirname(output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    peak_rss_mb = 0.0
    peak_time_iso: Optional[str] = None
    sample_count = 0
    start_time = time.time()

    cgroup_events_start = _read_cgroup_oom_events()
    cgroup_mem_start = _read_cgroup_memory()

    exit_reason = "unknown"

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Timestamp",
                "PID",
                "RSS_MB",
                "VIRT_MB",
                "CPU_Percent",
                "Procs",
                "Threads",
                "Status",
                "CgroupMemPercent",
            ]
        )

        # Warm up cpu_percent measurement
        try:
            process.cpu_percent(interval=None)
        except Exception:
            pass

        try:
            while not _shutdown:
                try:
                    if not process.is_running():
                        exit_reason = "not_running"
                        break

                    rss_bytes, vms_bytes, threads_total, procs_count = _aggregate_memory(process)
                    rss_mb = rss_bytes / (1024 * 1024)
                    virt_mb = vms_bytes / (1024 * 1024)

                    # cpu_percent with interval=None returns last computed value
                    cpu_pct = process.cpu_percent(interval=None)

                    status = "unknown"
                    try:
                        status = process.status()
                    except Exception:
                        pass

                    cg_mem = _read_cgroup_memory()
                    cg_pct = cg_mem.get("percent", None)
                    cg_pct_str = f"{cg_pct:.2f}" if isinstance(cg_pct, (float, int)) and cg_pct is not None else ""

                    timestamp = datetime.now().isoformat()
                    writer.writerow(
                        [
                            timestamp,
                            pid,
                            f"{rss_mb:.2f}",
                            f"{virt_mb:.2f}",
                            f"{cpu_pct:.1f}",
                            procs_count,
                            threads_total,
                            status,
                            cg_pct_str,
                        ]
                    )
                    f.flush()

                    sample_count += 1
                    if rss_mb > peak_rss_mb:
                        peak_rss_mb = rss_mb
                        peak_time_iso = timestamp

                    # Sleep with responsiveness to shutdown
                    slept = 0.0
                    while slept < interval and not _shutdown:
                        step = min(0.05, interval - slept)
                        time.sleep(step)
                        slept += step

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    exit_reason = "process_gone_or_denied"
                    break
        except KeyboardInterrupt:
            exit_reason = "keyboard_interrupt"
        finally:
            if _shutdown and exit_reason == "unknown":
                exit_reason = "terminated"

    duration = time.time() - start_time
    cgroup_events_end = _read_cgroup_oom_events()
    cgroup_mem_end = _read_cgroup_memory()

    summary = {
        "pid": pid,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "peak_time": peak_time_iso,
        "samples": sample_count,
        "duration_sec": round(duration, 2),
        "exit_reason": exit_reason,
        "timestamp": datetime.now().isoformat(),
        "cgroup_memory_start": cgroup_mem_start,
        "cgroup_memory_end": cgroup_mem_end,
        "cgroup_events_start": cgroup_events_start,
        "cgroup_events_end": cgroup_events_end,
    }

    with open(output_json, "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"\nMonitoring finished. Peak RSS: {peak_rss_mb:.2f} MB (samples: {sample_count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor process memory (RSS/VMS) and write CSV + JSON summary.")
    parser.add_argument("--pid", type=int, required=True, help="PID to monitor")
    parser.add_argument(
        "--output-dir",
        default="tests/performance/oom_repro/logs",
        help="Directory to write memory_log.csv and summary.json",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Sampling interval in seconds (supports sub-second, default 0.2)",
    )
    args = parser.parse_args()

    out_csv = os.path.join(args.output_dir, "memory_log.csv")
    out_json = os.path.join(args.output_dir, "summary.json")
    monitor(args.pid, out_csv, out_json, args.interval)
