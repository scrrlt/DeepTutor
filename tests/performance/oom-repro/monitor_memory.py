#!/usr/bin/env python3
# tests/performance/oom_repro/monitor_memory.py
"""
Lightweight process memory monitor for OOM repro runs.

Features and improvements:
- Aggregates RSS/VMS across the target process and its children.
- Sub-second sampling (default 0.2s) to capture fast OOM spikes.
- Writes CSV time series and a JSON summary with peak RSS and metadata.
- Attempts to detect cgroup OOM events (v1 and v2) when available.
- Robust to AccessDenied/NoSuchProcess and supports clean shutdown.
"""

import psutil
import time
import argparse
import csv
import os
import json
import signal
from datetime import datetime

# Sampling defaults
DEFAULT_INTERVAL = 0.2  # seconds

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGINT, _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)


def _read_cgroup_oom_events():
    """
    Try to read cgroup OOM events for both cgroup v1 and v2.
    Returns a dict with keys found and integer counts where available.
    """
    events = {}
    # cgroup v2
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

    # cgroup v1 (legacy)
    try:
        path_v1 = "/sys/fs/cgroup/memory/memory.oom_control"
        if os.path.exists(path_v1):
            with open(path_v1, "r") as f:
                content = f.read()
                # memory.oom_control contains "oom_kill_disable" and "under_oom"
                for line in content.splitlines():
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(':')
                            val = parts[-1]
                            try:
                                events[f"v1:{key}"] = int(val)
                            except ValueError:
                                events[f"v1:{key}"] = val
    except Exception:
        pass

    return events


def _aggregate_memory(proc):
    """
    Aggregate RSS and VMS across proc and all descendants.
    Returns (rss_bytes, vms_bytes).
    """
    rss = 0
    vms = 0
    try:
        procs = [proc] + proc.children(recursive=True)
        for p in procs:
            try:
                mi = p.memory_info()
                rss += getattr(mi, "rss", 0)
                vms += getattr(mi, "vms", 0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # If parent disappears, return zeros
        return 0, 0
    return rss, vms


def monitor(pid, output_csv, output_json, interval=DEFAULT_INTERVAL):
    """
    Monitor the given PID and write CSV and JSON summary.

    CSV columns:
      Timestamp, PID, RSS_MB, VIRT_MB, CPU_Percent, Swap_MB, Num_Threads, Status

    JSON summary includes peak_rss_mb, peak_time, samples, duration_sec, cgroup_events.
    """
    print(f"Monitoring PID {pid} (interval={interval}s)...")

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found.")
        return

    peak_rss_mb = 0.0
    peak_time_iso = None
    sample_count = 0
    start_time = time.time()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Open CSV and write header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Timestamp",
                "PID",
                "RSS_MB",
                "VIRT_MB",
                "CPU_Percent",
                "Swap_MB",
                "Num_Threads",
                "Status",
            ]
        )

        try:
            # Warm up cpu_percent measurement
            try:
                process.cpu_percent(interval=None)
            except Exception:
                pass

            while process.is_running() and not _shutdown:
                try:
                    rss_bytes, vms_bytes = _aggregate_memory(process)
                    rss_mb = rss_bytes / (1024 * 1024)
                    virt_mb = vms_bytes / (1024 * 1024)

                    # cpu_percent with interval=None returns last computed value; call once per loop
                    cpu_pct = process.cpu_percent(interval=None)

                    # swap may not be present on all platforms; best-effort
                    swap_mb = 0.0
                    try:
                        # psutil.Process.memory_info() may not include swap; use system swap as fallback
                        mi = process.memory_info()
                        if hasattr(mi, "swap"):
                            swap_mb = getattr(mi, "swap", 0) / (1024 * 1024)
                    except Exception:
                        swap_mb = 0.0

                    threads = process.num_threads()
                    status = process.status()

                    timestamp = datetime.now().isoformat()
                    writer.writerow(
                        [
                            timestamp,
                            pid,
                            f"{rss_mb:.2f}",
                            f"{virt_mb:.2f}",
                            f"{cpu_pct:.1f}",
                            f"{swap_mb:.2f}",
                            threads,
                            status,
                        ]
                    )
                    f.flush()

                    sample_count += 1
                    if rss_mb > peak_rss_mb:
                        peak_rss_mb = rss_mb
                        peak_time_iso = timestamp

                    # Sleep with small granularity to be responsive to shutdown
                    slept = 0.0
                    while slept < interval and not _shutdown:
                        time.sleep(min(0.05, interval - slept))
                        slept += min(0.05, interval - slept)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except KeyboardInterrupt:
            pass

    duration = time.time() - start_time

    # Attempt to read cgroup oom events (best-effort)
    cgroup_events = _read_cgroup_oom_events()

    summary = {
        "peak_rss_mb": round(peak_rss_mb, 2),
        "peak_time": peak_time_iso,
        "samples": sample_count,
        "duration_sec": round(duration, 2),
        "final_status": "completed" if not _shutdown else "terminated",
        "timestamp": datetime.now().isoformat(),
        "cgroup_events": cgroup_events,
    }

    # Write JSON summary
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