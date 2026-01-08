#!/usr/bin/env python3
"""
tests/performance/monitor_memory.py

Sidecar process to monitor memory usage of a target PID.
Writes statistics to a summary.json and detailed logs to memory_log.csv.
"""
import argparse
import csv
import json
import os
import time
import psutil
import sys

def monitor(pid: int, output_dir: str, interval: float = 0.2):
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "memory_log.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    peak_rss_mb = 0.0
    start_time = time.time()
    samples = 0

    print(f"Monitoring PID {pid} -> {output_dir}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rss_mb", "vms_mb", "cpu_percent"])

        while True:
            try:
                # Check if process is still alive
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    break

                # Gather stats
                mem = process.memory_info()
                rss_mb = mem.rss / (1024 * 1024)
                vms_mb = mem.vms / (1024 * 1024)
                cpu = process.cpu_percent(interval=0.0) # non-blocking

                peak_rss_mb = max(peak_rss_mb, rss_mb)
                samples += 1

                writer.writerow([time.time(), round(rss_mb, 2), round(vms_mb, 2), round(cpu, 2)])
                f.flush()

                time.sleep(interval)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                break

    duration = time.time() - start_time

    summary = {
        "pid": pid,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "duration_s": round(duration, 2),
        "samples": samples
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    monitor(args.pid, args.output_dir, args.interval)