This repository folder contains a safe, repeatable toolkit to reproduce and diagnose Out‑Of‑Memory (OOM) crashes for document ingestion pipelines. It focuses on deterministic reproduction, minimal host risk, and actionable artifacts so you can validate fixes without risking the developer workstation or display stack on APU systems.

Quick start
- Build and run (Docker, recommended)
chmod +x tests/performance/oom_repro/run_in_docker.sh
./tests/performance/oom_repro/run_in_docker.sh /path/to/large_doc.pdf 6g
- The script builds the image, runs the ingestion harness inside a container capped at the specified memory, and writes artifacts to tests/performance/oom_repro/logs/.
- Run subprocess harness directly
python3 tests/performance/oom_repro/ingest_isolated.py \
  --file /path/to/doc.pdf \
  --kb_name test_kb \
  --batch-size 10 \
  --max-bytes 5000000 \
  --output-dir tests/performance/oom_repro/logs
- Validate cleanup behavior
python3 tests/performance/oom_repro/simulate_pressure.py --target-mb 2000
Outputs and artifactsAll run artifacts are written to tests/performance/oom_repro/logs/ by default. Key files:- memory_log.csv — time series of RSS and VMS (per sample).
- summary.json — run summary including peak_rss_mb, duration_sec, samples, and cgroup_events if available.
- ingest_manifest.json — per‑batch status, timestamps, and failure classification (e.g., success, timeout, worker_crash, oom_killed).
- dmesg_excerpt.txt — kernel messages captured after a run (if available).
- container_manifest_TIMESTAMP.json — run metadata (pdf name, size, memory cap, exit code, OOM flag).
These artifacts are designed to be attached to bug reports and to drive post‑mortem analysis.Safety checklistBefore running any repro:- Save all open work and close unsaved documents.
- Close heavy GUI apps (browsers, editors, VMs) to reduce noise.
- Prefer Docker: use the Docker method to protect the host display and system.
- Have SSH or another device ready in case the test machine becomes unresponsive so you can collect logs without relying on the local display.
How to interpret results- Peak RSS in summary.json shows the highest aggregated RSS observed for the monitored PID and its children. Compare this to the container memory cap.
- ingest_manifest.json entries include batch_id, start_time, end_time, status, and error. Use status to quickly find failing batches.
- cgroup events in summary.json indicate kernel/cgroup OOM activity. If oom_kill or v2:oom counters increase, the kernel likely killed a process.
- memory_log.csv lets you plot memory curves to identify the “wall of OOM” pattern: a rapid RSS spike followed by process termination.
- dmesg_excerpt.txt often contains driver or kernel messages that explain display-driver crashes or GPU-related failures.
Recommended run parameters and CI guidance- Default memory cap: 6g for local repro; use 4g and 8g for additional sensitivity testing.
- Sampling interval: 0.2s for monitor to capture fast spikes.
- Batching defaults: batch_size=10, max_bytes=5_000_000, max_wait_ms=5000. Use the adaptive batching logic in the manager for production.
- CI smoke test: run the harness with a small synthetic PDF and the mock flush adapter inside a --memory=2g container. Assert peak_rss_mb < 95% of the cap and flush_count > 0. Keep heavy hardware-specific tests out of PR CI; run them on dedicated test machines.
Troubleshooting and next steps- If the container exits with OOMKilled: check container_manifest_TIMESTAMP.json and summary.json. Collect dmesg_excerpt.txt and the memory_log.csv to identify the offending batch.
- If monitor shows low RSS but display goes black: collect dmesg and driver logs from the affected machine; display-driver crashes can occur even when process RSS appears bounded.
- If simulate_pressure.py fails to reclaim memory: run it inside the same environment as the ingestion test and compare results; a failure suggests native allocations or driver-level allocations that Python GC cannot free.
- To reproduce a specific user report: ask the user to run the Docker script on their machine and share the logs/ bundle. The container is safe to run on user machines because it caps memory and mounts the PDF read‑only.
Minimal run checklist for bug reports- Run run_in_docker.sh with the reported PDF and memory cap.
- Zip tests/performance/oom_repro/logs/ and include: memory_log.csv, summary.json, ingest_manifest.json, and dmesg_excerpt.txt.
- Note the host OS, kernel version, GPU/APU model, and driver version. Attach these to the bug.
