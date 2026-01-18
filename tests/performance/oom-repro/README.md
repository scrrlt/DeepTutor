This folder contains a safe, repeatable toolkit for reproducing and diagnosing Out‑Of‑Memory (OOM) crashes in document ingestion pipelines.

The toolkit is designed for:

Deterministic reproduction of memory failures

Minimal host risk, even on APU and shared‑memory systems

Actionable artifacts suitable for post‑mortem analysis and bug reports

It allows you to validate memory‑safety fixes without risking the developer workstation, display stack, or GPU drivers.

Quick Start
1. Safe Reproduction (Docker — Recommended)
Run the ingestion harness inside a memory‑capped container.

bash
chmod +x tests/performance/oom_repro/run_in_docker.sh
./tests/performance/oom_repro/run_in_docker.sh /path/to/large_doc.pdf 6g
This will:

Build the repro image

Run the ingestion harness inside a container capped at the specified memory

Write all artifacts to tests/performance/oom_repro/logs/

2. Run the Subprocess Harness Directly (Advanced)
Run the isolated ingestion harness without Docker.

bash
python3 tests/performance/oom_repro/ingest_isolated.py \
  --file /path/to/doc.pdf \
  --kb_name test_kb \
  --batch-size 10 \
  --max-bytes 5000000 \
  --output-dir tests/performance/oom_repro/logs
This mode:

Spawns isolated worker processes

Reclaims native memory on worker exit

Is useful for debugging or controlled environments

3. Validate Cleanup Behavior
Verify that memory is reclaimed correctly in your environment.

bash
python3 tests/performance/oom_repro/simulate_pressure.py --target-mb 2000
If this test fails, native allocations (e.g., drivers, libraries, or frameworks) may not be releasing memory correctly.

Outputs & Artifacts
All run artifacts are written to:

Code
tests/performance/oom_repro/logs/
Key files include:

memory_log.csv  
Time‑series of RSS and VMS sampled at sub‑second resolution.

summary.json  
Run summary including:

peak_rss_mb

duration_sec

samples

cgroup_events (when available)

ingest_manifest.json  
Per‑batch execution metadata:

batch_id

status (success, timeout, worker_crash, oom_killed)

timing and error classification

dmesg_excerpt.txt (if available)  
Kernel messages captured after the run.

container_manifest_TIMESTAMP.json  
Container‑level metadata:

PDF name and size

memory cap

exit code

OOMKilled flag

These artifacts are intended to be attached directly to bug reports.

Safety Checklist
Before running any repro:

Save all open work and close unsaved documents

Close heavy GUI applications (browsers, editors, VMs)

Prefer Docker to protect the host system and display stack

Have SSH or another device available in case the test machine becomes unresponsive

How to Interpret Results
Peak RSS (summary.json)
Highest aggregated RSS observed for the monitored PID and its children.
Compare this to the container memory cap.

Batch failures (ingest_manifest.json)
Use status and error fields to quickly identify failing batches.

Cgroup events (summary.json)
If oom_kill or v2:oom counters increase, the kernel likely killed a process.

Memory curves (memory_log.csv)
Look for the classic “wall of OOM” pattern: a rapid RSS spike followed by termination.

Display or driver crashes  
If RSS appears bounded but the display goes black, inspect dmesg_excerpt.txt and GPU driver logs.

Recommended Parameters & CI Guidance
Local Reproduction
Memory cap: 6g (also test 4g and 8g)

Sampling interval: 0.2s

Batching defaults:

batch_size = 10

max_bytes = 5_000_000

max_wait_ms = 5000

CI Smoke Tests
Use a small synthetic PDF

Enable the mock flush adapter

Run inside a --memory=2g container

Assert:

peak_rss_mb < 95% of the cap

at least one successful flush

Keep hardware‑specific tests out of PR CI; run them on dedicated machines

Troubleshooting & Next Steps
Container exits with OOMKilled  
Inspect container_manifest_TIMESTAMP.json, summary.json, and memory_log.csv.

Low RSS but system instability  
Collect dmesg and GPU driver logs. Display‑driver crashes can occur even when process RSS appears bounded.

simulate_pressure.py fails  
Run it inside the same environment as ingestion. Failure suggests native or driver‑level memory retention.

Reproducing user reports  
Ask the user to run the Docker script and share the logs/ bundle.
The container is safe to run on user machines because it caps memory and mounts PDFs read‑only.

Minimal Checklist for Bug Reports
Run run_in_docker.sh with the reported PDF and memory cap

Zip tests/performance/oom_repro/logs/ and include:

memory_log.csv

summary.json

ingest_manifest.json

dmesg_excerpt.txt (if available)

Record:

Host OS

Kernel version

GPU/APU model

Driver version