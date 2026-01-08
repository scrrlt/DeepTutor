# OOM Reproduction and Memory Safety Toolkit

A deterministic, low‑risk framework for reproducing memory failures on APU and shared‑memory systems. It generates stable, reviewable artifacts for post‑mortem analysis and bug reporting, while protecting the host workstation, display stack, and GPU drivers.

---

## Quick Start

### 1. Safe Reproduction (Docker — Recommended)

Run the ingestion harness inside a memory‑capped container to isolate the host from instability.

```bash
chmod +x tests/performance/oom_repro/run_in_docker.sh
./tests/performance/oom_repro/run_in_docker.sh /path/to/large_doc.pdf 6g
```

**Execution Details**
- Reproduction image is built automatically.
- Harness runs inside a container restricted to the specified memory cap.
- All artifacts are written to `tests/performance/oom_repro/logs/`.

---

### 2. Manual Subprocess Harness (Advanced)

Run the isolated ingestion harness directly for debugging in controlled environments.

```bash
python3 tests/performance/oom_repro/ingest_isolated.py \
  --file /path/to/doc.pdf \
  --kb_name test_kb \
  --batch-size 10 \
  --max-bytes 5000000 \
  --output-dir tests/performance/oom_repro/logs
```

**Operational Characteristics**
- Worker processes are isolated and short‑lived.
- Native memory is reclaimed on worker exit.
- Intended for debugging or tightly controlled environments.

---

### 3. Validate Cleanup Behaviour

Verify memory reclamation using:

```bash
python3 tests/performance/oom_repro/simulate_pressure.py --target-mb 2000
```

> **Note**  
> Failure indicates native allocations (drivers, libraries, frameworks) are not releasing memory correctly.

---

## Outputs and Artifacts

All artifacts are stored in:  
`tests/performance/oom_repro/logs/`

### Key Files

| File                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `memory_log.csv`         | RSS/VMS time‑series sampled at sub‑second resolution.                       |
| `summary.json`           | peak_rss_mb, duration, sample count, cgroup events.                         |
| `ingest_manifest.json`   | Per‑batch metadata: batch_id, status, timing.                               |
| `dmesg_excerpt.txt`      | Kernel messages (if available).                                             |
| `container_manifest.json`| File size, memory caps, exit codes.                                         |

---

## Safety Checklist

Before running a reproduction:

- [ ] Save all open work and close unsaved documents.  
- [ ] Close browsers, editors, VMs, and other heavy applications.  
- [ ] Prefer Docker to protect the host system and display stack.  
- [ ] Ensure SSH access is available in case the machine becomes unresponsive.  

---

## Interpretation of Results

### Peak RSS
Found in `summary.json`; compare against container memory cap.

### Batch Failures
Inspect `ingest_manifest.json` for `status` and `error` fields.

### Cgroup Events
Increases in `oom_kill` or `v2:oom` indicate kernel‑level termination.

### Memory Curves
Use `memory_log.csv` to identify rapid RSS spikes before termination.

### Display / Driver Stability
If display fails while RSS remains bounded, inspect:
- `dmesg_excerpt.txt`
- GPU driver logs

---

## Recommended Parameters and CI Guidance

### Local Reproduction
- Memory cap: **6g** (also test at 4g and 8g)  
- Sampling interval: **0.2s**  
- Batching defaults:  
  - `batch_size = 10`  
  - `max_bytes = 5,000,000`  
  - `max_wait_ms = 5000`  

### CI Smoke Tests
- Use a small synthetic PDF.  
- Enable mock flush adapter.  
- Run inside a `--memory=2g` container.  
- Assertions:  
  - `peak_rss_mb` < 95% of cap  
  - At least one successful flush recorded  

---

## Troubleshooting

### Container OOMKilled
Inspect:
- `container_manifest.json`
- `summary.json`
- `memory_log.csv`

### System Instability with Low RSS
Check:
- `dmesg`
- GPU driver logs  
Driver crashes may occur even with bounded RSS.

### `simulate_pressure.py` Failure
Indicates native or driver‑level memory retention issues.

---

## Bug Report Requirements

Include:

- Logs from `run_in_docker.sh` using the reported file + memory cap.
- A compressed archive of `tests/performance/oom_repro/logs/` containing:  
  - `memory_log.csv`  
  - `summary.json`  
  - `ingest_manifest.json`  
  - `dmesg_excerpt.txt`  

### System Metadata
- Host OS + Kernel version  
- GPU/APU model + Driver version  