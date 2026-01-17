#!/usr/bin/env python3
# tests/performance/oom_repro/simulate_pressure.py
"""
Memory pressure simulator for validating cleanup behavior.

Goals:
- Create repeatable RSS pressure patterns (steady ramp, fragmentation-like churn).
- Verify that memory is reclaimed after releasing references + GC.
- Optionally test native allocator cleanup (PyTorch CPU + CUDA/MPS caches) if available.

Design notes:
- This script is intentionally simple and deterministic.
- It is not a leak detector on its own; it is an environment sanity check.
"""

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import psutil

logger = logging.getLogger("SimulatePressure")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class PressureResult:
    baseline_mb: float
    peak_mb: float
    final_mb: float
    reclaimed_mb: float
    reclaimed_ratio: float


def get_rss_mb() -> float:
    """Returns current Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def allocate_mb(size_mb: int) -> bytearray:
    """Allocates a block of memory of approximately size_mb MB."""
    if size_mb <= 0:
        size_mb = 1
    return bytearray(size_mb * 1024 * 1024)


def _maybe_sleep(ms: int) -> None:
    if ms and ms > 0:
        time.sleep(ms / 1000.0)


def test_native_cleanup(
    enable: bool,
    torch_tensor_mb: int,
    torch_device: str,
) -> None:
    """
    Best-effort native cleanup test.
    This checks whether RSS drops after releasing tensors and running cleanup hooks.
    """
    logger.info("--- Testing native cleanup ---")
    if not enable:
        logger.info("Native test skipped.")
        return

    try:
        import torch  # type: ignore
    except Exception:
        logger.warning("Torch not available; skipping native cleanup test.")
        return

    if torch_tensor_mb <= 0:
        logger.info("torch-tensor-mb <= 0; skipping tensor allocation.")
        return

    device = torch_device.lower().strip()
    if device not in ("cpu", "cuda", "mps", "auto"):
        logger.warning(f"Unknown torch device '{torch_device}', defaulting to auto.")
        device = "auto"

    chosen_device = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        chosen_device = "cuda"
    elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        chosen_device = "mps"
    elif device == "auto":
        if torch.cuda.is_available():
            chosen_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chosen_device = "mps"
        else:
            chosen_device = "cpu"

    # Allocate roughly torch_tensor_mb on the chosen device.
    # float32 = 4 bytes/elem => elems ~= (mb * 1024 * 1024) / 4
    elems = int((torch_tensor_mb * 1024 * 1024) / 4)
    if elems <= 0:
        elems = 1

    logger.info(f"Torch detected. Allocating ~{torch_tensor_mb}MB tensor on {chosen_device}...")
    before = get_rss_mb()
    t = torch.ones((elems,), dtype=torch.float32, device=chosen_device)
    mid = get_rss_mb()
    logger.info(f"Allocated tensor. RSS: {mid:.2f} MB (delta {mid - before:.2f} MB)")
    del t

    logger.info("Running cleanup hooks...")
    gc.collect()

    # CUDA cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass

    # MPS cleanup
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    after = get_rss_mb()
    logger.info(f"Post-cleanup RSS: {after:.2f} MB (reclaimed {mid - after:.2f} MB)")


def run_pattern_ramp(
    target_mb: int,
    steps: int,
    hold_ms: int,
) -> PressureResult:
    """
    Ramp pattern:
    - Allocate in 'steps' chunks until reaching ~target_mb total.
    - Hold briefly between steps.
    - Free all and measure reclaimed memory.
    """
    baseline = get_rss_mb()
    logger.info(f"Baseline RSS: {baseline:.2f} MB")

    if steps <= 0:
        steps = 4
    chunk_mb = max(1, int(target_mb / steps))

    bloat: List[bytearray] = []
    peak = baseline

    try:
        for i in range(steps):
            bloat.append(allocate_mb(chunk_mb))
            current = get_rss_mb()
            peak = max(peak, current)
            logger.info(f"Allocated step {i+1}/{steps}: +{chunk_mb}MB, RSS now {current:.2f} MB")
            _maybe_sleep(hold_ms)
    except MemoryError:
        logger.error("MemoryError during allocation (OOM hit in Python allocator).")

    peak = max(peak, get_rss_mb())
    logger.info(f"Peak RSS: {peak:.2f} MB")

    del bloat
    gc.collect()
    final = get_rss_mb()
    reclaimed = peak - final
    reclaimed_ratio = reclaimed / max(1e-9, (peak - baseline)) if peak > baseline else 1.0

    logger.info(f"Final RSS: {final:.2f} MB")
    logger.info(f"Reclaimed: {reclaimed:.2f} MB (ratio vs ramp delta: {reclaimed_ratio:.2f})")

    return PressureResult(
        baseline_mb=baseline,
        peak_mb=peak,
        final_mb=final,
        reclaimed_mb=reclaimed,
        reclaimed_ratio=reclaimed_ratio,
    )


def run_pattern_churn(
    target_mb: int,
    iterations: int,
    chunk_mb: int,
    live_set_mb: int,
    hold_ms: int,
) -> PressureResult:
    """
    Churn pattern (fragmentation-ish):
    - Maintain a live set of allocations up to live_set_mb.
    - Repeatedly allocate and free chunk_mb blocks to simulate allocator churn.
    - Measure peak and reclamation after final release.

    This is more representative of ingestion workloads that allocate/free many medium objects.
    """
    baseline = get_rss_mb()
    logger.info(f"Baseline RSS: {baseline:.2f} MB")

    if iterations <= 0:
        iterations = 100
    if chunk_mb <= 0:
        chunk_mb = 8
    if live_set_mb <= 0:
        live_set_mb = max(chunk_mb * 4, int(target_mb * 0.3))

    # Live set: keep some allocations around to keep RSS elevated.
    live_blocks: List[bytearray] = []
    live_target_blocks = max(1, int(live_set_mb / chunk_mb))

    # Churn blocks: allocate and immediately free to churn allocator.
    peak = baseline

    try:
        # Warm up live set
        for _ in range(live_target_blocks):
            live_blocks.append(allocate_mb(chunk_mb))
        peak = max(peak, get_rss_mb())
        logger.info(f"Live set established: ~{live_set_mb}MB (blocks={len(live_blocks)}), RSS {peak:.2f} MB")

        for i in range(iterations):
            # allocate
            tmp = allocate_mb(chunk_mb)
            # occasionally grow live set up to target_mb (bounded)
            if (i % 10 == 0) and (len(live_blocks) * chunk_mb < min(target_mb, live_set_mb * 2)):
                live_blocks.append(tmp)
                tmp = None  # type: ignore
            # free
            if tmp is not None:
                del tmp

            if i % 20 == 0:
                gc.collect()

            if i % 10 == 0:
                current = get_rss_mb()
                peak = max(peak, current)
                logger.info(f"Churn iter {i}/{iterations}, RSS {current:.2f} MB")
                _maybe_sleep(hold_ms)

    except MemoryError:
        logger.error("MemoryError during churn (OOM hit in Python allocator).")

    peak = max(peak, get_rss_mb())
    logger.info(f"Peak RSS: {peak:.2f} MB")

    del live_blocks
    gc.collect()
    final = get_rss_mb()
    reclaimed = peak - final
    reclaimed_ratio = reclaimed / max(1e-9, (peak - baseline)) if peak > baseline else 1.0

    logger.info(f"Final RSS: {final:.2f} MB")
    logger.info(f"Reclaimed: {reclaimed:.2f} MB (ratio vs churn delta: {reclaimed_ratio:.2f})")

    return PressureResult(
        baseline_mb=baseline,
        peak_mb=peak,
        final_mb=final,
        reclaimed_mb=reclaimed,
        reclaimed_ratio=reclaimed_ratio,
    )


def enforce_reclaim_threshold(result: PressureResult, min_reclaim_ratio: float, absolute_min_reclaim_mb: Optional[float]) -> None:
    """
    Decide pass/fail based on reclaimed ratio and (optionally) absolute reclaimed MB.
    """
    if absolute_min_reclaim_mb is not None and result.reclaimed_mb < absolute_min_reclaim_mb:
        logger.error(
            f"FAIL: reclaimed {result.reclaimed_mb:.2f}MB < absolute minimum {absolute_min_reclaim_mb:.2f}MB"
        )
        sys.exit(1)

    if result.reclaimed_ratio < min_reclaim_ratio:
        logger.error(
            f"FAIL: reclaimed ratio {result.reclaimed_ratio:.2f} < minimum {min_reclaim_ratio:.2f} (possible native retention/fragmentation)"
        )
        sys.exit(1)

    logger.info("PASS: memory reclaim appears acceptable for this environment.")


def main():
    parser = argparse.ArgumentParser(description="Simulate memory pressure and validate cleanup behavior.")
    parser.add_argument("--pattern", choices=["ramp", "churn"], default="ramp", help="Pressure pattern (default: ramp)")

    parser.add_argument("--target-mb", type=int, default=1000, help="Target pressure in MB (default: 1000)")
    parser.add_argument("--steps", type=int, default=4, help="Ramp steps (ramp pattern only, default: 4)")
    parser.add_argument("--iterations", type=int, default=120, help="Churn iterations (churn pattern only, default: 120)")
    parser.add_argument("--chunk-mb", type=int, default=16, help="Chunk size MB (churn pattern only, default: 16)")
    parser.add_argument("--live-set-mb", type=int, default=256, help="Approx live-set MB (churn pattern only, default: 256)")
    parser.add_argument("--hold-ms", type=int, default=50, help="Sleep between samples/steps (default: 50ms)")

    parser.add_argument(
        "--min-reclaim-ratio",
        type=float,
        default=0.50,
        help="Minimum reclaim ratio vs delta from baseline (default: 0.50)",
    )
    parser.add_argument(
        "--absolute-min-reclaim-mb",
        type=float,
        default=None,
        help="Optional absolute minimum reclaimed MB (default: unset)",
    )

    parser.add_argument("--native-test", action="store_true", help="Also run native cleanup test (PyTorch if available)")
    parser.add_argument("--torch-tensor-mb", type=int, default=512, help="Tensor allocation size (MB) (default: 512)")
    parser.add_argument(
        "--torch-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Where to allocate torch tensor (default: auto)",
    )

    args = parser.parse_args()

    logger.info(f"Starting memory pressure simulation: pattern={args.pattern}, target={args.target_mb}MB")

    if args.pattern == "ramp":
        result = run_pattern_ramp(
            target_mb=args.target_mb,
            steps=args.steps,
            hold_ms=args.hold_ms,
        )
    else:
        result = run_pattern_churn(
            target_mb=args.target_mb,
            iterations=args.iterations,
            chunk_mb=args.chunk_mb,
            live_set_mb=args.live_set_mb,
            hold_ms=args.hold_ms,
        )

    enforce_reclaim_threshold(
        result=result,
        min_reclaim_ratio=args.min_reclaim_ratio,
        absolute_min_reclaim_mb=args.absolute_min_reclaim_mb,
    )

    test_native_cleanup(
        enable=args.native_test,
        torch_tensor_mb=args.torch_tensor_mb,
        torch_device=args.torch_device,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
