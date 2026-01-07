# tests/performance/oom_repro/simulate_pressure.py

import gc
import os
import psutil
import time
import sys
import argparse
import logging

logger = logging.getLogger("SimulatePressure")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_rss_mb():
    """Returns current Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def allocate_mb(size_mb):
    """Allocates a block of memory of approximately size_mb MB."""
    return bytearray(size_mb * 1024 * 1024)

def test_native_cleanup(use_torch=False):
    """Tests whether native memory (e.g., PyTorch) is properly released after cleanup."""
    logger.info("--- Testing Native Cleanup ---")
    if not use_torch:
        logger.info("Torch test skipped.")
        return

    try:
        import torch
        logger.info("Torch detected. Allocating tensor...")
        t = torch.ones((125, 1024, 1024), dtype=torch.float32)  # ~500MB
        logger.info(f"Allocated tensor. RSS: {get_rss_mb():.2f} MB")
        del t

        logger.info("Running cleanup hooks...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info(f"Post-cleanup RSS: {get_rss_mb():.2f} MB")
    except ImportError:
        logger.warning("Torch not found. Skipping native tensor test.")
    except Exception as e:
        logger.error(f"Error during native cleanup test: {e}")

def main(target_mb):
    logger.info(f"Starting memory pressure simulation. Target: {target_mb} MB")
    baseline = get_rss_mb()
    logger.info(f"Baseline RSS: {baseline:.2f} MB")

    bloat = []
    try:
        chunk_size = max(1, int(target_mb * 0.2))  # 4 chunks of 20% each
        for i in range(4):
            bloat.append(allocate_mb(chunk_size))
            logger.info(f"Allocated {chunk_size}MB. Current RSS: {get_rss_mb():.2f} MB")
            time.sleep(0.1)
    except MemoryError:
        logger.error("OOM hit during allocation!")

    peak = get_rss_mb()
    logger.info(f"Peak RSS: {peak:.2f} MB")

    del bloat
    gc.collect()
    final = get_rss_mb()
    reclaimed = peak - final
    logger.info(f"Final RSS: {final:.2f} MB. Reclaimed: {reclaimed:.2f} MB")

    if reclaimed < (target_mb * 0.5):
        logger.error("FAIL: Less than 50% memory reclaimed. Potential native leak.")
        sys.exit(1)
    else:
        logger.info("PASS: Significant memory reclaimed.")

    test_native_cleanup(use_torch=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate memory pressure and validate cleanup.")
    parser.add_argument("--target-mb", type=int, default=1000, help="Target memory pressure in MB (default: 1000)")
    args = parser.parse_args()
    main(args.target_mb)