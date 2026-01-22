"""Fuzz tests for code executor deadlock and resource exhaustion scenarios.

These tests verify that the code executor is resilient to high-volume output,
infinite loops, and memory pressure without blocking the event loop or
deadlocking on pipe buffers.
"""

import pytest

from src.tools.code_executor import run_code


@pytest.mark.asyncio
async def test_executor_large_stdout_no_deadlock():
    """Test that large stdout does not deadlock the executor.

    A deadlock occurs when a process writes more than the
    OS pipe buffer size (~64KB) to stdout/stderr without the parent reading.
    This test ensures we consume streams concurrently to prevent blocking.
    """
    # Write 10MB to both stdout and stderr
    code = r"""
import sys
data = 'x' * (10 * 1024 * 1024)  # 10MB
sys.stdout.write(data)
sys.stderr.write(data)
print("done")
"""
    result = await run_code("python", code, timeout=30)

    # Should succeed without deadlock
    assert result["exit_code"] == 0
    # Verify we captured substantial output
    assert len(result["stdout"]) >= 10 * 1024 * 1024
    assert len(result["stderr"]) >= 10 * 1024 * 1024


@pytest.mark.asyncio
async def test_executor_timeout_kills_runaway():
    """Test that timeout properly kills runaway processes."""
    code = r"""
import time
while True:
    time.sleep(0.1)
"""
    result = await run_code("python", code, timeout=2)

    # Process should be killed
    assert result["exit_code"] == -1
    assert "timeout" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_executor_mixed_output_streams():
    """Test interleaved stdout and stderr without deadlock."""
    code = r"""
import sys
for i in range(100):
    sys.stdout.write(f"out{i}\n")
    sys.stderr.write(f"err{i}\n")
"""
    result = await run_code("python", code, timeout=10)

    assert result["exit_code"] == 0
    assert "out0" in result["stdout"]
    assert "out99" in result["stdout"]
    assert "err0" in result["stderr"]
    assert "err99" in result["stderr"]
