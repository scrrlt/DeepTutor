from unittest.mock import patch

import pytest

from src.tools.code_executor import run_code
from src.tools.code_runner.types import RunResult


@pytest.mark.asyncio
async def test_run_code_delegates(monkeypatch):
    # Patch LocalProcessRunner to avoid actually running anything
    fake_result = RunResult(
        stdout="ok\n",
        stderr="",
        artifacts=[],
        artifact_paths=[],
        exit_code=0,
        elapsed_ms=12.3,
    )

    async def fake_run(self, code, options):
        return fake_result

    with patch("src.tools.code_runner.runner.LocalProcessRunner.run", new=fake_run):
        result = await run_code("python", "print('hi')", timeout=2, assets_dir=None)

    assert result["stdout"] == "ok\n"
    assert result["exit_code"] == 0
    assert "elapsed_ms" in result
