import asyncio
import sys

import pytest

from src.tools.code_runner.runner import LocalProcessRunner
from src.tools.code_runner.types import ExecutionOptions
from src.tools.code_runner.workspace import WorkspaceManager


class FakeProcess:
    def __init__(self, stdout=b"ok\n", stderr=b"", returncode=0, delay=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._delay = delay
        self.killed = False

    async def communicate(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_local_process_runner_success(tmp_path, monkeypatch):
    ws = WorkspaceManager(project_root=tmp_path)
    ws.initialize()

    async def fake_create(*args, **kwargs):
        return FakeProcess(stdout=b"hello\n", stderr=b"", returncode=0, delay=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    runner = LocalProcessRunner(workspace_manager=ws, python_executable=sys.executable)

    options = ExecutionOptions(timeout=5, assets_dir=None)
    result = await runner.run("print('hello')", options)

    assert "hello" in result.stdout
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_local_process_runner_artifacts(tmp_path, monkeypatch):
    ws = WorkspaceManager(project_root=tmp_path)
    ws.initialize()

    # Create an assets dir and add a fake artifact
    assets = tmp_path / "artifacts"
    assets.mkdir()
    (assets / "out.txt").write_text("data")

    async def fake_create(*args, **kwargs):
        return FakeProcess(stdout=b"ok\n", stderr=b"", returncode=0, delay=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    runner = LocalProcessRunner(workspace_manager=ws, python_executable=sys.executable)

    options = ExecutionOptions(timeout=5, assets_dir=assets)
    result = await runner.run("print('hello')", options)

    assert result.exit_code == 0
    assert any("out.txt" in p or p.endswith("out.txt") for p in result.artifact_paths)
    assert any(a.name == "out.txt" for a in result.artifacts)
    # Size should match
    out = next(a for a in result.artifacts if a.name == "out.txt")
    assert out.size == 4


@pytest.mark.asyncio
async def test_local_process_runner_timeout(tmp_path, monkeypatch):
    ws = WorkspaceManager(project_root=tmp_path)
    ws.initialize()

    # Process that takes longer than timeout
    async def fake_create(*args, **kwargs):
        return FakeProcess(stdout=b"late\n", stderr=b"", returncode=0, delay=2)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    runner = LocalProcessRunner(workspace_manager=ws, python_executable=sys.executable)

    options = ExecutionOptions(timeout=0.1, assets_dir=None)
    result = await runner.run("import time\ntime.sleep(1)", options)

    assert result.exit_code == -1
    assert "timeout" in result.stderr.lower()


@pytest.mark.asyncio
async def test_local_process_runner_exception(tmp_path, monkeypatch):
    ws = WorkspaceManager(project_root=tmp_path)
    ws.initialize()

    async def fake_create(*args, **kwargs):
        raise OSError("failed to start")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    runner = LocalProcessRunner(workspace_manager=ws, python_executable=sys.executable)

    options = ExecutionOptions(timeout=5, assets_dir=None)
    result = await runner.run("print('hello')", options)

    assert result.exit_code == -1
    assert "failed to start" in result.stderr.lower()
