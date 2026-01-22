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

    async def communicate(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_resource_limits_passed_to_subprocess(tmp_path, monkeypatch):
    ws = WorkspaceManager(project_root=tmp_path)
    ws.initialize()

    captured = {}

    async def fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeProcess(stdout=b"ok\n", stderr=b"", returncode=0, delay=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    # Ensure resource module appears to be available
    import types

    fake_resource = types.SimpleNamespace()

    def fake_setrlimit(a, b):
        return None

    fake_resource.setrlimit = fake_setrlimit

    import sys as _sys

    _sys.modules["resource"] = fake_resource

    runner = LocalProcessRunner(
        workspace_manager=ws, python_executable=sys.executable
    )

    options = ExecutionOptions(
        timeout=5, assets_dir=None, cpu_time_limit=1, memory_limit_bytes=1024
    )
    result = await runner.run("print('hello')", options)

    # preexec_fn should be passed through kwargs
    assert "preexec_fn" in captured["kwargs"]


@pytest.mark.asyncio
async def test_resource_limits_not_supported_on_windows(tmp_path, monkeypatch):
    ws = WorkspaceManager(project_root=tmp_path)
    ws.initialize()

    captured = {}

    async def fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeProcess(stdout=b"ok\n", stderr=b"", returncode=0, delay=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    # Simulate no resource module
    import sys as _sys

    if "resource" in _sys.modules:
        del _sys.modules["resource"]

    runner = LocalProcessRunner(
        workspace_manager=ws, python_executable=sys.executable
    )

    options = ExecutionOptions(
        timeout=5, assets_dir=None, cpu_time_limit=1, memory_limit_bytes=1024
    )
    result = await runner.run("print('hello')", options)

    # preexec_fn should be present but will be None because we warn and ignore
    assert "preexec_fn" in captured["kwargs"]
    assert captured["kwargs"]["preexec_fn"] is None
