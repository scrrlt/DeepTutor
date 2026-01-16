#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the code_executor tool.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.tools.code_executor import run_code, ImportGuard, WorkspaceManager, CodeExecutionError


@pytest.mark.asyncio
async def test_run_code_success():
    """
    Tests that the run_code function can execute Python code and return the correct output.
    """
    code = "print('hello world')"
    result = await run_code("python", code)

    assert result["stdout"] == "hello world\n"
    assert result["stderr"] == ""
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_run_code_error():
    """
    Tests that the run_code function correctly handles exceptions raised by the executed code.
    """
    code = "raise ValueError('test error')"
    result = await run_code("python", code)

    assert result["stdout"] == ""
    assert "ValueError: test error" in result["stderr"]
    assert result["exit_code"] != 0


@pytest.mark.asyncio
async def test_run_code_timeout():
    """
    Tests that the run_code function correctly handles timeouts.
    """
    code = "import time; time.sleep(2)"
    result = await run_code("python", code, timeout=1)

    assert result["stdout"] == ""
    assert "Code execution timeout" in result["stderr"]
    assert result["exit_code"] == -1

def test_import_guard_success():
    """
    Tests that the ImportGuard correctly allows allowed modules.
    """
    code = "import os; import sys"
    ImportGuard.validate(code, allowed_imports=["os", "sys"])


def test_import_guard_error():
    """
    Tests that the ImportGuard correctly restricts not allowed modules.
    """
    code = "import requests"
    with pytest.raises(CodeExecutionError, match="requests"):
        ImportGuard.validate(code, allowed_imports=["os", "sys"])

def test_workspace_manager():
    """
    Tests that the WorkspaceManager correctly creates and manages the workspace.
    """
    with patch('src.tools.code_executor._load_config', return_value={}):
        manager = WorkspaceManager()
        assert manager.base_dir is not None
        
        with manager.create_temp_dir() as temp_dir:
            assert temp_dir.exists()
            assert str(manager.base_dir) in str(temp_dir)
