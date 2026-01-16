#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the SolveMemory.
"""

import pytest
import json
from pathlib import Path

from src.agents.solve.memory.solve_memory import SolveMemory, SolveChainStep, ToolCallRecord


@pytest.fixture
def solve_memory(tmp_path):
    """
    Provides a SolveMemory instance with a temporary output directory.
    """
    memory = SolveMemory(output_dir=str(tmp_path))
    yield memory


def test_solve_memory_initialization(solve_memory: SolveMemory):
    """
    Tests that the SolveMemory can be initialized correctly.
    """
    assert solve_memory.output_dir is not None
    assert solve_memory.solve_chains == []


def test_create_chains(solve_memory: SolveMemory):
    """
    Tests that the create_chains method correctly creates the solve chains.
    """
    chains = [SolveChainStep(step_id="step1", step_target="target1")]
    solve_memory.create_chains(chains)
    assert len(solve_memory.solve_chains) == 1
    assert solve_memory.metadata["total_steps"] == 1


def test_get_steps(solve_memory: SolveMemory):
    """
    Tests that the get_step and get_current_step methods correctly retrieve the steps.
    """
    chains = [
        SolveChainStep(step_id="step1", step_target="target1", status="done"),
        SolveChainStep(step_id="step2", step_target="target2", status="in_progress"),
    ]
    solve_memory.create_chains(chains)

    assert solve_memory.get_step("step1") is not None
    assert solve_memory.get_current_step().step_id == "step2"


def test_tool_calls(solve_memory: SolveMemory):
    """
    Tests that the tool call methods correctly manage the tool calls.
    """
    chains = [SolveChainStep(step_id="step1", step_target="target1")]
    solve_memory.create_chains(chains)

    record = solve_memory.append_tool_call("step1", "test_tool", "test_query")
    assert len(solve_memory.solve_chains[0].tool_calls) == 1

    solve_memory.update_tool_call_result(record.step_id, record.call_id, "raw", "summary")
    assert solve_memory.solve_chains[0].tool_calls[0].status == "success"


def test_step_status_updates(solve_memory: SolveMemory):
    """
    Tests that the step status update methods correctly update the status of the steps.
    """
    chains = [SolveChainStep(step_id="step1", step_target="target1")]
    solve_memory.create_chains(chains)

    solve_memory.mark_step_waiting_response("step1")
    assert solve_memory.get_step("step1").status == "waiting_response"

    solve_memory.submit_step_response("step1", "response")
    assert solve_memory.get_step("step1").status == "done"
    assert solve_memory.metadata["completed_steps"] == 1


def test_save_and_load(solve_memory: SolveMemory, tmp_path):
    """
    Tests that the save and load_or_create methods correctly save and load the memory.
    """
    chains = [SolveChainStep(step_id="step1", step_target="target1")]
    solve_memory.create_chains(chains)
    solve_memory.save()

    new_memory = SolveMemory.load_or_create(str(tmp_path))
    assert len(new_memory.solve_chains) == 1
    assert new_memory.solve_chains[0].step_id == "step1"


def test_load_legacy(tmp_path):
    """
    Tests that load_or_create correctly handles legacy memory files.
    """
    legacy_data = {
        "steps": [{"step_id": "s1", "plan": "p1", "content": "c1", "status": "completed"}]
    }
    legacy_path = tmp_path / "solve_memory.json"
    with open(legacy_path, "w") as f:
        json.dump(legacy_data, f)
        
    memory = SolveMemory.load_or_create(str(tmp_path))
    assert len(memory.solve_chains) == 1
    assert memory.solve_chains[0].step_id == "s1"
    assert memory.solve_chains[0].status == "done"
