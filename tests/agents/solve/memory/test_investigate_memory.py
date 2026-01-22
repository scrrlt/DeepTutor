#!/usr/bin/env python

"""
Tests for the InvestigateMemory.
"""

import json

import pytest

from src.agents.solve.memory.investigate_memory import (
    InvestigateMemory,
    KnowledgeItem,
)


@pytest.fixture
def investigate_memory(tmp_path):
    """
    Provides an InvestigateMemory instance with a temporary output directory.
    """
    memory = InvestigateMemory(output_dir=str(tmp_path))
    yield memory


def test_investigate_memory_initialization(
    investigate_memory: InvestigateMemory,
):
    """
    Tests that the InvestigateMemory can be initialized correctly.
    """
    assert investigate_memory.output_dir is not None
    assert investigate_memory.knowledge_chain == []
    assert investigate_memory.reflections.remaining_questions == []


def test_add_knowledge(investigate_memory: InvestigateMemory):
    """
    Tests that the add_knowledge method correctly adds a new knowledge item.
    """
    item = KnowledgeItem(
        cite_id="cite1",
        tool_type="test_tool",
        query="test_query",
        raw_result="test_result",
    )
    investigate_memory.add_knowledge(item)
    assert len(investigate_memory.knowledge_chain) == 1


def test_update_knowledge_summary(investigate_memory: InvestigateMemory):
    """
    Tests that the update_knowledge_summary method correctly updates the summary of a knowledge item.
    """
    item = KnowledgeItem(
        cite_id="cite1",
        tool_type="test_tool",
        query="test_query",
        raw_result="test_result",
    )
    investigate_memory.add_knowledge(item)
    investigate_memory.update_knowledge_summary("cite1", "new_summary")
    assert investigate_memory.knowledge_chain[0].summary == "new_summary"


def test_get_available_knowledge(investigate_memory: InvestigateMemory):
    """
    Tests that the get_available_knowledge method correctly retrieves the knowledge items.
    """
    item1 = KnowledgeItem(cite_id="cite1", tool_type="tool1", query="q1", raw_result="r1")
    item2 = KnowledgeItem(cite_id="cite2", tool_type="tool2", query="q2", raw_result="r2")
    investigate_memory.add_knowledge(item1)
    investigate_memory.add_knowledge(item2)

    assert len(investigate_memory.get_available_knowledge(tool_types=["tool1"])) == 1
    assert len(investigate_memory.get_available_knowledge(cite_ids=["cite2"])) == 1


def test_save_and_load(investigate_memory: InvestigateMemory, tmp_path):
    """
    Tests that the save and load_or_create methods correctly save and load the memory.
    """
    item = KnowledgeItem(
        cite_id="cite1",
        tool_type="test_tool",
        query="test_query",
        raw_result="test_result",
    )
    investigate_memory.add_knowledge(item)
    investigate_memory.save()

    new_memory = InvestigateMemory.load_or_create(str(tmp_path))
    assert len(new_memory.knowledge_chain) == 1
    assert new_memory.knowledge_chain[0].cite_id == "cite1"


def test_load_backward_compatibility(tmp_path):
    """
    Tests that the load_or_create method correctly handles backward compatibility.
    """
    v1_data = {
        "version": "1.0",
        "knowledge_chain": [
            {
                "knowledge_id": "kid1",
                "source_type": "st1",
                "query_text": "q1",
                "answer_raw": "a1",
            }
        ],
        "notes": [{"related_knowledge_ids": ["kid1"], "summary": "note_summary"}],
        "reflections": [{"action_items": ["ref1"]}],
    }

    file_path = tmp_path / "investigate_memory.json"
    with open(file_path, "w") as f:
        json.dump(v1_data, f)

    memory = InvestigateMemory.load_or_create(str(tmp_path))

    assert memory.version == "3.0"
    assert len(memory.knowledge_chain) == 1
    assert memory.knowledge_chain[0].cite_id == "kid1"
    assert memory.knowledge_chain[0].summary == "note_summary"
    assert memory.reflections.remaining_questions == ["ref1"]
