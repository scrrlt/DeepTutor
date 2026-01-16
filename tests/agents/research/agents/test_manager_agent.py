"""
Tests for the research's ManagerAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.research.agents.manager_agent import ManagerAgent
from src.agents.research.data_structures import DynamicTopicQueue, TopicBlock, TopicStatus


@pytest.fixture
def manager_agent():
    """
    Provides a ManagerAgent instance with a mocked queue.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        agent = ManagerAgent(config={})
        agent.set_queue(DynamicTopicQueue(research_id="test"))
        yield agent


def test_get_next_task(manager_agent: ManagerAgent):
    """
    Tests that the get_next_task method correctly retrieves the next task.
    """
    manager_agent.queue.add_block("t1", "o1")
    task = manager_agent.get_next_task()
    
    assert task is not None
    assert task.status == TopicStatus.RESEARCHING


def test_complete_task(manager_agent: ManagerAgent):
    """
    Tests that the complete_task method correctly marks a task as completed.
    """
    block = manager_agent.queue.add_block("t1", "o1")
    manager_agent.complete_task(block.block_id)
    
    assert manager_agent.queue.get_block_by_id(block.block_id).status == TopicStatus.COMPLETED


def test_fail_task(manager_agent: ManagerAgent):
    """
    Tests that the fail_task method correctly marks a task as failed.
    """
    block = manager_agent.queue.add_block("t1", "o1")
    manager_agent.fail_task(block.block_id)
    
    assert manager_agent.queue.get_block_by_id(block.block_id).status == TopicStatus.FAILED


def test_add_new_topic(manager_agent: ManagerAgent):
    """
    Tests that the add_new_topic method correctly adds a new topic.
    """
    manager_agent.add_new_topic("t1", "o1")
    assert manager_agent.queue.has_topic("t1")

def test_is_research_complete(manager_agent: ManagerAgent):
    """
    Tests that the is_research_complete method correctly checks if the research is complete.
    """
    assert manager_agent.is_research_complete() is True
    
    block = manager_agent.queue.add_block("t1", "o1")
    assert manager_agent.is_research_complete() is False
    
    manager_agent.complete_task(block.block_id)
    assert manager_agent.is_research_complete() is True


@pytest.mark.asyncio
async def test_async_methods(manager_agent: ManagerAgent):
    """
    Tests that the async methods work correctly.
    """
    manager_agent.queue.add_block("t1", "o1")
    task = await manager_agent.get_next_task_async()
    assert task is not None
    
    await manager_agent.complete_task_async(task.block_id)
    assert manager_agent.queue.get_block_by_id(task.block_id).status == TopicStatus.COMPLETED

