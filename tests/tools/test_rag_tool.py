# tests/tools/test_rag_tool.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.tools.rag_tool import rag_search

@pytest.mark.asyncio
async def test_rag_search_success():
    """Test successful retrieval from RAG engine."""
    
    # Mock the LightRAG wrapper or internal service
    # Adjust path to where 'query_rag' or the actual engine is imported in rag_tool.py
    with patch("src.tools.rag_tool.get_rag_engine") as mock_engine_getter:
        mock_engine = AsyncMock()
        mock_engine.query.return_value = "The answer is 42."
        mock_engine_getter.return_value = mock_engine
        
        # Test standard call
        result = await rag_search("What is the meaning?", kb_name="hitchhiker")
        
        assert result["status"] == "success"
        assert result["answer"] == "The answer is 42."
        # Verify engine was queried with correct params
        mock_engine.query.assert_called_with("What is the meaning?", param_mode="hybrid")

@pytest.mark.asyncio
async def test_rag_search_missing_kb():
    """Test error handling when KB is missing."""
    
    # Use a non-existent KB name
    # Assuming the tool checks for existence or catches the error
    with patch("src.tools.rag_tool.get_rag_engine", side_effect=ValueError("KB not found")):
        
        result = await rag_search("Query", kb_name="ghost_kb")
        
        assert result["status"] == "error"
        assert "KB not found" in result.get("message", "")

@pytest.mark.asyncio
async def test_rag_search_empty_query():
    """Test validation for empty queries."""
    result = await rag_search("", kb_name="test")
    # Should probably return error or empty result, depending on implementation
    # Adjust assertion based on your actual validation logic
    assert result["status"] == "error" or result["answer"] == ""