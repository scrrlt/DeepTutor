import pytest

from src.agents.solve.memory.citation_memory import CitationMemory


@pytest.fixture
def citation_memory(tmp_path) -> CitationMemory:
    """Provides a CitationMemory instance with a temporary output directory.

    Args:
        tmp_path (Path): Pytest fixture for a temporary directory.

    Yields:
        CitationMemory: An instance of the CitationMemory.
    """
    memory = CitationMemory(output_dir=str(tmp_path))
    yield memory


def test_citation_memory_initialization(citation_memory: CitationMemory):
    """
    Tests that the CitationMemory can be initialized correctly.
    """
    assert citation_memory.output_dir is not None
    assert citation_memory.citations == []


def test_add_citation(citation_memory: CitationMemory):
    """
    Tests that the add_citation method correctly adds a new citation.
    """
    cite_id = citation_memory.add_citation("test_tool", "test_query")
    assert len(citation_memory.citations) == 1
    assert citation_memory.get_citation(cite_id) is not None


def test_get_citation(citation_memory: CitationMemory):
    """
    Tests that the get_citation method correctly retrieves a citation.
    """
    cite_id = citation_memory.add_citation("test_tool", "test_query")
    citation = citation_memory.get_citation(cite_id)
    assert citation is not None
    assert citation.cite_id == cite_id


def test_update_citation(citation_memory: CitationMemory):
    """
    Tests that the update_citation method correctly updates a citation.
    """
    cite_id = citation_memory.add_citation("test_tool", "test_query")
    citation_memory.update_citation(cite_id, content="new_content")
    citation = citation_memory.get_citation(cite_id)
    assert citation.content == "new_content"


def test_save_and_load(citation_memory: CitationMemory, tmp_path):
    """
    Tests that the save and load_or_create methods correctly save and load the memory.
    """
    citation_memory.add_citation("test_tool", "test_query")
    citation_memory.save()

    new_memory = CitationMemory.load_or_create(str(tmp_path))
    assert len(new_memory.citations) == 1
    assert new_memory.citations[0].tool_type == "test_tool"


def test_format_citations_markdown(citation_memory: CitationMemory):
    """
    Tests that the format_citations_markdown method correctly formats the citations.
    """
    cite_id = citation_memory.add_citation(
        "test_tool", "test_query", content="test_content"
    )
    markdown = citation_memory.format_citations_markdown(used_cite_ids=[cite_id])

    assert "## Citations" in markdown
    assert cite_id in markdown
    assert "test_content" in markdown
