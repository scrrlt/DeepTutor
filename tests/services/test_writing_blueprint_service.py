"""Unit tests for writing blueprint algorithms and slop sanitization."""

from __future__ import annotations

import pytest
from deeptutor.services.writing.writing_blueprint_service import (
    analyse_historical_drift,
    calculate_prompt_retention_map,
    compute_inverse_coverage,
    extract_student_prose_segments,
    SlopPhraseSanitiser,
    StructuralEntropyExtrapolator,
    SyntacticTopologyEvaluator,
    perform_syllabus_gap_analysis,
)


@pytest.mark.asyncio
async def test_perform_syllabus_gap_analysis_orchestrates_correctly() -> None:
    from unittest.mock import AsyncMock, MagicMock, patch
    
    # Mock response
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[1.0, 0.0, 0.0])]
    
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    
    syllabus_chunks = [
        {"id": "s1", "module": "M1", "theory": "T1", "embedding": [0.0, 1.0, 0.0]}, # Sim 0.0
    ]
    
    with (
        patch("deeptutor.services.writing.provider_client.get_embedding_client", return_value=mock_client),
        patch("deeptutor.services.writing.provider_client.resolve_embedding_model", return_value="test-model"),
        patch("deeptutor.services.writing.provider_client.should_send_embedding_dimensions", return_value=False),
    ):
        gaps = await perform_syllabus_gap_analysis("Student text", syllabus_chunks, threshold=0.5)
        
    assert len(gaps) == 1
    assert gaps[0].associated_chunk_id == "s1"
    mock_client.embeddings.create.assert_awaited_once_with(model="test-model", input="Student text")


def test_extract_student_prose_segments_splits_correctly() -> None:
    text = (
        "This is a paragraph of student prose that is long enough to be counted.\n\n"
        "Feedback: This section was well written but needs more evidence.\n\n"
        "Another paragraph of prose that should be classified correctly."
    )
    prose, feedback = extract_student_prose_segments(text)
    
    assert len(prose) == 2
    assert len(feedback) == 1
    assert "student prose" in prose[0].text
    assert "Feedback:" in feedback[0].text


def test_analyse_historical_drift_computes_metrics() -> None:
    texts = [
        "The quick brown fox jumps over the lazy dog. It was an energetic fox.",
        "Another sentence here. This sentence is longer and contains more words for variance."
    ]
    profile = analyse_historical_drift(texts)
    
    assert profile.lexical_density_ttr > 0
    assert profile.mean_sentence_length > 0
    assert profile.sentence_length_variance >= 0


def test_calculate_prompt_retention_map_identifies_risk_zones() -> None:
    # (chunk_id, filename, token_count)
    chunks = [
        ("c1", "file1.txt", 100),
        ("c2", "file2.txt", 500),
        ("c3", "file3.txt", 100),
    ]
    # Total tokens = 100 + 100 + 500 + 100 = 800
    placements = calculate_prompt_retention_map(100, chunks)
    
    assert len(placements) == 3
    # c1 midpoint: 100 + 50 = 150. 150/800 = 0.1875 -> vulnerable_middle
    assert placements[0].chunk_id == "c1"
    assert placements[0].retention_risk_zone == "vulnerable_middle"


def test_compute_inverse_coverage_finds_gaps() -> None:
    outline_vector = [1.0, 0.0, 0.0]
    syllabus_chunks = [
        {"id": "s1", "module": "M1", "theory": "T1", "embedding": [1.0, 0.0, 0.0]}, # Sim 1.0
        {"id": "s2", "module": "M1", "theory": "T2", "embedding": [0.0, 1.0, 0.0]}, # Sim 0.0
    ]
    
    gaps = compute_inverse_coverage(outline_vector, syllabus_chunks, threshold=0.5)
    
    assert len(gaps) == 1
    assert gaps[0].associated_chunk_id == "s2"
    assert gaps[0].max_similarity_resolved == 0.0


def test_slop_phrase_sanitiser_removes_phrases() -> None:
    patterns = {
        "it is important to note": "",
        "delve into": "analyse",
    }
    sanitiser = SlopPhraseSanitiser(patterns)
    text = "It is important to note that we will delve into the details."
    sanitised = sanitiser.clear_slop(text)
    
    assert "important to note" not in sanitised
    assert "analyse" in sanitised
    assert "delve into" not in sanitised


def test_structural_entropy_extrapolator_verifies_cadence() -> None:
    # High entropy
    high_entropy = "Short sentence. This is a much longer sentence that provides more variance and detail to the paragraph. Another short one."
    # Low entropy
    low_entropy = "Short sentence. Another short. Small words."
    
    assert StructuralEntropyExtrapolator.verify_output_entropy(high_entropy) is True
    assert StructuralEntropyExtrapolator.verify_output_entropy(low_entropy) is False


def test_syntactic_topology_evaluator_calculates_metrics() -> None:
    segments = [
        "This is a sentence because it has a clause although it is simple.",
        "Whereas the second sentence is complex, the third is not."
    ]
    topology = SyntacticTopologyEvaluator.extract_topology(segments)
    
    assert topology.mean_dependency_depth > 1
    assert topology.branching_coefficient > 1
    assert topology.clause_count_variance >= 0