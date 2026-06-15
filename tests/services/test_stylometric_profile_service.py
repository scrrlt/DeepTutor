"""Unit tests for stylometric feature computation and profile persistence."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.services.writing.stylometric_profile_service import (
    compute_stylometric_features,
    extract_submission_metadata,
    is_high_score_submission,
    stylometric_distance,
    upsert_user_stylometric_profile,
    get_user_stylometric_profile,
)


def test_compute_stylometric_features_with_basic_text() -> None:
    text = "The quick brown fox jumps over the lazy dog. It was a very energetic fox."
    features = compute_stylometric_features(text)

    assert "token_diversity" in features
    assert "sentence_length_cv" in features
    assert features["token_diversity"] > 0
    # Basic check for deterministic output
    assert isinstance(features["token_diversity"], float)


def test_compute_stylometric_features_with_empty_text() -> None:
    features = compute_stylometric_features("")
    assert features["token_diversity"] == 0.0
    assert features["yule_i"] == 0.0


def test_extract_submission_metadata_finds_score_and_grade() -> None:
    text = "Overall Score: 85. Grade: High Distinction. Great work!"
    score, grade = extract_submission_metadata(text)
    assert score == 85.0
    assert grade == "High Distinction"


def test_extract_submission_metadata_handles_missing_markers() -> None:
    text = "No metrics here."
    score, grade = extract_submission_metadata(text)
    assert score is None
    assert grade is None


def test_is_high_score_submission_logic() -> None:
    assert is_high_score_submission(75.0, None) is True
    assert is_high_score_submission(74.0, "Distinction") is True
    assert is_high_score_submission(50.0, "Pass") is False
    assert is_high_score_submission(None, "High Distinction") is True


def test_stylometric_distance_computation() -> None:
    f1 = {"a": 1.0, "b": 2.0}
    f2 = {"a": 1.0, "b": 3.0}
    distance = stylometric_distance(f1, f2)
    assert distance == 1.0  # sqrt((1-1)^2 + (2-3)^2)


@pytest.mark.asyncio
async def test_upsert_and_get_profile(db_session: AsyncSession) -> None:
    user_id = "test-stylometry-user"
    features = {"token_diversity": 0.5, "sentence_length_cv": 0.2}
    exemplars = ["Sample sentence."]

    # First insert
    await upsert_user_stylometric_profile(
        db_session,
        user_id=user_id,
        features=features,
        exemplars=exemplars,
        score=80.0,
        grade="Distinction",
    )
    await db_session.commit()

    profile = await get_user_stylometric_profile(db_session, user_id)
    assert profile is not None
    assert profile.user_id == user_id
    assert profile.features["token_diversity"] == 0.5
    assert profile.sample_count == 1
    assert profile.last_score == 80.0

    # Second insert (rolling average)
    new_features = {"token_diversity": 0.6, "sentence_length_cv": 0.4}
    await upsert_user_stylometric_profile(
        db_session,
        user_id=user_id,
        features=new_features,
        exemplars=["Another sample."],
        score=90.0,
        grade="High Distinction",
    )
    await db_session.commit()

    updated = await get_user_stylometric_profile(db_session, user_id)
    assert updated.sample_count == 2
    # (0.5 + 0.6) / 2 = 0.55
    assert updated.features["token_diversity"] == 0.55
    assert len(updated.exemplars) == 2
    assert updated.last_score == 90.0
