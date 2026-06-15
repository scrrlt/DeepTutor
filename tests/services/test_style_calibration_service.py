"""Unit tests for binary style calibration engine."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from deeptutor.services.writing.style_calibration_service import (
    DEFAULT_CONTRAST_PAIRS,
    StyleCalibrationEngine,
)


def test_style_calibration_engine_lists_pairs() -> None:
    engine = StyleCalibrationEngine(DEFAULT_CONTRAST_PAIRS)
    pairs = engine.list_pairs()
    assert len(pairs) == len(DEFAULT_CONTRAST_PAIRS)
    assert pairs[0].pair_id == "sv_01"


def test_calculate_initial_profile_with_selections() -> None:
    engine = StyleCalibrationEngine(DEFAULT_CONTRAST_PAIRS)
    selections = [
        {"pair_id": "sv_01", "choice": "cadence_b"}, # weight 13.4
        {"pair_id": "pr_01", "choice": "cadence_a"}, # weight 0.31
        {"pair_id": "ld_01", "choice": "cadence_b"}, # weight 0.63
    ]
    
    profile = engine.calculate_initial_profile(selections)
    
    assert profile["sentence_length_variance"] == 13.4
    assert profile["passive_voice_ratio"] == 0.31
    assert profile["lexical_density_ttr"] == 0.63


def test_profile_to_feature_map_converts_correctly() -> None:
    weights = {
        "sentence_length_variance": 10.0,
        "passive_voice_ratio": 0.2,
        "lexical_density_ttr": 0.5,
    }
    feature_map = StyleCalibrationEngine.profile_to_feature_map(weights)
    
    assert feature_map["sentence_length_variance"] == 10.0
    # CV = 10 / 20 = 0.5
    assert feature_map["sentence_length_cv"] == 0.5
    assert feature_map["passive_voice_ratio"] == 0.2
    assert feature_map["lexical_density_ttr"] == 0.5
    assert feature_map["calibrated"] == 1.0


@pytest.mark.asyncio
async def test_commit_calibrated_profile(db_session: AsyncSession) -> None:
    engine = StyleCalibrationEngine(DEFAULT_CONTRAST_PAIRS)
    user_id = "calib-user"
    weights = {
        "sentence_length_variance": 8.0,
        "passive_voice_ratio": 0.15,
        "lexical_density_ttr": 0.6,
    }
    
    # Commit new
    row = await engine.commit_calibrated_profile(db_session, user_id=user_id, profile_weights=weights)
    assert row.user_id == user_id
    assert row.last_grade == "calibrated"

    # Update existing
    new_weights = {
        "sentence_length_variance": 12.0,
        "passive_voice_ratio": 0.25,
        "lexical_density_ttr": 0.4,
    }
    updated = await engine.commit_calibrated_profile(db_session, user_id=user_id, profile_weights=new_weights)
    assert updated.sample_count == 1
    assert "0.4" in updated.profile_json
