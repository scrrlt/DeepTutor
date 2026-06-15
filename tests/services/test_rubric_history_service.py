"""Unit tests for rubric assessment history and analytics."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from deeptutor.services.writing.rubric_history_service import (
    create_rubric_assessment,
    list_rubric_assessments,
    rubric_dashboard_summary,
    RubricCriterion,
)


@pytest.mark.asyncio
async def test_create_and_list_rubric_assessments(db_session: AsyncSession) -> None:
    user_id = "test-rubric-user"
    criteria = [
        RubricCriterion(key="C1", score=4.0, label="Excellent"),
        RubricCriterion(key="C2", score=3.0, label="Good"),
    ]
    
    # Create
    record = await create_rubric_assessment(
        db_session,
        user_id=user_id,
        unit_id="U1",
        topic="T1",
        grade_level="G1",
        cohort="C1",
        template_id="TPL1",
        comparison_group_id=None,
        is_improved_version=False,
        weighted_total=85.0,
        raw_total=7,
        raw_max=10,
        grade_descriptor="Distinction",
        criteria=criteria,
        improvement_suggestions=["Add more data"],
        assessor_notes="Well done",
    )
    
    assert record.id is not None
    assert record.weighted_total == 85.0
    assert len(record.criteria) == 2
    assert record.criteria[0].key == "C1"

    # List
    total, items = await list_rubric_assessments(db_session, user_id=user_id)
    assert total == 1
    assert items[0].id == record.id


@pytest.mark.asyncio
async def test_rubric_dashboard_summary(db_session: AsyncSession) -> None:
    user_id = "dash-user"
    criteria = [RubricCriterion(key="crit", score=5.0)]
    
    await create_rubric_assessment(
        db_session,
        user_id=user_id,
        template_id="TPL",
        is_improved_version=False,
        weighted_total=90.0,
        raw_total=5,
        raw_max=5,
        grade_descriptor="HD",
        criteria=criteria,
        improvement_suggestions=[],
        assessor_notes="",
    )
    await create_rubric_assessment(
        db_session,
        user_id=user_id,
        template_id="TPL",
        is_improved_version=False,
        weighted_total=70.0,
        raw_total=3,
        raw_max=5,
        grade_descriptor="D",
        criteria=criteria,
        improvement_suggestions=[],
        assessor_notes="",
    )
    
    summary = await rubric_dashboard_summary(db_session, unit_id=None, cohort=None)
    
    assert summary.total_assessments == 2
    assert summary.average_weighted_score == 80.0  # (90 + 70) / 2
    assert summary.grade_distribution["HD"] == 1
    assert summary.grade_distribution["D"] == 1
    assert summary.criterion_average_scores["crit"] == 5.0
