"""Persistence and analytics helpers for rubric assessment history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import median

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.models.rubric_assessment import RubricAssessment


@dataclass(frozen=True)
class RubricAssessmentRecord:
    """Typed projection for one rubric assessment row."""

    id: int
    user_id: str
    unit_id: str | None
    topic: str | None
    grade_level: str | None
    cohort: str | None
    template_id: str
    comparison_group_id: str | None
    is_improved_version: bool
    weighted_total: float
    raw_total: int
    raw_max: int
    grade_descriptor: str
    criteria: list[dict[str, object]]
    improvement_suggestions: list[str]
    assessor_notes: str
    created_at_iso: str


@dataclass(frozen=True)
class RubricDashboardSummary:
    """Aggregate metrics for rubric dashboard views."""

    total_assessments: int
    average_weighted_score: float
    median_weighted_score: float
    grade_distribution: dict[str, int]
    criterion_average_scores: dict[str, float]
    raw_scores: list[float]


@dataclass(frozen=True)
class PeerComparisonSummary:
    """Anonymized cohort comparison result for one assessment."""

    assessment_id: int
    cohort_size: int
    user_score: float
    cohort_average_score: float
    cohort_median_score: float
    percentile: float
    min_score: float
    max_score: float


def _to_record(row: RubricAssessment) -> RubricAssessmentRecord:
    """Convert ORM row to strongly typed record."""
    criteria_obj = json.loads(row.criteria_json)
    suggestions_obj = json.loads(row.improvement_suggestions_json)

    criteria: list[dict[str, object]] = []
    if isinstance(criteria_obj, list):
        for item in criteria_obj:
            if isinstance(item, dict):
                criteria.append(item)

    improvement_suggestions: list[str] = []
    if isinstance(suggestions_obj, list):
        for item in suggestions_obj:
            if isinstance(item, str):
                improvement_suggestions.append(item)

    return RubricAssessmentRecord(
        id=row.id,
        user_id=row.user_id,
        unit_id=row.unit_id,
        topic=row.topic,
        grade_level=row.grade_level,
        cohort=row.cohort,
        template_id=row.template_id,
        comparison_group_id=row.comparison_group_id,
        is_improved_version=row.is_improved_version,
        weighted_total=row.weighted_total,
        raw_total=row.raw_total,
        raw_max=row.raw_max,
        grade_descriptor=row.grade_descriptor,
        criteria=criteria,
        improvement_suggestions=improvement_suggestions,
        assessor_notes=row.assessor_notes,
        created_at_iso=row.created_at.isoformat(),
    )


async def create_rubric_assessment(
    db: AsyncSession,
    *,
    user_id: str,
    unit_id: str | None,
    topic: str | None,
    grade_level: str | None,
    cohort: str | None,
    template_id: str,
    comparison_group_id: str | None,
    is_improved_version: bool,
    weighted_total: float,
    raw_total: int,
    raw_max: int,
    grade_descriptor: str,
    criteria: list[dict[str, object]],
    improvement_suggestions: list[str],
    assessor_notes: str,
    auto_commit: bool = True,
) -> RubricAssessmentRecord:
    """Persist one rubric assessment entry and return typed projection."""
    row = RubricAssessment(
        user_id=user_id,
        unit_id=unit_id,
        topic=topic,
        grade_level=grade_level,
        cohort=cohort,
        template_id=template_id,
        comparison_group_id=comparison_group_id,
        is_improved_version=is_improved_version,
        weighted_total=weighted_total,
        raw_total=raw_total,
        raw_max=raw_max,
        grade_descriptor=grade_descriptor,
        criteria_json=json.dumps(criteria),
        improvement_suggestions_json=json.dumps(improvement_suggestions),
        assessor_notes=assessor_notes,
    )
    db.add(row)
    if auto_commit:
        await db.commit()
    else:
        await db.flush()
    await db.refresh(row)
    return _to_record(row)


async def list_rubric_assessments(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    user_id: str | None = None,
    unit_id: str | None = None,
    cohort: str | None = None,
) -> tuple[int, list[RubricAssessmentRecord]]:
    """Return total count and paginated rubric assessment records."""
    filters = []
    if user_id is not None:
        filters.append(RubricAssessment.user_id == user_id)
    if unit_id is not None:
        filters.append(RubricAssessment.unit_id == unit_id)
    if cohort is not None:
        filters.append(RubricAssessment.cohort == cohort)

    total_stmt = select(func.count(RubricAssessment.id))
    if filters:
        total_stmt = total_stmt.where(*filters)

    total_result = await db.execute(total_stmt)
    total = int(total_result.scalar_one())

    rows_stmt = (
        select(RubricAssessment)
        .order_by(desc(RubricAssessment.created_at), desc(RubricAssessment.id))
        .limit(limit)
        .offset(offset)
    )
    if filters:
        rows_stmt = rows_stmt.where(*filters)

    rows_result = await db.execute(rows_stmt)
    rows = rows_result.scalars().all()
    return total, [_to_record(row) for row in rows]


async def rubric_dashboard_summary(
    db: AsyncSession,
    *,
    unit_id: str | None,
    cohort: str | None,
) -> RubricDashboardSummary:
    """Compute aggregate rubric analytics for dashboard surfaces."""
    filters = []
    if unit_id is not None:
        filters.append(RubricAssessment.unit_id == unit_id)
    if cohort is not None:
        filters.append(RubricAssessment.cohort == cohort)

    stmt = select(
        RubricAssessment.weighted_total,
        RubricAssessment.grade_descriptor,
        RubricAssessment.criteria_json,
    )
    if filters:
        stmt = stmt.where(*filters)
    rows = (await db.execute(stmt.limit(2_000))).all()

    if not rows:
        return RubricDashboardSummary(
            total_assessments=0,
            average_weighted_score=0.0,
            median_weighted_score=0.0,
            grade_distribution={},
            criterion_average_scores={},
            raw_scores=[],
        )

    weighted_scores = [float(row.weighted_total) for row in rows]
    grade_distribution: dict[str, int] = {}
    criterion_accumulator: dict[str, list[float]] = {}

    for row in rows:
        descriptor = str(row.grade_descriptor)
        grade_distribution[descriptor] = (
            grade_distribution.get(descriptor, 0) + 1
        )
        try:
            criteria_obj = json.loads(str(row.criteria_json))
        except json.JSONDecodeError:
            continue

        if not isinstance(criteria_obj, list):
            continue
        for criterion in criteria_obj:
            if not isinstance(criterion, dict):
                continue
            key_obj = criterion.get("key")
            score_obj = criterion.get("score")
            if isinstance(key_obj, str) and isinstance(score_obj, (int, float)):
                criterion_accumulator.setdefault(key_obj, []).append(float(score_obj))

    criterion_average_scores: dict[str, float] = {}
    for key, values in criterion_accumulator.items():
        if values:
            criterion_average_scores[key] = round(sum(values) / len(values), 2)

    return RubricDashboardSummary(
        total_assessments=len(rows),
        average_weighted_score=round(sum(weighted_scores) / len(weighted_scores), 2),
        median_weighted_score=round(float(median(weighted_scores)), 2),
        grade_distribution=grade_distribution,
        criterion_average_scores=criterion_average_scores,
        raw_scores=weighted_scores,
    )


async def peer_comparison_summary(
    db: AsyncSession,
    *,
    assessment_id: int,
    cohort: str | None,
    unit_id: str | None,
    min_cohort_size: int = 3,
) -> PeerComparisonSummary:
    """Compute anonymized cohort comparison for a single assessment."""
    target_result = await db.execute(
        select(RubricAssessment).where(RubricAssessment.id == assessment_id)
    )
    target = target_result.scalar_one_or_none()
    if target is None:
        raise ValueError("Assessment not found")

    scoped_cohort = cohort if cohort is not None else target.cohort
    scoped_unit = unit_id if unit_id is not None else target.unit_id

    filters = [RubricAssessment.template_id == target.template_id]
    if scoped_cohort is not None:
        filters.append(RubricAssessment.cohort == scoped_cohort)
    if scoped_unit is not None:
        filters.append(RubricAssessment.unit_id == scoped_unit)

    scores_result = await db.execute(
        select(RubricAssessment.weighted_total).where(*filters)
    )
    scores = sorted(float(row.weighted_total) for row in scores_result.all())
    if len(scores) < min_cohort_size:
        raise ValueError("Cohort too small for anonymized comparison")

    user_score = float(target.weighted_total)
    less_or_equal = sum(1 for score in scores if score <= user_score)
    percentile = (less_or_equal / len(scores)) * 100.0

    return PeerComparisonSummary(
        assessment_id=assessment_id,
        cohort_size=len(scores),
        user_score=round(user_score, 2),
        cohort_average_score=round(sum(scores) / len(scores), 2),
        cohort_median_score=round(float(median(scores)), 2),
        percentile=round(percentile, 2),
        min_score=round(scores[0], 2),
        max_score=round(scores[-1], 2),
    )
