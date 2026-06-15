"""Tests for write router pattern checks and rubric workflows."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.api.routers.write import (
    AnalyseRubricRequest,
    AnalyseRubricResponse,
    CheckResponse,
    EntropyDraftRequest,
    InverseCoverageRequest,
    PromptRetentionRequest,
    ScrubResponse,
    StructureResponse,
    StyleCalibrationRequest,
    StylometricConstraintRequest,
    StylometricDriftRequest,
    analyse_rubric,
    analyse_structure,
    check_patterns,
    entropy_draft,
    get_calibration_quiz,
    inverse_coverage,
    prompt_retention_map,
    scrub_writing,
    stylometric_constraints,
    stylometric_drift,
    submit_calibration,
)
from deeptutor.models.user import User
from deeptutor.models.user_stylometric_profile import UserStylometricProfile
from deeptutor.services.writing.rubric_history_service import RubricDashboardSummary


class _FakeResp:
    """Simple fake response object for mocked chat completions."""

    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    """Mock chat completion API returning predictable rubric payloads."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    async def create(self, **_kwargs: object) -> _FakeResp:
        return _FakeResp(json.dumps(self._payload))


class _FakeClient:
    """Mock OpenAI client wrapper used by write router."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(payload))


@pytest.fixture
def mock_user(db_session: AsyncSession) -> User:
    """Create a direct-call test user object."""
    user = User(
        id="test-user-write",
        username="write_test",
        email="write_test@test.local",
        password_hash="hashed",
        role="student",
        is_active=True,
    )
    db_session.add(user)
    return user


def _rubric_payload(scores: dict[str, int]) -> dict[str, object]:
    """Build rubric payload with template-compatible criteria list."""
    criteria: list[dict[str, object]] = []
    for key, score in scores.items():
        criteria.append(
            {
                "key": key,
                "score": score,
                "feedback": f"Feedback for {key}",
                "evidence": f"Evidence for {key}",
            }
        )
    return {
        "criteria": criteria,
        "assessor_notes": "Clear strengths and next actions.",
    }


@pytest.mark.asyncio
async def test_check_patterns_detects_hedging(mock_user: User) -> None:
    """Pattern checks identify passive hedging and high-severity terms."""
    text = (
        "It has been argued that social media is important. "
        "It is important to note that students use platforms daily. "
        "This essay will explore the topic thoroughly."
    )

    result = await check_patterns(text=text, _user=mock_user)

    assert isinstance(result, CheckResponse)
    assert result.score > 0
    assert len(result.flags) > 0
    assert any(flag.severity == "high" for flag in result.flags)


@pytest.mark.asyncio
async def test_analyse_structure_detects_transitional_overload(mock_user: User) -> None:
    """Structure analysis flags repeated conjunctive openings."""
    text = (
        "Social media affects learning outcomes. "
        "Furthermore, platforms shape peer interaction. "
        "Moreover, notifications fragment attention. "
        "Additionally, engagement can become performative."
    )

    result = await analyse_structure(text=text, _user=mock_user)

    assert isinstance(result, StructureResponse)
    assert any(issue.issue_type == "transitional_overload" for issue in result.issues)


@pytest.mark.asyncio
async def test_scrub_writing_returns_changes(mock_user: User) -> None:
    """Scrub endpoint parses structured edit response."""
    text = "This essay will explore social media in education."
    payload = {
        "scrubbed_text": "Social media changes learning in structured classroom settings.",
        "changes": [
            {
                "original": "This essay will explore social media in education.",
                "replacement": "Social media changes learning in structured classroom settings.",
                "reason": "Removed roadmap opener.",
            }
        ],
    }

    with patch(
        "deeptutor.api.routers.write._get_client", return_value=_FakeClient(payload)
    ):
        result = await scrub_writing(text=text, _user=mock_user)

    assert isinstance(result, ScrubResponse)
    assert result.changes_count == 1
    assert "will explore" not in result.scrubbed_text


@pytest.mark.asyncio
async def test_analyse_rubric_direct_call_with_template(
    mock_user: User, db_session: AsyncSession
) -> None:
    """Direct rubric call returns template id and suggestions."""
    payload = _rubric_payload(
        {
            "thesis_strength": 8,
            "evidence_quality": 6,
            "argument_clarity": 7,
            "structure": 8,
            "expression": 6,
            "engagement": 5,
        }
    )

    with patch(
        "deeptutor.api.routers.write._get_client", return_value=_FakeClient(payload)
    ):
        result = await analyse_rubric(
            body=AnalyseRubricRequest(
                text="A draft with argument and moderate evidence quality.",
                template_id="standard",
                save_history=True,
                cohort="week-3",
            ),
            db=db_session,
            user=mock_user,
        )

    assert isinstance(result, AnalyseRubricResponse)
    assert result.template_id == "standard"
    assert result.assessment_id is not None
    assert len(result.improvement_suggestions) >= 1


@pytest.mark.asyncio
async def test_templates_endpoint_returns_defaults(
    client: AsyncClient,
    student_token: str,
) -> None:
    """Template listing exposes built-in rubric templates."""
    response = await client.get(
        "/api/write/rubric/templates",
        headers={"Authorization": f"Bearer {student_token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    template_ids = {template["id"] for template in payload["templates"]}
    assert "standard" in template_ids
    assert "evidence_heavy" in template_ids
    assert "critical_engagement" in template_ids


@pytest.mark.asyncio
async def test_compare_endpoint_returns_deltas(
    client: AsyncClient,
    student_token: str,
) -> None:
    """Compare endpoint returns before/after deltas and group id."""
    low_payload = _rubric_payload(
        {
            "thesis_strength": 4,
            "evidence_quality": 4,
            "argument_clarity": 4,
            "structure": 5,
            "expression": 4,
            "engagement": 3,
        }
    )
    high_payload = _rubric_payload(
        {
            "thesis_strength": 8,
            "evidence_quality": 8,
            "argument_clarity": 8,
            "structure": 8,
            "expression": 8,
            "engagement": 8,
        }
    )
    side_effect_clients = [_FakeClient(low_payload), _FakeClient(high_payload)]

    with patch(
        "deeptutor.api.routers.write._get_client", side_effect=side_effect_clients
    ):
        response = await client.post(
            "/api/write/analyse-rubric/compare",
            headers={"Authorization": f"Bearer {student_token}"},
            json={
                "original_text": "Baseline draft with weak support.",
                "improved_text": "Revised draft with stronger evidence and clarity.",
                "template_id": "standard",
                "cohort": "week-5",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison_group_id"]
    assert payload["weighted_total_delta"] > 0
    assert len(payload["criterion_deltas"]) == 6


@pytest.mark.asyncio
async def test_iterate_endpoint_supports_multi_topic(
    client: AsyncClient,
    student_token: str,
) -> None:
    """Iteration endpoint handles multiple topic and grade-level inputs."""
    payload = _rubric_payload(
        {
            "thesis_strength": 7,
            "evidence_quality": 7,
            "argument_clarity": 7,
            "structure": 7,
            "expression": 7,
            "engagement": 7,
        }
    )

    with patch(
        "deeptutor.api.routers.write._get_client", return_value=_FakeClient(payload)
    ):
        response = await client.post(
            "/api/write/analyse-rubric/iterate",
            headers={"Authorization": f"Bearer {student_token}"},
            json={
                "template_id": "standard",
                "items": [
                    {
                        "topic": "AI ethics in education",
                        "grade_level": "undergraduate",
                        "cohort": "week-1",
                        "text": "Essay draft one with clear claim and moderate support.",
                    },
                    {
                        "topic": "Climate adaptation policy",
                        "grade_level": "masters",
                        "cohort": "week-1",
                        "text": "Essay draft two with explicit argument and linked evidence.",
                    },
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["template_id"] == "standard"
    assert len(data["results"]) == 2
    assert data["results"][0]["topic"] == "AI ethics in education"


@pytest.mark.asyncio
async def test_history_dashboard_and_peer_comparison(
    client: AsyncClient,
    instructor_token: str,
    student_token: str,
) -> None:
    """History, dashboard, and peer comparison work with anonymized cohorts."""
    payload = _rubric_payload(
        {
            "thesis_strength": 7,
            "evidence_quality": 8,
            "argument_clarity": 7,
            "structure": 8,
            "expression": 7,
            "engagement": 8,
        }
    )

    with patch(
        "deeptutor.api.routers.write._get_client", return_value=_FakeClient(payload)
    ):
        assessment_ids: list[int] = []
        for idx in range(3):
            create_resp = await client.post(
                "/api/write/analyse-rubric",
                headers={"Authorization": f"Bearer {student_token}"},
                json={
                    "text": f"Essay draft cohort sample {idx} with stable argument and support.",
                    "template_id": "standard",
                    "cohort": "cohort-a",
                    "unit_id": "unit-1",
                    "save_history": True,
                },
            )
            assert create_resp.status_code == 200
            assessment_ids.append(create_resp.json()["assessment_id"])

    history_resp = await client.get(
        "/api/write/rubric/history",
        headers={"Authorization": f"Bearer {student_token}"},
        params={"limit": 10, "offset": 0},
    )
    assert history_resp.status_code == 200
    assert history_resp.json()["total"] >= 3

    dashboard_resp = await client.get(
        "/api/write/rubric/dashboard",
        headers={"Authorization": f"Bearer {instructor_token}"},
        params={"unit_id": "unit-1", "cohort": "cohort-a"},
    )
    assert dashboard_resp.status_code == 200
    dashboard_data = dashboard_resp.json()
    assert dashboard_data["total_assessments"] >= 3
    assert dashboard_data["average_weighted_score"] > 0

    peer_resp = await client.get(
        "/api/write/rubric/peer-comparison",
        headers={"Authorization": f"Bearer {student_token}"},
        params={
            "assessment_id": assessment_ids[0],
            "unit_id": "unit-1",
            "cohort": "cohort-a",
        },
    )
    assert peer_resp.status_code == 200
    peer_data = peer_resp.json()
    assert peer_data["cohort_size"] >= 3
    assert 0.0 <= peer_data["percentile"] <= 100.0


@pytest.mark.asyncio
async def test_invalid_custom_template_weight_fails(
    client: AsyncClient,
    student_token: str,
) -> None:
    """Custom template with invalid total weight is rejected."""
    payload = _rubric_payload(
        {
            "single": 5,
        }
    )

    with patch(
        "deeptutor.api.routers.write._get_client", return_value=_FakeClient(payload)
    ):
        response = await client.post(
            "/api/write/analyse-rubric",
            headers={"Authorization": f"Bearer {student_token}"},
            json={
                "text": "Essay with custom template.",
                "template_id": "custom",
                "custom_template": {
                    "id": "custom",
                    "name": "Custom",
                    "criteria": [
                        {
                            "key": "single",
                            "label": "Single Criterion",
                            "max_score": 10,
                            "weight": 0.8,
                            "description": "One criterion only.",
                        }
                    ],
                },
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_dashboard_chart_endpoint_returns_parallel_arrays(
    client: AsyncClient,
    instructor_token: str,
    student_token: str,
) -> None:
    """Dashboard chart endpoint returns parallel label/value arrays for plotting."""
    payload = _rubric_payload(
        {
            "thesis_strength": 8,
            "evidence_quality": 7,
            "argument_clarity": 9,
            "structure": 8,
            "expression": 7,
            "engagement": 8,
        }
    )

    with patch(
        "deeptutor.api.routers.write._get_client", return_value=_FakeClient(payload)
    ):
        for idx in range(3):
            create_resp = await client.post(
                "/api/write/analyse-rubric",
                headers={"Authorization": f"Bearer {student_token}"},
                json={
                    "text": f"Chart test essay draft {idx} with strong argument.",
                    "template_id": "standard",
                    "cohort": "chart-cohort",
                    "unit_id": "unit-chart",
                    "save_history": True,
                },
            )
            assert create_resp.status_code == 200

    chart_resp = await client.get(
        "/api/write/rubric/dashboard/chart",
        headers={"Authorization": f"Bearer {instructor_token}"},
        params={"unit_id": "unit-chart", "cohort": "chart-cohort"},
    )
    assert chart_resp.status_code == 200
    data = chart_resp.json()

    # grade_distribution — parallel labels + values
    grade_dist = data["grade_distribution"]
    assert isinstance(grade_dist["labels"], list)
    assert isinstance(grade_dist["values"], list)
    assert len(grade_dist["labels"]) == len(grade_dist["values"])
    assert len(grade_dist["labels"]) >= 1

    # criterion_averages — one entry per rubric criterion
    crit_avg = data["criterion_averages"]
    assert isinstance(crit_avg["labels"], list)
    assert isinstance(crit_avg["values"], list)
    assert len(crit_avg["labels"]) == len(crit_avg["values"])
    assert all(0.0 <= v <= 10.0 for v in crit_avg["values"])

    # score_histogram — 10 fixed bands (0–10 through 90–100)
    histogram = data["score_histogram"]
    assert len(histogram["labels"]) == 10
    assert len(histogram["values"]) == 10
    assert sum(histogram["values"]) >= 3  # all 3 saved assessments accounted for

    # summary KPIs
    summary = data["summary"]
    assert summary["total_assessments"] >= 3
    assert summary["average_score"] > 0.0
    assert summary["median_score"] > 0.0


@pytest.mark.asyncio
async def test_dashboard_chart_endpoint_orders_known_grades_before_unknown(
    client: AsyncClient,
    instructor_token: str,
) -> None:
    """Chart grade labels preserve academic-band order and append unknown labels."""
    summary = RubricDashboardSummary(
        total_assessments=5,
        average_weighted_score=68.4,
        median_weighted_score=69.0,
        grade_distribution={
            "Custom Distinction": 2,
            "Pass": 1,
            "Upper Second": 1,
            "Merit": 1,
        },
        criterion_average_scores={"argument_clarity": 8.4},
        raw_scores=[72.0, 68.0, 55.0, 81.0, 66.0],
    )

    with patch(
        "deeptutor.api.routers.write.rubric_dashboard_summary",
        new=AsyncMock(return_value=summary),
    ):
        chart_resp = await client.get(
            "/api/write/rubric/dashboard/chart",
            headers={"Authorization": f"Bearer {instructor_token}"},
        )

    assert chart_resp.status_code == 200
    data = chart_resp.json()
    assert data["grade_distribution"]["labels"] == [
        "Upper Second",
        "Pass",
        "Custom Distinction",
        "Merit",
    ]
    assert data["grade_distribution"]["values"] == [1.0, 1.0, 2.0, 1.0]


@pytest.mark.asyncio
async def test_stylometric_drift_returns_sorted_series(mock_user: User) -> None:
    """Drift endpoint returns chronological series and cadence alerts."""
    response = await stylometric_drift(
        body=StylometricDriftRequest(
            submissions=[
                {
                    "submission_id": "s2",
                    "semester": "2024-S2",
                    "submitted_at": "2024-10-01T10:00:00Z",
                    "text": "Feedback: weak references.\n\nThis argument was developed and was repeated. "
                    "It was framed and was presented in broad terms.",
                },
                {
                    "submission_id": "s1",
                    "semester": "2024-S1",
                    "submitted_at": "2024-03-01T10:00:00Z",
                    "text": "The paper evaluates policy trade-offs with direct evidence. "
                    "It compares alternatives and explains consequences clearly.",
                },
            ]
        ),
        _user=mock_user,
    )

    assert response.points[0].submission_id == "s1"
    assert len(response.series.submitted_at) == 2
    assert len(response.alerts) >= 1


@pytest.mark.asyncio
async def test_prompt_retention_map_flags_middle_chunks(mock_user: User) -> None:
    """Prompt retention map classifies middle placements as vulnerable."""
    response = await prompt_retention_map(
        body=PromptRetentionRequest(
            system_preamble_tokens=200,
            chunks=[
                {"chunk_id": "c1", "filename": "a.md", "token_count": 120},
                {"chunk_id": "c2", "filename": "b.md", "token_count": 400},
                {"chunk_id": "c3", "filename": "c.md", "token_count": 120},
            ],
        ),
        _user=mock_user,
    )

    assert response.total_tokens == 840
    assert response.vulnerable_middle_count >= 1


@pytest.mark.asyncio
async def test_inverse_coverage_uses_supplied_syllabus_chunks(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Inverse coverage isolates low-similarity syllabus gaps."""
    body = InverseCoverageRequest(
        outline_text="outline text",
        threshold=0.50,
        syllabus_chunks=[
            {"id": "t1", "module": "m1", "theory": "A", "embedding": [1.0, 0.0]},
            {"id": "t2", "module": "m1", "theory": "B", "embedding": [0.0, 1.0]},
        ],
    )

    with patch(
        "deeptutor.api.routers.write._embed_outline_text",
        new=AsyncMock(return_value=[1.0, 0.0]),
    ):
        response = await inverse_coverage(body=body, db=db_session, _user=mock_user)

    assert response.total_chunks_evaluated == 2
    assert len(response.gaps) == 1
    assert response.gaps[0].associated_chunk_id == "t2"


@pytest.mark.asyncio
async def test_stylometric_constraints_falls_back_without_profile(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Constraint compiler endpoint returns default instructions without profile row."""
    response = await stylometric_constraints(
        body=StylometricConstraintRequest(outline_node="Argument outline node"),
        db=db_session,
        user=mock_user,
    )

    assert response.profile_sample_count == 0
    assert len(response.instructions) == 2
    assert response.instructions[0]["role"] == "system"


@pytest.mark.asyncio
async def test_entropy_draft_validates_generated_variance(mock_user: User) -> None:
    """Entropy drafting endpoint evaluates structural variance on demand."""
    response = await entropy_draft(
        body=EntropyDraftRequest(
            notes="bullet notes",
            generated_prose=(
                "Brief claim. "
                "This sentence expands the analysis through multiple linked clauses and several concrete examples to force a markedly longer cadence profile for validation. "
                "Closing synthesis point."
            ),
        ),
        _user=mock_user,
    )

    assert "Sentence 1" in response.prompt
    assert response.entropy_passed is True


@pytest.mark.asyncio
async def test_get_calibration_quiz_returns_pairs(mock_user: User) -> None:
    """Calibration endpoint serves deterministic binary contrast pairs."""
    response = await get_calibration_quiz(_user=mock_user)

    assert response.total_pairs >= 6
    assert len(response.pairs) == response.total_pairs
    assert response.pairs[0].pair_id


@pytest.mark.asyncio
async def test_submit_calibration_commits_profile(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Calibration submission stores a baseline profile in user stylometric table."""
    response = await submit_calibration(
        body=StyleCalibrationRequest(
            selections=[
                {"pair_id": "sv_01", "choice": "cadence_b"},
                {"pair_id": "pr_01", "choice": "cadence_b"},
                {"pair_id": "ld_01", "choice": "cadence_b"},
            ],
            commit=True,
        ),
        db=db_session,
        user=mock_user,
    )

    assert response.committed is True
    assert response.valid_selection_count == 3
    assert response.profile.passive_voice_ratio < 0.18

    row = (
        await db_session.execute(
            select(UserStylometricProfile).where(
                UserStylometricProfile.user_id == mock_user.id
            )
        )
    ).scalar_one_or_none()
    assert row is not None
    assert row.last_grade == "calibrated"


@pytest.mark.asyncio
async def test_calibrate_endpoint_rejects_unknown_pair(
    client: AsyncClient,
    student_token: str,
) -> None:
    """Calibration endpoint rejects payloads with no valid pair identifiers."""
    response = await client.post(
        "/api/write/calibrate",
        headers={"Authorization": f"Bearer {student_token}"},
        json={
            "selections": [
                {"pair_id": "unknown_pair", "choice": "cadence_a"},
            ],
            "commit": False,
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_verify_topology_evaluates_syntactic_hierarchy(
    mock_user: User, db_session: AsyncSession
) -> None:
    """Topology verification computes depth, branching, and variance."""
    from deeptutor.api.routers.write import SyntacticTopologyRequest, verify_topology

    response = await verify_topology(
        body=SyntacticTopologyRequest(
            text=(
                "The policy was implemented. "
                "Because of delays, which affected coordination, the outcome was constrained. "
                "This shows the result."
            )
        ),
        db=db_session,
        user=mock_user,
    )

    assert response.mean_dependency_depth > 0
    assert response.branching_coefficient > 1.0
    assert response.baseline_available is False


@pytest.mark.asyncio
async def test_equalise_vocabulary_applies_substitutions(mock_user: User) -> None:
    """Lexical equaliser applies direct vocabulary substitutions."""
    from deeptutor.api.routers.write import (
        LexicalEqualiseRequest,
        equalise_vocabulary,
    )

    response = await equalise_vocabulary(
        body=LexicalEqualiseRequest(
            text="This is a testament to the crucial importance of delving into the topic.",
            substitutions={
                "testament": "evidence",
                "crucial": "key",
            },
        ),
        _user=mock_user,
    )

    assert (
        "testament" not in response.text.lower() or "evidence" in response.text.lower()
    )
    assert response.replacements_applied > 0


@pytest.mark.asyncio
async def test_validate_burstiness_windows_check_variance(mock_user: User) -> None:
    """Burstiness validator checks sentence-length variance per paragraph."""
    from deeptutor.api.routers.write import (
        BurstinessWindowRequest,
        validate_burstiness,
    )

    response = await validate_burstiness(
        body=BurstinessWindowRequest(
            text=(
                "Short. Very long sentence with multiple clauses and several ideas woven together. X.\n\n"
                "Brief. This extends with detail. Done."
            ),
            minimum_variance=2.0,
        ),
        _user=mock_user,
    )

    assert len(response.windows) >= 1
    assert isinstance(response.all_passed, bool)


@pytest.mark.asyncio
async def test_provenance_append_creates_hash_chained_receipt(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Provenance endpoint appends hash-chained mutation receipt."""
    from deeptutor.api.routers.write import (
        ProvenanceEventRequest,
        append_provenance_event,
    )

    response = await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-1",
            source_type="manual_typing",
            character_delta_count=42,
        ),
        db=db_session,
        user=mock_user,
    )

    assert response.id > 0
    assert response.sequence_index == 0
    assert response.signature_hash != ""
    assert response.previous_block_hash == "0" * 64


@pytest.mark.asyncio
async def test_provenance_ledger_returns_ordered_receipts(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Provenance ledger returns chain-ordered mutation receipts."""
    from deeptutor.api.routers.write import (
        ProvenanceEventRequest,
        append_provenance_event,
        get_provenance_ledger,
    )

    await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-2",
            source_type="manual_typing",
            character_delta_count=10,
        ),
        db=db_session,
        user=mock_user,
    )
    await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-2",
            source_type="api_scrub",
            character_delta_count=20,
        ),
        db=db_session,
        user=mock_user,
    )

    response = await get_provenance_ledger(
        document_id="doc-2",
        db=db_session,
        user=mock_user,
    )

    assert len(response.items) == 2
    assert response.items[0].sequence_index == 0
    assert response.items[1].sequence_index == 1
    assert response.items[1].previous_block_hash == response.items[0].signature_hash


@pytest.mark.asyncio
async def test_provenance_verify_reports_valid_chain(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Provenance verify endpoint reports valid hash chain after normal appends."""
    from deeptutor.api.routers.write import (
        ProvenanceEventRequest,
        append_provenance_event,
        verify_provenance_ledger,
    )

    await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-verify-valid",
            source_type="manual_typing",
            character_delta_count=12,
        ),
        db=db_session,
        user=mock_user,
    )
    await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-verify-valid",
            source_type="api_scrub",
            character_delta_count=4,
        ),
        db=db_session,
        user=mock_user,
    )

    response = await verify_provenance_ledger(
        document_id="doc-verify-valid",
        db=db_session,
        user=mock_user,
    )

    assert response.is_valid is True
    assert response.checked_count == 2
    assert response.first_invalid_sequence_index is None
    assert response.reason is None


@pytest.mark.asyncio
async def test_provenance_verify_detects_tampered_signature(
    mock_user: User,
    db_session: AsyncSession,
) -> None:
    """Provenance verify endpoint detects tampering when signature hash is modified."""
    from deeptutor.api.routers.write import (
        ProvenanceEventRequest,
        append_provenance_event,
        verify_provenance_ledger,
    )
    from deeptutor.models.text_mutation_receipt import TextMutationReceipt

    await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-verify-tamper",
            source_type="manual_typing",
            character_delta_count=20,
        ),
        db=db_session,
        user=mock_user,
    )
    await append_provenance_event(
        body=ProvenanceEventRequest(
            document_id="doc-verify-tamper",
            source_type="context_expansion",
            character_delta_count=8,
        ),
        db=db_session,
        user=mock_user,
    )

    rows = (
        (
            await db_session.execute(
                select(TextMutationReceipt)
                .where(TextMutationReceipt.user_id == mock_user.id)
                .where(TextMutationReceipt.document_id == "doc-verify-tamper")
                .order_by(TextMutationReceipt.sequence_index.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2

    rows[1].signature_hash = "tampered-signature"
    await db_session.commit()

    response = await verify_provenance_ledger(
        document_id="doc-verify-tamper",
        db=db_session,
        user=mock_user,
    )

    assert response.is_valid is False
    assert response.first_invalid_sequence_index == 1
    assert response.reason == "signature_mismatch"


@pytest.mark.asyncio
async def test_provenance_append_retries_after_integrity_conflict(
    mock_user: User,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance append retries when a sequence uniqueness conflict occurs."""
    from deeptutor.services.writing.authorship_receipt_service import append_mutation_receipt

    original_flush = db_session.flush
    flush_calls = 0

    async def flaky_flush(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 1:
            raise IntegrityError("INSERT", {}, RuntimeError("simulated conflict"))
        await original_flush()

    monkeypatch.setattr(db_session, "flush", flaky_flush)

    receipt = await append_mutation_receipt(
        db_session,
        user_id=mock_user.id,
        document_id="doc-retry",
        source_type="manual_typing",
        character_delta_count=7,
    )

    assert flush_calls >= 2
    assert receipt.sequence_index == 0
    assert receipt.previous_block_hash == "0" * 64
