"""Write assistance router — AI pattern detection and structural analysis.

Endpoints
---------
GET  /api/write/patterns          — serve the pattern config to the frontend (cached)
POST /api/write/check             — flag AI vocabulary and phrase patterns in text
POST /api/write/analyse-structure — structural tells (paragraph uniformity, thesis absence, transitional overload)
POST /api/write/scrub             — LLM-as-a-Judge rewrite pass to remove AI patterns
POST /api/write/analyse-rubric    — criterion-by-criterion essay assessment with weighted scoring

Design rationale: patterns live in config/ai_patterns.yml, not in source code.
The frontend downloads the vocabulary rules once and runs regex matching
client-side for real-time highlighting.  The structure endpoint handles analysis
that requires multi-paragraph context.  The scrub endpoint adds a second LLM
pass specialised in removing AI structural tells — separate from the generation
model so it can apply stricter constraints without degrading answer quality.
The rubric endpoint provides academic-grading-style assessment with six criteria
(thesis, evidence, clarity, structure, expression, engagement) scored 0–10 each
and aggregated to a 0–100 scale with UK academic grade descriptors.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import logging
import re
import statistics
import uuid
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.api.deps import get_current_user, get_db, require_instructor
from deeptutor.models.chunk import DocumentChunk
from deeptutor.models.user import User
from deeptutor.prompts import WRITE_SCRUB_SYSTEM
from deeptutor.services.writing.authorship_receipt_service import (
    MutationReceiptAppendError,
    append_mutation_receipt,
    list_mutation_receipts,
    verify_mutation_chain,
)
from deeptutor.services.writing.provider_client import (
    get_embedding_client,
    get_llm_client,
    resolve_embedding_model,
    resolve_llm_model,
    should_send_embedding_dimensions,
)
from deeptutor.services.writing.rubric_history_service import (
    create_rubric_assessment,
    list_rubric_assessments,
    peer_comparison_summary,
    rubric_dashboard_summary,
)
from deeptutor.services.writing.style_calibration_service import (
    DEFAULT_CONTRAST_PAIRS,
    PreferenceSignal,
    StyleCalibrationEngine,
)
from deeptutor.services.writing.stylometric_profile_service import (
    append_telemetry_events,
    get_user_stylometric_profile,
)
from deeptutor.services.writing.writing_blueprint_service import (
    LexicalEntropyEqualiser,
    SlidingWindowBurstinessController,
    StructuralEntropyExtrapolator,
    StylometricPromptCompiler,
    SyntacticTopologyEvaluator,
    analyse_historical_drift,
    calculate_prompt_retention_map,
    compute_inverse_coverage,
    extract_student_prose_segments,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/write", tags=["write"])

_PATTERNS_FILE = (
    Path(__file__).resolve().parent.parent.parent / "config" / "ai_patterns.yml"
)
_STYLE_CALIBRATION_ENGINE = StyleCalibrationEngine(DEFAULT_CONTRAST_PAIRS)


class PatternRule(TypedDict, total=False):
    """Typed rule schema loaded from ai_patterns.yml."""

    pattern: str
    category: str
    message: str
    suggestion: str
    severity: str


class ScrubChangePayload(TypedDict, total=False):
    """Typed change payload returned by scrub completion."""

    original: str
    replacement: str
    reason: str


_compiled_pattern_rules: list[tuple[re.Pattern[str], PatternRule]] = []


_ABBREVIATION_SENTINELS: tuple[str, ...] = (
    "e.g.",
    "i.e.",
    "et al.",
    "etc.",
    "mr.",
    "mrs.",
    "dr.",
    "prof.",
    "vs.",
)


def _get_client() -> AsyncOpenAI:
    return get_llm_client()


_PROVIDER_FAILURE_TYPES = (
    OpenAIRateLimitError,
    APIConnectionError,
    APIStatusError,
    RuntimeError,
    OSError,
    TimeoutError,
    ValueError,
    TypeError,
)


def _map_provider_failure(*, operation: str, exc: Exception) -> HTTPException:
    """Map provider/runtime failures to stable API error responses."""
    logger.warning("Provider failure during %s: %s", operation, exc)
    if isinstance(exc, OpenAIRateLimitError):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{operation} provider rate limit reached; retry shortly",
        )
    if isinstance(exc, (APIConnectionError, APIStatusError)):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{operation} provider is temporarily unavailable",
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"{operation} is temporarily unavailable",
    )


def _map_database_failure(*, operation: str, exc: Exception) -> HTTPException:
    """Map SQLAlchemy failures to stable API error responses."""
    logger.warning("Database failure during %s: %s", operation, exc)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"{operation} is temporarily unavailable",
    )


# ---------------------------------------------------------------------------
# Pattern config loader (module-level cache — loaded once per process)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_patterns() -> dict[str, object]:
    """Load and cache the YAML pattern config. Raises on missing/invalid file."""
    global _compiled_pattern_rules

    with _PATTERNS_FILE.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        data = {}

    _compiled_pattern_rules = []
    for section in ("vocabulary", "phrases", "ngram_phrases"):
        section_obj = data.get(section)
        if not isinstance(section_obj, list):
            continue
        for raw_rule in section_obj:
            if not isinstance(raw_rule, dict):
                continue
            pattern_obj = raw_rule.get("pattern")
            if not isinstance(pattern_obj, str):
                continue
            rule = cast(PatternRule, raw_rule)
            try:
                _compiled_pattern_rules.append(
                    (re.compile(pattern_obj, re.IGNORECASE), rule)
                )
            except re.error as exc:
                logger.warning("Skipping invalid pattern %r: %s", pattern_obj, exc)

    logger.info(
        "Loaded AI pattern config: %d vocab + %d phrase + %d ngram rules",
        len(data.get("vocabulary", [])),
        len(data.get("phrases", [])),
        len(data.get("ngram_phrases", [])),
    )
    return cast(dict[str, object], data)


def _split_sentences_safe(text: str) -> list[str]:
    """Split sentences while preserving common academic abbreviations."""
    if not text.strip():
        return []

    protected = text
    for abbreviation in _ABBREVIATION_SENTINELS:
        sentinel = abbreviation.replace(".", "<DOT>")
        protected = re.sub(
            re.escape(abbreviation),
            sentinel,
            protected,
            flags=re.IGNORECASE,
        )

    sentences = [
        segment.replace("<DOT>", ".").strip()
        for segment in re.split(r"(?<=[.!?])\s+", protected)
        if segment.strip()
    ]
    return sentences


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PatternFlag(BaseModel):
    """A single flagged match within the submitted text."""

    start: int
    end: int
    matched_text: str
    category: str
    message: str
    suggestion: str
    severity: str  # high | medium | low


class CheckResponse(BaseModel):
    """Result of a pattern-check request."""

    flags: list[PatternFlag]
    score: int  # 0–100 AI-signal score (higher = more AI-like signals present)
    summary: dict[str, int]  # count per category
    slop_density: float  # high-severity flags per 100 words (direct intensity metric)
    burstiness: float  # sentence-length CV: low = uniform = AI signal


class StylometricHotspot(BaseModel):
    """One deterministic stylometric hotspot in draft text."""

    start: int
    end: int
    text: str
    reason: str
    replacement_hint: str


class StylometricProfileResponse(BaseModel):
    """Deterministic stylometric profile payload."""

    token_diversity: float
    burstiness: float
    filler_density: float
    hotspots: list[StylometricHotspot]


class StylometricScrubResponse(BaseModel):
    """Sanitised prose plus deterministic replacement list."""

    scrubbed_text: str
    replacements: list[ScrubChange]
    low_entropy_detected: bool


class StructureIssue(BaseModel):
    """A structural tell detected at paragraph level."""

    issue_type: str
    message: str
    suggestion: str
    paragraph_index: int | None = None


class StructureResponse(BaseModel):
    """Result of a structure analysis request."""

    issues: list[StructureIssue]
    paragraph_count: int
    avg_word_count: float
    word_count_cv: float  # coefficient of variation — high = uniform length (AI signal)
    sentence_length_cv: float  # burstiness proxy — low = suspiciously consistent


class ScrubChange(BaseModel):
    """A single edit made by the LLM scrubbing pass."""

    original: str
    replacement: str
    reason: str


class ScrubResponse(BaseModel):
    """Result of the LLM-as-a-Judge scrub pass."""

    scrubbed_text: str
    changes: list[ScrubChange]
    changes_count: int


class RubricCriterion(BaseModel):
    """A single rubric criterion with descriptive feedback."""

    key: str
    criterion: str
    max_score: int
    score: int
    weight: float  # 0.0–1.0 proportion toward weighted total
    feedback: str
    evidence: str | None = None  # specific examples from text


class RubricScore(BaseModel):
    """Individual criterion scoring (not yet weighted)."""

    criteria: list[RubricCriterion]
    weighted_total: float  # 0.0–100.0
    raw_total: int  # unweighted sum of all criterion scores
    raw_max: int  # unweighted maximum possible
    assessor_notes: str


class AnalyseRubricResponse(BaseModel):
    """Result of rubric-mode essay analysis."""

    rubric: RubricScore
    grade_descriptor: str  # e.g. "First Class (90–100)", "Upper Second (70–89)", etc.
    template_id: str
    assessment_id: int | None = None
    improvement_suggestions: list[str]


class RubricCriterionTemplate(BaseModel):
    """Configuration for one rubric criterion."""

    key: str = Field(min_length=2, max_length=64)
    label: str = Field(min_length=2, max_length=120)
    max_score: int = Field(default=10, ge=1, le=100)
    weight: float = Field(gt=0.0, le=1.0)
    description: str = Field(min_length=5, max_length=1500)


class RubricTemplate(BaseModel):
    """Rubric template consisting of weighted criteria."""

    id: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=2, max_length=120)
    criteria: list[RubricCriterionTemplate]


class AnalyseRubricRequest(BaseModel):
    """Payload for rubric analysis with optional persistence metadata."""

    text: str = Field(min_length=20, max_length=50_000)
    template_id: str = Field(default="standard", min_length=2, max_length=64)
    custom_template: RubricTemplate | None = None
    unit_id: str | None = Field(default=None, min_length=1, max_length=64)
    topic: str | None = Field(default=None, max_length=256)
    grade_level: str | None = Field(default=None, max_length=64)
    cohort: str | None = Field(default=None, max_length=128)
    save_history: bool = True
    comparison_group_id: str | None = Field(default=None, max_length=64)
    is_improved_version: bool = False


class RubricComparisonDelta(BaseModel):
    """Score change for one criterion between two drafts."""

    key: str
    criterion: str
    before_score: int
    after_score: int
    delta: int


class CompareRubricRequest(BaseModel):
    """Payload to compare baseline and improved drafts."""

    original_text: str = Field(min_length=20, max_length=50_000)
    improved_text: str = Field(min_length=20, max_length=50_000)
    template_id: str = Field(default="standard", min_length=2, max_length=64)
    custom_template: RubricTemplate | None = None
    unit_id: str | None = Field(default=None, min_length=1, max_length=64)
    topic: str | None = Field(default=None, max_length=256)
    grade_level: str | None = Field(default=None, max_length=64)
    cohort: str | None = Field(default=None, max_length=128)
    save_history: bool = True


class CompareRubricResponse(BaseModel):
    """Comparison response for baseline and improved drafts."""

    comparison_group_id: str
    before: AnalyseRubricResponse
    after: AnalyseRubricResponse
    criterion_deltas: list[RubricComparisonDelta]
    weighted_total_delta: float


class RubricIterationItem(BaseModel):
    """One essay entry for multi-topic rubric iteration."""

    topic: str = Field(min_length=2, max_length=256)
    grade_level: str | None = Field(default=None, max_length=64)
    cohort: str | None = Field(default=None, max_length=128)
    unit_id: str | None = Field(default=None, min_length=1, max_length=64)
    text: str = Field(min_length=20, max_length=50_000)


class IterateRubricRequest(BaseModel):
    """Payload for rubric analysis over multiple topics/drafts."""

    items: list[RubricIterationItem] = Field(min_length=1, max_length=20)
    template_id: str = Field(default="standard", min_length=2, max_length=64)
    custom_template: RubricTemplate | None = None
    save_history: bool = True


class RubricIterationResult(BaseModel):
    """Result item for one multi-topic rubric analysis."""

    topic: str
    grade_level: str | None
    result: AnalyseRubricResponse


class IterateRubricResponse(BaseModel):
    """Response for multi-topic rubric analysis."""

    template_id: str
    results: list[RubricIterationResult]


class RubricAssessmentHistoryItem(BaseModel):
    """History item returned in rubric assessment listing."""

    id: int
    user_id: str
    unit_id: str | None
    topic: str | None
    grade_level: str | None
    cohort: str | None
    template_id: str
    is_improved_version: bool
    weighted_total: float
    grade_descriptor: str
    created_at: str


class RubricAssessmentHistoryPage(BaseModel):
    """Paginated rubric assessment history."""

    limit: int
    offset: int
    total: int
    items: list[RubricAssessmentHistoryItem]


class RubricDashboardResponse(BaseModel):
    """Aggregate rubric analytics for dashboard surfaces."""

    total_assessments: int
    average_weighted_score: float
    median_weighted_score: float
    grade_distribution: dict[str, int]
    criterion_average_scores: dict[str, float]


class PeerComparisonResponse(BaseModel):
    """Anonymized cohort score comparison for one assessment."""

    assessment_id: int
    cohort_size: int
    user_score: float
    cohort_average_score: float
    cohort_median_score: float
    percentile: float
    min_score: float
    max_score: float


class _ChartSeries(BaseModel):
    """Generic two-array chart series (labels + values)."""

    labels: list[str]
    values: list[float]


class RubricDashboardChartResponse(BaseModel):
    """Compact, chart-library-ready dashboard payload.

    All arrays are parallel (index *i* in ``labels`` corresponds to index *i* in
    ``values``) so they can be passed directly into Chart.js, Recharts, Nivo, or
    any similar library without client-side transformation.
    """

    # Bar / pie chart — grade band distribution
    grade_distribution: _ChartSeries
    # Radar / spider chart — per-criterion averages (0–10 scale)
    criterion_averages: _ChartSeries
    # Histogram — score distribution bucketed into 10-point bands
    score_histogram: _ChartSeries
    # Flat summary for headline KPI tiles
    summary: dict[str, float | int]


class AdversarialMarkingRequest(BaseModel):
    """Payload for adversarial multi-marker draft simulation."""

    text: str = Field(min_length=20, max_length=50_000)
    template_id: str = Field(default="standard", min_length=2, max_length=64)
    custom_template: RubricTemplate | None = None


class AdversarialFinding(BaseModel):
    """One deduction finding from a strict marker persona."""

    issue: str
    deduction_reason: str
    severity: Literal["low", "medium", "high"] = "medium"


class AdversarialAgentReport(BaseModel):
    """Single marker report with findings and recommendation."""

    agent: str
    focus: str
    findings: list[AdversarialFinding]
    priority_actions: list[str]


class AdversarialMarkingResponse(BaseModel):
    """Combined report from adversarial strict-marker simulation."""

    template_id: str
    reports: list[AdversarialAgentReport]
    combined_risk_summary: str


class ExportDraftRequest(BaseModel):
    """Draft export payload with format and citation-style options."""

    text: str = Field(min_length=20, max_length=60_000)
    title: str = Field(default="Assignment Draft", max_length=180)
    file_format: Literal["md", "docx", "pdf"] = "docx"
    citation_style: Literal["apa7", "harvard"] = "harvard"


class ExportDraftResponse(BaseModel):
    """Base64 encoded exported file payload."""

    filename: str
    mime_type: str
    content_base64: str
    preview_text: str


class WriteTelemetryEvent(BaseModel):
    """One low-level editing event emitted by the frontend editor."""

    op: Literal["insert", "delete"]
    text: str = Field(default="", max_length=2_000)
    source: Literal["keyboard", "clipboard", "system"] = "keyboard"
    is_paste: bool = False


class WriteTelemetryRequest(BaseModel):
    """Batch of editor telemetry events."""

    events: list[WriteTelemetryEvent] = Field(min_length=1, max_length=200)


class WriteTelemetryResponse(BaseModel):
    """Accepted telemetry event count."""

    accepted: int


class StyleCalibrationPairResponse(BaseModel):
    """One semantic contrast pair rendered for user selection."""

    pair_id: str
    sequence_index: int
    metric_target: Literal["sentence_variance", "passive_ratio", "lexical_density"]
    concept: str
    cadence_a_text: str
    cadence_b_text: str


class StyleCalibrationSelection(BaseModel):
    """One binary style preference decision."""

    pair_id: str = Field(min_length=1, max_length=64)
    choice: PreferenceSignal


class StyleCalibrationRequest(BaseModel):
    """Calibration submission payload."""

    selections: list[StyleCalibrationSelection] = Field(min_length=1, max_length=80)
    commit: bool = True


class StyleCalibrationProfile(BaseModel):
    """Computed baseline coefficients from binary contrast selections."""

    sentence_length_variance: float
    passive_voice_ratio: float
    lexical_density_ttr: float


class StyleCalibrationResponse(BaseModel):
    """Style calibration submission response."""

    profile: StyleCalibrationProfile
    valid_selection_count: int
    committed: bool


class StyleCalibrationQuizResponse(BaseModel):
    """Calibration quiz definition served to initialize frontend flow."""

    total_pairs: int
    pairs: list[StyleCalibrationPairResponse]


class HistoricalSubmissionItem(BaseModel):
    """One historical submission payload for drift analytics."""

    submission_id: str = Field(min_length=1, max_length=128)
    semester: str = Field(min_length=1, max_length=64)
    submitted_at: str = Field(min_length=4, max_length=64)
    text: str = Field(min_length=1, max_length=120_000)


class StylometricDriftRequest(BaseModel):
    """Chronological submission set for drift analysis."""

    submissions: list[HistoricalSubmissionItem] = Field(min_length=1, max_length=60)


class StylometricDriftPoint(BaseModel):
    """Per-submission stylometric metric point."""

    submission_id: str
    semester: str
    submitted_at: str
    lexical_density_ttr: float
    mean_sentence_length: float
    sentence_length_variance: float
    passive_voice_ratio: float
    drift_score: float
    prose_segments: int


class StylometricDriftAlert(BaseModel):
    """One drift alert emitted from chronological profile analysis."""

    submission_id: str
    semester: str
    severity: Literal["low", "medium", "high"]
    message: str


class StylometricDriftSeries(BaseModel):
    """Chart-friendly aligned arrays for chronological metrics."""

    submitted_at: list[str]
    lexical_density_ttr: list[float]
    sentence_length_variance: list[float]
    passive_voice_ratio: list[float]
    drift_score: list[float]


class StylometricDriftResponse(BaseModel):
    """Chronological stylometric drift output."""

    points: list[StylometricDriftPoint]
    alerts: list[StylometricDriftAlert]
    series: StylometricDriftSeries


class PromptChunkInput(BaseModel):
    """Chunk token payload for prompt retention mapping."""

    chunk_id: str
    filename: str
    token_count: int = Field(ge=1, le=40_000)


class PromptRetentionRequest(BaseModel):
    """Prompt context payload for lost-in-the-middle diagnostics."""

    system_preamble_tokens: int = Field(ge=0, le=20_000)
    chunks: list[PromptChunkInput] = Field(min_length=1, max_length=400)


class PromptRetentionPlacement(BaseModel):
    """One chunk placement with retention risk."""

    chunk_id: str
    filename: str
    token_index_start: int
    relative_position_percentage: float
    retention_risk_zone: Literal["optimal_top", "vulnerable_middle", "optimal_bottom"]


class PromptRetentionResponse(BaseModel):
    """Prompt retention map response."""

    total_tokens: int
    vulnerable_middle_count: int
    placements: list[PromptRetentionPlacement]


class SyllabusChunkPayload(BaseModel):
    """Optional syllabus chunk payload for direct inverse coverage calls."""

    id: str
    module: str
    theory: str
    embedding: list[float]


class InverseCoverageRequest(BaseModel):
    """Outline payload for latent boundary inverse coverage."""

    outline_text: str = Field(min_length=10, max_length=20_000)
    unit_id: str | None = None
    threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    max_chunks: int = Field(default=400, ge=10, le=2_000)
    syllabus_chunks: list[SyllabusChunkPayload] = Field(
        default_factory=list, max_length=2_000
    )


class InverseCoverageGap(BaseModel):
    """Coverage gap item with resolved max similarity."""

    module_name: str
    missing_theory_title: str
    associated_chunk_id: str
    max_similarity_resolved: float


class InverseCoverageResponse(BaseModel):
    """Inverse coverage analysis response."""

    threshold: float
    total_chunks_evaluated: int
    gaps: list[InverseCoverageGap]


class StylometricConstraintRequest(BaseModel):
    """Compile style constraints from profile and outline context."""

    outline_node: str = Field(min_length=3, max_length=4_000)


class StylometricConstraintResponse(BaseModel):
    """Compiled style constraints for generation payload injection."""

    instructions: list[dict[str, str]]
    profile_sample_count: int


class EntropyDraftRequest(BaseModel):
    """Build entropy-enforced drafting prompt and optional validation result."""

    notes: str = Field(min_length=5, max_length=8_000)
    generated_prose: str | None = Field(default=None, max_length=30_000)


class EntropyDraftResponse(BaseModel):
    """Structural entropy drafting response."""

    prompt: str
    entropy_passed: bool | None


class SyntacticTopologyRequest(BaseModel):
    """Payload for syntactic topology extraction and baseline check."""

    text: str = Field(min_length=20, max_length=60_000)


class SyntacticTopologyResponse(BaseModel):
    """Topology metrics plus optional baseline conformity flag."""

    mean_dependency_depth: float
    branching_coefficient: float
    clause_count_variance: float
    baseline_available: bool
    within_baseline: bool | None


class LexicalEqualiseRequest(BaseModel):
    """Payload for post-generation vocabulary equalisation."""

    text: str = Field(min_length=20, max_length=60_000)
    substitutions: dict[str, str] = Field(default_factory=dict, max_length=128)


class LexicalEqualiseResponse(BaseModel):
    """Equalised text output and replacement statistics."""

    text: str
    replacements_applied: int


class BurstinessWindowRequest(BaseModel):
    """Payload for rolling 3-paragraph burstiness validation."""

    text: str = Field(min_length=20, max_length=80_000)
    minimum_variance: float = Field(default=6.2, ge=0.0, le=40.0)


class BurstinessWindowResult(BaseModel):
    """Validation result for one paragraph window."""

    start_paragraph_index: int
    passed: bool


class BurstinessWindowResponse(BaseModel):
    """Sliding-window burstiness validation summary."""

    windows: list[BurstinessWindowResult]
    all_passed: bool


class ProvenanceEventRequest(BaseModel):
    """Mutation receipt append payload."""

    document_id: str = Field(min_length=1, max_length=64)
    source_type: Literal["manual_typing", "api_scrub", "context_expansion"]
    character_delta_count: int = Field(ge=0, le=50_000)


class ProvenanceEventResponse(BaseModel):
    """Single appended mutation receipt response."""

    id: int
    sequence_index: int
    signature_hash: str
    previous_block_hash: str


class ProvenanceLedgerItem(BaseModel):
    """One mutation receipt from the immutable ledger."""

    id: int
    sequence_index: int
    source_type: str
    character_delta_count: int
    timestamp_epoch: float
    previous_block_hash: str
    signature_hash: str


class ProvenanceLedgerResponse(BaseModel):
    """Ledger response for one user document."""

    document_id: str
    items: list[ProvenanceLedgerItem]


class ProvenanceVerifyResponse(BaseModel):
    """Verification result for immutable mutation chain integrity."""

    document_id: str
    checked_count: int
    is_valid: bool
    first_invalid_sequence_index: int | None
    reason: str | None


# ---------------------------------------------------------------------------
# GET /api/write/patterns
# ---------------------------------------------------------------------------


@router.get("/patterns")
async def get_patterns(
    _user: Annotated[User, Depends(get_current_user)],
) -> dict[str, object]:
    """Return the full AI pattern config so the frontend can run client-side matching.

    The response is stable for the process lifetime (YAML loaded once).
    Clients should cache this for the session.
    """
    return _load_patterns()


# ---------------------------------------------------------------------------
# POST /api/write/check
# ---------------------------------------------------------------------------


_SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}


_KNOWN_FILLER_PHRASES: tuple[tuple[str, str], ...] = (
    ("testament to", "direct evidence statement"),
    ("delve into", "examine"),
    ("it is crucial to note", ""),
    ("it is important to note", ""),
    ("moreover", ""),
    ("furthermore", ""),
)

_CITATION_TOKEN_RE = re.compile(r"\[(?:\d+|\d+\s*,\s*\d+)\]")


def _citation_tokens(text: str) -> list[str]:
    return _CITATION_TOKEN_RE.findall(text)


def _ensure_citation_integrity(original: str, transformed: str) -> None:
    """Reject rewriting outputs that lose existing citation markers."""
    original_tokens = _citation_tokens(original)
    if not original_tokens:
        return

    transformed_tokens = _citation_tokens(transformed)
    missing = sorted(set(original_tokens) - set(transformed_tokens))
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Citation integrity check failed; rewrite removed citation markers: "
                + ", ".join(missing)
            ),
        )


def _count_replacements(before: str, after: str) -> int:
    """Approximate replacement count for deterministic rewrite reporting."""
    if before == after:
        return 0
    return abs(len(before) - len(after))


@functools.lru_cache(maxsize=1)
def _compiled_slop_rules() -> list[tuple[re.Pattern[str], str]]:
    """Build deterministic slop-substitution rules from ai_patterns.yml."""
    config = _load_patterns()
    rule_pairs: list[tuple[re.Pattern[str], str]] = []
    replacement_hints: tuple[tuple[str, str], ...] = (
        ("delve", "analyse"),
        ("testament", "evidence"),
        ("underscore", "shows"),
        ("it is important to note", ""),
        ("it is crucial to note", ""),
        ("this essay will", ""),
        ("the purpose of this", ""),
    )

    for section in ("phrases", "ngram_phrases"):
        section_obj = config.get(section)
        if not isinstance(section_obj, list):
            continue
        for raw_rule in section_obj:
            if not isinstance(raw_rule, dict):
                continue
            pattern_obj = raw_rule.get("pattern")
            if not isinstance(pattern_obj, str):
                continue

            replacement = ""
            lowered = pattern_obj.lower()
            for marker, candidate in replacement_hints:
                if marker in lowered:
                    replacement = candidate
                    break

            try:
                compiled = re.compile(pattern_obj, re.IGNORECASE)
            except re.error:
                continue
            rule_pairs.append((compiled, replacement))

    return rule_pairs


def _token_diversity(text: str) -> float:
    tokens = [token for token in re.findall(r"[A-Za-z']+", text.lower()) if token]
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def _sentence_burstiness(text: str) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    lengths = [len(re.findall(r"[A-Za-z']+", sentence)) for sentence in sentences]
    if len(lengths) < 4:
        return 0.0
    mean_len = statistics.mean(lengths)
    return round((statistics.stdev(lengths) / mean_len) if mean_len > 0 else 0.0, 4)


def _stylometric_hotspots(text: str) -> list[StylometricHotspot]:
    hotspots: list[StylometricHotspot] = []
    lower = text.lower()
    for phrase, replacement in _KNOWN_FILLER_PHRASES:
        start = 0
        while True:
            idx = lower.find(phrase, start)
            if idx < 0:
                break
            end = idx + len(phrase)
            hotspots.append(
                StylometricHotspot(
                    start=idx,
                    end=end,
                    text=text[idx:end],
                    reason="Known high-frequency generative filler phrase",
                    replacement_hint=replacement,
                )
            )
            start = end
    return hotspots


def _apply_stylometric_scrub(text: str) -> tuple[str, list[ScrubChange]]:
    scrubbed = text
    replacements: list[ScrubChange] = []

    for pattern, replacement in _compiled_slop_rules():
        if pattern.search(scrubbed) is None:
            continue
        scrubbed = pattern.sub(replacement, scrubbed)
        replacements.append(
            ScrubChange(
                original=pattern.pattern,
                replacement=replacement,
                reason="Config-driven slop phrase sanitisation",
            )
        )

    for phrase, replacement in _KNOWN_FILLER_PHRASES:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        if pattern.search(scrubbed) is None:
            continue
        new_text = pattern.sub(replacement, scrubbed)
        replacements.append(
            ScrubChange(
                original=phrase,
                replacement=replacement,
                reason="Deterministic filler suppression",
            )
        )
        scrubbed = re.sub(r"\s{2,}", " ", new_text)
    return scrubbed.strip(), replacements


def _de_symmetrise_lists(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    buffer: list[str] = []

    def _flush_buffer() -> None:
        nonlocal buffer
        if len(buffer) >= 3:
            prose = "; ".join(item.strip("- *") for item in buffer if item.strip())
            out.append(f"{prose}.")
        else:
            out.extend(buffer)
        buffer = []

    for line in lines:
        if re.match(r"^\s*[-*+]\s+", line):
            buffer.append(line)
            continue
        if buffer:
            _flush_buffer()
        out.append(line)
    if buffer:
        _flush_buffer()
    return "\n".join(out)


def _balance_paragraph_lengths(text: str) -> str:
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    adjusted: list[str] = []
    for para in paragraphs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        lengths = [len(re.findall(r"[A-Za-z']+", sentence)) for sentence in sentences]
        if len(lengths) >= 3:
            mean_len = statistics.mean(lengths)
            stdev = statistics.stdev(lengths)
            if mean_len > 0 and (stdev / mean_len) < 0.22:
                mutated: list[str] = []
                for sentence in sentences:
                    words = len(re.findall(r"[A-Za-z']+", sentence))
                    if words > 28 and "," in sentence:
                        left, right = sentence.split(",", 1)
                        mutated.append(f"{left.strip()}.")
                        mutated.append(right.strip())
                    else:
                        mutated.append(sentence)
                adjusted.append(" ".join(mutated))
                continue
        adjusted.append(para)
    return "\n\n".join(adjusted)


def _build_docx_bytes(text: str, title: str) -> bytes:
    try:
        from docx import Document
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="python-docx is required for DOCX export",
        ) from exc

    doc = Document()
    doc.add_heading(title, level=1)
    for paragraph in re.split(r"\n{2,}", text):
        if paragraph.strip():
            doc.add_paragraph(paragraph.strip())
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _build_pdf_bytes(text: str, title: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="reportlab is required for PDF export",
        ) from exc

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 48
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(48, y, title)
    y -= 28
    pdf.setFont("Helvetica", 10)
    for paragraph in re.split(r"\n{2,}", text):
        for line in paragraph.strip().splitlines() or [paragraph.strip()]:
            if y < 60:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 48
            pdf.drawString(48, y, line[:120])
            y -= 14
        y -= 8
    pdf.save()
    return buffer.getvalue()


@router.post("/check")
async def check_patterns(
    text: Annotated[str, Body(embed=True, max_length=50_000)],
    _user: Annotated[User, Depends(get_current_user)],
) -> CheckResponse:
    """Scan *text* for AI vocabulary and phrase patterns.

    The engine applies every rule from ``ai_patterns.yml`` using
    ``re.finditer`` with ``re.IGNORECASE``. Overlapping matches are resolved
    by keeping the longest span.

    The AI-signal score is a simple weighted sum normalised to 0–100:
      - Each 'high' flag contributes 3 points
      - Each 'medium' flag contributes 2 points
      - Each 'low' flag contributes 1 point
    Score is capped at 100 and scaled relative to a 1,000-word document.
    """
    _load_patterns()

    # Find all non-overlapping matches, longest wins on overlap
    spans: list[tuple[int, int, PatternRule]] = []
    for pattern, rule in _compiled_pattern_rules:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), rule))

    # Resolve overlaps: sort by start, keep longest
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    resolved: list[tuple[int, int, PatternRule]] = []
    cursor = 0
    for start, end, rule in spans:
        if start >= cursor:
            resolved.append((start, end, rule))
            cursor = end

    flags = [
        PatternFlag(
            start=s,
            end=e,
            matched_text=text[s:e],
            category=rule.get("category", "unknown"),
            message=rule.get("message", ""),
            suggestion=rule.get("suggestion", ""),
            severity=rule.get("severity", "low"),
        )
        for s, e, rule in resolved
    ]

    # Summary by category
    summary: dict[str, int] = {}
    weight_sum = 0
    high_count = 0
    for flag in flags:
        summary[flag.category] = summary.get(flag.category, 0) + 1
        weight_sum += _SEVERITY_WEIGHT.get(flag.severity, 1)
        if flag.severity == "high":
            high_count += 1

    # Normalise score to word count (1000 words = 100% scale)
    word_count = max(1, len(text.split()))
    normalised = (weight_sum / word_count) * 1_000
    score = min(100, round(normalised))

    # Slop density: high-severity flags per 100 words
    slop_density = round((high_count / word_count) * 100, 2)

    # Burstiness: coefficient of variation of sentence lengths
    # Low CV = suspiciously uniform sentence lengths (AI tell)
    sentence_lengths = [len(s.split()) for s in _split_sentences_safe(text)]
    if len(sentence_lengths) >= 4:
        sl_mean = statistics.mean(sentence_lengths)
        sl_cv = (statistics.stdev(sentence_lengths) / sl_mean) if sl_mean > 0 else 1.0
    else:
        sl_cv = 1.0  # not enough sentences to measure

    return CheckResponse(
        flags=flags,
        score=score,
        summary=summary,
        slop_density=slop_density,
        burstiness=round(sl_cv, 3),
    )


@router.post("/stylometric-profile", response_model=StylometricProfileResponse)
async def stylometric_profile(
    text: Annotated[str, Body(embed=True, max_length=60_000)],
    _user: Annotated[User, Depends(get_current_user)],
) -> StylometricProfileResponse:
    """Generate deterministic stylometric profile signals for draft prose."""
    hotspots = _stylometric_hotspots(text)
    word_count = max(1, len(re.findall(r"[A-Za-z']+", text)))
    filler_density = round((len(hotspots) / word_count) * 100, 3)
    return StylometricProfileResponse(
        token_diversity=_token_diversity(text),
        burstiness=_sentence_burstiness(text),
        filler_density=filler_density,
        hotspots=hotspots,
    )


@router.post("/stylometric-scrub", response_model=StylometricScrubResponse)
async def stylometric_scrub(
    text: Annotated[str, Body(embed=True, max_length=60_000)],
    _user: Annotated[User, Depends(get_current_user)],
) -> StylometricScrubResponse:
    """Apply deterministic filler suppression and low-entropy highlighting."""
    scrubbed, replacements = _apply_stylometric_scrub(text)
    low_entropy = (
        _token_diversity(scrubbed) < 0.38 or _sentence_burstiness(scrubbed) < 0.22
    )
    return StylometricScrubResponse(
        scrubbed_text=scrubbed,
        replacements=replacements,
        low_entropy_detected=low_entropy,
    )


@router.post("/telemetry", response_model=WriteTelemetryResponse)
async def write_telemetry(
    body: WriteTelemetryRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> WriteTelemetryResponse:
    """Accept editor keystroke telemetry for asynchronous profile updates."""
    payload_events = [event.model_dump() for event in body.events]
    accepted = await append_telemetry_events(user_id=user.id, events=payload_events)
    return WriteTelemetryResponse(accepted=accepted)


@router.get("/calibrate", response_model=StyleCalibrationQuizResponse)
async def get_calibration_quiz(
    _user: Annotated[User, Depends(get_current_user)],
) -> StyleCalibrationQuizResponse:
    """Return deterministic contrast pairs for binary style calibration."""
    pairs = _STYLE_CALIBRATION_ENGINE.list_pairs()
    return StyleCalibrationQuizResponse(
        total_pairs=len(pairs),
        pairs=[
            StyleCalibrationPairResponse(
                pair_id=item.pair_id,
                sequence_index=index,
                metric_target=item.metric_target,
                concept=item.concept,
                cadence_a_text=item.cadence_a_text,
                cadence_b_text=item.cadence_b_text,
            )
            for index, item in enumerate(pairs, start=1)
        ],
    )


@router.post("/calibrate", response_model=StyleCalibrationResponse)
async def submit_calibration(
    body: StyleCalibrationRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> StyleCalibrationResponse:
    """Compute and optionally persist initial style baseline from binary selections."""
    known_pair_ids = {pair.pair_id for pair in _STYLE_CALIBRATION_ENGINE.list_pairs()}
    valid_selection_count = sum(
        1 for item in body.selections if item.pair_id in known_pair_ids
    )
    if valid_selection_count == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one selection must reference a valid calibration pair",
        )

    profile_weights = _STYLE_CALIBRATION_ENGINE.calculate_initial_profile(
        [item.model_dump() for item in body.selections]
    )

    if body.commit:
        try:
            await _STYLE_CALIBRATION_ENGINE.commit_calibrated_profile(
                db,
                user_id=user.id,
                profile_weights=profile_weights,
            )
            await db.commit()
        except SQLAlchemyError as exc:
            await db.rollback()
            raise _map_database_failure(
                operation="Style calibration persistence",
                exc=exc,
            ) from exc

    return StyleCalibrationResponse(
        profile=StyleCalibrationProfile(**profile_weights),
        valid_selection_count=valid_selection_count,
        committed=body.commit,
    )


def _machine_cadence_drift_score(
    *,
    lexical_density_ttr: float,
    sentence_length_variance: float,
    passive_voice_ratio: float,
) -> float:
    """Return normalized machine-cadence drift score against postgraduate bounds."""
    ttr_penalty = max(0.0, 0.52 - lexical_density_ttr) * 1.8
    variance_penalty = max(0.0, 5.5 - sentence_length_variance) * 0.08
    passive_penalty = max(0.0, passive_voice_ratio - 0.18) * 1.6
    score = ttr_penalty + variance_penalty + passive_penalty
    return round(min(1.0, score), 4)


async def _embed_outline_text(text: str) -> list[float]:
    """Embed outline text for inverse coverage analysis."""
    from deeptutor.config import get_settings

    settings = get_settings()
    request: dict[str, object] = {
        "input": [text],
        "model": resolve_embedding_model(),
    }
    if should_send_embedding_dimensions():
        request["dimensions"] = settings.embedding_dimensions

    client = get_embedding_client()
    try:
        response = await client.embeddings.create(**request)
    except _PROVIDER_FAILURE_TYPES as exc:
        raise _map_provider_failure(operation="Outline embedding", exc=exc) from exc
    return response.data[0].embedding


async def _load_unit_syllabus_chunks(
    db: AsyncSession,
    *,
    unit_id: str,
    max_chunks: int,
) -> list[dict[str, object]]:
    """Load syllabus/theory chunks with embeddings from a unit scope."""
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.scope == "unit")
        .where(DocumentChunk.scope_id == unit_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .limit(max_chunks)
    )
    rows = (await db.execute(stmt)).scalars().all()

    chunks: list[dict[str, object]] = []
    for row in rows:
        embedding_obj = row.embedding
        if not isinstance(embedding_obj, list):
            continue
        if not all(isinstance(value, (int, float)) for value in embedding_obj):
            continue

        theory = row.filename.rsplit(".", 1)[0] if row.filename else "unknown"
        chunks.append(
            {
                "id": row.id,
                "module": row.category or "course_context",
                "theory": theory,
                "embedding": [float(value) for value in embedding_obj],
            }
        )
    return chunks


@router.post("/stylometric-drift", response_model=StylometricDriftResponse)
async def stylometric_drift(
    body: StylometricDriftRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> StylometricDriftResponse:
    """Aggregate submissions chronologically and detect stylometric drift."""
    ordered_submissions = sorted(body.submissions, key=lambda item: item.submitted_at)
    points: list[StylometricDriftPoint] = []
    alerts: list[StylometricDriftAlert] = []

    for item in ordered_submissions:
        prose_segments, _feedback_segments = extract_student_prose_segments(item.text)
        prose_texts = [segment.text for segment in prose_segments] or [item.text]
        profile = analyse_historical_drift(prose_texts)
        drift_score = _machine_cadence_drift_score(
            lexical_density_ttr=profile.lexical_density_ttr,
            sentence_length_variance=profile.sentence_length_variance,
            passive_voice_ratio=profile.passive_voice_ratio,
        )

        points.append(
            StylometricDriftPoint(
                submission_id=item.submission_id,
                semester=item.semester,
                submitted_at=item.submitted_at,
                lexical_density_ttr=profile.lexical_density_ttr,
                mean_sentence_length=profile.mean_sentence_length,
                sentence_length_variance=profile.sentence_length_variance,
                passive_voice_ratio=profile.passive_voice_ratio,
                drift_score=drift_score,
                prose_segments=len(prose_segments),
            )
        )

        if drift_score >= 0.45 or profile.passive_voice_ratio > 0.28:
            alerts.append(
                StylometricDriftAlert(
                    submission_id=item.submission_id,
                    semester=item.semester,
                    severity="high",
                    message=(
                        "Elevated machine-cadence risk: low lexical diversity and "
                        "high passive ratio relative to postgraduate baseline."
                    ),
                )
            )
        elif drift_score >= 0.25:
            alerts.append(
                StylometricDriftAlert(
                    submission_id=item.submission_id,
                    semester=item.semester,
                    severity="medium",
                    message="Moderate stylistic drift detected against postgraduate bounds.",
                )
            )

    return StylometricDriftResponse(
        points=points,
        alerts=alerts,
        series=StylometricDriftSeries(
            submitted_at=[point.submitted_at for point in points],
            lexical_density_ttr=[point.lexical_density_ttr for point in points],
            sentence_length_variance=[
                point.sentence_length_variance for point in points
            ],
            passive_voice_ratio=[point.passive_voice_ratio for point in points],
            drift_score=[point.drift_score for point in points],
        ),
    )


@router.post("/prompt-retention-map", response_model=PromptRetentionResponse)
async def prompt_retention_map(
    body: PromptRetentionRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> PromptRetentionResponse:
    """Calculate lost-in-the-middle risk zones for current retrieval context."""
    placements = calculate_prompt_retention_map(
        body.system_preamble_tokens,
        [(item.chunk_id, item.filename, item.token_count) for item in body.chunks],
    )
    vulnerable_count = sum(
        1
        for placement in placements
        if placement.retention_risk_zone == "vulnerable_middle"
    )
    total_tokens = body.system_preamble_tokens + sum(
        item.token_count for item in body.chunks
    )

    return PromptRetentionResponse(
        total_tokens=total_tokens,
        vulnerable_middle_count=vulnerable_count,
        placements=[
            PromptRetentionPlacement(
                chunk_id=placement.chunk_id,
                filename=placement.filename,
                token_index_start=placement.token_index_start,
                relative_position_percentage=placement.relative_position_percentage,
                retention_risk_zone=placement.retention_risk_zone,
            )
            for placement in placements
        ],
    )


@router.post("/inverse-coverage", response_model=InverseCoverageResponse)
async def inverse_coverage(
    body: InverseCoverageRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
) -> InverseCoverageResponse:
    """Identify syllabus concepts weakly covered by an outline embedding."""
    outline_embedding = await _embed_outline_text(body.outline_text)

    syllabus_chunks: list[dict[str, object]]
    if body.syllabus_chunks:
        syllabus_chunks = [chunk.model_dump() for chunk in body.syllabus_chunks]
    elif body.unit_id:
        try:
            syllabus_chunks = await _load_unit_syllabus_chunks(
                db,
                unit_id=body.unit_id,
                max_chunks=body.max_chunks,
            )
        except SQLAlchemyError as exc:
            raise _map_database_failure(
                operation="Syllabus chunk loading",
                exc=exc,
            ) from exc
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="unit_id is required when syllabus_chunks are not supplied",
        )

    gaps = compute_inverse_coverage(
        outline_embedding,
        syllabus_chunks,
        threshold=body.threshold,
    )

    return InverseCoverageResponse(
        threshold=body.threshold,
        total_chunks_evaluated=len(syllabus_chunks),
        gaps=[
            InverseCoverageGap(
                module_name=gap.module_name,
                missing_theory_title=gap.missing_theory_title,
                associated_chunk_id=gap.associated_chunk_id,
                max_similarity_resolved=gap.max_similarity_resolved,
            )
            for gap in gaps
        ],
    )


@router.post("/stylometric-constraints", response_model=StylometricConstraintResponse)
async def stylometric_constraints(
    body: StylometricConstraintRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> StylometricConstraintResponse:
    """Compile profile-driven prompt constraints for style-aligned generation."""
    try:
        profile = await get_user_stylometric_profile(db, user.id)
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Stylometric profile lookup",
            exc=exc,
        ) from exc
    features = profile.features if profile is not None else {}
    compiler = StylometricPromptCompiler(features)
    instructions = compiler.compile_constrained_payload(body.outline_node)

    return StylometricConstraintResponse(
        instructions=instructions,
        profile_sample_count=profile.sample_count if profile is not None else 0,
    )


@router.post("/entropy-draft", response_model=EntropyDraftResponse)
async def entropy_draft(
    body: EntropyDraftRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> EntropyDraftResponse:
    """Build entropy-constrained drafting prompt and validate prose variance."""
    entropy_passed = (
        StructuralEntropyExtrapolator.verify_output_entropy(body.generated_prose)
        if body.generated_prose is not None
        else None
    )

    return EntropyDraftResponse(
        prompt=StructuralEntropyExtrapolator.append_entropy_instructions(body.notes),
        entropy_passed=entropy_passed,
    )


@router.post("/verify-topology", response_model=SyntacticTopologyResponse)
async def verify_topology(
    body: SyntacticTopologyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> SyntacticTopologyResponse:
    """Evaluate syntactic tree topology against optional user baseline."""
    segments = [para.strip() for para in body.text.split("\n\n") if para.strip()]
    topology = SyntacticTopologyEvaluator.extract_topology(segments)

    try:
        profile = await get_user_stylometric_profile(db, user.id)
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Topology baseline lookup",
            exc=exc,
        ) from exc
    baseline_available = profile is not None and profile.features
    within_baseline = None

    if baseline_available and profile.features:
        baseline_depth = profile.features.get("mean_dependency_depth", 2.0)
        baseline_branching = profile.features.get("branching_coefficient", 1.5)
        depth_variance = abs(topology.mean_dependency_depth - baseline_depth)
        branching_variance = abs(topology.branching_coefficient - baseline_branching)
        within_baseline = (depth_variance < 1.0) and (branching_variance < 0.5)

    return SyntacticTopologyResponse(
        mean_dependency_depth=topology.mean_dependency_depth,
        branching_coefficient=topology.branching_coefficient,
        clause_count_variance=topology.clause_count_variance,
        baseline_available=baseline_available,
        within_baseline=within_baseline,
    )


@router.post("/equalise-vocabulary", response_model=LexicalEqualiseResponse)
async def equalise_vocabulary(
    body: LexicalEqualiseRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> LexicalEqualiseResponse:
    """Replace out-of-distribution vocabulary with direct alternatives."""
    if not body.substitutions:
        return LexicalEqualiseResponse(text=body.text, replacements_applied=0)

    equaliser = LexicalEntropyEqualiser({}, body.substitutions)
    processed = equaliser.normalise_vocabulary_cadence(body.text)
    replacements_count = _count_replacements(body.text, processed)

    return LexicalEqualiseResponse(
        text=processed,
        replacements_applied=replacements_count,
    )


@router.post("/validate-burstiness", response_model=BurstinessWindowResponse)
async def validate_burstiness(
    body: BurstinessWindowRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> BurstinessWindowResponse:
    """Evaluate sliding-window sentence-length variance across paragraphs."""
    paragraphs = [p.strip() for p in body.text.split("\n\n") if p.strip()]
    controller = SlidingWindowBurstinessController(body.minimum_variance)
    results: list[BurstinessWindowResult] = []

    for idx, para in enumerate(paragraphs):
        passed = controller.verify_cadence_entropy(para)
        results.append(
            BurstinessWindowResult(
                start_paragraph_index=idx,
                passed=passed,
            )
        )

    all_passed = all(r.passed for r in results)
    return BurstinessWindowResponse(windows=results, all_passed=all_passed)


@router.post("/provenance/append", response_model=ProvenanceEventResponse)
async def append_provenance_event(
    body: ProvenanceEventRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> ProvenanceEventResponse:
    """Append one mutation receipt to the user's immutable provenance ledger."""
    try:
        receipt = await append_mutation_receipt(
            db,
            user_id=user.id,
            document_id=body.document_id,
            source_type=body.source_type,
            character_delta_count=body.character_delta_count,
        )
    except MutationReceiptAppendError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Unable to append provenance event due to concurrent write conflicts",
        ) from exc
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        raise _map_database_failure(
            operation="Provenance append persistence",
            exc=exc,
        ) from exc

    return ProvenanceEventResponse(
        id=receipt.id,
        sequence_index=receipt.sequence_index,
        signature_hash=receipt.signature_hash,
        previous_block_hash=receipt.previous_block_hash,
    )


@router.get("/provenance/ledger", response_model=ProvenanceLedgerResponse)
async def get_provenance_ledger(
    document_id: Annotated[str, Query(min_length=1, max_length=64)],
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> ProvenanceLedgerResponse:
    """Return immutable mutation history for one document."""
    try:
        receipts = await list_mutation_receipts(
            db,
            user_id=user.id,
            document_id=document_id,
            limit=200,
        )
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Provenance ledger lookup",
            exc=exc,
        ) from exc

    return ProvenanceLedgerResponse(
        document_id=document_id,
        items=[
            ProvenanceLedgerItem(
                id=r.id,
                sequence_index=r.sequence_index,
                source_type=r.source_type,
                character_delta_count=r.character_delta_count,
                timestamp_epoch=r.timestamp_epoch,
                previous_block_hash=r.previous_block_hash,
                signature_hash=r.signature_hash,
            )
            for r in receipts
        ],
    )


@router.get("/provenance/verify", response_model=ProvenanceVerifyResponse)
async def verify_provenance_ledger(
    document_id: Annotated[str, Query(min_length=1, max_length=64)],
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> ProvenanceVerifyResponse:
    """Verify hash-link integrity and sequence continuity for one document ledger."""
    try:
        result = await verify_mutation_chain(
            db,
            user_id=user.id,
            document_id=document_id,
            limit=2_000,
        )
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Provenance chain verification",
            exc=exc,
        ) from exc

    return ProvenanceVerifyResponse(
        document_id=document_id,
        checked_count=result.checked_count,
        is_valid=result.is_valid,
        first_invalid_sequence_index=result.first_invalid_sequence_index,
        reason=result.reason,
    )


# ---------------------------------------------------------------------------
# POST /api/write/analyse-structure
# ---------------------------------------------------------------------------


@router.post("/analyse-structure")
async def analyse_structure(
    text: Annotated[str, Body(embed=True, max_length=50_000)],
    _user: Annotated[User, Depends(get_current_user)],
) -> StructureResponse:
    """Detect structural AI tells that require multi-paragraph context.

    Checks performed
    ----------------
    1. Transitional overload — paragraph where ≥ 3 sentences start with listed adverbs
    2. Uniform paragraph length — coefficient of variation (CV) of word counts;
       CV < 0.20 flags the text as suspiciously uniform (typical LLM output)
    3. 'Further research' escape hatch — conclusion paragraph that punts
    4. Thesis absence — opening paragraph that contains no arguable claim
       (heuristic: no hedging verbs like 'argue', 'contend', 'demonstrate', etc.)
    """
    config = _load_patterns()
    transition_cfg_obj = config.get("transitional_overload", {})
    transition_cfg = transition_cfg_obj if isinstance(transition_cfg_obj, dict) else {}
    transition_terms_obj = transition_cfg.get("terms", [])
    transition_terms = [item for item in transition_terms_obj if isinstance(item, str)]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    issues: list[StructureIssue] = []

    word_counts: list[int] = []
    for i, para in enumerate(paragraphs):
        wc = len(para.split())
        word_counts.append(wc)

        # --- Transitional overload ---
        sentences = _split_sentences_safe(para)
        trans_hits = sum(
            1
            for s in sentences
            if any(s.strip().startswith(t) for t in transition_terms)
        )
        if trans_hits >= 3:
            issues.append(
                StructureIssue(
                    issue_type="transitional_overload",
                    message=f"Paragraph {i + 1}: {trans_hits} sentences open with conjunctive adverbs.",
                    suggestion=(
                        "Vary sentence openings. Transitional overload signals mechanical "
                        "argument construction rather than genuine logical flow."
                    ),
                    paragraph_index=i,
                )
            )

        # --- 'Further research needed' escape hatch ---
        if re.search(
            r"further (?:research|study|investigation) (?:is )?(?:needed|required|warranted)",
            para,
            re.IGNORECASE,
        ):
            issues.append(
                StructureIssue(
                    issue_type="further_research_escape",
                    message=f"Paragraph {i + 1}: 'further research needed' conclusion detected.",
                    suggestion=(
                        "Draw a concrete conclusion from your analysis. "
                        "Deferring to future research signals failure to commit to an argument."
                    ),
                    paragraph_index=i,
                )
            )

    # --- Uniform paragraph length (CV) ---
    avg_wc = statistics.mean(word_counts) if word_counts else 0.0
    cv = (
        (statistics.stdev(word_counts) / avg_wc)
        if len(word_counts) >= 3 and avg_wc > 0
        else 1.0
    )
    if cv < 0.20 and len(paragraphs) >= 4:
        issues.append(
            StructureIssue(
                issue_type="uniform_paragraph_length",
                message=f"Paragraph lengths are suspiciously uniform (CV={cv:.2f}). All paragraphs are ~{round(avg_wc)} words.",
                suggestion=(
                    "Vary paragraph depth. Complex arguments warrant longer development; "
                    "minor points can be brief. Mechanical uniformity is a strong AI signal."
                ),
                paragraph_index=None,
            )
        )

    # --- Thesis absence in opening paragraph ---
    if paragraphs:
        opening = paragraphs[0]
        arguable_verbs = r"\b(?:argue|contend|demonstrate|propose|challenge|critique|assert|show|claim)\b"
        if (
            not re.search(arguable_verbs, opening, re.IGNORECASE)
            and len(opening.split()) > 40
        ):
            issues.append(
                StructureIssue(
                    issue_type="thesis_absence",
                    message="Opening paragraph may lack a clear arguable thesis.",
                    suggestion=(
                        "Introduce a specific, contestable claim in the opening. "
                        "Summaries of background context without an argument are a structural tell."
                    ),
                    paragraph_index=0,
                )
            )

    # Burstiness: sentence-length coefficient of variation (computed once)
    _sent_splits = _split_sentences_safe(text)
    _sent_lengths = [len(s.split()) for s in _sent_splits]
    if len(_sent_lengths) >= 4:
        _sl_mean = statistics.mean(_sent_lengths)
        _sentence_length_cv = (
            statistics.stdev(_sent_lengths) / _sl_mean if _sl_mean > 0 else 1.0
        )
    else:
        _sentence_length_cv = 1.0

    return StructureResponse(
        issues=issues,
        paragraph_count=len(paragraphs),
        avg_word_count=round(avg_wc, 1),
        word_count_cv=round(cv, 3),
        sentence_length_cv=round(_sentence_length_cv, 3),
    )


# ---------------------------------------------------------------------------
# POST /api/write/scrub  — LLM-as-a-Judge rewrite pass
# ---------------------------------------------------------------------------


@router.post("/scrub")
async def scrub_writing(
    text: Annotated[str, Body(embed=True, max_length=20_000)],
    _user: Annotated[User, Depends(get_current_user)],
) -> ScrubResponse:
    """LLM-as-a-Judge rewrite pass that removes AI structural tells.

    This is a **separate** LLM call from the generation model, specialised
    purely in pattern removal.  It operates under strict constraints:
    - Does not add content or alter the writer's argument
    - Applies the six rules from ``WRITE_SCRUB_SYSTEM`` deterministically (temp=0.1)
    - Returns structured JSON so parsing is reliable

    The separation of concerns between the generation model and this scrubbing
    model means each can be optimised independently (e.g. use a faster/cheaper
    model here as the task is purely transformational, not creative).

    Args:
        text: Draft text to scrub (max 20,000 chars).

    Returns:
        :class:`ScrubResponse` with the rewritten text and a changelog.
    """
    client = _get_client()

    try:
        resp = await client.chat.completions.create(
            model=resolve_llm_model(),
            messages=[
                {"role": "system", "content": WRITE_SCRUB_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0.1,  # near-deterministic for consistent rule application
            max_tokens=4_000,
            response_format={"type": "json_object"},
        )
    except _PROVIDER_FAILURE_TYPES as exc:
        raise _map_provider_failure(operation="Scrub rewrite", exc=exc) from exc

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Scrub endpoint returned unparseable JSON — returning original text"
        )
        return ScrubResponse(scrubbed_text=text, changes=[], changes_count=0)

    scrubbed = data.get("scrubbed_text") or text
    raw_changes_obj = data.get("changes") or []
    raw_changes = (
        cast(list[ScrubChangePayload], raw_changes_obj)
        if isinstance(raw_changes_obj, list)
        else []
    )

    changes = [
        ScrubChange(
            original=c.get("original", ""),
            replacement=c.get("replacement", ""),
            reason=c.get("reason", ""),
        )
        for c in raw_changes
        if isinstance(c, dict)
    ]

    return ScrubResponse(
        scrubbed_text=scrubbed, changes=changes, changes_count=len(changes)
    )


@router.post("/export-draft", response_model=ExportDraftResponse)
async def export_draft(
    body: ExportDraftRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> ExportDraftResponse:
    """Export draft text with de-symmetrised layout and balanced structure."""
    processed = _de_symmetrise_lists(body.text)
    processed = _balance_paragraph_lengths(processed)
    processed, _ = _apply_stylometric_scrub(processed)

    style_header = "APA 7" if body.citation_style == "apa7" else "Harvard"
    processed = f"Citation style: {style_header}\n\n{processed}"

    if body.file_format == "md":
        content = processed.encode("utf-8")
        mime = "text/markdown"
        filename = f"{body.title.strip().replace(' ', '_')}.md"
    elif body.file_format == "docx":
        content = _build_docx_bytes(processed, body.title)
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        filename = f"{body.title.strip().replace(' ', '_')}.docx"
    else:
        content = _build_pdf_bytes(processed, body.title)
        mime = "application/pdf"
        filename = f"{body.title.strip().replace(' ', '_')}.pdf"

    return ExportDraftResponse(
        filename=filename,
        mime_type=mime,
        content_base64=base64.b64encode(content).decode("ascii"),
        preview_text=processed[:1200],
    )


# ---------------------------------------------------------------------------
# POST /api/write/analyse-rubric  — Criterion-by-criterion marking
# ---------------------------------------------------------------------------

_RUBRIC_TEMPLATES: dict[str, RubricTemplate] = {
    "standard": RubricTemplate(
        id="standard",
        name="Standard Academic",
        criteria=[
            RubricCriterionTemplate(
                key="thesis_strength",
                label="Thesis Strength",
                max_score=10,
                weight=1 / 6,
                description="Clear, contestable claim that frames the essay.",
            ),
            RubricCriterionTemplate(
                key="evidence_quality",
                label="Evidence Quality",
                max_score=10,
                weight=1 / 6,
                description="Credible support, examples, and source precision.",
            ),
            RubricCriterionTemplate(
                key="argument_clarity",
                label="Argument Clarity",
                max_score=10,
                weight=1 / 6,
                description="Reasoning chain from claim to evidence to inference.",
            ),
            RubricCriterionTemplate(
                key="structure",
                label="Structure & Organization",
                max_score=10,
                weight=1 / 6,
                description="Paragraph sequence and coherence against thesis.",
            ),
            RubricCriterionTemplate(
                key="expression",
                label="Expression & Style",
                max_score=10,
                weight=1 / 6,
                description="Precision, readability, and low hedge density.",
            ),
            RubricCriterionTemplate(
                key="engagement",
                label="Engagement with Sources",
                max_score=10,
                weight=1 / 6,
                description="Critical use of sources instead of summary-only reporting.",
            ),
        ],
    ),
    "evidence_heavy": RubricTemplate(
        id="evidence_heavy",
        name="Evidence-Focused",
        criteria=[
            RubricCriterionTemplate(
                key="thesis_strength",
                label="Thesis Strength",
                max_score=10,
                weight=0.15,
                description="Clear and specific claim.",
            ),
            RubricCriterionTemplate(
                key="evidence_quality",
                label="Evidence Quality",
                max_score=10,
                weight=0.3,
                description="Quality, specificity, and relevance of evidence.",
            ),
            RubricCriterionTemplate(
                key="argument_clarity",
                label="Argument Clarity",
                max_score=10,
                weight=0.2,
                description="Logical progression and inference quality.",
            ),
            RubricCriterionTemplate(
                key="structure",
                label="Structure & Organization",
                max_score=10,
                weight=0.1,
                description="Coherence of paragraph structure.",
            ),
            RubricCriterionTemplate(
                key="expression",
                label="Expression & Style",
                max_score=10,
                weight=0.1,
                description="Language precision and sentence control.",
            ),
            RubricCriterionTemplate(
                key="engagement",
                label="Engagement with Sources",
                max_score=10,
                weight=0.15,
                description="Critical positioning relative to source claims.",
            ),
        ],
    ),
    "critical_engagement": RubricTemplate(
        id="critical_engagement",
        name="Critical Engagement",
        criteria=[
            RubricCriterionTemplate(
                key="thesis_strength",
                label="Thesis Strength",
                max_score=10,
                weight=0.15,
                description="Focused claim and position.",
            ),
            RubricCriterionTemplate(
                key="evidence_quality",
                label="Evidence Quality",
                max_score=10,
                weight=0.2,
                description="Depth and reliability of support.",
            ),
            RubricCriterionTemplate(
                key="argument_clarity",
                label="Argument Clarity",
                max_score=10,
                weight=0.2,
                description="Reasoning and rebuttal quality.",
            ),
            RubricCriterionTemplate(
                key="structure",
                label="Structure & Organization",
                max_score=10,
                weight=0.1,
                description="Section and paragraph coherence.",
            ),
            RubricCriterionTemplate(
                key="expression",
                label="Expression & Style",
                max_score=10,
                weight=0.1,
                description="Clear expression with discipline-appropriate language.",
            ),
            RubricCriterionTemplate(
                key="engagement",
                label="Engagement with Sources",
                max_score=10,
                weight=0.25,
                description="Comparison, critique, and synthesis of source positions.",
            ),
        ],
    ),
}


def _resolve_template(
    template_id: str,
    custom_template: RubricTemplate | None,
) -> RubricTemplate:
    """Resolve template id with optional caller-provided custom template."""
    template = (
        custom_template
        if custom_template is not None
        else _RUBRIC_TEMPLATES.get(template_id)
    )
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown rubric template: {template_id}",
        )

    total_weight = sum(criterion.weight for criterion in template.criteria)
    if abs(total_weight - 1.0) > 1e-6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Rubric template weights must sum to 1.0",
        )
    return template


def _build_rubric_system_prompt(template: RubricTemplate) -> str:
    """Build structured prompt from template criteria."""
    criteria_lines = []
    for criterion in template.criteria:
        criteria_lines.append(
            f"- key={criterion.key}; label={criterion.label}; max_score={criterion.max_score}; "
            f"weight={criterion.weight:.3f}; guidance={criterion.description}"
        )

    return (
        "You are an academic assessor. Score the essay against each criterion and return JSON only.\n\n"
        "Criteria:\n" + "\n".join(criteria_lines) + "\n\n"
        "Return exactly this JSON schema:\n"
        "{\n"
        '  "criteria": [\n'
        '    {"key": "...", "score": <number>, "feedback": "...", "evidence": "..."}\n'
        "  ],\n"
        '  "assessor_notes": "2-3 sentences"\n'
        "}\n"
        "Do not include markdown fences."
    )


def _build_adversarial_system_prompt(
    template: RubricTemplate, *, agent: str, focus: str
) -> str:
    """Build strict marker simulation prompt for one persona."""
    rubric_lines = [
        f"- {criterion.label}: max={criterion.max_score}, weight={criterion.weight:.3f}, guidance={criterion.description}"
        for criterion in template.criteria
    ]
    return (
        "You are a strict academic marker. Return JSON only with concise findings.\n"
        f"Marker persona: {agent}. Focus area: {focus}.\n"
        "Rubric criteria:\n" + "\n".join(rubric_lines) + "\n"
        "Return schema:\n"
        "{\n"
        '  "findings": [{"issue": "...", "deduction_reason": "...", "severity": "low|medium|high"}],\n'
        '  "priority_actions": ["...", "..."]\n'
        "}\n"
        "Focus on concrete deduction triggers, not generic writing advice."
    )


async def _run_adversarial_agent(
    *,
    text: str,
    template: RubricTemplate,
    agent: str,
    focus: str,
) -> AdversarialAgentReport:
    """Run one strict marker persona and return structured findings."""
    client = _get_client()
    try:
        resp = await client.chat.completions.create(
            model=resolve_llm_model(),
            messages=[
                {
                    "role": "system",
                    "content": _build_adversarial_system_prompt(
                        template,
                        agent=agent,
                        focus=focus,
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=1_200,
            response_format={"type": "json_object"},
        )
    except _PROVIDER_FAILURE_TYPES as exc:
        raise _map_provider_failure(operation="Adversarial marking", exc=exc) from exc

    raw = resp.choices[0].message.content or "{}"
    findings: list[AdversarialFinding] = []
    priority_actions: list[str] = []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}

    findings_obj = payload.get("findings") if isinstance(payload, dict) else None
    if isinstance(findings_obj, list):
        for item in findings_obj:
            if not isinstance(item, dict):
                continue
            issue_obj = item.get("issue")
            reason_obj = item.get("deduction_reason")
            severity_obj = item.get("severity")
            if not isinstance(issue_obj, str) or not isinstance(reason_obj, str):
                continue
            severity: Literal["low", "medium", "high"] = "medium"
            if severity_obj in {"low", "medium", "high"}:
                severity = severity_obj
            findings.append(
                AdversarialFinding(
                    issue=issue_obj,
                    deduction_reason=reason_obj,
                    severity=severity,
                )
            )

    actions_obj = payload.get("priority_actions") if isinstance(payload, dict) else None
    if isinstance(actions_obj, list):
        for action in actions_obj:
            if isinstance(action, str) and action.strip():
                priority_actions.append(action.strip())

    if not findings:
        findings.append(
            AdversarialFinding(
                issue="No structured findings returned",
                deduction_reason="Model response was empty or non-compliant JSON.",
                severity="medium",
            )
        )
    if not priority_actions:
        priority_actions = ["Revise weak sections and re-run adversarial simulation."]

    return AdversarialAgentReport(
        agent=agent,
        focus=focus,
        findings=findings,
        priority_actions=priority_actions,
    )


def _get_grade_descriptor(weighted_score: float) -> str:
    """Map a 0–100 weighted rubric score to a UK academic grade descriptor."""
    if weighted_score >= 90:
        return "First Class (90–100) — Exceptional; well-structured argument with strong evidence and critical engagement."
    elif weighted_score >= 80:
        return "Upper Second (80–89) — Very good; clear thesis, strong support, mostly natural prose."
    elif weighted_score >= 70:
        return "Lower Second (70–79) — Good; coherent argument with adequate evidence; minor clarity issues."
    elif weighted_score >= 60:
        return "Upper Third (60–69) — Acceptable; argument present but some gaps in evidence or clarity."
    elif weighted_score >= 50:
        return "Lower Third (50–59) — Pass; argument weak or unclear; evidence sparse."
    else:
        return "Fail (0–49) — Does not meet acceptable standard; no clear thesis or argument structure."


async def _analyse_essay_rubric(
    text: str,
    *,
    template: RubricTemplate,
) -> dict[str, object]:
    """Call LLM to score essay with a selected rubric template."""
    client = _get_client()
    try:
        resp = await client.chat.completions.create(
            model=resolve_llm_model(),
            messages=[
                {"role": "system", "content": _build_rubric_system_prompt(template)},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=2_000,
            response_format={"type": "json_object"},
        )
    except _PROVIDER_FAILURE_TYPES as exc:
        raise _map_provider_failure(operation="Rubric analysis", exc=exc) from exc

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Rubric endpoint returned unparseable JSON")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Rubric assessment provider returned invalid JSON",
        ) from exc
    return data


def _score_to_int(value: object, *, fallback: int) -> int:
    """Convert unknown score payload to int using safe fallback."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        if value.strip().isdigit():
            return int(value.strip())
    return fallback


def _suggestions_from_criteria(criteria: list[RubricCriterion]) -> list[str]:
    """Generate actionable suggestions sorted by lowest criterion score."""
    guidance = {
        "thesis_strength": "Rewrite opening so paragraph one states one contestable claim and scope.",
        "evidence_quality": "Add at least two source-backed claims with direct references.",
        "argument_clarity": "Add one inference sentence after each evidence statement.",
        "structure": "Reorder paragraphs so each section advances one argument step.",
        "expression": "Replace hedge phrases with direct statements and precise verbs.",
        "engagement": "Add one comparison or critique between source positions.",
    }
    weak = sorted(criteria, key=lambda item: item.score)[:3]
    suggestions: list[str] = []
    for criterion in weak:
        if criterion.score <= int(criterion.max_score * 0.7):
            suggestions.append(
                guidance.get(
                    criterion.key,
                    f"Improve {criterion.criterion} with specific revisions.",
                )
            )
    return suggestions


async def _analyse_rubric_internal(
    *,
    body: AnalyseRubricRequest,
    user: User,
    db: AsyncSession,
) -> AnalyseRubricResponse:
    """Run rubric analysis with optional history persistence."""
    template = _resolve_template(body.template_id, body.custom_template)

    logger.info(
        "Rubric analysis requested template=%s text_len=%d",
        template.id,
        len(body.text),
    )

    rubric_data = await _analyse_essay_rubric(body.text, template=template)
    criteria_payload = rubric_data.get("criteria", [])

    parsed_criteria: list[RubricCriterion] = []
    raw_total = 0
    raw_max = 0
    weighted_total = 0.0

    criterion_by_key = {criterion.key: criterion for criterion in template.criteria}
    criteria_by_payload_key: dict[str, dict[str, object]] = {}

    if isinstance(criteria_payload, list):
        for item in criteria_payload:
            if isinstance(item, dict):
                key_obj = item.get("key")
                if isinstance(key_obj, str):
                    criteria_by_payload_key[key_obj] = item

    for criterion in template.criteria:
        payload = criteria_by_payload_key.get(criterion.key, {})
        fallback_score = max(1, criterion.max_score // 2)
        score = _score_to_int(payload.get("score"), fallback=fallback_score)
        score = max(0, min(criterion.max_score, score))
        feedback_obj = payload.get("feedback")
        evidence_obj = payload.get("evidence")
        feedback = (
            feedback_obj if isinstance(feedback_obj, str) else "No feedback returned."
        )
        evidence = (
            evidence_obj if isinstance(evidence_obj, str) and evidence_obj else None
        )

        raw_total += score
        raw_max += criterion.max_score
        weighted_total += (score / criterion.max_score) * criterion.weight * 100.0

        parsed_criteria.append(
            RubricCriterion(
                key=criterion.key,
                criterion=criterion.label,
                max_score=criterion.max_score,
                score=score,
                weight=criterion.weight,
                feedback=feedback,
                evidence=evidence,
            )
        )

    assessor_notes_obj = rubric_data.get("assessor_notes")
    assessor_notes = (
        assessor_notes_obj
        if isinstance(assessor_notes_obj, str)
        else "No notes provided."
    )
    grade_descriptor = _get_grade_descriptor(weighted_total)

    improvement_suggestions = _suggestions_from_criteria(parsed_criteria)

    response = AnalyseRubricResponse(
        rubric=RubricScore(
            criteria=parsed_criteria,
            weighted_total=round(weighted_total, 1),
            raw_total=raw_total,
            raw_max=raw_max,
            assessor_notes=assessor_notes,
        ),
        grade_descriptor=grade_descriptor,
        template_id=template.id,
        improvement_suggestions=improvement_suggestions,
    )

    if body.save_history:
        criteria_records: list[dict[str, object]] = []
        for item in response.rubric.criteria:
            criteria_records.append(
                {
                    "key": item.key,
                    "label": item.criterion,
                    "score": item.score,
                    "max_score": item.max_score,
                    "weight": item.weight,
                    "feedback": item.feedback,
                    "evidence": item.evidence or "",
                }
            )

        record = await create_rubric_assessment(
            db,
            user_id=user.id,
            unit_id=body.unit_id,
            topic=body.topic,
            grade_level=body.grade_level,
            cohort=body.cohort,
            template_id=template.id,
            comparison_group_id=body.comparison_group_id,
            is_improved_version=body.is_improved_version,
            weighted_total=response.rubric.weighted_total,
            raw_total=response.rubric.raw_total,
            raw_max=response.rubric.raw_max,
            grade_descriptor=response.grade_descriptor,
            criteria=criteria_records,
            improvement_suggestions=response.improvement_suggestions,
            assessor_notes=response.rubric.assessor_notes,
        )
        response.assessment_id = record.id

    return response


@router.post("/analyse-rubric")
async def analyse_rubric(
    body: AnalyseRubricRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> AnalyseRubricResponse:
    """Analyse one draft against a selected rubric template."""
    try:
        return await _analyse_rubric_internal(body=body, user=user, db=db)
    except SQLAlchemyError as exc:
        await db.rollback()
        raise _map_database_failure(
            operation="Rubric analysis persistence",
            exc=exc,
        ) from exc


@router.post("/adversarial-marking", response_model=AdversarialMarkingResponse)
async def adversarial_marking_simulation(
    body: AdversarialMarkingRequest,
    _db: Annotated[AsyncSession, Depends(get_db)],
    _user: Annotated[User, Depends(get_current_user)],
) -> AdversarialMarkingResponse:
    """Run a strict three-marker simulation to expose likely deduction points."""
    template = _resolve_template(body.template_id, body.custom_template)

    agents = [
        ("Agent A", "Pedantic citation checker"),
        ("Agent B", "Theory purist"),
        ("Agent C", "Practical structure reviewer"),
    ]
    reports = await asyncio.gather(
        *[
            _run_adversarial_agent(
                text=body.text,
                template=template,
                agent=agent,
                focus=focus,
            )
            for agent, focus in agents
        ]
    )

    high_findings = sum(
        1
        for report in reports
        for finding in report.findings
        if finding.severity == "high"
    )
    medium_findings = sum(
        1
        for report in reports
        for finding in report.findings
        if finding.severity == "medium"
    )

    summary = (
        f"High-risk deductions: {high_findings}; medium-risk deductions: {medium_findings}. "
        "Prioritize citation precision, theory integration, and argument flow before submission."
    )
    return AdversarialMarkingResponse(
        template_id=template.id,
        reports=reports,
        combined_risk_summary=summary,
    )


@router.get("/rubric/templates")
async def list_rubric_templates(
    _user: Annotated[User, Depends(get_current_user)],
) -> dict[str, list[RubricTemplate]]:
    """List available rubric templates for caller selection."""
    return {"templates": list(_RUBRIC_TEMPLATES.values())}


@router.post("/analyse-rubric/compare", response_model=CompareRubricResponse)
async def compare_rubric_assessments(
    body: CompareRubricRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> CompareRubricResponse:
    """Compare baseline and improved draft rubric outcomes."""
    comparison_group_id = str(uuid.uuid4())

    base_result = await _analyse_rubric_internal(
        body=AnalyseRubricRequest(
            text=body.original_text,
            template_id=body.template_id,
            custom_template=body.custom_template,
            unit_id=body.unit_id,
            topic=body.topic,
            grade_level=body.grade_level,
            cohort=body.cohort,
            save_history=False,
            comparison_group_id=comparison_group_id,
            is_improved_version=False,
        ),
        user=user,
        db=db,
    )
    improved_result = await _analyse_rubric_internal(
        body=AnalyseRubricRequest(
            text=body.improved_text,
            template_id=body.template_id,
            custom_template=body.custom_template,
            unit_id=body.unit_id,
            topic=body.topic,
            grade_level=body.grade_level,
            cohort=body.cohort,
            save_history=False,
            comparison_group_id=comparison_group_id,
            is_improved_version=True,
        ),
        user=user,
        db=db,
    )

    improved_by_key = {
        criterion.key: criterion for criterion in improved_result.rubric.criteria
    }
    deltas: list[RubricComparisonDelta] = []
    for criterion in base_result.rubric.criteria:
        target = improved_by_key.get(criterion.key)
        if target is None:
            continue
        deltas.append(
            RubricComparisonDelta(
                key=criterion.key,
                criterion=criterion.criterion,
                before_score=criterion.score,
                after_score=target.score,
                delta=target.score - criterion.score,
            )
        )

    if body.save_history:

        def _to_criteria_records(
            result: AnalyseRubricResponse,
        ) -> list[dict[str, object]]:
            records: list[dict[str, object]] = []
            for item in result.rubric.criteria:
                records.append(
                    {
                        "key": item.key,
                        "label": item.criterion,
                        "score": item.score,
                        "max_score": item.max_score,
                        "weight": item.weight,
                        "feedback": item.feedback,
                        "evidence": item.evidence or "",
                    }
                )
            return records

        try:
            base_record = await create_rubric_assessment(
                db,
                user_id=user.id,
                unit_id=body.unit_id,
                topic=body.topic,
                grade_level=body.grade_level,
                cohort=body.cohort,
                template_id=base_result.template_id,
                comparison_group_id=comparison_group_id,
                is_improved_version=False,
                weighted_total=base_result.rubric.weighted_total,
                raw_total=base_result.rubric.raw_total,
                raw_max=base_result.rubric.raw_max,
                grade_descriptor=base_result.grade_descriptor,
                criteria=_to_criteria_records(base_result),
                improvement_suggestions=base_result.improvement_suggestions,
                assessor_notes=base_result.rubric.assessor_notes,
                auto_commit=False,
            )
            improved_record = await create_rubric_assessment(
                db,
                user_id=user.id,
                unit_id=body.unit_id,
                topic=body.topic,
                grade_level=body.grade_level,
                cohort=body.cohort,
                template_id=improved_result.template_id,
                comparison_group_id=comparison_group_id,
                is_improved_version=True,
                weighted_total=improved_result.rubric.weighted_total,
                raw_total=improved_result.rubric.raw_total,
                raw_max=improved_result.rubric.raw_max,
                grade_descriptor=improved_result.grade_descriptor,
                criteria=_to_criteria_records(improved_result),
                improvement_suggestions=improved_result.improvement_suggestions,
                assessor_notes=improved_result.rubric.assessor_notes,
                auto_commit=False,
            )
            await db.commit()
            base_result.assessment_id = base_record.id
            improved_result.assessment_id = improved_record.id
        except SQLAlchemyError as exc:
            await db.rollback()
            raise _map_database_failure(
                operation="Rubric comparison persistence",
                exc=exc,
            ) from exc

    return CompareRubricResponse(
        comparison_group_id=comparison_group_id,
        before=base_result,
        after=improved_result,
        criterion_deltas=deltas,
        weighted_total_delta=round(
            improved_result.rubric.weighted_total - base_result.rubric.weighted_total,
            1,
        ),
    )


@router.post("/analyse-rubric/iterate", response_model=IterateRubricResponse)
async def iterate_rubric_assessments(
    body: IterateRubricRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
) -> IterateRubricResponse:
    """Run rubric analysis over multiple topics and grade levels."""
    results: list[RubricIterationResult] = []
    for item in body.items:
        try:
            result = await _analyse_rubric_internal(
                body=AnalyseRubricRequest(
                    text=item.text,
                    template_id=body.template_id,
                    custom_template=body.custom_template,
                    unit_id=item.unit_id,
                    topic=item.topic,
                    grade_level=item.grade_level,
                    cohort=item.cohort,
                    save_history=body.save_history,
                ),
                user=user,
                db=db,
            )
        except SQLAlchemyError as exc:
            await db.rollback()
            raise _map_database_failure(
                operation="Rubric iteration persistence",
                exc=exc,
            ) from exc
        results.append(
            RubricIterationResult(
                topic=item.topic,
                grade_level=item.grade_level,
                result=result,
            )
        )

    template = _resolve_template(body.template_id, body.custom_template)
    return IterateRubricResponse(template_id=template.id, results=results)


@router.get("/rubric/history", response_model=RubricAssessmentHistoryPage)
async def get_rubric_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    unit_id: str | None = Query(default=None),
    cohort: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RubricAssessmentHistoryPage:
    """Return rubric assessment history for current user with optional filters."""
    try:
        total, records = await list_rubric_assessments(
            db,
            limit=limit,
            offset=offset,
            user_id=user.id,
            unit_id=unit_id,
            cohort=cohort,
        )
    except SQLAlchemyError as exc:
        raise _map_database_failure(operation="Rubric history lookup", exc=exc) from exc
    return RubricAssessmentHistoryPage(
        limit=limit,
        offset=offset,
        total=total,
        items=[
            RubricAssessmentHistoryItem(
                id=record.id,
                user_id=record.user_id,
                unit_id=record.unit_id,
                topic=record.topic,
                grade_level=record.grade_level,
                cohort=record.cohort,
                template_id=record.template_id,
                is_improved_version=record.is_improved_version,
                weighted_total=record.weighted_total,
                grade_descriptor=record.grade_descriptor,
                created_at=record.created_at_iso,
            )
            for record in records
        ],
    )


@router.get("/rubric/dashboard", response_model=RubricDashboardResponse)
async def get_rubric_dashboard(
    unit_id: str | None = Query(default=None),
    cohort: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    _instructor: User = Depends(require_instructor),
) -> RubricDashboardResponse:
    """Return aggregate rubric analytics for instructors and admins."""
    try:
        summary = await rubric_dashboard_summary(db, unit_id=unit_id, cohort=cohort)
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Rubric dashboard aggregation",
            exc=exc,
        ) from exc
    return RubricDashboardResponse(
        total_assessments=summary.total_assessments,
        average_weighted_score=summary.average_weighted_score,
        median_weighted_score=summary.median_weighted_score,
        grade_distribution=summary.grade_distribution,
        criterion_average_scores=summary.criterion_average_scores,
    )


_GRADE_ORDER = [
    "First Class",
    "Upper Second",
    "Lower Second",
    "Third Class",
    "Pass",
    "Fail",
]

_SCORE_HISTOGRAM_BINS = [
    ("0–10", 0, 10),
    ("10–20", 10, 20),
    ("20–30", 20, 30),
    ("30–40", 30, 40),
    ("40–50", 40, 50),
    ("50–60", 50, 60),
    ("60–70", 60, 70),
    ("70–80", 70, 80),
    ("80–90", 80, 90),
    ("90–100", 90, 101),
]


@router.get("/rubric/dashboard/chart", response_model=RubricDashboardChartResponse)
async def get_rubric_dashboard_chart(
    unit_id: str | None = Query(default=None),
    cohort: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    _instructor: User = Depends(require_instructor),
) -> RubricDashboardChartResponse:
    """Compact chart-ready dashboard payload for direct use in plotting libraries.

    Returns parallel ``labels`` / ``values`` arrays for grade distribution,
    criterion averages, and a score histogram alongside flat summary KPIs.
    No client-side transformation is required — pass the arrays straight into
    Chart.js ``data.labels`` / ``data.datasets[0].data`` or equivalent.
    """
    try:
        summary = await rubric_dashboard_summary(db, unit_id=unit_id, cohort=cohort)
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Rubric dashboard chart aggregation",
            exc=exc,
        ) from exc

    # --- grade distribution (ordered by academic band) -----------------------
    grade_labels = [g for g in _GRADE_ORDER if g in summary.grade_distribution]
    remaining_grade_labels = sorted(
        g for g in summary.grade_distribution if g not in _GRADE_ORDER
    )
    grade_labels.extend(remaining_grade_labels)
    grade_counts = [float(summary.grade_distribution[g]) for g in grade_labels]

    # --- criterion averages (sorted descending by score) ---------------------
    crit_items = sorted(
        summary.criterion_average_scores.items(), key=lambda kv: kv[1], reverse=True
    )
    crit_labels = [k.replace("_", " ").title() for k, _ in crit_items]
    crit_scores = [round(v, 2) for _, v in crit_items]

    # --- score histogram (10-point bands) ------------------------------------
    raw_scores: list[float] = summary.raw_scores
    hist_counts: list[float] = []
    for _, lo, hi in _SCORE_HISTOGRAM_BINS:
        hist_counts.append(float(sum(1 for s in raw_scores if lo <= s < hi)))
    hist_labels = [label for label, _, _ in _SCORE_HISTOGRAM_BINS]

    return RubricDashboardChartResponse(
        grade_distribution=_ChartSeries(labels=grade_labels, values=grade_counts),
        criterion_averages=_ChartSeries(labels=crit_labels, values=crit_scores),
        score_histogram=_ChartSeries(labels=hist_labels, values=hist_counts),
        summary={
            "total_assessments": summary.total_assessments,
            "average_score": round(summary.average_weighted_score, 2),
            "median_score": round(summary.median_weighted_score, 2),
        },
    )


@router.get("/rubric/peer-comparison", response_model=PeerComparisonResponse)
async def get_peer_comparison(
    assessment_id: int = Query(ge=1),
    unit_id: str | None = Query(default=None),
    cohort: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> PeerComparisonResponse:
    """Return anonymized cohort comparison for one saved assessment."""
    try:
        summary = await peer_comparison_summary(
            db,
            assessment_id=assessment_id,
            cohort=cohort,
            unit_id=unit_id,
            min_cohort_size=3,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except SQLAlchemyError as exc:
        raise _map_database_failure(
            operation="Peer comparison aggregation",
            exc=exc,
        ) from exc

    return PeerComparisonResponse(
        assessment_id=summary.assessment_id,
        cohort_size=summary.cohort_size,
        user_score=summary.user_score,
        cohort_average_score=summary.cohort_average_score,
        cohort_median_score=summary.cohort_median_score,
        percentile=summary.percentile,
        min_score=summary.min_score,
        max_score=summary.max_score,
    )
