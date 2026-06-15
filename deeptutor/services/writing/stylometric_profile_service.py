"""Stylometric profile computation, persistence, and telemetry aggregation."""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.models.user_stylometric_profile import UserStylometricProfile

try:
    from redis.exceptions import RedisError
except ImportError:

    class RedisError(Exception):
        """Fallback redis error type when redis package is unavailable."""


logger = logging.getLogger(__name__)

STREAM_KEY = "leo_rag:stylometry:events"
STREAM_CURSOR_KEY = "leo_rag:stylometry:cursor"


@dataclass(frozen=True)
class StylometricProfileSnapshot:
    """Typed view of one user stylometric profile row."""

    user_id: str
    features: dict[str, float]
    exemplars: list[str]
    sample_count: int
    last_score: float | None
    last_grade: str | None


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def compute_stylometric_features(text: str) -> dict[str, float]:
    """Compute deterministic stylometric features from plain text."""
    words = [token for token in re.findall(r"[A-Za-z']+", text.lower()) if token]
    if not words:
        return {
            "token_diversity": 0.0,
            "yule_i": 0.0,
            "sentence_length_cv": 0.0,
            "passive_ratio": 0.0,
            "pronoun_density": 0.0,
            "punctuation_density": 0.0,
            "burstiness": 0.0,
        }

    frequencies: dict[str, int] = {}
    for word in words:
        frequencies[word] = frequencies.get(word, 0) + 1

    m1 = float(len(words))
    m2 = float(sum(count * count for count in frequencies.values()))
    yule_i = ((m1 * m1) / (m2 - m1)) if (m2 - m1) > 0 else 0.0

    sentences = _split_sentences(text)
    sentence_lengths = [
        len(re.findall(r"[A-Za-z']+", sentence)) for sentence in sentences
    ]
    mean_sentence = _safe_mean([float(length) for length in sentence_lengths])
    if len(sentence_lengths) >= 2 and mean_sentence > 0:
        sentence_cv = statistics.stdev(sentence_lengths) / mean_sentence
    else:
        sentence_cv = 0.0

    passive_markers = re.findall(
        r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", text, flags=re.IGNORECASE
    )
    pronouns = re.findall(
        r"\b(?:i|we|you|he|she|they|my|our|their|his|her)\b", text, flags=re.IGNORECASE
    )
    punctuation_marks = re.findall(r"[,:;!?()\-]", text)

    return {
        "token_diversity": round(len(frequencies) / len(words), 5),
        "yule_i": round(yule_i, 5),
        "sentence_length_cv": round(sentence_cv, 5),
        "passive_ratio": round(len(passive_markers) / max(1, len(sentences)), 5),
        "pronoun_density": round(len(pronouns) / len(words), 5),
        "punctuation_density": round(len(punctuation_marks) / len(text), 5),
        "burstiness": round(sentence_cv, 5),
    }


def extract_submission_metadata(text_corpus: str) -> tuple[float | None, str | None]:
    """Scrape score and grade markers from graded assignment text."""
    score_pattern = re.compile(
        r"\b(?:score|mark|total)\s*[:\-]*\s*(\d{1,2}(?:\.\d)?)\s*(?:/\s*100)?",
        re.IGNORECASE,
    )
    grade_pattern = re.compile(
        r"\b(fail|pass|credit|distinction|high\s+distinction)\b",
        re.IGNORECASE,
    )

    score_match = score_pattern.search(text_corpus)
    grade_match = grade_pattern.search(text_corpus)

    resolved_score = float(score_match.group(1)) if score_match else None
    resolved_grade = grade_match.group(1).title() if grade_match else None
    return resolved_score, resolved_grade


def parse_graded_assignment_text(text: str) -> tuple[list[str], list[str]]:
    """Split graded assignment text into prose blocks and feedback blocks."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    prose_blocks: list[str] = []
    feedback_blocks: list[str] = []

    feedback_re = re.compile(
        r"\b(?:feedback|marker comment|criterion|rubric|grade|score|overall comment)\b",
        re.IGNORECASE,
    )

    for paragraph in paragraphs:
        if len(paragraph) < 40:
            continue
        if feedback_re.search(paragraph):
            feedback_blocks.append(paragraph)
        else:
            prose_blocks.append(paragraph)

    return prose_blocks, feedback_blocks


def _merge_features(
    current: dict[str, float],
    incoming: dict[str, float],
    existing_samples: int,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for key in incoming:
        current_value = float(current.get(key, 0.0))
        incoming_value = float(incoming.get(key, 0.0))
        merged[key] = round(
            ((current_value * existing_samples) + incoming_value)
            / (existing_samples + 1),
            5,
        )
    return merged


def stylometric_distance(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> float:
    """Compute Euclidean distance across common stylometric dimensions."""
    keys = sorted(set(candidate.keys()) & set(baseline.keys()))
    if not keys:
        return 0.0
    squared = sum((candidate[key] - baseline[key]) ** 2 for key in keys)
    return float(squared**0.5)


def is_high_score_submission(score: float | None, grade: str | None) -> bool:
    """Return True when grade metadata indicates distinction-level quality."""
    if score is not None and score >= 75.0:
        return True
    if grade is None:
        return False
    grade_norm = grade.lower().strip()
    return grade_norm in {"distinction", "high distinction"}


async def get_user_stylometric_profile(
    db: AsyncSession,
    user_id: str,
) -> StylometricProfileSnapshot | None:
    """Load a user's persisted stylometric profile."""
    result = await db.execute(
        select(UserStylometricProfile).where(UserStylometricProfile.user_id == user_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None

    features_obj = json.loads(row.profile_json)
    exemplars_obj = json.loads(row.exemplar_json)

    features: dict[str, float] = {}
    if isinstance(features_obj, dict):
        for key, value in features_obj.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                features[key] = float(value)

    exemplars: list[str] = []
    if isinstance(exemplars_obj, list):
        for item in exemplars_obj:
            if isinstance(item, str) and item.strip():
                exemplars.append(item.strip())

    return StylometricProfileSnapshot(
        user_id=row.user_id,
        features=features,
        exemplars=exemplars,
        sample_count=row.sample_count,
        last_score=row.last_score,
        last_grade=row.last_grade,
    )


async def upsert_user_stylometric_profile(
    db: AsyncSession,
    *,
    user_id: str,
    features: dict[str, float],
    exemplars: list[str],
    score: float | None,
    grade: str | None,
) -> StylometricProfileSnapshot:
    """Insert or update a user's stylometric profile with rolling averages."""
    result = await db.execute(
        select(UserStylometricProfile).where(UserStylometricProfile.user_id == user_id)
    )
    existing = result.scalar_one_or_none()

    unique_exemplars = [item.strip() for item in exemplars if item.strip()][:12]

    if existing is None:
        row = UserStylometricProfile(
            user_id=user_id,
            profile_json=json.dumps(features),
            exemplar_json=json.dumps(unique_exemplars),
            sample_count=1,
            last_score=score,
            last_grade=grade,
        )
        db.add(row)
        await db.flush()
        return StylometricProfileSnapshot(
            user_id=user_id,
            features=features,
            exemplars=unique_exemplars,
            sample_count=1,
            last_score=score,
            last_grade=grade,
        )

    current_obj = json.loads(existing.profile_json)
    current_features: dict[str, float] = {}
    if isinstance(current_obj, dict):
        for key, value in current_obj.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                current_features[key] = float(value)

    merged = _merge_features(current_features, features, existing.sample_count)

    existing_exemplar_obj = json.loads(existing.exemplar_json)
    existing_exemplars: list[str] = []
    if isinstance(existing_exemplar_obj, list):
        for item in existing_exemplar_obj:
            if isinstance(item, str) and item.strip():
                existing_exemplars.append(item.strip())

    exemplar_union = (unique_exemplars + existing_exemplars)[:12]

    existing.profile_json = json.dumps(merged)
    existing.exemplar_json = json.dumps(exemplar_union)
    existing.sample_count = existing.sample_count + 1
    existing.last_score = score
    existing.last_grade = grade

    return StylometricProfileSnapshot(
        user_id=user_id,
        features=merged,
        exemplars=exemplar_union,
        sample_count=existing.sample_count,
        last_score=score,
        last_grade=grade,
    )


async def append_telemetry_events(
    *,
    user_id: str,
    events: list[dict[str, object]],
) -> int:
    """Push write telemetry events to Redis stream for asynchronous processing."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning(
            "redis is unavailable; telemetry event dropped for user %s", user_id
        )
        return 0

    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2)

    accepted = 0
    try:
        for event in events:
            payload = {
                "user_id": user_id,
                "op": str(event.get("op", "insert")),
                "text": str(event.get("text", ""))[:2_000],
                "source": str(event.get("source", "keyboard")),
                "is_paste": "1" if bool(event.get("is_paste", False)) else "0",
            }
            await client.xadd(STREAM_KEY, payload)
            accepted += 1
    except (RedisError, OSError, RuntimeError, TimeoutError) as exc:
        logger.warning("Telemetry stream unavailable for user %s: %s", user_id, exc)
        return 0
    finally:
        await client.aclose()

    return accepted


async def aggregate_telemetry_stream(
    db: AsyncSession,
    *,
    max_events: int = 5_000,
) -> dict[str, int]:
    """Aggregate stream events and update user stylometric profiles."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning("redis package missing; telemetry aggregation skipped")
        return {"processed": 0, "profiles_updated": 0}

    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2)

    try:
        last_id = await client.get(STREAM_CURSOR_KEY)
    except (RedisError, OSError, RuntimeError, TimeoutError) as exc:
        logger.warning("Telemetry cursor unavailable: %s", exc)
        await client.aclose()
        return {"processed": 0, "profiles_updated": 0}
    cursor = last_id if isinstance(last_id, str) else "0-0"

    try:
        entries = await client.xrange(STREAM_KEY, min=cursor, max="+", count=max_events)
    except (RedisError, OSError, RuntimeError, TimeoutError) as exc:
        logger.warning("Telemetry stream read unavailable: %s", exc)
        await client.aclose()
        return {"processed": 0, "profiles_updated": 0}
    if not entries:
        await client.aclose()
        return {"processed": 0, "profiles_updated": 0}

    grouped: dict[str, list[str]] = {}
    latest_id = cursor
    processed = 0

    for event_id, values in entries:
        latest_id = event_id
        if event_id == cursor:
            continue
        user_id = str(values.get("user_id", "")).strip()
        op = str(values.get("op", "insert"))
        source = str(values.get("source", "keyboard"))
        is_paste = str(values.get("is_paste", "0")) == "1"
        text = str(values.get("text", ""))
        if not user_id or op != "insert" or source != "keyboard" or is_paste:
            continue
        grouped.setdefault(user_id, []).append(text)
        processed += 1

    profiles_updated = 0
    for user_id, fragments in grouped.items():
        merged_text = " ".join(fragment for fragment in fragments if fragment.strip())
        if len(merged_text) < 80:
            continue
        features = compute_stylometric_features(merged_text)
        await upsert_user_stylometric_profile(
            db,
            user_id=user_id,
            features=features,
            exemplars=[],
            score=None,
            grade=None,
        )
        profiles_updated += 1

    try:
        await client.set(STREAM_CURSOR_KEY, latest_id)
    except (RedisError, OSError, RuntimeError, TimeoutError) as exc:
        logger.warning("Telemetry cursor update unavailable: %s", exc)
    await client.aclose()
    return {"processed": processed, "profiles_updated": profiles_updated}
