"""Deterministic algorithms for advanced writing feature blueprints."""

from __future__ import annotations

import math
import re
import statistics
from pydantic import BaseModel
from pathlib import Path
from typing import Literal

from yaml import safe_load


type TokenArray = list[str]
type PatternDictionary = dict[str, str]
type PromptInstructionList = list[dict[str, str]]
type BiasPayloadMap = dict[str, int]
type Vector = list[float]
type VocabularyFrequencyMap = dict[str, int]
type SubstitutionDictionary = dict[str, str]


class StylometricProfile(BaseModel):
    """Strict Pydantic contract for chronological stylometric indicators."""

    lexical_density_ttr: float
    mean_sentence_length: float
    sentence_length_variance: float
    passive_voice_ratio: float


class TextSegment(BaseModel):
    """Strict Pydantic contract for a paragraph segment with source coordinates."""

    start_index: int
    end_index: int
    text: str


class ContextChunkPlacement(BaseModel):
    """Strict Pydantic contract for chunk placement inside prompt context."""

    chunk_id: str
    filename: str
    token_index_start: int
    relative_position_percentage: float
    retention_risk_zone: Literal["optimal_top", "vulnerable_middle", "optimal_bottom"]


class SyllabusGap(BaseModel):
    """Strict Pydantic contract for syllabus coverage gaps."""

    module_name: str
    missing_theory_title: str
    associated_chunk_id: str
    max_similarity_resolved: float


def extract_student_prose_segments(
    text: str,
) -> tuple[list[TextSegment], list[TextSegment]]:
    """Split assignment text into prose and feedback segments with coordinates."""
    paragraph_pattern = re.compile(r"(?s)(.*?)(?:\n{2,}|$)")
    feedback_re = re.compile(
        r"\b(?:feedback|marker comment|criterion|rubric|grade|score|overall comment)\b",
        re.IGNORECASE,
    )

    prose_segments: list[TextSegment] = []
    feedback_segments: list[TextSegment] = []

    for match in paragraph_pattern.finditer(text):
        paragraph = match.group(1).strip()
        if len(paragraph) < 40:
            continue

        segment = TextSegment(
            start_index=match.start(1),
            end_index=match.end(1),
            text=paragraph,
        )

        if feedback_re.search(paragraph):
            feedback_segments.append(segment)
        else:
            prose_segments.append(segment)

    return prose_segments, feedback_segments


def analyse_historical_drift(
    document_texts: list[str],
) -> StylometricProfile:
    """Extract chronological stylistic indicators from a collection of student texts."""
    combined_text = " ".join(document_texts).strip()
    if not combined_text:
        return StylometricProfile(0.0, 0.0, 0.0, 0.0)

    words: TokenArray = re.findall(r"\b[a-zA-Z]+\b", combined_text.lower())
    sentences: list[str] = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", combined_text) if s.strip()
    ]

    if not words or not sentences:
        return StylometricProfile(0.0, 0.0, 0.0, 0.0)

    ttr = len(set(words)) / len(words)

    sentence_lengths = [len(s.split()) for s in sentences]
    mean_len = statistics.mean(sentence_lengths)
    var_len = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    passive_pattern = re.compile(
        r"\b(am|is|are|was|were|be|been|being)\b\s+([a-z]+ed)\b", re.IGNORECASE
    )
    passive_matches = passive_pattern.findall(combined_text)
    passive_ratio = len(passive_matches) / len(sentences)

    return StylometricProfile(
        lexical_density_ttr=round(ttr, 4),
        mean_sentence_length=round(mean_len, 2),
        sentence_length_variance=round(var_len, 2),
        passive_voice_ratio=round(passive_ratio, 4),
    )


def calculate_prompt_retention_map(
    system_preamble_tokens: int,
    retrieved_chunks: list[tuple[str, str, int]],
) -> list[ContextChunkPlacement]:
    """Map chunk locations inside prompt context and classify retention risk."""
    total_tokens = system_preamble_tokens + sum(c[2] for c in retrieved_chunks)
    current_cursor = system_preamble_tokens
    placement_map: list[ContextChunkPlacement] = []

    for chunk_id, filename, token_count in retrieved_chunks:
        mid_point = current_cursor + (token_count // 2)
        percentage = mid_point / total_tokens if total_tokens > 0 else 0.0

        if percentage < 0.15:
            risk_zone: Literal["optimal_top", "vulnerable_middle", "optimal_bottom"] = (
                "optimal_top"
            )
        elif percentage > 0.85:
            risk_zone = "optimal_bottom"
        else:
            risk_zone = "vulnerable_middle"

        placement_map.append(
            ContextChunkPlacement(
                chunk_id=chunk_id,
                filename=filename,
                token_index_start=current_cursor,
                relative_position_percentage=round(percentage, 3),
                retention_risk_zone=risk_zone,
            )
        )
        current_cursor += token_count

    return placement_map


def compute_inverse_coverage(
    outline_embedding: Vector,
    syllabus_chunks: list[dict[str, object]],
    *,
    threshold: float = 0.50,
) -> list[SyllabusGap]:
    """Execute matrix coverage comparisons to reveal missing conceptual assertions."""

    def cosine_similarity(v1: Vector, v2: Vector) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
        magnitude_v1 = math.sqrt(sum(a * a for a in v1))
        magnitude_v2 = math.sqrt(sum(b * b for b in v2))
        if magnitude_v1 == 0.0 or magnitude_v2 == 0.0:
            return 0.0
        return dot_product / (magnitude_v1 * magnitude_v2)

    gaps: list[SyllabusGap] = []
    for chunk in syllabus_chunks:
        chunk_vector_obj = chunk.get("embedding")
        if not isinstance(chunk_vector_obj, list):
            continue

        if not all(isinstance(value, (int, float)) for value in chunk_vector_obj):
            continue

        chunk_vector = [float(value) for value in chunk_vector_obj]
        similarity = cosine_similarity(outline_embedding, chunk_vector)

        if similarity < threshold:
            gaps.append(
                SyllabusGap(
                    module_name=str(chunk.get("module", "unknown")),
                    missing_theory_title=str(chunk.get("theory", "unknown")),
                    associated_chunk_id=str(chunk.get("id", "unknown")),
                    max_similarity_resolved=round(similarity, 4),
                )
            )

    return sorted(gaps, key=lambda item: item.max_similarity_resolved)


async def perform_syllabus_gap_analysis(
    student_text: str,
    syllabus_chunks: list[dict[str, object]],
    *,
    threshold: float = 0.5,
) -> list[SyllabusGap]:
    """Orchestrate embedding generation and coverage analysis for a student text."""
    from deeptutor.services.writing.provider_client import (
        get_embedding_client,
        resolve_embedding_model,
        should_send_embedding_dimensions,
    )

    client = get_embedding_client()
    model = resolve_embedding_model()

    kwargs: dict[str, object] = {"model": model, "input": student_text}
    if should_send_embedding_dimensions():
        # text-embedding-3-small uses 1536 by default.
        kwargs["dimensions"] = 1536

    response = await client.embeddings.create(**kwargs)
    student_embedding = response.data[0].embedding

    return compute_inverse_coverage(student_embedding, syllabus_chunks, threshold=threshold)


class SlopPhraseSanitiser:
    """Identifies and removes machine-slop terminology via regex mapping."""

    def __init__(self, patterns: PatternDictionary) -> None:
        self._compiled_rules: list[tuple[re.Pattern[str], str]] = [
            (re.compile(raw_pattern, re.IGNORECASE), replacement)
            for raw_pattern, replacement in patterns.items()
        ]

    def clear_slop(self, raw_prose: str) -> str:
        """Apply deterministic phrase substitutions to completion output."""
        sanitised_prose = raw_prose
        for pattern, replacement in self._compiled_rules:
            sanitised_prose = pattern.sub(replacement, sanitised_prose)

        return re.sub(r"\s+", " ", sanitised_prose).strip()


class StylometricPromptCompiler:
    """Translate profile metrics into strict generation constraints."""

    def __init__(self, profile_metrics: dict[str, float]) -> None:
        self.mean_sentence_len = profile_metrics.get("mean_sentence_length", 20.0)
        self.ttr = profile_metrics.get(
            "lexical_density_ttr", profile_metrics.get("token_diversity", 0.55)
        )
        self.variance = profile_metrics.get(
            "sentence_length_variance",
            profile_metrics.get("sentence_length_cv", 8.0),
        )

    def compile_constrained_payload(
        self, student_outline_node: str
    ) -> PromptInstructionList:
        """Construct a prompt array packed with style-matching constraints."""
        system_instruction = (
            "Adhere to these structural writing constraints to match the author's voice:\n"
            f"1. Target mean sentence length around {self.mean_sentence_len:.2f} words.\n"
            f"2. Maintain sentence length variation near {self.variance:.2f}.\n"
            f"3. Maintain vocabulary diversity near TTR {self.ttr:.3f}.\n"
            "4. Avoid passive hedging structures and machine transitional cliches."
        )

        return [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": f"Draft prose for this outline node: {student_outline_node}",
            },
        ]


class StructuralEntropyExtrapolator:
    """Validate and enforce sentence-length entropy in generated prose."""

    @staticmethod
    def verify_output_entropy(generated_prose: str) -> bool:
        """Evaluate sentence length variance to detect machine cadence."""
        sentences = [s.strip() for s in generated_prose.split(".") if s.strip()]
        if len(sentences) < 3:
            return True

        word_counts = [len(s.split()) for s in sentences]
        variance = statistics.stdev(word_counts)
        return variance >= 5.5

    @staticmethod
    def append_entropy_instructions(user_notes: str) -> str:
        """Inject structural sequence instructions into the drafting prompt."""
        pattern_directive = (
            "Structure the output paragraph using this sentence profile:\n"
            "- Sentence 1: brief direct claim (under 12 words).\n"
            "- Sentence 2: extended analytical explanation with two clauses (over 28 words).\n"
            "- Sentence 3: precise summary tied to evidence (15-20 words).\n"
            "Maintain structural variation across all generated sentences."
        )
        return f"{pattern_directive}\n\nDraft prose for these notes:\n{user_notes}"


class SyntacticTopology(BaseModel):
    """Strict Pydantic contract for hierarchical grammar topology metrics."""

    mean_dependency_depth: float
    branching_coefficient: float
    clause_count_variance: float


class SyntacticTopologyEvaluator:
    """Evaluate grammatical hierarchy markers from prose text blocks."""

    @staticmethod
    def extract_topology(text_segments: list[str]) -> SyntacticTopology:
        """Calculate depth and branching metrics from text segment sequence."""
        combined_text = " ".join(text_segments).strip()
        if not combined_text:
            return SyntacticTopology(0.0, 0.0, 0.0)

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", combined_text) if s.strip()
        ]
        if not sentences:
            return SyntacticTopology(0.0, 0.0, 0.0)

        clause_markers = re.compile(
            r"\b(because|although|which|whereas|whereby|since|if|unless|while|after|before)\b",
            re.IGNORECASE,
        )

        depths: list[int] = []
        branching_factors: list[int] = []

        for sentence in sentences:
            matches = clause_markers.findall(sentence)
            clause_count = len(matches) + 1
            branching_factors.append(clause_count)

            internal_bounds = len(re.findall(r"[,;]", sentence))
            calculated_depth = 1 + clause_count + (internal_bounds // 2)
            depths.append(calculated_depth)

        mean_depth = statistics.mean(depths) if depths else 1.0
        mean_branching = (
            statistics.mean(branching_factors) if branching_factors else 1.0
        )
        variance_clauses = (
            statistics.stdev(branching_factors) if len(branching_factors) > 1 else 0.0
        )

        return SyntacticTopology(
            mean_dependency_depth=round(mean_depth, 2),
            branching_coefficient=round(mean_branching, 2),
            clause_count_variance=round(variance_clauses, 2),
        )


class LexicalEntropyEqualiser:
    """Adjust vocabulary choices to align with user-typical distribution."""

    def __init__(
        self,
        historical_vocabulary: VocabularyFrequencyMap,
        substitution_rules: SubstitutionDictionary,
    ) -> None:
        self.user_words = historical_vocabulary
        self.rules = {
            re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE): value
            for key, value in substitution_rules.items()
        }

    def normalise_vocabulary_cadence(self, draft_prose: str) -> str:
        """Replace out-of-distribution machine terms with direct alternatives."""
        processed_text = draft_prose
        for compiled_pattern, replacement in self.rules.items():
            processed_text = compiled_pattern.sub(replacement, processed_text)
        return processed_text


class SlidingWindowBurstinessController:
    """Validate rolling sentence-length variance across paragraph windows."""

    def __init__(self, minimum_allowed_variance: float = 6.2) -> None:
        self.min_variance = minimum_allowed_variance

    def verify_cadence_entropy(self, paragraph_block: str) -> bool:
        """Evaluate sentence-length variation threshold for one paragraph block."""
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph_block) if s.strip()
        ]
        if len(sentences) < 3:
            return True

        word_counts = [len(s.split()) for s in sentences]
        rolling_variance = statistics.stdev(word_counts)
        return rolling_variance >= self.min_variance


def _normalise_literal_phrase(pattern: str) -> str | None:
    value = pattern
    value = value.replace("\\b", "")
    value = value.replace("(?:", "(")
    if re.search(r"[\[\]{}+*?^$]", value):
        return None
    if "(" in value or ")" in value or "|" in value:
        return None
    value = value.replace("\\", "")
    normalised = re.sub(r"\s+", " ", value).strip().lower()
    return normalised if normalised else None


def build_sanitiser_patterns_from_config(data: dict[str, object]) -> PatternDictionary:
    """Compile deterministic replacement mappings from ai_patterns config."""
    replacements: PatternDictionary = {}
    sections = ("phrases", "ngram_phrases")

    canonical_replacements: dict[str, str] = {
        "it is important to note": "",
        "it is crucial to note": "",
        "delve into": "analyse",
        "a testament to the pivotal role of": "central to",
        "crucially, this underscores": "this shows",
        "this essay will explore": "",
        "the purpose of this essay is": "",
    }

    for section in sections:
        section_obj = data.get(section)
        if not isinstance(section_obj, list):
            continue

        for item in section_obj:
            if not isinstance(item, dict):
                continue
            pattern_obj = item.get("pattern")
            if not isinstance(pattern_obj, str):
                continue

            literal_phrase = _normalise_literal_phrase(pattern_obj)
            if literal_phrase is None:
                continue

            replacement = ""
            for key, candidate in canonical_replacements.items():
                if key in literal_phrase:
                    replacement = candidate
                    break
            replacements[pattern_obj] = replacement

    return replacements


def load_sanitiser_from_pattern_file(pattern_file: Path) -> SlopPhraseSanitiser:
    """Load ai_patterns YAML and construct a deterministic slop sanitiser."""
    if not pattern_file.exists():
        return SlopPhraseSanitiser({})

    loaded = safe_load(pattern_file.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return SlopPhraseSanitiser({})

    replacements = build_sanitiser_patterns_from_config(loaded)
    return SlopPhraseSanitiser(replacements)


def compile_bias_map_from_terms(
    tokenizer_engine: object,
    banned_words: list[str],
) -> BiasPayloadMap:
    """Compile absolute negative logit bias mapping for banned terms."""
    bias_payload: BiasPayloadMap = {}

    if not hasattr(tokenizer_engine, "encode"):
        return bias_payload

    encode_fn = tokenizer_engine.encode
    if not callable(encode_fn):
        return bias_payload

    for word in banned_words:
        encoded_obj = encode_fn(word)
        if not isinstance(encoded_obj, list):
            continue
        token_ids = [token for token in encoded_obj if isinstance(token, int)]
        for token_id in token_ids:
            bias_payload[str(token_id)] = -100

    return bias_payload
