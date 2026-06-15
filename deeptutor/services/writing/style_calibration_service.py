"""Binary style calibration engine for initializing user style baselines."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deeptutor.models.user_stylometric_profile import UserStylometricProfile
from deeptutor.services.writing.types import PreferenceSignal, WeightUpdateMap


class ContrastPair(BaseModel):
    """Strict Pydantic contract for a single style metric contrast pair."""

    pair_id: str
    metric_target: Literal["sentence_variance", "passive_ratio", "lexical_density"]
    concept: str
    cadence_a_text: str
    cadence_b_text: str
    cadence_a_weight: float
    cadence_b_weight: float


_DEFAULT_PROFILE: WeightUpdateMap = {
    "sentence_length_variance": 12.5,
    "passive_voice_ratio": 0.18,
    "lexical_density_ttr": 0.58,
}


DEFAULT_CONTRAST_PAIRS: list[ContrastPair] = [
    ContrastPair(
        pair_id="sv_01",
        metric_target="sentence_variance",
        concept="Policy adoption outcomes",
        cadence_a_text=(
            "The policy was implemented in multiple areas. "
            "Its effects were observed in routine contexts. "
            "The findings were presented in a stable sequence."
        ),
        cadence_b_text=(
            "The policy changed outcomes quickly. "
            "In some departments it reduced delays, while in others it exposed coordination gaps that required targeted fixes. "
            "Those mixed effects sharpen the final recommendation."
        ),
        cadence_a_weight=5.8,
        cadence_b_weight=13.4,
    ),
    ContrastPair(
        pair_id="sv_02",
        metric_target="sentence_variance",
        concept="Assessment feedback use",
        cadence_a_text=(
            "Feedback was collected and reviewed. "
            "The themes were grouped and summarised. "
            "An adjustment was made in response."
        ),
        cadence_b_text=(
            "Students used targeted feedback to revise claims. "
            "Some shortened weak sections, whereas others expanded evidence where markers identified unsupported inferences. "
            "Those changes improved argument force."
        ),
        cadence_a_weight=6.2,
        cadence_b_weight=14.1,
    ),
    ContrastPair(
        pair_id="pr_01",
        metric_target="passive_ratio",
        concept="Research method explanation",
        cadence_a_text=(
            "Data were collected from two cohorts, and patterns were identified before conclusions were drawn."
        ),
        cadence_b_text=(
            "We collected data from two cohorts, identified the patterns, and then drew the conclusion."
        ),
        cadence_a_weight=0.31,
        cadence_b_weight=0.09,
    ),
    ContrastPair(
        pair_id="pr_02",
        metric_target="passive_ratio",
        concept="Theory comparison",
        cadence_a_text=(
            "The frameworks were contrasted, and key assumptions were highlighted before a judgement was formed."
        ),
        cadence_b_text=(
            "The analysis contrasts the frameworks, highlights their assumptions, and forms a clear judgement."
        ),
        cadence_a_weight=0.27,
        cadence_b_weight=0.08,
    ),
    ContrastPair(
        pair_id="ld_01",
        metric_target="lexical_density",
        concept="Causal argument clarity",
        cadence_a_text=(
            "This is a very important point that shows how a lot of factors are involved in the issue."
        ),
        cadence_b_text=(
            "The claim isolates three causal drivers and links each to a measurable policy outcome."
        ),
        cadence_a_weight=0.46,
        cadence_b_weight=0.63,
    ),
    ContrastPair(
        pair_id="ld_02",
        metric_target="lexical_density",
        concept="Evidence interpretation",
        cadence_a_text=(
            "It is important to note that the evidence gives a broad idea of the main trend."
        ),
        cadence_b_text=(
            "The evidence indicates a consistent upward trend in participation after intervention."
        ),
        cadence_a_weight=0.48,
        cadence_b_weight=0.66,
    ),
]


class StyleCalibrationEngine:
    """Process binary choices over contrast pairs into a baseline style profile."""

    def __init__(self, contrast_registry: list[ContrastPair]) -> None:
        self._registry = {cp.pair_id: cp for cp in contrast_registry}

    def list_pairs(self) -> list[ContrastPair]:
        """Return all configured contrast pairs in deterministic order."""
        return list(self._registry.values())

    def calculate_initial_profile(
        self,
        selections: list[dict[str, str]],
    ) -> WeightUpdateMap:
        """Aggregate binary preferences into concrete stylometric parameters."""
        profile_weights: WeightUpdateMap = dict(_DEFAULT_PROFILE)

        counts: dict[str, int] = {
            "sentence_variance": 0,
            "passive_ratio": 0,
            "lexical_density": 0,
        }
        accumulators: dict[str, float] = {
            "sentence_variance": 0.0,
            "passive_ratio": 0.0,
            "lexical_density": 0.0,
        }

        for selection in selections:
            pair_id = selection.get("pair_id")
            chosen_signal = selection.get("choice")

            if pair_id is None or chosen_signal not in ("cadence_a", "cadence_b"):
                continue
            if pair_id not in self._registry:
                continue

            pair = self._registry[pair_id]
            metric = pair.metric_target

            weight = (
                pair.cadence_a_weight
                if chosen_signal == "cadence_a"
                else pair.cadence_b_weight
            )
            accumulators[metric] += weight
            counts[metric] += 1

        if counts["sentence_variance"] > 0:
            profile_weights["sentence_length_variance"] = (
                accumulators["sentence_variance"] / counts["sentence_variance"]
            )
        if counts["passive_ratio"] > 0:
            profile_weights["passive_voice_ratio"] = (
                accumulators["passive_ratio"] / counts["passive_ratio"]
            )
        if counts["lexical_density"] > 0:
            profile_weights["lexical_density_ttr"] = (
                accumulators["lexical_density"] / counts["lexical_density"]
            )

        profile_weights["sentence_length_variance"] = round(
            profile_weights["sentence_length_variance"], 4
        )
        profile_weights["passive_voice_ratio"] = round(
            profile_weights["passive_voice_ratio"], 5
        )
        profile_weights["lexical_density_ttr"] = round(
            profile_weights["lexical_density_ttr"], 5
        )
        return profile_weights

    @staticmethod
    def profile_to_feature_map(profile_weights: WeightUpdateMap) -> dict[str, float]:
        """Map calibration weights to the canonical stored stylometric feature schema."""
        variance = float(profile_weights["sentence_length_variance"])
        # Approximate CV against a 20-word sentence baseline used in prompts.
        sentence_length_cv = max(0.0, min(1.2, variance / 20.0))

        lexical_density_ttr = float(profile_weights["lexical_density_ttr"])
        passive_voice_ratio = float(profile_weights["passive_voice_ratio"])

        return {
            "sentence_length_variance": round(variance, 4),
            "sentence_length_cv": round(sentence_length_cv, 5),
            "passive_voice_ratio": round(passive_voice_ratio, 5),
            "passive_ratio": round(passive_voice_ratio, 5),
            "lexical_density_ttr": round(lexical_density_ttr, 5),
            "token_diversity": round(lexical_density_ttr, 5),
            "calibrated": 1.0,
        }

    async def commit_calibrated_profile(
        self,
        db: AsyncSession,
        *,
        user_id: str,
        profile_weights: WeightUpdateMap,
    ) -> UserStylometricProfile:
        """Persist calibrated profile as the current baseline for the user."""
        feature_map = self.profile_to_feature_map(profile_weights)

        existing = (
            await db.execute(
                select(UserStylometricProfile).where(
                    UserStylometricProfile.user_id == user_id
                )
            )
        ).scalar_one_or_none()

        if existing is None:
            row = UserStylometricProfile(
                user_id=user_id,
                profile_json=json.dumps(feature_map),
                exemplar_json="[]",
                sample_count=1,
                last_score=None,
                last_grade="calibrated",
            )
            db.add(row)
            await db.flush()
            return row

        existing.profile_json = json.dumps(feature_map)
        existing.sample_count = max(1, existing.sample_count)
        existing.last_grade = "calibrated"
        await db.flush()
        return existing
