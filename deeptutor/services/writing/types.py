"""Shared type definitions for the writing service layer."""

from __future__ import annotations

from typing import Literal

# PEP 695 type aliases for provider and client identifiers
type ProviderName = Literal[
    "openai",
    "gemini",
    "openrouter",
    "groq",
    "together",
    "deepseek",
    "xai",
    "mistral",
    "ollama",
    "custom",
]

type ClientKind = Literal["llm", "embedding"]

type EditSource = Literal["manual_typing", "api_scrub", "context_expansion"]

type PreferenceSignal = Literal["cadence_a", "cadence_b"]
type WeightUpdateMap = dict[str, float]

type TokenArray = list[str]
type PatternDictionary = dict[str, str]
type PromptInstructionList = list[dict[str, str]]
type BiasPayloadMap = dict[str, int]
type Vector = list[float]
type VocabularyFrequencyMap = dict[str, int]
type SubstitutionDictionary = dict[str, str]
