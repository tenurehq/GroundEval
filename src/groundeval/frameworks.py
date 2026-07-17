from __future__ import annotations

FRAMEWORK_ADAPTERS = frozenset({
    "crewai",
    "langgraph",
    "maf",
    "openai_agents",
})


def normalize_framework(value: object) -> str:
    return str(value or "").strip().lower()


def is_framework_adapter(value: object) -> bool:
    return normalize_framework(value) in FRAMEWORK_ADAPTERS
