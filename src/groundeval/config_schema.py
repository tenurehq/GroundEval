"""
groundeval/config_schema.py
=============================
Config schema definition and validation.

Called at startup by both cmd_generate and cmd_eval. Rejects unknown
top-level keys and warns when known keys are absent or empty.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("groundeval.config_schema")

KNOWN_TOP_LEVEL_KEYS: set[str] = {
    "output_dir",
    "artifacts_dir",
    "use_event_log_policy",
    "actors",
    "roles",
    "provider",
    "provider_path",
    "model",
    "api_key",
    "api_key_env",
    "base_url",
    "temperature",
    "max_tokens",
    "max_retries",
    "llm_question_prose",
    "llm_model",
    "perspective",
    "perspective_actors",
    "causal_links",
    "silence_pairs",
    "easy_ratio",
    "medium_ratio",
    "hard_ratio",
    "max_perspective_questions",
    "max_counterfactual_questions",
    "max_silence_questions",
    "max_questions_per_actor",
    "max_questions_per_event_type",
    "max_links_per_spec",
    "trivial_search_space",
    "seed",
}


WARN_IF_EMPTY = {
    "causal_links": "COUNTERFACTUAL",
    "silence_pairs": "SILENCE",
}


def validate_config(cfg: dict[str, Any], *, command: str) -> None:
    """
    Validate a config dict before generation or evaluation.

    - Rejects unknown top-level keys (hard error)
    - Warns when causal_links or silence_pairs are missing/empty
    - Does NOT validate nested structure beyond top-level keys
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a YAML mapping, got {type(cfg).__name__}")

    unknown = set(cfg.keys()) - KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(KNOWN_TOP_LEVEL_KEYS))}"
        )

    for key, track_name in WARN_IF_EMPTY.items():
        val = cfg.get(key)
        if not val:
            logger.warning(
                f"  Config key '{key}' is absent or empty. "
                f"No {track_name} questions will be generated."
            )
