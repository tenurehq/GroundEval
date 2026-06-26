"""
groundeval/config_schema.py
=============================
Config schema definition and validation for task-contract evaluation.

Called at startup by cmd_task and cmd_validate. Rejects unknown
top-level keys and warns when expected keys are absent. Validates
critical subkeys that would cause silent zero-scores or hard crashes
if missing or malformed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("groundeval.config_schema")

KNOWN_TOP_LEVEL_KEYS: set[str] = {
    "agent",
    "output_dir",
    "artifacts_dir",
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
    "task_contracts",
    "seed",
}


def validate_config(cfg: dict[str, Any], *, command: str) -> None:
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a YAML mapping, got {type(cfg).__name__}")

    unknown = set(cfg.keys()) - KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(KNOWN_TOP_LEVEL_KEYS))}"
        )

    task_contracts = cfg.get("task_contracts", [])
    if not task_contracts:
        logger.warning(
            "  Config key 'task_contracts' is absent or empty. "
            "No task contracts will be evaluated."
        )
        return

    _validate_agent_block(cfg)
    _validate_task_contracts(task_contracts, cfg)


def _validate_agent_block(cfg: dict[str, Any]) -> None:
    agent_cfg = cfg.get("agent", {})
    if not agent_cfg:
        return

    framework = agent_cfg.get("framework", "")

    if framework == "crewai":
        crew_class = agent_cfg.get("crew_class", "")
        if not crew_class:
            raise ValueError(
                "agent.crew_class is required when agent.framework is 'crewai'. "
                "Set crew_class to the dotted Python path of your @CrewBase class."
            )

    output_mode = agent_cfg.get("output_mode", "")
    if output_mode and output_mode not in ("auto", "pydantic", "raw"):
        logger.warning(
            f"agent.output_mode is '{output_mode}'. "
            f"Expected one of: auto, pydantic, raw. Falling back to 'auto'."
        )

    tool_map = agent_cfg.get("tool_map", {})
    if tool_map:
        for tool_name, verb in tool_map.items():
            if verb not in ("fetch", "search"):
                logger.warning(
                    f"agent.tool_map: '{tool_name}' maps to '{verb}'. "
                    f"Expected 'fetch' or 'search'. Using 'fetch' as fallback."
                )


def _validate_task_contracts(
    task_contracts: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    actors_declared = set(cfg.get("actors", {}).keys())
    roles_declared = set(cfg.get("roles", {}).keys())

    fixture_count = 0
    corpus_count = 0

    for tc in task_contracts:
        name = tc.get("name", "unnamed")

        preconditions = tc.get("preconditions", [])
        if not preconditions:
            raise ValueError(
                f"Task contract '{name}' has no preconditions. "
                f"At least one precondition is required for meaningful evaluation."
            )

        decision_field = tc.get("decision_field", "should_act")
        if decision_field not in ("should_act", "all_preconditions_pass", "reasoning"):
            logger.warning(
                f"Task contract '{name}': decision_field is '{decision_field}'. "
                f"The output schema includes 'should_act' but not '{decision_field}'. "
                f"The agent may produce a field the scorer cannot find."
            )

        allowed_tools = tc.get("allowed_tools", {})
        if allowed_tools:
            fixture_count += 1
            _validate_allowed_tools(name, allowed_tools, cfg.get("roles", {}))
        else:
            corpus_count += 1

        _validate_preconditions(name, preconditions, allowed_tools)

        actor = tc.get("actor")
        if actor and actors_declared and actor not in actors_declared:
            logger.warning(
                f"Task contract '{name}': actor '{actor}' is not declared "
                f"in the top-level 'actors' map. "
                f"Role resolution will fail and Perspective scoring will "
                f"not enforce boundaries."
            )

        role = tc.get("role")
        if role and roles_declared and role not in roles_declared:
            logger.warning(
                f"Task contract '{name}': role '{role}' is not declared "
                f"in the top-level 'roles' map. "
                f"Subsystem access will not be enforced."
            )

    if fixture_count > 0 and corpus_count > 0:
        logger.warning(
            f"Mixed evaluation modes: {fixture_count} fixture contract(s) "
            f"and {corpus_count} corpus contract(s). "
            f"Fixture contracts will use FixtureBackend; corpus contracts "
            f"will use FileCorpusAdapter from artifacts_dir."
        )


def _validate_allowed_tools(
    contract_name: str,
    allowed_tools: dict[str, dict[str, Any]],
    roles_declared: dict[str, dict[str, Any]] | None = None,
) -> None:
    roles_declared = roles_declared or {}
    for tool_name, tool_cfg in allowed_tools.items():
        returns = tool_cfg.get("returns", {})
        if not returns:
            logger.warning(
                f"Task contract '{contract_name}': allowed_tool '{tool_name}' "
                f"has an empty 'returns' dict. The fixture will return no "
                f"useful data, and scoring will produce zeros."
            )

        entity_arg = tool_cfg.get("entity_arg", "")
        artifact_id = tool_cfg.get("artifact_id", "")
        if not entity_arg and not artifact_id:
            logger.warning(
                f"Task contract '{contract_name}': allowed_tool '{tool_name}' "
                f"has neither 'entity_arg' nor 'artifact_id'. "
                f"The runtime will not know which artifact ID to resolve."
            )

        if entity_arg and entity_arg not in ("artifact_id",) and not artifact_id:
            logger.info(
                f"Task contract '{contract_name}': '{tool_name}' uses "
                f"entity_arg='{entity_arg}' but has no artifact_id. "
                f"The FixtureBackend will use the entity_arg value as the "
                f"artifact ID at runtime."
            )

        subsystem = tool_cfg.get("subsystem", "")
        roles_declared = {}
        if subsystem:
            found_in_any_role = False
            for role_name, role_cfg in roles_declared.items():
                if subsystem in role_cfg.get("subsystems", []):
                    found_in_any_role = True
                    break
            if roles_declared and not found_in_any_role:
                logger.warning(
                    f"Task contract '{contract_name}': allowed_tool '{tool_name}' "
                    f"declares subsystem '{subsystem}' which is not in any "
                    f"declared role's subsystems list. "
                    f"All actors will be gated from this artifact."
                )


def _validate_preconditions(
    contract_name: str,
    preconditions: list[dict[str, Any]],
    allowed_tools: dict[str, dict[str, Any]],
) -> None:
    is_fixture = bool(allowed_tools)

    declared_artifact_ids: set[str] = set()
    if is_fixture:
        for tool_cfg in allowed_tools.values():
            aid = tool_cfg.get("artifact_id") or ""
            if aid:
                declared_artifact_ids.add(aid)

    for pc in preconditions:
        check_name = pc.get("check", "unnamed")
        gt_field = pc.get("ground_truth_field", "")
        required_facts = pc.get("required_facts", [])

        if not gt_field:
            logger.warning(
                f"Task contract '{contract_name}': precondition "
                f"'{check_name}' has no ground_truth_field. "
                f"Counterfactual scoring will mark this as unsupported."
            )
            continue

        if "." not in gt_field:
            logger.warning(
                f"Task contract '{contract_name}': ground_truth_field "
                f"'{gt_field}' for precondition '{check_name}' is not in "
                f"'artifact_id.field_name' format. "
                f"Evidence resolution will fail."
            )
            continue

        artifact_ref = gt_field.split(".")[0]
        field_ref = gt_field.rsplit(".", 1)[-1]

        if (
            is_fixture
            and declared_artifact_ids
            and artifact_ref not in declared_artifact_ids
        ):
            logger.warning(
                f"Task contract '{contract_name}': ground_truth_field "
                f"'{gt_field}' references artifact '{artifact_ref}' which "
                f"is not declared in any allowed_tools artifact_id. "
                f"Evidence resolution will return None."
            )

        if required_facts and field_ref not in required_facts:
            logger.warning(
                f"Task contract '{contract_name}': ground_truth_field "
                f"'{gt_field}' resolves to field '{field_ref}', but this "
                f"field is not in required_facts: {required_facts}. "
                f"The uncovered-facts loop will not check this field."
            )

        if not required_facts:
            logger.warning(
                f"Task contract '{contract_name}': precondition "
                f"'{check_name}' has an empty required_facts list. "
                f"Silence scoring will have nothing to verify."
            )
