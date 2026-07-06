from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger('groundeval.config_schema')

KNOWN_TOP_LEVEL_KEYS: set[str] = {'agent', 'output_dir', 'artifacts_dir', 'actors', 'roles', 'provider', 'provider_path', 'model', 'api_key', 'api_key_env', 'base_url', 'temperature', 'max_tokens', 'max_retries', 'task_contracts', 'seed', 'groundeval'}


def validate_config(cfg: dict[str, Any], *, command: str) -> None:
    if not isinstance(cfg, dict):
        raise TypeError(f'Config must be a YAML mapping, got {type(cfg).__name__}')
    unknown = set(cfg.keys()) - KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}. Known keys: {', '.join(sorted(KNOWN_TOP_LEVEL_KEYS))}")
    task_contracts = cfg.get('task_contracts', [])
    if not task_contracts:
        logger.warning("  Config key 'task_contracts' is absent or empty. No task contracts will be evaluated.")
        return
    _validate_agent_block(cfg)
    _validate_task_contracts(task_contracts, cfg, command=command)


def _validate_agent_block(cfg: dict[str, Any]) -> None:
    agent_cfg = cfg.get('agent', {})
    if not agent_cfg:
        return
    framework = agent_cfg.get('framework', '')
    agent_class = agent_cfg.get('agent_class', '')
    if framework in {'crewai', 'maf', 'langgraph'} and not agent_class:
        raise ValueError(f"agent.agent_class is required when agent.framework is '{framework}'. Set agent_class to the dotted Python path of your framework entry class.")
    output_mode = agent_cfg.get('output_mode', '')
    if output_mode and output_mode not in ('auto', 'pydantic', 'raw'):
        logger.warning(f"agent.output_mode is '{output_mode}'. Expected one of: auto, pydantic, raw. Falling back to 'auto'.")


def _validate_task_contracts(task_contracts: list[dict[str, Any]], cfg: dict[str, Any], *, command: str) -> None:
    actors_declared = set(cfg.get('actors', {}).keys())
    roles_declared = set(cfg.get('roles', {}).keys())
    fixture_count = 0
    corpus_count = 0
    tool_contract_count = 0
    framework = str(cfg.get('agent', {}).get('framework', '') or '').lower()
    for tc in task_contracts:
        name = tc.get('name', 'unnamed')
        preconditions = tc.get('preconditions', [])
        required_agents = tc.get('required_agents', [])
        required_handoffs = tc.get('required_handoffs', [])
        required_agent_tool_expectations = tc.get('required_agent_tool_expectations', [])
        has_multi_agent_requirements = any([bool(required_agents), bool(required_handoffs), bool(required_agent_tool_expectations)])
        if not preconditions and not has_multi_agent_requirements:
            raise ValueError(f"Task contract '{name}' has no preconditions and no multi-agent requirements. At least one precondition or one multi-agent requirement is required for meaningful evaluation.")
        decision_field = tc.get('decision_field', 'should_act')
        if decision_field not in ('should_act', 'all_preconditions_pass', 'should_escalate', 'reasoning'):
            logger.warning(f"Task contract '{name}': decision_field is '{decision_field}'. The output schema includes 'should_act' but not '{decision_field}'. The agent may produce a field the scorer cannot find.")
        allowed_tools = tc.get('allowed_tools', {})
        tool_expectations = tc.get('tool_expectations', [])
        if tool_expectations:
            tool_contract_count += 1
            _validate_tool_expectations(name, tool_expectations)
        elif allowed_tools:
            fixture_count += 1
            _validate_allowed_tools(name, allowed_tools, cfg.get('roles', {}))
        else:
            corpus_count += 1
        _validate_preconditions(name, preconditions, allowed_tools, tool_expectations)
        _validate_multi_agent_requirements(contract_name=name, tc=tc, framework=framework)
        actor = tc.get('actor')
        if actor and actors_declared and actor not in actors_declared:
            logger.warning(f"Task contract '{name}': actor '{actor}' is not declared in the top-level 'actors' map. Role resolution will fail and Perspective scoring will not enforce boundaries.")
        role = tc.get('role')
        if role and roles_declared and role not in roles_declared:
            logger.warning(f"Task contract '{name}': role '{role}' is not declared in the top-level 'roles' map. Subsystem access will not be enforced.")
    if command == 'observe' and framework in {'crewai', 'maf', 'langgraph'} and corpus_count > 0:
        logger.info('Framework observe scoring uses framework-native scoring and does not require artifacts_dir.')
    if fixture_count > 0 and corpus_count > 0:
        logger.warning(f'Mixed evaluation modes: {fixture_count} fixture contract(s) and {corpus_count} corpus contract(s). Fixture contracts will use FixtureBackend; corpus contracts will use FileCorpusAdapter from artifacts_dir.')
    if tool_contract_count > 0 and corpus_count > 0:
        logger.warning(f'Mixed framework-native and corpus contracts detected: {tool_contract_count} framework contract(s) and {corpus_count} corpus contract(s).')


def _validate_tool_expectations(contract_name: str, tool_expectations: list[dict[str, Any]]) -> None:
    if not isinstance(tool_expectations, list):
        raise ValueError(f"Task contract '{contract_name}': tool_expectations must be a list.")
    for i, exp in enumerate(tool_expectations, start=1):
        tool_name = exp.get('tool', '')
        if not tool_name:
            raise ValueError(f"Task contract '{contract_name}': tool_expectations[{i}] has no tool name.")
        if 'expected_return' not in exp:
            logger.warning(f"Task contract '{contract_name}': tool_expectation '{tool_name}' has no expected_return. Scoring will only verify the call occurred.")
        match_args = exp.get('match_args', {})
        if match_args and not isinstance(match_args, dict):
            raise ValueError(f"Task contract '{contract_name}': tool_expectation '{tool_name}' match_args must be a mapping.")
        expected_return = exp.get('expected_return', {})
        if expected_return and not isinstance(expected_return, dict):
            raise ValueError(f"Task contract '{contract_name}': tool_expectation '{tool_name}' expected_return must be a mapping.")


def _validate_allowed_tools(contract_name: str, allowed_tools: dict[str, dict[str, Any]], roles_declared: dict[str, dict[str, Any]] | None = None) -> None:
    roles_declared = roles_declared or {}
    for tool_name, tool_cfg in allowed_tools.items():
        returns = tool_cfg.get('returns', {})
        if not returns:
            logger.warning(f"Task contract '{contract_name}': allowed_tool '{tool_name}' has an empty 'returns' dict. The fixture will return no useful data, and scoring will produce zeros.")
        entity_arg = tool_cfg.get('entity_arg', '')
        artifact_id = tool_cfg.get('artifact_id', '')
        if not entity_arg and not artifact_id:
            logger.warning(f"Task contract '{contract_name}': allowed_tool '{tool_name}' has neither 'entity_arg' nor 'artifact_id'. The runtime will not know which artifact ID to resolve.")
        if entity_arg and entity_arg not in ('artifact_id',) and not artifact_id:
            logger.info(f"Task contract '{contract_name}': '{tool_name}' uses entity_arg='{entity_arg}' but has no artifact_id. The FixtureBackend will use the entity_arg value as the artifact ID at runtime.")
        subsystem = tool_cfg.get('subsystem', '')
        if subsystem:
            found_in_any_role = False
            for role_cfg in roles_declared.values():
                if subsystem in role_cfg.get('subsystems', []):
                    found_in_any_role = True
                    break
            if roles_declared and not found_in_any_role:
                logger.warning(f"Task contract '{contract_name}': allowed_tool '{tool_name}' declares subsystem '{subsystem}' which is not in any declared role's subsystems list. All actors will be gated from this artifact.")


def _validate_preconditions(contract_name: str, preconditions: list[dict[str, Any]], allowed_tools: dict[str, dict[str, Any]], tool_expectations: list[dict[str, Any]] | None = None) -> None:
    is_fixture = bool(allowed_tools)
    is_tool_contract = bool(tool_expectations)
    expected_tool_names = {exp.get('tool') for exp in (tool_expectations or []) if exp.get('tool')}
    declared_artifact_ids: set[str] = set()
    if is_fixture:
        for tool_cfg in allowed_tools.values():
            aid = tool_cfg.get('artifact_id') or ''
            if aid:
                declared_artifact_ids.add(aid)
    for pc in preconditions:
        check_name = pc.get('check', 'unnamed')
        gt_field = pc.get('ground_truth_field', '')
        required_facts = pc.get('required_facts', [])
        if is_tool_contract:
            required_tool = pc.get('required_tool', '')
            if required_tool and expected_tool_names and required_tool not in expected_tool_names:
                logger.warning(f"Task contract '{contract_name}': precondition '{check_name}' requires tool '{required_tool}', but that tool is not listed in tool_expectations.")
            expected_field = pc.get('expected_field', '')
            if not required_tool and not expected_field:
                logger.warning(f"Task contract '{contract_name}': precondition '{check_name}' has neither required_tool nor expected_field. It may not be scoreable in observe+score mode.")
            continue
        if not gt_field:
            logger.warning(f"Task contract '{contract_name}': precondition '{check_name}' has no ground_truth_field. Counterfactual scoring will mark this as unsupported.")
            continue
        if '.' not in gt_field:
            logger.warning(f"Task contract '{contract_name}': ground_truth_field '{gt_field}' for precondition '{check_name}' is not in 'artifact_id.field_name' format. Evidence resolution will fail.")
            continue
        artifact_ref = gt_field.split('.')[0]
        field_ref = gt_field.rsplit('.', 1)[-1]
        if is_fixture and declared_artifact_ids and artifact_ref not in declared_artifact_ids:
            logger.warning(f"Task contract '{contract_name}': ground_truth_field '{gt_field}' references artifact '{artifact_ref}' which is not declared in any allowed_tools artifact_id. Evidence resolution will return None.")
        if required_facts and field_ref not in required_facts:
            logger.warning(f"Task contract '{contract_name}': ground_truth_field '{gt_field}' resolves to field '{field_ref}', but this field is not in required_facts: {required_facts}. The uncovered-facts loop will not check this field.")
        if not required_facts:
            logger.warning(f"Task contract '{contract_name}': precondition '{check_name}' has an empty required_facts list. Silence scoring will have nothing to verify.")


def _validate_multi_agent_requirements(contract_name: str, tc: dict[str, Any], framework: str = '') -> None:
    required_agents = tc.get('required_agents', [])
    required_handoffs = tc.get('required_handoffs', [])
    required_agent_tool_expectations = tc.get('required_agent_tool_expectations', [])
    has_multi_agent_requirements = any([bool(required_agents), bool(required_handoffs), bool(required_agent_tool_expectations)])
    if required_agents and not isinstance(required_agents, list):
        raise ValueError(f"Task contract '{contract_name}': required_agents must be a list.")
    for i, item in enumerate(required_agents, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Task contract '{contract_name}': required_agents[{i}] must be a mapping.")
        if not item.get('agent_name') and not item.get('agent_id'):
            raise ValueError(f"Task contract '{contract_name}': required_agents[{i}] must declare agent_name or agent_id.")
    if required_handoffs and not isinstance(required_handoffs, list):
        raise ValueError(f"Task contract '{contract_name}': required_handoffs must be a list.")
    for i, item in enumerate(required_handoffs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Task contract '{contract_name}': required_handoffs[{i}] must be a mapping.")
        if not item.get('from_agent') and not item.get('from_executor_id'):
            raise ValueError(f"Task contract '{contract_name}': required_handoffs[{i}] must declare from_agent or from_executor_id.")
        if not item.get('to_agent') and not item.get('to_executor_id'):
            raise ValueError(f"Task contract '{contract_name}': required_handoffs[{i}] must declare to_agent or to_executor_id.")
    if required_agent_tool_expectations and not isinstance(required_agent_tool_expectations, list):
        raise ValueError(f"Task contract '{contract_name}': required_agent_tool_expectations must be a list.")
    for i, item in enumerate(required_agent_tool_expectations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Task contract '{contract_name}': required_agent_tool_expectations[{i}] must be a mapping.")
        if not item.get('tool'):
            raise ValueError(f"Task contract '{contract_name}': required_agent_tool_expectations[{i}] must declare tool.")
        if not item.get('agent_name') and not item.get('agent_id'):
            raise ValueError(f"Task contract '{contract_name}': required_agent_tool_expectations[{i}] must declare agent_name or agent_id.")
    if has_multi_agent_requirements and framework and framework not in {'maf', 'crewai', 'langgraph'}:
        logger.warning(f"Task contract '{contract_name}': multi-agent requirements are declared, but framework '{framework}' may not yet emit enough normalized agent, handoff, and per-agent tool-call metadata for full deterministic scoring.")
