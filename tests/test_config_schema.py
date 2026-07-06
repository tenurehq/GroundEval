import logging

import pytest

from groundeval.config_schema import KNOWN_TOP_LEVEL_KEYS, validate_config


def _task_contract(**overrides):
    base = {
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    }
    base.update(overrides)
    return base


def test_known_top_level_keys_cover_current_schema():
    expected = {
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
        "groundeval",
    }
    assert expected.issubset(KNOWN_TOP_LEVEL_KEYS)


def test_validate_config_rejects_non_mapping():
    with pytest.raises(TypeError, match="Config must be a YAML mapping"):
        validate_config([], command="task")


def test_validate_config_rejects_unknown_top_level_key():
    cfg = {"task_contracts": [], "unknown_key": 1}
    with pytest.raises(ValueError, match="Unknown config key"):
        validate_config(cfg, command="task")


def test_validate_config_error_lists_multiple_unknown_keys():
    cfg = {
        "task_contracts": [],
        "zzz": 1,
        "aaa": 2,
    }
    with pytest.raises(ValueError) as exc:
        validate_config(cfg, command="task")
    msg = str(exc.value)
    assert "aaa" in msg
    assert "zzz" in msg


def test_validate_config_warns_and_returns_when_task_contracts_missing(caplog):
    cfg = {"seed": 1}
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "task_contracts" in caplog.text
    assert "absent or empty" in caplog.text


def test_validate_config_warns_and_returns_when_task_contracts_empty(caplog):
    cfg = {"task_contracts": []}
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "task_contracts" in caplog.text
    assert "absent or empty" in caplog.text


def test_validate_config_accepts_broad_valid_top_level_config():
    cfg = {
        "agent": {"framework": "custom"},
        "output_dir": "./out",
        "artifacts_dir": "./data",
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
        "provider": "openai",
        "provider_path": "pkg.module.Provider",
        "model": "gpt-4o",
        "api_key": "x",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://example.com",
        "temperature": 0.1,
        "max_tokens": 123,
        "max_retries": 3,
        "seed": 7,
        "groundeval": {"config_status": "draft"},
        "task_contracts": [_task_contract()],
    }
    validate_config(cfg, command="task")


def test_validate_config_agent_block_absent_passes():
    validate_config({"task_contracts": [_task_contract()]}, command="task")


def test_validate_config_empty_agent_block_passes():
    validate_config({"agent": {}, "task_contracts": [_task_contract()]}, command="task")


def test_validate_config_crewai_requires_agent_class():
    cfg = {
        "agent": {"framework": "crewai"},
        "task_contracts": [_task_contract()],
    }
    with pytest.raises(ValueError, match="agent.agent_class is required"):
        validate_config(cfg, command="task")


def test_validate_config_maf_requires_agent_class():
    cfg = {
        "agent": {"framework": "maf"},
        "task_contracts": [_task_contract()],
    }
    with pytest.raises(ValueError, match="agent.agent_class is required"):
        validate_config(cfg, command="task")


def test_validate_config_crewai_with_agent_class_passes():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [_task_contract()],
    }
    validate_config(cfg, command="task")


def test_validate_config_maf_with_agent_class_passes():
    cfg = {
        "agent": {"framework": "maf", "agent_class": "x.y.Agent"},
        "task_contracts": [_task_contract()],
    }
    validate_config(cfg, command="task")


def test_validate_config_bad_output_mode_warns(caplog):
    cfg = {
        "agent": {"output_mode": "xml"},
        "task_contracts": [_task_contract()],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "agent.output_mode" in caplog.text


@pytest.mark.parametrize("mode", ["auto", "pydantic", "raw"])
def test_validate_config_good_output_modes_do_not_warn(mode, caplog):
    cfg = {
        "agent": {"output_mode": mode},
        "task_contracts": [_task_contract()],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "agent.output_mode" not in caplog.text


def test_validate_config_task_contract_requires_preconditions_or_multi_agent_requirements():
    cfg = {
        "task_contracts": [{"name": "t1", "preconditions": []}],
    }
    with pytest.raises(
        ValueError, match="has no preconditions and no multi-agent requirements"
    ):
        validate_config(cfg, command="task")


def test_validate_config_empty_preconditions_allowed_with_required_agents():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agents": [{"agent_name": "planner"}],
            }
        ],
    }
    validate_config(cfg, command="observe")


def test_validate_config_empty_preconditions_allowed_with_required_handoffs():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_handoffs": [{"from_agent": "a", "to_agent": "b"}],
            }
        ],
    }
    validate_config(cfg, command="observe")


def test_validate_config_empty_preconditions_allowed_with_required_agent_tool_expectations():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agent_tool_expectations": [
                    {"agent_name": "planner", "tool": "fetch_customer"}
                ],
            }
        ],
    }
    validate_config(cfg, command="observe")


@pytest.mark.parametrize(
    "decision_field",
    ["should_act", "all_preconditions_pass", "should_escalate", "reasoning"],
)
def test_validate_config_allowed_decision_fields_do_not_warn(decision_field, caplog):
    cfg = {
        "task_contracts": [_task_contract(decision_field=decision_field)],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "decision_field" not in caplog.text


def test_validate_config_custom_decision_field_warns(caplog):
    cfg = {
        "task_contracts": [_task_contract(decision_field="custom_flag")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "decision_field" in caplog.text
    assert "custom_flag" in caplog.text


def test_validate_config_tool_expectations_must_be_list():
    cfg = {
        "task_contracts": [
            _task_contract(tool_expectations={"tool": "fetch_customer"})
        ],
    }
    with pytest.raises(ValueError, match="tool_expectations must be a list"):
        validate_config(cfg, command="observe")


def test_validate_config_tool_expectation_requires_tool_name():
    cfg = {
        "task_contracts": [_task_contract(tool_expectations=[{"match_args": {}}])],
    }
    with pytest.raises(ValueError, match="has no tool name"):
        validate_config(cfg, command="observe")


def test_validate_config_tool_expectation_missing_expected_return_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(tool_expectations=[{"tool": "fetch_customer"}])
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="observe")
    assert "has no expected_return" in caplog.text


def test_validate_config_tool_expectation_match_args_must_be_mapping():
    cfg = {
        "task_contracts": [
            _task_contract(
                tool_expectations=[{"tool": "fetch_customer", "match_args": ["bad"]}]
            )
        ],
    }
    with pytest.raises(ValueError, match="match_args must be a mapping"):
        validate_config(cfg, command="observe")


def test_validate_config_tool_expectation_expected_return_must_be_mapping():
    cfg = {
        "task_contracts": [
            _task_contract(
                tool_expectations=[
                    {"tool": "fetch_customer", "expected_return": ["bad"]}
                ]
            )
        ],
    }
    with pytest.raises(ValueError, match="expected_return must be a mapping"):
        validate_config(cfg, command="observe")


def test_validate_config_valid_tool_expectation_passes():
    cfg = {
        "task_contracts": [
            _task_contract(
                tool_expectations=[
                    {
                        "tool": "fetch_customer",
                        "match_args": {"artifact_id": "crm-1"},
                        "expected_return": {"status": "active"},
                    }
                ]
            )
        ],
    }
    validate_config(cfg, command="observe")


def test_validate_config_allowed_tool_empty_returns_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                allowed_tools={
                    "fetch_customer": {"returns": {}},
                }
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "empty 'returns' dict" in caplog.text


def test_validate_config_allowed_tool_missing_entity_arg_and_artifact_id_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                allowed_tools={
                    "fetch_customer": {"returns": {"status": "active"}},
                }
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "neither 'entity_arg' nor 'artifact_id'" in caplog.text


def test_validate_config_allowed_tool_nonstandard_entity_arg_logs_info(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                allowed_tools={
                    "fetch_customer": {
                        "entity_arg": "customer_id",
                        "returns": {"status": "active"},
                    }
                }
            )
        ],
    }
    with caplog.at_level(logging.INFO):
        validate_config(cfg, command="task")
    assert "entity_arg='customer_id'" in caplog.text


def test_validate_config_allowed_tool_subsystem_not_in_any_role_warns(caplog):
    cfg = {
        "roles": {"sales": {"subsystems": ["crm"]}},
        "task_contracts": [
            _task_contract(
                allowed_tools={
                    "fetch_customer": {
                        "artifact_id": "crm-1",
                        "subsystem": "email",
                        "returns": {"status": "active"},
                    }
                }
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "not in any declared role's subsystems list" in caplog.text


def test_validate_config_allowed_tool_subsystem_in_declared_role_does_not_warn(caplog):
    cfg = {
        "roles": {"sales": {"subsystems": ["crm", "email"]}},
        "task_contracts": [
            _task_contract(
                allowed_tools={
                    "fetch_customer": {
                        "artifact_id": "crm-1",
                        "subsystem": "email",
                        "returns": {"status": "active"},
                    }
                }
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "not in any declared role's subsystems list" not in caplog.text


def test_validate_config_precondition_missing_ground_truth_field_warns(caplog):
    cfg = {
        "task_contracts": [_task_contract(preconditions=[{"check": "pc1"}])],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "has no ground_truth_field" in caplog.text


def test_validate_config_precondition_bad_ground_truth_field_format_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[{"check": "pc1", "ground_truth_field": "status"}]
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "artifact_id.field_name" in caplog.text


def test_validate_config_fixture_precondition_unknown_artifact_reference_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[
                    {
                        "check": "pc1",
                        "ground_truth_field": "missing.status",
                        "required_facts": ["status"],
                    }
                ],
                allowed_tools={
                    "fetch_customer": {
                        "artifact_id": "crm-1",
                        "returns": {"status": "active"},
                    }
                },
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "references artifact 'missing'" in caplog.text


def test_validate_config_precondition_field_not_in_required_facts_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[
                    {
                        "check": "pc1",
                        "ground_truth_field": "crm-1.status",
                        "required_facts": ["plan"],
                    }
                ],
                allowed_tools={
                    "fetch_customer": {
                        "artifact_id": "crm-1",
                        "returns": {"status": "active", "plan": "gold"},
                    }
                },
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "not in required_facts" in caplog.text


def test_validate_config_precondition_empty_required_facts_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[
                    {
                        "check": "pc1",
                        "ground_truth_field": "crm-1.status",
                        "required_facts": [],
                    }
                ]
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "empty required_facts list" in caplog.text


def test_validate_config_valid_fixture_precondition_no_related_warning(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[
                    {
                        "check": "pc1",
                        "ground_truth_field": "crm-1.status",
                        "required_facts": ["status"],
                    }
                ],
                allowed_tools={
                    "fetch_customer": {
                        "artifact_id": "crm-1",
                        "returns": {"status": "active"},
                    }
                },
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "ground_truth_field" not in caplog.text
    assert "required_facts" not in caplog.text


def test_validate_config_framework_precondition_required_tool_not_in_expectations_warns(
    caplog,
):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[
                    {
                        "check": "pc1",
                        "required_tool": "fetch_customer",
                    }
                ],
                tool_expectations=[
                    {"tool": "search_customer", "expected_return": {"x": 1}}
                ],
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="observe")
    assert "requires tool 'fetch_customer'" in caplog.text


def test_validate_config_framework_precondition_missing_required_tool_and_expected_field_warns(
    caplog,
):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[{"check": "pc1"}],
                tool_expectations=[
                    {"tool": "fetch_customer", "expected_return": {"x": 1}}
                ],
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="observe")
    assert "neither required_tool nor expected_field" in caplog.text


def test_validate_config_framework_precondition_with_expected_field_passes_without_that_warning(
    caplog,
):
    cfg = {
        "task_contracts": [
            _task_contract(
                preconditions=[{"check": "pc1", "expected_field": "status"}],
                tool_expectations=[
                    {"tool": "fetch_customer", "expected_return": {"status": "active"}}
                ],
            )
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="observe")
    assert "neither required_tool nor expected_field" not in caplog.text


def test_validate_config_required_agents_must_be_list():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agents": "bad",
            }
        ],
    }
    with pytest.raises(ValueError, match="required_agents must be a list"):
        validate_config(cfg, command="observe")


def test_validate_config_required_agents_items_must_be_mappings():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agents": ["bad"],
            }
        ],
    }
    with pytest.raises(ValueError, match="required_agents\\[1\\] must be a mapping"):
        validate_config(cfg, command="observe")


def test_validate_config_required_agents_item_needs_name_or_id():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agents": [{}],
            }
        ],
    }
    with pytest.raises(ValueError, match="must declare agent_name or agent_id"):
        validate_config(cfg, command="observe")


def test_validate_config_required_handoffs_must_be_list():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_handoffs": "bad",
            }
        ],
    }
    with pytest.raises(ValueError, match="required_handoffs must be a list"):
        validate_config(cfg, command="observe")


def test_validate_config_required_handoffs_items_must_be_mappings():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_handoffs": ["bad"],
            }
        ],
    }
    with pytest.raises(ValueError, match="required_handoffs\\[1\\] must be a mapping"):
        validate_config(cfg, command="observe")


def test_validate_config_required_handoffs_need_from_side():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_handoffs": [{"to_agent": "b"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="must declare from_agent or from_executor_id"):
        validate_config(cfg, command="observe")


def test_validate_config_required_handoffs_need_to_side():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_handoffs": [{"from_agent": "a"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="must declare to_agent or to_executor_id"):
        validate_config(cfg, command="observe")


def test_validate_config_required_agent_tool_expectations_must_be_list():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agent_tool_expectations": "bad",
            }
        ],
    }
    with pytest.raises(
        ValueError, match="required_agent_tool_expectations must be a list"
    ):
        validate_config(cfg, command="observe")


def test_validate_config_required_agent_tool_expectations_items_must_be_mappings():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agent_tool_expectations": ["bad"],
            }
        ],
    }
    with pytest.raises(
        ValueError, match="required_agent_tool_expectations\\[1\\] must be a mapping"
    ):
        validate_config(cfg, command="observe")


def test_validate_config_required_agent_tool_expectations_need_tool():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agent_tool_expectations": [{"agent_name": "planner"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="must declare tool"):
        validate_config(cfg, command="observe")


def test_validate_config_required_agent_tool_expectations_need_agent_name_or_id():
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agent_tool_expectations": [{"tool": "fetch_customer"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="must declare agent_name or agent_id"):
        validate_config(cfg, command="observe")


def test_validate_config_multi_agent_requirements_warn_for_non_framework_native(caplog):
    cfg = {
        "agent": {"framework": "custom"},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [],
                "required_agents": [{"agent_name": "planner"}],
            }
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="observe")
    assert "multi-agent requirements are declared" in caplog.text
    assert "framework 'custom'" in caplog.text


def test_validate_config_mixed_fixture_and_corpus_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                name="fixture",
                allowed_tools={
                    "fetch_customer": {
                        "artifact_id": "crm-1",
                        "returns": {"status": "active"},
                    }
                },
            ),
            _task_contract(name="corpus"),
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "Mixed evaluation modes" in caplog.text


def test_validate_config_mixed_framework_and_corpus_warns(caplog):
    cfg = {
        "task_contracts": [
            _task_contract(
                name="framework",
                tool_expectations=[
                    {"tool": "fetch_customer", "expected_return": {"status": "active"}}
                ],
            ),
            _task_contract(name="corpus"),
        ],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="observe")
    assert "Mixed framework-native and corpus contracts detected" in caplog.text


def test_validate_config_observe_logs_framework_native_scoring_info_for_crewai(caplog):
    cfg = {
        "agent": {"framework": "crewai", "agent_class": "x.y.Agent"},
        "task_contracts": [
            _task_contract(name="corpus"),
        ],
    }
    with caplog.at_level(logging.INFO):
        validate_config(cfg, command="observe")
    assert "framework-native scoring" in caplog.text.lower()


def test_validate_config_observe_logs_framework_native_scoring_info_for_maf(caplog):
    cfg = {
        "agent": {"framework": "maf", "agent_class": "x.y.Agent"},
        "task_contracts": [
            _task_contract(name="corpus"),
        ],
    }
    with caplog.at_level(logging.INFO):
        validate_config(cfg, command="observe")
    assert "framework-native scoring" in caplog.text.lower()


def test_validate_config_actor_not_declared_warns(caplog):
    cfg = {
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
        "task_contracts": [_task_contract(actor="bob")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "actor 'bob' is not declared" in caplog.text


def test_validate_config_actor_declared_does_not_warn(caplog):
    cfg = {
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
        "task_contracts": [_task_contract(actor="alice")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "is not declared in the top-level 'actors' map" not in caplog.text


def test_validate_config_role_not_declared_warns(caplog):
    cfg = {
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
        "task_contracts": [_task_contract(role="admin")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "role 'admin' is not declared" in caplog.text


def test_validate_config_role_declared_does_not_warn(caplog):
    cfg = {
        "actors": {"alice": "engineer"},
        "roles": {"engineer": {"subsystems": ["jira"]}},
        "task_contracts": [_task_contract(role="engineer")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "is not declared in the top-level 'roles' map" not in caplog.text


def test_validate_config_actor_warning_not_emitted_when_no_actors_declared(caplog):
    cfg = {
        "roles": {"engineer": {"subsystems": ["jira"]}},
        "task_contracts": [_task_contract(actor="alice")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "is not declared in the top-level 'actors' map" not in caplog.text


def test_validate_config_role_warning_not_emitted_when_no_roles_declared(caplog):
    cfg = {
        "actors": {"alice": "engineer"},
        "task_contracts": [_task_contract(role="engineer")],
    }
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "is not declared in the top-level 'roles' map" not in caplog.text
