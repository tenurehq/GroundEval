import logging

import pytest

from groundeval.config_schema import (
    KNOWN_TOP_LEVEL_KEYS,
    validate_config,
)


def test_known_keys_includes_new_keys():
    """The new task-contract keys are present."""
    assert "task_contracts" in KNOWN_TOP_LEVEL_KEYS
    assert "seed" in KNOWN_TOP_LEVEL_KEYS
    assert "actors" in KNOWN_TOP_LEVEL_KEYS
    assert "roles" in KNOWN_TOP_LEVEL_KEYS


def test_known_keys_excludes_old_keys():
    """Old three-question model keys are gone."""
    assert "causal_links" not in KNOWN_TOP_LEVEL_KEYS
    assert "silence_pairs" not in KNOWN_TOP_LEVEL_KEYS
    assert "perspective" not in KNOWN_TOP_LEVEL_KEYS
    assert "perspective_actors" not in KNOWN_TOP_LEVEL_KEYS
    assert "use_event_log_policy" not in KNOWN_TOP_LEVEL_KEYS
    assert "easy_ratio" not in KNOWN_TOP_LEVEL_KEYS
    assert "max_perspective_questions" not in KNOWN_TOP_LEVEL_KEYS
    assert "max_counterfactual_questions" not in KNOWN_TOP_LEVEL_KEYS
    assert "max_silence_questions" not in KNOWN_TOP_LEVEL_KEYS
    assert "llm_question_prose" not in KNOWN_TOP_LEVEL_KEYS
    assert "llm_model" not in KNOWN_TOP_LEVEL_KEYS


def test_validate_config_accepts_valid_keys():
    """Valid config with all known keys passes."""
    cfg = {
        "task_contracts": [
            {"name": "t1", "preconditions": [{"check": "pc1", "description": "d"}]}
        ],
        "actors": {"agent": "sales_rep"},
        "roles": {"sales_rep": {"subsystems": ["crm"]}},
        "seed": 42,
    }
    validate_config(cfg, command="task")


def test_validate_config_rejects_unknown_keys():
    """Unknown top-level keys raise ValueError."""
    cfg = {
        "task_contracts": [],
        "not_a_real_key": "oops",
    }
    with pytest.raises(ValueError, match="Unknown config key"):
        validate_config(cfg, command="task")


def test_validate_config_rejects_multiple_unknown_keys():
    """All unknown keys are reported in error message."""
    cfg = {
        "task_contracts": [],
        "causal_links": [],
        "silence_pairs": [],
        "perspective": {},
    }
    with pytest.raises(ValueError, match="Unknown config key"):
        validate_config(cfg, command="task")


def test_validate_config_accepts_minimal_config():
    """Minimal config with just task_contracts passes."""
    cfg = {"task_contracts": []}
    validate_config(cfg, command="task")  # Should not raise, but may warn


def test_validate_config_warns_empty_contracts(caplog):
    """Warns when task_contracts is empty."""
    cfg = {"task_contracts": []}
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "absent or empty" in caplog.text


def test_validate_config_warns_missing_contracts(caplog):
    """Warns when task_contracts key is entirely missing."""
    cfg = {"seed": 42}
    with caplog.at_level(logging.WARNING):
        validate_config(cfg, command="task")
    assert "absent or empty" in caplog.text


def test_validate_config_rejects_non_dict():
    """Non-dict input raises TypeError."""
    with pytest.raises(TypeError, match="must be a YAML mapping"):
        validate_config(["not", "a", "dict"], command="task")  # type: ignore


def test_validate_config_accepts_all_provider_keys():
    """All provider-related keys are accepted."""
    cfg = {
        "task_contracts": [],
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key": "sk-xxx",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.example.com",
        "temperature": 0.2,
        "max_tokens": 2048,
        "max_retries": 5,
    }
    validate_config(cfg, command="task")  # Should not raise


def test_validate_config_accepts_output_keys():
    """Output directory key is accepted."""
    cfg = {
        "task_contracts": [],
        "output_dir": "./my_results",
        "artifacts_dir": "./my_artifacts",
    }
    validate_config(cfg, command="task")  # Should not raise


class TestAgentBlockValidation:
    def test_crewai_without_agent_class_raises(self):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
            "agent": {"framework": "crewai"},
        }
        with pytest.raises(ValueError, match="agent.agent_class is required"):
            validate_config(cfg, command="task")

    def test_crewai_with_agent_class_passes(self):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
            "agent": {"framework": "crewai", "agent_class": "my_project.crew.MyCrew"},
        }
        validate_config(cfg, command="task")

    def test_bad_output_mode_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
            "agent": {"output_mode": "xml"},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "output_mode" in caplog.text

    def test_tool_map_bad_verb_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
            "agent": {"tool_map": {"fetch_customer": "send", "search_docs": "fetch"}},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "tool_map" in caplog.text

    def test_tool_map_good_verbs_no_warning(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
            "agent": {"tool_map": {"fetch_customer": "fetch", "search_docs": "search"}},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "tool_map" not in caplog.text

    def test_no_agent_block_passes(self):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
        }
        validate_config(cfg, command="task")

    def test_agent_block_empty_passes(self):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                }
            ],
            "agent": {},
        }
        validate_config(cfg, command="task")


class TestTaskContractValidation:
    def test_no_preconditions_raises(self):
        cfg = {
            "task_contracts": [
                {"name": "t1", "preconditions": []},
            ],
        }
        with pytest.raises(ValueError, match="has no preconditions"):
            validate_config(cfg, command="task")

    def test_good_decision_field_passes(self):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "decision_field": "should_act",
                },
            ],
        }
        validate_config(cfg, command="task")

    def test_unknown_decision_field_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "decision_field": "should_escalate",
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "decision_field" in caplog.text

    def test_actor_not_declared_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "actor": "bob",
                },
            ],
            "actors": {"alice": "engineer"},
            "roles": {"engineer": {"subsystems": ["crm"]}},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "actor 'bob' is not declared" in caplog.text

    def test_actor_declared_passes(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "actor": "alice",
                },
            ],
            "actors": {"alice": "engineer"},
            "roles": {"engineer": {"subsystems": ["crm"]}},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "actor" not in caplog.text

    def test_role_not_declared_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "role": "admin",
                },
            ],
            "actors": {"alice": "engineer"},
            "roles": {"engineer": {"subsystems": ["crm"]}},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "role 'admin' is not declared" in caplog.text

    def test_role_declared_passes(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "role": "engineer",
                },
            ],
            "actors": {"alice": "engineer"},
            "roles": {"engineer": {"subsystems": ["crm"]}},
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "role" not in caplog.text

    def test_mixed_fixture_and_corpus_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "fixture_task",
                    "preconditions": [{"check": "pc1", "description": "d"}],
                    "allowed_tools": {
                        "fetch_customer": {
                            "artifact_id": "crm",
                            "returns": {"status": "active"},
                        }
                    },
                },
                {
                    "name": "corpus_task",
                    "preconditions": [{"check": "pc2", "description": "d"}],
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "Mixed evaluation modes" in caplog.text


class TestAllowedToolsValidation:
    def test_empty_returns_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "artifact_id": "crm",
                            "returns": {},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "empty 'returns' dict" in caplog.text

    def test_no_entity_arg_and_no_artifact_id_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "returns": {"status": "active"},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "neither 'entity_arg' nor 'artifact_id'" in caplog.text

    def test_entity_arg_without_artifact_id_logs_info(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "entity_arg": "customer_id",
                            "returns": {"status": "active"},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.INFO):
            validate_config(cfg, command="task")
        assert "entity_arg" in caplog.text

    def test_valid_allowed_tool_passes(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "artifact_id": "crm",
                            "returns": {"status": "active"},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "allowed_tool" not in caplog.text


class TestPreconditionsValidation:
    def test_no_ground_truth_field_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {"check": "pc1", "description": "d"},
                    ],
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "no ground_truth_field" in caplog.text

    def test_ground_truth_field_no_dot_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "status",
                        },
                    ],
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "not in 'artifact_id.field_name' format" in caplog.text

    def test_artifact_ref_not_in_fixture_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "missing_artifact.status",
                            "required_facts": ["status"],
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "artifact_id": "crm",
                            "returns": {"status": "active"},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "references artifact 'missing_artifact'" in caplog.text

    def test_field_not_in_required_facts_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                            "required_facts": ["plan_tier"],
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "artifact_id": "crm",
                            "returns": {"status": "active"},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "not in required_facts" in caplog.text

    def test_empty_required_facts_warns(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                            "required_facts": [],
                        }
                    ],
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert "empty required_facts" in caplog.text

    def test_valid_precondition_passes(self, caplog):
        cfg = {
            "task_contracts": [
                {
                    "name": "t1",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "d",
                            "ground_truth_field": "crm.status",
                            "required_facts": ["status"],
                        }
                    ],
                    "allowed_tools": {
                        "fetch_customer": {
                            "artifact_id": "crm",
                            "returns": {"status": "active"},
                        }
                    },
                },
            ],
        }
        with caplog.at_level(logging.WARNING):
            validate_config(cfg, command="task")
        assert caplog.text == "" or "allowed_tool" not in caplog.text
