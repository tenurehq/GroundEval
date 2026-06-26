import argparse
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from groundeval.run import (
    _merge_with_defaults,
    _validate_config,
    cmd_task,
    _build_agent_fn,
    main,
)


# ── _merge_with_defaults ────────────────────────────────────


def test_merge_with_defaults_overrides():
    """main config values override defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        defaults_path = Path(tmp) / "config" / "evaluation.yaml"
        defaults_path.parent.mkdir(parents=True, exist_ok=True)
        defaults_path.write_text(yaml.dump({"seed": 1, "temperature": 0.5}))

        with patch("groundeval.run.Path", wraps=Path) as mock_path:
            # Only intercept the one specific path
            original = Path

            def _path_wrapper(p):
                if str(p) == "config/evaluation.yaml":
                    return original(defaults_path)
                return original(p)

            mock_path.side_effect = _path_wrapper

            result = _merge_with_defaults({"seed": 99, "provider": "anthropic"})

    assert result["seed"] == 99
    assert result["temperature"] == 0.5
    assert result["provider"] == "anthropic"


def test_merge_with_defaults_missing_file():
    """when defaults file is missing, main config returned as-is."""
    main_cfg = {"seed": 42, "provider": "openai"}
    with patch("pathlib.Path.exists", return_value=False):
        result = _merge_with_defaults(main_cfg)
    assert result == main_cfg


def test_merge_with_defaults_empty_main():
    """empty main config gets all defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        defaults_path = Path(tmp) / "config" / "evaluation.yaml"
        defaults_path.parent.mkdir(parents=True, exist_ok=True)
        defaults = {"seed": 42, "temperature": 0.0, "max_tokens": 1024}
        defaults_path.write_text(yaml.dump(defaults))

        with patch("groundeval.run.Path", wraps=Path) as mock_path:
            original = Path

            def _path_wrapper(p):
                if str(p) == "config/evaluation.yaml":
                    return original(defaults_path)
                return original(p)

            mock_path.side_effect = _path_wrapper

            result = _merge_with_defaults({})

    assert result["seed"] == 42
    assert result["temperature"] == 0.0


def test_validate_config_empty_contracts(caplog):
    """warns and raises when no task_contracts defined and no artifacts dir."""
    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _validate_config({"task_contracts": []})
    assert "No task_contracts defined" in caplog.text


def test_validate_config_with_contracts(caplog):
    """logs precondition count for each contract."""
    with tempfile.TemporaryDirectory() as tmp:
        art_dir = Path(tmp) / "task_artifacts"
        art_dir.mkdir()
        (art_dir / "dummy.json").write_text("{}")

        cfg = {
            "task_contracts": [
                {
                    "name": "task_a",
                    "preconditions": [
                        {"check": "pc1", "description": "Check one"},
                        {"check": "pc2", "description": "Check two"},
                    ],
                }
            ],
            "actors": {"agent": "sales_rep"},
            "roles": {"sales_rep": {"subsystems": ["crm"]}},
            "artifacts_dir": str(art_dir),
        }
        with caplog.at_level(logging.INFO):
            _validate_config(cfg)
    assert "task_a" in caplog.text
    assert "2 precondition" in caplog.text


def test_validate_config_missing_artifacts_dir(caplog):
    """raises when artifacts dir doesn't exist."""
    cfg = {"task_contracts": [], "artifacts_dir": "/nonexistent/path/12345"}
    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _validate_config(cfg)
    assert "No task_contracts defined" in caplog.text


def test_validate_config_with_artifacts(caplog):
    """logs artifact count when directory exists with json files."""
    with tempfile.TemporaryDirectory() as tmp:
        art_dir = Path(tmp) / "task_artifacts"
        art_dir.mkdir()
        (art_dir / "a.json").write_text('{"id": "a1"}')
        (art_dir / "b.json").write_text('{"id": "a2"}')
        (art_dir / "readme.txt").write_text("not json")

        cfg = {"task_contracts": [], "artifacts_dir": str(art_dir)}
        with caplog.at_level(logging.INFO):
            _validate_config(cfg)
        assert "2 JSON files" in caplog.text


def test_validate_config_actors_and_roles(caplog):
    """logs actor and role declarations."""
    with tempfile.TemporaryDirectory() as tmp:
        art_dir = Path(tmp) / "task_artifacts"
        art_dir.mkdir()
        (art_dir / "dummy.json").write_text("{}")

        cfg = {
            "task_contracts": [],
            "actors": {"agent": "sales_rep", "auditor": "compliance"},
            "roles": {
                "sales_rep": {"subsystems": ["crm", "email"]},
                "compliance": {"subsystems": ["audit_trail"]},
            },
            "artifacts_dir": str(art_dir),
        }
        with caplog.at_level(logging.INFO):
            _validate_config(cfg)
    assert "2 declared" in caplog.text
    assert "sales_rep" in caplog.text
    assert "compliance" in caplog.text


def test_build_agent_fn_returns_callable():
    """returns a callable agent function."""
    cfg = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
    with patch("groundeval.providers.AnthropicProvider") as mock_prov:
        mock_instance = MagicMock()
        mock_prov.from_config.return_value = mock_instance
        fn = _build_agent_fn(cfg)
        assert callable(fn)


def test_build_agent_fn_uses_config_provider():
    """respects provider key in config."""
    cfg = {"provider": "openai", "model": "gpt-4o"}
    with patch("groundeval.providers.ModelProvider.from_config") as mock_from_config:
        mock_instance = MagicMock()
        mock_from_config.return_value = mock_instance
        _build_agent_fn(cfg)
        mock_from_config.assert_called_once_with(cfg)


def test_cmd_task_no_contracts_raises():
    """SystemExit when no task_contracts in config."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as f:
        yaml.dump({"task_contracts": []}, f)
        f.flush()

        args = argparse.Namespace(
            config=f.name,
            model="claude-sonnet-4-6",
            max_steps=10,
        )
        with pytest.raises(SystemExit, match="No task_contracts"):
            cmd_task(args)


def test_cmd_task_writes_results():
    """full run writes results file."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.yaml"
        artifacts_path = Path(tmp) / "task_artifacts"
        artifacts_path.mkdir()
        (artifacts_path / "crm_account.json").write_text(
            json.dumps({
                "id": "crm_account",
                "subsystem": "crm",
                "account_status": "active",
            })
        )

        config = {
            "output_dir": str(tmp),
            "artifacts_dir": str(artifacts_path),
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "actors": {"agent": "sales_rep"},
            "roles": {"sales_rep": {"subsystems": ["crm"]}},
            "task_contracts": [
                {
                    "name": "test_task",
                    "task_description": "Verify the customer status.",
                    "preconditions": [
                        {
                            "check": "customer_is_active",
                            "description": "Customer must be active.",
                            "ground_truth_field": "crm_account.account_status",
                        }
                    ],
                }
            ],
        }
        config_path.write_text(yaml.dump(config))

        # We need defaults to exist
        defaults_path = Path(tmp) / "config" / "evaluation.yaml"
        defaults_path.parent.mkdir(parents=True, exist_ok=True)
        defaults_path.write_text(yaml.dump({"seed": 42, "temperature": 0.0}))

        mock_trajectory = MagicMock()
        mock_trajectory.horizon_violations = 0
        mock_trajectory.actor_gate_violations = 0
        mock_trajectory.subsystem_violations = 0
        mock_trajectory.dead_ends_hit = 0
        mock_trajectory.dead_ends_recovered = 0
        mock_trajectory.tool_calls = []
        mock_trajectory.prompt_tokens = 10
        mock_trajectory.completion_tokens = 5
        mock_trajectory.budget_exceeded = False

        mock_trajectory.task_id = "test_task"

        mock_answer = {
            "preconditions_verified": [
                {
                    "check": "customer_is_active",
                    "passed": True,
                    "facts_found": {"account_status": "active"},
                    "evidence_artifacts": ["crm_account"],
                }
            ],
            "all_preconditions_pass": True,
            "reasoning": "Found active account.",
        }

        def _fake_agent_fn(question, context, tools, max_steps, runtime=None):
            return mock_trajectory, mock_answer

        with patch("groundeval.run._build_agent_fn", return_value=_fake_agent_fn):
            with patch("groundeval.run.Path", wraps=Path) as mock_path:
                original = Path

                def _path_wrapper(p):
                    s = str(p)
                    if s == "config/evaluation.yaml":
                        return original(defaults_path)
                    return original(p)

                mock_path.side_effect = _path_wrapper

                args = argparse.Namespace(
                    config=str(config_path),
                    model="claude-sonnet-4-6",
                    max_steps=5,
                )
                cmd_task(args)

        # Check results file was written
        results_files = list(Path(tmp).glob("task_results_*.json"))
        assert len(results_files) >= 1

        with open(results_files[0]) as f:
            data = json.load(f)
        assert data["meta"]["evaluation_mode"] == "task_contract"
        assert data["summary"]["n_tasks"] == 1
        assert "counterfactual_score" in data["summary"]


def test_cmd_task_respects_seed(caplog):
    """logs seed when set."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.yaml"
        artifacts_path = Path(tmp) / "task_artifacts"
        artifacts_path.mkdir()
        (artifacts_path / "a.json").write_text('{"id": "a", "subsystem": "crm"}')

        config = {
            "output_dir": str(tmp),
            "artifacts_dir": str(artifacts_path),
            "seed": 12345,
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "actors": {"agent": "sales_rep"},
            "roles": {"sales_rep": {"subsystems": ["crm"]}},
            "task_contracts": [
                {
                    "name": "t1",
                    "task_description": "Do the thing.",
                    "preconditions": [{"check": "pc1", "description": "Check one."}],
                }
            ],
        }
        config_path.write_text(yaml.dump(config))

        defaults_path = Path(tmp) / "config" / "evaluation.yaml"
        defaults_path.parent.mkdir(parents=True, exist_ok=True)
        defaults_path.write_text(yaml.dump({"seed": 1}))

        mock_t = MagicMock()
        mock_t.horizon_violations = 0
        mock_t.actor_gate_violations = 0
        mock_t.subsystem_violations = 0
        mock_t.dead_ends_hit = 0
        mock_t.dead_ends_recovered = 0
        mock_t.tool_calls = []
        mock_t.prompt_tokens = 0
        mock_t.completion_tokens = 0
        mock_t.budget_exceeded = False
        mock_t.task_id = "t1"
        mock_a = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": [],
                }
            ],
            "all_preconditions_pass": True,
            "reasoning": "ok",
        }

        def fake_agent(*args, **kwargs):
            return mock_t, mock_a

        with patch("groundeval.run._build_agent_fn", return_value=fake_agent):
            with patch("groundeval.run.Path", wraps=Path) as mock_path:
                original = Path

                def _pw(p):
                    if str(p) == "config/evaluation.yaml":
                        return original(defaults_path)
                    return original(p)

                mock_path.side_effect = _pw
                with caplog.at_level(logging.INFO):
                    args = argparse.Namespace(
                        config=str(config_path),
                        model="claude-sonnet-4-6",
                        max_steps=3,
                    )
                    cmd_task(args)

    assert "12345" in caplog.text


def test_cmd_task_multiple_contracts():
    """multiple contracts produce multiple results."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "config.yaml"
        artifacts_path = Path(tmp) / "task_artifacts"
        artifacts_path.mkdir()
        (artifacts_path / "a.json").write_text(
            '{"id": "a", "subsystem": "crm", "status": "ok"}'
        )

        config = {
            "output_dir": str(tmp),
            "artifacts_dir": str(artifacts_path),
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "actors": {"agent": "sales_rep"},
            "roles": {"sales_rep": {"subsystems": ["crm"]}},
            "task_contracts": [
                {
                    "name": "task_1",
                    "task_description": "First task.",
                    "preconditions": [
                        {
                            "check": "pc1",
                            "description": "A",
                            "ground_truth_field": "a.status",
                        }
                    ],
                },
                {
                    "name": "task_2",
                    "task_description": "Second task.",
                    "preconditions": [{"check": "pc1", "description": "B"}],
                },
            ],
        }
        config_path.write_text(yaml.dump(config))

        defaults_path = Path(tmp) / "config" / "evaluation.yaml"
        defaults_path.parent.mkdir(parents=True, exist_ok=True)
        defaults_path.write_text(yaml.dump({"seed": 1}))

        mock_t = MagicMock()
        mock_t.horizon_violations = 0
        mock_t.actor_gate_violations = 0
        mock_t.subsystem_violations = 0
        mock_t.dead_ends_hit = 0
        mock_t.dead_ends_recovered = 0
        mock_t.tool_calls = []
        mock_t.prompt_tokens = 0
        mock_t.completion_tokens = 0
        mock_t.budget_exceeded = False
        mock_t.task_id = "t"
        mock_a = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": [],
                }
            ],
            "all_preconditions_pass": True,
            "reasoning": "ok",
        }

        def fake_agent(*args, **kwargs):
            return mock_t, mock_a

        with patch("groundeval.run._build_agent_fn", return_value=fake_agent):
            with patch("groundeval.run.Path", wraps=Path) as mock_path:
                original = Path

                def _pw(p):
                    if str(p) == "config/evaluation.yaml":
                        return original(defaults_path)
                    return original(p)

                mock_path.side_effect = _pw

                args = argparse.Namespace(
                    config=str(config_path),
                    model="claude-sonnet-4-6",
                    max_steps=3,
                )
                cmd_task(args)

        results_files = list(Path(tmp).glob("task_results_*.json"))
        assert len(results_files) >= 1
        with open(results_files[0]) as f:
            data = json.load(f)
        assert data["summary"]["n_tasks"] == 2


# ── main ────────────────────────────────────────────────────


def test_main_task_command():
    """main with 'task' command dispatches to cmd_task."""
    with patch("sys.argv", ["groundeval", "task", "--config", "/fake/config.yaml"]):
        with patch("groundeval.run.cmd_task") as mock_cmd:
            main()
            mock_cmd.assert_called_once()


def test_main_validate_command():
    """main with 'validate' command calls _validate_config."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w") as f:
        yaml.dump({"task_contracts": []}, f)
        f.flush()

        with patch("sys.argv", ["groundeval", "validate", "--config", f.name]):
            with patch("groundeval.run._validate_config") as mock_val:
                main()
                mock_val.assert_called_once()


def test_main_no_command_raises():
    """main exits when no subcommand given."""
    with patch("sys.argv", ["groundeval"]):
        with pytest.raises(SystemExit):
            main()
