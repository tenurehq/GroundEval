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


def test_merge_with_defaults_overrides(isolate_filesystem):
    """main config values override defaults."""
    sandbox = isolate_filesystem

    defaults_path = sandbox / "config" / "evaluation.yaml"
    defaults_path.parent.mkdir(parents=True, exist_ok=True)
    defaults_path.write_text(yaml.dump({"seed": 1, "temperature": 0.5}))

    with patch("groundeval.run.Path", wraps=Path) as mock_path:
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


def test_merge_with_defaults_empty_main(isolate_filesystem):
    """empty main config gets all defaults."""
    sandbox = isolate_filesystem

    defaults_path = sandbox / "config" / "evaluation.yaml"
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
            _validate_config({
                "task_contracts": [],
                "artifacts_dir": "/nonexistent/path/xyz789",
            })
    assert "No task_contracts defined" in caplog.text


def test_validate_config_with_contracts(isolate_filesystem, caplog):
    """logs precondition count for each contract."""
    sandbox = isolate_filesystem

    art_dir = sandbox / "task_artifacts"
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


def test_validate_config_with_artifacts(isolate_filesystem, caplog):
    """logs artifact count when directory exists with json files."""
    sandbox = isolate_filesystem

    art_dir = sandbox / "task_artifacts"
    art_dir.mkdir()
    (art_dir / "a.json").write_text('{"id": "a1"}')
    (art_dir / "b.json").write_text('{"id": "a2"}')
    (art_dir / "readme.txt").write_text("not json")

    cfg = {"task_contracts": [], "artifacts_dir": str(art_dir)}
    with caplog.at_level(logging.INFO):
        _validate_config(cfg)
    assert "2 JSON files" in caplog.text


def test_validate_config_actors_and_roles(isolate_filesystem, caplog):
    """logs actor and role declarations."""
    sandbox = isolate_filesystem

    art_dir = sandbox / "task_artifacts"
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


def test_cmd_task_no_contracts_raises(isolate_filesystem):
    """SystemExit when no task_contracts in config."""
    sandbox = isolate_filesystem

    art_dir = sandbox / "data"
    art_dir.mkdir()
    (art_dir / "dummy.json").write_text("{}")

    config_path = sandbox / "config.yaml"
    config_path.write_text(
        yaml.dump({
            "task_contracts": [],
            "artifacts_dir": str(art_dir),
        })
    )

    args = argparse.Namespace(
        config=str(config_path),
        model="claude-sonnet-4-6",
        max_steps=10,
    )
    with pytest.raises(SystemExit, match="No task_contracts"):
        cmd_task(args)


def test_cmd_task_writes_results(isolate_filesystem):
    """full run writes results file."""
    sandbox = isolate_filesystem

    config_path = sandbox / "config.yaml"
    artifacts_path = sandbox / "task_artifacts"
    artifacts_path.mkdir()
    (artifacts_path / "crm_account.json").write_text(
        json.dumps({
            "id": "crm_account",
            "subsystem": "crm",
            "account_status": "active",
        })
    )

    config = {
        "output_dir": str(sandbox),
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

    defaults_path = sandbox / "config" / "evaluation.yaml"
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

    results_files = list(sandbox.glob("task_results_*.json"))
    assert len(results_files) >= 1

    with open(results_files[0]) as f:
        data = json.load(f)
    assert data["meta"]["evaluation_mode"] == "task_contract"
    assert data["summary"]["n_tasks"] == 1
    assert "counterfactual_score" in data["summary"]


def test_cmd_task_respects_seed(isolate_filesystem, caplog):
    """logs seed when set."""
    sandbox = isolate_filesystem

    config_path = sandbox / "config.yaml"
    artifacts_path = sandbox / "task_artifacts"
    artifacts_path.mkdir()
    (artifacts_path / "a.json").write_text('{"id": "a", "subsystem": "crm"}')

    config = {
        "output_dir": str(sandbox),
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

    defaults_path = sandbox / "config" / "evaluation.yaml"
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


def test_cmd_task_multiple_contracts(isolate_filesystem):
    """multiple contracts produce multiple results."""
    sandbox = isolate_filesystem

    config_path = sandbox / "config.yaml"
    artifacts_path = sandbox / "task_artifacts"
    artifacts_path.mkdir()
    (artifacts_path / "a.json").write_text(
        '{"id": "a", "subsystem": "crm", "status": "ok"}'
    )

    config = {
        "output_dir": str(sandbox),
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

    defaults_path = sandbox / "config" / "evaluation.yaml"
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

    results_files = list(sandbox.glob("task_results_*.json"))
    assert len(results_files) >= 1
    with open(results_files[0]) as f:
        data = json.load(f)
    assert data["summary"]["n_tasks"] == 2


def test_main_task_command():
    """main with 'task' command dispatches to cmd_task."""
    with patch("sys.argv", ["groundeval", "task", "--config", "/fake/config.yaml"]):
        with patch("groundeval.run.cmd_task") as mock_cmd:
            main()
            mock_cmd.assert_called_once()


def test_main_validate_command(isolate_filesystem):
    """main with 'validate' command calls _validate_config."""
    sandbox = isolate_filesystem

    config_path = sandbox / "validate_test.yaml"
    config_path.write_text(yaml.dump({"task_contracts": []}))

    with patch("sys.argv", ["groundeval", "validate", "--config", str(config_path)]):
        with patch("groundeval.run._validate_config") as mock_val:
            main()
            mock_val.assert_called_once()


def test_main_no_command_raises():
    """main exits when no subcommand given."""
    with patch("sys.argv", ["groundeval"]):
        with pytest.raises(SystemExit):
            main()


def test_cmd_observe_creates_output(isolate_filesystem):
    sandbox = isolate_filesystem

    mock_crew = MagicMock()
    mock_agent = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "fetch_data"

    def fake_run(**kwargs):
        return {"result": "ok"}

    mock_tool._run = fake_run
    mock_agent.tools = [mock_tool]
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = '{"should_act": true}'
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        args = argparse.Namespace(
            command="observe",
            framework="crewai",
            agent_class="my.crew",
            no_draft=False,
            draft_mode="standard",
            output=str(sandbox),
            max_steps=10,
        )
        from groundeval.run import cmd_observe

        cmd_observe(args)

    assert (sandbox / "observed_run.json").exists()
    assert (sandbox / "observe_report.md").exists()
    assert (sandbox / "draft_config" / "config.yaml").exists()


def test_cmd_observe_no_draft_flag(tmp_path):
    mock_crew = MagicMock()
    mock_agent = MagicMock()
    mock_agent.tools = []
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = "{}"
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        args = argparse.Namespace(
            command="observe",
            framework="crewai",
            agent_class="my.crew",
            no_draft=True,
            draft_mode=None,
            output=str(tmp_path),
            max_steps=10,
        )
        from groundeval.run import cmd_observe

        cmd_observe(args)

    assert (tmp_path / "observed_run.json").exists()
    assert (tmp_path / "observe_report.md").exists()
    assert not (tmp_path / "draft_config").exists()


def test_cmd_draft_from_saved_run(tmp_path):
    run_data = {
        "run_id": "saved_run_001",
        "framework": "crewai",
        "agent_class": "my.crew.Class",
        "tool_calls": [
            {
                "tool_name": "fetch_customer",
                "arguments": {"id": "1"},
                "return_value": {"plan": "enterprise"},
                "latency_ms": 10.0,
            }
        ],
        "final_answer": {"should_act": True},
        "total_latency_ms": 100.0,
    }
    run_path = tmp_path / "observed_run.json"
    run_path.write_text(json.dumps(run_data))

    args = argparse.Namespace(
        command="draft",
        from_run=str(run_path),
        draft_mode="standard",
        output=str(tmp_path),
    )

    from groundeval.run import cmd_draft

    cmd_draft(args)

    assert (tmp_path / "draft_config" / "config.yaml").exists()
    assert (tmp_path / "draft_config" / "REVIEW.md").exists()


def test_cmd_draft_missing_file():
    args = argparse.Namespace(
        command="draft",
        from_run="/nonexistent/path/observed_run.json",
        draft_mode="standard",
        output=None,
    )
    from groundeval.run import cmd_draft

    with pytest.raises(FileNotFoundError, match="not found"):
        cmd_draft(args)


def test_cmd_draft_default_output_is_run_parent(tmp_path):
    sub = tmp_path / "runs"
    sub.mkdir()
    run_data = {
        "run_id": "r",
        "framework": "crewai",
        "agent_class": "x",
        "tool_calls": [],
        "final_answer": {},
        "total_latency_ms": 0,
    }
    run_path = sub / "observed_run.json"
    run_path.write_text(json.dumps(run_data))

    args = argparse.Namespace(
        command="draft",
        from_run=str(run_path),
        draft_mode="conservative",
        output=None,
    )

    from groundeval.run import cmd_draft

    cmd_draft(args)

    assert (sub / "draft_config" / "config.yaml").exists()


def test_cmd_observe_aggressive_mode_flag(tmp_path):
    mock_crew = MagicMock()
    mock_agent = MagicMock()
    mock_agent.tools = []
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = "{}"
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        args = argparse.Namespace(
            command="observe",
            framework="crewai",
            agent_class="my.crew",
            no_draft=False,
            draft_mode="aggressive",
            output=str(tmp_path),
            max_steps=5,
        )
        from groundeval.run import cmd_observe

        cmd_observe(args)

    with open(tmp_path / "draft_config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["groundeval"]["draft_mode"] == "aggressive"


def test_cmd_task_warns_on_draft_config(isolate_filesystem, caplog):
    sandbox = isolate_filesystem

    config_path = sandbox / "config.yaml"
    artifacts_path = sandbox / "task_artifacts"
    artifacts_path.mkdir()
    (artifacts_path / "a.json").write_text('{"id": "a", "subsystem": "crm"}')

    config = {
        "output_dir": str(sandbox),
        "artifacts_dir": str(artifacts_path),
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "actors": {"agent": "sales_rep"},
        "roles": {"sales_rep": {"subsystems": ["crm"]}},
        "groundeval": {
            "config_status": "draft",
            "generated_from_observation": True,
            "reviewed": False,
        },
        "task_contracts": [
            {
                "name": "t1",
                "task_description": "Do the thing.",
                "preconditions": [{"check": "pc1", "description": "A"}],
            }
        ],
    }
    config_path.write_text(yaml.dump(config))

    defaults_path = sandbox / "config" / "evaluation.yaml"
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

            with caplog.at_level(logging.WARNING):
                args = argparse.Namespace(
                    config=str(config_path),
                    model="claude-sonnet-4-6",
                    max_steps=3,
                    allow_draft_config=False,
                )
                cmd_task(args)

    assert "not been marked reviewed" in caplog.text


def test_cmd_task_allow_draft_config_suppresses_warning(isolate_filesystem, caplog):
    sandbox = isolate_filesystem

    config_path = sandbox / "config.yaml"
    artifacts_path = sandbox / "task_artifacts"
    artifacts_path.mkdir()
    (artifacts_path / "a.json").write_text('{"id": "a"}')

    config = {
        "output_dir": str(sandbox),
        "artifacts_dir": str(artifacts_path),
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "actors": {"agent": "x"},
        "roles": {"x": {"subsystems": ["crm"]}},
        "groundeval": {
            "config_status": "draft",
            "generated_from_observation": True,
            "reviewed": False,
        },
        "task_contracts": [
            {
                "name": "t",
                "task_description": "d",
                "preconditions": [{"check": "p", "description": "d"}],
            }
        ],
    }
    config_path.write_text(yaml.dump(config))

    defaults_path = sandbox / "config" / "evaluation.yaml"
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
                "check": "p",
                "passed": True,
                "facts_found": {},
                "evidence_artifacts": [],
            }
        ],
        "all_preconditions_pass": True,
        "reasoning": "",
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

            with caplog.at_level(logging.WARNING):
                args = argparse.Namespace(
                    config=str(config_path),
                    model="claude-sonnet-4-6",
                    max_steps=3,
                    allow_draft_config=True,
                )
                cmd_task(args)

    assert "not been marked reviewed" not in caplog.text


def test_main_observe_command(isolate_filesystem):
    sandbox = isolate_filesystem

    mock_crew = MagicMock()
    mock_agent = MagicMock()
    mock_agent.tools = []
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = "{}"
    mock_crew.kickoff.return_value = result_obj

    config_path = sandbox / "config.yaml"
    config_path.write_text("task_contracts: []")

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        with patch(
            "sys.argv",
            [
                "groundeval",
                "observe",
                "--framework",
                "crewai",
                "--agent-class",
                "my.crew.Class",
                "--output",
                str(sandbox),
                "--no-draft",
            ],
        ):
            main()

    assert (sandbox / "observed_run.json").exists()


def test_main_draft_command(isolate_filesystem):
    sandbox = isolate_filesystem

    run_data = {
        "run_id": "r",
        "framework": "crewai",
        "agent_class": "x",
        "tool_calls": [],
        "final_answer": {},
        "total_latency_ms": 0,
    }
    run_path = sandbox / "run.json"
    run_path.write_text(json.dumps(run_data))

    with patch(
        "sys.argv",
        [
            "groundeval",
            "draft",
            "--from-run",
            str(run_path),
            "--output",
            str(sandbox),
        ],
    ):
        main()

    assert (sandbox / "draft_config" / "config.yaml").exists()


def test_main_validate_mark_reviewed(isolate_filesystem):
    sandbox = isolate_filesystem

    config_path = sandbox / "config.yaml"
    config = {
        "task_contracts": [],
        "groundeval": {
            "config_status": "draft",
            "generated_from_observation": True,
            "reviewed": False,
        },
    }
    config_path.write_text(yaml.dump(config))

    art_dir = sandbox / "task_artifacts"
    art_dir.mkdir()
    (art_dir / "a.json").write_text("{}")

    with patch(
        "sys.argv",
        [
            "groundeval",
            "validate",
            "--config",
            str(config_path),
            "--mark-reviewed",
        ],
    ):
        with patch("groundeval.run._validate_config"):
            main()

    with open(str(config_path)) as f:
        updated = yaml.safe_load(f)

    assert updated["groundeval"]["config_status"] == "reviewed"
    assert updated["groundeval"]["reviewed"] is True
