import argparse
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from groundeval.run import (
    _build_agent_fn,
    _is_framework_config,
    _load_observe_score_config,
    _merge_with_defaults,
    _validate_config,
    cmd_draft,
    cmd_observe,
    cmd_task,
    main,
)


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


def test_is_framework_config_cases():
    assert _is_framework_config({"agent": {"framework": "crewai"}}) is True
    assert _is_framework_config({"agent": {"framework": "maf"}}) is True
    assert _is_framework_config({"agent": {"framework": "CREWAI"}}) is True
    assert _is_framework_config({"agent": {"framework": "custom"}}) is False
    assert _is_framework_config({"agent": {}}) is False
    assert _is_framework_config({}) is False


def test_merge_with_defaults_missing_file_returns_main_config():
    cfg = {"seed": 1, "provider": "openai"}
    with patch("pathlib.Path.exists", return_value=False):
        out = _merge_with_defaults(cfg)
    assert out == cfg
    assert out is not cfg


def test_merge_with_defaults_uses_defaults_and_main_overrides(tmp_path):
    defaults = tmp_path / "config" / "evaluation.yaml"
    _write_yaml(defaults, {"seed": 1, "temperature": 0.5, "model": "a"})

    with patch("groundeval.run.Path", wraps=Path) as mock_path:
        original = Path

        def side_effect(p):
            if str(p) == "config/evaluation.yaml":
                return original(defaults)
            return original(p)

        mock_path.side_effect = side_effect
        merged = _merge_with_defaults({"seed": 2, "provider": "openai"})

    assert merged["seed"] == 2
    assert merged["temperature"] == 0.5
    assert merged["model"] == "a"
    assert merged["provider"] == "openai"


def test_merge_with_defaults_requires_mapping(tmp_path):
    defaults = tmp_path / "config" / "evaluation.yaml"
    defaults.parent.mkdir(parents=True, exist_ok=True)
    defaults.write_text("- bad\n- list\n")

    with patch("groundeval.run.Path", wraps=Path) as mock_path:
        original = Path

        def side_effect(p):
            if str(p) == "config/evaluation.yaml":
                return original(defaults)
            return original(p)

        mock_path.side_effect = side_effect
        with pytest.raises(ValueError, match="evaluation.yaml must be a YAML mapping"):
            _merge_with_defaults({"seed": 1})


def test_load_observe_score_config_resolves_relative_artifacts_dir(tmp_path):
    cfg_path = tmp_path / "configs" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.dump({"artifacts_dir": "../data", "task_contracts": []}))

    out = _load_observe_score_config(cfg_path)
    assert Path(out["artifacts_dir"]).is_absolute()
    assert str(Path(out["artifacts_dir"])) == str(
        (cfg_path.parent / "../data").resolve()
    )


def test_load_observe_score_config_preserves_absolute_artifacts_dir(tmp_path):
    art = tmp_path / "data"
    art.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump({"artifacts_dir": str(art), "task_contracts": []}))

    out = _load_observe_score_config(cfg_path)
    assert out["artifacts_dir"] == str(art)


def test_load_observe_score_config_requires_mapping(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("- not\n- mapping\n")
    with pytest.raises(ValueError, match="Expected YAML mapping"):
        _load_observe_score_config(cfg_path)


def test_validate_config_logs_contract_info_and_fixture_mode(caplog):
    cfg = {
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [{"check": "pc1"}],
                "allowed_tools": {"fetch_customer": {"returns": {"status": "ok"}}},
            }
        ]
    }
    with caplog.at_level(logging.INFO):
        _validate_config(cfg, mode="task")
    assert "Task 't1': 1 precondition" in caplog.text
    assert "fixture mode" in caplog.text
    assert "Validation complete" in caplog.text


def test_validate_config_requires_artifacts_dir_for_task_mode(tmp_path):
    cfg = {
        "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
        "artifacts_dir": str(tmp_path / "missing"),
    }
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _validate_config(cfg, mode="task")


def test_validate_config_requires_json_files_when_corpus_needed(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "README.txt").write_text("hello")
    cfg = {
        "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
        "artifacts_dir": str(art),
    }
    with pytest.raises(FileNotFoundError, match="contains no JSON files"):
        _validate_config(cfg, mode="task")


def test_validate_config_observe_score_non_framework_requires_corpus(tmp_path):
    cfg = {
        "agent": {"framework": "custom"},
        "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
        "artifacts_dir": str(tmp_path / "missing"),
    }
    with pytest.raises(FileNotFoundError):
        _validate_config(cfg, mode="observe_score")


def test_validate_config_observe_score_framework_no_corpus_needed(caplog):
    cfg = {
        "agent": {"framework": "crewai"},
        "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
    }
    with caplog.at_level(logging.INFO):
        _validate_config(cfg, mode="observe_score")
    assert "not required for framework observe scoring" in caplog.text


def test_validate_config_logs_actors_roles(tmp_path, caplog):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "a.json").write_text("{}")
    cfg = {
        "task_contracts": [],
        "artifacts_dir": str(art),
        "actors": {"alice": "sales", "bob": "eng"},
        "roles": {
            "sales": {"subsystems": ["crm", "email"]},
            "eng": {"subsystems": ["jira"]},
        },
    }
    with caplog.at_level(logging.INFO):
        _validate_config(cfg, mode="task")
    assert "Actors: 2 declared" in caplog.text
    assert "Roles: 2 declared" in caplog.text
    assert "sales: ['crm', 'email']" in caplog.text
    assert "eng: ['jira']" in caplog.text


def test_build_agent_fn_rejects_crewai_task_mode():
    with pytest.raises(SystemExit, match="observe --score"):
        _build_agent_fn({"agent": {"framework": "crewai"}})


def test_build_agent_fn_rejects_maf_task_mode():
    with pytest.raises(SystemExit, match="observe --score"):
        _build_agent_fn({"agent": {"framework": "maf"}})


def test_build_agent_fn_uses_model_provider():
    fake_provider = object()
    fake_agent_fn = object()
    with patch(
        "groundeval.providers.ModelProvider.from_config", return_value=fake_provider
    ) as p1:
        with patch(
            "groundeval.providers.build_agent_fn", return_value=fake_agent_fn
        ) as p2:
            out = _build_agent_fn({"provider": "openai"})
    assert out is fake_agent_fn
    p1.assert_called_once()
    p2.assert_called_once_with(fake_provider)


def test_cmd_task_requires_config_mapping(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("- bad\n")
    args = argparse.Namespace(
        config=str(cfg_path),
        model="gpt-4o",
        max_steps=5,
        allow_draft_config=False,
    )
    with pytest.raises(ValueError, match="Expected YAML mapping"):
        cmd_task(args)


def test_cmd_task_no_task_contracts_exits(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, {"task_contracts": []})

    defaults = tmp_path / "config" / "evaluation.yaml"
    _write_yaml(defaults, {})

    args = argparse.Namespace(
        config=str(cfg_path),
        model="gpt-4o",
        max_steps=5,
        allow_draft_config=False,
    )

    with pytest.raises(SystemExit, match="No task_contracts defined"):
        cmd_task(args)


def test_cmd_task_warns_for_unreviewed_draft_config(tmp_path, caplog):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "a.json").write_text(json.dumps({"id": "a", "subsystem": "crm"}))

    cfg = {
        "output_dir": str(tmp_path),
        "artifacts_dir": str(art),
        "provider": "openai",
        "groundeval": {
            "generated_from_observation": True,
            "reviewed": False,
            "config_status": "draft",
        },
        "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
    }
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, cfg)

    defaults = tmp_path / "config" / "evaluation.yaml"
    _write_yaml(defaults, {})

    fake_result = MagicMock()
    fake_result.task_name = "t1"
    fake_result.counterfactual_score = 0.0
    fake_result.silence_score = 0.0
    fake_result.perspective_score = 0.0
    fake_result.overall_score = 0.0
    fake_result.horizon_violations = 0
    fake_result.actor_gate_violations = 0
    fake_result.subsystem_violations = 0

    with patch("groundeval.run._build_agent_fn", return_value=lambda **kwargs: None):
        with patch("groundeval.run.run_all_tasks", return_value=[fake_result]):
            with patch(
                "groundeval.scorers.aggregate_task_results",
                return_value={
                    "n_tasks": 1,
                    "counterfactual_score": 0.0,
                    "silence_score": 0.0,
                    "perspective_score": 0.0,
                    "overall_score": 0.0,
                    "accuracy": 0.0,
                    "total_violations": 0,
                    "per_task": [],
                },
            ):
                with patch("groundeval.run.Path", wraps=Path) as mock_path:
                    original = Path

                    def side_effect(p):
                        if str(p) == "config/evaluation.yaml":
                            return original(defaults)
                        return original(p)

                    mock_path.side_effect = side_effect

                    with caplog.at_level(logging.WARNING):
                        args = argparse.Namespace(
                            config=str(cfg_path),
                            model="gpt-4o",
                            max_steps=5,
                            allow_draft_config=False,
                        )
                        cmd_task(args)

    assert "not been marked reviewed" in caplog.text


def test_cmd_task_allow_draft_config_suppresses_warning(tmp_path, caplog):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "a.json").write_text(json.dumps({"id": "a", "subsystem": "crm"}))

    cfg = {
        "output_dir": str(tmp_path),
        "artifacts_dir": str(art),
        "provider": "openai",
        "groundeval": {
            "generated_from_observation": True,
            "reviewed": False,
            "config_status": "draft",
        },
        "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
    }
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, cfg)

    defaults = tmp_path / "config" / "evaluation.yaml"
    _write_yaml(defaults, {})

    fake_result = MagicMock()
    fake_result.task_name = "t1"
    fake_result.counterfactual_score = 0.0
    fake_result.silence_score = 0.0
    fake_result.perspective_score = 0.0
    fake_result.overall_score = 0.0
    fake_result.horizon_violations = 0
    fake_result.actor_gate_violations = 0
    fake_result.subsystem_violations = 0

    with patch("groundeval.run._build_agent_fn", return_value=lambda **kwargs: None):
        with patch("groundeval.run.run_all_tasks", return_value=[fake_result]):
            with patch(
                "groundeval.scorers.aggregate_task_results",
                return_value={
                    "n_tasks": 1,
                    "counterfactual_score": 0.0,
                    "silence_score": 0.0,
                    "perspective_score": 0.0,
                    "overall_score": 0.0,
                    "accuracy": 0.0,
                    "total_violations": 0,
                    "per_task": [],
                },
            ):
                with patch("groundeval.run.Path", wraps=Path) as mock_path:
                    original = Path

                    def side_effect(p):
                        if str(p) == "config/evaluation.yaml":
                            return original(defaults)
                        return original(p)

                    mock_path.side_effect = side_effect

                    with caplog.at_level(logging.WARNING):
                        args = argparse.Namespace(
                            config=str(cfg_path),
                            model="gpt-4o",
                            max_steps=5,
                            allow_draft_config=True,
                        )
                        cmd_task(args)

    assert "not been marked reviewed" not in caplog.text


def test_cmd_task_writes_results_file_and_uses_contract_roles(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "a.json").write_text(
        json.dumps({"id": "a", "subsystem": "crm", "status": "active"})
    )

    cfg = {
        "output_dir": str(tmp_path),
        "artifacts_dir": str(art),
        "provider": "openai",
        "model": "gpt-4o",
        "actors": {"top_actor": "top_role"},
        "roles": {"top_role": {"subsystems": ["email"]}},
        "task_contracts": [
            {
                "name": "t1",
                "preconditions": [{"check": "pc1"}],
                "actors": {"contract_actor": "contract_role"},
                "roles": {"contract_role": {"subsystems": ["crm"]}},
            }
        ],
    }
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, cfg)

    defaults = tmp_path / "config" / "evaluation.yaml"
    _write_yaml(defaults, {})

    fake_result = MagicMock()
    fake_result.task_name = "t1"
    fake_result.counterfactual_score = 1.0
    fake_result.silence_score = 1.0
    fake_result.perspective_score = 1.0
    fake_result.overall_score = 1.0
    fake_result.horizon_violations = 0
    fake_result.actor_gate_violations = 0
    fake_result.subsystem_violations = 0

    captured = {}

    def fake_run_all_tasks(**kwargs):
        captured.update(kwargs)
        return [fake_result]

    with patch("groundeval.run._build_agent_fn", return_value=lambda **kwargs: None):
        with patch("groundeval.run.run_all_tasks", side_effect=fake_run_all_tasks):
            with patch(
                "groundeval.scorers.aggregate_task_results",
                return_value={
                    "n_tasks": 1,
                    "counterfactual_score": 1.0,
                    "silence_score": 1.0,
                    "perspective_score": 1.0,
                    "overall_score": 1.0,
                    "accuracy": 1.0,
                    "total_violations": 0,
                    "per_task": [],
                },
            ):
                with patch("groundeval.run.Path", wraps=Path) as mock_path:
                    original = Path

                    def side_effect(p):
                        if str(p) == "config/evaluation.yaml":
                            return original(defaults)
                        return original(p)

                    mock_path.side_effect = side_effect

                    args = argparse.Namespace(
                        config=str(cfg_path),
                        model="gpt-4o",
                        max_steps=5,
                        allow_draft_config=False,
                    )
                    cmd_task(args)

    policy = captured["policy"]
    assert policy.role_for_actor("contract_actor") == "contract_role"
    files = list(tmp_path.glob("task_results_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["meta"]["evaluation_mode"] == "task_contract"
    assert payload["summary"]["n_tasks"] == 1


def test_cmd_observe_no_draft_writes_basic_outputs(tmp_path):
    fake_observed = MagicMock()
    fake_observed.to_dict.return_value = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "final_answer": {},
        "total_latency_ms": 0.0,
    }
    fake_observed.framework_extra = None
    fake_observed.tool_calls = []
    fake_observed.run_id = "r1"
    fake_observed.framework = "custom"
    fake_observed.agent_class = "pkg.Agent"
    fake_observed.total_latency_ms = 0.0

    with patch("groundeval.observe.observe_agent", return_value=fake_observed):
        args = argparse.Namespace(
            framework="custom",
            agent_class="pkg.Agent",
            no_draft=True,
            draft_mode="standard",
            output=str(tmp_path),
            max_steps=5,
            score=False,
            config=None,
            allow_draft_config=False,
        )
        cmd_observe(args)

    observed_runs = list(tmp_path.glob("observed_run_*.json"))
    reports = list(tmp_path.glob("observe_report_*.md"))
    diagrams = list(tmp_path.glob("observe_diagram_*.pdf"))

    assert len(observed_runs) == 1
    assert len(reports) == 1
    assert len(diagrams) == 1


def test_cmd_observe_with_draft_writes_draft_outputs(tmp_path):
    fake_observed = MagicMock()
    fake_observed.tool_calls = []
    fake_observed.run_id = "r1"

    with patch("groundeval.observe.observe_agent", return_value=fake_observed):
        with patch("groundeval.observe.DraftGenerator") as gen_cls:
            gen = MagicMock()
            gen_cls.return_value = gen
            with patch("groundeval.observe.write_draft_output") as writer:
                args = argparse.Namespace(
                    framework="custom",
                    agent_class="pkg.Agent",
                    no_draft=False,
                    draft_mode="aggressive",
                    output=str(tmp_path),
                    max_steps=5,
                    score=False,
                    config=None,
                    allow_draft_config=False,
                )
                cmd_observe(args)

    gen_cls.assert_called_once_with(fake_observed, mode="aggressive")
    writer.assert_called_once()


def test_cmd_observe_score_requires_config(tmp_path):
    fake_observed = MagicMock()
    fake_observed.tool_calls = []
    fake_observed.run_id = "r1"
    fake_observed.framework_extra = None
    fake_observed.to_dict.return_value = {}

    with patch("groundeval.observe.observe_agent", return_value=fake_observed):
        args = argparse.Namespace(
            framework="custom",
            agent_class="pkg.Agent",
            no_draft=True,
            draft_mode="standard",
            output=str(tmp_path),
            max_steps=5,
            score=True,
            config=None,
            allow_draft_config=False,
        )
        with pytest.raises(SystemExit, match="requires --config"):
            cmd_observe(args)


def test_cmd_observe_score_rejects_unreviewed_draft_config(tmp_path):
    fake_observed = MagicMock()
    fake_observed.tool_calls = []
    fake_observed.run_id = "r1"
    fake_observed.framework_extra = None
    fake_observed.to_dict.return_value = {}

    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "groundeval": {"generated_from_observation": True, "reviewed": False},
            "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
        },
    )

    with patch("groundeval.observe.observe_agent", return_value=fake_observed):
        args = argparse.Namespace(
            framework="custom",
            agent_class="pkg.Agent",
            no_draft=True,
            draft_mode="standard",
            output=str(tmp_path),
            max_steps=5,
            score=True,
            config=str(cfg_path),
            allow_draft_config=False,
        )
        with pytest.raises(SystemExit, match="has not been marked reviewed"):
            cmd_observe(args)


def test_cmd_observe_score_writes_observed_scores(tmp_path):
    fake_observed = MagicMock()
    fake_observed.tool_calls = []
    fake_observed.run_id = "r1"
    fake_observed.framework_extra = None
    fake_observed.framework = "custom"
    fake_observed.agent_class = "pkg.Agent"
    fake_observed.total_latency_ms = 0.0
    fake_observed.to_dict.return_value = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "final_answer": {},
        "total_latency_ms": 0.0,
    }

    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "a.json").write_text("{}")

    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "artifacts_dir": str(art),
            "task_contracts": [{"name": "t1", "preconditions": [{"check": "pc1"}]}],
        },
    )

    payload = {
        "summary": {
            "n_tasks": 1,
            "counterfactual_score": 0.1,
            "silence_score": 0.2,
            "perspective_score": 0.3,
            "overall_score": 0.2,
            "accuracy": 0.0,
            "total_violations": 1,
        },
        "meta": {},
        "results": [],
        "trajectories": [],
    }

    with patch("groundeval.observe.observe_agent", return_value=fake_observed):
        with patch("groundeval.observe.score_observed_run", return_value=([], payload)):
            args = argparse.Namespace(
                framework="custom",
                agent_class="pkg.Agent",
                no_draft=True,
                draft_mode="standard",
                output=str(tmp_path),
                max_steps=5,
                score=True,
                config=str(cfg_path),
                allow_draft_config=False,
            )
            cmd_observe(args)

    observed_runs = list(tmp_path.glob("observed_run_*.json"))
    reports = list(tmp_path.glob("observe_report_*.md"))
    diagrams = list(tmp_path.glob("observe_diagram_*.pdf"))
    score_files = list(tmp_path.glob("observed_scores_*.json"))

    assert len(observed_runs) == 1
    assert len(reports) == 1
    assert len(diagrams) == 1
    assert len(score_files) == 1


def test_cmd_draft_requires_existing_file(tmp_path):
    args = argparse.Namespace(
        from_run=str(tmp_path / "missing.json"),
        draft_mode="standard",
        output=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="Observed run file not found"):
        cmd_draft(args)


def test_cmd_draft_reads_run_and_writes_outputs(tmp_path):
    run_json = tmp_path / "observed_run.json"
    run_json.write_text(
        json.dumps({
            "run_id": "r1",
            "framework": "custom",
            "agent_class": "pkg.Agent",
            "tool_calls": [],
            "final_answer": {},
            "total_latency_ms": 0.0,
        })
    )

    args = argparse.Namespace(
        from_run=str(run_json),
        draft_mode="conservative",
        output=str(tmp_path),
    )

    with patch("groundeval.observe.DraftGenerator") as gen_cls:
        gen = MagicMock()
        gen_cls.return_value = gen
        with patch("groundeval.observe.write_draft_output") as writer:
            cmd_draft(args)

    gen_cls.assert_called_once()
    writer.assert_called_once()


def test_main_task_dispatch():
    with patch("sys.argv", ["groundeval", "task", "--config", "config.yaml"]):
        with patch("groundeval.run.cmd_task") as cmd:
            main()
            cmd.assert_called_once()


def test_main_observe_dispatch():
    with patch(
        "sys.argv",
        [
            "groundeval",
            "observe",
            "--framework",
            "custom",
            "--agent-class",
            "pkg.Agent",
        ],
    ):
        with patch("groundeval.run.cmd_observe") as cmd:
            main()
            cmd.assert_called_once()


def test_main_draft_dispatch(tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps({
            "run_id": "r1",
            "framework": "custom",
            "agent_class": "pkg.Agent",
            "tool_calls": [],
            "final_answer": {},
            "total_latency_ms": 0.0,
        })
    )
    with patch("sys.argv", ["groundeval", "draft", "--from-run", str(run_json)]):
        with patch("groundeval.run.cmd_draft") as cmd:
            main()
            cmd.assert_called_once()


def test_main_validate_marks_reviewed(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "task_contracts": [],
            "groundeval": {
                "config_status": "draft",
                "generated_from_observation": True,
                "reviewed": False,
            },
        },
    )

    with patch(
        "sys.argv",
        ["groundeval", "validate", "--config", str(cfg_path), "--mark-reviewed"],
    ):
        with patch("groundeval.run._validate_config"):
            main()

    updated = yaml.safe_load(cfg_path.read_text())
    assert updated["groundeval"]["config_status"] == "reviewed"
    assert updated["groundeval"]["reviewed"] is True


def test_main_validate_requires_mapping(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("- bad\n")
    with patch("sys.argv", ["groundeval", "validate", "--config", str(cfg_path)]):
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            main()
