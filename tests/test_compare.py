import json

import pytest

from groundeval.compare import compare_json_files


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_text(path, text):
    path.write_text(text, encoding="utf-8")


def test_compare_observed_scores_no_meaningful_differences(tmp_path):
    payload = {
        "summary": {
            "counterfactual_score": 0.5,
            "silence_score": 0.6,
            "perspective_score": 0.7,
            "overall_score": 0.8,
            "accuracy": 0.9,
            "total_violations": 1,
        },
        "results": [
            {
                "task_name": "task-a",
                "overall_score": 0.8,
                "horizon_violations": 0,
                "actor_gate_violations": 0,
                "subsystem_violations": 0,
                "dead_ends_hit": 0,
                "precondition_results": [],
                "meta": {},
            }
        ],
        "trajectories": [
            {
                "task_id": "task-a",
                "tool_calls": [{"tool_name": "fetch_customer"}],
            }
        ],
    }
    old_file = tmp_path / "old_scores.json"
    new_file = tmp_path / "new_scores.json"
    _write_json(old_file, payload)
    _write_json(new_file, payload)

    report = compare_json_files(old_file, new_file)

    assert "GroundEval Compare" in report
    assert "No meaningful differences found." in report


def test_compare_observed_scores_detects_summary_per_task_violations_and_trajectory_changes(
    tmp_path,
):
    old_payload = {
        "summary": {
            "counterfactual_score": 0.5,
            "silence_score": 0.6,
            "perspective_score": 0.7,
            "overall_score": 0.8,
            "accuracy": 0.9,
            "total_violations": 1,
        },
        "results": [
            {
                "task_name": "task-a",
                "overall_score": 0.8,
                "horizon_violations": 0,
                "actor_gate_violations": 0,
                "subsystem_violations": 0,
                "dead_ends_hit": 0,
                "precondition_results": [
                    {
                        "check": "customer_ok",
                        "verified": True,
                        "evidence_supported": True,
                        "reasons": [],
                    }
                ],
                "meta": {
                    "multi_agent": {
                        "required_agents": [],
                        "required_handoffs": [],
                        "required_agent_tool_expectations": [],
                    }
                },
            }
        ],
        "trajectories": [
            {
                "task_id": "task-a",
                "tool_calls": [
                    {"tool_name": "fetch_customer", "agent_name": "alice"},
                    {"tool_name": "search_tickets", "agent_name": "alice"},
                ],
            }
        ],
    }
    new_payload = {
        "summary": {
            "counterfactual_score": 0.4,
            "silence_score": 0.6,
            "perspective_score": 0.9,
            "overall_score": 0.7,
            "accuracy": 0.8,
            "total_violations": 3,
        },
        "results": [
            {
                "task_name": "task-a",
                "overall_score": 0.3,
                "horizon_violations": 2,
                "actor_gate_violations": 1,
                "subsystem_violations": 0,
                "dead_ends_hit": 1,
                "precondition_results": [
                    {
                        "check": "customer_ok",
                        "verified": False,
                        "evidence_supported": False,
                        "error": "missing evidence",
                        "reasons": ["status unavailable"],
                    }
                ],
                "meta": {
                    "multi_agent": {
                        "required_agents": [
                            {
                                "requirement": {"agent_name": "planner"},
                                "observed": False,
                            }
                        ],
                        "required_handoffs": [
                            {
                                "requirement": {
                                    "from_agent": "planner",
                                    "to_agent": "reviewer",
                                },
                                "observed": False,
                            }
                        ],
                        "required_agent_tool_expectations": [
                            {
                                "requirement": {
                                    "agent_name": "reviewer",
                                    "tool": "fetch_case",
                                },
                                "observed": False,
                            }
                        ],
                    }
                },
            }
        ],
        "trajectories": [
            {
                "task_id": "task-a",
                "tool_calls": [
                    {
                        "tool_name": "fetch_customer",
                        "agent_name": "alice",
                        "returned_empty": True,
                    },
                    {
                        "tool_name": "fetch_case",
                        "agent_name": "bob",
                        "horizon_violation": True,
                        "actor_gate_violation": True,
                        "subsystem_violation": True,
                    },
                ],
            }
        ],
    }
    old_file = tmp_path / "old_scores.json"
    new_file = tmp_path / "new_scores.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Scores changed:" in report
    assert "- counterfactual_score: 0.50 → 0.40" in report
    assert "- perspective_score: 0.70 → 0.90" in report
    assert "- overall_score: 0.80 → 0.70" in report
    assert "- accuracy: 0.90 → 0.80" in report
    assert "- total_violations: 1 → 3" in report
    assert "Per-task changes:" in report
    assert "- task-a: overall_score 0.80 → 0.30" in report
    assert "New violations:" in report
    assert "task-a: horizon_violations=2" in report
    assert "task-a: actor_gate_violations=1" in report
    assert "task-a: dead_ends_hit=1" in report
    assert "task-a: precondition 'customer_ok' error: missing evidence" in report
    assert "task-a: precondition 'customer_ok' evidence unsupported" in report
    assert "task-a: precondition 'customer_ok' not verified" in report
    assert "task-a: precondition 'customer_ok' reason: status unavailable" in report
    assert "task-a: required agent not observed: planner" in report
    assert "task-a: required handoff not observed: planner → reviewer" in report
    assert "task-a: required agent tool not observed: reviewer -> fetch_case" in report
    assert "task-a: alice fetch_customer call 1 returned empty" in report
    assert "task-a: bob fetch_case call 2 horizon violation" in report
    assert "task-a: bob fetch_case call 2 actor gate violation" in report
    assert "task-a: bob fetch_case call 2 subsystem violation" in report
    assert "Trajectory diff:" in report
    assert "- old: fetch_customer → search_tickets" in report
    assert "- new: fetch_customer → fetch_case" in report


def test_compare_observed_scores_detects_fixed_violations(tmp_path):
    old_payload = {
        "summary": {},
        "results": [
            {
                "task_name": "task-a",
                "overall_score": 0.2,
                "horizon_violations": 1,
                "actor_gate_violations": 0,
                "subsystem_violations": 0,
                "dead_ends_hit": 0,
                "precondition_results": [
                    {
                        "check": "pc1",
                        "verified": False,
                        "evidence_supported": False,
                        "reasons": ["bad source"],
                    }
                ],
                "meta": {},
            }
        ],
        "trajectories": [{"task_id": "task-a", "tool_calls": []}],
    }
    new_payload = {
        "summary": {},
        "results": [
            {
                "task_name": "task-a",
                "overall_score": 0.9,
                "horizon_violations": 0,
                "actor_gate_violations": 0,
                "subsystem_violations": 0,
                "dead_ends_hit": 0,
                "precondition_results": [
                    {
                        "check": "pc1",
                        "verified": True,
                        "evidence_supported": True,
                        "reasons": [],
                    }
                ],
                "meta": {},
            }
        ],
        "trajectories": [{"task_id": "task-a", "tool_calls": []}],
    }
    old_file = tmp_path / "old_scores.json"
    new_file = tmp_path / "new_scores.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Fixed violations:" in report
    assert "task-a: horizon_violations=1" in report
    assert "task-a: precondition 'pc1' evidence unsupported" in report
    assert "task-a: precondition 'pc1' not verified" in report
    assert "task-a: precondition 'pc1' reason: bad source" in report


def test_compare_observed_scores_per_task_added_and_missing(tmp_path):
    old_payload = {
        "summary": {},
        "results": [
            {"task_name": "task-a", "overall_score": 0.4},
            {"task_name": "task-b", "overall_score": 0.5},
        ],
        "trajectories": [],
    }
    new_payload = {
        "summary": {},
        "results": [
            {"task_name": "task-a", "overall_score": 0.4},
            {"task_name": "task-c", "overall_score": 0.9},
        ],
        "trajectories": [],
    }
    old_file = tmp_path / "old_scores.json"
    new_file = tmp_path / "new_scores.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Per-task changes:" in report
    assert "- task-b: missing in new file" in report
    assert "- task-c: added in new file" in report


def test_compare_observed_scores_multiple_task_trajectory_diff_format(tmp_path):
    old_payload = {
        "summary": {},
        "results": [
            {"task_name": "task-a", "overall_score": 0.5},
            {"task_name": "task-b", "overall_score": 0.5},
        ],
        "trajectories": [
            {"task_id": "task-a", "tool_calls": [{"tool_name": "fetch_customer"}]},
            {"task_id": "task-b", "tool_calls": [{"tool_name": "search_docs"}]},
        ],
    }
    new_payload = {
        "summary": {},
        "results": [
            {"task_name": "task-a", "overall_score": 0.5},
            {"task_name": "task-b", "overall_score": 0.5},
        ],
        "trajectories": [
            {"task_id": "task-a", "tool_calls": [{"tool_name": "fetch_case"}]},
            {"task_id": "task-b", "tool_calls": [{"tool_name": "search_docs"}]},
        ],
    }
    old_file = tmp_path / "old_scores.json"
    new_file = tmp_path / "new_scores.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Trajectory diff:" in report
    assert "- task: task-a" in report
    assert "  old: fetch_customer" in report
    assert "  new: fetch_case" in report


def test_compare_observed_run_no_meaningful_differences(tmp_path):
    payload = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [
            {"tool_name": "fetch_customer"},
            {"tool_name": "search_docs"},
        ],
        "final_answer": {"should_act": True},
    }
    old_file = tmp_path / "old_run.json"
    new_file = tmp_path / "new_run.json"
    _write_json(old_file, payload)
    _write_json(new_file, payload)

    report = compare_json_files(old_file, new_file)

    assert "No meaningful differences found." in report


def test_compare_observed_run_detects_count_sequence_and_final_answer_shape_changes(
    tmp_path,
):
    old_payload = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [
            {"tool_name": "fetch_customer"},
            {"tool_name": "search_docs"},
        ],
        "final_answer": {
            "should_act": True,
            "preconditions_verified": [{"check": "pc1"}],
        },
    }
    new_payload = {
        "run_id": "r2",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [
            {"tool_name": "fetch_customer"},
            {"tool_name": "fetch_case"},
            {"tool_name": "search_docs"},
        ],
        "final_answer": {
            "should_act": True,
            "reasoning": "updated",
            "preconditions_verified": [{"check": "pc1"}, {"check": "pc2"}],
        },
    }
    old_file = tmp_path / "old_run.json"
    new_file = tmp_path / "new_run.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Tool call count changed:" in report
    assert "- old: 2" in report
    assert "- new: 3" in report
    assert "Trajectory diff:" in report
    assert "- old: fetch_customer → search_docs" in report
    assert "- new: fetch_customer → fetch_case → search_docs" in report
    assert "Final answer diff:" in report
    assert "- added keys: reasoning" in report
    assert "- preconditions_verified count: 1 → 2" in report


def test_compare_observed_run_detects_removed_final_answer_keys(tmp_path):
    old_payload = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "final_answer": {
            "should_act": True,
            "reasoning": "because",
            "extra": "x",
        },
    }
    new_payload = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "final_answer": {
            "should_act": True,
        },
    }
    old_file = tmp_path / "old_run.json"
    new_file = tmp_path / "new_run.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Final answer diff:" in report
    assert "- removed keys: extra, reasoning" in report


def test_compare_observed_run_detects_final_answer_content_change_when_not_dicts(
    tmp_path,
):
    old_payload = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "final_answer": "plain text answer",
    }
    new_payload = {
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "final_answer": ["plain", "text", "answer"],
    }
    old_file = tmp_path / "old_run.json"
    new_file = tmp_path / "new_run.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Final answer diff:" in report
    assert "- final_answer content changed" in report


def test_compare_observed_run_handles_missing_optional_fields_as_empty(tmp_path):
    old_payload = {
        "run_id": "r1",
        "framework": "custom",
        "tool_calls": None,
        "final_answer": None,
    }
    new_payload = {
        "run_id": "r2",
        "framework": "custom",
        "tool_calls": [],
    }
    old_file = tmp_path / "old_run.json"
    new_file = tmp_path / "new_run.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Final answer diff:" in report
    assert "- final_answer content changed" in report


def test_compare_task_results_no_meaningful_differences(tmp_path):
    payload = {
        "meta": {"model": "x"},
        "summary": {
            "counterfactual_score": 0.5,
            "silence_score": 0.6,
            "perspective_score": 0.7,
            "overall_score": 0.8,
            "accuracy": 0.9,
            "total_violations": 1,
            "per_task": [
                {"task_name": "task-a", "overall_score": 0.8},
            ],
        },
    }
    old_file = tmp_path / "old_task_results.json"
    new_file = tmp_path / "new_task_results.json"
    _write_json(old_file, payload)
    _write_json(new_file, payload)

    report = compare_json_files(old_file, new_file)

    assert "No meaningful differences found." in report


def test_compare_task_results_detects_summary_and_per_task_changes(tmp_path):
    old_payload = {
        "meta": {"model": "x"},
        "summary": {
            "counterfactual_score": 0.5,
            "silence_score": 0.6,
            "perspective_score": 0.7,
            "overall_score": 0.8,
            "accuracy": 0.9,
            "total_violations": 1,
            "per_task": [
                {"task_name": "task-a", "overall_score": 0.8},
                {"task_name": "task-b", "overall_score": 0.5},
            ],
        },
    }
    new_payload = {
        "meta": {"model": "y"},
        "summary": {
            "counterfactual_score": 0.4,
            "silence_score": 0.6,
            "perspective_score": 0.9,
            "overall_score": 0.7,
            "accuracy": 0.8,
            "total_violations": 2,
            "per_task": [
                {"task_name": "task-a", "overall_score": 0.3},
                {"task_name": "task-c", "overall_score": 1.0},
            ],
        },
    }
    old_file = tmp_path / "old_task_results.json"
    new_file = tmp_path / "new_task_results.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Scores changed:" in report
    assert "- counterfactual_score: 0.50 → 0.40" in report
    assert "- perspective_score: 0.70 → 0.90" in report
    assert "- overall_score: 0.80 → 0.70" in report
    assert "- accuracy: 0.90 → 0.80" in report
    assert "- total_violations: 1 → 2" in report
    assert "Per-task changes:" in report
    assert "- task-a: overall_score 0.80 → 0.30" in report
    assert "- task-b: missing in new file" in report
    assert "- task-c: added in new file" in report


def test_compare_mismatched_payload_types_aborts(tmp_path):
    old_payload = {
        "summary": {},
        "results": [],
        "trajectories": [],
    }
    new_payload = {
        "run_id": "r1",
        "framework": "custom",
        "tool_calls": [],
    }
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "File types differ:" in report
    assert "- old: observed_scores" in report
    assert "- new: observed_run" in report
    assert (
        "Comparison aborted because the two JSON files are not the same payload type."
        in report
    )


def test_compare_unsupported_json_shape(tmp_path):
    old_payload = {"hello": "world"}
    new_payload = {"hello": "again"}
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "Unsupported JSON shape." in report


def test_compare_missing_old_file_raises(tmp_path):
    new_file = tmp_path / "new.json"
    _write_json(new_file, {"meta": {}, "summary": {}})

    with pytest.raises(FileNotFoundError, match="Compare file not found"):
        compare_json_files(tmp_path / "missing.json", new_file)


def test_compare_missing_new_file_raises(tmp_path):
    old_file = tmp_path / "old.json"
    _write_json(old_file, {"meta": {}, "summary": {}})

    with pytest.raises(FileNotFoundError, match="Compare file not found"):
        compare_json_files(old_file, tmp_path / "missing.json")


def test_compare_malformed_old_json_raises_decode_error(tmp_path):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    _write_text(old_file, '{"broken": ')
    _write_json(new_file, {"meta": {}, "summary": {}})

    with pytest.raises(json.JSONDecodeError):
        compare_json_files(old_file, new_file)


def test_compare_malformed_new_json_raises_decode_error(tmp_path):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    _write_json(old_file, {"meta": {}, "summary": {}})
    _write_text(new_file, '{"broken": ')

    with pytest.raises(json.JSONDecodeError):
        compare_json_files(old_file, new_file)


def test_compare_non_dict_json_values_are_unknown_shape(tmp_path):
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    _write_json(old_file, ["not", "a", "dict"])
    _write_json(new_file, ["still", "not", "a", "dict"])

    report = compare_json_files(old_file, new_file)

    assert "Unsupported JSON shape." in report


def test_compare_observed_scores_incomplete_payload_raises(tmp_path):
    old_payload = {
        "summary": None,
        "results": None,
        "trajectories": None,
    }
    new_payload = {
        "summary": {},
        "results": [],
        "trajectories": [],
    }
    old_file = tmp_path / "old_scores.json"
    new_file = tmp_path / "new_scores.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    with pytest.raises(TypeError, match="NoneType"):
        compare_json_files(old_file, new_file)


def test_compare_task_results_incomplete_payload_still_compares_safely(tmp_path):
    old_payload = {
        "meta": {},
        "summary": None,
    }
    new_payload = {
        "meta": {"model": "x"},
        "summary": {},
    }
    old_file = tmp_path / "old_task_results.json"
    new_file = tmp_path / "new_task_results.json"
    _write_json(old_file, old_payload)
    _write_json(new_file, new_payload)

    report = compare_json_files(old_file, new_file)

    assert "No meaningful differences found." in report
