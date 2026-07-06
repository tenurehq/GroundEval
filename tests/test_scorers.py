from groundeval.adapters import InMemoryCorpusAdapter, YamlAccessPolicy
from groundeval.core import AgentTrajectory, TaskContract, ToolCall
from groundeval.scorers import (
    _call_matches_args,
    _combine,
    _find_matching_calls,
    _get_nested,
    _merge_precondition_details,
    _resolve_from_corpus,
    _return_contains_expected,
    _values_match,
    aggregate_task_results,
    score_counterfactual_answer,
    score_counterfactual_trajectory,
    score_framework_observed_run,
    score_perspective_answer,
    score_perspective_trajectory,
    score_silence_answer,
    score_silence_trajectory,
    score_task_run,
)


def _tool_call(
    name="fetch_customer",
    args=None,
    result_ids=None,
    returned_empty=False,
    horizon=False,
    actor_gate=False,
    subsystem=False,
    timestamp_applied=None,
    latency_ms=1.0,
    agent_name=None,
    node_name=None,
    workflow_run_id=None,
    branch_id=None,
    call_id=None,
    parent_event_id=None,
    observed_return_value=None,
):
    call = ToolCall(
        tool_name=name,
        arguments=args if args is not None else {},
        result_ids=result_ids if result_ids is not None else [],
        timestamp_applied=timestamp_applied,
        horizon_violation=horizon,
        actor_gate_violation=actor_gate,
        subsystem_violation=subsystem,
        returned_empty=returned_empty,
        latency_ms=latency_ms,
        agent_name=agent_name,
        node_name=node_name,
        workflow_run_id=workflow_run_id,
        branch_id=branch_id,
        call_id=call_id,
        parent_event_id=parent_event_id,
    )
    if observed_return_value is not None:
        call.observed_return_value = observed_return_value
    return call


def test_values_match_cases():
    assert _values_match("active", "active") is True
    assert _values_match("Active", "active") is True
    assert _values_match(" 42 ", 42) is True
    assert _values_match(None, None) is True
    assert _values_match("x", "y") is False


def test_get_nested_and_resolve_from_corpus():
    assert _get_nested({"a": {"b": [{"c": 1}]}}, "a.b.0.c") == 1
    corpus = InMemoryCorpusAdapter([{"id": "a1", "nested": {"x": 2}}])
    assert _resolve_from_corpus("a1.nested.x", corpus) == 2
    assert _resolve_from_corpus("a1.missing", corpus) is None
    assert _resolve_from_corpus("badpath", corpus) is None


def test_call_matches_args_and_return_contains_expected():
    assert _call_matches_args({"artifact_id": "A1"}, {"artifact_id": "a1"}) is True
    assert _call_matches_args({"x": 1}, {"y": 1}) is False
    assert _return_contains_expected({"a": {"b": 2}}, {"a.b": 2}) is True
    assert _return_contains_expected({"a": 1}, {"a": 2}) is False
    assert _return_contains_expected([], {"a": 1}) is False
    assert _return_contains_expected({"a": "Hello"}, {"a": "hello"}) is True


def test_find_matching_calls():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            _tool_call("fetch_customer", {"artifact_id": "a1"}),
            _tool_call("fetch_customer", {"artifact_id": "a2"}),
            _tool_call("search_docs", {"query": "x"}),
        ],
    )
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
        "tool_expectations": [
            {"tool": "fetch_customer", "match_args": {"artifact_id": "a2"}}
        ],
    })
    matches = _find_matching_calls(traj, contract.tool_expectations[0])
    assert len(matches) == 1
    assert matches[0].arguments["artifact_id"] == "a2"


def test_combine_weights():
    assert _combine("COUNTERFACTUAL", 1.0, 0.0) == 0.5
    assert _combine("SILENCE", 1.0, 0.0) == 0.3
    assert _combine("PERSPECTIVE", 1.0, 0.0) == 0.4
    assert _combine("UNKNOWN", 1.0, 0.0) == 0.5


def test_score_counterfactual_answer_happy_path():
    corpus = InMemoryCorpusAdapter([{"id": "a1", "status": "active"}])
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [
            {
                "check": "pc1",
                "ground_truth_field": "a1.status",
                "required_facts": ["status"],
            }
        ],
    })
    answer = {
        "preconditions_verified": [
            {
                "check": "pc1",
                "passed": True,
                "facts_found": {"status": "active"},
                "evidence_artifacts": ["a1"],
            }
        ]
    }
    score, correct, details = score_counterfactual_answer(answer, contract, corpus)
    assert score == 1.0
    assert correct is True
    assert details[0]["evidence_supported"] is True


def test_score_counterfactual_answer_missing_precondition_and_no_evidence():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [
            {"check": "pc1", "ground_truth_field": "a1.status"},
            {"check": "pc2", "ground_truth_field": "a2.status"},
        ],
    })
    answer = {
        "preconditions_verified": [
            {
                "check": "pc1",
                "passed": True,
                "facts_found": {"status": "active"},
                "evidence_artifacts": [],
            }
        ]
    }
    score, correct, details = score_counterfactual_answer(answer, contract)
    assert score == 0.0
    assert correct is False
    assert len(details) == 2
    assert details[0]["error"] == "no evidence artifacts cited"
    assert details[1]["error"] == "precondition not checked by agent"


def test_score_counterfactual_answer_checks_uncovered_required_facts_against_evidence():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "status": "active", "plan": "gold"},
        {"id": "a2", "status": "inactive", "plan": "silver"},
    ])
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [
            {
                "check": "pc1",
                "ground_truth_field": "a1.status",
                "required_facts": ["status", "plan"],
            }
        ],
    })
    answer = {
        "preconditions_verified": [
            {
                "check": "pc1",
                "passed": True,
                "facts_found": {"status": "active", "plan": "gold"},
                "evidence_artifacts": ["a1"],
            }
        ]
    }
    score, correct, details = score_counterfactual_answer(answer, contract, corpus)
    assert score == 1.0
    assert details[0]["evidence_supported"] is True


def test_score_counterfactual_answer_uncovered_fact_fails_if_not_in_evidence():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "status": "active", "plan": "gold"},
        {"id": "a2", "status": "inactive", "plan": "silver"},
    ])
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [
            {
                "check": "pc1",
                "ground_truth_field": "a1.status",
                "required_facts": ["status", "plan"],
            }
        ],
    })
    answer = {
        "preconditions_verified": [
            {
                "check": "pc1",
                "passed": True,
                "facts_found": {"status": "active", "plan": "silver"},
                "evidence_artifacts": ["a1"],
            }
        ]
    }
    score, correct, details = score_counterfactual_answer(answer, contract, corpus)
    assert score == 0.0
    assert correct is False
    assert details[0]["evidence_supported"] is False


def test_score_counterfactual_answer_failures():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1", "ground_truth_field": "a1.status"}],
    })
    score, correct, details = score_counterfactual_answer({}, contract)
    assert score == 0.0
    assert correct is False
    assert details == []


def test_score_counterfactual_trajectory_uses_citations_and_penalties():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            _tool_call(result_ids=["a1"]),
            _tool_call(result_ids=["a2"], returned_empty=True, horizon=True),
        ],
        final_answer={
            "preconditions_verified": [
                {"check": "pc1", "evidence_artifacts": ["a1", "a3"]}
            ]
        },
        dead_ends_recovered=0,
    )
    score = score_counterfactual_trajectory(
        traj, TaskContract.from_dict({"name": "t1", "preconditions": []})
    )
    assert 0.0 <= score <= 1.0


def test_score_counterfactual_trajectory_top_level_evidence_artifacts_count():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[_tool_call(result_ids=["a1", "a2"])],
        final_answer={"evidence_artifacts": ["a1", "a2"]},
        dead_ends_recovered=0,
    )
    score = score_counterfactual_trajectory(
        traj, TaskContract.from_dict({"name": "t1", "preconditions": []})
    )
    assert score > 0.8


def test_score_silence_answer_basic():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}, {"check": "pc2"}],
    })
    answer = {
        "preconditions_verified": [
            {"check": "pc1", "facts_found": {"status": "active"}},
            {"check": "pc2", "facts_found": {}},
        ]
    }
    score, correct, details = score_silence_answer(answer, contract)
    assert score == 0.5
    assert correct is False
    assert len(details) == 2


def test_score_silence_answer_zero_preconditions_is_perfect():
    contract = TaskContract.from_dict({"name": "t1", "preconditions": []})
    score, correct, details = score_silence_answer(
        {"preconditions_verified": []}, contract
    )
    assert score == 1.0
    assert correct is True
    assert details == []


def test_score_silence_trajectory_diversity_and_dead_end_effects():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
        "role": "sales",
        "roles": {"sales": {"subsystems": ["crm", "email", "jira"]}},
    })
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            _tool_call(args={"artifact_type": "crm"}, result_ids=["a1"]),
            _tool_call(
                args={"artifact_type": "email"}, result_ids=["a2"], returned_empty=True
            ),
        ],
        dead_ends_recovered=1,
    )
    score = score_silence_trajectory(traj, contract, corpus)
    assert 0.0 <= score <= 1.0


def test_score_silence_trajectory_without_available_subsystems_uses_zero_diversity():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[_tool_call(args={}, result_ids=["a1"])],
        dead_ends_recovered=0,
    )
    score = score_silence_trajectory(
        traj, TaskContract.from_dict({"name": "t1", "preconditions": []})
    )
    assert 0.0 <= score <= 1.0


def test_score_perspective_answer_with_policy_and_corpus():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    policy = YamlAccessPolicy({
        "actors": {"alice": "sales"},
        "roles": {"sales": {"subsystems": ["crm"]}},
    })
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
        "actor": "alice",
        "role": "sales",
    })
    answer = {
        "preconditions_verified": [
            {"check": "pc1", "evidence_artifacts": ["a1", "a2"]}
        ],
        "reasoning": "done",
    }
    score, correct, details = score_perspective_answer(answer, contract, policy, corpus)
    assert score == 0.5
    assert correct is False
    assert details == []


def test_score_perspective_answer_no_policy_context_defaults_to_full_score():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    answer = {
        "preconditions_verified": [{"check": "pc1"}],
        "reasoning": "done",
    }
    score, correct, details = score_perspective_answer(answer, contract)
    assert score == 1.0
    assert correct is True


def test_score_perspective_answer_missing_reasoning_or_preconditions_is_zero():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    score1, _, _ = score_perspective_answer(
        {"preconditions_verified": [{"check": "pc1"}]}, contract
    )
    score2, _, _ = score_perspective_answer({"reasoning": "x"}, contract)
    assert score1 == 0.0
    assert score2 == 0.0


def test_score_perspective_trajectory_penalties():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            _tool_call(actor_gate=True),
            _tool_call(subsystem=True),
            _tool_call(horizon=True, returned_empty=True),
        ],
        dead_ends_recovered=0,
    )
    score = score_perspective_trajectory(
        traj, TaskContract.from_dict({"name": "t1", "preconditions": []})
    )
    assert 0.0 <= score <= 1.0


def test_score_task_run_returns_populated_result():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "status": "active", "subsystem": "crm"}
    ])
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [
            {
                "check": "pc1",
                "ground_truth_field": "a1.status",
                "required_facts": ["status"],
            }
        ],
    })
    answer = {
        "preconditions_verified": [
            {
                "check": "pc1",
                "passed": True,
                "facts_found": {"status": "active"},
                "evidence_artifacts": ["a1"],
            }
        ],
        "reasoning": "ok",
    }
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[_tool_call(result_ids=["a1"])],
        final_answer=answer,
    )
    result = score_task_run(traj, answer, contract, corpus=corpus)
    assert result.task_name == "t1"
    assert 0.0 <= result.overall_score <= 1.0
    assert "cf_answer" in result.meta
    assert "sl_answer" in result.meta
    assert "ps_answer" in result.meta


def test_score_framework_observed_run_expected_return_partial_match_scores_half_trajectory_credit():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [],
        "tool_expectations": [
            {
                "tool": "fetch_customer",
                "match_args": {"artifact_id": "a1"},
                "expected_return": {"status": "active"},
            }
        ],
    })
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            _tool_call(
                "fetch_customer",
                {"artifact_id": "a1"},
                ["a1"],
                observed_return_value={"status": "inactive"},
            )
        ],
        final_answer={"reasoning": "x"},
    )
    result = score_framework_observed_run(
        trajectory=traj,
        final_answer=traj.final_answer,
        contract=contract,
    )
    assert result.meta["cf_trajectory"] == 0.5


def test_merge_precondition_details_merges_by_check():
    merged = _merge_precondition_details(
        [{"check": "pc1", "a": 1}],
        [{"check": "pc1", "b": 2}, {"check": "pc2", "c": 3}],
    )
    by_check = {x["check"]: x for x in merged}
    assert by_check["pc1"]["a"] == 1
    assert by_check["pc1"]["b"] == 2
    assert by_check["pc2"]["c"] == 3


def test_aggregate_task_results_basic():
    r1 = score_task_run(
        AgentTrajectory(task_id="t1"),
        {},
        TaskContract.from_dict({"name": "t1", "preconditions": [{"check": "pc1"}]}),
    )
    r2 = score_task_run(
        AgentTrajectory(task_id="t2"),
        {},
        TaskContract.from_dict({"name": "t2", "preconditions": [{"check": "pc1"}]}),
    )
    agg = aggregate_task_results([r1, r2])
    assert agg["n_tasks"] == 2
    assert "overall_score" in agg
    assert "per_task" in agg
