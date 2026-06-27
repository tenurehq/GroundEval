import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from groundeval.core import (
    ToolCall,
    AgentTrajectory,
    TaskPrecondition,
    TaskContract,
    TaskEvalResult,
    GatedRuntime,
    ANSWER_SCHEMA_TASK,
    _get_nested,
)


# ── ToolCall ────────────────────────────────────────────────


def test_tool_call_defaults():
    tc = ToolCall(
        tool_name="fetch_artifact",
        arguments={"artifact_id": "a1"},
        result_ids=["a1"],
        timestamp_applied=None,
        horizon_violation=False,
        actor_gate_violation=False,
        subsystem_violation=False,
        returned_empty=False,
        latency_ms=12.5,
    )
    assert tc.tool_name == "fetch_artifact"
    assert tc.result_ids == ["a1"]
    assert tc.latency_ms == 12.5
    assert not tc.horizon_violation


def test_tool_call_violation_flags():
    tc = ToolCall(
        tool_name="search_artifacts",
        arguments={"query": "x", "artifact_type": "audit_trail"},
        result_ids=[],
        timestamp_applied="2026-01-15",
        horizon_violation=True,
        actor_gate_violation=False,
        subsystem_violation=True,
        returned_empty=True,
        latency_ms=0.0,
    )
    assert tc.horizon_violation
    assert tc.subsystem_violation
    assert tc.returned_empty
    assert tc.result_ids == []


# ── AgentTrajectory ─────────────────────────────────────────


def test_agent_trajectory_defaults():
    traj = AgentTrajectory(task_id="task_1")
    assert traj.task_id == "task_1"
    assert traj.tool_calls == []
    assert traj.final_answer == {}
    assert traj.horizon_violations == 0
    assert traj.actor_gate_violations == 0
    assert traj.subsystem_violations == 0
    assert traj.dead_ends_hit == 0
    assert traj.dead_ends_recovered == 0
    assert not traj.budget_exceeded


def test_agent_trajectory_with_calls():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            ToolCall(
                tool_name="fetch",
                arguments={"artifact_id": "a"},
                result_ids=["a"],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=True,
                returned_empty=False,
                latency_ms=5.0,
            )
        ],
        final_answer={"all_preconditions_pass": True},
        prompt_tokens=100,
        completion_tokens=50,
        budget_exceeded=False,
    )
    assert len(traj.tool_calls) == 1
    assert traj.final_answer["all_preconditions_pass"] is True
    assert traj.prompt_tokens == 100
    assert traj.completion_tokens == 50


# ── TaskPrecondition ────────────────────────────────────────


def test_task_precondition_from_dict_full():
    d = {
        "check": "customer_active",
        "description": "Customer must be active in CRM.",
        "required_facts": ["account_status"],
        "ground_truth_field": "crm_account.account_status",
    }
    pc = TaskPrecondition.from_dict(d)
    assert pc.check == "customer_active"
    assert pc.description == "Customer must be active in CRM."
    assert pc.required_facts == ["account_status"]
    assert pc.ground_truth_field == "crm_account.account_status"


def test_task_precondition_from_dict_minimal():
    d = {"check": "just_a_check"}
    pc = TaskPrecondition.from_dict(d)
    assert pc.check == "just_a_check"
    assert pc.description == "just_a_check"  # falls back to check
    assert pc.required_facts == []
    assert pc.ground_truth_field == ""


def test_task_precondition_to_dict():
    pc = TaskPrecondition(
        check="domain_clean",
        description="Domain must not be spoofed.",
        required_facts=["reputation"],
        ground_truth_field="domain_lookup.reputation",
    )
    d = pc.to_dict()
    assert d["check"] == "domain_clean"
    assert d["ground_truth_field"] == "domain_lookup.reputation"
    assert "reputation" in d["required_facts"]


# ── TaskContract ────────────────────────────────────────────


def test_task_contract_from_dict():
    d = {
        "name": "verify_recipient",
        "task_description": "Verify recipient before sending email.",
        "preconditions": [
            {"check": "pc1", "description": "First", "ground_truth_field": "a.status"},
            {"check": "pc2", "description": "Second"},
        ],
        "valid_action": "all_preconditions_pass",
        "decision_field": "should_send_email",
        "actor": "agent",
        "role": "sales_rep",
    }
    tc = TaskContract.from_dict(d)
    assert tc.name == "verify_recipient"
    assert tc.task_description == "Verify recipient before sending email."
    assert len(tc.preconditions) == 2
    assert tc.preconditions[0].check == "pc1"
    assert tc.preconditions[0].ground_truth_field == "a.status"
    assert tc.decision_field == "should_send_email"
    assert tc.actor == "agent"
    assert tc.role == "sales_rep"


def test_task_contract_from_dict_defaults():
    d = {"name": "minimal", "preconditions": []}
    tc = TaskContract.from_dict(d)
    assert tc.name == "minimal"
    assert tc.task_description == "minimal"
    assert tc.valid_action == "all_preconditions_pass"
    assert tc.decision_field == "should_act"
    assert tc.artifacts_dir == "./data"
    assert tc.actor is None
    assert tc.role is None
    assert tc.actors == {}
    assert tc.roles == {}


def test_task_contract_to_dict():
    tc = TaskContract(
        name="t1",
        task_description="Do the thing.",
        preconditions=[TaskPrecondition(check="pc", description="d")],
        actors={"agent1": "role_a"},
        roles={"role_a": {"subsystems": ["crm"]}},
    )
    d = tc.to_dict()
    assert d["name"] == "t1"
    assert len(d["preconditions"]) == 1
    assert d["actors"] == {"agent1": "role_a"}


def test_task_contract_multi_actor():
    d = {
        "name": "multi",
        "task_description": "Multi-agent task.",
        "preconditions": [{"check": "pc", "description": "d"}],
        "actors": {"agent1": "role_a", "agent2": "role_b"},
        "roles": {
            "role_a": {"subsystems": ["crm"]},
            "role_b": {"subsystems": ["audit"]},
        },
    }
    tc = TaskContract.from_dict(d)
    assert len(tc.actors) == 2
    assert tc.actors["agent1"] == "role_a"
    assert tc.roles["role_a"]["subsystems"] == ["crm"]


# ── TaskEvalResult ──────────────────────────────────────────


def test_task_eval_result_scores():
    result = TaskEvalResult(
        task_name="test",
        counterfactual_score=0.85,
        silence_score=0.72,
        perspective_score=0.90,
        overall_score=0.823,
        answer_correct=True,
        horizon_violations=1,
        actor_gate_violations=0,
        subsystem_violations=2,
    )
    assert result.counterfactual_score == 0.85
    assert result.perspective_score == 0.90
    assert result.answer_correct


def test_task_eval_result_to_dict():
    result = TaskEvalResult(
        task_name="test",
        counterfactual_score=0.5,
        silence_score=0.6,
        perspective_score=0.7,
        overall_score=0.6,
        answer_correct=False,
        horizon_violations=2,
        actor_gate_violations=1,
        subsystem_violations=3,
    )
    d = result.to_dict()
    assert d["total_violations"] == 6
    assert d["counterfactual_score"] == 0.5
    assert d["answer_correct"] is False


def test_task_eval_result_default_meta():
    result = TaskEvalResult(
        task_name="t",
        counterfactual_score=1.0,
        silence_score=1.0,
        perspective_score=1.0,
        overall_score=1.0,
        answer_correct=True,
    )
    assert result.meta == {}
    assert result.tool_call_count == 0
    assert result.prompt_tokens == 0


def test_task_eval_result_all_zeros():
    """Result with all zero scores should still serialize correctly."""
    result = TaskEvalResult(
        task_name="zero",
        counterfactual_score=0.0,
        silence_score=0.0,
        perspective_score=0.0,
        overall_score=0.0,
        answer_correct=False,
    )
    d = result.to_dict()
    assert d["total_violations"] == 0
    assert d["overall_score"] == 0.0


def test_task_eval_result_max_violations():
    """Result with many violations should sum correctly."""
    result = TaskEvalResult(
        task_name="bad",
        counterfactual_score=0.1,
        silence_score=0.1,
        perspective_score=0.1,
        overall_score=0.1,
        answer_correct=False,
        horizon_violations=10,
        actor_gate_violations=20,
        subsystem_violations=30,
    )
    assert result.to_dict()["total_violations"] == 60


# ── ANSWER_SCHEMA_TASK ──────────────────────────────────────


def test_answer_schema_task_structure():
    schema = ANSWER_SCHEMA_TASK
    assert schema["type"] == "object"
    assert "preconditions_verified" in schema["required"]
    assert "all_preconditions_pass" in schema["required"]
    assert "reasoning" in schema["required"]
    assert "preconditions_verified" in schema["properties"]
    items = schema["properties"]["preconditions_verified"]["items"]
    assert "check" in items["required"]
    assert "passed" in items["required"]
    assert "facts_found" in items["required"]


def test_answer_schema_task_optional_fields():
    schema = ANSWER_SCHEMA_TASK
    props = schema["properties"]
    assert "should_act" in props
    assert "evidence_artifacts" in props


def test_answer_schema_task_can_validate_empty_preconditions():
    """Schema should accept an empty preconditions_verified list."""
    import jsonschema

    answer = {
        "preconditions_verified": [],
        "all_preconditions_pass": True,
        "reasoning": "Nothing to check.",
    }
    jsonschema.validate(answer, ANSWER_SCHEMA_TASK)  # should not raise


# ── _get_nested ─────────────────────────────────────────────


def test_get_nested_simple():
    d = {"a": {"b": "value"}}
    assert _get_nested(d, "a.b") == "value"


def test_get_nested_list_index():
    d = {"items": [{"name": "first"}, {"name": "second"}]}
    assert _get_nested(d, "items.0.name") == "first"
    assert _get_nested(d, "items.1.name") == "second"


def test_get_nested_missing_key():
    d = {"a": 1}
    assert _get_nested(d, "b") is None
    assert _get_nested(d, "a.b.c") is None


def test_get_nested_bad_index():
    d = {"items": [1, 2]}
    assert _get_nested(d, "items.5") is None
    assert _get_nested(d, "items.abc") is None


def test_get_nested_non_container():
    d = {"a": 42}
    assert _get_nested(d, "a.b") is None


def test_get_nested_empty_dict():
    assert _get_nested({}, "anything") is None


def test_get_nested_empty_path():
    d = {"a": 1}
    assert _get_nested(d, "") is None


def test_get_nested_dotted_key_looks_like_index():
    """A key that is a digit string but in a dict should not be treated as list index."""
    d = {"0": "zero", "1": "one"}
    assert _get_nested(d, "0") == "zero"


def test_get_nested_value_is_none():
    d = {"a": {"b": None}}
    assert _get_nested(d, "a.b") is None


def test_get_nested_value_is_zero():
    d = {"count": 0}
    assert _get_nested(d, "count") == 0


def test_get_nested_value_is_empty_string():
    d = {"a": {"b": ""}}
    assert _get_nested(d, "a.b") == ""


def test_get_nested_value_is_false():
    d = {"flag": False}
    assert _get_nested(d, "flag") is False


def test_get_nested_out_of_bounds_negative():
    d = {"items": [1, 2, 3]}
    assert _get_nested(d, "items.-1") is None


def test_gated_runtime_record_filters_timestamps_in_results():
    """Results with timestamps past as_of should be filtered by _record, not returned."""
    corpus = _make_corpus([
        {"id": "old", "subsystem": "crm", "timestamp": "2025-01-01T00:00:00"},
        {"id": "new", "subsystem": "crm", "timestamp": "2027-01-01T00:00:00"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus, policy=policy, task_id="test", as_of="2026-01-01T00:00:00"
    )
    results = runtime.search("")
    ids = {r.get("id", r.get("_id")) for r in results}
    assert "old" in ids
    assert "new" not in ids


def _make_corpus(artifacts: list[dict]):
    """Helper: build an in-memory corpus for testing."""
    from groundeval.adapters import InMemoryCorpusAdapter

    return InMemoryCorpusAdapter(artifacts)


def _make_policy(actors: dict, roles: dict):
    """Helper: build a YamlAccessPolicy."""
    from groundeval.adapters import YamlAccessPolicy

    return YamlAccessPolicy({"actors": actors, "roles": roles})


def test_gated_runtime_fetch_found():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "account_status": "active"}
    ])
    policy = _make_policy(
        {"alice": "engineer"},
        {"engineer": {"subsystems": ["crm"]}},
    )
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor="alice",
        actor_visible_artifacts={"a1"},
        actor_subsystem_access={"crm"},
    )
    doc = runtime.fetch("a1")
    assert doc is not None
    assert doc["account_status"] == "active"

    traj = runtime.trajectory()
    assert len(traj.tool_calls) == 1
    assert traj.tool_calls[0].tool_name == "fetch_artifact"


def test_gated_runtime_fetch_not_found():
    corpus = _make_corpus([])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    doc = runtime.fetch("missing")
    assert doc is None

    traj = runtime.trajectory()
    assert traj.tool_calls[0].returned_empty


def test_gated_runtime_fetch_subsystem_violation():
    corpus = _make_corpus([{"id": "a1", "subsystem": "audit_trail", "data": "secret"}])
    policy = _make_policy(
        {"alice": "engineer"},
        {"engineer": {"subsystems": ["crm", "email"]}},
    )
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor="alice",
        actor_subsystem_access={"crm", "email"},
    )
    doc = runtime.fetch("a1")
    assert doc is None

    traj = runtime.trajectory()
    assert traj.subsystem_violations >= 1


def test_gated_runtime_fetch_actor_gate_violation():
    """Actor gate only applies when actor_visible_artifacts is non-empty."""
    corpus = _make_corpus([{"id": "a1", "subsystem": "crm", "data": "visible_to_bob"}])
    policy = _make_policy(
        {"alice": "engineer"},
        {"engineer": {"subsystems": ["crm"]}},
    )
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor="alice",
        actor_visible_artifacts={"some_other_artifact"},  # non-empty, excludes a1
        actor_subsystem_access={"crm"},
    )
    doc = runtime.fetch("a1")
    assert doc is None
    traj = runtime.trajectory()
    assert traj.actor_gate_violations >= 1


def test_gated_runtime_fetch_horizon_violation():
    corpus = _make_corpus([
        {
            "id": "a1",
            "subsystem": "crm",
            "timestamp": "2026-06-15T00:00:00",
            "data": "future",
        }
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        as_of="2026-01-01T00:00:00",
    )
    doc = runtime.fetch("a1")
    assert doc is None

    traj = runtime.trajectory()
    assert traj.horizon_violations >= 1 or traj.tool_calls[0].horizon_violation


def test_gated_runtime_search():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "name": "Acme Corp"},
        {"id": "a2", "subsystem": "crm", "name": "Acme Logistics"},
        {"id": "a3", "subsystem": "email", "name": "Acme Mail"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("Acme", limit=5)
    assert len(results) == 3

    traj = runtime.trajectory()
    assert traj.tool_calls[0].tool_name == "search_artifacts"


def test_gated_runtime_search_type_filter():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "name": "Common"},
        {"id": "a2", "subsystem": "email", "name": "Common"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("Common", artifact_type="email")
    assert len(results) == 1
    assert results[0]["id"] == "a2"


def test_gated_runtime_search_subsystem_violation():
    corpus = _make_corpus([{"id": "a1", "subsystem": "audit_trail", "name": "Secret"}])
    policy = _make_policy(
        {"alice": "engineer"},
        {"engineer": {"subsystems": ["crm"]}},
    )
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor_subsystem_access={"crm"},
    )
    results = runtime.search("Secret", artifact_type="audit_trail")
    assert results == []

    traj = runtime.trajectory()
    assert traj.subsystem_violations >= 1


def test_gated_runtime_search_strips_full_content():
    """Search results should only include metadata fields, not full content."""
    corpus = _make_corpus([
        {
            "id": "a1",
            "subsystem": "crm",
            "name": "Acme",
            "secret_field": "should_be_removed",
            "body": "should_also_be_removed",
            "timestamp": "2026-01-01T00:00:00",
            "summary": "A summary",
        }
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("Acme")
    assert len(results) == 1
    doc = results[0]
    assert "id" in doc or "_id" in doc
    assert "subsystem" in doc or doc.get("subsystem") == "crm"
    assert "timestamp" in doc
    assert "summary" in doc
    assert "secret_field" not in doc
    assert "body" not in doc


def test_gated_runtime_empty_trajectory():
    corpus = _make_corpus([])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    traj = runtime.trajectory()
    assert traj.task_id == "test"
    assert traj.tool_calls == []
    assert traj.horizon_violations == 0
    assert traj.actor_gate_violations == 0
    assert traj.subsystem_violations == 0
    assert traj.dead_ends_hit == 0
    assert traj.dead_ends_recovered == 0


def test_gated_runtime_dead_end_detection():
    corpus = _make_corpus([])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")

    # Call that returns empty
    runtime.fetch("missing1")
    # Call that also returns empty
    runtime.fetch("missing2")
    # Call that succeeds (use a real artifact)
    corpus._by_id["found"] = {"id": "found", "subsystem": "crm"}
    runtime.fetch("found")

    traj = runtime.trajectory()
    assert traj.dead_ends_hit >= 2
    assert traj.dead_ends_recovered == 1  # recovered after the second dead end


def test_gated_runtime_dead_end_no_recovery():
    corpus = _make_corpus([])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")

    runtime.fetch("missing1")
    runtime.fetch("missing2")

    traj = runtime.trajectory()
    assert traj.dead_ends_hit == 2
    assert traj.dead_ends_recovered == 0  # never recovered


def test_gated_runtime_timestamp_of():
    corpus = _make_corpus([{"id": "a1", "timestamp": "2026-03-15T10:00:00"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    assert runtime.timestamp_of("a1") == "2026-03-15T10:00:00"
    assert runtime.timestamp_of("missing") is None


def test_gated_runtime_subsystem_of():
    corpus = _make_corpus([{"id": "a1", "subsystem": "crm"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    assert runtime.subsystem_of("a1") == "crm"
    assert runtime.subsystem_of("missing") is None


def test_gated_runtime_list_ids():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    assert set(runtime.list_ids()) == {"a1", "a2"}
    assert runtime.list_ids(subsystem="crm") == ["a1"]


def test_gated_runtime_list_ids_actor_visible_filter():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor_visible_artifacts={"a1"},
    )
    ids = runtime.list_ids()
    assert ids == ["a1"]


def test_gated_runtime_list_ids_subsystem_blocked():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "audit_trail"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor_subsystem_access={"crm"},
    )
    ids = runtime.list_ids(subsystem="audit_trail")
    assert ids == []


def test_gated_runtime_call_log():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "name": "Acme"},
        {"id": "a2", "subsystem": "crm", "name": "Beta"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    runtime.fetch("a1")
    runtime.search("Beta")

    log = runtime.call_log
    assert len(log) == 2
    assert log[0].tool_name == "fetch_artifact"
    assert log[0].arguments["artifact_id"] == "a1"
    assert log[1].tool_name == "search_artifacts"
    assert log[1].arguments["query"] == "Beta"


def test_gated_runtime_all_subsystems():
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
        {"id": "a3"},  # no subsystem
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    # all_subsystems is populated by task_eval, not by GatedRuntime itself
    # Initially empty
    assert runtime.all_subsystems == []

    # task_eval sets _all_subsystems after construction
    runtime._all_subsystems = ["crm", "email"]
    assert runtime.all_subsystems == ["crm", "email"]


def test_gated_runtime_fetch_with_none_timestamp():
    """Artifact with timestamp=None should not trigger horizon violation."""
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "timestamp": None, "data": "ok"}
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus, policy=policy, task_id="test", as_of="2026-01-01T00:00:00"
    )
    doc = runtime.fetch("a1")
    assert doc is not None
    traj = runtime.trajectory()
    assert not traj.tool_calls[0].horizon_violation


def test_gated_runtime_fetch_artifact_no_subsystem():
    """Artifact with no subsystem field should be visible when actor has subsystem restrictions."""
    corpus = _make_corpus([{"id": "a1", "data": "no subsystem here"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor_subsystem_access={"crm"},
    )
    doc = runtime.fetch("a1")
    # No subsystem on artifact, so subsystem check passes (sub is None, not in set)
    assert doc is not None


def test_gated_runtime_fetch_same_id_twice():
    """Fetching the same artifact twice should record two tool calls."""
    corpus = _make_corpus([{"id": "a1", "subsystem": "crm", "data": "ok"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    runtime.fetch("a1")
    runtime.fetch("a1")
    assert len(runtime.call_log) == 2
    assert runtime.call_log[0].tool_name == "fetch_artifact"
    assert runtime.call_log[1].tool_name == "fetch_artifact"


def test_gated_runtime_search_empty_corpus():
    """Search on an empty corpus returns empty list, not error."""
    corpus = _make_corpus([])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("anything")
    assert results == []


def test_gated_runtime_search_zero_limit():
    """Search with limit=0 returns empty list."""
    corpus = _make_corpus([{"id": "a1", "subsystem": "crm", "name": "Acme"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("Acme", limit=0)
    assert results == []


def test_gated_runtime_search_no_subsystem_filter():
    """Search without artifact_type includes all subsystems."""
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "name": "Common"},
        {"id": "a2", "subsystem": "email", "name": "Common"},
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("Common")
    assert len(results) == 2


def test_gated_runtime_search_results_have_no_full_content():
    """After search, results must not contain body/secret fields — only metadata."""
    corpus = _make_corpus([
        {
            "id": "a1",
            "subsystem": "crm",
            "title": "Doc 1",
            "body": "very long content that should be stripped",
            "secret": "should not appear",
            "nested": {"key": "value"},
        }
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    results = runtime.search("Doc")
    assert len(results) == 1
    doc = results[0]
    assert "body" not in doc
    assert "secret" not in doc
    assert "nested" not in doc


def test_gated_runtime_fetch_after_search_strips():
    """fetch returns full content; search strips. Verify they differ."""
    corpus = _make_corpus([
        {
            "id": "a1",
            "subsystem": "crm",
            "title": "Doc",
            "body": "full body text",
            "timestamp": "2026-01-01",
        }
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")

    search_results = runtime.search("Doc")
    fetch_result = runtime.fetch("a1")

    assert "body" not in search_results[0]
    assert "body" in fetch_result


def test_gated_runtime_actor_none_no_gates():
    """When actor is None, actor_gate and subsystem checks should not apply."""
    corpus = _make_corpus([{"id": "a1", "subsystem": "audit_trail", "data": "secret"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor=None,
        actor_visible_artifacts=None,
        actor_subsystem_access=None,
    )
    doc = runtime.fetch("a1")
    assert doc is not None
    traj = runtime.trajectory()
    assert traj.actor_gate_violations == 0
    assert traj.subsystem_violations == 0


def test_gated_runtime_search_with_actor_filter():
    """actor filter in search arguments does NOT affect gating — only artifact_type does."""
    corpus = _make_corpus([
        {"id": "a1", "subsystem": "crm", "actors": ["alice"], "name": "Doc"}
    ])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="test",
        actor_subsystem_access={"crm"},
    )
    # The 'actor' argument in search is passed to corpus.search but not gated by runtime
    results = runtime.search("Doc")
    assert len(results) == 1


def test_gated_runtime_subsequent_trajectory_calls_consistent():
    """Multiple trajectory() calls should return the same data."""
    corpus = _make_corpus([{"id": "a1", "subsystem": "crm"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    runtime.fetch("a1")

    t1 = runtime.trajectory()
    t2 = runtime.trajectory()
    assert t1.horizon_violations == t2.horizon_violations
    assert t1.tool_calls == t2.tool_calls


def test_gated_runtime_no_tool_calls_trajectory_has_zeros():
    """Trajectory with no calls should have all zero counts."""
    corpus = _make_corpus([])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    traj = runtime.trajectory()
    assert traj.tool_calls == []
    assert traj.dead_ends_hit == 0
    assert traj.dead_ends_recovered == 0


def test_gated_runtime_fetch_then_search_then_empty():
    """Three dead ends: two empty, one recovery — correct count."""
    corpus = _make_corpus([{"id": "found", "subsystem": "crm"}])
    policy = _make_policy({}, {})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="test")
    runtime.fetch("missing")  # dead end
    runtime.fetch("missing2")  # dead end
    runtime.fetch("found")  # recovery
    runtime.fetch("missing3")  # dead end (no recovery after)
    runtime.fetch("missing4")  # dead end (no recovery after)

    traj = runtime.trajectory()
    assert traj.dead_ends_hit == 4
    assert traj.dead_ends_recovered == 1  # only between missing2 and found
