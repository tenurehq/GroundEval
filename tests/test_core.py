from groundeval.adapters import InMemoryCorpusAdapter, YamlAccessPolicy
from groundeval.core import (
    ANSWER_SCHEMA_TASK,
    AgentTrajectory,
    AllowedTool,
    FixtureBackend,
    GatedRuntime,
    TaskContract,
    TaskEvalResult,
    TaskPrecondition,
    ToolCall,
    ToolExpectation,
)


def test_toolcall_full_shape():
    call = ToolCall(
        tool_name="fetch_artifact",
        arguments={"artifact_id": "a1"},
        result_ids=["a1"],
        timestamp_applied="2026-01-01T00:00:00",
        horizon_violation=False,
        actor_gate_violation=True,
        subsystem_violation=False,
        returned_empty=False,
        latency_ms=12.5,
        agent_name="alice",
        node_name="node-1",
        workflow_run_id="wf-1",
        branch_id="b1",
        call_id="c1",
        parent_event_id="p1",
    )
    assert call.tool_name == "fetch_artifact"
    assert call.arguments["artifact_id"] == "a1"
    assert call.result_ids == ["a1"]
    assert call.actor_gate_violation is True
    assert call.agent_name == "alice"
    assert call.node_name == "node-1"
    assert call.workflow_run_id == "wf-1"
    assert call.branch_id == "b1"
    assert call.call_id == "c1"
    assert call.parent_event_id == "p1"


def test_tool_expectation_from_dict_defaults_and_to_dict():
    exp = ToolExpectation.from_dict({"tool": "fetch_customer"})
    assert exp.tool == "fetch_customer"
    assert exp.match_args == {}
    assert exp.expected_return == {}
    assert exp.to_dict()["tool"] == "fetch_customer"


def test_agenttrajectory_to_dict_preserves_nested_dataclasses():
    traj = AgentTrajectory(
        task_id="t1",
        tool_calls=[
            ToolCall(
                tool_name="fetch_artifact",
                arguments={"artifact_id": "a1"},
                result_ids=["a1"],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=False,
                returned_empty=False,
                latency_ms=1.0,
            )
        ],
        cited_artifacts=["a1"],
        final_answer={"ok": True},
        total_latency_ms=5.0,
        prompt_tokens=10,
        completion_tokens=20,
    )
    data = traj.to_dict()
    assert data["task_id"] == "t1"
    assert data["tool_calls"][0]["tool_name"] == "fetch_artifact"
    assert data["final_answer"]["ok"] is True
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 20


def test_tool_expectation_from_dict_with_values():
    exp = ToolExpectation.from_dict({
        "tool": "fetch_customer",
        "match_args": {"artifact_id": "a1"},
        "expected_return": {"status": "active"},
    })
    assert exp.tool == "fetch_customer"
    assert exp.match_args == {"artifact_id": "a1"}
    assert exp.expected_return == {"status": "active"}


def test_task_precondition_from_dict_defaults_and_to_dict():
    pc = TaskPrecondition.from_dict({"check": "customer_ok"})
    assert pc.check == "customer_ok"
    assert pc.description == "customer_ok"
    assert pc.required_facts == []
    assert pc.ground_truth_field == ""
    assert pc.required_tool == ""
    assert pc.expected_field == ""
    assert pc.to_dict()["check"] == "customer_ok"


def test_task_precondition_from_dict_full():
    pc = TaskPrecondition.from_dict({
        "check": "customer_ok",
        "description": "Verify customer status",
        "required_facts": ["status"],
        "ground_truth_field": "a1.status",
        "required_tool": "fetch_customer",
        "expected_field": "status",
    })
    assert pc.description == "Verify customer status"
    assert pc.required_facts == ["status"]
    assert pc.ground_truth_field == "a1.status"
    assert pc.required_tool == "fetch_customer"
    assert pc.expected_field == "status"


def test_allowed_tool_from_dict_defaults_and_to_dict():
    tool = AllowedTool.from_dict("fetch_customer", {})
    assert tool.tool_name == "fetch_customer"
    assert tool.entity_arg == ""
    assert tool.returns == {}
    assert tool.action is False
    assert tool.artifact_id == ""
    assert tool.subsystem == ""
    assert tool.timestamp == ""
    assert tool.to_dict()["artifact_id"] == ""


def test_allowed_tool_from_dict_full():
    tool = AllowedTool.from_dict(
        "fetch_customer",
        {
            "entity_arg": "artifact_id",
            "returns": {"status": "active"},
            "action": True,
            "artifact_id": "crm-1",
            "subsystem": "crm",
            "timestamp": "2026-01-01T00:00:00",
        },
    )
    assert tool.entity_arg == "artifact_id"
    assert tool.returns["status"] == "active"
    assert tool.action is True
    assert tool.artifact_id == "crm-1"
    assert tool.subsystem == "crm"
    assert tool.timestamp == "2026-01-01T00:00:00"


def test_task_contract_defaults():
    contract = TaskContract.from_dict({"name": "t1", "preconditions": []})
    assert contract.name == "t1"
    assert contract.task_description == "t1"
    assert contract.valid_action == "all_preconditions_pass"
    assert contract.decision_field == "should_act"
    assert contract.artifacts_dir == "./data"
    assert contract.actor is None
    assert contract.role is None
    assert contract.actors == {}
    assert contract.roles == {}
    assert contract.inputs == {}
    assert contract.allowed_tools == []
    assert contract.tool_expectations == []
    assert contract.expected_action is None
    assert contract.action_tool == ""
    assert contract.required_agents == []
    assert contract.required_handoffs == []
    assert contract.required_agent_tool_expectations == []
    assert contract.is_fixture_mode is False
    assert contract.is_framework_contract is False


def test_task_contract_from_dict_full_and_to_dict():
    contract = TaskContract.from_dict({
        "name": "t1",
        "task_description": "Verify customer",
        "preconditions": [
            {
                "check": "pc1",
                "description": "status active",
                "required_facts": ["status"],
                "ground_truth_field": "crm-1.status",
                "required_tool": "fetch_customer",
                "expected_field": "status",
            }
        ],
        "valid_action": "all_preconditions_pass",
        "decision_field": "should_act",
        "artifacts_dir": "./fixtures",
        "actor": "alice",
        "role": "sales",
        "actors": {"alice": "sales"},
        "roles": {"sales": {"subsystems": ["crm"]}},
        "inputs": {"as_of": "2026-01-01T00:00:00"},
        "allowed_tools": {
            "fetch_customer": {
                "entity_arg": "artifact_id",
                "returns": {"status": "active"},
                "artifact_id": "crm-1",
                "subsystem": "crm",
                "timestamp": "2026-01-01T00:00:00",
            }
        },
        "tool_expectations": [
            {
                "tool": "fetch_customer",
                "match_args": {"artifact_id": "crm-1"},
                "expected_return": {"status": "active"},
            }
        ],
        "expected_action": True,
        "action_tool": "send_email",
        "required_agents": [{"agent_name": "planner"}],
        "required_handoffs": [{"from_agent": "planner", "to_agent": "executor"}],
        "required_agent_tool_expectations": [
            {"agent_name": "planner", "tool": "fetch_customer"}
        ],
    })
    data = contract.to_dict()
    assert contract.is_fixture_mode is True
    assert contract.is_framework_contract is True
    assert data["name"] == "t1"
    assert data["preconditions"][0]["check"] == "pc1"
    assert data["allowed_tools"][0]["tool_name"] == "fetch_customer"
    assert data["tool_expectations"][0]["tool"] == "fetch_customer"
    assert data["expected_action"] is True


def test_task_contract_framework_detection_variants():
    c1 = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1", "required_tool": "fetch_customer"}],
    })
    c2 = TaskContract.from_dict({
        "name": "t2",
        "preconditions": [{"check": "pc1", "expected_field": "status"}],
    })
    c3 = TaskContract.from_dict({
        "name": "t3",
        "preconditions": [{"check": "pc1"}],
        "tool_expectations": [{"tool": "fetch_customer"}],
    })
    c4 = TaskContract.from_dict({
        "name": "t4",
        "preconditions": [{"check": "pc1"}],
    })
    assert c1.is_framework_contract is True
    assert c2.is_framework_contract is True
    assert c3.is_framework_contract is True
    assert c4.is_framework_contract is False


def test_task_eval_result_to_dict_total_violations():
    result = TaskEvalResult(
        task_name="t1",
        counterfactual_score=0.1,
        silence_score=0.2,
        perspective_score=0.3,
        overall_score=0.2,
        answer_correct=False,
        horizon_violations=1,
        actor_gate_violations=2,
        subsystem_violations=3,
    )
    data = result.to_dict()
    assert data["total_violations"] == 6


def test_fixture_backend_duplicate_artifact_ids_overwrite_last():
    t1 = AllowedTool(
        tool_name="tool1", artifact_id="a1", returns={"x": 1}, subsystem="crm"
    )
    t2 = AllowedTool(
        tool_name="tool2", artifact_id="a1", returns={"x": 2}, subsystem="email"
    )
    backend = FixtureBackend([t1, t2])
    doc = backend.fetch("a1")
    assert doc["x"] == 2
    assert doc["subsystem"] == "email"


def test_fixture_backend_fetch_search_and_metadata():
    backend = FixtureBackend([
        AllowedTool(
            tool_name="fetch_customer",
            artifact_id="crm-1",
            returns={"status": "active", "name": "Acme"},
            subsystem="crm",
            timestamp="2026-01-01T00:00:00",
        ),
        AllowedTool(
            tool_name="fetch_email",
            artifact_id="email-1",
            returns={"subject": "Hello"},
            subsystem="email",
        ),
    ])

    fetched = backend.fetch("crm-1")
    assert fetched["id"] == "crm-1"
    assert fetched["subsystem"] == "crm"
    assert fetched["timestamp"] == "2026-01-01T00:00:00"
    assert backend.fetch("crm-1", as_of="2025-01-01T00:00:00") is None
    assert backend.fetch("missing") is None

    results = backend.search("Acme")
    assert len(results) == 1
    assert results[0]["id"] == "crm-1"
    assert backend.search("Acme", artifact_type="email") == []
    assert backend.search("Acme", limit=0) == []
    assert backend.timestamp_of("crm-1") == "2026-01-01T00:00:00"
    assert backend.timestamp_of("missing") is None
    assert backend.subsystem_of("email-1") == "email"
    assert backend.subsystem_of("missing") is None
    assert set(backend.list_ids()) == {"crm-1", "email-1"}
    assert backend.list_ids(subsystem="crm") == ["crm-1"]


def test_fixture_backend_search_respects_as_of_and_case_insensitive_match():
    backend = FixtureBackend([
        AllowedTool(
            tool_name="fetch_customer",
            artifact_id="crm-1",
            returns={"name": "ACME Corp"},
            subsystem="crm",
            timestamp="2027-01-01T00:00:00",
        ),
        AllowedTool(
            tool_name="fetch_customer2",
            artifact_id="crm-2",
            returns={"name": "Acme Europe"},
            subsystem="crm",
            timestamp="2025-01-01T00:00:00",
        ),
    ])
    results = backend.search("acme", as_of="2026-01-01T00:00:00")
    assert len(results) == 1
    assert results[0]["id"] == "crm-2"


def test_gated_runtime_call_log_returns_copy():
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "crm"}])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")
    runtime.fetch("a1")
    log1 = runtime.call_log
    log2 = runtime.call_log
    assert log1 is not log2
    log1.append("mutate")
    assert len(runtime.call_log) == 1


def test_gated_runtime_all_subsystems_returns_copy():
    corpus = InMemoryCorpusAdapter([])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")
    runtime._all_subsystems = ["crm"]
    s1 = runtime.all_subsystems
    s2 = runtime.all_subsystems
    assert s1 == ["crm"]
    assert s1 is not s2
    s1.append("email")
    assert runtime.all_subsystems == ["crm"]


def test_gated_runtime_fetch_success():
    corpus = InMemoryCorpusAdapter([
        {
            "id": "a1",
            "subsystem": "crm",
            "timestamp": "2025-01-01T00:00:00",
            "name": "Acme",
        }
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        as_of="2026-01-01T00:00:00",
        actor_visible_artifacts={"a1"},
        actor_subsystem_access={"crm"},
    )
    doc = runtime.fetch("a1")
    assert doc["name"] == "Acme"
    traj = runtime.trajectory()
    assert len(traj.tool_calls) == 1
    assert traj.horizon_violations == 0
    assert traj.actor_gate_violations == 0
    assert traj.subsystem_violations == 0


def test_gated_runtime_fetch_missing_artifact_records_dead_end():
    corpus = InMemoryCorpusAdapter([])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")
    assert runtime.fetch("missing") is None
    traj = runtime.trajectory()
    assert len(traj.tool_calls) == 1
    assert traj.tool_calls[0].returned_empty is True
    assert traj.dead_ends_hit == 1


def test_gated_runtime_fetch_horizon_violation():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm", "timestamp": "2027-01-01T00:00:00"}
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        as_of="2026-01-01T00:00:00",
    )
    assert runtime.fetch("a1") is None
    traj = runtime.trajectory()
    assert traj.horizon_violations == 1
    assert traj.tool_calls[0].returned_empty is True


def test_gated_runtime_fetch_actor_gate_violation_only():
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "crm"}])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_visible_artifacts={"other"},
        actor_subsystem_access={"crm"},
    )
    assert runtime.fetch("a1") is None
    traj = runtime.trajectory()
    assert traj.actor_gate_violations == 1
    assert traj.subsystem_violations == 0


def test_gated_runtime_fetch_subsystem_violation_only():
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "email"}])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_subsystem_access={"crm"},
    )
    assert runtime.fetch("a1") is None
    traj = runtime.trajectory()
    assert traj.subsystem_violations == 1
    assert traj.actor_gate_violations == 0


def test_gated_runtime_fetch_combined_actor_and_subsystem_violation():
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "email"}])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_visible_artifacts={"other"},
        actor_subsystem_access={"crm"},
    )
    assert runtime.fetch("a1") is None
    traj = runtime.trajectory()
    assert traj.actor_gate_violations == 1
    assert traj.subsystem_violations == 1


def test_gated_runtime_search_strips_non_search_fields():
    corpus = InMemoryCorpusAdapter([
        {
            "id": "a1",
            "subsystem": "crm",
            "title": "Doc",
            "summary": "Short",
            "description": "Desc",
            "body": "secret",
            "private": "remove",
            "timestamp": "2026-01-01T00:00:00",
        }
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")
    results = runtime.search("Doc")
    assert len(results) == 1
    assert "title" in results[0]
    assert "summary" in results[0]
    assert "description" in results[0]
    assert "body" not in results[0]
    assert "private" not in results[0]


def test_gated_runtime_search_empty_result_records_dead_end():
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "crm", "title": "Doc"}])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")
    assert runtime.search("missing") == []
    traj = runtime.trajectory()
    assert traj.tool_calls[0].returned_empty is True
    assert traj.dead_ends_hit == 1


def test_gated_runtime_search_blocked_by_requested_artifact_type():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "email", "title": "Mail"}
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_subsystem_access={"crm"},
    )
    results = runtime.search("Mail", artifact_type="email")
    assert results == []
    traj = runtime.trajectory()
    assert traj.subsystem_violations == 1
    assert traj.tool_calls[0].returned_empty is True


def test_gated_runtime_search_as_of_filters_future_records():
    corpus = InMemoryCorpusAdapter([
        {
            "id": "a1",
            "subsystem": "crm",
            "title": "Future",
            "timestamp": "2027-01-01T00:00:00",
        },
        {
            "id": "a2",
            "subsystem": "crm",
            "title": "Past",
            "timestamp": "2025-01-01T00:00:00",
        },
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        as_of="2026-01-01T00:00:00",
    )
    results = runtime.search("crm")
    ids = {r["id"] for r in results}
    assert "a1" not in ids
    assert "a2" in ids


def test_gated_runtime_list_ids_unrestricted():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")
    assert set(runtime.list_ids()) == {"a1", "a2"}


def test_gated_runtime_list_ids_with_actor_visibility_only():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_visible_artifacts={"a1"},
    )
    assert runtime.list_ids() == ["a1"]


def test_gated_runtime_list_ids_with_visibility_and_subsystem_filter():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
        {"id": "a3", "subsystem": "crm"},
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_visible_artifacts={"a1", "a2"},
    )
    assert runtime.list_ids(subsystem="crm") == ["a1"]


def test_gated_runtime_list_ids_subsystem_request_blocked():
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        actor_subsystem_access={"crm"},
    )
    assert runtime.list_ids(subsystem="email") == []


def test_gated_runtime_trajectory_dead_end_recovery_multiple_transitions():
    corpus = InMemoryCorpusAdapter([
        {"id": "ok1", "subsystem": "crm"},
        {"id": "ok2", "subsystem": "crm"},
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(corpus=corpus, policy=policy, task_id="t1")

    runtime.fetch("missing-1")
    runtime.fetch("ok1")
    runtime.fetch("missing-2")
    runtime.fetch("missing-3")
    runtime.fetch("ok2")

    traj = runtime.trajectory()
    assert traj.dead_ends_hit == 3
    assert traj.dead_ends_recovered == 2


def test_gated_runtime_trajectory_counts_multiple_violation_types():
    corpus = InMemoryCorpusAdapter([
        {"id": "future", "subsystem": "crm", "timestamp": "2027-01-01T00:00:00"},
        {"id": "hidden", "subsystem": "crm"},
        {"id": "wrong_sub", "subsystem": "email"},
    ])
    policy = YamlAccessPolicy({"actors": {}, "roles": {}})
    runtime = GatedRuntime(
        corpus=corpus,
        policy=policy,
        task_id="t1",
        as_of="2026-01-01T00:00:00",
        actor_visible_artifacts={"future", "wrong_sub"},
        actor_subsystem_access={"crm"},
    )

    runtime.fetch("future")
    runtime.fetch("hidden")
    runtime.fetch("wrong_sub")

    traj = runtime.trajectory()
    assert traj.horizon_violations == 1
    assert traj.actor_gate_violations == 1
    assert traj.subsystem_violations == 1


def test_answer_schema_task_structure():
    assert ANSWER_SCHEMA_TASK["type"] == "object"
    assert "preconditions_verified" in ANSWER_SCHEMA_TASK["required"]
    assert "all_preconditions_pass" in ANSWER_SCHEMA_TASK["required"]
    assert "reasoning" in ANSWER_SCHEMA_TASK["required"]
    assert ANSWER_SCHEMA_TASK["properties"]["preconditions_verified"]["type"] == "array"
