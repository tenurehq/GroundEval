from groundeval.framework_adapters.framework_observation import (
    ObservedAgent,
    ObservedApprovalEvent,
    ObservedCheckpointEvent,
    ObservedContextEvent,
    ObservedError,
    ObservedEvent,
    ObservedHandoff,
    ObservedModelEvent,
    ObservedRun,
    ObservedWorkflow,
    ObservedWorkflowNode,
)
from groundeval.observe import ObservedToolCall


def test_observed_run_to_dict_and_from_dict_round_trip_full():
    run = ObservedRun(
        run_id="r1",
        framework="maf",
        agent_class="pkg.Agent",
        started_at="1.0",
        completed_at="2.0",
        total_latency_ms=1000.0,
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"id": "a1"},
                latency_ms=10.0,
                agent_id="agent-1",
                agent_name="planner",
                node_name="node-1",
                workflow_run_id="wf-1",
                branch_id="b1",
                parent_event_id="evt-1",
            )
        ],
        events=[
            ObservedEvent(
                event_type="otel.span",
                timestamp="1.1",
                agent_name="planner",
                node_name="node-1",
                workflow_run_id="wf-1",
                branch_id="b1",
                parent_event_id="evt-1",
                payload={"a": 1},
            )
        ],
        agents=[
            ObservedAgent(
                agent_id="agent-1",
                agent_name="planner",
                agent_description="plans",
                role="planner",
                node_usage=["node-1"],
                tool_call_count=1,
            )
        ],
        workflow=ObservedWorkflow(
            workflow_id="wf-1",
            workflow_name="Main",
            workflow_description="desc",
            node_count=1,
            nodes=[
                ObservedWorkflowNode(
                    node_id="node-1",
                    node_type="executor",
                    entered_at="1.0",
                    exited_at="1.5",
                    agent_name="planner",
                )
            ],
            handoff_count=1,
            handoffs=[
                ObservedHandoff(
                    from_executor_id="agent-1",
                    to_executor_id="agent-2",
                    timestamp="1.2",
                    payload_type="handoff",
                )
            ],
            branch_count=1,
        ),
        approvals=[
            ObservedApprovalEvent(
                request_id="req-1",
                source_executor_id="agent-1",
                status="approved",
                timestamp="1.3",
                request_type="safety",
                response_type="ok",
            )
        ],
        checkpoints=[
            ObservedCheckpointEvent(
                event_type="checkpoint.saved",
                timestamp="1.4",
                executor_id="agent-1",
                state_keys=["k1", "k2"],
                version="v1",
            )
        ],
        context_events=[
            ObservedContextEvent(
                event_type="context.loaded",
                timestamp="1.5",
                source_names=["crm", "email"],
                size_bytes=123,
                redacted_preview="preview",
            )
        ],
        model_events=[
            ObservedModelEvent(
                event_type="model.call.completed",
                timestamp="1.6",
                model_name="gpt-4o",
                provider_name="openai",
                input_tokens=10,
                output_tokens=5,
                finish_reason="stop",
                tool_schemas_count=2,
            )
        ],
        final_output={"should_act": True},
        errors=[
            ObservedError(
                error_type="SpanError",
                message="boom",
                timestamp="1.7",
                executor_id="agent-1",
                traceback="tb",
            )
        ],
        capabilities={"tool_calls": True, "workflow_nodes": True},
    )

    data = run.to_dict()
    restored = ObservedRun.from_dict(data)

    assert restored.run_id == "r1"
    assert restored.framework == "maf"
    assert restored.tool_calls[0].tool_name == "fetch_customer"
    assert restored.events[0].event_type == "otel.span"
    assert restored.agents[0].agent_name == "planner"
    assert restored.workflow.workflow_id == "wf-1"
    assert restored.workflow.nodes[0].node_id == "node-1"
    assert restored.workflow.handoffs[0].to_executor_id == "agent-2"
    assert restored.approvals[0].request_id == "req-1"
    assert restored.checkpoints[0].version == "v1"
    assert restored.context_events[0].size_bytes == 123
    assert restored.model_events[0].model_name == "gpt-4o"
    assert restored.errors[0].message == "boom"
    assert restored.capabilities["tool_calls"] is True


def test_observed_run_from_dict_without_workflow():
    data = {
        "run_id": "r1",
        "framework": "maf",
        "agent_class": "pkg.Agent",
        "tool_calls": [],
        "evidence": [],
        "events": [],
        "agents": [],
        "workflow": None,
        "approvals": [],
        "checkpoints": [],
        "context_events": [],
        "model_events": [],
        "final_output": {},
        "errors": [],
        "capabilities": {},
    }
    restored = ObservedRun.from_dict(data)
    assert restored.workflow is None
    assert restored.run_id == "r1"


def test_observed_run_to_dict_minimal_fields():
    run = ObservedRun(
        run_id="r1",
        framework="maf",
        agent_class="pkg.Agent",
    )

    data = run.to_dict()
    assert data["run_id"] == "r1"
    assert data["framework"] == "maf"
    assert data["agent_class"] == "pkg.Agent"
    assert data["tool_calls"] == []
    assert data["events"] == []
    assert data["agents"] == []
    assert data["errors"] == []


def test_observed_run_from_dict_minimal_fields_defaults():
    data = {
        "run_id": "r1",
        "framework": "maf",
        "agent_class": "pkg.Agent",
    }

    restored = ObservedRun.from_dict(data)
    assert restored.run_id == "r1"
    assert restored.framework == "maf"
    assert restored.agent_class == "pkg.Agent"
    assert restored.tool_calls == []
    assert restored.events == []
    assert restored.agents == []
    assert restored.workflow is None
    assert restored.final_output is None
    assert restored.errors == []
    assert restored.capabilities == {}


def test_observed_run_from_dict_partial_workflow_defaults():
    data = {
        "run_id": "r1",
        "framework": "maf",
        "agent_class": "pkg.Agent",
        "workflow": {
            "workflow_id": "wf-1",
        },
    }

    restored = ObservedRun.from_dict(data)
    assert restored.workflow is not None
    assert restored.workflow.workflow_id == "wf-1"
    assert restored.workflow.node_count == 0
    assert restored.workflow.nodes == []
    assert restored.workflow.handoff_count == 0
    assert restored.workflow.handoffs == []
    assert restored.workflow.branch_count == 0


def test_observed_run_round_trip_with_empty_nested_objects():
    run = ObservedRun(
        run_id="r2",
        framework="maf",
        agent_class="pkg.Agent",
        workflow=ObservedWorkflow(workflow_id="wf-2"),
        approvals=[],
        checkpoints=[],
        context_events=[],
        model_events=[],
        errors=[],
        capabilities={"tool_calls": False},
    )

    restored = ObservedRun.from_dict(run.to_dict())
    assert restored.workflow is not None
    assert restored.workflow.workflow_id == "wf-2"
    assert restored.capabilities["tool_calls"] is False
