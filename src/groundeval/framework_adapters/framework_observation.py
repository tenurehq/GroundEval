from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..observe import ObservedToolCall


@dataclass
class ObservedWorkflowNode:
    node_id: str
    node_type: str = ""
    entered_at: str | None = None
    exited_at: str | None = None
    agent_name: str | None = None


@dataclass
class ObservedHandoff:
    from_executor_id: str
    to_executor_id: str
    timestamp: str | None = None
    payload_type: str | None = None


@dataclass
class ObservedAgent:
    agent_id: str
    agent_name: str | None = None
    agent_description: str | None = None
    role: str | None = None
    node_usage: list[str] = field(default_factory=list)
    tool_call_count: int = 0


@dataclass
class ObservedWorkflow:
    workflow_id: str
    workflow_name: str | None = None
    workflow_description: str | None = None
    node_count: int = 0
    nodes: list[ObservedWorkflowNode] = field(default_factory=list)
    handoff_count: int = 0
    handoffs: list[ObservedHandoff] = field(default_factory=list)
    branch_count: int = 0


@dataclass
class ObservedEvent:
    event_type: str
    timestamp: str | None = None
    agent_name: str | None = None
    node_name: str | None = None
    workflow_run_id: str | None = None
    branch_id: str | None = None
    parent_event_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservedApprovalEvent:
    request_id: str
    source_executor_id: str
    status: str
    timestamp: str | None = None
    request_type: str | None = None
    response_type: str | None = None


@dataclass
class ObservedCheckpointEvent:
    event_type: str
    timestamp: str | None = None
    executor_id: str | None = None
    state_keys: list[str] = field(default_factory=list)
    version: str | None = None


@dataclass
class ObservedContextEvent:
    event_type: str
    timestamp: str | None = None
    source_names: list[str] = field(default_factory=list)
    size_bytes: int = 0
    redacted_preview: str | None = None


@dataclass
class ObservedModelEvent:
    event_type: str
    timestamp: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    tool_schemas_count: int = 0


@dataclass
class ObservedError:
    error_type: str
    message: str
    timestamp: str | None = None
    executor_id: str | None = None
    traceback: str | None = None


@dataclass
class ObservedRun:
    run_id: str
    framework: str
    agent_class: str
    started_at: str | None = None
    completed_at: str | None = None
    total_latency_ms: float = 0.0
    tool_calls: list[ObservedToolCall] = field(default_factory=list)
    events: list[ObservedEvent] = field(default_factory=list)
    agents: list[ObservedAgent] = field(default_factory=list)
    workflow: ObservedWorkflow | None = None
    approvals: list[ObservedApprovalEvent] = field(default_factory=list)
    checkpoints: list[ObservedCheckpointEvent] = field(default_factory=list)
    context_events: list[ObservedContextEvent] = field(default_factory=list)
    model_events: list[ObservedModelEvent] = field(default_factory=list)
    final_output: Any = None
    errors: list[ObservedError] = field(default_factory=list)
    capabilities: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "agent_class": self.agent_class,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_latency_ms": self.total_latency_ms,
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "events": [asdict(e) for e in self.events],
            "agents": [asdict(a) for a in self.agents],
            "workflow": asdict(self.workflow) if self.workflow else None,
            "approvals": [asdict(a) for a in self.approvals],
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "context_events": [asdict(ce) for ce in self.context_events],
            "model_events": [asdict(me) for me in self.model_events],
            "final_output": self.final_output,
            "errors": [asdict(e) for e in self.errors],
            "capabilities": dict(self.capabilities),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObservedRun":
        workflow_data = d.get("workflow")
        workflow = None
        if workflow_data:
            workflow = ObservedWorkflow(
                workflow_id=workflow_data["workflow_id"],
                workflow_name=workflow_data.get("workflow_name"),
                workflow_description=workflow_data.get("workflow_description"),
                node_count=workflow_data.get("node_count", 0),
                nodes=[
                    ObservedWorkflowNode(**node)
                    for node in workflow_data.get("nodes", [])
                ],
                handoff_count=workflow_data.get("handoff_count", 0),
                handoffs=[
                    ObservedHandoff(**handoff)
                    for handoff in workflow_data.get("handoffs", [])
                ],
                branch_count=workflow_data.get("branch_count", 0),
            )

        return cls(
            run_id=d["run_id"],
            framework=d["framework"],
            agent_class=d["agent_class"],
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            total_latency_ms=d.get("total_latency_ms", 0.0),
            tool_calls=[ObservedToolCall(**tc) for tc in d.get("tool_calls", [])],
            events=[ObservedEvent(**e) for e in d.get("events", [])],
            agents=[ObservedAgent(**a) for a in d.get("agents", [])],
            workflow=workflow,
            approvals=[ObservedApprovalEvent(**a) for a in d.get("approvals", [])],
            checkpoints=[
                ObservedCheckpointEvent(**c) for c in d.get("checkpoints", [])
            ],
            context_events=[
                ObservedContextEvent(**ce) for ce in d.get("context_events", [])
            ],
            model_events=[ObservedModelEvent(**me) for me in d.get("model_events", [])],
            final_output=d.get("final_output"),
            errors=[ObservedError(**e) for e in d.get("errors", [])],
            capabilities=d.get("capabilities", {}),
        )
