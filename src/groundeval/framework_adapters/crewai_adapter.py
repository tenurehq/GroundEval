from __future__ import annotations

import importlib
import json
import logging
import time
from typing import Any

from ..observe import AgentObserver, ObservedToolCall, RecordingRuntime
from .framework_observation import (
    ObservedAgent,
    ObservedError,
    ObservedEvent,
    ObservedHandoff,
    ObservedModelEvent,
    ObservedRun as RichObservedRun,
    ObservedWorkflow,
    ObservedWorkflowNode,
)

logger = logging.getLogger("groundeval.adapters.crewai")


def _load_crew(agent_class_path: str) -> Any:
    module_path, attr_name = agent_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    crew_obj = getattr(module, attr_name)

    if callable(crew_obj) and not isinstance(crew_obj, type):
        return crew_obj()

    if isinstance(crew_obj, type):
        if hasattr(crew_obj, "crew") and callable(crew_obj.crew):
            instance = crew_obj()
            return instance.crew()
        return crew_obj()

    return crew_obj


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _parse_jsonish(value: Any) -> Any:
    value = _jsonish(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in '[{"':
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _event_type_name(event: Any) -> str:
    explicit = _safe_getattr(event, "type")
    if explicit:
        return str(explicit)
    return event.__class__.__name__


def _event_timestamp(event: Any) -> str | None:
    value = _safe_getattr(event, "timestamp")
    if value is None:
        return None
    return str(value)


def _event_id(event: Any) -> str | None:
    value = _safe_getattr(event, "event_id")
    return str(value) if value is not None else None


def _parent_event_id(event: Any) -> str | None:
    for key in ("parent_event_id", "triggered_by_event_id", "started_event_id"):
        value = _safe_getattr(event, key)
        if value is not None:
            return str(value)
    return None


def _agent_id_from_event(event: Any) -> str | None:
    value = _safe_getattr(event, "agent_id")
    if value is not None:
        return str(value)
    agent = _safe_getattr(event, "agent")
    if agent is not None:
        value = _safe_getattr(agent, "id")
        if value is not None:
            return str(value)
    return None


def _agent_name_from_event(event: Any) -> str | None:
    for key in ("agent_role", "agent_name"):
        value = _safe_getattr(event, key)
        if value is not None:
            return str(value)
    agent = _safe_getattr(event, "agent")
    if agent is not None:
        value = _safe_getattr(agent, "role")
        if value is not None:
            return str(value)
    return None


def _task_id_from_event(event: Any) -> str | None:
    value = _safe_getattr(event, "task_id")
    return str(value) if value is not None else None


def _task_name_from_event(event: Any) -> str | None:
    value = _safe_getattr(event, "task_name")
    if value is not None:
        return str(value)
    task = _safe_getattr(event, "task")
    if task is not None:
        description = _safe_getattr(task, "description")
        if description is not None:
            return str(description)
    return None


def _tool_name_from_event(event: Any) -> str | None:
    value = _safe_getattr(event, "tool_name")
    return str(value) if value is not None else None


def _tool_args_from_event(event: Any) -> dict[str, Any]:
    value = _safe_getattr(event, "tool_args")
    if isinstance(value, dict):
        return dict(value)
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _tool_output_from_event(event: Any) -> Any:
    value = _safe_getattr(event, "output")
    if value is None:
        return None
    return _parse_jsonish(value)


def _llm_model_name(event: Any) -> str | None:
    for key in ("model", "model_name"):
        value = _safe_getattr(event, key)
        if value is not None:
            return str(value)
    llm = _safe_getattr(event, "llm")
    if llm is not None:
        for key in ("model", "model_name"):
            value = _safe_getattr(llm, key)
            if value is not None:
                return str(value)
    return None


def _llm_provider_name(event: Any) -> str | None:
    llm = _safe_getattr(event, "llm")
    if llm is not None:
        return str(llm.__class__.__module__.split(".")[0])
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_crew_output(result: Any) -> dict[str, Any]:
    if hasattr(result, "pydantic") and result.pydantic is not None:
        pydantic_result = result.pydantic
        if hasattr(pydantic_result, "model_dump") and callable(
            pydantic_result.model_dump
        ):
            dumped = pydantic_result.model_dump()
            if isinstance(dumped, dict):
                return dumped

    raw = getattr(result, "raw", "")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    if hasattr(result, "model_dump") and callable(result.model_dump):
        try:
            dumped = result.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    return {"raw_output": str(result)[:1000]}


class _CrewAIEventCollector:
    def __init__(self, run_id: str, agent_class: str):
        self.run_id = run_id
        self.agent_class = agent_class
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.events: list[ObservedEvent] = []
        self.tool_calls: list[ObservedToolCall] = []
        self.agents: dict[str, ObservedAgent] = {}
        self.workflow_nodes: dict[str, ObservedWorkflowNode] = {}
        self.handoffs: list[ObservedHandoff] = []
        self.model_events: list[ObservedModelEvent] = []
        self.errors: list[ObservedError] = []
        self.final_output: Any = None
        self._listener = None
        self._bus = None

    def install(self) -> None:
        try:
            from crewai.events import BaseEventListener, crewai_event_bus
        except Exception as exc:
            raise RuntimeError(
                "CrewAI observe requires CrewAI event listeners and crewai_event_bus."
            ) from exc

        collector = self

        class GroundEvalCrewAIListener(BaseEventListener):
            def __init__(self):
                super().__init__()

            def setup_listeners(self, event_bus):
                @event_bus.on(object)
                def on_any_event(source, event):
                    collector.capture(source, event)

        self._listener = GroundEvalCrewAIListener()
        self._bus = crewai_event_bus

    def flush(self) -> None:
        if self._bus is None:
            return
        flush = getattr(self._bus, "flush", None)
        if callable(flush):
            flush()

    def capture(self, source: Any, event: Any) -> None:
        event_type = _event_type_name(event)
        timestamp = _event_timestamp(event)
        parent_event_id = _parent_event_id(event)
        agent_id = _agent_id_from_event(event)
        agent_name = _agent_name_from_event(event)
        task_id = _task_id_from_event(event)
        task_name = _task_name_from_event(event)

        payload = {}
        for key, value in vars(event).items():
            payload[str(key)] = _jsonish(value)
        payload["event_class"] = event.__class__.__name__
        payload["source_class"] = (
            source.__class__.__name__ if source is not None else None
        )

        self.events.append(
            ObservedEvent(
                event_type=event_type,
                timestamp=timestamp,
                agent_name=agent_name,
                node_name=task_name,
                workflow_run_id=self.run_id,
                branch_id=None,
                parent_event_id=parent_event_id,
                payload=payload,
            )
        )

        if agent_id or agent_name:
            key = agent_id or agent_name or "unknown"
            if key not in self.agents:
                self.agents[key] = ObservedAgent(
                    agent_id=agent_id or key,
                    agent_name=agent_name or key,
                    role=agent_name,
                )

        node_key = task_id or task_name
        if node_key:
            existing = self.workflow_nodes.get(str(node_key))
            if existing is None:
                self.workflow_nodes[str(node_key)] = ObservedWorkflowNode(
                    node_id=str(node_key),
                    node_type=event_type,
                    entered_at=timestamp,
                    exited_at=None,
                    agent_name=agent_name,
                )
            elif timestamp:
                existing.exited_at = timestamp
                if not existing.agent_name and agent_name:
                    existing.agent_name = agent_name

        lower = event_type.lower()

        if "kickoffstarted" in lower and self.started_at is None:
            self.started_at = time.time()

        if "kickoffcompleted" in lower:
            self.completed_at = time.time()
            output = _safe_getattr(event, "output")
            if output is not None:
                self.final_output = _jsonish(output)

        if "kickofffailed" in lower:
            self.completed_at = time.time()
            self.errors.append(
                ObservedError(
                    error_type=event.__class__.__name__,
                    message=str(_safe_getattr(event, "error", "Crew kickoff failed")),
                    timestamp=timestamp,
                )
            )

        if (
            "agentexecutionerror" in lower
            or "taskfailed" in lower
            or "llmcallfailed" in lower
            or "toolusageerror" in lower
            or "toolexecutionerror" in lower
        ):
            self.errors.append(
                ObservedError(
                    error_type=event.__class__.__name__,
                    message=str(
                        _safe_getattr(event, "error")
                        or _safe_getattr(event, "error_message")
                        or event_type
                    ),
                    timestamp=timestamp,
                    executor_id=agent_id,
                )
            )

        if "llmcallcompleted" in lower:
            self.model_events.append(
                ObservedModelEvent(
                    event_type="model.call.completed",
                    timestamp=timestamp,
                    model_name=_llm_model_name(event),
                    provider_name=_llm_provider_name(event),
                    input_tokens=_coerce_int(
                        _safe_getattr(event, "prompt_tokens")
                        or _safe_getattr(event, "input_tokens")
                    ),
                    output_tokens=_coerce_int(
                        _safe_getattr(event, "completion_tokens")
                        or _safe_getattr(event, "output_tokens")
                    ),
                    finish_reason=_safe_getattr(event, "finish_reason"),
                    tool_schemas_count=0,
                )
            )

        if "toolusagefinished" in lower:
            tool_name = _tool_name_from_event(event) or "unknown_tool"
            tool_args = _tool_args_from_event(event)
            output = _tool_output_from_event(event)

            observed_call = ObservedToolCall(
                tool_name=tool_name,
                arguments=tool_args,
                return_value=output,
                latency_ms=0.0,
                agent_id=agent_id,
                agent_name=agent_name,
                node_name=task_name,
                workflow_run_id=self.run_id,
                branch_id=None,
                parent_event_id=parent_event_id or _event_id(event),
            )
            self.tool_calls.append(observed_call)

            if agent_id and agent_id in self.agents:
                self.agents[agent_id].tool_call_count += 1
            elif agent_name:
                for observed_agent in self.agents.values():
                    if observed_agent.agent_name == agent_name:
                        observed_agent.tool_call_count += 1
                        break

        if "delegation" in lower:
            from_executor_id = str(
                _safe_getattr(event, "agent_id")
                or _safe_getattr(event, "from_agent")
                or _safe_getattr(event, "source_agent_id")
                or ""
            )
            to_executor_id = str(
                _safe_getattr(event, "target_agent_id")
                or _safe_getattr(event, "to_agent")
                or _safe_getattr(event, "endpoint")
                or ""
            )
            if from_executor_id and to_executor_id:
                self.handoffs.append(
                    ObservedHandoff(
                        from_executor_id=from_executor_id,
                        to_executor_id=to_executor_id,
                        timestamp=timestamp,
                        payload_type=event_type,
                    )
                )

    def to_rich_observed_run(self) -> RichObservedRun:
        workflow = None
        if self.workflow_nodes or self.handoffs:
            workflow = ObservedWorkflow(
                workflow_id=self.run_id,
                workflow_name="CrewAI observed workflow",
                workflow_description=None,
                node_count=len(self.workflow_nodes),
                nodes=list(self.workflow_nodes.values()),
                handoff_count=len(self.handoffs),
                handoffs=self.handoffs,
                branch_count=0,
            )

        started_at = str(self.started_at) if self.started_at is not None else None
        completed_at = str(self.completed_at) if self.completed_at is not None else None
        total_latency_ms = 0.0
        if self.started_at is not None and self.completed_at is not None:
            total_latency_ms = (self.completed_at - self.started_at) * 1000

        capabilities = {
            "event_bus": True,
            "tool_calls": bool(self.tool_calls),
            "agent_turns": bool(self.agents),
            "workflow_nodes": bool(self.workflow_nodes),
            "handoffs": bool(self.handoffs),
            "approvals": False,
            "checkpoints": False,
            "context_injection": False,
            "model_calls": bool(self.model_events),
        }

        return RichObservedRun(
            run_id=self.run_id,
            framework="crewai",
            agent_class=self.agent_class,
            started_at=started_at,
            completed_at=completed_at,
            total_latency_ms=total_latency_ms,
            tool_calls=list(self.tool_calls),
            events=self.events,
            agents=list(self.agents.values()),
            workflow=workflow,
            approvals=[],
            checkpoints=[],
            context_events=[],
            model_events=self.model_events,
            final_output=self.final_output,
            errors=self.errors,
            capabilities=capabilities,
        )


class CrewAIObserver(AgentObserver):
    def load_agent(self, class_path: str) -> Any:
        return _load_crew(class_path)

    def instrument_agent(
        self,
        agent: Any,
        recording: RecordingRuntime,
    ) -> Any:
        run_id = f"crewai_observed_{agent.__class__.__module__.replace('.', '_')}_{agent.__class__.__name__}_{int(time.time())}"
        collector = _CrewAIEventCollector(
            run_id=run_id,
            agent_class=f"{agent.__class__.__module__}.{agent.__class__.__name__}",
        )
        collector.install()
        agent._groundeval_recording = recording
        agent._groundeval_crewai_collector = collector
        return agent

    def execute_agent(self, agent: Any) -> Any:
        collector = getattr(agent, "_groundeval_crewai_collector", None)
        recording = getattr(agent, "_groundeval_recording", None)

        if collector is None:
            raise RuntimeError("CrewAI observer missing installed event collector.")

        collector.started_at = time.time()
        result = None
        try:
            result = agent.kickoff()
            return result
        finally:
            collector.completed_at = time.time()
            collector.flush()

            if collector.final_output is None and result is not None:
                collector.final_output = _parse_crew_output(result)

            rich_run = collector.to_rich_observed_run()
            agent._groundeval_framework_observed_run = rich_run

            if recording is not None:
                for tc in rich_run.tool_calls:
                    recording.record(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        return_value=tc.return_value,
                        latency_ms=tc.latency_ms,
                        agent_id=tc.agent_id,
                        agent_name=tc.agent_name,
                        node_name=tc.node_name,
                        workflow_run_id=tc.workflow_run_id,
                        branch_id=tc.branch_id,
                        parent_event_id=tc.parent_event_id,
                    )

    def set_max_steps(self, agent: Any, max_steps: int) -> None:
        if hasattr(agent, "max_iter"):
            try:
                agent.max_iter = max_steps
            except Exception:
                pass
