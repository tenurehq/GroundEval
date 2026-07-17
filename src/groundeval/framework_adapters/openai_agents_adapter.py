from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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

logger = logging.getLogger("groundeval.adapters.openai_agents")


@dataclass
class OpenAIAgentsEntry:
    agent: Any
    input: Any = ""
    context: Any = None
    run_config: Any = None


def _load_openai_agents_entry(class_path: str) -> Any:
    module_path, attr_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    entry = getattr(module, attr_name)
    if isinstance(entry, type):
        return entry()
    if callable(entry) and not _looks_like_agent(entry):
        return entry()
    return entry


def _looks_like_agent(value: Any) -> bool:
    return bool(
        value is not None
        and hasattr(value, "name")
        and hasattr(value, "tools")
        and hasattr(value, "handoffs")
    )


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_name(agent: Any) -> str:
    return str(getattr(agent, "name", None) or agent.__class__.__name__)


def _agent_id(agent: Any) -> str:
    explicit = getattr(agent, "id", None) or getattr(agent, "agent_id", None)
    return str(explicit or f"openai-agents:{_agent_name(agent)}")


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", None) or tool.__class__.__name__)


def _tool_call_id(context: Any) -> str | None:
    value = getattr(context, "tool_call_id", None)
    return str(value) if value is not None else None


def _tool_arguments(context: Any) -> dict[str, Any]:
    value = getattr(context, "tool_arguments", None)
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        return parsed
    if parsed is None:
        return {}
    return {"raw": parsed}


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _timestamp_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def _latency_ms(started_at: str | None, ended_at: str | None) -> float:
    start = _timestamp_seconds(started_at)
    end = _timestamp_seconds(ended_at)
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start) * 1000)


def _span_payload(span: Any) -> dict[str, Any]:
    exported = None
    try:
        exported = span.export()
    except Exception:
        pass
    if isinstance(exported, dict):
        return _jsonish(exported)
    data = getattr(span, "span_data", None)
    data_export = None
    if data is not None and hasattr(data, "export"):
        try:
            data_export = data.export()
        except Exception:
            pass
    return {
        "id": getattr(span, "span_id", None),
        "trace_id": getattr(span, "trace_id", None),
        "parent_id": getattr(span, "parent_id", None),
        "started_at": getattr(span, "started_at", None),
        "ended_at": getattr(span, "ended_at", None),
        "span_data": _jsonish(data_export),
        "error": _jsonish(getattr(span, "error", None)),
    }


def _span_data(span: Any) -> tuple[str, dict[str, Any]]:
    data = getattr(span, "span_data", None)
    span_type = str(getattr(data, "type", "unknown") or "unknown")
    exported = None
    if data is not None and hasattr(data, "export"):
        try:
            exported = data.export()
        except Exception:
            pass
    return span_type, exported if isinstance(exported, dict) else {}


class _OpenAITraceProcessor:
    def __init__(self, collector: "_OpenAIAgentsCollector"):
        self.collector = collector
        self.active = True

    def on_trace_start(self, trace: Any) -> None:
        if self.active:
            self.collector.on_trace_start(trace)

    def on_trace_end(self, trace: Any) -> None:
        if self.active:
            self.collector.on_trace_end(trace)

    def on_span_start(self, span: Any) -> None:
        if self.active:
            self.collector.on_span_start(span)

    def on_span_end(self, span: Any) -> None:
        if self.active:
            self.collector.on_span_end(span)

    def shutdown(self) -> None:
        self.active = False

    def force_flush(self) -> None:
        return None


class _OpenAIAgentsHooks:
    def __init__(self, collector: "_OpenAIAgentsCollector"):
        self.collector = collector

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: str | None, input_items: list[Any]) -> None:
        self.collector.record_event(
            "openai_agents.llm_start",
            agent=agent,
            payload={"system_prompt": system_prompt, "input_items": _jsonish(input_items)},
        )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.collector.record_model_response(agent, response)

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        self.collector.record_agent_start(agent)

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        self.collector.record_agent_end(agent, output)

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        self.collector.record_handoff(from_agent, to_agent)

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.collector.record_tool_start(context, agent, tool)

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: object) -> None:
        self.collector.record_tool_end(context, agent, tool, result)


class _OpenAIAgentsCollector:
    def __init__(self, run_id: str, agent_class: str):
        self.run_id = run_id
        self.agent_class = agent_class
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.events: list[ObservedEvent] = []
        self.tool_calls: list[ObservedToolCall] = []
        self.agents: dict[str, ObservedAgent] = {}
        self.nodes: dict[str, ObservedWorkflowNode] = {}
        self.handoffs: list[ObservedHandoff] = []
        self.model_events: list[ObservedModelEvent] = []
        self.errors: list[ObservedError] = []
        self.final_output: Any = None
        self.trace_id: str | None = None
        self.workflow_name: str | None = None
        self._pending_tools: dict[str, dict[str, Any]] = {}
        self._hook_tool_keys: set[tuple[str, str]] = set()
        self.hooks = _OpenAIAgentsHooks(self)
        self.processor = _OpenAITraceProcessor(self)

    def install(self) -> None:
        try:
            from agents import add_trace_processor
        except Exception as exc:
            raise RuntimeError(
                "OpenAI Agents observe requires the openai-agents package."
            ) from exc
        add_trace_processor(self.processor)

    def deactivate(self) -> None:
        self.processor.active = False

    def record_event(
        self,
        event_type: str,
        agent: Any = None,
        payload: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        agent_name = _agent_name(agent) if agent is not None else None
        self.events.append(
            ObservedEvent(
                event_type=event_type,
                timestamp=timestamp or _now(),
                agent_name=agent_name,
                node_name=agent_name,
                workflow_run_id=self.trace_id or self.run_id,
                parent_event_id=parent_event_id,
                payload=payload or {},
            )
        )

    def ensure_agent(self, agent: Any) -> ObservedAgent:
        agent_id = _agent_id(agent)
        if agent_id not in self.agents:
            description = getattr(agent, "instructions", None)
            self.agents[agent_id] = ObservedAgent(
                agent_id=agent_id,
                agent_name=_agent_name(agent),
                agent_description=str(description) if isinstance(description, str) else None,
                role=_agent_name(agent),
            )
        return self.agents[agent_id]

    def record_agent_start(self, agent: Any) -> None:
        observed = self.ensure_agent(agent)
        timestamp = _now()
        node = self.nodes.get(observed.agent_id)
        if node is None:
            self.nodes[observed.agent_id] = ObservedWorkflowNode(
                node_id=observed.agent_id,
                node_type="openai_agents.agent",
                entered_at=timestamp,
                agent_name=observed.agent_name,
            )
        else:
            node.entered_at = node.entered_at or timestamp
        self.record_event("openai_agents.agent_start", agent=agent)

    def record_agent_end(self, agent: Any, output: Any) -> None:
        observed = self.ensure_agent(agent)
        timestamp = _now()
        node = self.nodes.get(observed.agent_id)
        if node is None:
            node = ObservedWorkflowNode(
                node_id=observed.agent_id,
                node_type="openai_agents.agent",
                entered_at=timestamp,
                agent_name=observed.agent_name,
            )
            self.nodes[observed.agent_id] = node
        node.exited_at = timestamp
        self.final_output = _parse_jsonish(output)
        self.record_event(
            "openai_agents.agent_end", agent=agent, payload={"output": _jsonish(output)}
        )

    def record_handoff(self, from_agent: Any, to_agent: Any) -> None:
        source = self.ensure_agent(from_agent)
        target = self.ensure_agent(to_agent)
        timestamp = _now()
        self.handoffs.append(
            ObservedHandoff(
                from_executor_id=source.agent_id,
                to_executor_id=target.agent_id,
                timestamp=timestamp,
                payload_type="openai_agents.handoff",
            )
        )
        self.record_event(
            "openai_agents.handoff",
            agent=from_agent,
            timestamp=timestamp,
            payload={"from_agent": source.agent_name, "to_agent": target.agent_name},
        )

    def record_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        observed = self.ensure_agent(agent)
        call_id = _tool_call_id(context) or uuid.uuid4().hex
        name = _tool_name(tool)
        arguments = _tool_arguments(context)
        self._pending_tools[call_id] = {
            "started": time.time(),
            "agent_id": observed.agent_id,
            "agent_name": observed.agent_name,
            "tool_name": name,
            "arguments": arguments,
        }
        self.record_event(
            "openai_agents.tool_start",
            agent=agent,
            parent_event_id=call_id,
            payload={"tool_name": name, "arguments": arguments, "tool_call_id": call_id},
        )

    def record_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        observed = self.ensure_agent(agent)
        call_id = _tool_call_id(context)
        pending = self._pending_tools.pop(call_id, None) if call_id else None
        if pending is None:
            for key, value in list(self._pending_tools.items()):
                if value["tool_name"] == _tool_name(tool) and value["agent_id"] == observed.agent_id:
                    call_id = key
                    pending = self._pending_tools.pop(key)
                    break
        arguments = pending["arguments"] if pending else _tool_arguments(context)
        latency = (time.time() - pending["started"]) * 1000 if pending else 0.0
        parsed_result = _parse_jsonish(result)
        self.tool_calls.append(
            ObservedToolCall(
                tool_name=_tool_name(tool),
                arguments=arguments,
                return_value=parsed_result,
                latency_ms=latency,
                agent_id=observed.agent_id,
                agent_name=observed.agent_name,
                node_name=observed.agent_name,
                workflow_run_id=self.trace_id or self.run_id,
                parent_event_id=call_id,
            )
        )
        observed.tool_call_count += 1
        self._hook_tool_keys.add((_tool_name(tool), json.dumps(arguments, sort_keys=True, default=str)))
        self.record_event(
            "openai_agents.tool_end",
            agent=agent,
            parent_event_id=call_id,
            payload={"tool_name": _tool_name(tool), "return_value": _jsonish(parsed_result)},
        )

    def record_model_response(self, agent: Any, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            usage = getattr(response, "usage_data", None)
        input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "requests", None)
        output_tokens = getattr(usage, "output_tokens", None)
        model = getattr(response, "model", None) or getattr(response, "model_name", None)
        self.model_events.append(
            ObservedModelEvent(
                event_type="model.call.completed",
                timestamp=_now(),
                model_name=str(model) if model else None,
                provider_name="openai",
                input_tokens=_coerce_int(input_tokens),
                output_tokens=_coerce_int(output_tokens),
            )
        )
        self.record_event(
            "openai_agents.llm_end", agent=agent, payload={"response": _jsonish(response)}
        )

    def on_trace_start(self, trace: Any) -> None:
        self.trace_id = str(getattr(trace, "trace_id", None) or self.run_id)
        self.workflow_name = str(getattr(trace, "name", None) or "OpenAI Agents workflow")
        self.record_event(
            "openai_agents.trace_start",
            payload={"trace_id": self.trace_id, "workflow_name": self.workflow_name},
        )

    def on_trace_end(self, trace: Any) -> None:
        self.record_event(
            "openai_agents.trace_end",
            payload={"trace_id": getattr(trace, "trace_id", None)},
        )

    def on_span_start(self, span: Any) -> None:
        span_type, data = _span_data(span)
        self.record_event(
            f"openai_agents.span_start.{span_type}",
            payload={"span": _span_payload(span), "span_data": _jsonish(data)},
            parent_event_id=getattr(span, "parent_id", None),
            timestamp=getattr(span, "started_at", None),
        )

    def on_span_end(self, span: Any) -> None:
        span_type, data = _span_data(span)
        self.record_event(
            f"openai_agents.span_end.{span_type}",
            payload={"span": _span_payload(span), "span_data": _jsonish(data)},
            parent_event_id=getattr(span, "parent_id", None),
            timestamp=getattr(span, "ended_at", None),
        )
        error = getattr(span, "error", None)
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            self.errors.append(
                ObservedError(
                    error_type=f"OpenAIAgents{span_type.title()}SpanError",
                    message=str(message),
                    timestamp=getattr(span, "ended_at", None),
                    traceback=json.dumps(_jsonish(error), default=str),
                )
            )
        if span_type == "agent":
            name = str(data.get("name") or "unknown")
            agent_id = f"openai-agents:{name}"
            if agent_id not in self.agents:
                self.agents[agent_id] = ObservedAgent(agent_id=agent_id, agent_name=name, role=name)
        elif span_type == "handoff":
            source = data.get("from_agent")
            target = data.get("to_agent")
            if source and target:
                source_id = f"openai-agents:{source}"
                target_id = f"openai-agents:{target}"
                duplicate = any(
                    h.from_executor_id == source_id and h.to_executor_id == target_id
                    for h in self.handoffs
                )
                if not duplicate:
                    self.handoffs.append(
                        ObservedHandoff(
                            from_executor_id=source_id,
                            to_executor_id=target_id,
                            timestamp=getattr(span, "ended_at", None),
                            payload_type="openai_agents.handoff_span",
                        )
                    )
        elif span_type == "function":
            name = str(data.get("name") or "function")
            parsed_args = _parse_jsonish(data.get("input"))
            arguments = parsed_args if isinstance(parsed_args, dict) else {"raw": parsed_args}
            key = (name, json.dumps(arguments, sort_keys=True, default=str))
            if key not in self._hook_tool_keys:
                self.tool_calls.append(
                    ObservedToolCall(
                        tool_name=name,
                        arguments=arguments,
                        return_value=_parse_jsonish(data.get("output")),
                        latency_ms=_latency_ms(
                            getattr(span, "started_at", None), getattr(span, "ended_at", None)
                        ),
                        workflow_run_id=self.trace_id or self.run_id,
                        parent_event_id=getattr(span, "parent_id", None),
                    )
                )
        elif span_type == "generation":
            usage = data.get("usage") or {}
            self.model_events.append(
                ObservedModelEvent(
                    event_type="model.call.completed",
                    timestamp=getattr(span, "ended_at", None),
                    model_name=str(data.get("model")) if data.get("model") else None,
                    provider_name="openai",
                    input_tokens=_coerce_int(
                        usage.get("input_tokens") or usage.get("prompt_tokens")
                    ) if isinstance(usage, dict) else None,
                    output_tokens=_coerce_int(
                        usage.get("output_tokens") or usage.get("completion_tokens")
                    ) if isinstance(usage, dict) else None,
                )
            )

    def record_exception(self, exc: BaseException) -> None:
        self.errors.append(
            ObservedError(
                error_type=exc.__class__.__name__,
                message=str(exc),
                timestamp=_now(),
                traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
        )

    def to_rich_run(self) -> RichObservedRun:
        workflow = ObservedWorkflow(
            workflow_id=self.trace_id or self.run_id,
            workflow_name=self.workflow_name or "OpenAI Agents workflow",
            node_count=len(self.nodes),
            nodes=list(self.nodes.values()),
            handoff_count=len(self.handoffs),
            handoffs=self.handoffs,
        )
        total_latency = 0.0
        if self.started_at is not None and self.completed_at is not None:
            total_latency = (self.completed_at - self.started_at) * 1000
        return RichObservedRun(
            run_id=self.trace_id or self.run_id,
            framework="openai_agents",
            agent_class=self.agent_class,
            started_at=str(self.started_at) if self.started_at is not None else None,
            completed_at=str(self.completed_at) if self.completed_at is not None else None,
            total_latency_ms=total_latency,
            tool_calls=self.tool_calls,
            events=self.events,
            agents=list(self.agents.values()),
            workflow=workflow,
            model_events=self.model_events,
            final_output=self.final_output,
            errors=self.errors,
            capabilities={
                "native_tracing": True,
                "lifecycle_hooks": True,
                "tool_calls": bool(self.tool_calls),
                "agent_turns": bool(self.agents),
                "workflow_nodes": bool(self.nodes),
                "handoffs": bool(self.handoffs),
                "approvals": False,
                "checkpoints": False,
                "context_injection": False,
                "model_calls": bool(self.model_events),
            },
        )


def _entry_parts(entry: Any) -> tuple[Any, Any, Any, Any]:
    if isinstance(entry, OpenAIAgentsEntry):
        return entry.agent, entry.input, entry.context, entry.run_config
    if isinstance(entry, tuple) and len(entry) == 2 and _looks_like_agent(entry[0]):
        return entry[0], entry[1], None, None
    if isinstance(entry, dict) and _looks_like_agent(entry.get("agent")):
        return (
            entry["agent"],
            entry.get("input", ""),
            entry.get("context"),
            entry.get("run_config"),
        )
    if _looks_like_agent(entry):
        input_value = getattr(entry, "groundeval_input", None)
        if input_value is None:
            input_value = os.environ.get("GROUNDEVAL_AGENT_INPUT", "")
        return entry, input_value, None, None
    nested = getattr(entry, "agent", None)
    if _looks_like_agent(nested):
        return (
            nested,
            getattr(entry, "input", getattr(entry, "groundeval_input", "")),
            getattr(entry, "context", None),
            getattr(entry, "run_config", None),
        )
    return None, None, None, None


def _run_entry(entry: Any, collector: _OpenAIAgentsCollector, max_steps: int) -> Any:
    custom = getattr(entry, "run_groundeval", None)
    if callable(custom):
        result = custom(hooks=collector.hooks, max_turns=max_steps)
        if inspect.isawaitable(result):
            import asyncio

            return asyncio.run(result)
        return result

    agent, input_value, context, run_config = _entry_parts(entry)
    if agent is not None:
        try:
            from agents import Runner
        except Exception as exc:
            raise RuntimeError(
                "OpenAI Agents observe requires the openai-agents package."
            ) from exc
        kwargs: dict[str, Any] = {
            "max_turns": max_steps,
            "hooks": collector.hooks,
        }
        if context is not None:
            kwargs["context"] = context
        if run_config is not None:
            kwargs["run_config"] = run_config
        return Runner.run_sync(agent, input_value, **kwargs)

    run = getattr(entry, "run", None)
    if callable(run):
        result = run()
        if inspect.isawaitable(result):
            import asyncio

            return asyncio.run(result)
        return result
    if callable(entry):
        result = entry()
        if inspect.isawaitable(result):
            import asyncio

            return asyncio.run(result)
        return result
    raise TypeError(
        "OpenAI Agents adapter expected an Agent, OpenAIAgentsEntry, (agent, input) tuple, or an entry object with run_groundeval()."
    )


def _final_output(result: Any) -> Any:
    value = getattr(result, "final_output", result)
    return _parse_jsonish(value)


class _InstrumentedOpenAIAgentsEntry:
    def __init__(self, entry: Any, collector: _OpenAIAgentsCollector, recording: RecordingRuntime):
        self.entry = entry
        self._groundeval_openai_agents_collector = collector
        self._groundeval_recording = recording
        self._groundeval_max_steps = 10


class OpenAIAgentsObserver(AgentObserver):
    def load_agent(self, class_path: str) -> Any:
        return _load_openai_agents_entry(class_path)

    def instrument_agent(self, agent: Any, recording: RecordingRuntime) -> Any:
        run_id = f"openai_agents_observed_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        collector = _OpenAIAgentsCollector(
            run_id=run_id,
            agent_class=f"{agent.__class__.__module__}.{agent.__class__.__name__}",
        )
        collector.install()
        return _InstrumentedOpenAIAgentsEntry(agent, collector, recording)

    def execute_agent(self, agent: Any) -> Any:
        collector = agent._groundeval_openai_agents_collector
        recording = agent._groundeval_recording
        max_steps = int(getattr(agent, "_groundeval_max_steps", 10))
        result = None
        collector.started_at = time.time()
        try:
            result = _run_entry(agent.entry, collector, max_steps)
            collector.final_output = _final_output(result)
            return result
        except Exception as exc:
            collector.record_exception(exc)
            raise
        finally:
            collector.completed_at = time.time()
            collector.deactivate()
            rich_run = collector.to_rich_run()
            agent._groundeval_framework_observed_run = rich_run
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
        if max_steps <= 0:
            raise ValueError("OpenAI Agents max_steps must be greater than zero.")
        agent._groundeval_max_steps = int(max_steps)


def generate_openai_agents_report(run: RichObservedRun) -> str:
    workflow = run.workflow
    lines = [
        "# GroundEval OpenAI Agents Observation Report",
        "",
        "## Summary",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Entry class: `{run.agent_class}`",
        f"- Total latency: {run.total_latency_ms:.0f}ms",
        f"- Agents observed: {len(run.agents)}",
        f"- Tool calls recorded: {len(run.tool_calls)}",
        f"- Handoffs recorded: {workflow.handoff_count if workflow else 0}",
        f"- Model calls recorded: {len(run.model_events)}",
        "",
        "## Capabilities",
        "",
        "| Capability | Observed |",
        "|---|---|",
    ]
    for key, value in sorted(run.capabilities.items()):
        lines.append(f"| {key} | {'Yes' if value else 'No'} |")
    lines.extend(["", "## Agent Inventory", "", "| Agent | ID | Tool Calls |", "|---|---|---:|"])
    for agent in run.agents:
        lines.append(f"| {agent.agent_name or ''} | `{agent.agent_id}` | {agent.tool_call_count} |")
    lines.extend(["", "## Tool Calls", "", "| Tool | Agent | Arguments | Return | Latency |", "|---|---|---|---|---:|"])
    for call in run.tool_calls:
        args = json.dumps(call.arguments, default=str)[:300]
        result = json.dumps(call.return_value, default=str)[:300]
        lines.append(f"| `{call.tool_name}` | {call.agent_name or ''} | `{args}` | `{result}` | {call.latency_ms:.0f}ms |")
    lines.extend(["", "## Handoffs", "", "| From | To | Timestamp |", "|---|---|---|"])
    for handoff in workflow.handoffs if workflow else []:
        lines.append(f"| `{handoff.from_executor_id}` | `{handoff.to_executor_id}` | {handoff.timestamp or ''} |")
    lines.extend(["", "## Errors", "", "| Type | Message | Timestamp |", "|---|---|---|"])
    for error in run.errors:
        lines.append(f"| {error.error_type} | {error.message} | {error.timestamp or ''} |")
    lines.extend(["", "## Final Output", "", "```json", json.dumps(run.final_output, indent=2, default=str), "```", ""])
    return "\n".join(lines)
