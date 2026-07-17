from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import time
from typing import Any

from ..observe import AgentObserver, ObservedToolCall, RecordingRuntime
from .framework_observation import (
    ObservedAgent,
    ObservedError,
    ObservedEvent,
    ObservedHandoff,
    ObservedModelEvent,
    ObservedRun as MafObservedRun,
    ObservedWorkflow,
    ObservedWorkflowNode,
)

logger = logging.getLogger("groundeval.adapters.maf")

_DEFAULT_MAF_SOURCE = "Experimental.Microsoft.Agents.AI"


def _load_maf_agent(agent_class_path: str) -> Any:
    module_path, attr_name = agent_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    agent_obj = getattr(module, attr_name)

    if callable(agent_obj) and not isinstance(agent_obj, type):
        return agent_obj()

    if isinstance(agent_obj, type):
        return agent_obj()

    return agent_obj


class _GroundEvalSpanExporter:
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: list[Any]) -> Any:
        self.spans.extend(spans)
        try:
            from opentelemetry.sdk.trace.export import SpanExportResult

            return SpanExportResult.SUCCESS
        except Exception:
            return 0

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _install_in_memory_otel_exporter() -> _GroundEvalSpanExporter:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except Exception as exc:
        raise RuntimeError(
            "MAF observe requires opentelemetry-api and opentelemetry-sdk. "
            "Microsoft Agent Framework installs these for observability."
        ) from exc

    exporter = _GroundEvalSpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = trace.get_tracer_provider()
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)
        return exporter

    provider = TracerProvider(
        resource=Resource.create({"service.name": "groundeval-maf-observe"})
    )
    provider.add_span_processor(processor)
    try:
        trace.set_tracer_provider(provider)
    except Exception:
        logger.debug(
            "Unable to replace global OpenTelemetry tracer provider",
            exc_info=True,
        )
    return exporter


def _enable_maf_instrumentation() -> None:
    os.environ.setdefault("ENABLE_INSTRUMENTATION", "true")
    os.environ.setdefault("ENABLE_SENSITIVE_DATA", "true")

    try:
        from agent_framework.observability import enable_instrumentation
    except Exception as exc:
        raise RuntimeError(
            "MAF observe requires Microsoft Agent Framework. Install the "
            "agent_framework package used by your project."
        ) from exc

    try:
        enable_instrumentation(enable_sensitive_data=True)
        return
    except TypeError:
        pass

    try:
        enable_instrumentation(True)
        return
    except TypeError:
        pass

    enable_instrumentation()


async def _await_if_needed(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


async def _collect_async_iter(value: Any) -> Any:
    if hasattr(value, "__aiter__"):
        chunks = []
        async for update in value:
            text = getattr(update, "text", None)
            if text is not None:
                chunks.append(str(text))
            else:
                chunks.append(str(update))
        return "".join(chunks)
    return value


def _run_maf_entity(entity: Any) -> Any:
    async def _run_async() -> Any:
        if hasattr(entity, "run") and callable(entity.run):
            try:
                result = entity.run()
            except TypeError:
                result = entity.run("")
            result = await _await_if_needed(result)
            result = await _collect_async_iter(result)
            return result

        if callable(entity):
            result = entity()
            result = await _await_if_needed(result)
            result = await _collect_async_iter(result)
            return result

        raise TypeError(
            "MAF adapter expected an object with .run(...) or a callable entry point."
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            raise RuntimeError(
                "MAF observe cannot run inside an already-running asyncio loop. "
                "Call GroundEval from a normal CLI process."
            )
    except RuntimeError as exc:
        if "already-running asyncio loop" in str(exc):
            raise
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_run_async())
        finally:
            loop.close()

    return asyncio.run(_run_async())


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
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


def _span_name(span: Any) -> str:
    return getattr(span, "name", "") or "unknown"


def _span_attrs(span: Any) -> dict[str, Any]:
    attrs = getattr(span, "attributes", None) or {}
    return {str(k): _jsonish(v) for k, v in dict(attrs).items()}


def _span_time(value: Any) -> str | None:
    if not value:
        return None
    try:
        return str(float(value) / 1_000_000_000.0)
    except Exception:
        return str(value)


def _span_latency_ms(span: Any) -> float:
    start = getattr(span, "start_time", None)
    end = getattr(span, "end_time", None)
    try:
        if start is not None and end is not None:
            return max(0.0, (float(end) - float(start)) / 1_000_000.0)
    except Exception:
        pass
    return 0.0


def _first(attrs: dict[str, Any], keys: list[str], contains: str | None = None) -> Any:
    for key in keys:
        if key in attrs:
            return attrs[key]
    if contains:
        needle = contains.lower()
        for key, value in attrs.items():
            if needle in key.lower():
                return value
    return None


def _extract_tool_name(name: str, attrs: dict[str, Any]) -> str:
    explicit = _first(
        attrs,
        [
            "gen_ai.tool.name",
            "gen_ai.function.name",
            "function.name",
            "tool.name",
            "name",
        ],
    )
    if explicit:
        return str(explicit)
    for prefix in ("execute_tool ", "tool ", "function "):
        if name.startswith(prefix):
            return name[len(prefix) :].strip() or name
    return name


def _extract_tool_args(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = _first(
        attrs,
        [
            "gen_ai.tool.call.arguments",
            "gen_ai.function.arguments",
            "function.arguments",
            "tool.arguments",
            "arguments",
            "args",
            "executor.input",
        ],
        contains="arguments",
    )
    parsed = _parse_jsonish(raw)
    if isinstance(parsed, dict):
        return parsed
    if parsed is None:
        return {}
    return {"raw": parsed}


def _extract_tool_result(attrs: dict[str, Any]) -> Any:
    raw = _first(
        attrs,
        [
            "gen_ai.tool.call.result",
            "gen_ai.function.result",
            "function.result",
            "tool.result",
            "result",
            "return_value",
            "executor.output",
        ],
        contains="result",
    )
    return _parse_jsonish(raw)


def _extract_final_output(raw_result: Any, spans: list[Any]) -> Any:
    if raw_result is not None:
        return _parse_jsonish(raw_result)

    for span in reversed(spans):
        attrs = _span_attrs(span)
        candidate = _first(
            attrs,
            [
                "gen_ai.response.text",
                "gen_ai.response.content",
                "gen_ai.output",
                "output",
                "response",
                "executor.output",
            ],
            contains="response",
        )
        if candidate is not None:
            return _parse_jsonish(candidate)

    return None


def _is_tool_span(name: str, attrs: dict[str, Any]) -> bool:
    operation = str(attrs.get("gen_ai.operation.name", "")).lower()
    lower = name.lower()
    return (
        lower.startswith("execute_tool")
        or "execute_tool" in lower
        or operation in {"execute_tool", "tool_call", "function_call"}
        or "function.invocation" in lower
    )


def _is_agent_span(name: str, attrs: dict[str, Any]) -> bool:
    operation = str(attrs.get("gen_ai.operation.name", "")).lower()
    lower = name.lower()
    return lower.startswith("invoke_agent") or operation == "invoke_agent"


def _is_chat_span(name: str, attrs: dict[str, Any]) -> bool:
    operation = str(attrs.get("gen_ai.operation.name", "")).lower()
    lower = name.lower()
    return lower.startswith("chat ") or operation in {"chat", "generate_content"}


def _is_workflow_build_span(name: str) -> bool:
    return name.lower() == "workflow.build"


def _is_workflow_run_span(name: str) -> bool:
    lower = name.lower()
    return lower in {"workflow.run", "workflow.session", "workflow_invoke"}


def _is_executor_span(name: str) -> bool:
    return name.lower().startswith("executor.process")


def _is_edge_group_span(name: str) -> bool:
    return name.lower().startswith("edge_group.process")


def _is_message_send_span(name: str) -> bool:
    return name.lower() == "message.send"


def _safe_span_ids(span: Any) -> tuple[str | None, str | None, str | None]:
    ctx = None
    try:
        ctx = span.get_span_context()
    except Exception:
        ctx = getattr(span, "context", None)

    trace_id = None
    span_id = None
    if ctx is not None:
        trace_id = getattr(ctx, "trace_id", None)
        span_id = getattr(ctx, "span_id", None)
        trace_id = hex(trace_id) if isinstance(trace_id, int) else str(trace_id)
        span_id = hex(span_id) if isinstance(span_id, int) else str(span_id)

    parent = getattr(span, "parent", None)
    parent_id = getattr(parent, "span_id", None) if parent is not None else None
    parent_id = (
        hex(parent_id)
        if isinstance(parent_id, int)
        else str(parent_id)
        if parent_id
        else None
    )
    return trace_id, span_id, parent_id


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _tool_agent_identity(
    attrs: dict[str, Any],
    agents: dict[str, ObservedAgent],
) -> tuple[str | None, str | None]:
    agent_id = attrs.get("gen_ai.agent.id") or attrs.get("agent.id")
    agent_name = attrs.get("gen_ai.agent.name") or attrs.get("agent.name")

    if agent_id is not None:
        agent_id = str(agent_id)
    if agent_name is not None:
        agent_name = str(agent_name)

    if agent_id and not agent_name and agent_id in agents:
        agent_name = agents[agent_id].agent_name

    if agent_name and not agent_id:
        for known_id, known_agent in agents.items():
            if known_agent.agent_name == agent_name:
                agent_id = known_id
                break

    return agent_id, agent_name


def maf_spans_to_observed_run(
    *,
    spans: list[Any],
    raw_result: Any,
    agent_class: str,
    run_id: str,
    started_at: float,
    completed_at: float,
) -> MafObservedRun:
    tool_calls: list[ObservedToolCall] = []
    events: list[ObservedEvent] = []
    agents: dict[str, ObservedAgent] = {}
    workflow_nodes: dict[str, ObservedWorkflowNode] = {}
    handoffs: list[ObservedHandoff] = []
    model_events: list[ObservedModelEvent] = []
    errors: list[ObservedError] = []

    workflow_id = None
    workflow_name = None
    workflow_description = None

    for span in spans:
        name = _span_name(span)
        attrs = _span_attrs(span)
        trace_id, span_id, parent_id = _safe_span_ids(span)
        started = _span_time(getattr(span, "start_time", None))
        ended = _span_time(getattr(span, "end_time", None))

        event_payload = {
            "span_name": name,
            "trace_id": trace_id,
            "span_id": span_id,
            "attributes": attrs,
        }
        events.append(
            ObservedEvent(
                event_type="otel.span",
                timestamp=started,
                workflow_run_id=workflow_id or run_id,
                parent_event_id=parent_id,
                payload=event_payload,
            )
        )

        for span_event in getattr(span, "events", []) or []:
            span_event_attrs = {
                str(k): _jsonish(v)
                for k, v in dict(getattr(span_event, "attributes", {}) or {}).items()
            }
            events.append(
                ObservedEvent(
                    event_type=f"otel.event.{getattr(span_event, 'name', 'unknown')}",
                    timestamp=_span_time(getattr(span_event, "timestamp", None)),
                    workflow_run_id=workflow_id or run_id,
                    parent_event_id=span_id,
                    payload=span_event_attrs,
                )
            )

        status = getattr(span, "status", None)
        status_code = str(getattr(status, "status_code", "")).upper()
        if "ERROR" in status_code:
            errors.append(
                ObservedError(
                    error_type="OpenTelemetrySpanError",
                    message=str(getattr(status, "description", "") or name),
                    timestamp=ended or started,
                )
            )

        if _is_agent_span(name, attrs):
            agent_id = str(
                attrs.get("gen_ai.agent.id")
                or attrs.get("agent.id")
                or attrs.get("gen_ai.agent.name")
                or name.replace("invoke_agent", "").strip()
                or "unknown"
            )
            agent_name = str(
                attrs.get("gen_ai.agent.name") or attrs.get("agent.name") or agent_id
            )
            agents[agent_id] = ObservedAgent(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_description=attrs.get("gen_ai.request.instructions"),
            )

        if _is_tool_span(name, attrs):
            tool_name = _extract_tool_name(name, attrs)
            agent_id, agent_name = _tool_agent_identity(attrs, agents)
            node_name = str(
                attrs.get("executor.id") or attrs.get("edge_group.id") or name
            )
            tool_call = ObservedToolCall(
                tool_name=tool_name,
                arguments=_extract_tool_args(attrs),
                return_value=_extract_tool_result(attrs),
                latency_ms=_span_latency_ms(span),
                agent_id=agent_id,
                agent_name=agent_name,
                node_name=node_name,
                workflow_run_id=workflow_id or run_id,
                branch_id=None,
                parent_event_id=parent_id,
            )
            tool_calls.append(tool_call)
            if agent_id and agent_id in agents:
                agents[agent_id].tool_call_count += 1

        if _is_chat_span(name, attrs):
            model_events.append(
                ObservedModelEvent(
                    event_type="model.call.completed",
                    timestamp=ended or started,
                    model_name=str(
                        attrs.get("gen_ai.request.model")
                        or attrs.get("gen_ai.response.model")
                        or name.replace("chat", "").strip()
                        or "unknown"
                    ),
                    provider_name=attrs.get("gen_ai.system"),
                    input_tokens=_coerce_int(
                        attrs.get("gen_ai.usage.input_tokens")
                        or attrs.get("gen_ai.usage.prompt_tokens")
                    ),
                    output_tokens=_coerce_int(
                        attrs.get("gen_ai.usage.output_tokens")
                        or attrs.get("gen_ai.usage.completion_tokens")
                    ),
                    finish_reason=attrs.get("gen_ai.response.finish_reason"),
                    tool_schemas_count=_coerce_int(
                        attrs.get("gen_ai.request.tool_schemas.count")
                    )
                    or 0,
                )
            )

        if _is_workflow_build_span(name) or _is_workflow_run_span(name):
            workflow_id = str(attrs.get("workflow.id") or workflow_id or run_id)
            workflow_name = attrs.get("workflow.name") or workflow_name
            workflow_description = (
                attrs.get("workflow.description") or workflow_description
            )

        if (
            _is_executor_span(name)
            or _is_edge_group_span(name)
            or _is_workflow_run_span(name)
        ):
            node_id = str(
                attrs.get("executor.id")
                or attrs.get("edge_group.id")
                or attrs.get("workflow.id")
                or span_id
                or name
            )
            workflow_nodes[node_id] = ObservedWorkflowNode(
                node_id=node_id,
                node_type=name,
                entered_at=started,
                exited_at=ended,
                agent_name=attrs.get("gen_ai.agent.name"),
            )

        if _is_message_send_span(name):
            source_id = attrs.get("message.source_id")
            target_id = attrs.get("message.target_id") or attrs.get(
                "message.destination_executor_id"
            )
            if source_id and target_id:
                handoffs.append(
                    ObservedHandoff(
                        from_executor_id=str(source_id),
                        to_executor_id=str(target_id),
                        timestamp=started,
                        payload_type=attrs.get("message.type"),
                    )
                )

            events.append(
                ObservedEvent(
                    event_type="workflow.message.send",
                    timestamp=started,
                    workflow_run_id=workflow_id or run_id,
                    parent_event_id=parent_id,
                    payload={
                        "message.type": attrs.get("message.type"),
                        "message.source_id": attrs.get("message.source_id"),
                        "message.target_id": attrs.get("message.target_id"),
                        "message.destination_executor_id": attrs.get(
                            "message.destination_executor_id"
                        ),
                        "message.content": attrs.get("message.content"),
                    },
                )
            )

    final_output = _extract_final_output(raw_result, spans)

    workflow = None
    if (
        workflow_nodes
        or workflow_id
        or workflow_name
        or workflow_description
        or handoffs
    ):
        workflow = ObservedWorkflow(
            workflow_id=workflow_id or run_id,
            workflow_name=workflow_name or "MAF workflow trace",
            workflow_description=workflow_description,
            node_count=len(workflow_nodes),
            nodes=list(workflow_nodes.values()),
            handoff_count=len(handoffs),
            handoffs=handoffs,
        )

    capabilities = {
        "otel_spans": bool(spans),
        "tool_calls": bool(tool_calls),
        "agent_turns": bool(agents),
        "workflow_nodes": bool(workflow_nodes),
        "handoffs": bool(handoffs),
        "approvals": False,
        "checkpoints": False,
        "context_injection": False,
        "model_calls": bool(model_events),
        "message_send_spans": any(
            _is_message_send_span(_span_name(span)) for span in spans
        ),
        "workflow_build_spans": any(
            _is_workflow_build_span(_span_name(span)) for span in spans
        ),
        "workflow_run_spans": any(
            _is_workflow_run_span(_span_name(span)) for span in spans
        ),
    }

    return MafObservedRun(
        run_id=run_id,
        framework="maf",
        agent_class=agent_class,
        started_at=str(started_at),
        completed_at=str(completed_at),
        total_latency_ms=(completed_at - started_at) * 1000,
        tool_calls=tool_calls,
        events=events,
        agents=list(agents.values()),
        workflow=workflow,
        model_events=model_events,
        final_output=final_output,
        errors=errors,
        capabilities=capabilities,
    )


class MafObserver(AgentObserver):
    def load_agent(self, class_path: str) -> Any:
        return _load_maf_agent(class_path)

    def instrument_agent(
        self,
        agent: Any,
        recording: RecordingRuntime,
    ) -> Any:
        agent._groundeval_recording = recording
        return agent

    def execute_agent(self, agent: Any) -> Any:
        recording: RecordingRuntime | None = getattr(
            agent, "_groundeval_recording", None
        )

        exporter = _install_in_memory_otel_exporter()
        _enable_maf_instrumentation()

        agent_class = f"{agent.__class__.__module__}.{agent.__class__.__name__}"
        run_id = f"maf_observed_{agent_class.replace('.', '_')}_{int(time.time())}"
        started_at = time.time()
        raw_result = None
        try:
            raw_result = _run_maf_entity(agent)
            return raw_result
        finally:
            completed_at = time.time()
            spans = list(exporter.spans)
            maf_run = maf_spans_to_observed_run(
                spans=spans,
                raw_result=raw_result,
                agent_class=agent_class,
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
            )
            agent._groundeval_framework_observed_run = maf_run
            if recording is not None:
                for tc in maf_run.tool_calls:
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
            raise ValueError("MAF max_steps must be greater than zero.")

        configured = False
        targets = [
            agent,
            getattr(agent, "options", None),
            getattr(agent, "config", None),
            getattr(agent, "settings", None),
        ]

        for target in targets:
            if target is None:
                continue

            if isinstance(target, dict):
                for key in ("max_steps", "max_iterations", "max_turns"):
                    if key in target:
                        target[key] = int(max_steps)
                        configured = True
                continue

            for attribute in ("max_steps", "max_iterations", "max_turns"):
                if hasattr(target, attribute):
                    try:
                        setattr(target, attribute, int(max_steps))
                        configured = True
                    except Exception:
                        pass

        agent._groundeval_max_steps = int(max_steps)

        if not configured:
            logger.warning(
                "The MAF entry object exposes no recognized native step-limit "
                "setting. max_steps cannot be enforced portably for this object."
            )


def generate_maf_report(maf_run: MafObservedRun) -> str:
    lines = [
        "# GroundEval MAF Observation Report",
        "",
        f"Run ID: `{maf_run.run_id}`",
        f"Framework: {maf_run.framework}",
        f"Entry class: {maf_run.agent_class}",
        f"Total latency: {maf_run.total_latency_ms:.0f}ms",
        f"Tool calls recorded: {len(maf_run.tool_calls)}",
        "",
        "## Capabilities Observed",
        "",
    ]

    for cap, observed in sorted(maf_run.capabilities.items()):
        lines.append(f"- {cap}: {'Yes' if observed else 'No'}")
    lines.append("")

    if maf_run.agents:
        lines.extend([
            "## Agent Inventory",
            "",
            "| Agent | ID | Tool Calls |",
            "|---|---|---:|",
        ])
        for a in maf_run.agents:
            lines.append(
                f"| {a.agent_name or a.agent_id} | `{a.agent_id}` | {a.tool_call_count} |"
            )
        lines.append("")

    if maf_run.workflow:
        lines.extend([
            "## Workflow Summary",
            "",
            f"- Workflow ID: {maf_run.workflow.workflow_id}",
            f"- Workflow Name: {maf_run.workflow.workflow_name or ''}",
            f"- Nodes: {maf_run.workflow.node_count}",
            f"- Handoffs: {maf_run.workflow.handoff_count}",
            f"- Branches: {maf_run.workflow.branch_count}",
            "",
        ])
        if maf_run.workflow.nodes:
            lines.extend([
                "| Node | Type | Agent | Entered | Exited |",
                "|---|---|---|---|---|",
            ])
            for n in maf_run.workflow.nodes:
                lines.append(
                    f"| `{n.node_id}` | {n.node_type} | {n.agent_name or ''} | {n.entered_at or ''} | {n.exited_at or ''} |"
                )
            lines.append("")
        if maf_run.workflow.handoffs:
            lines.extend([
                "| From | To | Timestamp | Payload Type |",
                "|---|---|---|---|",
            ])
            for h in maf_run.workflow.handoffs:
                lines.append(
                    f"| `{h.from_executor_id}` | `{h.to_executor_id}` | {h.timestamp or ''} | {h.payload_type or ''} |"
                )
            lines.append("")

    if maf_run.model_events:
        lines.extend([
            "## Model Calls",
            "",
            "| Model | Provider | Input Tokens | Output Tokens | Finish Reason |",
            "|---|---|---:|---:|---|",
        ])
        for m in maf_run.model_events:
            lines.append(
                f"| {m.model_name or ''} | {m.provider_name or ''} | "
                f"{m.input_tokens if m.input_tokens is not None else ''} | "
                f"{m.output_tokens if m.output_tokens is not None else ''} | "
                f"{m.finish_reason or ''} |"
            )
        lines.append("")

    if maf_run.tool_calls:
        lines.extend(["## Tool Calls", ""])
        for i, tc in enumerate(maf_run.tool_calls, start=1):
            lines.extend([
                f"### {i}. {tc.tool_name}",
                f"- Latency: {tc.latency_ms:.0f}ms",
                f"- Agent: {tc.agent_name or ''} ({tc.agent_id or 'unknown'})",
                f"- Node: {tc.node_name or ''}",
                f"- Arguments: `{json.dumps(tc.arguments, default=str)}`",
                f"- Return: `{json.dumps(tc.return_value, default=str)[:1000]}`",
                "",
            ])

    if maf_run.errors:
        lines.extend([
            "## Errors",
            "",
            "| Type | Message | Timestamp |",
            "|---|---|---|",
        ])
        for e in maf_run.errors:
            lines.append(f"| {e.error_type} | {e.message} | {e.timestamp or ''} |")
        lines.append("")

    if maf_run.events:
        lines.extend([
            "## Event Timeline",
            "",
            "| Event | Timestamp | Details |",
            "|---|---|---|",
        ])
        for event in maf_run.events[:200]:
            details = json.dumps(event.payload, default=str)[:500]
            lines.append(
                f"| {event.event_type} | {event.timestamp or ''} | `{details}` |"
            )
        if len(maf_run.events) > 200:
            lines.append(
                f"| ... | ... | {len(maf_run.events) - 200} additional events omitted |"
            )
        lines.append("")

    lines.extend([
        "## Final Output",
        "",
        "```json",
        json.dumps(maf_run.final_output, indent=2, default=str),
        "```",
        "",
    ])
    return "\n".join(lines)
