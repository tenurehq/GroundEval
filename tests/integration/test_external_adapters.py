import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from groundeval.observe import (
    DraftGenerator,
    observe_agent,
    score_observed_run,
    write_draft_output,
)
from groundeval.run import _write_basic_observation_outputs, cmd_observe


class FakeSpanContext:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id


class FakeParentContext:
    def __init__(self, span_id):
        self.span_id = span_id


class FakeStatus:
    def __init__(self, status_code="OK", description=""):
        self.status_code = status_code
        self.description = description


class FakeSpanEvent:
    def __init__(self, name, timestamp, attributes=None):
        self.name = name
        self.timestamp = timestamp
        self.attributes = attributes or {}


class FakeSpan:
    def __init__(
        self,
        name,
        attributes=None,
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        trace_id=0x1,
        span_id=0x2,
        parent_id=None,
        status=None,
        events=None,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.start_time = start_time
        self.end_time = end_time
        self._ctx = FakeSpanContext(trace_id, span_id)
        self.parent = FakeParentContext(parent_id) if parent_id is not None else None
        self.status = status or FakeStatus()
        self.events = events or []

    def get_span_context(self):
        return self._ctx


class FakeSimpleSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter


class FakeTracerProvider:
    def __init__(self, resource=None):
        self.resource = resource
        self.processors = []

    def add_span_processor(self, processor):
        self.processors.append(processor)


class FakeTraceModule:
    def __init__(self):
        self._provider = FakeTracerProvider()

    def get_tracer_provider(self):
        return self._provider

    def set_tracer_provider(self, provider):
        self._provider = provider


class FakeCrewBus:
    def __init__(self):
        self.handlers = []

    def on(self, _event_type):
        def decorator(fn):
            self.handlers.append(fn)
            return fn

        return decorator

    def emit(self, source, event):
        for handler in list(self.handlers):
            handler(source, event)

    def flush(self):
        return None


class FakeBaseEventListener:
    def __init__(self):
        self._groundeval_initialized = True
        crewai_events = sys.modules.get("crewai.events")
        if crewai_events is not None:
            event_bus = getattr(crewai_events, "crewai_event_bus", None)
            setup = getattr(self, "setup_listeners", None)
            if event_bus is not None and callable(setup):
                setup(event_bus)


class FakeCrewEvent:
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeCrewResult:
    def __init__(self, raw):
        self.raw = raw
        self.pydantic = None


class FakeCrew:
    def __init__(self, bus, case, scenario):
        self._bus = bus
        self._case = case
        self._scenario = scenario
        self.max_iter = 99

    def kickoff(self):
        agent = SimpleNamespace(
            id=f"{self._case.framework}-agent-1", role="orchestrator"
        )
        source = SimpleNamespace(__class__=SimpleNamespace(__name__="CrewSource"))
        self._bus.emit(
            source,
            FakeCrewEvent(
                "KickoffStarted",
                timestamp="1.0",
                agent=agent,
                agent_id=None if self._scenario.missing_metadata else agent.id,
                agent_role=None if self._scenario.missing_metadata else agent.role,
                task_id="task-1",
                task_name=None if self._scenario.missing_metadata else "account_review",
            ),
        )
        self._bus.emit(
            source,
            FakeCrewEvent(
                "LLMCallCompleted",
                timestamp="1.1",
                agent=agent,
                agent_id=agent.id,
                agent_role=agent.role,
                model="test-model",
                prompt_tokens=10,
                completion_tokens=20,
                finish_reason="stop",
                task_name="account_review",
            ),
        )
        for idx, call in enumerate(self._scenario.tool_calls, start=1):
            event_type = "ToolUsageFinished"
            output = call["return_value"]
            if call.get("error_event"):
                event_type = "ToolUsageError"
                output = None
            tool_args = call["raw_args"]
            kwargs = {
                "timestamp": f"1.{idx + 1}",
                "event_id": f"evt-{idx}",
                "parent_event_id": None
                if self._scenario.missing_metadata
                else f"parent-{idx}",
                "agent": agent,
                "agent_id": None if self._scenario.missing_metadata else agent.id,
                "agent_role": None if self._scenario.missing_metadata else agent.role,
                "task_id": None if self._scenario.missing_metadata else f"task-{idx}",
                "task_name": None
                if self._scenario.missing_metadata
                else call.get("node_name"),
                "tool_name": call["tool_name"],
                "tool_args": tool_args,
                "output": json.dumps(output)
                if isinstance(output, (dict, list))
                else output,
            }
            if call.get("error_event"):
                kwargs["error"] = call.get("error_message", "tool failed")
            self._bus.emit(source, FakeCrewEvent(event_type, **kwargs))
        self._bus.emit(
            source,
            FakeCrewEvent(
                "DelegationEvent",
                timestamp="1.9",
                agent_id=agent.id,
                source_agent_id="account_review",
                target_agent_id="routing",
            ),
        )
        if self._scenario.emit_error:
            self._bus.emit(
                source,
                FakeCrewEvent(
                    "KickoffFailed",
                    timestamp="2.0",
                    agent=agent,
                    agent_id=agent.id,
                    agent_role=agent.role,
                    error="crew kickoff failed",
                ),
            )
        else:
            self._bus.emit(
                source,
                FakeCrewEvent(
                    "KickoffCompleted",
                    timestamp="2.0",
                    agent=agent,
                    agent_id=agent.id,
                    agent_role=agent.role,
                    output=self._scenario.final_answer,
                ),
            )
        return FakeCrewResult(self._scenario.final_answer)


class FakeStaticNode:
    def __init__(self, node_id):
        self.id = node_id


class FakeStaticEdge:
    def __init__(self, source, target):
        self.source = source
        self.target = target


class FakeStaticGraph:
    def __init__(self, node_names):
        self.nodes = {name: FakeStaticNode(name) for name in node_names}
        self.edges = (
            [FakeStaticEdge(node_names[0], node_names[-1])] if node_names else []
        )


class FakeLangGraph:
    def __init__(self, case, scenario):
        self._case = case
        self._scenario = scenario

    def compile(self):
        return self

    def get_graph(self):
        names = [
            call.get("node_name")
            for call in self._scenario.tool_calls
            if call.get("node_name")
        ]
        if not names:
            names = ["account_review", "routing"]
        ordered = []
        for name in names:
            if name not in ordered:
                ordered.append(name)
        return FakeStaticGraph(ordered)

    def get_subgraphs(self):
        return ["root", "branch-a", "branch-b"]

    async def astream(self, inputs, config=None, stream_mode=None, subgraphs=None):
        callback = config["callbacks"][0]
        chain_run_id = "chain-run-1"
        chain_metadata = (
            {}
            if self._scenario.missing_metadata
            else {"langgraph_node": "account_review"}
        )
        callback.on_chain_start(
            {"name": "account_review"},
            {"customer": "Acme"},
            run_id=chain_run_id,
            metadata=chain_metadata,
        )
        callback.on_llm_start(
            {"name": "planner"},
            ["review account"],
            run_id="llm-start-1",
            parent_run_id=chain_run_id,
            metadata=chain_metadata,
        )
        fake_generation = SimpleNamespace(
            message=SimpleNamespace(
                response_metadata={"finish_reason": "stop", "model_name": "test-model"},
                usage_metadata={"input_tokens": 10, "output_tokens": 20},
                tool_calls=[{"name": "lookup_customer_account"}],
            )
        )
        fake_response = SimpleNamespace(
            llm_output={
                "provider": "test-provider",
                "model_name": "test-model",
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            generations=[[fake_generation]],
        )
        callback.on_llm_end(
            fake_response, run_id="llm-end-1", parent_run_id=chain_run_id
        )
        for idx, call in enumerate(self._scenario.tool_calls, start=1):
            metadata = (
                {}
                if self._scenario.missing_metadata
                else {"langgraph_node": call.get("node_name")}
            )
            tool_input = (
                call["raw_args"]
                if isinstance(call["raw_args"], str)
                else json.dumps(call["raw_args"])
                if call["raw_args"] is not None
                else ""
            )
            callback.on_tool_start(
                {"name": call["tool_name"]},
                tool_input,
                run_id=f"tool-run-{idx}",
                parent_run_id=None if self._scenario.missing_metadata else chain_run_id,
                metadata=metadata,
                inputs=call["inputs_for_callback"],
            )
            if call.get("error_event"):
                callback.on_tool_error(
                    RuntimeError(call.get("error_message", "tool failed")),
                    run_id=f"tool-run-{idx}",
                    parent_run_id=chain_run_id,
                )
            else:
                callback.on_tool_end(
                    call["return_value"],
                    run_id=f"tool-run-{idx}",
                    parent_run_id=chain_run_id,
                )
            ns = tuple(
                x
                for x in (
                    "root",
                    None
                    if self._scenario.missing_metadata
                    else f"{call.get('node_name')}:task",
                )
                if x
            )
            yield {
                "type": "debug",
                "ns": ns,
                "data": {
                    "node": call.get("node_name"),
                    "input": call["raw_args"],
                    "output": call.get("return_value"),
                },
            }
            yield {
                "type": "updates",
                "ns": ns,
                "data": {
                    call.get("node_name") or call["tool_name"]: call.get("return_value")
                },
            }
        callback.on_chain_end({"done": True}, run_id=chain_run_id)
        if self._scenario.raise_stream_error:
            raise RuntimeError("langgraph stream failed")
        yield {
            "type": "values",
            "ns": ("root", "routing:task"),
            "data": self._scenario.final_answer,
        }


class FakeMafEntity:
    def __init__(self, scenario):
        self._scenario = scenario

    def run(self, *args, **kwargs):
        if self._scenario.raise_stream_error:
            raise RuntimeError("maf run failed")
        if isinstance(self._scenario.final_answer, str):
            return self._scenario.final_answer
        return json.dumps(self._scenario.final_answer)


@dataclass
class Scenario:
    name: str
    tool_calls: list[dict]
    final_answer: object
    reviewed_required_tool: str = "lookup_customer_account"
    reviewed_expected_return: dict | None = None
    emit_error: bool = False
    raise_stream_error: bool = False
    missing_metadata: bool = False


@dataclass
class AdapterCase:
    framework: str

    def base_tool_calls(self):
        return [
            {
                "tool_name": "lookup_customer_account",
                "raw_args": {"customer": "Acme", "channel": "email"},
                "inputs_for_callback": {"customer": "Acme", "channel": "email"},
                "return_value": {
                    "customer": "Acme",
                    "account_status": "delinquent",
                    "risk": "high",
                    "last_contacted": "2026-01-12",
                },
                "node_name": "account_review",
            },
            {
                "tool_name": "read_support_history",
                "raw_args": {"customer": "Acme", "window_days": 30},
                "inputs_for_callback": {"customer": "Acme", "window_days": 30},
                "return_value": [
                    {"name": "Nadia", "role": "sales"},
                    {"name": "Jax", "role": "support"},
                ],
                "node_name": "support_review",
            },
            {
                "tool_name": "calculate_renewal_risk",
                "raw_args": {"customer": "Acme"},
                "inputs_for_callback": {"customer": "Acme"},
                "return_value": {
                    "account": {"name": "Acme", "status": "delinquent"},
                    "checks": {"billing_reviewed": True, "legal_hold": False},
                },
                "node_name": "risk_review",
            },
            {
                "tool_name": "manual_review_summary",
                "raw_args": {"customer": "Acme"},
                "inputs_for_callback": {"customer": "Acme"},
                "return_value": "Manual review completed: account is safe to contact.",
                "node_name": "manual_review",
            },
            {
                "tool_name": "route_case_owner",
                "raw_args": {"customer": "Acme", "owner": "renewals"},
                "inputs_for_callback": {"customer": "Acme", "owner": "renewals"},
                "return_value": {},
                "node_name": "routing",
            },
            {
                "tool_name": "update_account_status",
                "raw_args": {"customer": "Acme", "status": "reviewed"},
                "inputs_for_callback": {"customer": "Acme", "status": "reviewed"},
                "return_value": [],
                "node_name": "status_update",
            },
        ]

    def final_answer_dict(self):
        return {
            "preconditions_verified": [
                {
                    "check": "account_review_complete",
                    "passed": True,
                    "facts_found": {"account_status": "delinquent", "risk": "high"},
                    "evidence_artifacts": [],
                },
                {
                    "check": "support_history_reviewed",
                    "passed": True,
                    "facts_found": {
                        "history_summary": "recent support interactions reviewed"
                    },
                    "evidence_artifacts": [],
                },
            ],
            "all_preconditions_pass": True,
            "should_act": True,
            "reasoning": "Native framework tools confirmed account risk and support context.",
        }

    def scenarios(self):
        happy = Scenario(
            name="happy_path_native_trace",
            tool_calls=self.base_tool_calls(),
            final_answer=self.final_answer_dict(),
        )
        missing_tool_calls = [
            call
            for call in self.base_tool_calls()
            if call["tool_name"] != "lookup_customer_account"
        ]
        missing_tool = Scenario(
            name="missing_required_native_tool",
            tool_calls=missing_tool_calls,
            final_answer=self.final_answer_dict(),
        )
        wrong_tool_calls = [dict(call) for call in self.base_tool_calls()]
        wrong_tool_calls[0]["tool_name"] = "lookup_customer_profile"
        wrong_tool = Scenario(
            name="wrong_native_tool_called",
            tool_calls=wrong_tool_calls,
            final_answer=self.final_answer_dict(),
        )
        wrong_return_calls = [dict(call) for call in self.base_tool_calls()]
        wrong_return_calls[0]["return_value"] = {
            "customer": "Acme",
            "account_status": "current",
            "risk": "low",
        }
        wrong_return = Scenario(
            name="wrong_native_return_value",
            tool_calls=wrong_return_calls,
            final_answer=self.final_answer_dict(),
            reviewed_expected_return={"account_status": "delinquent", "risk": "high"},
        )
        malformed_calls = [dict(call) for call in self.base_tool_calls()]
        malformed_calls[0]["raw_args"] = '{"customer": "Acme"'
        malformed_calls[0]["inputs_for_callback"] = None
        malformed_calls[1]["raw_args"] = "customer=Acme"
        malformed_calls[1]["inputs_for_callback"] = None
        malformed_calls[2]["raw_args"] = ["Acme", "billing"]
        malformed_calls[2]["inputs_for_callback"] = None
        malformed_calls[3]["raw_args"] = None
        malformed_calls[3]["inputs_for_callback"] = None
        malformed_args = Scenario(
            name="malformed_or_non_dict_tool_arguments",
            tool_calls=malformed_calls,
            final_answer=self.final_answer_dict(),
        )
        final_answer_json = Scenario(
            name="final_answer_json_string",
            tool_calls=self.base_tool_calls(),
            final_answer=json.dumps(self.final_answer_dict()),
        )
        final_answer_text = Scenario(
            name="final_answer_plain_text",
            tool_calls=self.base_tool_calls(),
            final_answer="Manual review completed with no structured answer fields.",
        )
        final_answer_list = Scenario(
            name="final_answer_list",
            tool_calls=self.base_tool_calls(),
            final_answer=["safe", "contact"],
        )
        missing_metadata_calls = [dict(call) for call in self.base_tool_calls()]
        missing_metadata = Scenario(
            name="missing_metadata",
            tool_calls=missing_metadata_calls,
            final_answer=self.final_answer_dict(),
            missing_metadata=True,
        )
        duplicate_calls = [dict(call) for call in self.base_tool_calls()]
        duplicate_calls.insert(
            1,
            {
                "tool_name": "lookup_customer_account",
                "raw_args": {"customer": "Acme", "channel": "phone"},
                "inputs_for_callback": {"customer": "Acme", "channel": "phone"},
                "return_value": {
                    "customer": "Acme",
                    "account_status": "delinquent",
                    "risk": "medium",
                },
                "node_name": "account_review",
            },
        )
        duplicate = Scenario(
            name="duplicate_native_tool_calls",
            tool_calls=duplicate_calls,
            final_answer=self.final_answer_dict(),
        )
        error_calls = [dict(call) for call in self.base_tool_calls()]
        error_calls[1]["error_event"] = True
        error_calls[1]["error_message"] = "support history unavailable"
        adapter_error = Scenario(
            name="adapter_error_or_failed_tool_event",
            tool_calls=error_calls,
            final_answer=self.final_answer_dict(),
            emit_error=False,
        )
        return [
            happy,
            missing_tool,
            wrong_tool,
            wrong_return,
            malformed_args,
            final_answer_json,
            final_answer_text,
            final_answer_list,
            missing_metadata,
            duplicate,
            adapter_error,
        ]

    def install_modules(self, monkeypatch, scenario):
        if self.framework == "crewai":
            self._install_crewai(monkeypatch, scenario)
        elif self.framework == "langgraph":
            self._install_langgraph(monkeypatch, scenario)
        elif self.framework == "maf":
            self._install_maf(monkeypatch, scenario)
        else:
            raise AssertionError(self.framework)

    def _install_crewai(self, monkeypatch, scenario):
        bus = FakeCrewBus()
        crewai_mod = types.ModuleType("crewai")
        crewai_events_mod = types.ModuleType("crewai.events")
        crewai_events_mod.BaseEventListener = FakeBaseEventListener
        crewai_events_mod.crewai_event_bus = bus
        test_agents_mod = types.ModuleType("test_agents")
        agent_mod = types.ModuleType("test_agents.fake_crewai")
        agent_mod.entry = lambda: FakeCrew(bus, self, scenario)
        monkeypatch.setitem(sys.modules, "crewai", crewai_mod)
        monkeypatch.setitem(sys.modules, "crewai.events", crewai_events_mod)
        monkeypatch.setitem(sys.modules, "test_agents", test_agents_mod)
        monkeypatch.setitem(sys.modules, "test_agents.fake_crewai", agent_mod)

    def _install_langgraph(self, monkeypatch, scenario):
        test_agents_mod = types.ModuleType("test_agents")
        agent_mod = types.ModuleType("test_agents.fake_langgraph")
        agent_mod.entry = lambda: FakeLangGraph(self, scenario)
        monkeypatch.setitem(sys.modules, "test_agents", test_agents_mod)
        monkeypatch.setitem(sys.modules, "test_agents.fake_langgraph", agent_mod)

    def _install_maf(self, monkeypatch, scenario):
        fake_trace = FakeTraceModule()
        otel_mod = types.ModuleType("opentelemetry")
        trace_mod = types.ModuleType("opentelemetry.trace")
        trace_mod.get_tracer_provider = fake_trace.get_tracer_provider
        trace_mod.set_tracer_provider = fake_trace.set_tracer_provider
        sdk_mod = types.ModuleType("opentelemetry.sdk")
        sdk_trace_mod = types.ModuleType("opentelemetry.sdk.trace")
        sdk_trace_mod.TracerProvider = FakeTracerProvider
        sdk_export_mod = types.ModuleType("opentelemetry.sdk.trace.export")
        sdk_export_mod.SimpleSpanProcessor = FakeSimpleSpanProcessor
        sdk_export_mod.SpanExportResult = SimpleNamespace(SUCCESS=1)
        sdk_resources_mod = types.ModuleType("opentelemetry.sdk.resources")
        sdk_resources_mod.Resource = SimpleNamespace(create=lambda data: data)
        agent_framework_mod = types.ModuleType("agent_framework")
        observability_mod = types.ModuleType("agent_framework.observability")
        observability_mod.enable_instrumentation = lambda *args, **kwargs: None
        test_agents_mod = types.ModuleType("test_agents")
        agent_mod = types.ModuleType("test_agents.fake_maf")
        agent_mod.entry = lambda: FakeMafEntity(scenario)
        monkeypatch.setitem(sys.modules, "opentelemetry", otel_mod)
        monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)
        monkeypatch.setitem(sys.modules, "opentelemetry.sdk", sdk_mod)
        monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", sdk_trace_mod)
        monkeypatch.setitem(
            sys.modules, "opentelemetry.sdk.trace.export", sdk_export_mod
        )
        monkeypatch.setitem(
            sys.modules, "opentelemetry.sdk.resources", sdk_resources_mod
        )
        monkeypatch.setitem(sys.modules, "agent_framework", agent_framework_mod)
        monkeypatch.setitem(
            sys.modules, "agent_framework.observability", observability_mod
        )
        monkeypatch.setitem(sys.modules, "test_agents", test_agents_mod)
        monkeypatch.setitem(sys.modules, "test_agents.fake_maf", agent_mod)
        import groundeval.framework_adapters.maf_adapter as maf_adapter

        real_install = maf_adapter._install_in_memory_otel_exporter

        def wrapped_install():
            exporter = real_install()
            spans = []
            spans.append(
                FakeSpan(
                    "workflow.build",
                    {
                        "workflow.id": "maf-workflow-1",
                        "workflow.name": "maf workflow",
                        "workflow.description": "test workflow",
                    },
                    span_id=0x10,
                    trace_id=0x100,
                )
            )
            spans.append(
                FakeSpan(
                    "invoke_agent orchestrator",
                    {
                        "gen_ai.agent.id": None
                        if scenario.missing_metadata
                        else "maf-agent-1",
                        "gen_ai.agent.name": None
                        if scenario.missing_metadata
                        else "orchestrator",
                        "gen_ai.request.instructions": "review account",
                    },
                    span_id=0x11,
                    trace_id=0x100,
                )
            )
            spans.append(
                FakeSpan(
                    "workflow.run",
                    {"workflow.id": "maf-workflow-1"},
                    span_id=0x12,
                    trace_id=0x100,
                )
            )
            spans.append(
                FakeSpan(
                    "chat completion",
                    {
                        "gen_ai.request.model": "test-model",
                        "gen_ai.system": "test-provider",
                        "gen_ai.usage.input_tokens": 10,
                        "gen_ai.usage.output_tokens": 20,
                        "gen_ai.response.finish_reason": "stop",
                        "gen_ai.request.tool_schemas.count": 2,
                    },
                    span_id=0x13,
                    trace_id=0x100,
                )
            )
            for idx, call in enumerate(scenario.tool_calls, start=1):
                attrs = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": call["tool_name"],
                    "gen_ai.tool.call.arguments": json.dumps(call["raw_args"])
                    if isinstance(call["raw_args"], (dict, list))
                    else call["raw_args"],
                    "gen_ai.tool.call.result": json.dumps(call.get("return_value"))
                    if isinstance(call.get("return_value"), (dict, list))
                    else call.get("return_value"),
                    "gen_ai.agent.id": None
                    if scenario.missing_metadata
                    else "maf-agent-1",
                    "gen_ai.agent.name": None
                    if scenario.missing_metadata
                    else "orchestrator",
                    "executor.id": None
                    if scenario.missing_metadata
                    else call.get("node_name"),
                }
                status = (
                    FakeStatus("ERROR", call.get("error_message", "tool failed"))
                    if call.get("error_event")
                    else FakeStatus()
                )
                spans.append(
                    FakeSpan(
                        f"execute_tool {call['tool_name']}",
                        attrs,
                        start_time=1_000_000_000 + idx * 1_000_000,
                        end_time=1_500_000_000 + idx * 1_000_000,
                        span_id=0x20 + idx,
                        trace_id=0x100,
                        parent_id=0x11,
                        status=status,
                        events=[
                            FakeSpanEvent(
                                "tool.progress",
                                1_100_000_000 + idx * 1_000_000,
                                {"step": idx},
                            )
                        ],
                    )
                )
            spans.append(
                FakeSpan(
                    "message.send",
                    {
                        "message.source_id": "account_review",
                        "message.target_id": "routing",
                        "message.type": "handoff",
                        "message.content": "route this case",
                    },
                    span_id=0x40,
                    trace_id=0x100,
                    parent_id=0x12,
                )
            )
            exporter.export(spans)
            return exporter

        monkeypatch.setattr(
            maf_adapter, "_install_in_memory_otel_exporter", wrapped_install
        )

    def agent_class_path(self):
        return f"test_agents.fake_{self.framework}.entry"

    def reviewed_config(
        self,
        scenario,
        required_tool=None,
        expected_return=None,
        require_agent_metadata=False,
    ):
        required_tool = (
            required_tool
            if required_tool is not None
            else scenario.reviewed_required_tool
        )
        expected_return = (
            expected_return
            if expected_return is not None
            else scenario.reviewed_expected_return
            or {"account_status": "delinquent", "risk": "high"}
        )
        cfg = {
            "output_dir": "./eval_output",
            "agent": {
                "framework": self.framework,
                "agent_class": self.agent_class_path(),
            },
            "task_contracts": [
                {
                    "name": "native_tool_contract",
                    "task_description": "Validate native tool preservation without demo assumptions.",
                    "preconditions": [
                        {
                            "check": "account_review_complete",
                            "description": "Account review must use native lookup tool.",
                            "required_facts": ["account_status", "risk"],
                            "required_tool": required_tool,
                            "expected_field": "account_status",
                        },
                        {
                            "check": "support_history_reviewed",
                            "description": "Support history must be represented in final answer.",
                            "required_facts": ["history_summary"],
                            "required_tool": "read_support_history",
                            "expected_field": "history_summary",
                        },
                    ],
                    "decision_field": "should_act",
                    "tool_expectations": [
                        {
                            "tool": required_tool,
                            "match_args": {"customer": "Acme"},
                            "expected_return": expected_return,
                        },
                        {
                            "tool": "read_support_history",
                            "match_args": {"customer": "Acme"},
                        },
                        {
                            "tool": "calculate_renewal_risk",
                            "match_args": {"customer": "Acme"},
                            "expected_return": {
                                "account.name": "Acme",
                                "checks.billing_reviewed": True,
                            },
                        },
                    ],
                }
            ],
            "groundeval": {
                "config_status": "reviewed",
                "generated_from_observation": True,
                "reviewed": True,
            },
        }
        if require_agent_metadata:
            cfg["task_contracts"][0]["required_agents"] = [
                {"agent_name": "orchestrator"}
            ]
        return cfg


CASES = [AdapterCase("crewai"), AdapterCase("langgraph"), AdapterCase("maf")]


@pytest.fixture(
    params=[(case, scenario) for case in CASES for scenario in case.scenarios()],
    ids=lambda item: f"{item[0].framework}:{item[1].name}",
)
def adapter_scenario(request, monkeypatch):
    case, scenario = request.param
    case.install_modules(monkeypatch, scenario)
    monkeypatch.setattr("groundeval.run._timestamp_suffix", lambda: "20240101_000000")

    def fake_write_observe_diagram_pdf(output_dir, observed, direction="TD"):
        path = Path(output_dir) / "observe_diagram.pdf"
        path.write_bytes(b"%PDF-1.4\n% GroundEval test diagram\n")
        return path

    monkeypatch.setattr(
        "groundeval.diagram.write_observe_diagram_pdf", fake_write_observe_diagram_pdf
    )
    return case, scenario


def assert_no_demo_leakage(text):
    assert "agent.tool_map" not in text
    assert "tool_map.yaml" not in text
    assert "fetch_artifact" not in text
    assert "search_artifacts" not in text
    assert "FileCorpusAdapter" not in text


def read_json(path):
    assert path.exists()
    assert path.read_bytes()
    with path.open() as f:
        return json.load(f)


def read_yaml(path):
    assert path.exists()
    assert path.read_bytes()
    with path.open() as f:
        return yaml.safe_load(f)


def assert_observe_artifacts(base_dir, framework):
    required = [
        base_dir / "observed_run_20240101_000000.json",
        base_dir / "observe_report_20240101_000000.md",
        base_dir / "observe_diagram_20240101_000000.pdf",
    ]
    optional = []
    if framework == "crewai":
        optional.append(base_dir / "observed_run_crewai_20240101_000000.json")
    if framework == "langgraph":
        optional.append(base_dir / "observed_run_langgraph_20240101_000000.json")
        optional.append(base_dir / "observe_report_langgraph_20240101_000000.md")
    if framework == "maf":
        optional.append(base_dir / "observed_run_maf_20240101_000000.json")
        optional.append(base_dir / "observe_report_maf_20240101_000000.md")
    for path in required:
        assert path.exists(), str(path)
        assert path.read_bytes(), str(path)
    existing_optional = []
    for path in optional:
        if path.exists():
            assert path.read_bytes(), str(path)
            existing_optional.append(path)
    return required + existing_optional


def assert_draft_artifacts(base_dir):
    expected = [
        base_dir / "observed_run.json",
        base_dir / "observe_report.md",
        base_dir / "draft_config" / "config.yaml",
        base_dir / "draft_config" / "REVIEW.md",
        base_dir / "draft_config" / "task_contracts" / "inferred_task.yaml",
    ]
    for path in expected:
        assert path.exists(), str(path)
        assert path.read_bytes(), str(path)
    return expected


def assert_score_artifacts(base_dir):
    score_path = base_dir / "observed_scores_20240101_000000.json"
    assert score_path.exists()
    assert score_path.read_bytes()
    payload = read_json(score_path)
    return score_path, payload


def assert_written_files_clean(base_dir):
    for path in base_dir.rglob("*"):
        if path.is_file():
            assert_no_demo_leakage(path.read_text(errors="ignore"))


def execute_full_pipeline(case, scenario, config, output_root):
    observed = observe_agent(case.framework, case.agent_class_path(), max_steps=7)
    observe_dir = output_root / "observe_output"
    _write_basic_observation_outputs(observe_dir, observed, "20240101_000000")
    draft_dir = output_root / "draft_output"
    write_draft_output(draft_dir, observed, DraftGenerator(observed, mode="standard"))
    reviewed_path = output_root / "reviewed_config.yaml"
    reviewed_path.write_text(yaml.dump(config, sort_keys=False))
    score_dir = output_root / "scored_output"
    score_dir.mkdir(parents=True, exist_ok=True)
    _, payload = score_observed_run(observed, config, config_path=reviewed_path)
    score_path = score_dir / "observed_scores_20240101_000000.json"
    score_path.write_text(json.dumps(payload, indent=2, default=str))
    return observed, observe_dir, draft_dir, score_dir


def tool_names_from_observed_json(payload):
    return [tc["tool_name"] for tc in payload["tool_calls"]]


def tool_names_from_score_payload(payload):
    names = []
    for traj in payload.get("trajectories", []):
        for call in traj.get("tool_calls", []):
            names.append(call["tool_name"])
    return names


def first_result(payload):
    return payload["results"][0]


def reasons_for_check(result, check_name):
    for item in result["precondition_results"]:
        if item["check"] == check_name:
            return item.get("reasons", [])
    return []


def assert_common_native_shapes(payload, duplicate_expected=False):
    calls = payload["tool_calls"]
    names = [tc["tool_name"] for tc in calls]
    if "lookup_customer_account" in names:
        first_lookup = next(
            tc for tc in calls if tc["tool_name"] == "lookup_customer_account"
        )
        if isinstance(first_lookup["return_value"], dict):
            assert "artifact_id" not in first_lookup["return_value"]
            assert "document_id" not in first_lookup["return_value"]
            assert "ticket_id" not in first_lookup["return_value"]
    if "read_support_history" in names:
        support_call = next(
            tc for tc in calls if tc["tool_name"] == "read_support_history"
        )
        if (
            isinstance(support_call["return_value"], list)
            and support_call["return_value"]
        ):
            assert isinstance(support_call["return_value"][0], dict)
    if "calculate_renewal_risk" in names:
        nested_call = next(
            tc for tc in calls if tc["tool_name"] == "calculate_renewal_risk"
        )
        if isinstance(nested_call["return_value"], dict):
            assert "account" in nested_call["return_value"]
    if "manual_review_summary" in names:
        text_call = next(
            tc for tc in calls if tc["tool_name"] == "manual_review_summary"
        )
        assert isinstance(text_call["return_value"], str)
    if "route_case_owner" in names:
        empty_dict_call = next(
            tc for tc in calls if tc["tool_name"] == "route_case_owner"
        )
        assert empty_dict_call["return_value"] == {}
    if "update_account_status" in names:
        empty_list_call = next(
            tc for tc in calls if tc["tool_name"] == "update_account_status"
        )
        assert empty_list_call["return_value"] == []
    if duplicate_expected:
        assert names.count("lookup_customer_account") == 2


@pytest.mark.parametrize("target_case", ["happy_path_native_trace"])
def test_adapter_happy_path_runs_full_pipeline_and_reads_written_files(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("case_output")
    )
    assert observed.framework == case.framework
    observe_files = assert_observe_artifacts(observe_dir, case.framework)
    draft_files = assert_draft_artifacts(draft_dir)
    score_path, score_payload = assert_score_artifacts(score_dir)
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    draft_cfg = read_yaml(draft_dir / "draft_config" / "config.yaml")
    draft_report = (draft_dir / "observe_report.md").read_text()
    review_md = (draft_dir / "draft_config" / "REVIEW.md").read_text()
    observed_names = tool_names_from_observed_json(observed_payload)
    assert observed_names == [call["tool_name"] for call in scenario.tool_calls]
    assert_common_native_shapes(observed_payload)
    assert draft_cfg["agent"]["framework"] == case.framework
    assert "tool_map" not in draft_cfg["agent"]
    draft_tools = [
        item["tool"]
        for item in draft_cfg["task_contracts"][0].get("tool_expectations", [])
    ]
    for call in scenario.tool_calls:
        assert call["tool_name"] in draft_tools
    assert (
        "lookup_customer_account" in draft_report
        or "lookup_customer_account" in review_md
        or "lookup_customer_account" in json.dumps(draft_cfg)
    )
    assert score_payload["meta"]["framework"] == case.framework
    assert score_payload["meta"]["framework_native_scoring"] is True
    assert score_payload["meta"]["run_id"] == observed.run_id
    score_names = tool_names_from_score_payload(score_payload)
    assert "lookup_customer_account" in score_names
    assert "read_support_history" in score_names
    assert score_payload["summary"]["overall_score"] > 0
    assert score_path.read_text()
    assert_written_files_clean(Path("case_output"))
    assert observe_files and draft_files


@pytest.mark.parametrize("target_case", ["missing_required_native_tool"])
def test_adapter_missing_required_native_tool_survives_to_written_score_diagnostics(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario, required_tool="lookup_customer_account")
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("missing_tool_output")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    assert "lookup_customer_account" not in tool_names_from_observed_json(
        observed_payload
    )
    draft_cfg = read_yaml(draft_dir / "draft_config" / "config.yaml")
    draft_tools = [
        item["tool"]
        for item in draft_cfg["task_contracts"][0].get("tool_expectations", [])
    ]
    assert "lookup_customer_account" not in draft_tools
    score_path, score_payload = assert_score_artifacts(score_dir)
    result = first_result(score_payload)
    reasons = reasons_for_check(result, "account_review_complete")
    assert (
        "required_tool_not_called" in reasons or "required_tool_not_observed" in reasons
    )
    assert score_payload["summary"]["overall_score"] < 1.0
    assert "lookup_customer_account" in score_path.read_text()
    assert_written_files_clean(Path("missing_tool_output"))


@pytest.mark.parametrize("target_case", ["wrong_native_tool_called"])
def test_adapter_wrong_native_tool_name_stays_wrong_through_written_files(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario, required_tool="lookup_customer_account")
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("wrong_tool_output")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    names = tool_names_from_observed_json(observed_payload)
    assert "lookup_customer_profile" in names
    assert "lookup_customer_account" not in names
    score_path, score_payload = assert_score_artifacts(score_dir)
    result = first_result(score_payload)
    reasons = reasons_for_check(result, "account_review_complete")
    assert (
        "required_tool_not_called" in reasons or "required_tool_not_observed" in reasons
    )
    score_text = score_path.read_text()
    assert "lookup_customer_profile" in score_text
    assert "lookup_customer_account" in score_text
    assert_written_files_clean(Path("wrong_tool_output"))


@pytest.mark.parametrize("target_case", ["wrong_native_return_value"])
def test_adapter_wrong_native_return_value_is_preserved_and_penalized_in_written_score(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(
        scenario,
        required_tool="lookup_customer_account",
        expected_return={"account_status": "delinquent", "risk": "high"},
    )
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("wrong_return_output")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    lookup = next(
        tc
        for tc in observed_payload["tool_calls"]
        if tc["tool_name"] == "lookup_customer_account"
    )
    assert lookup["return_value"]["account_status"] == "current"
    assert lookup["return_value"]["risk"] == "low"
    score_path, score_payload = assert_score_artifacts(score_dir)
    result = first_result(score_payload)
    assert result["meta"]["framework_mode"] is True
    assert result["meta"]["cf_trajectory"] < 1.0
    assert score_payload["summary"]["overall_score"] < 1.0
    score_text = score_path.read_text()
    assert "lookup_customer_account" in score_text
    assert_written_files_clean(Path("wrong_return_output"))


@pytest.mark.parametrize("target_case", ["malformed_or_non_dict_tool_arguments"])
def test_adapter_malformed_or_non_dict_args_do_not_break_written_outputs(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("malformed_args_output")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    calls = {tc["tool_name"]: tc for tc in observed_payload["tool_calls"]}
    assert isinstance(calls["lookup_customer_account"]["arguments"], dict)
    assert isinstance(calls["read_support_history"]["arguments"], dict)
    assert isinstance(calls["calculate_renewal_risk"]["arguments"], dict)
    assert isinstance(calls["manual_review_summary"]["arguments"], dict)
    assert_draft_artifacts(draft_dir)
    score_path, score_payload = assert_score_artifacts(score_dir)
    assert score_payload["meta"]["framework"] == case.framework
    assert score_path.read_text()
    assert_written_files_clean(Path("malformed_args_output"))


@pytest.mark.parametrize(
    "target_case",
    ["final_answer_json_string", "final_answer_plain_text", "final_answer_list"],
)
def test_adapter_non_dict_final_answer_surfaces_in_written_outputs_and_scoring(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path(f"final_answer_output_{scenario.name}")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    report_text = (draft_dir / "observe_report.md").read_text()
    final_answer = observed_payload["final_answer"]

    if case.framework == "crewai":
        if scenario.name == "final_answer_json_string":
            assert isinstance(final_answer, str)
            assert "account_review_complete" in final_answer
        elif scenario.name == "final_answer_plain_text":
            assert (
                final_answer
                == "Manual review completed with no structured answer fields."
            )
            assert (
                "Manual review completed with no structured answer fields."
                in report_text
            )
        elif scenario.name == "final_answer_list":
            assert final_answer == ["safe", "contact"]
            assert "safe" in report_text

    elif case.framework == "langgraph":
        assert isinstance(final_answer, dict)
        assert final_answer == {"status_update": []}

    elif case.framework == "maf":
        if scenario.name == "final_answer_json_string":
            assert isinstance(final_answer, dict)
            assert "account_review_complete" in json.dumps(final_answer)
        elif scenario.name == "final_answer_plain_text":
            assert (
                final_answer
                == "Manual review completed with no structured answer fields."
            )
            assert (
                "Manual review completed with no structured answer fields."
                in report_text
            )
        elif scenario.name == "final_answer_list":
            assert final_answer == ["safe", "contact"]
            assert "safe" in report_text

    score_path, score_payload = assert_score_artifacts(score_dir)
    if scenario.name != "final_answer_json_string":
        assert score_payload["summary"]["overall_score"] < 1.0
    assert score_path.read_text()
    assert_written_files_clean(Path(f"final_answer_output_{scenario.name}"))


@pytest.mark.parametrize("target_case", ["missing_metadata"])
def test_adapter_missing_metadata_remains_absent_but_pipeline_still_writes_files(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("missing_metadata_output")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    assert observed_payload["tool_calls"]
    first_call = observed_payload["tool_calls"][0]

    if case.framework == "crewai":
        assert first_call.get("agent_id") == "crewai-agent-1"
        assert first_call.get("agent_name") == "orchestrator"
        assert first_call.get("node_name") in (None, "", "None")

    elif case.framework == "langgraph":
        assert first_call.get("agent_id")
        assert first_call.get("agent_name")
        assert first_call.get("node_name")

    elif case.framework == "maf":
        assert first_call.get("agent_id") in (None, "", "None")
        assert first_call.get("agent_name") in (None, "", "None")
        assert first_call.get("node_name") == "execute_tool lookup_customer_account"
    assert_observe_artifacts(observe_dir, case.framework)
    score_path, score_payload = assert_score_artifacts(score_dir)
    assert score_payload["meta"]["framework"] == case.framework
    assert score_path.read_text()
    assert_written_files_clean(Path("missing_metadata_output"))


@pytest.mark.parametrize("target_case", ["duplicate_native_tool_calls"])
def test_adapter_duplicate_native_tool_calls_preserve_order_and_written_scoreability(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("duplicate_tool_output")
    )
    observed_payload = read_json(observe_dir / "observed_run_20240101_000000.json")
    names = tool_names_from_observed_json(observed_payload)
    assert names.count("lookup_customer_account") == 2
    lookup_calls = [
        tc
        for tc in observed_payload["tool_calls"]
        if tc["tool_name"] == "lookup_customer_account"
    ]
    assert lookup_calls[0]["arguments"]["channel"] == "email"
    assert lookup_calls[1]["arguments"]["channel"] == "phone"
    assert_common_native_shapes(observed_payload, duplicate_expected=True)
    score_path, score_payload = assert_score_artifacts(score_dir)
    assert "lookup_customer_account" in tool_names_from_score_payload(score_payload)
    assert score_path.read_text()
    assert_written_files_clean(Path("duplicate_tool_output"))


@pytest.mark.parametrize("target_case", ["adapter_error_or_failed_tool_event"])
def test_adapter_error_or_failed_tool_event_reaches_observe_and_score_files(
    adapter_scenario, target_case
):
    case, scenario = adapter_scenario
    if scenario.name != target_case:
        pytest.skip("scenario filter")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path("error_output")
    )
    rich_path = observe_dir / f"observed_run_{case.framework}_20240101_000000.json"
    base_observed = read_json(observe_dir / "observed_run_20240101_000000.json")

    if rich_path.exists():
        rich_observed = read_json(rich_path)
        rich_text = rich_path.read_text()
        assert "errors" in rich_observed
        if case.framework in {"crewai", "langgraph"}:
            assert "support history unavailable" in rich_text or rich_observed["errors"]
        if case.framework == "maf":
            assert rich_observed["errors"]
    else:
        assert base_observed["framework"] == case.framework
    score_path, score_payload = assert_score_artifacts(score_dir)
    assert score_payload["summary"]["overall_score"] <= 1.0
    assert score_path.read_text()
    assert_written_files_clean(Path("error_output"))


def test_observe_only_cases_always_write_and_read_observe_and_draft_files(
    adapter_scenario,
):
    case, scenario = adapter_scenario
    observed = observe_agent(case.framework, case.agent_class_path(), max_steps=7)
    observe_dir = Path(f"observe_only_{case.framework}_{scenario.name}")
    draft_dir = Path(f"draft_only_{case.framework}_{scenario.name}")
    _write_basic_observation_outputs(observe_dir, observed, "20240101_000000")
    write_draft_output(draft_dir, observed, DraftGenerator(observed, mode="standard"))
    assert_observe_artifacts(observe_dir, case.framework)
    assert_draft_artifacts(draft_dir)
    observe_json = read_json(observe_dir / "observed_run_20240101_000000.json")
    draft_cfg = read_yaml(draft_dir / "draft_config" / "config.yaml")
    assert observe_json["framework"] == case.framework
    assert draft_cfg["agent"]["framework"] == case.framework
    assert_written_files_clean(Path(f"."))


def test_written_score_file_contains_expected_meta_and_native_tool_names(
    adapter_scenario,
):
    case, scenario = adapter_scenario
    if scenario.name != "happy_path_native_trace":
        pytest.skip("targeted score file assertions only for happy path")
    config = case.reviewed_config(scenario)
    observed, observe_dir, draft_dir, score_dir = execute_full_pipeline(
        case, scenario, config, Path(f"score_meta_{case.framework}")
    )
    score_path, score_payload = assert_score_artifacts(score_dir)
    assert score_payload["meta"]["framework"] == case.framework
    assert score_payload["meta"]["run_id"] == observed.run_id
    assert score_payload["meta"]["framework_native_scoring"] is True
    score_text = score_path.read_text()
    score_tool_names = tool_names_from_score_payload(score_payload)
    for expected_name in [call["tool_name"] for call in scenario.tool_calls]:
        assert expected_name in score_tool_names
    assert "lookup_customer_account" in score_text
    assert "read_support_history" in score_text
    assert_written_files_clean(Path(f"score_meta_{case.framework}"))
