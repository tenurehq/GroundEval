import asyncio
import sys
import types
from unittest.mock import Mock, patch

import pytest

from groundeval.framework_adapters.framework_observation import (
    ObservedAgent,
    ObservedError,
    ObservedHandoff,
    ObservedModelEvent,
    ObservedRun,
    ObservedWorkflow,
)
from groundeval.framework_adapters.openai_agents_adapter import (
    OpenAIAgentsEntry,
    OpenAIAgentsObserver,
    _InstrumentedOpenAIAgentsEntry,
    _OpenAIAgentsCollector,
    _OpenAITraceProcessor,
    _agent_id,
    _agent_name,
    _coerce_int,
    _entry_parts,
    _final_output,
    _jsonish,
    _latency_ms,
    _load_openai_agents_entry,
    _looks_like_agent,
    _parse_jsonish,
    _run_entry,
    _span_data,
    _span_payload,
    _timestamp_seconds,
    _tool_arguments,
    _tool_call_id,
    _tool_name,
    generate_openai_agents_report,
)
from groundeval.observe import ObservedToolCall, RecordingRuntime


class _FactoryModule:
    pass


class _FakeAgent:
    def __init__(self, name="planner", agent_id=None, instructions="Plan carefully"):
        self.name = name
        self.tools = []
        self.handoffs = []
        self.instructions = instructions
        if agent_id is not None:
            self.id = agent_id


class _FakeTool:
    def __init__(self, name="lookup"):
        self.name = name


class _ModelDumpObj:
    def model_dump(self):
        return {"a": 1}


class _DictObj:
    def dict(self):
        return {"b": 2}


class _ToDictObj:
    def to_dict(self):
        return {"c": 3}


class _AllSerializersFail:
    def model_dump(self):
        raise RuntimeError("model")

    def dict(self):
        raise RuntimeError("dict")

    def to_dict(self):
        raise RuntimeError("to_dict")

    def __str__(self):
        return "fallback"


class _SpanData:
    def __init__(self, span_type, payload=None, raises=False):
        self.type = span_type
        self.payload = payload if payload is not None else {}
        self.raises = raises

    def export(self):
        if self.raises:
            raise RuntimeError("export failed")
        return self.payload


class _Span:
    def __init__(
        self,
        span_type="unknown",
        payload=None,
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
        parent_id="parent-1",
        error=None,
        export_value=None,
        export_raises=False,
        data_raises=False,
    ):
        self.span_data = _SpanData(span_type, payload, raises=data_raises)
        self.started_at = started_at
        self.ended_at = ended_at
        self.parent_id = parent_id
        self.span_id = "span-1"
        self.trace_id = "trace-1"
        self.error = error
        self.export_value = export_value
        self.export_raises = export_raises

    def export(self):
        if self.export_raises:
            raise RuntimeError("span export failed")
        return self.export_value


@pytest.fixture
def collector():
    return _OpenAIAgentsCollector(run_id="run-1", agent_class="pkg.Agent")


def test_load_entry_instantiates_class():
    class Entry:
        pass

    mod = _FactoryModule()
    mod.Entry = Entry
    with patch("importlib.import_module", return_value=mod):
        result = _load_openai_agents_entry("pkg.Entry")
    assert isinstance(result, Entry)


def test_load_entry_calls_function_factory():
    mod = _FactoryModule()
    sentinel = object()
    mod.make_entry = lambda: sentinel
    with patch("importlib.import_module", return_value=mod):
        result = _load_openai_agents_entry("pkg.make_entry")
    assert result is sentinel


def test_load_entry_preserves_agent_even_when_callable():
    class CallableAgent(_FakeAgent):
        def __call__(self):
            raise AssertionError("agent must not be called while loading")

    mod = _FactoryModule()
    agent = CallableAgent()
    mod.agent = agent
    with patch("importlib.import_module", return_value=mod):
        result = _load_openai_agents_entry("pkg.agent")
    assert result is agent


def test_looks_like_agent_requires_all_native_attributes():
    assert _looks_like_agent(_FakeAgent()) is True
    assert _looks_like_agent(None) is False
    assert _looks_like_agent(types.SimpleNamespace(name="x", tools=[])) is False
    assert _looks_like_agent(types.SimpleNamespace(name="x", handoffs=[])) is False


def test_jsonish_handles_primitives_and_serializer_protocols():
    assert _jsonish(None) is None
    assert _jsonish("x") == "x"
    assert _jsonish(1) == 1
    assert _jsonish([1]) == [1]
    assert _jsonish({"x": 1}) == {"x": 1}
    assert _jsonish(_ModelDumpObj()) == {"a": 1}
    assert _jsonish(_DictObj()) == {"b": 2}
    assert _jsonish(_ToDictObj()) == {"c": 3}
    assert _jsonish(_AllSerializersFail()) == "fallback"


def test_parse_jsonish_parses_json_and_preserves_plain_or_invalid_text():
    assert _parse_jsonish('{"a": 1}') == {"a": 1}
    assert _parse_jsonish('[1, 2]') == [1, 2]
    assert _parse_jsonish('"hello"') == "hello"
    assert _parse_jsonish("plain") == "plain"
    assert _parse_jsonish("{broken") == "{broken"


def test_identity_and_tool_helpers_use_explicit_values_and_fallbacks():
    explicit = _FakeAgent(name="planner", agent_id="agent-7")
    implicit = _FakeAgent(name="reviewer")
    unnamed = types.SimpleNamespace()
    assert _agent_name(explicit) == "planner"
    assert _agent_id(explicit) == "agent-7"
    assert _agent_id(implicit) == "openai-agents:reviewer"
    assert _agent_name(unnamed) == "SimpleNamespace"
    assert _tool_name(_FakeTool("fetch")) == "fetch"
    assert _tool_name(types.SimpleNamespace()) == "SimpleNamespace"


def test_tool_context_helpers_parse_dict_scalar_none_and_call_id():
    assert _tool_call_id(types.SimpleNamespace(tool_call_id=123)) == "123"
    assert _tool_call_id(types.SimpleNamespace()) is None
    assert _tool_arguments(types.SimpleNamespace(tool_arguments='{"id": "a1"}')) == {"id": "a1"}
    assert _tool_arguments(types.SimpleNamespace(tool_arguments='"hello"')) == {"raw": "hello"}
    assert _tool_arguments(types.SimpleNamespace(tool_arguments=None)) == {}


def test_numeric_and_timestamp_helpers_cover_valid_and_invalid_values():
    assert _coerce_int("5") == 5
    assert _coerce_int(7.9) == 7
    assert _coerce_int(None) is None
    assert _coerce_int("bad") is None
    assert _timestamp_seconds("1970-01-01T00:00:01Z") == 1.0
    assert _timestamp_seconds("2.5") == 2.5
    assert _timestamp_seconds(None) is None
    assert _timestamp_seconds("bad") is None


def test_latency_handles_iso_numeric_invalid_and_reverse_ranges():
    assert _latency_ms("1", "2.5") == 1500.0
    assert _latency_ms("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z") == 1000.0
    assert _latency_ms("2", "1") == 0.0
    assert _latency_ms("bad", "1") == 0.0


def test_span_payload_prefers_span_export_dict():
    span = _Span(export_value={"native": True})
    assert _span_payload(span) == {"native": True}


def test_span_payload_falls_back_to_attributes_and_data_export():
    span = _Span(
        span_type="function",
        payload={"name": "lookup"},
        error={"message": "boom"},
        export_raises=True,
    )
    payload = _span_payload(span)
    assert payload["id"] == "span-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["parent_id"] == "parent-1"
    assert payload["span_data"] == {"name": "lookup"}
    assert payload["error"] == {"message": "boom"}


def test_span_data_handles_missing_non_dict_and_failed_export():
    assert _span_data(types.SimpleNamespace()) == ("unknown", {})
    assert _span_data(_Span("agent", payload="not-a-dict")) == ("agent", {})
    assert _span_data(_Span("agent", data_raises=True)) == ("agent", {})


def test_trace_processor_forwards_only_while_active():
    target = Mock()
    processor = _OpenAITraceProcessor(target)
    trace = object()
    span = object()
    processor.on_trace_start(trace)
    processor.on_trace_end(trace)
    processor.on_span_start(span)
    processor.on_span_end(span)
    target.on_trace_start.assert_called_once_with(trace)
    target.on_trace_end.assert_called_once_with(trace)
    target.on_span_start.assert_called_once_with(span)
    target.on_span_end.assert_called_once_with(span)
    assert processor.force_flush() is None
    processor.shutdown()
    processor.on_trace_start(trace)
    processor.on_span_end(span)
    assert target.on_trace_start.call_count == 1
    assert target.on_span_end.call_count == 1


def test_collector_install_registers_processor(collector):
    add_trace_processor = Mock()
    fake_agents = types.ModuleType("agents")
    fake_agents.add_trace_processor = add_trace_processor
    with patch.dict(sys.modules, {"agents": fake_agents}):
        collector.install()
    add_trace_processor.assert_called_once_with(collector.processor)


def test_collector_install_wraps_missing_dependency(collector):
    with patch.dict(sys.modules, {"agents": None}):
        with pytest.raises(RuntimeError, match="openai-agents package"):
            collector.install()


def test_async_hooks_forward_all_lifecycle_events(collector):
    agent = _FakeAgent("planner")
    target = _FakeAgent("reviewer")
    tool = _FakeTool("lookup")
    context = types.SimpleNamespace(tool_call_id="c1", tool_arguments='{"id": "a1"}')
    response = types.SimpleNamespace(
        model="gpt-test",
        usage=types.SimpleNamespace(input_tokens=3, output_tokens=4),
    )

    async def invoke():
        await collector.hooks.on_agent_start(context, agent)
        await collector.hooks.on_llm_start(context, agent, "system", [{"role": "user"}])
        await collector.hooks.on_tool_start(context, agent, tool)
        await collector.hooks.on_tool_end(context, agent, tool, '{"ok": true}')
        await collector.hooks.on_llm_end(context, agent, response)
        await collector.hooks.on_handoff(context, agent, target)
        await collector.hooks.on_agent_end(context, target, '{"done": true}')

    asyncio.run(invoke())
    event_types = [event.event_type for event in collector.events]
    assert "openai_agents.agent_start" in event_types
    assert "openai_agents.llm_start" in event_types
    assert "openai_agents.tool_start" in event_types
    assert "openai_agents.tool_end" in event_types
    assert "openai_agents.llm_end" in event_types
    assert "openai_agents.handoff" in event_types
    assert "openai_agents.agent_end" in event_types
    assert collector.tool_calls[0].return_value == {"ok": True}
    assert collector.final_output == {"done": True}


def test_record_event_uses_trace_id_agent_and_explicit_metadata(collector):
    collector.trace_id = "trace-9"
    agent = _FakeAgent("planner")
    collector.record_event(
        "custom",
        agent=agent,
        payload={"x": 1},
        parent_event_id="parent",
        timestamp="now",
    )
    event = collector.events[0]
    assert event.event_type == "custom"
    assert event.timestamp == "now"
    assert event.agent_name == "planner"
    assert event.node_name == "planner"
    assert event.workflow_run_id == "trace-9"
    assert event.parent_event_id == "parent"
    assert event.payload == {"x": 1}


def test_ensure_agent_is_idempotent_and_records_string_description(collector):
    agent = _FakeAgent("planner", "agent-1", "instructions")
    first = collector.ensure_agent(agent)
    second = collector.ensure_agent(agent)
    assert first is second
    assert len(collector.agents) == 1
    assert first.agent_description == "instructions"
    assert first.role == "planner"


def test_ensure_agent_ignores_non_string_description(collector):
    agent = _FakeAgent(instructions={"dynamic": True})
    observed = collector.ensure_agent(agent)
    assert observed.agent_description is None


def test_agent_start_end_create_and_update_workflow_node(collector):
    agent = _FakeAgent("planner", "agent-1")
    collector.record_agent_start(agent)
    entered = collector.nodes["agent-1"].entered_at
    collector.record_agent_start(agent)
    assert collector.nodes["agent-1"].entered_at == entered
    collector.record_agent_end(agent, '{"should_act": true}')
    node = collector.nodes["agent-1"]
    assert node.node_type == "openai_agents.agent"
    assert node.agent_name == "planner"
    assert node.exited_at is not None
    assert collector.final_output == {"should_act": True}


def test_agent_end_without_start_still_creates_node(collector):
    agent = _FakeAgent("reviewer", "agent-2")
    collector.record_agent_end(agent, "plain")
    assert collector.nodes["agent-2"].entered_at is not None
    assert collector.nodes["agent-2"].exited_at is not None
    assert collector.final_output == "plain"


def test_handoff_records_agents_edge_and_event(collector):
    source = _FakeAgent("planner", "a1")
    target = _FakeAgent("reviewer", "a2")
    collector.record_handoff(source, target)
    handoff = collector.handoffs[0]
    assert handoff.from_executor_id == "a1"
    assert handoff.to_executor_id == "a2"
    assert handoff.payload_type == "openai_agents.handoff"
    assert len(collector.agents) == 2
    assert collector.events[-1].payload == {"from_agent": "planner", "to_agent": "reviewer"}


def test_tool_lifecycle_records_metadata_and_latency(collector):
    collector.trace_id = "trace-1"
    agent = _FakeAgent("planner", "a1")
    tool = _FakeTool("lookup")
    context = types.SimpleNamespace(tool_call_id="call-1", tool_arguments='{"id": "x"}')
    with patch("groundeval.framework_adapters.openai_agents_adapter.time.time", side_effect=[10.0, 10.25]):
        collector.record_tool_start(context, agent, tool)
        collector.record_tool_end(context, agent, tool, '[{"id": "x"}]')
    call = collector.tool_calls[0]
    assert call.tool_name == "lookup"
    assert call.arguments == {"id": "x"}
    assert call.return_value == [{"id": "x"}]
    assert call.latency_ms == 250.0
    assert call.agent_id == "a1"
    assert call.agent_name == "planner"
    assert call.node_name == "planner"
    assert call.workflow_run_id == "trace-1"
    assert call.parent_event_id == "call-1"
    assert collector.agents["a1"].tool_call_count == 1
    assert collector._pending_tools == {}


def test_tool_end_matches_pending_operation_without_call_id(collector):
    agent = _FakeAgent("planner", "a1")
    tool = _FakeTool("lookup")
    start_context = types.SimpleNamespace(tool_call_id="generated", tool_arguments='{"id": 1}')
    end_context = types.SimpleNamespace(tool_arguments='{"ignored": true}')
    collector.record_tool_start(start_context, agent, tool)
    collector.record_tool_end(end_context, agent, tool, "done")
    assert collector.tool_calls[0].arguments == {"id": 1}
    assert collector.tool_calls[0].parent_event_id == "generated"
    assert collector._pending_tools == {}


def test_tool_end_without_start_uses_context_and_zero_latency(collector):
    agent = _FakeAgent("planner", "a1")
    tool = _FakeTool("lookup")
    context = types.SimpleNamespace(tool_call_id="unknown", tool_arguments='"raw input"')
    collector.record_tool_end(context, agent, tool, "plain result")
    call = collector.tool_calls[0]
    assert call.arguments == {"raw": "raw input"}
    assert call.return_value == "plain result"
    assert call.latency_ms == 0.0
    assert call.parent_event_id == "unknown"


def test_tool_start_without_id_generates_identifier(collector):
    agent = _FakeAgent()
    with patch("groundeval.framework_adapters.openai_agents_adapter.uuid.uuid4") as uuid4:
        uuid4.return_value.hex = "generated-id"
        collector.record_tool_start(types.SimpleNamespace(tool_arguments=None), agent, _FakeTool())
    assert "generated-id" in collector._pending_tools
    assert collector.events[-1].parent_event_id == "generated-id"


def test_model_response_reads_usage_and_usage_data_fallback(collector):
    agent = _FakeAgent()
    response_one = types.SimpleNamespace(
        model="gpt-a",
        usage=types.SimpleNamespace(input_tokens="3", output_tokens="4"),
    )
    response_two = types.SimpleNamespace(
        model_name="gpt-b",
        usage=None,
        usage_data=types.SimpleNamespace(requests="5", output_tokens=6),
    )
    collector.record_model_response(agent, response_one)
    collector.record_model_response(agent, response_two)
    assert [(e.model_name, e.input_tokens, e.output_tokens) for e in collector.model_events] == [
        ("gpt-a", 3, 4),
        ("gpt-b", 5, 6),
    ]
    assert all(e.provider_name == "openai" for e in collector.model_events)


def test_trace_lifecycle_uses_defaults_when_metadata_missing(collector):
    collector.on_trace_start(types.SimpleNamespace())
    assert collector.trace_id == "run-1"
    assert collector.workflow_name == "OpenAI Agents workflow"
    collector.on_trace_end(types.SimpleNamespace())
    assert [e.event_type for e in collector.events] == [
        "openai_agents.trace_start",
        "openai_agents.trace_end",
    ]


def test_trace_lifecycle_records_explicit_identity(collector):
    trace = types.SimpleNamespace(trace_id="trace-7", name="Workflow")
    collector.on_trace_start(trace)
    collector.on_trace_end(trace)
    assert collector.trace_id == "trace-7"
    assert collector.workflow_name == "Workflow"
    assert collector.events[0].workflow_run_id == "trace-7"


def test_span_start_records_type_payload_parent_and_timestamp(collector):
    span = _Span("agent", {"name": "planner"})
    collector.on_span_start(span)
    event = collector.events[0]
    assert event.event_type == "openai_agents.span_start.agent"
    assert event.timestamp == span.started_at
    assert event.parent_event_id == "parent-1"
    assert event.payload["span_data"] == {"name": "planner"}


def test_agent_span_registers_agent_inventory(collector):
    collector.on_span_end(_Span("agent", {"name": "reviewer"}))
    assert collector.agents["openai-agents:reviewer"].agent_name == "reviewer"
    collector.on_span_end(_Span("agent", {"name": "reviewer"}))
    assert len(collector.agents) == 1


def test_agent_span_with_missing_name_uses_unknown(collector):
    collector.on_span_end(_Span("agent", {}))
    assert "openai-agents:unknown" in collector.agents


def test_handoff_span_records_edge_and_deduplicates_hook_edge(collector):
    source = _FakeAgent("planner")
    target = _FakeAgent("reviewer")
    collector.record_handoff(source, target)
    collector.on_span_end(_Span("handoff", {"from_agent": "planner", "to_agent": "reviewer"}))
    assert len(collector.handoffs) == 1


def test_handoff_span_records_new_edge_and_ignores_incomplete_data(collector):
    collector.on_span_end(_Span("handoff", {"from_agent": "planner", "to_agent": "reviewer"}))
    collector.on_span_end(_Span("handoff", {"from_agent": "planner"}))
    assert len(collector.handoffs) == 1
    assert collector.handoffs[0].payload_type == "openai_agents.handoff_span"


def test_function_span_records_tool_call_with_parsed_data(collector):
    collector.trace_id = "trace-1"
    span = _Span(
        "function",
        {"name": "lookup", "input": '{"id": "a1"}', "output": '{"ok": true}'},
        started_at="1",
        ended_at="1.5",
    )
    collector.on_span_end(span)
    call = collector.tool_calls[0]
    assert call.tool_name == "lookup"
    assert call.arguments == {"id": "a1"}
    assert call.return_value == {"ok": True}
    assert call.latency_ms == 500.0
    assert call.workflow_run_id == "trace-1"
    assert call.parent_event_id == "parent-1"


def test_function_span_wraps_scalar_input_and_uses_defaults(collector):
    collector.on_span_end(_Span("function", {"input": '"hello"'}))
    call = collector.tool_calls[0]
    assert call.tool_name == "function"
    assert call.arguments == {"raw": "hello"}
    assert call.return_value is None


def test_function_span_is_deduplicated_after_hook_tool_call(collector):
    agent = _FakeAgent()
    tool = _FakeTool("lookup")
    context = types.SimpleNamespace(tool_call_id="c1", tool_arguments='{"id": "a1"}')
    collector.record_tool_start(context, agent, tool)
    collector.record_tool_end(context, agent, tool, '{"ok": true}')
    collector.on_span_end(
        _Span("function", {"name": "lookup", "input": '{"id": "a1"}', "output": '{"ok": true}'})
    )
    assert len(collector.tool_calls) == 1


def test_generation_span_records_model_usage_with_aliases(collector):
    collector.on_span_end(
        _Span(
            "generation",
            {
                "model": "gpt-test",
                "usage": {"prompt_tokens": "8", "completion_tokens": "9"},
            },
        )
    )
    event = collector.model_events[0]
    assert event.model_name == "gpt-test"
    assert event.provider_name == "openai"
    assert event.input_tokens == 8
    assert event.output_tokens == 9


def test_generation_span_tolerates_non_mapping_usage(collector):
    collector.on_span_end(_Span("generation", {"model": "gpt-test", "usage": "unknown"}))
    event = collector.model_events[0]
    assert event.input_tokens is None
    assert event.output_tokens is None


def test_span_error_dict_and_string_are_recorded(collector):
    collector.on_span_end(_Span("function", {}, error={"message": "tool failed", "code": 500}))
    collector.on_span_end(_Span("generation", {}, error="model failed"))
    assert collector.errors[0].error_type == "OpenAIAgentsFunctionSpanError"
    assert collector.errors[0].message == "tool failed"
    assert '"code": 500' in collector.errors[0].traceback
    assert collector.errors[1].message == "model failed"


def test_record_exception_preserves_type_message_and_traceback(collector):
    try:
        raise ValueError("boom")
    except ValueError as exc:
        collector.record_exception(exc)
    error = collector.errors[0]
    assert error.error_type == "ValueError"
    assert error.message == "boom"
    assert "ValueError: boom" in error.traceback


def test_to_rich_run_builds_workflow_latency_and_capabilities(collector):
    collector.started_at = 1.0
    collector.completed_at = 2.25
    collector.trace_id = "trace-1"
    collector.workflow_name = "Workflow"
    agent = _FakeAgent("planner", "a1")
    collector.record_agent_start(agent)
    collector.record_agent_end(agent, {"ok": True})
    collector.record_model_response(agent, types.SimpleNamespace(model="gpt", usage=None, usage_data=None))
    run = collector.to_rich_run()
    assert run.run_id == "trace-1"
    assert run.framework == "openai_agents"
    assert run.total_latency_ms == 1250.0
    assert run.workflow.workflow_id == "trace-1"
    assert run.workflow.workflow_name == "Workflow"
    assert run.workflow.node_count == 1
    assert run.final_output == {"ok": True}
    assert run.capabilities["native_tracing"] is True
    assert run.capabilities["agent_turns"] is True
    assert run.capabilities["workflow_nodes"] is True
    assert run.capabilities["model_calls"] is True
    assert run.capabilities["tool_calls"] is False


def test_to_rich_run_defaults_latency_and_workflow_identity(collector):
    run = collector.to_rich_run()
    assert run.run_id == "run-1"
    assert run.total_latency_ms == 0.0
    assert run.workflow.workflow_name == "OpenAI Agents workflow"
    assert run.workflow.handoff_count == 0
    assert run.capabilities["approvals"] is False
    assert run.capabilities["checkpoints"] is False


def test_entry_parts_supports_all_declared_entry_shapes(monkeypatch):
    agent = _FakeAgent()
    wrapped = OpenAIAgentsEntry(agent, "input", "context", "config")
    assert _entry_parts(wrapped) == (agent, "input", "context", "config")
    assert _entry_parts((agent, "tuple input")) == (agent, "tuple input", None, None)
    assert _entry_parts({"agent": agent, "input": "dict input", "context": 1, "run_config": 2}) == (
        agent,
        "dict input",
        1,
        2,
    )
    monkeypatch.setenv("GROUNDEVAL_AGENT_INPUT", "environment input")
    assert _entry_parts(agent) == (agent, "environment input", None, None)
    agent.groundeval_input = "attribute input"
    assert _entry_parts(agent) == (agent, "attribute input", None, None)
    nested = types.SimpleNamespace(agent=agent, input="nested input", context="ctx", run_config="cfg")
    assert _entry_parts(nested) == (agent, "nested input", "ctx", "cfg")
    assert _entry_parts(object()) == (None, None, None, None)


def test_run_entry_custom_sync_receives_hooks_and_limit(collector):
    class Entry:
        def run_groundeval(self, **kwargs):
            self.kwargs = kwargs
            return {"ok": True}

    entry = Entry()
    assert _run_entry(entry, collector, 7) == {"ok": True}
    assert entry.kwargs == {"hooks": collector.hooks, "max_turns": 7}


def test_run_entry_custom_async_is_awaited(collector):
    class Entry:
        async def run_groundeval(self, **kwargs):
            return kwargs["max_turns"]

    assert _run_entry(Entry(), collector, 6) == 6


def test_run_entry_native_agent_calls_runner_with_optional_values(collector):
    agent = _FakeAgent()
    runner = types.SimpleNamespace(run_sync=Mock(return_value={"ok": True}))
    fake_agents = types.ModuleType("agents")
    fake_agents.Runner = runner
    entry = OpenAIAgentsEntry(agent, "hello", context="ctx", run_config="cfg")
    with patch.dict(sys.modules, {"agents": fake_agents}):
        result = _run_entry(entry, collector, 9)
    assert result == {"ok": True}
    runner.run_sync.assert_called_once_with(
        agent,
        "hello",
        max_turns=9,
        hooks=collector.hooks,
        context="ctx",
        run_config="cfg",
    )


def test_run_entry_native_agent_omits_absent_optional_values(collector):
    agent = _FakeAgent()
    runner = types.SimpleNamespace(run_sync=Mock(return_value="done"))
    fake_agents = types.ModuleType("agents")
    fake_agents.Runner = runner
    with patch.dict(sys.modules, {"agents": fake_agents}):
        _run_entry((agent, "hello"), collector, 3)
    runner.run_sync.assert_called_once_with(
        agent,
        "hello",
        max_turns=3,
        hooks=collector.hooks,
    )


def test_run_entry_native_agent_wraps_missing_dependency(collector):
    with patch.dict(sys.modules, {"agents": None}):
        with pytest.raises(RuntimeError, match="openai-agents package"):
            _run_entry(_FakeAgent(), collector, 3)


def test_run_entry_supports_sync_and_async_run_methods(collector):
    class SyncEntry:
        def run(self):
            return "sync"

    class AsyncEntry:
        async def run(self):
            return "async"

    assert _run_entry(SyncEntry(), collector, 1) == "sync"
    assert _run_entry(AsyncEntry(), collector, 1) == "async"


def test_run_entry_supports_sync_and_async_callables(collector):
    class SyncCallable:
        def __call__(self):
            return "sync"

    class AsyncCallable:
        async def __call__(self):
            return "async"

    assert _run_entry(SyncCallable(), collector, 1) == "sync"
    assert _run_entry(AsyncCallable(), collector, 1) == "async"


def test_run_entry_rejects_unsupported_entry(collector):
    with pytest.raises(TypeError, match="expected an Agent"):
        _run_entry(object(), collector, 1)


def test_final_output_prefers_result_attribute_and_parses_json():
    assert _final_output(types.SimpleNamespace(final_output='{"ok": true}')) == {"ok": True}
    assert _final_output({"ok": True}) == {"ok": True}
    assert _final_output(types.SimpleNamespace(final_output=None)) is None


def test_observer_load_agent_delegates_to_loader():
    observer = OpenAIAgentsObserver()
    with patch(
        "groundeval.framework_adapters.openai_agents_adapter._load_openai_agents_entry",
        return_value="entry",
    ) as loader:
        assert observer.load_agent("pkg.Entry") == "entry"
    loader.assert_called_once_with("pkg.Entry")


def test_observer_instrument_agent_installs_collector_and_wraps_entry():
    observer = OpenAIAgentsObserver()
    entry = object()
    recording = RecordingRuntime()
    with patch.object(_OpenAIAgentsCollector, "install") as install:
        wrapped = observer.instrument_agent(entry, recording)
    assert isinstance(wrapped, _InstrumentedOpenAIAgentsEntry)
    assert wrapped.entry is entry
    assert wrapped._groundeval_recording is recording
    assert wrapped._groundeval_max_steps == 10
    assert wrapped._groundeval_openai_agents_collector.agent_class == "builtins.object"
    install.assert_called_once()


def test_observer_execute_success_sets_rich_run_and_recording():
    observer = OpenAIAgentsObserver()
    collector = _OpenAIAgentsCollector("run-1", "pkg.Agent")
    recording = RecordingRuntime()
    wrapped = _InstrumentedOpenAIAgentsEntry(object(), collector, recording)
    wrapped._groundeval_max_steps = 4

    def run_entry(entry, active_collector, max_steps):
        assert entry is wrapped.entry
        assert active_collector is collector
        assert max_steps == 4
        active_collector.tool_calls.append(
            ObservedToolCall(
                tool_name="lookup",
                arguments={"id": "a1"},
                return_value={"ok": True},
                latency_ms=2.0,
                agent_id="a1",
                agent_name="planner",
                node_name="planner",
                workflow_run_id="run-1",
                parent_event_id="call-1",
            )
        )
        return types.SimpleNamespace(final_output='{"done": true}')

    with patch("groundeval.framework_adapters.openai_agents_adapter._run_entry", side_effect=run_entry):
        result = observer.execute_agent(wrapped)
    assert result.final_output == '{"done": true}'
    assert collector.final_output == {"done": True}
    assert collector.started_at is not None
    assert collector.completed_at is not None
    assert collector.processor.active is False
    assert wrapped._groundeval_framework_observed_run.final_output == {"done": True}
    assert len(recording.call_log) == 1
    assert recording.call_log[0].agent_id == "a1"
    assert recording.call_log[0].parent_event_id == "call-1"


def test_observer_execute_failure_records_error_finalizes_and_reraises():
    observer = OpenAIAgentsObserver()
    collector = _OpenAIAgentsCollector("run-1", "pkg.Agent")
    wrapped = _InstrumentedOpenAIAgentsEntry(object(), collector, RecordingRuntime())
    with patch(
        "groundeval.framework_adapters.openai_agents_adapter._run_entry",
        side_effect=ValueError("boom"),
    ):
        with pytest.raises(ValueError, match="boom"):
            observer.execute_agent(wrapped)
    assert collector.completed_at is not None
    assert collector.processor.active is False
    assert collector.errors[0].error_type == "ValueError"
    assert collector.errors[0].message == "boom"
    assert wrapped._groundeval_framework_observed_run.errors[0].message == "boom"


def test_observer_set_max_steps_accepts_positive_and_rejects_nonpositive():
    observer = OpenAIAgentsObserver()
    wrapped = types.SimpleNamespace()
    observer.set_max_steps(wrapped, 7)
    assert wrapped._groundeval_max_steps == 7
    with pytest.raises(ValueError, match="greater than zero"):
        observer.set_max_steps(wrapped, 0)
    with pytest.raises(ValueError, match="greater than zero"):
        observer.set_max_steps(wrapped, -1)


def test_generate_report_contains_all_sections_and_values():
    run = ObservedRun(
        run_id="r1",
        framework="openai_agents",
        agent_class="pkg.Agent",
        total_latency_ms=123.0,
        agents=[ObservedAgent("a1", "planner", tool_call_count=1)],
        tool_calls=[
            ObservedToolCall(
                "lookup",
                {"id": "a1"},
                {"ok": True},
                5.0,
                agent_name="planner",
            )
        ],
        workflow=ObservedWorkflow(
            workflow_id="wf1",
            handoff_count=1,
            handoffs=[ObservedHandoff("a1", "a2", "now", "handoff")],
        ),
        model_events=[ObservedModelEvent("model.call.completed", model_name="gpt")],
        final_output={"done": True},
        errors=[ObservedError("ValueError", "boom", "later")],
        capabilities={"tool_calls": True, "handoffs": True},
    )
    report = generate_openai_agents_report(run)
    assert "GroundEval OpenAI Agents Observation Report" in report
    assert "## Summary" in report
    assert "## Capabilities" in report
    assert "## Agent Inventory" in report
    assert "## Tool Calls" in report
    assert "## Handoffs" in report
    assert "## Errors" in report
    assert "## Final Output" in report
    assert "planner" in report
    assert "lookup" in report
    assert "boom" in report
    assert '"done": true' in report


def test_generate_report_handles_missing_workflow_and_empty_collections():
    run = ObservedRun(
        run_id="r1",
        framework="openai_agents",
        agent_class="pkg.Agent",
        workflow=None,
        final_output=None,
        capabilities={},
    )
    report = generate_openai_agents_report(run)
    assert "Handoffs recorded: 0" in report
    assert "Agents observed: 0" in report
    assert "Tool calls recorded: 0" in report
    assert "null" in report
