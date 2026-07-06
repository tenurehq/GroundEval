import types
from unittest.mock import patch

import pytest

from groundeval.framework_adapters.maf_adapter import (
    MafObserver,
    _GroundEvalSpanExporter,
    _await_if_needed,
    _build_maf_eval_agent_fn,
    _collect_async_iter,
    _enable_maf_instrumentation,
    _extract_final_output,
    _extract_tool_args,
    _extract_tool_name,
    _extract_tool_result,
    _first,
    _is_agent_span,
    _is_chat_span,
    _is_edge_group_span,
    _is_executor_span,
    _is_message_send_span,
    _is_tool_span,
    _is_workflow_build_span,
    _is_workflow_run_span,
    _jsonish,
    _load_maf_agent,
    _parse_jsonish,
    _run_maf_entity,
    _safe_span_ids,
    _span_attrs,
    _span_latency_ms,
    _span_name,
    _span_time,
    _tool_agent_identity,
    maf_spans_to_observed_run,
    generate_maf_report,
)


class _FactoryModule:
    pass


def test_load_maf_agent_function_factory():
    mod = _FactoryModule()
    mod.make_agent = lambda: "agent"
    with patch("importlib.import_module", return_value=mod):
        assert _load_maf_agent("x.make_agent") == "agent"


def test_load_maf_agent_class():
    class Agent:
        pass

    mod = _FactoryModule()
    mod.Agent = Agent
    with patch("importlib.import_module", return_value=mod):
        out = _load_maf_agent("x.Agent")
    assert isinstance(out, Agent)


def test_load_maf_agent_object():
    mod = _FactoryModule()
    sentinel = object()
    mod.value = sentinel
    with patch("importlib.import_module", return_value=mod):
        assert _load_maf_agent("x.value") is sentinel


def test_span_exporter_collects_and_reports_success():
    exp = _GroundEvalSpanExporter()
    result = exp.export([1, 2, 3])
    assert exp.spans == [1, 2, 3]
    assert exp.force_flush() is True
    assert exp.shutdown() is None
    assert result is not None


def test_enable_maf_instrumentation_fallback_paths():
    fake_mod = types.SimpleNamespace(
        enable_instrumentation=lambda enable_sensitive_data=True: None
    )
    with patch.dict("sys.modules", {"agent_framework.observability": fake_mod}):
        _enable_maf_instrumentation()


@pytest.mark.anyio
async def test_await_if_needed():
    async def coro():
        return 1

    assert await _await_if_needed(coro()) == 1
    assert await _await_if_needed(2) == 2


@pytest.mark.anyio
async def test_collect_async_iter():
    class Update:
        def __init__(self, text):
            self.text = text

    class Stream:
        def __aiter__(self):
            async def gen():
                yield Update("a")
                yield Update("b")

            return gen()

    assert await _collect_async_iter(Stream()) == "ab"
    assert await _collect_async_iter({"x": 1}) == {"x": 1}


def test_run_maf_entity_with_run_method():
    class Agent:
        def run(self):
            return {"ok": True}

    assert _run_maf_entity(Agent()) == {"ok": True}


def test_run_maf_entity_with_callable():
    class Agent:
        def __call__(self):
            return {"ok": True}

    assert _run_maf_entity(Agent()) == {"ok": True}


def test_run_maf_entity_invalid_type():
    with pytest.raises(TypeError, match="expected an object with .run"):
        _run_maf_entity(object())


def test_jsonish_and_parse_jsonish():
    assert _jsonish({"a": 1}) == {"a": 1}
    assert _parse_jsonish('{"a": 1}') == {"a": 1}
    assert _parse_jsonish("plain text") == "plain text"


def test_span_helpers():
    span = types.SimpleNamespace(
        name="execute_tool fetch_customer",
        attributes={"a": 1},
        start_time=1_000_000_000,
        end_time=2_000_000_000,
    )
    assert _span_name(span) == "execute_tool fetch_customer"
    assert _span_attrs(span) == {"a": 1}
    assert _span_time(1_000_000_000) == "1.0"
    assert _span_latency_ms(span) == 1000.0


def test_first_and_extractors():
    attrs = {
        "gen_ai.tool.name": "fetch_customer",
        "gen_ai.tool.call.arguments": '{"artifact_id": "a1"}',
        "gen_ai.tool.call.result": '{"id": "a1"}',
    }
    assert _first(attrs, ["x", "gen_ai.tool.name"]) == "fetch_customer"
    assert _extract_tool_name("execute_tool something", attrs) == "fetch_customer"
    assert _extract_tool_args(attrs) == {"artifact_id": "a1"}
    assert _extract_tool_result(attrs) == {"id": "a1"}


def test_span_classifiers():
    assert _is_tool_span("execute_tool x", {"gen_ai.operation.name": ""}) is True
    assert _is_agent_span("invoke_agent planner", {"gen_ai.operation.name": ""}) is True
    assert _is_chat_span("chat model", {"gen_ai.operation.name": ""}) is True
    assert _is_workflow_build_span("workflow.build") is True
    assert _is_workflow_run_span("workflow.run") is True
    assert _is_executor_span("executor.process.node") is True
    assert _is_edge_group_span("edge_group.process.node") is True
    assert _is_message_send_span("message.send") is True


def test_safe_span_ids():
    class Ctx:
        trace_id = 10
        span_id = 11

    span = types.SimpleNamespace(
        get_span_context=lambda: Ctx(),
        parent=types.SimpleNamespace(span_id=12),
    )
    trace_id, span_id, parent_id = _safe_span_ids(span)
    assert trace_id.startswith("0x")
    assert span_id.startswith("0x")
    assert parent_id.startswith("0x")


def test_extract_final_output_fallbacks():
    assert _extract_final_output({"ok": True}, []) == {"ok": True}

    class Raw:
        raw = '{"ok": true}'

    raw = Raw()
    assert _extract_final_output(raw, []) == str(raw)

    class Text:
        text = '{"ok": true}'

    text = Text()
    assert _extract_final_output(text, []) == str(text)

    class Dump:
        def model_dump(self):
            return {"ok": True}

    dump = Dump()
    assert _extract_final_output(dump, []) == {"ok": True}


def test_maf_spans_to_observed_run_basic():
    class Status:
        status_code = "OK"
        description = ""

    class Span:
        def __init__(self, name, attrs):
            self.name = name
            self.attributes = attrs
            self.start_time = 1_000_000_000
            self.end_time = 2_000_000_000
            self.status = Status()
            self.parent = None
            self.events = []

        def get_span_context(self):
            return types.SimpleNamespace(trace_id=1, span_id=2)

    spans = [
        Span(
            "invoke_agent planner",
            {"gen_ai.agent.id": "agent-1", "gen_ai.agent.name": "planner"},
        ),
        Span(
            "execute_tool fetch_customer",
            {
                "gen_ai.tool.name": "fetch_customer",
                "gen_ai.tool.call.arguments": '{"artifact_id": "a1"}',
                "gen_ai.tool.call.result": '{"id": "a1", "subsystem": "crm"}',
                "gen_ai.agent.id": "agent-1",
                "gen_ai.agent.name": "planner",
                "executor.id": "node-1",
            },
        ),
        Span(
            "chat gpt-4o",
            {
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.system": "openai",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
        ),
        Span("workflow.build", {"workflow.id": "wf-1", "workflow.name": "Main"}),
        Span("workflow.run", {"workflow.id": "wf-1"}),
        Span(
            "message.send",
            {
                "message.source_id": "agent-1",
                "message.target_id": "agent-2",
                "message.type": "handoff",
            },
        ),
    ]

    observed = maf_spans_to_observed_run(
        spans=spans,
        raw_result={"should_act": True},
        agent_class="pkg.Agent",
        run_id="run-1",
        started_at=1.0,
        completed_at=2.0,
    )

    assert observed.framework == "maf"
    assert len(observed.tool_calls) == 1
    assert observed.tool_calls[0].tool_name == "fetch_customer"
    assert len(observed.agents) == 1
    assert observed.workflow.workflow_id == "wf-1"
    assert observed.workflow.handoff_count == 1
    assert len(observed.model_events) == 1
    assert observed.final_output == {"should_act": True}


def test_maf_observer_execute_agent_records_framework_run():
    observer = MafObserver()

    class Agent:
        pass

    agent = Agent()

    from groundeval.observe import RecordingRuntime

    recording = RecordingRuntime()
    agent._groundeval_recording = recording

    fake_run = types.SimpleNamespace(
        run_id="run-1",
        framework="maf",
        agent_class="pkg.Agent",
        tool_calls=[],
        final_output={"ok": True},
        total_latency_ms=1.0,
        to_dict=lambda: {
            "run_id": "run-1",
            "framework": "maf",
            "agent_class": "pkg.Agent",
            "tool_calls": [],
            "final_output": {"ok": True},
            "total_latency_ms": 1.0,
        },
    )

    with patch(
        "groundeval.framework_adapters.maf_adapter._install_in_memory_otel_exporter"
    ) as p1:
        with patch(
            "groundeval.framework_adapters.maf_adapter._enable_maf_instrumentation"
        ):
            with patch(
                "groundeval.framework_adapters.maf_adapter._run_maf_entity",
                return_value={"ok": True},
            ):
                with patch(
                    "groundeval.framework_adapters.maf_adapter.maf_spans_to_observed_run",
                    return_value=fake_run,
                ):
                    p1.return_value = types.SimpleNamespace(spans=[])
                    out = observer.execute_agent(agent)

    assert out == {"ok": True}
    assert hasattr(agent, "_groundeval_framework_observed_run")


def test_build_maf_eval_agent_fn_raises():
    with pytest.raises(RuntimeError, match="observe --score"):
        _build_maf_eval_agent_fn("pkg.Agent")


def test_generate_maf_report_contains_key_sections():
    from groundeval.framework_adapters.framework_observation import (
        ObservedAgent,
        ObservedHandoff,
        ObservedModelEvent,
        ObservedRun,
        ObservedWorkflow,
        ObservedWorkflowNode,
    )
    from groundeval.observe import ObservedToolCall

    run = ObservedRun(
        run_id="r1",
        framework="maf",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                "fetch_customer", {"artifact_id": "a1"}, {"id": "a1"}, 10.0
            )
        ],
        agents=[
            ObservedAgent(agent_id="agent-1", agent_name="planner", tool_call_count=1)
        ],
        workflow=ObservedWorkflow(
            workflow_id="wf-1",
            workflow_name="Main",
            node_count=1,
            nodes=[ObservedWorkflowNode(node_id="node-1", node_type="executor")],
            handoff_count=1,
            handoffs=[ObservedHandoff(from_executor_id="a", to_executor_id="b")],
        ),
        model_events=[
            ObservedModelEvent(event_type="model.call.completed", model_name="gpt-4o")
        ],
        final_output={"should_act": True},
        capabilities={"tool_calls": True},
    )

    report = generate_maf_report(run)
    assert "GroundEval MAF Observation Report" in report
    assert "Workflow Summary" in report
    assert "Final Output" in report


def test_extract_tool_args_non_dict_returns_raw_wrapper():
    attrs = {"arguments": '"hello"'}
    out = _extract_tool_args(attrs)
    assert out == {"raw": "hello"}


def test_extract_tool_args_none_returns_empty_dict():
    out = _extract_tool_args({})
    assert out == {}


def test_extract_tool_result_none_returns_none():
    out = _extract_tool_result({})
    assert out is None


def test_extract_tool_name_falls_back_to_prefix_parsing():
    out = _extract_tool_name("execute_tool fetch_orders", {})
    assert out == "fetch_orders"


def test_extract_tool_name_returns_original_name_when_no_signal():
    out = _extract_tool_name("mystery_span", {})
    assert out == "mystery_span"


def test_extract_final_output_from_span_response_fallback():
    class Status:
        status_code = "OK"
        description = ""

    class Span:
        def __init__(self):
            self.name = "chat gpt"
            self.attributes = {"gen_ai.response.text": '{"ok": true}'}
            self.start_time = 1
            self.end_time = 2
            self.status = Status()
            self.parent = None
            self.events = []

        def get_span_context(self):
            return types.SimpleNamespace(trace_id=1, span_id=2)

    out = _extract_final_output(None, [Span()])
    assert out == {"ok": True}


def test_extract_final_output_text_non_json_returns_string_fallback():
    class Text:
        text = "plain response"

    text = Text()
    out = _extract_final_output(text, [])
    assert out == str(text)


def test_safe_span_ids_handles_missing_context():
    span = types.SimpleNamespace(parent=None)
    trace_id, span_id, parent_id = _safe_span_ids(span)
    assert trace_id is None
    assert span_id is None
    assert parent_id is None


def test_is_tool_span_detects_function_invocation():
    assert _is_tool_span("some function.invocation call", {}) is True


def test_tool_agent_identity_backfills_name_from_agents():
    agents = {
        "agent-1": types.SimpleNamespace(agent_name="planner"),
    }
    attrs = {"gen_ai.agent.id": "agent-1"}
    agent_id, agent_name = _tool_agent_identity(attrs, agents)
    assert agent_id == "agent-1"
    assert agent_name == "planner"


def test_tool_agent_identity_backfills_id_from_name():
    agents = {
        "agent-1": types.SimpleNamespace(agent_name="planner"),
    }
    attrs = {"gen_ai.agent.name": "planner"}
    agent_id, agent_name = _tool_agent_identity(attrs, agents)
    assert agent_id == "agent-1"
    assert agent_name == "planner"


def test_maf_spans_to_observed_run_records_error_span():
    class Status:
        status_code = "ERROR"
        description = "boom"

    class Span:
        def __init__(self):
            self.name = "workflow.run"
            self.attributes = {"workflow.id": "wf-1"}
            self.start_time = 1_000_000_000
            self.end_time = 2_000_000_000
            self.status = Status()
            self.parent = None
            self.events = []

        def get_span_context(self):
            return types.SimpleNamespace(trace_id=1, span_id=2)

    observed = maf_spans_to_observed_run(
        spans=[Span()],
        raw_result={"ok": True},
        agent_class="pkg.Agent",
        run_id="run-1",
        started_at=1.0,
        completed_at=2.0,
    )

    assert len(observed.errors) == 1
    assert observed.errors[0].message == "boom"


def test_maf_spans_to_observed_run_records_span_events():
    class Status:
        status_code = "OK"
        description = ""

    class SpanEvent:
        def __init__(self):
            self.name = "partial"
            self.timestamp = 1_500_000_000
            self.attributes = {"k": "v"}

    class Span:
        def __init__(self):
            self.name = "workflow.run"
            self.attributes = {"workflow.id": "wf-1"}
            self.start_time = 1_000_000_000
            self.end_time = 2_000_000_000
            self.status = Status()
            self.parent = None
            self.events = [SpanEvent()]

        def get_span_context(self):
            return types.SimpleNamespace(trace_id=1, span_id=2)

    observed = maf_spans_to_observed_run(
        spans=[Span()],
        raw_result={"ok": True},
        agent_class="pkg.Agent",
        run_id="run-1",
        started_at=1.0,
        completed_at=2.0,
    )

    event_types = [e.event_type for e in observed.events]
    assert "otel.span" in event_types
    assert "otel.event.partial" in event_types


def test_maf_spans_to_observed_run_tool_result_list_is_preserved():
    class Status:
        status_code = "OK"
        description = ""

    class Span:
        def __init__(self):
            self.name = "execute_tool fetch_customer"
            self.attributes = {
                "gen_ai.tool.name": "fetch_customer",
                "gen_ai.tool.call.arguments": '{"artifact_id": "a1"}',
                "gen_ai.tool.call.result": '[{"id": "a1", "subsystem": "crm"}]',
                "gen_ai.agent.id": "agent-1",
                "gen_ai.agent.name": "planner",
            }
            self.start_time = 1_000_000_000
            self.end_time = 2_000_000_000
            self.status = Status()
            self.parent = None
            self.events = []

        def get_span_context(self):
            return types.SimpleNamespace(trace_id=1, span_id=2)

    observed = maf_spans_to_observed_run(
        spans=[Span()],
        raw_result={"ok": True},
        agent_class="pkg.Agent",
        run_id="run-1",
        started_at=1.0,
        completed_at=2.0,
    )

    assert len(observed.tool_calls) == 1
    assert observed.tool_calls[0].return_value == [{"id": "a1", "subsystem": "crm"}]


def test_generate_maf_report_with_errors_and_events():
    from groundeval.framework_adapters.framework_observation import (
        ObservedError,
        ObservedEvent,
        ObservedRun,
    )

    run = ObservedRun(
        run_id="r1",
        framework="maf",
        agent_class="pkg.Agent",
        tool_calls=[],
        events=[
            ObservedEvent(event_type="otel.span", timestamp="1.0", payload={"a": 1})
        ],
        final_output={"ok": True},
        errors=[ObservedError(error_type="X", message="boom", timestamp="1.1")],
        capabilities={},
    )

    report = generate_maf_report(run)
    assert "Errors" in report
    assert "Event Timeline" in report
    assert "boom" in report
