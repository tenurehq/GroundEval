import types
from unittest.mock import patch

import pytest

from groundeval.framework_adapters.langgraph_adapter import (
    LangGraphObserver,
    _GroundEvalLangChainCallbackHandler,
    _LangGraphEventCollector,
    _coerce_int,
    _extract_node_name_from_ns,
    _jsonish,
    _load_langgraph,
    _now_str,
    _parse_jsonish,
    _require_python_311,
    _summarize_value,
    generate_langgraph_report,
)
from groundeval.framework_adapters.framework_observation import (
    ObservedAgent,
    ObservedHandoff,
    ObservedRun as FrameworkObservedRun,
)
from groundeval.observe import ObservedToolCall, RecordingRuntime


class _FactoryModule:
    pass


class _FakeCompiledGraph:
    def __init__(self):
        self.stream_calls = []
        self.astream_calls = []

    def stream(self, *args, **kwargs):
        self.stream_calls.append((args, kwargs))
        return iter([])

    async def astream(self, *args, **kwargs):
        self.astream_calls.append((args, kwargs))
        if False:
            yield None


class _NoStreamGraph:
    pass


class _ModelDumpObj:
    def model_dump(self):
        return {"a": 1}


class _DictObj:
    def dict(self):
        return {"b": 2}


class _ToDictObj:
    def to_dict(self):
        return {"c": 3}


class _BadJsonObj:
    pass


class _FakeStaticEdge:
    def __init__(self, source, target):
        self.source = source
        self.target = target


class _FakeStaticGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes if nodes is not None else {}
        self.edges = edges if edges is not None else []


class _FakeGraph:
    def __init__(
        self,
        *,
        stream_chunks=None,
        astream_chunks=None,
        static_graph=None,
        subgraphs=None,
    ):
        self._stream_chunks = stream_chunks or []
        self._astream_chunks = astream_chunks or []
        self._static_graph = (
            static_graph if static_graph is not None else _FakeStaticGraph()
        )
        self._subgraphs = subgraphs if subgraphs is not None else []

    def get_graph(self):
        return self._static_graph

    def get_subgraphs(self):
        return self._subgraphs

    def stream(self, *args, **kwargs):
        for chunk in self._stream_chunks:
            yield chunk

    async def astream(self, *args, **kwargs):
        for chunk in self._astream_chunks:
            yield chunk


class _GraphFactory:
    def __call__(self):
        return _FakeCompiledGraph()


class _GraphClass:
    def __init__(self):
        self.inner = _FakeCompiledGraph()

    def compile(self):
        return self.inner


class _NodeWithId:
    def __init__(self, node_id):
        self.id = node_id


class _ResponseWithLLMOutput:
    def __init__(self):
        self.llm_output = {
            "model_name": "gpt-test",
            "provider": "openai",
            "token_usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
        msg = types.SimpleNamespace(
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 5, "output_tokens": 6},
            tool_calls=[{"name": "t1"}],
        )
        generation = types.SimpleNamespace(message=msg)
        self.generations = [[generation]]


class _CallbackError(Exception):
    pass


@pytest.fixture
def collector():
    graph = _FakeGraph()
    return _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )


def test_require_python_311_passes_for_newer_python():
    with patch(
        "groundeval.framework_adapters.langgraph_adapter.sys.version_info", (3, 11, 0)
    ):
        _require_python_311()


def test_require_python_311_raises_for_older_python():
    with patch(
        "groundeval.framework_adapters.langgraph_adapter.sys.version_info", (3, 10, 9)
    ):
        with pytest.raises(RuntimeError, match=r"Python 3.11\+"):
            _require_python_311()


def test_load_langgraph_from_function_factory():
    mod = _FactoryModule()
    mod.make_graph = lambda: _FakeCompiledGraph()
    with patch("importlib.import_module", return_value=mod):
        graph = _load_langgraph("x.make_graph")
    assert isinstance(graph, _FakeCompiledGraph)


def test_load_langgraph_from_class_with_compile():
    mod = _FactoryModule()
    mod.Graph = _GraphClass
    with patch("importlib.import_module", return_value=mod):
        graph = _load_langgraph("x.Graph")
    assert isinstance(graph, _FakeCompiledGraph)


def test_load_langgraph_existing_object_with_stream():
    mod = _FactoryModule()
    obj = _FakeCompiledGraph()
    mod.graph = obj
    with patch("importlib.import_module", return_value=mod):
        graph = _load_langgraph("x.graph")
    assert graph is obj


def test_load_langgraph_raises_when_missing_stream_interfaces():
    mod = _FactoryModule()
    mod.bad = _NoStreamGraph()
    with patch("importlib.import_module", return_value=mod):
        with pytest.raises(TypeError, match="compiled graph"):
            _load_langgraph("x.bad")


def test_jsonish_handles_primitives_and_model_like_objects():
    assert _jsonish(None) is None
    assert _jsonish("x") == "x"
    assert _jsonish(1) == 1
    assert _jsonish([1, 2]) == [1, 2]
    assert _jsonish({"a": 1}) == {"a": 1}
    assert _jsonish(_ModelDumpObj()) == {"a": 1}
    assert _jsonish(_DictObj()) == {"b": 2}
    assert _jsonish(_ToDictObj()) == {"c": 3}
    assert isinstance(_jsonish(_BadJsonObj()), str)


def test_parse_jsonish_parses_json_strings_and_preserves_non_json():
    assert _parse_jsonish('{"a": 1}') == {"a": 1}
    assert _parse_jsonish("[1, 2]") == [1, 2]
    assert _parse_jsonish('"hello"') == "hello"
    assert _parse_jsonish("plain text") == "plain text"


def test_now_str_returns_string_number():
    out = _now_str()
    assert isinstance(out, str)
    float(out)


def test_coerce_int_handles_valid_invalid_and_none():
    assert _coerce_int(None) is None
    assert _coerce_int("5") == 5
    assert _coerce_int(7.9) == 7
    assert _coerce_int("bad") is None


def test_summarize_value_truncates_long_content():
    text = _summarize_value({"x": "a" * 500}, limit=20)
    assert text.endswith("...")
    assert len(text) <= 23


def test_extract_node_name_from_ns_handles_multiple_forms():
    assert _extract_node_name_from_ns(None) is None
    assert _extract_node_name_from_ns(()) is None
    assert _extract_node_name_from_ns(["node:abc"]) == "node"
    assert _extract_node_name_from_ns(["node_only"]) == "node_only"


def test_callback_handler_chain_start_end_and_error_record_events(collector):
    handler = _GroundEvalLangChainCallbackHandler(collector)
    handler.on_chain_start(
        serialized={"name": "nodeA"},
        inputs={"q": 1},
        run_id="r1",
        parent_run_id="p1",
        tags=["t"],
        metadata={"langgraph_node": "nodeA:branch"},
    )
    handler.on_chain_end(
        outputs={"ok": True},
        run_id="r1",
        parent_run_id="p1",
        tags=["t"],
    )
    handler.on_chain_error(
        _CallbackError("boom"),
        run_id="r1",
        parent_run_id="p1",
        tags=["t"],
    )
    event_types = [e.event_type for e in collector.events]
    assert "langchain.callback.chain_start" in event_types
    assert "langchain.callback.chain_end" in event_types
    assert "langchain.callback.chain_error" in event_types
    assert any(err.message == "boom" for err in collector.errors)


def test_callback_handler_tool_and_retriever_lifecycle_records_child_operations(
    collector,
):
    handler = _GroundEvalLangChainCallbackHandler(collector)
    handler.on_tool_start(
        serialized={"name": "lookup"},
        input_str='{"artifact_id": "a1"}',
        run_id="tool-1",
        parent_run_id="parent-1",
        tags=["x"],
        metadata={"langgraph_node": "node1:branch"},
        inputs=None,
    )
    handler.on_tool_end(
        output='{"id": "a1", "subsystem": "crm"}',
        run_id="tool-1",
        parent_run_id="parent-1",
    )
    handler.on_retriever_start(
        serialized={"name": "retrieverA"},
        query="hello",
        run_id="ret-1",
        parent_run_id="parent-2",
        metadata={"langgraph_node": "node2:branch"},
    )
    handler.on_retriever_end(
        documents=[{"id": "a2"}], run_id="ret-1", parent_run_id="parent-2"
    )
    assert len(collector.tool_calls) == 2
    assert collector.tool_calls[0].tool_name == "lookup"
    assert collector.tool_calls[0].arguments == {"artifact_id": "a1"}
    assert collector.tool_calls[1].tool_name == "retrieverA"
    assert collector.tool_calls[1].arguments == {"query": "hello"}


def test_callback_handler_tool_start_wraps_non_dict_input(collector):
    handler = _GroundEvalLangChainCallbackHandler(collector)
    handler.on_tool_start(
        serialized={"name": "lookup"},
        input_str='"hello"',
        run_id="tool-1",
        metadata={},
        inputs=None,
    )
    handler.on_tool_end(output='"done"', run_id="tool-1")
    assert collector.tool_calls[0].arguments == {"input": "hello"}


def test_callback_handler_tool_error_and_retriever_error_record_failures(collector):
    handler = _GroundEvalLangChainCallbackHandler(collector)
    handler.on_tool_start(
        serialized={"name": "lookup"},
        input_str='{"artifact_id": "a1"}',
        run_id="tool-1",
        metadata={"langgraph_node": "node1"},
    )
    handler.on_tool_error(_CallbackError("tool boom"), run_id="tool-1")
    handler.on_retriever_start(
        serialized={"name": "ret"},
        query="q",
        run_id="ret-1",
        metadata={"langgraph_node": "node2"},
    )
    handler.on_retriever_error(_CallbackError("ret boom"), run_id="ret-1")
    assert any(e.message == "tool boom" for e in collector.errors)
    assert any(e.message == "ret boom" for e in collector.errors)
    assert collector.child_ops_by_run_id == {}


def test_callback_handler_llm_and_chat_events_record_model_event(collector):
    handler = _GroundEvalLangChainCallbackHandler(collector)
    handler.on_llm_start(serialized={"name": "llm"}, prompts=["p"], run_id="r1")
    handler.on_llm_end(response=_ResponseWithLLMOutput(), run_id="r1")
    handler.on_llm_error(_CallbackError("llm boom"), run_id="r1")
    handler.on_chat_model_start(
        serialized={"name": "chat"}, messages=[["m"]], run_id="r2"
    )
    assert len(collector.model_events) == 1
    model_event = collector.model_events[0]
    assert model_event.model_name == "gpt-test"
    assert model_event.provider_name == "openai"
    assert model_event.input_tokens == 3
    assert model_event.output_tokens == 4
    assert model_event.finish_reason == "stop"
    assert model_event.tool_schemas_count == 1


def test_callback_handler_custom_event_records_event(collector):
    handler = _GroundEvalLangChainCallbackHandler(collector)
    handler.on_custom_event(
        name="custom-x",
        data={"a": 1},
        run_id="r1",
        metadata={"langgraph_node": "nodeC"},
    )
    assert collector.events[-1].event_type == "langchain.callback.custom_event"


def test_collector_resolve_node_name_priority_order(collector):
    assert (
        collector.resolve_node_name_from_callback(
            metadata={"langgraph_node": "node1:abc"}
        )
        == "node1"
    )
    assert (
        collector.resolve_node_name_from_callback(metadata={"node_name": "node2"})
        == "node2"
    )
    assert (
        collector.resolve_node_name_from_callback(
            metadata={"checkpoint_ns": "root/node3"}
        )
        == "node3"
    )
    assert (
        collector.resolve_node_name_from_callback(kwargs={"name": "node4"}) == "node4"
    )
    assert (
        collector.resolve_node_name_from_callback(serialized={"name": "node5"})
        == "node5"
    )
    assert collector.resolve_node_name_from_callback() is None


def test_collector_note_node_start_end_and_stream_updates(collector):
    collector.note_node_start(node_name="node1", arguments={"a": 1})
    collector.note_node_end(node_name="node1", return_value={"b": 2})
    collector.note_node_from_stream(
        node_name="node2", branch_id="b1", arguments={"x": 1}, return_value={"id": "a1"}
    )
    assert collector.node_state["node1"]["arguments"] == {"a": 1}
    assert collector.node_state["node1"]["return_value"] == {"b": 2}
    assert collector.node_state["node2"]["branch_id"] == "b1"
    assert "b1" in collector.distinct_branch_ids


def test_collector_start_finish_and_fail_child_operations(collector):
    collector.start_child_operation(
        run_id="r1",
        node_name="node1",
        tool_name="lookup",
        arguments={"artifact_id": "a1"},
        parent_event_id="p1",
        source="child_tool",
        kind="tool",
    )
    collector.finish_child_operation(
        run_id="r1", node_name="node1", return_value={"id": "a1"}
    )
    assert len(collector.tool_calls) == 1
    assert collector.tool_calls[0].tool_name == "lookup"
    collector.start_child_operation(
        run_id="r2",
        node_name="node2",
        tool_name="lookup2",
        arguments={},
        parent_event_id=None,
        source="child_tool",
        kind="tool",
    )
    collector.fail_child_operation(
        run_id="r2", node_name="node2", error=_CallbackError("boom")
    )
    assert any(e.message == "boom" for e in collector.errors)


def test_collector_finish_child_operation_ignores_unknown_run_id(collector):
    collector.finish_child_operation(
        run_id="missing", node_name="node", return_value={"id": "a1"}
    )
    assert collector.tool_calls == []


def test_collector_record_model_event_uses_fallback_usage_metadata():
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=_FakeGraph(),
        recording=RecordingRuntime(),
    )
    msg = types.SimpleNamespace(
        response_metadata={"finish_reason": "done", "model_name": "model-x"},
        usage_metadata={"input_tokens": 9, "output_tokens": 10},
        tool_calls=[{"name": "t1"}, {"name": "t2"}],
    )
    generation = types.SimpleNamespace(message=msg)
    response = types.SimpleNamespace(llm_output={}, generations=[[generation]])
    collector.record_model_event(response=response, timestamp="1.0")
    event = collector.model_events[0]
    assert event.model_name == "model-x"
    assert event.input_tokens == 9
    assert event.output_tokens == 10
    assert event.tool_schemas_count == 2


def test_process_stream_chunk_supports_all_emitted_shapes(collector):
    collector.process_stream_chunk({
        "type": "values",
        "ns": ["node1:abc"],
        "data": {"final": True},
    })
    collector.process_stream_chunk(("updates", {"node2": {"id": "a1"}}))
    collector.process_stream_chunk((
        ("root", "review:task"),
        "updates",
        {"review": {"ok": True}},
    ))
    collector.process_stream_chunk((
        ("root", "review:task"),
        "values",
        {"should_act": True},
    ))
    collector.process_stream_chunk("weird")

    event_types = [event.event_type for event in collector.events]
    three_part_events = [
        event for event in collector.events if event.branch_id == "root/review:task"
    ]

    assert "langgraph.stream.values" in event_types
    assert "langgraph.stream.updates" in event_types
    assert "langgraph.stream.unknown" in event_types
    assert [event.event_type for event in three_part_events] == [
        "langgraph.stream.updates",
        "langgraph.stream.values",
    ]
    assert all(event.node_name == "review" for event in three_part_events)
    assert collector.final_output_from_values == {"should_act": True}
    assert not any(
        event.event_type == "langgraph.stream.unknown" for event in three_part_events
    )


def test_runtime_handoffs_capture_sequential_transitions_without_duplicates(
    collector,
):
    collector.static_handoffs = [
        ObservedHandoff(
            "review",
            "routing",
            payload_type="langgraph.static_edge",
        )
    ]

    collector.process_stream_chunk(("updates", {"review": {"ok": True}}))
    collector.process_stream_chunk(("updates", {"review": {"ok": True}}))
    collector.process_stream_chunk(("updates", {"routing": {"owner": "renewals"}}))
    collector.process_stream_chunk(("updates", {"routing": {"owner": "renewals"}}))

    assert len(collector.dynamic_handoffs) == 1
    handoff = collector.dynamic_handoffs[0]
    assert handoff.from_executor_id == "review"
    assert handoff.to_executor_id == "routing"
    assert handoff.payload_type == "langgraph.runtime_transition"


def test_runtime_handoffs_are_branch_scoped_and_parallel_updates_are_unordered(
    collector,
):
    collector.process_stream_chunk((("branch-a",), "updates", {"review": {"ok": True}}))
    collector.process_stream_chunk((
        ("branch-b",),
        "updates",
        {"support": {"ok": True}},
    ))
    collector.process_stream_chunk((
        ("branch-a",),
        "updates",
        {"risk": {"ok": True}, "routing": {"ok": True}},
    ))
    collector.process_stream_chunk((
        ("branch-b",),
        "updates",
        {"routing": {"ok": True}},
    ))

    assert len(collector.dynamic_handoffs) == 1
    handoff = collector.dynamic_handoffs[0]
    assert handoff.from_executor_id == "support"
    assert handoff.to_executor_id == "routing"


def test_runtime_handoff_must_match_static_topology_when_available(collector):
    collector.static_handoffs = [
        ObservedHandoff(
            "review",
            "approved",
            payload_type="langgraph.static_edge",
        )
    ]

    collector.process_stream_chunk(("updates", {"review": {"ok": True}}))
    collector.process_stream_chunk(("updates", {"routing": {"ok": True}}))

    assert collector.dynamic_handoffs == []


def test_extract_node_data_from_stream_debug_and_node_scoped_updates(collector):
    collector._extract_node_data_from_stream(
        mode="debug",
        data={"name": "nodeA", "input": {"x": 1}, "result": {"id": "a1"}},
        node_name=None,
        branch_id="b1",
    )
    collector._extract_node_data_from_stream(
        mode="values",
        data={"id": "a2"},
        node_name="nodeB",
        branch_id="b2",
    )
    assert collector.node_state["nodeA"]["arguments"] == {"x": 1}
    assert collector.node_state["nodeA"]["return_value"] == {"id": "a1"}
    assert collector.node_state["nodeB"]["return_value"] == {"id": "a2"}


def test_final_output_precedence(collector):
    collector.final_output_from_values = {"a": 1}
    collector.final_output_from_last_dict = {"b": 2}
    collector.raw_result = {"c": 3}
    assert collector._final_output() == {"a": 1}
    collector.final_output_from_values = None
    assert collector._final_output() == {"b": 2}
    collector.final_output_from_last_dict = None
    assert collector._final_output() == {"c": 3}
    collector.raw_result = "text"
    assert collector._final_output() == {"raw_output": "text"}


def test_introspect_static_graph_records_nodes_edges_and_subgraphs():
    graph = _FakeGraph(
        static_graph=_FakeStaticGraph(
            nodes={"node1": object(), "node2": object()},
            edges=[_FakeStaticEdge("node1", "node2")],
        ),
        subgraphs=[("sg1", object()), "sg2"],
    )
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )
    collector.introspect_static_graph()
    assert collector.static_graph_available is True
    assert collector.subgraph_introspection_available is True
    assert collector.static_nodes == {"node1", "node2"}
    assert len(collector.static_handoffs) == 1
    event_types = [e.event_type for e in collector.events]
    assert "langgraph.introspection.static_nodes" in event_types
    assert "langgraph.introspection.subgraphs" in event_types


def test_introspect_static_graph_handles_failures():
    class BadGraph:
        def get_graph(self):
            raise RuntimeError("no graph")

        def get_subgraphs(self):
            raise RuntimeError("no subgraphs")

    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=BadGraph(),
        recording=RecordingRuntime(),
    )
    collector.introspect_static_graph()
    event_types = [e.event_type for e in collector.events]
    assert "langgraph.introspection.unavailable" in event_types
    assert "langgraph.introspection.subgraphs_unavailable" in event_types


def test_execute_async_passes_recursion_limit_to_astream():
    class AsyncGraph:
        def __init__(self):
            self.kwargs = None
            self._groundeval_max_steps = 7

        async def astream(self, *args, **kwargs):
            self.kwargs = kwargs
            yield {
                "type": "values",
                "ns": ["node1:branch"],
                "data": {"should_act": True},
            }

    graph = AsyncGraph()
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )

    __import__("asyncio").run(collector._execute_async(graph))

    assert graph.kwargs["config"]["recursion_limit"] == 7


def test_execute_async_omits_recursion_limit_when_not_configured():
    class AsyncGraph:
        def __init__(self):
            self.kwargs = None

        async def astream(self, *args, **kwargs):
            self.kwargs = kwargs
            if False:
                yield None

    graph = AsyncGraph()
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )

    __import__("asyncio").run(collector._execute_async(graph))

    assert "recursion_limit" not in graph.kwargs["config"]


def test_execute_async_passes_recursion_limit_to_stream():
    class StreamGraph:
        def __init__(self):
            self.kwargs = None
            self._groundeval_max_steps = 5

        def stream(self, *args, **kwargs):
            self.kwargs = kwargs
            return iter([])

    graph = StreamGraph()
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )

    __import__("asyncio").run(collector._execute_async(graph))

    assert graph.kwargs["config"]["recursion_limit"] == 5


def test_execute_async_prefers_astream_when_available():
    graph = _FakeGraph(
        astream_chunks=[
            {"type": "values", "ns": ["node1:branch"], "data": {"should_act": True}}
        ]
    )
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )
    result = __import__("asyncio").run(collector._execute_async(graph))
    assert result == {"should_act": True}


def test_execute_async_uses_stream_when_no_astream():
    class StreamOnlyGraph:
        def stream(self, *args, **kwargs):
            yield {
                "type": "values",
                "ns": ["node1:branch"],
                "data": {"should_act": True},
            }

    graph = StreamOnlyGraph()
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=graph,
        recording=RecordingRuntime(),
    )
    result = __import__("asyncio").run(collector._execute_async(graph))
    assert result == {"should_act": True}


def test_execute_async_raises_when_no_stream_interfaces():
    collector = _LangGraphEventCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=object(),
        recording=RecordingRuntime(),
    )
    with pytest.raises(TypeError, match="compiled graph"):
        __import__("asyncio").run(collector._execute_async(object()))


def test_execute_records_errors_and_raises():
    class BadCollector(_LangGraphEventCollector):
        async def _execute_async(self, agent):
            raise _CallbackError("boom")

    collector = BadCollector(
        run_id="run-1",
        agent_class="pkg.Agent",
        graph=_FakeGraph(),
        recording=RecordingRuntime(),
    )
    with pytest.raises(_CallbackError, match="boom"):
        collector.execute(_FakeGraph())
    assert len(collector.errors) == 1
    assert collector.errors[0].message == "boom"


def test_to_observed_run_builds_capabilities_and_workflow(collector):
    collector.started_at = 1.0
    collector.completed_at = 2.0
    collector.final_output_from_values = {"should_act": True}
    collector.static_nodes = {"node1"}
    collector.node_state = {
        "node1": {"entered_at": "1.0", "exited_at": "2.0", "return_value": {"id": "a1"}}
    }
    run = collector.to_observed_run({"ignored": True})
    assert run.framework == "langgraph"
    assert run.final_output == {"should_act": True}
    assert run.total_latency_ms == 1000.0
    assert run.capabilities["workflow_nodes"] is True


def test_to_observed_run_agents_round_trip_as_dataclasses(collector):
    collector.started_at = 1.0
    collector.completed_at = 2.0
    collector.final_output_from_values = {"should_act": True}
    collector.static_nodes = {"review"}
    collector.node_state = {
        "review": {
            "entered_at": "1.0",
            "exited_at": "2.0",
            "return_value": {"ok": True},
        }
    }
    collector.tool_calls = [
        ObservedToolCall(
            tool_name="lookup",
            arguments={"customer": "Acme"},
            return_value={"ok": True},
            latency_ms=1.0,
            agent_id="langgraph:review",
            agent_name="review",
            node_name="review",
        )
    ]
    collector.child_tool_calls_by_node = {"review": list(collector.tool_calls)}

    run = collector.to_observed_run({"should_act": True})
    restored = FrameworkObservedRun.from_dict(run.to_dict())

    assert all(isinstance(agent, ObservedAgent) for agent in run.agents)
    assert all(isinstance(agent, ObservedAgent) for agent in restored.agents)
    assert len(restored.agents) == 1
    assert restored.agents[0].agent_id == "langgraph:review"
    assert restored.agents[0].agent_name == "review"
    assert restored.agents[0].role == "review"
    assert restored.agents[0].tool_call_count == 1


def test_langgraph_observer_load_agent_calls_version_check_and_loader():
    observer = LangGraphObserver()
    with patch(
        "groundeval.framework_adapters.langgraph_adapter._require_python_311"
    ) as req:
        with patch(
            "groundeval.framework_adapters.langgraph_adapter._load_langgraph",
            return_value="graph",
        ) as load:
            out = observer.load_agent("pkg.graph")
    assert out == "graph"
    req.assert_called_once()
    load.assert_called_once_with("pkg.graph")


def test_langgraph_observer_instrument_agent_sets_collector_and_recording():
    observer = LangGraphObserver()
    agent = _FakeGraph()
    recording = RecordingRuntime()
    with patch("groundeval.framework_adapters.langgraph_adapter._require_python_311"):
        out = observer.instrument_agent(agent, recording)
    assert out is agent
    assert hasattr(agent, "_groundeval_langgraph_collector")
    assert agent._groundeval_recording is recording


def test_langgraph_observer_execute_agent_populates_framework_run_and_recording():
    observer = LangGraphObserver()

    class Agent:
        pass

    agent = Agent()
    recording = RecordingRuntime()
    agent._groundeval_recording = recording

    fake_tool_call = ObservedToolCall(
        tool_name="lookup",
        arguments={"artifact_id": "a1"},
        return_value={"id": "a1"},
        latency_ms=1.0,
        node_name="node1",
        workflow_run_id="wf-1",
        branch_id="b1",
        parent_event_id="p1",
    )
    fake_run = types.SimpleNamespace(tool_calls=[fake_tool_call])

    class FakeCollector:
        def execute(self, agent_obj):
            return {"ok": True}

        def to_observed_run(self, result):
            return fake_run

    agent._groundeval_langgraph_collector = FakeCollector()
    out = observer.execute_agent(agent)
    assert out == {"ok": True}
    assert agent._groundeval_framework_observed_run is fake_run
    assert len(recording.call_log) == 1
    assert recording.call_log[0].tool_name == "lookup"


def test_langgraph_observer_execute_agent_still_sets_framework_run_on_error():
    observer = LangGraphObserver()

    class Agent:
        pass

    agent = Agent()
    agent._groundeval_recording = RecordingRuntime()
    fake_run = types.SimpleNamespace(tool_calls=[])

    class FakeCollector:
        def execute(self, agent_obj):
            raise _CallbackError("boom")

        def to_observed_run(self, result):
            assert result is None
            return fake_run

    agent._groundeval_langgraph_collector = FakeCollector()
    with pytest.raises(_CallbackError, match="boom"):
        observer.execute_agent(agent)
    assert agent._groundeval_framework_observed_run is fake_run


def test_langgraph_observer_set_max_steps_sets_attribute_and_rejects_invalid_values():
    observer = LangGraphObserver()
    agent = types.SimpleNamespace()

    observer.set_max_steps(agent, 7)

    assert agent._groundeval_max_steps == 7
    with pytest.raises(ValueError, match="greater than zero"):
        observer.set_max_steps(agent, 0)
    with pytest.raises(ValueError, match="greater than zero"):
        observer.set_max_steps(agent, -1)


def test_generate_langgraph_report_contains_key_sections():
    from groundeval.framework_adapters.framework_observation import (
        ObservedError,
        ObservedEvent,
        ObservedModelEvent,
        ObservedRun,
        ObservedWorkflow,
        ObservedWorkflowNode,
        ObservedHandoff,
    )

    run = ObservedRun(
        run_id="r1",
        framework="langgraph",
        agent_class="pkg.Graph",
        started_at="1.0",
        completed_at="2.0",
        total_latency_ms=1000.0,
        tool_calls=[
            ObservedToolCall(
                tool_name="lookup",
                arguments={"artifact_id": "a1"},
                return_value={"id": "a1"},
                latency_ms=10.0,
                node_name="node1",
            )
        ],
        events=[ObservedEvent(event_type="evt", timestamp="1.5", node_name="node1")],
        agents=[],
        workflow=ObservedWorkflow(
            workflow_id="wf-1",
            workflow_name="Main",
            node_count=1,
            nodes=[
                ObservedWorkflowNode(
                    node_id="node1", node_type="langgraph.node.observed_operation"
                )
            ],
            handoff_count=1,
            handoffs=[
                ObservedHandoff(
                    from_executor_id="a", to_executor_id="b", payload_type="edge"
                )
            ],
            branch_count=1,
        ),
        model_events=[
            ObservedModelEvent(event_type="model.call.completed", model_name="gpt-test")
        ],
        final_output={"should_act": True},
        errors=[ObservedError(error_type="X", message="boom", timestamp="1.9")],
        capabilities={"tool_calls": True, "workflow_nodes": True},
    )
    report = generate_langgraph_report(run)
    assert "GroundEval LangGraph Observation Report" in report
    assert "Capabilities" in report
    assert "Workflow Nodes" in report
    assert "Observed Operations" in report
    assert "Static Handoffs" in report
    assert "Model Events" in report
    assert "Raw Framework Events" in report
    assert "Errors" in report


def test_debug_and_update_for_same_transition_do_not_duplicate_handoff(collector):
    collector.process_stream_chunk((
        ("root",),
        "debug",
        {"node": "review", "output": {"ok": True}},
    ))
    collector.process_stream_chunk((("root",), "updates", {"review": {"ok": True}}))
    collector.process_stream_chunk((
        ("root",),
        "debug",
        {"node": "routing", "output": {"ok": True}},
    ))
    collector.process_stream_chunk((("root",), "updates", {"routing": {"ok": True}}))

    assert len(collector.dynamic_handoffs) == 1
    handoff = collector.dynamic_handoffs[0]
    assert handoff.from_executor_id == "review"
    assert handoff.to_executor_id == "routing"
    assert handoff.payload_type == "langgraph.runtime_transition"


def test_workflow_preserves_static_and_runtime_handoff_types(collector):
    collector.static_handoffs = [
        ObservedHandoff(
            from_executor_id="review",
            to_executor_id="routing",
            payload_type="langgraph.static_edge",
        )
    ]

    collector.process_stream_chunk(("updates", {"review": {"ok": True}}))
    collector.process_stream_chunk(("updates", {"routing": {"ok": True}}))

    workflow = collector._build_workflow(final_output={"should_act": True})
    payload_types = [handoff.payload_type for handoff in workflow.handoffs]

    assert payload_types.count("langgraph.static_edge") == 1
    assert payload_types.count("langgraph.runtime_transition") == 1
    assert workflow.handoff_count == 2
