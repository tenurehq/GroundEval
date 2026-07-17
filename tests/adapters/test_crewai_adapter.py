import types
from unittest.mock import patch

import pytest

from groundeval.framework_adapters.crewai_adapter import (
    CrewAIObserver,
    _CrewAIEventCollector,
    _agent_id_from_event,
    _agent_name_from_event,
    _coerce_int,
    _event_id,
    _event_timestamp,
    _event_type_name,
    _jsonish,
    _llm_model_name,
    _llm_provider_name,
    _load_crew,
    _parent_event_id,
    _parse_crew_output,
    _parse_jsonish,
    _safe_getattr,
    _task_id_from_event,
    _task_name_from_event,
    _tool_args_from_event,
    _tool_name_from_event,
    _tool_output_from_event,
)


class _PydanticLike:
    def model_dump(self):
        return {"value": 1}


class _DictLike:
    def dict(self):
        return {"value": 2}


class _ToDictLike:
    def to_dict(self):
        return {"value": 3}


class _ModelDumpRaises:
    def model_dump(self):
        raise RuntimeError("boom")

    def __str__(self):
        return "fallback-str"


class _DictRaises:
    def dict(self):
        raise RuntimeError("boom")

    def __str__(self):
        return "fallback-str"


class _ToDictRaises:
    def to_dict(self):
        raise RuntimeError("boom")

    def __str__(self):
        return "fallback-str"


class _ResultWithPydantic:
    def __init__(self):
        self.pydantic = _PydanticLike()
        self.raw = ""


class _ResultWithRawDict:
    def __init__(self):
        self.raw = {"ok": True}


class _ResultWithRawJson:
    def __init__(self):
        self.raw = '{"ok": true}'


class _ResultBad:
    def __init__(self):
        self.raw = None

    def __str__(self):
        return "fallback"


class _CrewResultModelOnly:
    def model_dump(self):
        return {"ok": True}


class _FakeTask:
    def __init__(self, description="orig"):
        self.description = description


class _FakeCrew:
    def __init__(self, result=None):
        self.tasks = [_FakeTask()]
        self.max_iter = None
        self._result = result if result is not None else _ResultWithRawJson()

    def kickoff(self):
        return self._result


class _FactoryModule:
    pass


def _question():
    return types.SimpleNamespace(
        question_id="q1",
        question_text="Question text",
    )


def test_load_crew_from_function_factory():
    mod = _FactoryModule()
    mod.make_crew = lambda: "factory-result"
    with patch("importlib.import_module", return_value=mod):
        assert _load_crew("x.make_crew") == "factory-result"


def test_load_crew_from_class_with_crew_method():
    class Builder:
        def crew(self):
            return "crew-result"

    mod = _FactoryModule()
    mod.Builder = Builder
    with patch("importlib.import_module", return_value=mod):
        assert _load_crew("x.Builder") == "crew-result"


def test_load_crew_from_plain_class_instance():
    class Plain:
        pass

    mod = _FactoryModule()
    mod.Plain = Plain
    with patch("importlib.import_module", return_value=mod):
        out = _load_crew("x.Plain")
    assert isinstance(out, Plain)


def test_load_crew_from_plain_object():
    mod = _FactoryModule()
    sentinel = object()
    mod.value = sentinel
    with patch("importlib.import_module", return_value=mod):
        assert _load_crew("x.value") is sentinel


def test_jsonish_handles_common_and_fallback_types():
    assert _jsonish(None) is None
    assert _jsonish({"a": 1}) == {"a": 1}
    assert _jsonish(_PydanticLike()) == {"value": 1}
    assert _jsonish(_DictLike()) == {"value": 2}
    assert _jsonish(_ToDictLike()) == {"value": 3}
    assert _jsonish(_ModelDumpRaises()) == "fallback-str"
    assert _jsonish(_DictRaises()) == "fallback-str"
    assert _jsonish(_ToDictRaises()) == "fallback-str"


def test_parse_jsonish_variants():
    assert _parse_jsonish('{"a": 1}') == {"a": 1}
    assert _parse_jsonish('["a"]') == ["a"]
    assert _parse_jsonish('"hello"') == "hello"
    assert _parse_jsonish("{bad") == "{bad"
    assert _parse_jsonish(3) == 3


def test_safe_getattr_returns_default_on_attribute_error():
    class Bad:
        def __getattr__(self, name):
            raise RuntimeError("bad getattr")

    assert _safe_getattr(Bad(), "x", "fallback") == "fallback"


def test_event_helpers_cover_primary_and_fallback_fields():
    agent = types.SimpleNamespace(id="agent-1", role="planner")
    task = types.SimpleNamespace(description="review customer")

    event = types.SimpleNamespace(
        type="CustomType",
        timestamp=123,
        event_id="evt-1",
        parent_event_id="parent-1",
        agent=agent,
        task=task,
        tool_name="fetch_customer",
        tool_args='{"artifact_id": "a1"}',
        output='{"id": "a1"}',
    )

    assert _event_type_name(event) == "CustomType"
    assert _event_timestamp(event) == "123"
    assert _event_id(event) == "evt-1"
    assert _parent_event_id(event) == "parent-1"
    assert _agent_id_from_event(event) == "agent-1"
    assert _agent_name_from_event(event) == "planner"
    assert _task_name_from_event(event) == "review customer"
    assert _tool_name_from_event(event) == "fetch_customer"
    assert _tool_args_from_event(event) == {"artifact_id": "a1"}
    assert _tool_output_from_event(event) == {"id": "a1"}


def test_event_helper_fallbacks_cover_secondary_fields():
    event_triggered = types.SimpleNamespace(triggered_by_event_id="p1")
    event_started = types.SimpleNamespace(started_event_id="p2")
    event_task_id = types.SimpleNamespace(task_id="task-1")
    event_tool_bad_args = types.SimpleNamespace(tool_args="{bad json")
    event_no_output = types.SimpleNamespace(output=None)

    assert _parent_event_id(event_triggered) == "p1"
    assert _parent_event_id(event_started) == "p2"
    assert _task_id_from_event(event_task_id) == "task-1"
    assert _tool_args_from_event(event_tool_bad_args) == {}
    assert _tool_output_from_event(event_no_output) is None


def test_llm_helpers_and_int_coercion():
    LlmCls = type("ChatModel", (), {})
    LlmCls.__module__ = "openai.client"
    llm = LlmCls()
    llm.model = "gpt-4o"

    event = types.SimpleNamespace(llm=llm)

    assert _llm_model_name(event) == "gpt-4o"
    assert _llm_provider_name(event) == "openai"
    assert _coerce_int("12") == 12
    assert _coerce_int(None) is None
    assert _coerce_int("abc") is None


def test_parse_crew_output_prefers_pydantic():
    assert _parse_crew_output(_ResultWithPydantic()) == {"value": 1}


def test_parse_crew_output_raw_dict():
    assert _parse_crew_output(_ResultWithRawDict()) == {"ok": True}


def test_parse_crew_output_raw_json():
    assert _parse_crew_output(_ResultWithRawJson()) == {"ok": True}


def test_parse_crew_output_model_dump_fallback():
    assert _parse_crew_output(_CrewResultModelOnly()) == {"ok": True}


def test_parse_crew_output_non_dict_paths_fall_back():
    class RawList:
        raw = '["a", "b"]'

    class EmptyRawNonDictDump:
        raw = ""

        def model_dump(self):
            return ["x"]

    assert "raw_output" in _parse_crew_output(RawList())
    assert "raw_output" in _parse_crew_output(EmptyRawNonDictDump())
    assert "raw_output" in _parse_crew_output(_ResultBad())


def test_collector_install_raises_when_crewai_events_missing():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")
    with pytest.raises(RuntimeError, match="requires CrewAI event listeners"):
        collector.install()


def test_collector_install_sets_listener_and_bus():
    class FakeBus:
        def on(self, event_type):
            def decorator(fn):
                self.fn = fn
                return fn

            return decorator

    class BaseEventListener:
        def __init__(self):
            pass

    fake_mod = types.SimpleNamespace(
        BaseEventListener=BaseEventListener,
        crewai_event_bus=FakeBus(),
    )

    with patch.dict("sys.modules", {"crewai.events": fake_mod}):
        collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")
        collector.install()

    assert collector._listener is not None
    assert collector._bus is not None


def test_collector_flush_branches():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")
    collector.flush()

    class Bus:
        def __init__(self):
            self.called = False

        def flush(self):
            self.called = True

    bus = Bus()
    collector._bus = bus
    collector.flush()
    assert bus.called is True


def test_collector_capture_tool_usage_and_kickoff_completed():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    ToolFinished = type("ToolUsageFinishedEvent", (), {})
    tool_event = ToolFinished()
    tool_event.tool_name = "fetch_customer"
    tool_event.tool_args = {"artifact_id": "a1"}
    tool_event.output = {"id": "a1", "subsystem": "crm"}
    tool_event.timestamp = "2026-01-01T00:00:00"
    tool_event.agent_id = "agent-1"
    tool_event.agent_role = "planner"
    tool_event.task_name = "task-a"
    tool_event.event_id = "e1"

    KickoffDone = type("CrewKickoffCompletedEvent", (), {})
    kickoff_event = KickoffDone()
    kickoff_event.output = {"should_act": True}
    kickoff_event.timestamp = "2026-01-01T00:00:01"

    collector.capture(source=object(), event=tool_event)
    collector.capture(source=object(), event=kickoff_event)

    rich = collector.to_rich_observed_run()
    assert len(rich.tool_calls) == 1
    assert rich.tool_calls[0].tool_name == "fetch_customer"
    assert rich.final_output == {"should_act": True}


def test_collector_capture_kickoff_started_sets_started_at():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    Started = type("CrewKickoffStartedEvent", (), {})
    event = Started()
    event.timestamp = "2026-01-01T00:00:00"

    collector.capture(source=object(), event=event)

    assert collector.started_at is not None


def test_collector_capture_error_events():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    KickoffFailed = type("CrewKickoffFailedEvent", (), {})
    kickoff_failed = KickoffFailed()
    kickoff_failed.timestamp = "2026-01-01T00:00:00"
    kickoff_failed.error = "boom"

    TaskFailed = type("TaskFailedEvent", (), {})
    task_failed = TaskFailed()
    task_failed.timestamp = "2026-01-01T00:00:01"
    task_failed.error_message = "task failure"
    task_failed.agent_id = "agent-9"

    ToolError = type("ToolUsageErrorEvent", (), {})
    tool_error = ToolError()
    tool_error.timestamp = "2026-01-01T00:00:02"
    tool_error.error_message = "tool failed"
    tool_error.agent_id = "agent-1"

    collector.capture(source=object(), event=kickoff_failed)
    collector.capture(source=object(), event=task_failed)
    collector.capture(source=object(), event=tool_error)

    rich = collector.to_rich_observed_run()
    assert len(rich.errors) == 3
    assert rich.errors[0].message == "boom"
    assert rich.errors[1].executor_id == "agent-9"
    assert rich.errors[2].executor_id == "agent-1"


def test_collector_capture_llm_completed_records_model_event():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    Done = type("LLMCallCompletedEvent", (), {})
    event = Done()
    event.timestamp = "2026-01-01T00:00:00"
    event.model_name = "gpt-4o"
    event.prompt_tokens = "12"
    event.completion_tokens = "7"
    event.finish_reason = "stop"

    collector.capture(source=object(), event=event)

    rich = collector.to_rich_observed_run()
    assert len(rich.model_events) == 1
    assert rich.model_events[0].model_name == "gpt-4o"
    assert rich.model_events[0].input_tokens == 12
    assert rich.model_events[0].output_tokens == 7
    assert rich.model_events[0].finish_reason == "stop"


def test_collector_capture_delegation_primary_and_fallback_fields():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    DelegationA = type("DelegationEvent", (), {})
    event_a = DelegationA()
    event_a.timestamp = "2026-01-01T00:00:00"
    event_a.agent_id = "agent-1"
    event_a.target_agent_id = "agent-2"

    DelegationB = type("DelegationEvent", (), {})
    event_b = DelegationB()
    event_b.timestamp = "2026-01-01T00:00:01"
    event_b.from_agent = "agent-a"
    event_b.to_agent = "agent-b"

    collector.capture(source=object(), event=event_a)
    collector.capture(source=object(), event=event_b)

    rich = collector.to_rich_observed_run()
    assert rich.workflow is not None
    assert rich.workflow.handoff_count == 2
    assert rich.workflow.handoffs[0].from_executor_id == "agent-1"
    assert rich.workflow.handoffs[0].to_executor_id == "agent-2"
    assert rich.workflow.handoffs[1].from_executor_id == "agent-a"
    assert rich.workflow.handoffs[1].to_executor_id == "agent-b"


def test_collector_capture_nested_agent_task_and_parent_event_fallback():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    agent = types.SimpleNamespace(id="agent-1", role="planner")
    task = types.SimpleNamespace(description="review-task")

    EventCls = type("ToolUsageFinishedEvent", (), {})
    event = EventCls()
    event.agent = agent
    event.task = task
    event.tool_name = "fetch_customer"
    event.tool_args = '{"artifact_id": "a1"}'
    event.output = '{"id": "a1", "subsystem": "crm"}'
    event.timestamp = "2026-01-01T00:00:00"
    event.triggered_by_event_id = "parent-1"

    collector.capture(source=object(), event=event)

    rich = collector.to_rich_observed_run()
    assert len(rich.tool_calls) == 1
    assert rich.tool_calls[0].arguments == {"artifact_id": "a1"}
    assert rich.tool_calls[0].agent_id == "agent-1"
    assert rich.tool_calls[0].agent_name == "planner"
    assert rich.tool_calls[0].node_name == "review-task"
    assert rich.tool_calls[0].parent_event_id == "parent-1"


def test_collector_updates_existing_workflow_node_exit_time():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    EventA = type("TaskStartedEvent", (), {})
    event1 = EventA()
    event1.task_id = "task-1"
    event1.task_name = "task-name"
    event1.timestamp = "1.0"

    EventB = type("TaskCompletedEvent", (), {})
    event2 = EventB()
    event2.task_id = "task-1"
    event2.task_name = "task-name"
    event2.timestamp = "2.0"

    collector.capture(source=object(), event=event1)
    collector.capture(source=object(), event=event2)

    rich = collector.to_rich_observed_run()
    assert rich.workflow is not None
    assert rich.workflow.nodes[0].node_id == "task-1"
    assert rich.workflow.nodes[0].entered_at == "1.0"
    assert rich.workflow.nodes[0].exited_at == "2.0"


def test_collector_tool_count_increments_by_agent_name_when_id_missing():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    AgentEvt = type("AgentStartedEvent", (), {})
    event1 = AgentEvt()
    event1.agent_name = "planner"
    event1.timestamp = "1.0"

    ToolEvt = type("ToolUsageFinishedEvent", (), {})
    event2 = ToolEvt()
    event2.agent_name = "planner"
    event2.tool_name = "fetch_customer"
    event2.tool_args = {}
    event2.output = {"id": "a1", "subsystem": "crm"}
    event2.timestamp = "2.0"

    collector.capture(source=object(), event=event1)
    collector.capture(source=object(), event=event2)

    rich = collector.to_rich_observed_run()
    assert len(rich.agents) == 1
    assert rich.agents[0].tool_call_count == 1


def test_collector_uses_explicit_event_type_attribute():
    collector = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")

    EventCls = type("OtherClassName", (), {})
    event = EventCls()
    event.type = "CustomType"
    event.timestamp = "2026-01-01T00:00:00"

    collector.capture(source=object(), event=event)

    rich = collector.to_rich_observed_run()
    assert rich.events[0].event_type == "CustomType"


def test_to_rich_observed_run_with_and_without_workflow():
    empty = _CrewAIEventCollector(run_id="r1", agent_class="pkg.Crew")
    rich_empty = empty.to_rich_observed_run()
    assert rich_empty.workflow is None
    assert rich_empty.capabilities["tool_calls"] is False
    assert rich_empty.capabilities["model_calls"] is False

    collector = _CrewAIEventCollector(run_id="r2", agent_class="pkg.Crew")
    collector.started_at = 10.0
    collector.completed_at = 12.5

    ToolFinished = type("ToolUsageFinishedEvent", (), {})
    event = ToolFinished()
    event.tool_name = "fetch_customer"
    event.tool_args = {"artifact_id": "a1"}
    event.output = {"id": "a1", "subsystem": "crm"}
    event.timestamp = "2026-01-01T00:00:00"
    event.agent_id = "agent-1"
    event.agent_role = "planner"
    event.task_name = "task-a"
    event.event_id = "e1"

    collector.capture(source=object(), event=event)

    rich = collector.to_rich_observed_run()
    assert rich.total_latency_ms == 2500.0
    assert rich.workflow is not None
    assert rich.capabilities["tool_calls"] is True
    assert rich.capabilities["agent_turns"] is True
    assert rich.capabilities["workflow_nodes"] is True


def test_observer_load_agent_delegates():
    observer = CrewAIObserver()
    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value="loaded",
    ):
        assert observer.load_agent("pkg.Crew") == "loaded"


def test_observer_instrument_agent_sets_recording_and_collector():
    observer = CrewAIObserver()

    class Agent:
        pass

    agent = Agent()

    from groundeval.observe import RecordingRuntime

    recording = RecordingRuntime()

    with patch(
        "groundeval.framework_adapters.crewai_adapter._CrewAIEventCollector.install"
    ):
        out = observer.instrument_agent(agent, recording)

    assert out is agent
    assert agent._groundeval_recording is recording
    assert hasattr(agent, "_groundeval_crewai_collector")


def test_observer_execute_agent_requires_collector():
    observer = CrewAIObserver()

    class Agent:
        pass

    with pytest.raises(RuntimeError, match="missing installed event collector"):
        observer.execute_agent(Agent())


def test_observer_execute_agent_records_framework_run_and_flushes():
    observer = CrewAIObserver()

    class Collector:
        def __init__(self):
            self.started_at = None
            self.completed_at = None
            self.final_output = None
            self.flushed = False
            self.tool_calls = []

        def flush(self):
            self.flushed = True

        def to_rich_observed_run(self):
            from groundeval.framework_adapters.framework_observation import ObservedRun

            return ObservedRun(
                run_id="r1",
                framework="crewai",
                agent_class="pkg.Crew",
                tool_calls=[],
                final_output={"should_act": True},
            )

    class Agent:
        def kickoff(self):
            return _ResultWithRawJson()

    agent = Agent()
    agent._groundeval_crewai_collector = Collector()

    from groundeval.observe import RecordingRuntime

    agent._groundeval_recording = RecordingRuntime()

    result = observer.execute_agent(agent)
    assert isinstance(result, _ResultWithRawJson)
    assert agent._groundeval_crewai_collector.flushed is True
    assert hasattr(agent, "_groundeval_framework_observed_run")


def test_observer_execute_agent_still_finalizes_when_kickoff_raises():
    observer = CrewAIObserver()

    class Collector:
        def __init__(self):
            self.started_at = None
            self.completed_at = None
            self.final_output = None
            self.flushed = False

        def flush(self):
            self.flushed = True

        def to_rich_observed_run(self):
            from groundeval.framework_adapters.framework_observation import ObservedRun

            return ObservedRun(
                run_id="r1",
                framework="crewai",
                agent_class="pkg.Crew",
                tool_calls=[],
                final_output=None,
            )

    class Agent:
        def kickoff(self):
            raise RuntimeError("kickoff failed")

    agent = Agent()
    agent._groundeval_crewai_collector = Collector()

    with pytest.raises(RuntimeError, match="kickoff failed"):
        observer.execute_agent(agent)

    assert agent._groundeval_crewai_collector.flushed is True
    assert hasattr(agent, "_groundeval_framework_observed_run")


def test_observer_set_max_steps_success_and_failure():
    observer = CrewAIObserver()

    class GoodAgent:
        def __init__(self):
            self.max_iter = 1

    good = GoodAgent()
    observer.set_max_steps(good, 9)
    assert good.max_iter == 9

    class BadAgent:
        @property
        def max_iter(self):
            return 1

        @max_iter.setter
        def max_iter(self, value):
            raise RuntimeError("cannot set")

    observer.set_max_steps(BadAgent(), 5)
