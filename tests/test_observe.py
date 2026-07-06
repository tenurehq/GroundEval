import json
from pathlib import Path

import yaml

from groundeval.observe import (
    DraftGenerator,
    ObservedRun,
    ObservedToolCall,
    RecordingRuntime,
    _get_observer,
    _parse_observed_answer,
    observe_agent,
    register_observer,
    write_draft_output,
)


class _FakeResultWithRawJson:
    def __init__(self, raw):
        self.raw = raw


class _FakePydanticModel:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _FakeResultWithPydantic:
    def __init__(self, data):
        self.pydantic = _FakePydanticModel(data)


class _FakeModelDump:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return dict(self._data)


class _FakeToDict:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


class _FakeObserver:
    def __init__(self, result=None, rich_run=None):
        self.result = result
        self.rich_run = rich_run
        self.loaded_class_path = None
        self.instrumented_agent = None
        self.max_steps_set = None
        self.executed_agent = None

    def load_agent(self, class_path):
        self.loaded_class_path = class_path
        return _FakeAgent()

    def instrument_agent(self, agent, recording):
        self.instrumented_agent = agent
        agent._groundeval_recording = recording
        if self.rich_run is not None:
            agent._groundeval_framework_observed_run = self.rich_run
        return agent

    def execute_agent(self, agent):
        self.executed_agent = agent
        return self.result

    def set_max_steps(self, agent, max_steps):
        self.max_steps_set = max_steps
        agent.max_steps = max_steps


class _FakeAgent:
    pass


class _FakeRichRun:
    def __init__(self, run_id="rich-run-1", final_output=None, total_latency_ms=123.0):
        self.run_id = run_id
        self.final_output = final_output
        self.total_latency_ms = total_latency_ms

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "framework": "fake",
            "agent_class": "pkg.Agent",
            "tool_calls": [],
            "events": [],
            "agents": [],
            "workflow": None,
            "approvals": [],
            "checkpoints": [],
            "context_events": [],
            "model_events": [],
            "final_output": self.final_output,
            "errors": [],
            "capabilities": {},
        }


def test_observed_tool_call_dataclass_fields():
    call = ObservedToolCall(
        tool_name="fetch_customer",
        arguments={"artifact_id": "a1"},
        return_value={"id": "a1"},
        latency_ms=12.5,
        agent_id="agent-1",
        agent_name="planner",
        node_name="node-1",
        workflow_run_id="wf-1",
        branch_id="b1",
        parent_event_id="evt-1",
    )
    assert call.tool_name == "fetch_customer"
    assert call.arguments == {"artifact_id": "a1"}
    assert call.return_value == {"id": "a1"}
    assert call.latency_ms == 12.5
    assert call.agent_id == "agent-1"
    assert call.agent_name == "planner"
    assert call.node_name == "node-1"
    assert call.workflow_run_id == "wf-1"
    assert call.branch_id == "b1"
    assert call.parent_event_id == "evt-1"


def test_observed_run_to_dict_and_from_dict_round_trip():
    run = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
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
        final_answer={"should_act": True},
        total_latency_ms=50.0,
        framework_extra={"x": 1},
    )
    data = run.to_dict()
    restored = ObservedRun.from_dict(data)

    assert restored.run_id == "r1"
    assert restored.framework == "custom"
    assert restored.agent_class == "pkg.Agent"
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0].tool_name == "fetch_customer"
    assert restored.final_answer == {"should_act": True}
    assert restored.total_latency_ms == 50.0
    assert restored.framework_extra == {"x": 1}


def test_observed_run_from_dict_defaults():
    restored = ObservedRun.from_dict({
        "run_id": "r1",
        "framework": "custom",
        "agent_class": "pkg.Agent",
    })
    assert restored.run_id == "r1"
    assert restored.tool_calls == []
    assert restored.final_answer is None
    assert restored.total_latency_ms == 0.0
    assert restored.framework_extra is None


def test_recording_runtime_records_and_returns_copy():
    runtime = RecordingRuntime()
    runtime.record(
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

    log1 = runtime.call_log
    log2 = runtime.call_log

    assert len(log1) == 1
    assert log1 is not log2
    assert log1[0].tool_name == "fetch_customer"

    log1.append("mutated")
    assert len(runtime.call_log) == 1


def test_get_observer_returns_registered_observer():
    observer = _FakeObserver()
    register_observer("unit_test_framework", observer)
    assert _get_observer("unit_test_framework") is observer


def test_get_observer_raises_for_unknown_framework():
    try:
        _get_observer("definitely_unknown_framework")
        assert False
    except ValueError as exc:
        assert "No observer registered for framework" in str(exc)


def test_parse_observed_answer_none_and_primitives():
    assert _parse_observed_answer(None) is None
    assert _parse_observed_answer({"a": 1}) == {"a": 1}
    assert _parse_observed_answer([1, 2]) == [1, 2]
    assert _parse_observed_answer(3) == 3
    assert _parse_observed_answer(True) is True
    assert _parse_observed_answer("plain text") == "plain text"


def test_parse_observed_answer_json_string_variants():
    assert _parse_observed_answer('{"a": 1}') == {"a": 1}
    assert _parse_observed_answer("[1, 2]") == [1, 2]
    assert _parse_observed_answer('"hello"') == "hello"


def test_parse_observed_answer_invalid_json_string_returns_empty_dict():
    assert _parse_observed_answer("{bad json") == {}


def test_parse_observed_answer_uses_raw_dict():
    result = _FakeResultWithRawJson({"a": 1})
    assert _parse_observed_answer(result) == {"a": 1}


def test_parse_observed_answer_uses_raw_json_string():
    result = _FakeResultWithRawJson('{"a": 1}')
    assert _parse_observed_answer(result) == {"a": 1}


def test_parse_observed_answer_uses_pydantic_model_dump():
    result = _FakeResultWithPydantic({"a": 1})
    assert _parse_observed_answer(result) == {"a": 1}


def test_parse_observed_answer_uses_model_dump_on_object():
    result = _FakeModelDump({"a": 1})
    assert _parse_observed_answer(result) == {"a": 1}


def test_parse_observed_answer_uses_to_dict_on_object():
    result = _FakeToDict({"a": 1})
    assert _parse_observed_answer(result) == {"a": 1}


def test_parse_observed_answer_falls_back_to_string():
    class X:
        def __str__(self):
            return "fallback-string"

    assert _parse_observed_answer(X()) == "fallback-string"


def test_observe_agent_happy_path_without_rich_framework_run():
    observer = _FakeObserver(result={"should_act": True})
    register_observer("observe_unit_basic", observer)

    observed = observe_agent(
        framework="observe_unit_basic",
        class_path="pkg.Agent",
        max_steps=7,
    )

    assert observed.framework == "observe_unit_basic"
    assert observed.agent_class == "pkg.Agent"
    assert observed.final_answer == {"should_act": True}
    assert observed.framework_extra is None
    assert observer.loaded_class_path == "pkg.Agent"
    assert observer.max_steps_set == 7


def test_observe_agent_uses_rich_framework_run_when_present():
    rich = _FakeRichRun(
        run_id="rich-123",
        final_output={"should_act": False},
        total_latency_ms=999.0,
    )
    observer = _FakeObserver(result={"should_act": True}, rich_run=rich)
    register_observer("observe_unit_rich", observer)

    observed = observe_agent(
        framework="observe_unit_rich",
        class_path="pkg.Agent",
        max_steps=3,
    )

    assert observed.run_id == "rich-123"
    assert observed.final_answer == {"should_act": False}
    assert observed.total_latency_ms == 999.0
    assert observed.framework_extra is not None
    assert observed.framework_extra["run_id"] == "rich-123"


def test_draft_generator_generate_from_structured_answer():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=10.0,
            )
        ],
        final_answer={
            "preconditions_verified": [
                {
                    "check": "customer_ok",
                    "facts_found": {"status": "active", "plan": "gold"},
                }
            ],
            "should_act": True,
        },
        total_latency_ms=10.0,
    )

    config = DraftGenerator(observed).generate()
    task = config["task_contracts"][0]

    assert config["agent"]["framework"] == "custom"
    assert config["agent"]["agent_class"] == "pkg.Agent"
    assert task["name"] == "inferred_task"
    assert task["decision_field"] == "should_act"
    assert task["preconditions"][0]["check"] == "customer_ok"
    assert task["preconditions"][0]["required_tool"] == "fetch_customer"
    assert task["preconditions"][0]["expected_field"] == "status"
    assert task["tool_expectations"][0]["tool"] == "fetch_customer"


def test_draft_generator_generate_without_structured_answer_infers_from_tool_calls():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active", "plan": "gold"},
                latency_ms=10.0,
            )
        ],
        final_answer={"reasoning": "done"},
        total_latency_ms=10.0,
    )

    config = DraftGenerator(observed).generate()
    task = config["task_contracts"][0]

    assert len(task["preconditions"]) == 1
    assert task["preconditions"][0]["check"] == "fetch_customer_observed"
    assert task["preconditions"][0]["required_tool"] == "fetch_customer"
    assert task["preconditions"][0]["expected_field"] == "status"


def test_draft_generator_conservative_mode_returns_no_preconditions_without_structured_answer():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=10.0,
            )
        ],
        final_answer={"reasoning": "done"},
        total_latency_ms=10.0,
    )

    config = DraftGenerator(observed, mode="conservative").generate()
    task = config["task_contracts"][0]

    assert task["preconditions"] == []
    assert len(task["tool_expectations"]) == 1


def test_draft_generator_deduplicates_tool_expectations_by_name_and_args():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=10.0,
            ),
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=11.0,
            ),
        ],
        final_answer={},
        total_latency_ms=10.0,
    )

    config = DraftGenerator(observed).generate()
    expectations = config["task_contracts"][0]["tool_expectations"]

    assert len(expectations) == 1
    assert expectations[0]["tool"] == "fetch_customer"


def test_draft_generator_picks_non_default_decision_field():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[],
        final_answer={"should_escalate": True},
        total_latency_ms=10.0,
    )

    config = DraftGenerator(observed).generate()
    assert config["task_contracts"][0]["decision_field"] == "should_escalate"


def test_generate_review_checklist_contains_expected_sections():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=10.0,
            )
        ],
        final_answer={"should_act": True},
        total_latency_ms=10.0,
    )

    text = DraftGenerator(observed).generate_review_checklist()

    assert "GroundEval Draft Config Review Checklist" in text
    assert "Observed Native Tools" in text
    assert "Decision Field" in text
    assert "groundeval validate --config draft_config/config.yaml" in text


def test_generate_observe_report_contains_tool_calls_and_final_answer():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=10.0,
                agent_id="agent-1",
                agent_name="planner",
                node_name="node-1",
            )
        ],
        final_answer={"should_act": True},
        total_latency_ms=25.0,
    )

    text = DraftGenerator(observed).generate_observe_report()

    assert "GroundEval Observation Report" in text
    assert "fetch_customer" in text
    assert "planner" in text
    assert "node-1" in text
    assert '"should_act": true' in text.lower()


def test_generate_observe_report_truncates_large_return_preview():
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="big_tool",
                arguments={},
                return_value={"text": "x" * 1000},
                latency_ms=10.0,
            )
        ],
        final_answer={},
        total_latency_ms=25.0,
    )

    text = DraftGenerator(observed).generate_observe_report()
    assert "..." in text


def test_write_draft_output_writes_expected_files(tmp_path):
    observed = ObservedRun(
        run_id="r1",
        framework="custom",
        agent_class="pkg.Agent",
        tool_calls=[
            ObservedToolCall(
                tool_name="fetch_customer",
                arguments={"artifact_id": "a1"},
                return_value={"status": "active"},
                latency_ms=10.0,
            )
        ],
        final_answer={"should_act": True},
        total_latency_ms=10.0,
    )

    out = write_draft_output(tmp_path, observed, DraftGenerator(observed))

    assert out == Path(tmp_path)
    assert (out / "observed_run.json").exists()
    assert (out / "observe_report.md").exists()
    assert (out / "draft_config" / "config.yaml").exists()
    assert (out / "draft_config" / "REVIEW.md").exists()
    assert (out / "draft_config" / "task_contracts" / "inferred_task.yaml").exists()

    observed_json = json.loads((out / "observed_run.json").read_text())
    assert observed_json["run_id"] == "r1"

    config = yaml.safe_load((out / "draft_config" / "config.yaml").read_text())
    assert config["agent"]["framework"] == "custom"
    assert config["groundeval"]["generated_from_observation"] is True

    task_contract = yaml.safe_load(
        (out / "draft_config" / "task_contracts" / "inferred_task.yaml").read_text()
    )
    assert task_contract["name"] == "inferred_task"
