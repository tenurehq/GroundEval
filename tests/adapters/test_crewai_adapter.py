from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from groundeval.framework_adapters.crewai_adapter import (
    _deep_merge,
    _default_for_schema,
    _extract_return_schema,
    _find_tool,
    _wrapped_tool_factory,
    _infer_tool_verb,
    _parse_crew_output,
    build_crewai_agent_fn,
    build_fixture_return,
)
from groundeval.core import AllowedTool, TaskContract


class FakePydanticModel:
    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "email": {"type": "string"},
                "name": {"type": "string"},
                "account_status": {"type": "string"},
                "score": {"type": "integer"},
                "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "nested": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "zip": {"type": "integer"},
                    },
                },
                "nullable_field": {"type": "string", "nullable": True},
            },
        }


class FakeQuestion:
    def __init__(
        self,
        question_id="Q1",
        question_text="Do the thing",
        expected_answer_schema=None,
    ):
        self.question_id = question_id
        self.question_text = question_text
        self.expected_answer_schema = expected_answer_schema


class FakeTask:
    def __init__(self, description="Default desc", expected_output="Default output"):
        self.description = description
        self.expected_output = expected_output


class FakeAgent:
    def __init__(self, tools=None):
        self.tools = tools or []


class FakeResult:
    def __init__(self, raw=None, pydantic=None):
        self._raw = raw
        self.pydantic = pydantic

    @property
    def raw(self):
        return self._raw


class FakeTool:
    def __init__(self, name, func=None):
        self.name = name
        self.func = func
        self.args_schema = None

    def _run(self, **kwargs):
        return self.func(**kwargs) if self.func else {}


class FakeCrew:
    def __init__(self, agents=None, tasks=None, max_iter=None):
        self.agents = agents or []
        self.tasks = tasks or []
        self.max_iter = max_iter

    def kickoff(self):
        return FakeResult(raw=json.dumps({"should_act": True, "reasoning": "done"}))

    def __deepcopy__(self, memo):
        return FakeCrew(
            agents=list(self.agents),
            tasks=list(self.tasks),
            max_iter=self.max_iter,
        )


class FakeRuntime:
    def __init__(self, fetch_return=None, search_return=None):
        self.fetch_return = fetch_return
        self.search_return = search_return or []
        self.fetch_calls = []
        self.search_calls = []

    def fetch(self, artifact_id):
        self.fetch_calls.append(artifact_id)
        return self.fetch_return

    def search(self, query="", artifact_type=None, limit=10):
        self.search_calls.append((query, artifact_type, limit))
        return self.search_return

    def trajectory(self):
        traj = MagicMock()
        traj.tool_calls = []
        traj.horizon_violations = []
        traj.actor_gate_violations = []
        traj.subsystem_violations = []
        traj.dead_ends_hit = 0
        traj.dead_ends_recovered = 0
        return traj


def make_contract(
    name="test",
    inputs=None,
    allowed_tools=None,
    expected_action=None,
    action_tool="",
    decision_field="should_act",
):
    return TaskContract(
        name=name,
        task_description="Test task",
        preconditions=[],
        inputs=inputs or {},
        allowed_tools=allowed_tools or [],
        expected_action=expected_action,
        action_tool=action_tool,
        decision_field=decision_field,
    )


class TestDefaultForSchema:
    def test_non_dict_returns_none(self):
        assert _default_for_schema(None) is None
        assert _default_for_schema("nope") is None
        assert _default_for_schema(42) is None

    def test_ref_returns_empty_dict(self):
        assert _default_for_schema({"$ref": "#/some/ref"}) == {}

    def test_string_default(self):
        assert _default_for_schema({"type": "string"}) == ""

    def test_number_default(self):
        assert _default_for_schema({"type": "number"}) == 0

    def test_integer_default(self):
        assert _default_for_schema({"type": "integer"}) == 0

    def test_boolean_default(self):
        assert _default_for_schema({"type": "boolean"}) is False

    def test_array_default(self):
        assert _default_for_schema({"type": "array"}) == []

    def test_object_default(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"},
                "active": {"type": "boolean"},
            },
        }
        assert _default_for_schema(schema) == {
            "id": "",
            "count": 0,
            "active": False,
        }

    def test_nested_object_default(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "zip": {"type": "integer"},
                    },
                },
                "name": {"type": "string"},
            },
        }
        assert _default_for_schema(schema) == {
            "address": {"street": "", "zip": 0},
            "name": "",
        }

    def test_unknown_type_returns_none(self):
        assert _default_for_schema({"type": "unknown"}) is None

    def test_no_type_key_defaults_to_string(self):
        assert _default_for_schema({}) == ""


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": "", "b": 0}
        overrides = {"a": "hello"}
        assert _deep_merge(base, overrides) == {"a": "hello", "b": 0}

    def test_nested_merge(self):
        base = {"outer": {"x": "", "y": 0}}
        overrides = {"outer": {"x": "hello"}}
        assert _deep_merge(base, overrides) == {"outer": {"x": "hello", "y": 0}}

    def test_override_adds_new_keys(self):
        base = {"a": ""}
        overrides = {"b": "new", "c": 42}
        assert _deep_merge(base, overrides) == {"a": "", "b": "new", "c": 42}

    def test_override_scalar_wins_over_whole_dict(self):
        base = {"nested": {"x": "", "y": 0}}
        overrides = {"nested": "scalar"}
        assert _deep_merge(base, overrides) == {"nested": "scalar"}

    def test_empty_overrides(self):
        base = {"a": "", "b": 0}
        assert _deep_merge(base, {}) == {"a": "", "b": 0}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": "hello"}) == {"a": "hello"}


class TestFindTool:
    def test_finds_tool_by_name(self):
        tools = [FakeTool("alpha"), FakeTool("beta"), FakeTool("gamma")]
        result = _find_tool("beta", tools)
        assert result is tools[1]

    def test_not_found_returns_none(self):
        tools = [FakeTool("alpha")]
        assert _find_tool("nope", tools) is None

    def test_empty_list_returns_none(self):
        assert _find_tool("anything", []) is None

    def test_tool_without_name_skipped(self):
        tool = object()
        assert _find_tool("anything", [tool]) is None


class TestExtractReturnSchema:
    def test_from_func_return_annotation(self):
        tool = FakeTool(name="test")
        tool.func = lambda: None
        tool.func.__annotations__ = {"return": FakePydanticModel}
        schema = _extract_return_schema(tool)
        assert schema is not None
        assert schema["type"] == "object"
        assert "id" in schema["properties"]

    def test_from_args_schema(self):
        tool = FakeTool(name="test")
        tool.args_schema = FakePydanticModel
        schema = _extract_return_schema(tool)
        assert schema is not None
        assert schema["type"] == "object"

    def test_from_run_method(self):
        class SubTool(FakeTool):
            def _run(self, **kwargs):
                pass

        SubTool._run.__annotations__ = {"return": FakePydanticModel}
        tool = SubTool(name="test")
        schema = _extract_return_schema(tool)
        assert schema is not None
        assert schema["type"] == "object"

    def test_no_schema_found_returns_none(self):
        tool = FakeTool(name="bare")
        assert _extract_return_schema(tool) is None

    def test_exception_during_extraction_returns_none(self):
        tool = FakeTool(name="bad")
        tool.func = lambda: None
        tool.func.__annotations__ = {"return": object()}
        assert _extract_return_schema(tool) is None


class TestBuildFixtureReturn:
    def test_tool_not_found_returns_declared_as_is(self):
        declared = {"email": "jane@acme.com"}
        result = build_fixture_return("nonexistent", declared, [])
        assert result == declared

    def test_no_schema_returns_declared_as_is(self):
        tool = FakeTool(name="bare")
        declared = {"email": "jane@acme.com"}
        result = build_fixture_return("bare", declared, [tool])
        assert result == declared

    def test_merges_declared_into_schema_defaults(self):
        tool = FakeTool(name="fetch")
        tool.func = lambda: None
        tool.func.__annotations__ = {"return": FakePydanticModel}
        declared = {"id": "CUST-001", "email": "jane@acme.com", "name": "Jane"}
        result = build_fixture_return("fetch", declared, [tool])
        assert result["id"] == "CUST-001"
        assert result["email"] == "jane@acme.com"
        assert result["name"] == "Jane"
        assert result["account_status"] == ""
        assert result["score"] == 0
        assert result["active"] is False
        assert result["tags"] == []
        assert result["nested"] == {"street": "", "zip": 0}

    def test_non_dict_defaulted_returns_declared(self):
        class ArrayModel:
            @classmethod
            def model_json_schema(cls):
                return {"type": "array"}

        tool = FakeTool(name="list_tool")
        tool.func = lambda: None
        tool.func.__annotations__ = {"return": ArrayModel}
        declared = {"key": "val"}
        result = build_fixture_return("list_tool", declared, [tool])
        assert result == declared


class TestInferToolVerb:
    def test_explicit_tool_map_wins(self):
        assert _infer_tool_verb("my_tool", {"my_tool": "search"}) == "search"

    def test_search_words(self):
        for word in (
            "search_customers",
            "query_db",
            "find_all",
            "list_items",
            "discover",
        ):
            assert _infer_tool_verb(word) == "search"

    def test_fetch_words(self):
        for word in (
            "fetch_customer",
            "get_record",
            "retrieve_doc",
            "read_file",
            "lookup_key",
        ):
            assert _infer_tool_verb(word) == "fetch"

    def test_default_fetch(self):
        assert _infer_tool_verb("do_something") == "fetch"

    def test_tool_map_missing_key_falls_back(self):
        assert _infer_tool_verb("search_docs", {"other": "fetch"}) == "search"

    def test_none_tool_map(self):
        assert _infer_tool_verb("fetch_data", None) == "fetch"


class TestParseCrewOutput:
    def test_raw_is_dict(self):
        result = FakeResult(raw={"should_act": True})
        out = _parse_crew_output(result, "auto", None)
        assert out == {"should_act": True}

    def test_raw_is_valid_json_string(self):
        result = FakeResult(raw='{"should_act": true, "reasoning": "ok"}')
        out = _parse_crew_output(result, "auto", None)
        assert out == {"should_act": True, "reasoning": "ok"}

    def test_raw_is_invalid_json_fallback_to_reasoning(self):
        result = FakeResult(raw="This is not JSON at all")
        out = _parse_crew_output(result, "auto", None)
        assert "reasoning" in out

    def test_raw_empty_string(self):
        result = FakeResult(raw="")
        out = _parse_crew_output(result, "auto", None)
        assert "reasoning" in out

    def test_result_without_raw_attribute(self):
        result = object()
        out = _parse_crew_output(result, "auto", None)
        assert "reasoning" in out

    def test_pydantic_output_model_dump(self):
        pydantic = MagicMock()
        pydantic.model_dump.return_value = {"should_act": False, "reasoning": "nope"}
        result = FakeResult(raw="ignored", pydantic=pydantic)
        out = _parse_crew_output(result, "pydantic", None)
        assert out == {"should_act": False, "reasoning": "nope"}

    def test_pydantic_output_dict_method(self):
        pydantic = MagicMock()
        del pydantic.model_dump
        pydantic.dict.return_value = {"should_act": True}
        result = FakeResult(raw="ignored", pydantic=pydantic)
        out = _parse_crew_output(result, "pydantic", None)
        assert out == {"should_act": True}

    def test_pydantic_output_fallback_to_dict_constructor(self):
        pydantic = MagicMock()
        del pydantic.model_dump
        del pydantic.dict
        result = FakeResult(raw="ignored", pydantic=pydantic)
        out = _parse_crew_output(result, "pydantic", None)
        assert isinstance(out, dict)

    def test_pydantic_mode_no_pydantic_falls_through(self):
        result = FakeResult(raw='{"should_act": true}')
        out = _parse_crew_output(result, "pydantic", None)
        assert out == {"should_act": True}

    def test_answer_key_extraction(self):
        result = FakeResult(raw='{"outer": {"inner": "val"}, "other": 1}')
        out = _parse_crew_output(result, "auto", "outer")
        assert out == {"inner": "val"}

    def test_answer_key_extraction_nested_in_pydantic(self):
        pydantic = MagicMock()
        pydantic.model_dump.return_value = {"outer": {"inner": "val"}}
        result = FakeResult(raw="ignored", pydantic=pydantic)
        out = _parse_crew_output(result, "pydantic", "outer")
        assert out == {"inner": "val"}

    def test_answer_key_not_found_returns_full_dict(self):
        result = FakeResult(raw='{"a": 1}')
        out = _parse_crew_output(result, "auto", "nonexistent")
        assert out == {"a": 1}

    def test_reasoning_truncated_at_500(self):
        long_text = "x" * 600
        result = FakeResult(raw=long_text)
        out = _parse_crew_output(result, "auto", None)
        assert len(out["reasoning"]) <= 500


class TestBuildCrewaiAgentFn:
    def test_no_contract_raises_runtime_error(self):
        agent_fn = build_crewai_agent_fn("some.module.Crew")
        with pytest.raises(RuntimeError, match="TaskContract"):
            agent_fn(FakeQuestion(), None, None, 5, None)

    def test_contract_passed_returns_callable(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            assert callable(agent_fn)

    def test_fixture_mode_agent_fn_runs(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            allowed = AllowedTool(
                tool_name="fetch_customer",
                entity_arg="customer_id",
                returns={"id": "CUST-001", "email": "jane@acme.com"},
            )
            contract = make_contract(
                inputs={"customer_id": "CUST-001"},
                allowed_tools=[allowed],
            )
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert isinstance(traj.task_id, str)
            assert "should_act" in answer

    def test_fixture_mode_injects_action_when_missing(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("send_email")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            original_kickoff = FakeCrew.kickoff
            FakeCrew.kickoff = lambda self: FakeResult(raw=json.dumps({}))

            try:
                allowed = AllowedTool(
                    tool_name="send_email",
                    action=True,
                    returns={"sent": True},
                )
                contract = make_contract(
                    allowed_tools=[allowed],
                    expected_action=False,
                    action_tool="send_email",
                    decision_field="should_act",
                )
                agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
                traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
                assert answer["should_act"] is False
            finally:
                FakeCrew.kickoff = original_kickoff

    def test_fixture_mode_does_not_overwrite_existing_decision(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("send_email")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            allowed = AllowedTool(tool_name="send_email", action=True, returns={})
            contract = make_contract(
                allowed_tools=[allowed],
                expected_action=False,
                action_tool="send_email",
                decision_field="should_act",
            )
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert answer["should_act"] is True

    def test_corpus_mode_agent_fn_runs(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            runtime = FakeRuntime(fetch_return={"id": "CUST-001"})
            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, runtime)
            assert isinstance(traj.task_id, str)
            assert "should_act" in answer

    def test_max_iter_set_on_crew(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
                max_iter=0,
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            agent_fn(FakeQuestion(), None, None, 25, None)


class TestWrappedToolFactory:
    def test_returns_fixture_data_via_runtime(self):
        tool = FakeTool(name="fetch_customer")
        tool.func = lambda: None
        tool.func.__annotations__ = {"return": FakePydanticModel}

        runtime = FakeRuntime(fetch_return={"id": "CUST-001", "email": "jane@acme.com"})
        allowed = AllowedTool(
            tool_name="fetch_customer",
            entity_arg="customer_id",
            returns={"id": "CUST-001", "email": "jane@acme.com"},
        )
        wrapped = _wrapped_tool_factory(
            "fetch_customer", tool, "fetch", runtime, allowed
        )
        result = wrapped._run(customer_id="CUST-001")
        assert result["id"] == "CUST-001"
        assert result["email"] == "jane@acme.com"

    def test_passes_artifact_id_to_runtime(self):
        tool = FakeTool(name="fetch_customer")
        runtime = FakeRuntime(fetch_return={"id": "CUST-001"})
        allowed = AllowedTool(
            tool_name="fetch_customer",
            entity_arg="customer_id",
            returns={"id": "CUST-001"},
        )
        wrapped = _wrapped_tool_factory(
            "fetch_customer", tool, "fetch", runtime, allowed
        )
        wrapped._run(customer_id="CUST-001")
        assert "CUST-001" in runtime.fetch_calls

    def test_runtime_none_falls_back_to_original_run(self):
        tool = FakeTool(name="fetch_customer")
        captured = {}

        def original_fn(**kwargs):
            captured["called"] = True
            captured["kwargs"] = kwargs
            return {"original": "result"}

        tool._run = original_fn
        allowed = AllowedTool(
            tool_name="fetch_customer",
            returns={"id": "FIXED"},
        )
        wrapped = _wrapped_tool_factory("fetch_customer", tool, "fetch", None, allowed)
        result = wrapped._run(customer_id="CUST-001")
        assert captured.get("called") is True
        assert result == {"original": "result"}

    def test_entity_arg_fallback_to_artifact_id(self):
        tool = FakeTool(name="fetch_customer")
        runtime = FakeRuntime(fetch_return={"id": "CUST-001"})
        allowed = AllowedTool(
            tool_name="fetch_customer",
            entity_arg="",
            returns={"id": "CUST-001"},
        )
        wrapped = _wrapped_tool_factory(
            "fetch_customer", tool, "fetch", runtime, allowed
        )
        wrapped._run(artifact_id="CUST-001")
        assert "CUST-001" in runtime.fetch_calls

    def test_search_verb_routes_to_runtime_search(self):
        tool = FakeTool(name="search_docs")
        runtime = FakeRuntime(search_return=[{"id": "DOC-1"}])
        allowed = AllowedTool(tool_name="search_docs", returns={})
        wrapped = _wrapped_tool_factory("search_docs", tool, "search", runtime, allowed)
        result = wrapped._run(query="test", limit=5)
        assert result == [{"id": "DOC-1"}]
        assert runtime.search_calls == [("test", None, 5)]

    def test_fetch_with_no_artifact_id_returns_empty(self):
        tool = FakeTool(name="fetch_customer")
        runtime = FakeRuntime(fetch_return={"id": "CUST-001"})
        allowed = AllowedTool(
            tool_name="fetch_customer",
            entity_arg="customer_id",
            returns={"id": "CUST-001"},
        )
        wrapped = _wrapped_tool_factory(
            "fetch_customer", tool, "fetch", runtime, allowed
        )
        result = wrapped._run()
        assert result == {}

    def test_fetch_runtime_returns_none_returns_empty(self):
        tool = FakeTool(name="fetch_customer")
        runtime = FakeRuntime(fetch_return=None)
        allowed = AllowedTool(
            tool_name="fetch_customer",
            entity_arg="customer_id",
            returns={"id": "CUST-001"},
        )
        wrapped = _wrapped_tool_factory(
            "fetch_customer", tool, "fetch", runtime, allowed
        )
        result = wrapped._run(customer_id="CUST-001")
        assert result == {}

    def test_default_verb_falls_back_to_original_run(self):
        tool = FakeTool(name="do_something")

        def original_fn(**kwargs):
            return {"custom": "result"}

        tool._run = original_fn
        runtime = FakeRuntime()
        allowed = None
        wrapped = _wrapped_tool_factory(
            "do_something", tool, "unknown_verb", runtime, allowed
        )
        result = wrapped._run()
        assert result == {"custom": "result"}


class TestAgentFnTrajectoryPopulation:
    def test_runtime_trajectory_populated_into_trajectory(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            runtime = FakeRuntime(fetch_return={"id": "CUST-001"})
            runtime.traj_mock = MagicMock()
            runtime.traj_mock.tool_calls = [
                {"tool": "fetch_customer", "args": {"customer_id": "CUST-001"}}
            ]
            runtime.traj_mock.horizon_violations = [
                {"artifact_id": "CUST-002", "reason": "outside visibility cone"}
            ]
            runtime.traj_mock.actor_gate_violations = [
                {"actor": "bob", "artifact": "CUST-003"}
            ]
            runtime.traj_mock.subsystem_violations = [
                {"artifact_id": "CUST-004", "subsystem": "finance"}
            ]
            runtime.traj_mock.dead_ends_hit = 3
            runtime.traj_mock.dead_ends_recovered = 1
            runtime.trajectory = lambda: runtime.traj_mock

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, runtime)

            assert len(traj.tool_calls) == 1
            assert traj.tool_calls[0]["tool"] == "fetch_customer"
            assert len(traj.horizon_violations) == 1
            assert traj.horizon_violations[0]["artifact_id"] == "CUST-002"
            assert len(traj.actor_gate_violations) == 1
            assert len(traj.subsystem_violations) == 1
            assert traj.dead_ends_hit == 3
            assert traj.dead_ends_recovered == 1

    def test_runtime_none_skips_trajectory_population(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)

            assert traj.tool_calls == []
            assert traj.horizon_violations == 0
            assert traj.actor_gate_violations == 0
            assert traj.subsystem_violations == 0
            assert traj.dead_ends_hit == 0
            assert traj.dead_ends_recovered == 0

    def test_trajectory_final_answer_set(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)

            assert traj.final_answer is not None
            assert traj.final_answer["should_act"] is True
            assert traj.task_id == "Q1"


class TestBuildCrewaiAgentFnEdgeCases:
    def test_crew_with_no_tasks(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert "should_act" in answer

    def test_crew_with_no_agents(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(agents=[], tasks=[FakeTask()])
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert "should_act" in answer

    def test_crew_with_agent_no_tools(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert "should_act" in answer

    def test_runtime_none_in_corpus_mode(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert "should_act" in answer

    def test_multiple_agents_all_wrapped(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            mock_crew = FakeCrew(
                agents=[
                    FakeAgent(tools=[FakeTool("fetch_customer")]),
                    FakeAgent(tools=[FakeTool("fetch_email_draft")]),
                ],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            runtime = FakeRuntime(fetch_return={"id": "X"})
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, runtime)
            assert "should_act" in answer

    def test_expected_answer_schema_appended(self):
        task = FakeTask()
        mock_crew = FakeCrew(
            agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
            tasks=[task],
        )

        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew",
            return_value=mock_crew,
        ):
            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            schema = {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
            question = FakeQuestion(expected_answer_schema=schema)
            traj, answer = agent_fn(question, None, None, 5, None)
            assert "MUST be valid JSON" in task.expected_output

    def test_inputs_injected_into_task_description(self):
        task = FakeTask(description="Original description")
        mock_crew = FakeCrew(
            agents=[FakeAgent(tools=[FakeTool("fetch_customer")])],
            tasks=[task],
        )

        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew",
            return_value=mock_crew,
        ):
            allowed = AllowedTool(
                tool_name="fetch_customer",
                entity_arg="customer_id",
                returns={"id": "CUST-001"},
            )
            contract = make_contract(
                inputs={"customer_id": "CUST-001", "draft_id": "DRAFT-001"},
                allowed_tools=[allowed],
            )
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert "customer_id" in task.description
            assert "CUST-001" in task.description
            assert "draft_id" in task.description

    def test_tool_without_name_attribute(self):
        with patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew"
        ) as mock_load:
            bare = object()
            mock_crew = FakeCrew(
                agents=[FakeAgent(tools=[bare])],
                tasks=[FakeTask()],
            )
            mock_load.return_value = mock_crew

            contract = make_contract()
            agent_fn = build_crewai_agent_fn("some.module.Crew", contract=contract)
            traj, answer = agent_fn(FakeQuestion(), None, None, 5, None)
            assert "should_act" in answer
