import json
from unittest.mock import MagicMock

import pytest

from groundeval.core import AgentTrajectory
from groundeval.providers import (
    _build_system_prompt,
    _dispatch_tool,
    _extract_json_from_text,
    _to_strict_schema,
    _build_submit_answer_tool,
    build_agent_fn,
)


def test_build_system_prompt_contains_schema():
    prompt = _build_system_prompt(
        question_text="What is the status?",
        context=None,
    )
    assert "TASK" in prompt
    assert "submit_answer" in prompt
    assert "preconditions_verified" in prompt


def test_build_system_prompt_includes_actor_details():
    prompt = _build_system_prompt(
        question_text="Verify preconditions.",
        context=None,
        actor="alice",
        actor_role="engineer",
        as_of_time="2026-01-15T00:00:00",
    )
    assert "alice" in prompt
    assert "engineer" in prompt
    assert "2026-01-15" in prompt


def test_build_system_prompt_no_context_includes_tools():
    prompt = _build_system_prompt(
        question_text="Check this.",
        context=None,
    )
    assert "fetch_artifact" in prompt or "search_artifacts" in prompt


def test_build_system_prompt_with_context_shows_context():
    prompt = _build_system_prompt(
        question_text="Check this.",
        context="Some artifact text here.",
    )
    assert "Some artifact text here" in prompt
    assert "Available context" in prompt


def test_build_system_prompt_custom_schema():
    custom = {
        "type": "object",
        "properties": {"custom_field": {"type": "boolean"}},
    }
    prompt = _build_system_prompt(
        question_text="Check.",
        context=None,
        expected_answer_schema=custom,
    )
    assert "custom_field" in prompt


def test_build_system_prompt_defaults_to_task_schema():
    prompt = _build_system_prompt(
        question_text="Check.",
        context=None,
    )
    assert "preconditions_verified" in prompt


def test_dispatch_tool_fetch_found():
    runtime = MagicMock()
    runtime.fetch.return_value = {"id": "a1"}
    result = _dispatch_tool("fetch_artifact", {"artifact_id": "a1"}, runtime)
    assert result == {"id": "a1"}
    runtime.fetch.assert_called_once_with("a1")


def test_dispatch_tool_fetch_not_found():
    runtime = MagicMock()
    runtime.fetch.return_value = None
    result = _dispatch_tool("fetch_artifact", {"artifact_id": "missing"}, runtime)
    assert result == {}


def test_dispatch_tool_search():
    runtime = MagicMock()
    runtime.search.return_value = [{"id": "a1"}, {"id": "a2"}]
    result = _dispatch_tool(
        "search_artifacts",
        {"query": "hello", "artifact_type": "jira", "limit": 5},
        runtime,
    )
    assert result == [{"id": "a1"}, {"id": "a2"}]
    runtime.search.assert_called_once_with(query="hello", artifact_type="jira", limit=5)


def test_dispatch_tool_search_defaults():
    runtime = MagicMock()
    runtime.search.return_value = []
    result = _dispatch_tool(
        "search_artifacts",
        {"query": "q"},
        runtime,
    )
    assert result == []
    runtime.search.assert_called_once_with(query="q", artifact_type=None, limit=10)


def test_dispatch_tool_no_runtime():
    result = _dispatch_tool("fetch_artifact", {"artifact_id": "a1"}, None)
    assert result == {"error": "No runtime available for tool calls"}


def test_dispatch_tool_submit_answer():
    result = _dispatch_tool("submit_answer", {}, MagicMock())
    assert result == {"status": "answer_recorded"}


def test_dispatch_tool_unknown_tool():
    result = _dispatch_tool("unknown_thing", {"x": 1}, MagicMock())
    assert "error" in result


def test_dispatch_tool_generate_email():
    runtime = MagicMock()
    result = _dispatch_tool("generate_email", {"customer_id": "cust_42"}, runtime)
    assert result["customer_id"] == "cust_42"
    assert result["customer_name"] == "Jenny Fields"
    assert result["email"] == "jenny@gmail.com"
    assert result["subject"] == "Exclusive offer for Jenny Fields"
    assert "special offer" in result["body"]


def test_dispatch_tool_generate_email_default_customer_id():
    runtime = MagicMock()
    result = _dispatch_tool("generate_email", {}, runtime)
    assert result["customer_id"] == ""


def test_dispatch_tool_unknown_tool_message():
    runtime = MagicMock()
    result = _dispatch_tool("nonexistent_tool", {"a": 1}, runtime)
    assert "error" in result
    assert "unknown tool" in result["error"]


def test_extract_json_from_text_empty_string():
    assert _extract_json_from_text("") == {}


def test_extract_json_from_text_only_whitespace():
    assert _extract_json_from_text("   \n\t  ") == {}


def test_extract_json_from_text_nested_braces():
    text = 'prefix {"a": {"b": {"c": 1}}} suffix'
    result = _extract_json_from_text(text)
    assert result == {"b": {"c": 1}} or result == {"a": {"b": {"c": 1}}}


def test_to_strict_schema_array_of_primitives():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False
    assert "additionalProperties" not in strict["properties"]["tags"]


def test_to_strict_schema_empty_schema():
    schema = {"type": "object"}
    strict = _to_strict_schema(schema)
    assert strict == {"type": "object"}


def test_to_strict_schema_object_without_properties():
    schema = {
        "type": "object",
        "properties": {
            "data": {"type": "object"},
        },
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False
    assert "additionalProperties" not in strict["properties"]["data"]


def test_build_system_prompt_with_actor_no_role_or_time():
    prompt = _build_system_prompt(
        question_text="Verify preconditions.",
        context=None,
        actor="alice",
    )
    assert "alice" in prompt
    assert "unknown" in prompt


def test_build_system_prompt_max_steps_reflected():
    prompt = _build_system_prompt(
        question_text="Check.",
        context=None,
        max_steps=7,
    )
    assert "7 turns" in prompt


def test_build_submit_answer_tool_schema_required_fields():
    tool = _build_submit_answer_tool()
    assert "answer" in tool["input_schema"]["required"]
    assert tool["input_schema"]["additionalProperties"] is False


def test_build_submit_answer_tool_custom_schema_strict():
    custom = {
        "type": "object",
        "properties": {
            "nested_obj": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
            },
        },
    }
    tool = _build_submit_answer_tool(expected_answer_schema=custom)
    answer_schema = tool["input_schema"]["properties"]["answer"]
    assert answer_schema["properties"]["nested_obj"]["additionalProperties"] is False


def test_build_agent_fn_tools_arg_is_ignored():
    fake = _FakeProvider()
    agent_fn = build_agent_fn(fake)

    class FakeQuestion:
        question_id = "task_999"
        question_text = "Ignored tools test."
        expected_answer_schema = None
        actor = None
        actor_role = None
        as_of_time = None

    fake_question = FakeQuestion()
    dummy_tools = [{"name": "should_be_ignored"}]

    traj, answer = agent_fn(
        fake_question, context=None, tools=dummy_tools, max_steps=3, runtime=None
    )

    assert fake.last_call["task_id"] == "task_999"


def test_extract_json_from_text_single_block():
    text = 'some text {"a": 1} more'
    assert _extract_json_from_text(text) == {"a": 1}


def test_extract_json_from_text_no_json():
    assert _extract_json_from_text("no json here at all") == {}


def test_extract_json_from_text_malformed_then_valid():
    text = 'bad { not json } then {"good": 1}'
    assert _extract_json_from_text(text) == {"good": 1}


def test_extract_json_from_text_nested_objects():
    text = '{"outer": {"inner": "value"}} trailing'
    result = _extract_json_from_text(text)
    assert result == {"inner": "value"} or "inner" in str(result)


def test_extract_json_from_text_array():
    text = "[1, 2, 3]"
    result = _extract_json_from_text(text)
    assert result == [1, 2, 3] or result == {}


def test_to_strict_schema_adds_additionalProperties():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        },
    }
    strict = _to_strict_schema(schema)
    assert strict.get("additionalProperties") is False
    assert strict["properties"]["nested"].get("additionalProperties") is False


def test_to_strict_schema_already_strict():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string"}},
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False


def test_to_strict_schema_deep_nesting():
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {"leaf": {"type": "boolean"}},
                    },
                },
            },
        },
    }
    strict = _to_strict_schema(schema)
    assert strict["properties"]["level1"]["additionalProperties"] is False
    assert (
        strict["properties"]["level1"]["properties"]["level2"]["additionalProperties"]
        is False
    )


def test_to_strict_schema_leaves_non_objects_alone():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False
    # String and integer schemas don't need additionalProperties
    assert "additionalProperties" not in strict["properties"]["name"]


def test_to_strict_schema_handles_arrays():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            },
        },
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["items"]["items"]["additionalProperties"] is False


def test_build_submit_answer_tool_structure():
    tool = _build_submit_answer_tool()
    assert tool["name"] == "submit_answer"
    assert tool.get("strict") is True
    assert "answer" in tool["input_schema"]["properties"]


def test_build_submit_answer_tool_uses_task_schema_by_default():
    tool = _build_submit_answer_tool()
    answer_schema = tool["input_schema"]["properties"]["answer"]
    assert "preconditions_verified" in answer_schema["properties"]
    assert "all_preconditions_pass" in answer_schema.get("required", [])


def test_build_submit_answer_tool_uses_custom_schema():
    custom = {
        "type": "object",
        "properties": {"custom_field": {"type": "boolean"}},
        "required": ["custom_field"],
    }
    tool = _build_submit_answer_tool(expected_answer_schema=custom)
    answer_schema = tool["input_schema"]["properties"]["answer"]
    assert "custom_field" in tool["input_schema"]["properties"]["answer"]["properties"]
    assert "custom_field" in answer_schema.get("required", [])


class _FakeProvider:
    """Minimal provider stub for testing build_agent_fn."""

    def __init__(self):
        self.last_call = {}

    def run_agent(self, **kwargs):
        self.last_call = kwargs
        traj = AgentTrajectory(task_id=kwargs.get("task_id", "unknown"))
        return traj, {"result": "ok"}


def test_build_agent_fn_forwards_all_fields():
    fake = _FakeProvider()
    agent_fn = build_agent_fn(fake)

    class FakeQuestion:
        question_id = "task_123"
        question_text = "Verify this."
        expected_answer_schema = {"type": "object"}
        actor = "alice"
        actor_role = "engineer"
        as_of_time = "2026-01-15"

    fake_question = FakeQuestion()
    runtime_mock = MagicMock()

    traj, answer = agent_fn(
        fake_question, context=None, tools=[], max_steps=8, runtime=runtime_mock
    )

    assert fake.last_call["task_id"] == "task_123"
    assert fake.last_call["question_text"] == "Verify this."
    assert fake.last_call["context"] is None
    assert fake.last_call["runtime"] is runtime_mock
    assert fake.last_call["max_steps"] == 8
    assert fake.last_call["expected_answer_schema"] == {"type": "object"}
    assert fake.last_call["actor"] == "alice"
    assert fake.last_call["actor_role"] == "engineer"
    assert fake.last_call["as_of_time"] == "2026-01-15"


def test_build_agent_fn_missing_optional_fields():
    fake = _FakeProvider()
    agent_fn = build_agent_fn(fake)

    class FakeQuestion:
        question_id = "task_456"
        question_text = "Minimal question."
        expected_answer_schema = None

    fake_question = FakeQuestion()

    traj, answer = agent_fn(
        fake_question, context=None, tools=[], max_steps=3, runtime=None
    )

    assert fake.last_call["task_id"] == "task_456"
    assert fake.last_call["actor"] is None
    assert fake.last_call["actor_role"] is None
    assert fake.last_call["as_of_time"] is None
