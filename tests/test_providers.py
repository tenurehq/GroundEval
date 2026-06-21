import json
from unittest.mock import MagicMock

import pytest

from groundeval.core import EvalQuestion
from groundeval.providers import (
    _build_system_prompt,
    _dispatch_tool,
    _extract_json_from_text,
    _to_strict_schema,
    _build_submit_answer_tool,
)


def test_build_system_prompt_contains_schema():
    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="Could Alice know?",
        difficulty="easy",
        ground_truth={},
    )
    prompt = _build_system_prompt(q, context=None)
    assert "PERSPECTIVE" in prompt
    assert "submit_answer" in prompt


def test_build_system_prompt_perspective_context():
    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        actor="alice",
        actor_role="engineer",
        as_of_time="2026-01-15T00:00:00",
    )
    prompt = _build_system_prompt(q, context=None)
    assert "alice" in prompt
    assert "engineer" in prompt
    assert "2026-01-15" in prompt


def test_dispatch_tool_fetch():
    runtime = MagicMock()
    runtime.fetch.return_value = {"id": "a1"}
    result = _dispatch_tool("fetch_artifact", {"artifact_id": "a1"}, runtime)
    assert result == {"id": "a1"}
    runtime.fetch.assert_called_once_with("a1")


def test_dispatch_tool_search():
    runtime = MagicMock()
    runtime.search.return_value = [{"id": "a1"}]
    result = _dispatch_tool(
        "search_artifacts",
        {"query": "hello", "artifact_type": "jira", "limit": 5},
        runtime,
    )
    assert result == [{"id": "a1"}]
    runtime.search.assert_called_once_with(query="hello", artifact_type="jira", limit=5)


def test_dispatch_tool_no_runtime():
    result = _dispatch_tool("fetch_artifact", {"artifact_id": "a1"}, None)
    assert result == {"error": "No runtime available for tool calls"}


def test_dispatch_tool_submit_answer():
    result = _dispatch_tool("submit_answer", {}, MagicMock())
    assert result == {"status": "answer_recorded"}


def test_extract_json_from_text_single_block():
    text = 'some text {"a": 1} more'
    assert _extract_json_from_text(text) == {"a": 1}


def test_extract_json_from_text_prefers_last_block():
    text = '{"a": 1} and then {"b": 2}'
    assert _extract_json_from_text(text) == {"b": 2}


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


def test_build_submit_answer_tool_structure():
    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
    )
    tool = _build_submit_answer_tool(q)
    assert tool["name"] == "submit_answer"
    assert tool.get("strict") is True
    assert "answer" in tool["input_schema"]["properties"]


def test_extract_json_from_text_no_json():
    assert _extract_json_from_text("no json here at all") == {}


def test_extract_json_from_text_malformed_then_valid():
    text = 'bad { not json } then {"good": 1}'
    assert _extract_json_from_text(text) == {"good": 1}


def test_dispatch_tool_unknown_tool():
    result = _dispatch_tool("unknown_thing", {"x": 1}, MagicMock())
    assert "error" in result


def test_build_system_prompt_no_context_includes_tools():
    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
    )
    prompt = _build_system_prompt(q, context=None)
    assert "fetch_artifact" in prompt or "search_artifacts" in prompt


def test_to_strict_schema_already_strict():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string"}},
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["x"].get("additionalProperties") is None


def test_build_submit_answer_tool_uses_question_schema():
    custom = {
        "type": "object",
        "properties": {"custom_field": {"type": "boolean"}},
        "required": ["custom_field"],
    }
    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        expected_answer_schema=custom,
    )
    tool = _build_submit_answer_tool(q)
    assert "custom_field" in tool["input_schema"]["properties"]["answer"]["properties"]
