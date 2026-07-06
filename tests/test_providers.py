from unittest.mock import MagicMock, patch

import pytest

from groundeval.core import AgentTrajectory
from groundeval.providers import (
    ModelProvider,
    _build_submit_answer_tool,
    _build_system_prompt,
    _dispatch_tool,
    _extract_json_from_text,
    _to_strict_schema,
    build_agent_fn,
)


def test_to_strict_schema_adds_additional_properties_false_recursively():
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                },
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                },
            },
            "name": {"type": "string"},
        },
    }
    strict = _to_strict_schema(schema)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["nested"]["additionalProperties"] is False
    assert strict["properties"]["items"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in strict["properties"]["name"]


def test_to_strict_schema_object_without_properties_is_unchanged():
    schema = {"type": "object"}
    assert _to_strict_schema(schema) == {"type": "object"}


def test_to_strict_schema_does_not_mutate_input():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "object", "properties": {"y": {"type": "string"}}}
        },
    }
    original = {
        "type": "object",
        "properties": {
            "x": {"type": "object", "properties": {"y": {"type": "string"}}}
        },
    }
    _to_strict_schema(schema)
    assert schema == original


def test_build_submit_answer_tool_uses_default_schema():
    tool = _build_submit_answer_tool()
    assert tool["name"] == "submit_answer"
    assert tool["strict"] is True
    assert "answer" in tool["input_schema"]["properties"]
    assert tool["input_schema"]["additionalProperties"] is False


def test_build_submit_answer_tool_uses_custom_schema():
    tool = _build_submit_answer_tool({
        "type": "object",
        "properties": {"custom": {"type": "boolean"}},
    })
    answer_schema = tool["input_schema"]["properties"]["answer"]
    assert "custom" in answer_schema["properties"]


def test_build_system_prompt_with_and_without_context():
    prompt1 = _build_system_prompt("Question?", context=None, max_steps=5)
    assert "fetch_artifact" in prompt1
    assert "submit_answer" in prompt1

    prompt2 = _build_system_prompt("Question?", context="Artifact text", max_steps=7)
    assert "Artifact text" in prompt2
    assert "7 turns" in prompt2


def test_build_system_prompt_includes_actor_metadata():
    prompt = _build_system_prompt(
        "Question?",
        context=None,
        actor="alice",
        actor_role="engineer",
        as_of_time="2026-01-15T00:00:00",
    )
    assert "alice" in prompt
    assert "engineer" in prompt
    assert "2026-01-15" in prompt


def test_build_system_prompt_uses_custom_schema_text():
    prompt = _build_system_prompt(
        "Question?",
        context=None,
        expected_answer_schema={
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        },
    )
    assert "flag" in prompt


def test_dispatch_tool_submit_answer():
    assert _dispatch_tool("submit_answer", {}, None) == {"status": "answer_recorded"}


def test_dispatch_tool_fetch_and_search():
    runtime = MagicMock()
    runtime.fetch.return_value = {"id": "a1"}
    runtime.search.return_value = [{"id": "a1"}]

    assert _dispatch_tool("fetch_artifact", {"artifact_id": "a1"}, runtime) == {
        "id": "a1"
    }
    assert _dispatch_tool(
        "search_artifacts", {"query": "q", "artifact_type": "crm", "limit": 5}, runtime
    ) == [{"id": "a1"}]


def test_dispatch_tool_fetch_returns_empty_dict_for_none():
    runtime = MagicMock()
    runtime.fetch.return_value = None
    assert _dispatch_tool("fetch_artifact", {"artifact_id": "missing"}, runtime) == {}


def test_dispatch_tool_search_defaults():
    runtime = MagicMock()
    runtime.search.return_value = []
    assert _dispatch_tool("search_artifacts", {"query": "x"}, runtime) == []
    runtime.search.assert_called_once_with(query="x", artifact_type=None, limit=10)


def test_dispatch_tool_missing_runtime_and_unknown_tool():
    assert _dispatch_tool("fetch_artifact", {"artifact_id": "a1"}, None) == {
        "error": "No runtime available for tool calls"
    }
    out = _dispatch_tool("unknown", {}, MagicMock())
    assert "error" in out


def test_dispatch_tool_generate_email():
    out = _dispatch_tool("generate_email", {"customer_id": "cust-1"}, MagicMock())
    assert out["customer_id"] == "cust-1"
    assert "Jenny Fields" in out["subject"]


def test_extract_json_from_text_behavior():
    assert _extract_json_from_text("") == {}
    assert _extract_json_from_text("not json") == {}

    obj = _extract_json_from_text('{"a": 1}')
    assert isinstance(obj, dict)
    assert obj["a"] == 1


def test_extract_json_from_text_array_returns_empty_dict():
    assert _extract_json_from_text("[1,2,3]") == {}


class _FakeProvider:
    def __init__(self):
        self.kwargs = None

    def run_agent(self, **kwargs):
        self.kwargs = kwargs
        return AgentTrajectory(task_id=kwargs["task_id"]), {"ok": True}


def test_build_agent_fn_forwards_expected_fields():
    provider = _FakeProvider()
    agent_fn = build_agent_fn(provider)

    class Q:
        question_id = "q1"
        question_text = "Question?"
        expected_answer_schema = {"type": "object"}
        actor = "alice"
        actor_role = "engineer"
        as_of_time = "2026-01-01T00:00:00"

    runtime = MagicMock()
    traj, answer = agent_fn(Q(), context=None, tools=[], max_steps=3, runtime=runtime)

    assert provider.kwargs["task_id"] == "q1"
    assert provider.kwargs["question_text"] == "Question?"
    assert provider.kwargs["runtime"] is runtime
    assert provider.kwargs["actor"] == "alice"
    assert answer == {"ok": True}
    assert traj.task_id == "q1"


def test_build_agent_fn_missing_optional_fields_defaults_to_none():
    provider = _FakeProvider()
    agent_fn = build_agent_fn(provider)

    class Q:
        question_id = "q2"
        question_text = "Question?"
        expected_answer_schema = None

    traj, answer = agent_fn(
        Q(), context=None, tools=["ignored"], max_steps=2, runtime=None
    )
    assert provider.kwargs["actor"] is None
    assert provider.kwargs["actor_role"] is None
    assert provider.kwargs["as_of_time"] is None


def test_model_provider_from_config_openai(monkeypatch):
    fake = object()
    with patch("groundeval.providers.OpenAIProvider", return_value=fake) as mock_cls:
        out = ModelProvider.from_config({"provider": "openai", "model": "gpt-4o"})
    assert out is fake
    mock_cls.assert_called_once()


def test_model_provider_from_config_anthropic(monkeypatch):
    fake = object()
    with patch("groundeval.providers.AnthropicProvider", return_value=fake) as mock_cls:
        out = ModelProvider.from_config({
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
        })
    assert out is fake
    mock_cls.assert_called_once()


def test_model_provider_from_config_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    fake = object()
    with patch("groundeval.providers.OpenAIProvider", return_value=fake) as mock_cls:
        ModelProvider.from_config({"provider": "openai"})
    assert mock_cls.call_args.kwargs["api_key"] == "secret"


def test_model_provider_from_config_custom_provider_path():
    class CustomProvider:
        @classmethod
        def from_config(cls, cfg):
            return "custom-provider"

    fake_module = MagicMock()
    fake_module.CustomProvider = CustomProvider

    with patch("importlib.import_module", return_value=fake_module):
        out = ModelProvider.from_config({
            "provider": "other",
            "provider_path": "pkg.CustomProvider",
        })
    assert out == "custom-provider"


def test_model_provider_from_config_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        ModelProvider.from_config({"provider": "nope"})
