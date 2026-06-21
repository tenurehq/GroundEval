"""
Model provider layer. Two providers built-in: Anthropic and OpenAI.
Extend by subclassing ModelProvider.

TWO DISTINCT USES
-----------------
1. Prose generation (question text rewriting during `generate`)
   - Single turn, short completion, no tools
   - Called via provider.complete(prompt, max_tokens)

2. Agent loop (running the model against eval questions during `eval`)
   - Multi-turn with tool calls (fetch_artifact, search_artifacts)
   - Structured JSON final answer required (enforced via expected_answer_schema)
   - Called via provider.run_agent(question, context, runtime, max_steps)
   - Returns (AgentTrajectory, dict)


ANTHROPIC IMPLEMENTATION NOTES
-------------------------------
The Anthropic provider uses two SDK features instead of a hand-rolled loop:

- Tool Runner (`client.beta.messages.tool_runner()`): handles the
  request/response cycle, conversation state, and tool dispatch. We define
  `fetch_artifact` and `search_artifacts` as `@beta_tool`-decorated closures
  over the question's `GatedRuntime` and let the runner drive the loop. We
  iterate manually (rather than draining the whole runner) so we can stop
  the instant `submit_answer` is called and enforce `max_steps` ourselves.
- Strict tool use (`strict: true` on `submit_answer`): the answer schema
  varies per question (`expected_answer_schema`, falling back to
  `ANSWER_SCHEMAS[question_type]`), so we build the `submit_answer` tool
  definition per-question rather than reusing one static definition, and
  compile it into a strict-compliant schema (`additionalProperties: false`
  injected at every object level) before sending it. This guarantees the
  `answer` dict Claude returns already matches the expected schema, so we
  no longer need to regex-scrape JSON out of free text on this path.

The OpenAI provider is unaffected by either of the above — both are
Anthropic-specific SDK/API features — and keeps its original hand-rolled
loop and the `_extract_json_from_text` fallback.

`expected_answer_schema` / `ANSWER_SCHEMAS` remain the single, provider-
agnostic contract used for scoring and for every other provider. Strict mode
only changes how the Anthropic provider *enforces* that same schema; it does
not introduce an Anthropic-specific schema.

LAST-STEP FORCED ANSWER
-----------------------
Both providers now inject a final-answer tool on the last allowed step,
mirroring the OrgForge harness pattern:

- Anthropic: rebuilds the tool runner with only `submit_answer` available
  and injects a user message instructing the model this is its last turn.
- OpenAI: filters `tools` to only `submit_answer` and uses model-specific
  `tool_choice`. Native OpenAI models work with `"auto"` and a single tool.
  Models routed through Azure that are actually Mistral/Qwen/Llama behind
  the endpoint need `"required"` to force a tool call when only one tool
  is available.

`budget_exceeded` is set on the trajectory when the loop exhausts
`max_steps` without `answer_submitted` being true.

"""

from __future__ import annotations

import copy
from enum import Enum
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, cast

from .core import (
    AgentTrajectory,
    EvalQuestion,
    GatedRuntime,
    ANSWER_SCHEMAS,
)

logger = logging.getLogger("groundeval.providers")


_TOOLS_FETCH_AND_SEARCH = [
    {
        "name": "fetch_artifact",
        "description": (
            "Retrieve a single artifact by its ID. "
            "Returns the artifact dict or null if not found or gated."
        ),
        "input_schema": {
            "type": "object",
            "required": ["artifact_id"],
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "The artifact ID to fetch.",
                }
            },
        },
    },
    {
        "name": "search_artifacts",
        "description": (
            "Full-text search over the artifact corpus. "
            "Returns a list of matching artifact dicts. "
            "Use filters to narrow results when you know the subsystem, "
            "actor, or time window you're interested in."
        ),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms or artifact ID.",
                },
                "artifact_type": {
                    "type": "string",
                    "description": (
                        "Optional subsystem filter. Examples: 'jira', 'confluence', "
                        "'slack', 'email', 'zendesk', 'git'. Restricts results to "
                        "artifacts from the specified subsystem."
                    ),
                },
                "actor": {
                    "type": "string",
                    "description": (
                        "Optional actor filter. Only return artifacts associated "
                        "with this actor. Useful when investigating what a specific "
                        "person did or had access to."
                    ),
                },
                "from_date": {
                    "type": "string",
                    "description": (
                        "Optional lower bound on artifact timestamp. "
                        "ISO date (YYYY-MM-DD) or full ISO-8601 string. "
                        "Only return artifacts created on or after this date. "
                        "Combine with the system's time horizon to narrow to "
                        "a specific investigation window."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10, max 50).",
                    "default": 10,
                },
            },
        },
    },
]

_TOOLS_OPENAI_BASE = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in _TOOLS_FETCH_AND_SEARCH
]


def _build_tools_for_corpus(
    subsystems: list[str],
) -> list[dict]:
    """Build fetch_artifact and search_artifacts tool definitions with the
    actual subsystem list from the corpus baked into artifact_type's enum."""
    valid_subsystems = sorted(set(s for s in subsystems if s))
    enum_desc = (
        f"Must be one of: {', '.join(valid_subsystems)}."
        if valid_subsystems
        else "No subsystems available."
    )

    return [
        {
            "name": "fetch_artifact",
            "description": (
                "Retrieve a single artifact by its ID. "
                "Returns the artifact dict or null if not found or gated."
            ),
            "input_schema": {
                "type": "object",
                "required": ["artifact_id"],
                "properties": {
                    "artifact_id": {
                        "type": "string",
                        "description": "The artifact ID to fetch.",
                    }
                },
            },
        },
        {
            "name": "search_artifacts",
            "description": (
                "Full-text search over the artifact corpus. "
                "Returns a list of matching artifact dicts. "
                "You MUST specify the subsystem to search within."
            ),
            "input_schema": {
                "type": "object",
                "required": ["query", "artifact_type"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms or artifact ID.",
                    },
                    "artifact_type": {
                        "type": "string",
                        "enum": valid_subsystems,
                        "description": (
                            f"REQUIRED. The subsystem to search within. {enum_desc}"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10, max 50).",
                        "default": 10,
                    },
                },
            },
        },
    ]


def _build_openai_tools(subsystems: list[str]) -> list[dict]:
    """Build OpenAI-compatible tool definitions with dynamic subsystem enum."""
    tools = _build_tools_for_corpus(subsystems)
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Compile an ANSWER_SCHEMAS-style JSON Schema into a strict-tool-use
    compliant schema.

    Strict mode (https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)
    requires `additionalProperties: false` on every object node, not just
    the root. ANSWER_SCHEMAS / expected_answer_schema are written as plain
    JSON Schema for general provider-agnostic use, so we deep-copy and
    inject that constraint here rather than requiring every schema author
    to know about Anthropic's strict-mode requirements.
    """
    compiled = copy.deepcopy(schema)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node.setdefault("additionalProperties", False)
                for prop_schema in node["properties"].values():
                    _walk(prop_schema)
            elif "items" in node:
                _walk(node["items"])
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(compiled)
    return compiled


def _build_submit_answer_tool(question: EvalQuestion) -> dict[str, Any]:
    """
    Build the submit_answer tool definition for this question, with
    strict: true so the `answer` field is guaranteed to match the
    question's expected_answer_schema (or the track default) before it
    ever reaches us.
    """
    schema = question.expected_answer_schema or ANSWER_SCHEMAS.get(
        question.question_type, {}
    )
    answer_schema = _to_strict_schema(schema) if schema else {"type": "object"}

    return {
        "name": "submit_answer",
        "description": (
            "Submit your final answer. Call this exactly once when you are done "
            "researching. The answer must conform to the expected schema."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": answer_schema},
            "additionalProperties": False,
        },
    }


class ModelProvider(ABC):
    """
    Base class for model providers.

    Subclass this to add new providers (Gemini, Mistral, etc.).
    The two methods to implement are complete() and run_agent().
    """

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.temperature = float(kwargs.get("temperature", 0.0))
        self.max_tokens = int(kwargs.get("max_tokens", 1024))

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 2056) -> str:
        """
        Single-turn completion. Used for question prose generation.
        Returns the response text.
        """
        ...

    @abstractmethod
    def run_agent(
        self,
        question: EvalQuestion,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        """
        Run the agent loop against a single question.

        - context: pre-built string of artifact text (context-injection mode)
        - runtime: GatedRuntime providing gated fetch/search (tool mode)
        Exactly one of context or runtime will be non-None.

        Returns (trajectory, final_answer_dict).
        The trajectory's tool_calls will be populated by GatedRuntime if
        runtime is provided; the caller in run.py merges them back.
        """
        ...

    @classmethod
    def from_config(cls, cfg: dict) -> ModelProvider:
        """Build the right provider from a config dict."""
        provider_name = cfg.get("provider", "anthropic").lower()
        model = cfg.get("model", None)
        default_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(provider_name, "")
        api_key = cfg.get("api_key") or os.environ.get(
            cfg.get("api_key_env", default_env)
        )
        kwargs = {
            "temperature": cfg.get("temperature", 0.0),
            "max_tokens": cfg.get("max_tokens", 1024),
            "base_url": cfg.get("base_url"),
            "max_retries": cfg.get("max_retries", 3),
        }

        if provider_name == "anthropic":
            default_model = "claude-sonnet-4-6"
            return AnthropicProvider(
                model=model or default_model,
                api_key=api_key,
                **kwargs,
            )
        elif provider_name == "openai":
            default_model = "gpt-4o"
            return OpenAIProvider(
                model=model or default_model,
                api_key=api_key,
                **kwargs,
            )
        elif "provider_path" in cfg:
            import importlib

            path = cfg["provider_path"]
            module_path, class_name = path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            provider_cls = getattr(mod, class_name)
            return provider_cls.from_config(cfg)
        else:
            raise ValueError(
                f"Unknown provider '{provider_name}'. Supported: anthropic, openai"
            )


def _build_system_prompt(
    question: EvalQuestion, context: str | None, max_steps: int = 5
) -> str:
    schema = question.expected_answer_schema or ANSWER_SCHEMAS.get(
        question.question_type, {}
    )
    schema_str = json.dumps(schema, indent=2)

    parts = [
        "You are an expert evaluator. Answer the question below by researching the "
        "available artifacts. You reason carefully over artifacts to answer complex questions. "
        "You cite evidence by artifact ID, stay within stated constraints, and never guess. "
        "You have access to the full conversation history. Never call a tool to retrieve "
        "an artifact you have already retrieved in a previous step. Re-read the earlier result instead."
        "When you have enough information, call submit_answer "
        "with a JSON object matching the required schema. ",
        f"\nQuestion type: {question.question_type}",
        f"\nRequired answer schema:\n{schema_str}",
    ]

    if question.question_type == "PERSPECTIVE" and question.actor:
        parts.append(
            f"\nYou are reasoning from the perspective of: {question.actor} "
            f"(role: {question.actor_role or 'unknown'}) "
            f"as of: {(question.as_of_time or '')[:10]}. "
            f"Only consider information accessible to this actor at that time."
        )

    if context:
        parts.append(f"\n\nAvailable context:\n{context}")
    else:
        parts.append(
            "\n\nUse the fetch_artifact and search_artifacts tools to retrieve "
            "relevant information before submitting your answer."
        )

    parts.append(
        f"\n\nYou have at most {max_steps} turns to research. "
        "On your final turn, you MUST call submit_answer with whatever "
        "you have found. Do not call fetch_artifact or search_artifacts "
        "on your last turn."
    )

    return "\n".join(parts)


def _dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    runtime: GatedRuntime | None,
) -> Any:
    """Execute a tool call against the GatedRuntime and return the result."""
    if tool_name == "submit_answer":
        return {"status": "answer_recorded"}

    if runtime is None:
        return {"error": "No runtime available for tool calls"}

    if tool_name == "fetch_artifact":
        result = runtime.fetch(tool_input.get("artifact_id", ""))
        return result if result is not None else {"error": "not found or gated"}

    if tool_name == "search_artifacts":
        results = runtime.search(
            query=tool_input.get("query", ""),
            artifact_type=tool_input.get("artifact_type"),
            limit=int(tool_input.get("limit", 10)),
        )
        return results

    return {"error": f"unknown tool: {tool_name}"}


class AnthropicProvider(ModelProvider):
    """
    Anthropic provider using the Messages API and the SDK's Tool Runner.

    Tool use: driven by `client.beta.messages.tool_runner()`. fetch_artifact
    and search_artifacts are defined as `@beta_tool`-decorated closures over
    the question's GatedRuntime, so the runner can call them directly
    instead of us hand-dispatching tool_use blocks.

    Structured output: enforced via submit_answer with strict: true. The
    tool's input_schema is compiled per-question from
    question.expected_answer_schema (or ANSWER_SCHEMAS[question_type]), so
    the `answer` field the model returns is already guaranteed to match the
    expected schema — no post-hoc JSON extraction or validation needed on
    this path.

    Last-step forcing: on the final turn, the tool runner is rebuilt with
    only submit_answer available and a user message instructs the model
    this is its last turn.
    """

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        try:
            import anthropic as _anthropic

            self._client = _anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def complete(self, prompt: str, max_tokens: int = 2056) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    def run_agent(
        self,
        question: EvalQuestion,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        trajectory = AgentTrajectory(
            question_id=question.question_id,
            question_type=question.question_type,
        )

        system = _build_system_prompt(question, context, max_steps)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": question.question_text}
        ]

        subsystem_list = runtime.all_subsystems if runtime is not None else []

        mapping = {}
        for s in subsystem_list:
            key = s.upper().replace("-", "_").replace(".", "_")
            while key in mapping:
                key = key + "_"
            mapping[key] = s
        ArtifactType = Enum("ArtifactType", mapping or {"__NONE__": "__none__"})

        beta_tool = self._anthropic.beta_tool

        @beta_tool
        def fetch_artifact(artifact_id: str) -> str:
            """Retrieve a single artifact by its ID.

            Args:
                artifact_id (str): The artifact ID to fetch.
            Returns:
                str: JSON-encoded artifact dict, or an error dict if not
                    found or gated.
            """
            if runtime is None:
                return json.dumps({"error": "No runtime available for tool calls"})
            result = runtime.fetch(artifact_id)
            return json.dumps(
                result if result is not None else {"error": "not found or gated"},
                default=str,
            )

        @beta_tool
        def search_artifacts(
            query: str, artifact_type: ArtifactType, limit: int = 10
        ) -> str:
            """Full-text search over the artifact corpus.

            Args:
                query (str): Search terms.
                artifact_type (str): Optional subsystem filter (e.g. 'jira', 'email').
                limit (int): Max results (default 10).
            Returns:
                str: JSON-encoded list of matching artifact dicts.
            """
            if runtime is None:
                return json.dumps({"error": "No runtime available for tool calls"})

            at: str = (
                artifact_type.value
                if hasattr(artifact_type, "value")
                else str(artifact_type)
            )
            results = runtime.search(query=query, artifact_type=at, limit=int(limit))
            return json.dumps(results, default=str)

        submit_answer_def = _build_submit_answer_tool(question)
        tools: list[Any] = [submit_answer_def]
        if runtime is not None:
            tools = [fetch_artifact, search_artifacts, submit_answer_def]

        final_answer: dict[str, Any] = {}
        answer_submitted = False
        t_start = time.time()
        prompt_tokens = 0
        completion_tokens = 0

        runner = self._client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            tools=tools,
            messages=cast(Any, messages),
        )

        step = 0
        try:
            for response in runner:
                step += 1

                if step == max_steps and not answer_submitted:
                    tools = [submit_answer_def]
                    messages.append({
                        "role": "user",
                        "content": (
                            "This is your final turn. You must call "
                            "submit_answer now with your best answer "
                            "based on what you have found. Do not call "
                            "any other tool."
                        ),
                    })
                    runner = self._client.beta.messages.tool_runner(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        system=system,
                        tools=tools,
                        messages=cast(Any, messages),
                    )
                    continue

                if response.usage:
                    prompt_tokens += response.usage.input_tokens
                    completion_tokens += response.usage.output_tokens

                for block in response.content:
                    block_type = getattr(block, "type", None)
                    block_name = getattr(block, "name", None)
                    if block_type == "tool_use" and block_name == "submit_answer":
                        block_input = getattr(block, "input", {})
                        final_answer = block_input.get("answer", block_input)
                        answer_submitted = True

                if answer_submitted or step >= max_steps:
                    break
        except Exception as exc:
            logger.error(f"Anthropic tool runner error: {exc}")

        trajectory.total_latency_ms = (time.time() - t_start) * 1000
        trajectory.prompt_tokens = prompt_tokens
        trajectory.completion_tokens = completion_tokens
        trajectory.budget_exceeded = step >= max_steps and not answer_submitted

        return trajectory, final_answer


class OpenAIProvider(ModelProvider):
    """
    OpenAI provider using the Chat Completions API with function calling.

    Tool use: OpenAI function calling (tool_calls / tool role messages).
    Structured output: enforced via submit_answer function + json_object mode
    as a fallback if the model doesn't call submit_answer.

    Last-step forcing: on the final turn, tools are filtered to only
    submit_answer. Model-specific tool_choice:
    - Native OpenAI / Azure OpenAI: "auto" with a single tool works.
    - Mistral/Qwen/Llama behind Azure: "required" forces a tool call when
      only one tool is available. Detected from base_url and model name.

    API key:  OPENAI_API_KEY env var, or api_key in config.
    """

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=kwargs.get("base_url"),
                max_retries=int(kwargs.get("max_retries", 3)),
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._base_url = kwargs.get("base_url", "")
        self._model_lower = self.model.lower()

    def _is_non_openai_model(self) -> bool:
        """Returns True if the endpoint is likely routing to a non-OpenAI model
        (Mistral, Qwen, Llama, Ollama) that needs tool_choice='required' to
        force a tool call when only one tool is available."""
        url_lower = self._base_url.lower() if self._base_url else ""
        model_lower = self._model_lower
        non_openai_markers = {
            "mistral",
            "moonshot",
            "moonshotai",
            "qwen",
            "llama",
            "gemma",
            "ollama",
            "11434",
            "deepseek",
        }
        return any(m in url_lower for m in non_openai_markers) or any(
            m in model_lower for m in non_openai_markers
        )

    def complete(self, prompt: str, max_tokens: int = 2056) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        if not content:
            logger.warning(
                "[providers] Empty completion content. "
                f"finish_reason={choice.finish_reason!r} "
                f"message={choice.message!r} "
                f"usage={response.usage!r}"
            )
        return content

    def run_agent(
        self,
        question: EvalQuestion,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        trajectory = AgentTrajectory(
            question_id=question.question_id,
            question_type=question.question_type,
        )

        system = _build_system_prompt(question, context, max_steps)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question.question_text},
        ]

        answer_schema = question.expected_answer_schema or ANSWER_SCHEMAS.get(
            question.question_type, {}
        )
        submit_answer_tool = {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": (
                    "Submit your final answer. Call this exactly once when you are "
                    "done researching. The answer must conform to the expected schema."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["answer"],
                    "properties": {
                        "answer": answer_schema
                        if answer_schema
                        else {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
        }

        if runtime is not None:
            subsystem_list = (
                runtime.all_subsystems if hasattr(runtime, "all_subsystems") else []
            )
            valid_subsystems = sorted(set(s for s in subsystem_list if s))
            dynamic_base = [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_artifact",
                        "description": (
                            "Retrieve a single artifact by its ID. "
                            "Returns the artifact dict or null if not found or gated."
                        ),
                        "parameters": {
                            "type": "object",
                            "required": ["artifact_id"],
                            "properties": {
                                "artifact_id": {
                                    "type": "string",
                                    "description": "The artifact ID to fetch.",
                                }
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_artifacts",
                        "description": (
                            "Full-text search over the artifact corpus. "
                            "Returns a list of matching artifact dicts. "
                            "You MUST specify the subsystem to search within."
                        ),
                        "parameters": {
                            "type": "object",
                            "required": ["query", "artifact_type"],
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search terms or artifact ID.",
                                },
                                "artifact_type": {
                                    "type": "string",
                                    "enum": valid_subsystems,
                                    "description": (
                                        f"REQUIRED. The subsystem to search within. "
                                        f"Must be one of: {', '.join(valid_subsystems)}."
                                        if valid_subsystems
                                        else "No subsystems available."
                                    ),
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Max results (default 10, max 50).",
                                    "default": 10,
                                },
                            },
                        },
                    },
                },
            ]
            tools = dynamic_base + [submit_answer_tool]
        else:
            tools = [submit_answer_tool]

        final_answer: dict[str, Any] = {}
        t_start = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        budget_exceeded = False

        for step in range(max_steps):
            current_tools = tools
            current_tool_choice: Any = "auto"

            if step == max_steps - 1:
                current_tools = [
                    t for t in tools if t["function"]["name"] == "submit_answer"
                ]
                messages.append({
                    "role": "user",
                    "content": (
                        "This is your final turn. You must call "
                        "submit_answer now with your best answer "
                        "based on what you have found. Do not call "
                        "any other tool."
                    ),
                })

                current_tool_choice = {
                    "type": "function",
                    "function": {"name": "submit_answer"},
                }

            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    tools=current_tools,  # type: ignore[arg-type]
                    tool_choice=current_tool_choice,  # type: ignore[arg-type]
                    messages=messages,  # type: ignore[arg-type]
                    parallel_tool_calls=False,
                )
            except Exception as exc:
                logger.error(f"OpenAI API error on step {step}: {exc}")
                break

            if response.usage:
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

            choice = response.choices[0]
            msg = choice.message
            messages.append(msg.model_dump())

            answer_submitted = False

            if msg.tool_calls:
                tool_results = []
                for tc in msg.tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    fn_name = fn.name
                    try:
                        fn_input = json.loads(fn.arguments)
                    except json.JSONDecodeError:
                        fn_input = {}

                    if fn_name == "submit_answer":
                        final_answer = fn_input.get("answer", fn_input)
                        answer_submitted = True
                        result = {"status": "answer_recorded"}
                    else:
                        result = _dispatch_tool(fn_name, fn_input, runtime)

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })

                messages.extend(tool_results)

                if answer_submitted:
                    break

            elif choice.finish_reason in ("stop", "length"):
                if msg.content:
                    final_answer = _extract_json_from_text(msg.content)
                break

            time.sleep(3)
        else:
            budget_exceeded = True

        trajectory.total_latency_ms = (time.time() - t_start) * 1000
        trajectory.prompt_tokens = prompt_tokens
        trajectory.completion_tokens = completion_tokens
        trajectory.budget_exceeded = budget_exceeded

        return trajectory, final_answer


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """
    Last-resort extraction of a JSON object from free-form model output.
    Looks for the last {...} block in the text.
    """
    import re

    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning("Could not extract JSON from model output")
        return {}


def build_prose_fn(provider: ModelProvider):
    """
    Returns a llm_fn(prompt: str) -> str callable suitable for
    passing to QuestionGenerator.
    """

    def llm_fn(prompt: str) -> str:
        return provider.complete(prompt, max_tokens=2056)

    return llm_fn


def build_agent_fn(provider: ModelProvider):
    """
    Returns an agent_fn compatible with run.py's _run_one signature:
        agent_fn(question, context, tools, max_steps, runtime=None)
        -> (AgentTrajectory, dict)

    The tools argument from run.py is ignored here — the provider owns
    its tool definitions internally. runtime is passed through to
    provider.run_agent().
    """

    def agent_fn(question, context, tools, max_steps, runtime=None):
        return provider.run_agent(
            question=question,
            context=context,
            runtime=runtime,
            max_steps=max_steps,
        )

    return agent_fn
