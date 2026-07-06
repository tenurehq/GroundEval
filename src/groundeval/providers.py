"""
Model provider layer. Two providers built-in: Anthropic and OpenAI.
Extend by subclassing ModelProvider.

SINGLE USE: Agent loop
----------------------
- Multi-turn with tool calls (fetch_artifact, search_artifacts)
- Structured JSON final answer required (enforced via expected_answer_schema)
- Called via provider.run_agent(question_text, ..., runtime, max_steps)
- Returns (AgentTrajectory, dict)
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
from json_repair import loads as repair_loads

from .core import (
    AgentTrajectory,
    GatedRuntime,
    ANSWER_SCHEMA_TASK,
)

logger = logging.getLogger("groundeval.providers")


def _to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Compile a JSON Schema into a strict-tool-use compliant schema.

    Strict mode requires `additionalProperties: false` on every object node.
    We deep-copy and inject that constraint here.
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


def _build_submit_answer_tool(
    expected_answer_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the submit_answer tool definition with strict: true.
    Uses the provided schema or falls back to ANSWER_SCHEMA_TASK.
    """
    schema = expected_answer_schema or ANSWER_SCHEMA_TASK
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
    The method to implement is run_agent().
    """

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.temperature = float(kwargs.get("temperature", 0.0))
        self.max_tokens = int(kwargs.get("max_tokens", 1024))

    @abstractmethod
    def run_agent(
        self,
        task_id: str,
        question_text: str,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
        expected_answer_schema: dict[str, Any] | None = None,
        actor: str | None = None,
        actor_role: str | None = None,
        as_of_time: str | None = None,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        """
        Run the agent loop against a single task.

        - task_id: unique identifier for this run (used in trajectory)
        - question_text: the task description + preconditions text
        - context: pre-built string of artifact text (context-injection mode)
        - runtime: GatedRuntime providing gated fetch/search (tool mode)
        - max_steps: maximum tool call turns
        - expected_answer_schema: JSON schema for the answer (defaults to ANSWER_SCHEMA_TASK)
        - actor, actor_role, as_of_time: optional perspective metadata

        Exactly one of context or runtime will be non-None.

        Returns (trajectory, final_answer_dict).
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
    question_text: str,
    context: str | None,
    max_steps: int = 5,
    expected_answer_schema: dict[str, Any] | None = None,
    actor: str | None = None,
    actor_role: str | None = None,
    as_of_time: str | None = None,
) -> str:
    schema = expected_answer_schema or ANSWER_SCHEMA_TASK
    schema_str = json.dumps(schema, indent=2)

    parts = [
        "You are an expert evaluator. Answer the question below by researching the "
        "available artifacts. You reason carefully over artifacts to answer complex questions. "
        "You cite evidence by artifact ID, stay within stated constraints, and never guess. "
        "You have access to the full conversation history. Never call a tool to retrieve "
        "an artifact you have already retrieved in a previous step. Re-read the earlier result instead. "
        "When you have enough information, call submit_answer "
        "with a JSON object matching the required schema. ",
        "\nQuestion type: TASK",
        f"\nRequired answer schema:\n{schema_str}",
    ]

    if actor:
        parts.append(
            f"\nYou are reasoning from the perspective of: {actor} "
            f"(role: {actor_role or 'unknown'}) "
            f"as of: {(as_of_time or '')[:10]}. "
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
        return result if result is not None else {}

    if tool_name == "search_artifacts":
        results = runtime.search(
            query=tool_input.get("query", ""),
            artifact_type=tool_input.get("artifact_type"),
            limit=int(tool_input.get("limit", 10)),
        )
        return results

    if tool_name == "generate_email":
        customer_id = tool_input.get("customer_id", "")
        return {
            "customer_id": customer_id,
            "customer_name": "Jenny Fields",
            "email": "jenny@gmail.com",
            "subject": "Exclusive offer for Jenny Fields",
            "body": "Dear Jenny Fields, we have a special offer for you...",
        }

    return {"error": f"unknown tool: {tool_name}"}


class AnthropicProvider(ModelProvider):
    """
    Anthropic provider using the Messages API and the SDK's Tool Runner.

    Tool use: driven by `client.beta.messages.tool_runner()`. fetch_artifact
    and search_artifacts are defined as `@beta_tool`-decorated closures over
    the GatedRuntime.

    Structured output: enforced via submit_answer with strict: true.

    Last-step forcing: on the final turn, the tool runner is rebuilt with
    only submit_answer available.
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

    def run_agent(
        self,
        task_id: str,
        question_text: str,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
        expected_answer_schema: dict[str, Any] | None = None,
        actor: str | None = None,
        actor_role: str | None = None,
        as_of_time: str | None = None,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        trajectory = AgentTrajectory(task_id=task_id)

        system = _build_system_prompt(
            question_text=question_text,
            context=context,
            max_steps=max_steps,
            expected_answer_schema=expected_answer_schema,
            actor=actor,
            actor_role=actor_role,
            as_of_time=as_of_time,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": question_text}]

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
                result if result is not None else {},
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

        @beta_tool
        def generate_email(customer_id: str) -> str:
            draft = {
                "customer_id": customer_id,
                "customer_name": "Jenny Fields",
                "email": "jenny@gmail.com",
                "subject": "Exclusive offer for Jenny Fields",
                "body": "Dear Jenny Fields, we have a special offer for you...",
            }
            return json.dumps(draft)

        submit_answer_def = _build_submit_answer_tool(expected_answer_schema)
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
    submit_answer.
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
        """Returns True if the endpoint is likely routing to a non-OpenAI model."""
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

    def run_agent(
        self,
        task_id: str,
        question_text: str,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
        expected_answer_schema: dict[str, Any] | None = None,
        actor: str | None = None,
        actor_role: str | None = None,
        as_of_time: str | None = None,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        trajectory = AgentTrajectory(task_id=task_id)

        system = _build_system_prompt(
            question_text=question_text,
            context=context,
            max_steps=max_steps,
            expected_answer_schema=expected_answer_schema,
            actor=actor,
            actor_role=actor_role,
            as_of_time=as_of_time,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question_text},
        ]

        answer_schema = expected_answer_schema or ANSWER_SCHEMA_TASK
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
                {
                    "type": "function",
                    "function": {
                        "name": "generate_email",
                        "description": (
                            "Generate an email draft for a given customer ID. "
                            "Returns the draft with customer_name, email_address, subject, and body."
                        ),
                        "parameters": {
                            "type": "object",
                            "required": ["customer_id"],
                            "properties": {
                                "customer_id": {
                                    "type": "string",
                                    "description": "The customer ID to generate an email draft for.",
                                }
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
                status_code = getattr(exc, "status_code", None)
                if status_code in (401, 403):
                    logger.error(f"OpenAI auth error: {exc}")
                    raise SystemExit(
                        f"Authentication failed (HTTP {status_code}). "
                        "Check your api_key in config.yaml or the "
                        "OPENAI_API_KEY environment variable."
                    )
                raise

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
                        fn_input = repair_loads(fn.arguments)
                    except Exception:
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
    if not text or not text.strip():
        return {}

    try:
        result = repair_loads(text)
    except Exception:
        logger.warning("Could not extract JSON from model output")
        return {}
    if isinstance(result, dict):
        return result
    logger.warning("Could not extract JSON from model output")
    return {}


def build_agent_fn(provider: ModelProvider):
    """
    Returns an agent_fn compatible with task_eval.py's signature:
        agent_fn(question, context, tools, max_steps, runtime=None)
        -> (AgentTrajectory, dict)

    The tools argument is ignored — the provider owns its tool definitions internally.
    """

    def agent_fn(question, context, tools, max_steps, runtime=None):
        return provider.run_agent(
            task_id=question.question_id,
            question_text=question.question_text,
            context=context,
            runtime=runtime,
            max_steps=max_steps,
            expected_answer_schema=question.expected_answer_schema,
            actor=getattr(question, "actor", None),
            actor_role=getattr(question, "actor_role", None),
            as_of_time=getattr(question, "as_of_time", None),
        )

    return agent_fn
