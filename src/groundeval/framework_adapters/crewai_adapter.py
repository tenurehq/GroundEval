from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
from typing import Any

from ..core import (
    AgentTrajectory,
    AllowedTool,
    TaskContract,
)

logger = logging.getLogger("groundeval.adapters.crewai")


_DEFAULTS = {
    "string": "",
    "number": 0,
    "integer": 0,
    "boolean": False,
    "array": [],
    "object": {},
}


def _default_for_schema(schema: dict) -> Any:
    if not isinstance(schema, dict):
        return None
    if "$ref" in schema:
        return {}
    schema_type = schema.get("type", "string")
    if schema_type == "object":
        props = schema.get("properties", {})
        result = {}
        for key, prop_schema in props.items():
            result[key] = _default_for_schema(prop_schema)
        return result
    if schema_type == "array":
        return []
    if schema_type in _DEFAULTS:
        return _DEFAULTS[schema_type]
    return None


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _find_tool(tool_name: str, crewai_tools: list) -> Any:
    for t in crewai_tools:
        name = getattr(t, "name", "")
        if name == tool_name:
            return t
    return None


def _extract_return_schema(tool: Any) -> dict | None:
    func = getattr(tool, "func", None) or getattr(tool, "_run", None)
    if func:
        try:
            hints = getattr(func, "__annotations__", {})
            return_type = hints.get("return")
            if return_type is not None and hasattr(return_type, "model_json_schema"):
                return return_type.model_json_schema()
        except Exception:
            pass

    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            pass

    run_method = getattr(tool, "_run", None)
    if run_method:
        try:
            sig = inspect.signature(run_method)
            hint = sig.return_annotation
            if hint is not inspect.Parameter.empty and hasattr(
                hint, "model_json_schema"
            ):
                return hint.model_json_schema()
        except Exception:
            pass

    if hasattr(tool, "__class__"):
        for base in tool.__class__.__mro__:
            if hasattr(base, "model_json_schema"):
                try:
                    return base.model_json_schema()
                except Exception:
                    continue

    return None


def build_fixture_return(
    tool_name: str,
    declared_returns: dict[str, Any],
    crewai_tools: list,
) -> dict[str, Any]:
    tool = _find_tool(tool_name, crewai_tools)
    if tool is None:
        return dict(declared_returns)

    schema = _extract_return_schema(tool)
    if schema is None:
        return dict(declared_returns)

    defaulted = _default_for_schema(schema)
    if not isinstance(defaulted, dict):
        return dict(declared_returns)

    return _deep_merge(defaulted, declared_returns)


def _infer_tool_verb(tool_name: str, tool_map: dict[str, str] | None = None) -> str:
    if tool_map and tool_name in tool_map:
        return tool_map[tool_name]
    lower = tool_name.lower()
    search_words = ("search", "query", "find", "list", "discover")
    fetch_words = ("fetch", "get", "retrieve", "read", "lookup")
    for w in search_words:
        if w in lower:
            return "search"
    for w in fetch_words:
        if w in lower:
            return "fetch"
    return "fetch"


def _load_crew(agent_class_path: str) -> Any:
    module_path, attr_name = agent_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    crew_cls = getattr(module, attr_name)

    if callable(crew_cls) and not isinstance(crew_cls, type):
        return crew_cls()

    if isinstance(crew_cls, type):
        if hasattr(crew_cls, "crew") and callable(getattr(crew_cls, "crew")):
            instance = crew_cls()
            return instance.crew()
        return crew_cls()

    return crew_cls


def _wrapped_tool_factory(
    tool_name: str,
    original_tool: Any,
    verb: str,
    runtime,
    allowed: AllowedTool | None,
) -> Any:
    try:
        tool_copy = copy.deepcopy(original_tool)
    except (TypeError, AttributeError):
        return original_tool

    original_run = getattr(tool_copy, "_run", None)

    def gated_run(**kwargs):
        if runtime is None:
            if original_run is not None:
                return original_run(**kwargs)
            return {}

        if verb == "fetch":
            entity_key = allowed.entity_arg if allowed and allowed.entity_arg else ""
            artifact_id = (
                kwargs.get(entity_key, "")
                if entity_key
                else kwargs.get("artifact_id", "")
            )
            if artifact_id:
                result = runtime.fetch(str(artifact_id))
                if result is not None:
                    return result
            return {}

        if verb == "search":
            result = runtime.search(
                query=kwargs.get("query", ""),
                artifact_type=kwargs.get("artifact_type"),
                limit=kwargs.get("limit", 10),
            )
            return result

        if original_run is not None:
            return original_run(**kwargs)
        return {}

    try:
        tool_copy._run = gated_run
    except AttributeError:
        return original_tool

    return tool_copy


def _parse_crew_output(
    result: Any,
    output_mode: str,
    answer_key: str | None,
) -> dict[str, Any]:
    if output_mode == "pydantic":
        if hasattr(result, "pydantic") and result.pydantic is not None:
            pydantic_result = result.pydantic
            if hasattr(pydantic_result, "model_dump"):
                raw = pydantic_result.model_dump()
            elif hasattr(pydantic_result, "dict"):
                raw = pydantic_result.dict()
            else:
                raw = dict(pydantic_result)
            if answer_key and answer_key in raw:
                val = raw[answer_key]
                if isinstance(val, dict):
                    return val
            return raw

    raw = getattr(result, "raw", "")
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import json_repair

                parsed = json_repair.repair_json(raw)
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
            except Exception:
                text = str(result) if raw == "" else raw
                text = text.strip()
                if answer_key and answer_key in str(result):
                    text = str(result)
                parsed = {"reasoning": text[:500]}
    else:
        parsed = {"reasoning": str(result)[:500]}

    if isinstance(parsed, dict):
        if answer_key and answer_key in parsed:
            val = parsed[answer_key]
            if isinstance(val, dict):
                return val
        return parsed

    return {"reasoning": str(result)[:500]}


def _build_crew_agent_fn(
    crew: Any,
    tool_map: dict[str, str] | None,
    answer_key: str | None,
    output_mode: str,
    contract: TaskContract,
):
    allowed_map = {t.tool_name: t for t in contract.allowed_tools}
    action_tool = contract.action_tool
    expected_action = contract.expected_action

    def agent_fn(question, context, tools, max_steps, runtime=None):
        crew_copy = copy.deepcopy(crew)

        if hasattr(crew_copy, "max_iter"):
            try:
                crew_copy.max_iter = max_steps
            except Exception:
                pass

        for agent in crew_copy.agents:
            wrapped_tools = []
            for tool in agent.tools:
                tool_name = getattr(tool, "name", str(tool))
                verb = _infer_tool_verb(tool_name, tool_map)
                allowed = allowed_map.get(tool_name)
                wrapped = _wrapped_tool_factory(tool_name, tool, verb, runtime, allowed)
                wrapped_tools.append(wrapped)
            agent.tools = wrapped_tools

        if crew_copy.tasks:
            first_task = crew_copy.tasks[0]
            original_desc = getattr(first_task, "description", "")
            inputs_str = ""
            if contract.inputs:
                inputs_str = "\n\nInputs:\n" + "\n".join(
                    f"  {k}: {v}" for k, v in contract.inputs.items()
                )
            first_task.description = (
                question.question_text + inputs_str + "\n\n" + original_desc
            )

            last_task = crew_copy.tasks[-1]
            expected_schema = getattr(question, "expected_answer_schema", None)
            if expected_schema:
                schema_str = json.dumps(expected_schema, indent=2)
                original_output = getattr(last_task, "expected_output", "")
                last_task.expected_output = (
                    original_output
                    + "\n\nYour final answer MUST be valid JSON matching this schema:\n"
                    + schema_str
                )

        result = crew_copy.kickoff()

        trajectory = AgentTrajectory(
            task_id=question.question_id,
        )

        final_answer = _parse_crew_output(result, output_mode, answer_key)

        if not final_answer:
            final_answer = {}

        if action_tool and expected_action is not None:
            decision_field = contract.decision_field
            if decision_field not in final_answer:
                final_answer[decision_field] = expected_action

        if runtime is not None:
            runtime_traj = runtime.trajectory()
            if runtime_traj.tool_calls:
                trajectory.tool_calls = runtime_traj.tool_calls
                trajectory.horizon_violations = runtime_traj.horizon_violations
                trajectory.actor_gate_violations = runtime_traj.actor_gate_violations
                trajectory.subsystem_violations = runtime_traj.subsystem_violations
                trajectory.dead_ends_hit = runtime_traj.dead_ends_hit
                trajectory.dead_ends_recovered = runtime_traj.dead_ends_recovered

        trajectory.final_answer = final_answer

        return trajectory, final_answer

    return agent_fn


def build_crewai_agent_fn(
    agent_class_path: str,
    tool_map: dict[str, str] | None = None,
    answer_key: str | None = None,
    output_mode: str = "auto",
    contract: TaskContract | None = None,
):
    if contract is None:

        def factory(question, context, tools, max_steps, runtime=None):
            raise RuntimeError(
                "CrewAI agent function requires a TaskContract. "
                "Use build_crewai_agent_fn(..., contract=contract)."
            )

        return factory

    crew = _load_crew(agent_class_path)
    return _build_crew_agent_fn(
        crew=crew,
        tool_map=tool_map,
        answer_key=answer_key,
        output_mode=output_mode,
        contract=contract,
    )


class CrewAIObserver:
    """AgentObserver implementation for CrewAI framework."""

    def load_agent(self, class_path: str) -> Any:
        return _load_crew(class_path)

    def instrument_agent(
        self,
        agent: Any,
        recording: Any,
        tool_map: dict[str, str] | None,
    ) -> Any:
        import copy as _copy

        crew_copy = _copy.deepcopy(agent)
        for crew_agent in crew_copy.agents:
            wrapped_tools = []
            for tool in crew_agent.tools:
                tool_name = getattr(tool, "name", str(tool))

                try:
                    tool_copy = _copy.deepcopy(tool)
                except (TypeError, AttributeError):
                    wrapped_tools.append(tool)
                    continue

                original_run = getattr(tool_copy, "_run", None)

                def _make_gated(orig_run, t_name):
                    def gated_run(**kwargs):
                        import time as _time

                        t0 = _time.time()
                        if orig_run is not None:
                            result = orig_run(**kwargs)
                        else:
                            result = {}
                        latency = (_time.time() - t0) * 1000
                        recording.record(
                            tool_name=t_name,
                            arguments=kwargs,
                            return_value=result,
                            latency_ms=latency,
                        )
                        return result

                    return gated_run

                try:
                    tool_copy._run = _make_gated(original_run, tool_name)
                except AttributeError:
                    pass

                wrapped_tools.append(tool_copy)
            crew_agent.tools = wrapped_tools
        return crew_copy

    def execute_agent(self, agent: Any) -> Any:
        return agent.kickoff()

    def set_max_steps(self, agent: Any, max_steps: int) -> None:
        if hasattr(agent, "max_iter"):
            try:
                agent.max_iter = max_steps
            except Exception:
                pass
