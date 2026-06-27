from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from .core import (
    AgentTrajectory,
    ToolCall,
    TaskPrecondition,
    AllowedTool,
    TaskContract,
)

logger = logging.getLogger("groundeval.observe")

try:
    import json_repair as _json_repair
except ImportError:
    _json_repair = None


@dataclass
class ObservedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    return_value: Any
    latency_ms: float


@dataclass
class ObservedRun:
    run_id: str
    framework: str
    agent_class: str
    tool_calls: list[ObservedToolCall] = field(default_factory=list)
    final_answer: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservedRun":
        return cls(
            run_id=d["run_id"],
            framework=d["framework"],
            agent_class=d["agent_class"],
            tool_calls=[ObservedToolCall(**tc) for tc in d.get("tool_calls", [])],
            final_answer=d.get("final_answer", {}),
            total_latency_ms=d.get("total_latency_ms", 0.0),
        )


class RecordingRuntime:
    def __init__(self):
        self._call_log: list[ObservedToolCall] = []
        self._trajectory_log: list[ToolCall] = []

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        return_value: Any,
        latency_ms: float,
    ) -> None:
        self._call_log.append(
            ObservedToolCall(
                tool_name=tool_name,
                arguments=arguments,
                return_value=return_value,
                latency_ms=latency_ms,
            )
        )

    @property
    def call_log(self) -> list[ObservedToolCall]:
        return list(self._call_log)


@runtime_checkable
class AgentObserver(Protocol):
    """Protocol for framework-specific agent observability adapters.

    Each adapter implementation handles loading, instrumenting, and
    executing an agent for a particular agent framework (CrewAI,
    LangChain, AutoGen, etc.).  The observe_agent() dispatcher calls
    these methods so the rest of the observation pipeline (recording,
    draft generation) stays framework-agnostic.

    Implement this protocol once per framework and register it in
    _OBSERVER_REGISTRY.
    """

    def load_agent(self, class_path: str) -> Any:
        """Import and return an agent instance given a dotted path."""
        ...

    def instrument_agent(
        self,
        agent: Any,
        recording: RecordingRuntime,
        tool_map: dict[str, str] | None,
    ) -> Any:
        """Deep-copy *agent*, wrap tools with recording hooks, return copy."""
        ...

    def execute_agent(self, agent: Any) -> Any:
        """Run the instrumented agent and return its raw result."""
        ...

    def set_max_steps(self, agent: Any, max_steps: int) -> None:
        """Configure a step / iteration limit on the agent if supported."""
        ...


_OBSERVER_REGISTRY: dict[str, AgentObserver] = {}


def register_observer(framework: str, observer: AgentObserver) -> None:
    _OBSERVER_REGISTRY[framework] = observer


def _get_observer(framework: str) -> AgentObserver:
    if framework not in _OBSERVER_REGISTRY:
        _try_auto_register(framework)
    if framework not in _OBSERVER_REGISTRY:
        raise ValueError(
            f"No observer registered for framework '{framework}'. "
            f"Registered frameworks: {list(_OBSERVER_REGISTRY.keys())}. "
            f"Ensure the adapter package is installed and importable."
        )
    return _OBSERVER_REGISTRY[framework]


def _try_auto_register(framework: str) -> None:
    if framework == "crewai":
        try:
            from .framework_adapters.crewai_adapter import CrewAIObserver

            register_observer("crewai", CrewAIObserver())
        except ImportError:
            pass


def _tool_name_to_verb(tool_name: str) -> str:
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


def _extract_return_schema_from_tool(tool: Any) -> dict | None:
    import inspect

    func = None
    if "_run" in tool.__dict__:
        func = tool._run
    elif "func" in tool.__dict__:
        func = tool.func

    if func is not None:
        try:
            hints = getattr(func, "__annotations__", {})
            return_type = hints.get("return")
            if return_type is not None and hasattr(return_type, "model_json_schema"):
                return return_type.model_json_schema()
        except Exception:
            pass

    args_schema = None
    if "args_schema" in tool.__dict__:
        args_schema = tool.args_schema

    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            pass

    if func is not None:
        try:
            sig = inspect.signature(func)
            hint = sig.return_annotation
            if hint is not inspect.Parameter.empty and hasattr(
                hint, "model_json_schema"
            ):
                return hint.model_json_schema()
        except Exception:
            pass

    return None


def _parse_observed_answer(result: Any) -> dict[str, Any]:
    raw = getattr(result, "raw", "")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            if _json_repair is not None:
                try:
                    repaired = _json_repair.repair_json(raw)
                    if isinstance(repaired, str):
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            return parsed
                    elif isinstance(repaired, dict):
                        return repaired
                except Exception:
                    pass
    if hasattr(result, "pydantic") and result.pydantic is not None:
        pydantic_result = result.pydantic
        if hasattr(pydantic_result, "model_dump") and callable(
            pydantic_result.model_dump
        ):
            return pydantic_result.model_dump()
        if hasattr(pydantic_result, "dict") and callable(pydantic_result.dict):
            return pydantic_result.dict()
        try:
            return dict(pydantic_result)
        except Exception:
            pass
    return {"raw_output": str(result)[:1000]}


def observe_agent(
    framework: str,
    class_path: str,
    tool_map: dict[str, str] | None = None,
    max_steps: int = 10,
) -> ObservedRun:
    observer = _get_observer(framework)

    agent = observer.load_agent(class_path)
    recording = RecordingRuntime()

    run_id = f"observed_{class_path.replace('.', '_')}_{int(time.time())}"

    instrumented = observer.instrument_agent(agent, recording, tool_map)
    observer.set_max_steps(instrumented, max_steps)

    t_start = time.time()
    result = observer.execute_agent(instrumented)
    total_latency = (time.time() - t_start) * 1000

    final_answer = _parse_observed_answer(result)

    return ObservedRun(
        run_id=run_id,
        framework=framework,
        agent_class=class_path,
        tool_calls=recording.call_log,
        final_answer=final_answer,
        total_latency_ms=total_latency,
    )


def observe_crew(
    crew_class_path: str,
    tool_map: dict[str, str] | None = None,
    max_steps: int = 10,
) -> ObservedRun:
    """Deprecated: use observe_agent(framework='crewai', ...) instead."""
    return observe_agent(
        framework="crewai",
        class_path=crew_class_path,
        tool_map=tool_map,
        max_steps=max_steps,
    )


class DraftGenerator:
    def __init__(self, observed: ObservedRun, mode: str = "standard"):
        self._observed = observed
        self._mode = mode

    def generate(self) -> dict[str, Any]:
        tool_map_entries = self._infer_tool_map()
        preconditions = self._infer_preconditions()
        allowed_tools = self._infer_allowed_tools()
        roles = self._infer_roles()

        decision_field = "should_act"
        if self._observed.final_answer:
            for candidate in (
                "should_act",
                "all_preconditions_pass",
                "should_escalate",
            ):
                if candidate in self._observed.final_answer:
                    decision_field = candidate
                    break

        task_contract = {
            "name": "inferred_task",
            "task_description": "Inferred from observed agent run. Update with your domain description.",
            "preconditions": preconditions,
            "valid_action": "all_preconditions_pass",
            "decision_field": decision_field,
        }

        if allowed_tools:
            task_contract["allowed_tools"] = allowed_tools

        config = {
            "output_dir": "./eval_output",
            "agent": {
                "framework": self._observed.framework,
                "agent_class": self._observed.agent_class,
                "tool_map": tool_map_entries,
            },
            "roles": roles,
            "task_contracts": [task_contract],
            "groundeval": {
                "config_status": "draft",
                "generated_from_observation": True,
                "reviewed": False,
                "draft_mode": self._mode,
                "observed_run_id": self._observed.run_id,
            },
        }

        return config

    def _infer_tool_map(self) -> dict[str, str]:
        tool_map: dict[str, str] = {}
        seen = set()
        for tc in self._observed.tool_calls:
            if tc.tool_name not in seen:
                seen.add(tc.tool_name)
                tool_map[tc.tool_name] = _tool_name_to_verb(tc.tool_name)
        return tool_map

    def _infer_preconditions(self) -> list[dict[str, Any]]:
        answer = self._observed.final_answer or {}
        preconditions_verified = answer.get("preconditions_verified", [])

        if preconditions_verified:
            return self._infer_from_structured_answer(preconditions_verified)

        if self._mode == "conservative":
            return []

        return self._infer_from_tool_patterns()

    def _infer_from_structured_answer(
        self, preconditions_verified: list[dict]
    ) -> list[dict[str, Any]]:
        preconditions = []
        for pc in preconditions_verified:
            check = pc.get("check", "unknown_check")
            facts = pc.get("facts_found", {})
            evidence = pc.get("evidence_artifacts", [])

            fact_keys = list(facts.keys()) if facts else []
            gt_field = ""
            if fact_keys and evidence:
                gt_field = f"{evidence[0]}.{fact_keys[0]}"

            inferred_reason = f"Observed check '{check}' in agent answer."
            if evidence:
                inferred_reason += f" Evidence artifacts: {evidence}."

            preconditions.append({
                "check": check,
                "description": f"Agent must verify: {check}",
                "required_facts": fact_keys,
                "ground_truth_field": gt_field,
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "source": "structured_answer",
                    "reason": inferred_reason,
                },
            })

        return preconditions

    def _infer_from_tool_patterns(self) -> list[dict[str, Any]]:
        if self._mode == "conservative":
            return []

        fetch_calls = [
            tc
            for tc in self._observed.tool_calls
            if _tool_name_to_verb(tc.tool_name) == "fetch"
        ]

        preconditions = []
        for tc in fetch_calls:
            check_name = (
                tc.tool_name
                .replace("fetch_", "")
                .replace("get_", "")
                .replace("retrieve_", "")
            )
            check = f"{check_name}_verified"

            return_val = tc.return_value
            fact_keys = []
            if isinstance(return_val, dict):
                fact_keys = [
                    k
                    for k in return_val.keys()
                    if k not in ("id", "_id", "subsystem", "timestamp", "type")
                ]

            preconditions.append({
                "check": check,
                "description": f"Agent must verify {check_name}.",
                "required_facts": fact_keys,
                "ground_truth_field": "",
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "source": "tool_call",
                    "tool_name": tc.tool_name,
                    "reason": f"Inferred from observed fetch call to '{tc.tool_name}'.",
                },
            })

        if self._mode == "aggressive":
            search_calls = [
                tc
                for tc in self._observed.tool_calls
                if _tool_name_to_verb(tc.tool_name) == "search"
            ]
            for tc in search_calls:
                check_name = (
                    tc.tool_name
                    .replace("search_", "")
                    .replace("query_", "")
                    .replace("find_", "")
                )
                check = f"{check_name}_performed"
                preconditions.append({
                    "check": check,
                    "description": f"Agent must search {check_name}.",
                    "required_facts": [],
                    "ground_truth_field": "",
                    "review_required": True,
                    "inferred_from": {
                        "run_id": self._observed.run_id,
                        "source": "tool_call",
                        "tool_name": tc.tool_name,
                        "reason": f"Inferred from observed search call to '{tc.tool_name}'. May not be a required precondition.",
                    },
                })

        return preconditions

    def _infer_allowed_tools(self) -> dict[str, dict[str, Any]]:
        allowed: dict[str, dict[str, Any]] = {}
        seen = set()

        for tc in self._observed.tool_calls:
            if tc.tool_name in seen:
                continue
            seen.add(tc.tool_name)

            verb = _tool_name_to_verb(tc.tool_name)

            entry: dict[str, Any] = {
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "reason": f"Observed call to '{tc.tool_name}' with arguments {list(tc.arguments.keys())}.",
                },
            }

            entity_arg = ""
            for arg_name in tc.arguments.keys():
                if arg_name.lower() in (
                    "artifact_id",
                    "ticket_id",
                    "customer_id",
                    "id",
                    "entity_id",
                ):
                    entity_arg = arg_name
                    break
            if entity_arg:
                entry["entity_arg"] = entity_arg

            if isinstance(tc.return_value, dict):
                stripped = {
                    k: v
                    for k, v in tc.return_value.items()
                    if k not in ("id", "_id", "subsystem", "timestamp", "type")
                    and not isinstance(v, (dict, list))
                }
                if stripped:
                    entry["returns"] = stripped

            artifact_id = entity_arg or tc.tool_name
            entry["artifact_id"] = artifact_id

            allowed[tc.tool_name] = entry

        return allowed

    def _infer_roles(self) -> dict[str, dict[str, Any]]:
        subsystems: set[str] = set()
        for tc in self._observed.tool_calls:
            return_val = tc.return_value
            if isinstance(return_val, dict):
                sub = return_val.get("subsystem", "")
                if sub:
                    subsystems.add(sub)

        if not subsystems:
            return {}

        return {
            "agent": {
                "subsystems": sorted(subsystems),
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "reason": "Subsystems inferred from tool names and return values.",
                },
            },
        }

    def generate_review_checklist(self) -> str:
        lines = [
            "# GroundEval Draft Config Review Checklist",
            "",
            f"Config generated from observed run: `{self._observed.run_id}`",
            f"Framework: {self._observed.framework}",
            f"Crew class: {self._observed.agent_class}",
            f"Draft mode: {self._mode}",
            "",
            "## Before you can use this config for deterministic scoring:",
            "",
        ]

        preconditions = self._infer_preconditions()
        if preconditions:
            lines.append("### Preconditions")
            lines.append("")
            for pc in preconditions:
                lines.append(
                    f"- [ ] **{pc['check']}**: {pc.get('inferred_from', {}).get('reason', 'Review needed.')}"
                )
                if pc.get("ground_truth_field"):
                    lines.append(
                        f"  - Ground truth field: `{pc['ground_truth_field']}` — verify this is the canonical source of truth."
                    )
                if pc.get("required_facts"):
                    lines.append(
                        f"  - Required facts: {pc['required_facts']} — verify these are all necessary and sufficient."
                    )
            lines.append("")

        allowed = self._infer_allowed_tools()
        if allowed:
            lines.append("### Allowed Tools / Fixture Returns")
            lines.append("")
            for name, cfg in allowed.items():
                lines.append(
                    f"- [ ] **{name}**: entity_arg=`{cfg.get('entity_arg', '')}`, artifact_id=`{cfg.get('artifact_id', '')}`"
                )
                returns = cfg.get("returns", {})
                if returns:
                    lines.append(
                        f"  - Returns: {returns} — verify these are the correct ground truth values."
                    )
            lines.append("")

        roles = self._infer_roles()
        if roles:
            lines.append("### Roles and Subsystem Access")
            lines.append("")
            for role_name, role_cfg in roles.items():
                lines.append(
                    f"- [ ] **{role_name}**: subsystems={role_cfg.get('subsystems', [])} — verify allowed access boundaries."
                )
            lines.append("")

        lines.append("### Decision Field")
        decision_field = "should_act"
        answer = self._observed.final_answer or {}
        for candidate in ("should_act", "all_preconditions_pass", "should_escalate"):
            if candidate in answer:
                decision_field = candidate
                break
        lines.append(
            f"- [ ] `decision_field: {decision_field}` — verify this matches your domain convention."
        )
        lines.append("")

        lines.append("### General")
        lines.append("")
        lines.append("- [ ] Task description updated from inferred placeholder.")
        lines.append(
            "- [ ] Valid action logic (`all_preconditions_pass`) matches domain rules."
        )
        lines.append("- [ ] Artifact ground truth values verified as canonical.")
        lines.append("")
        lines.append("## After review")
        lines.append("")
        lines.append("Run: `groundeval validate --config draft_config/config.yaml`")
        lines.append("Then: `groundeval task --config draft_config/config.yaml`")
        lines.append("")
        lines.append(
            "Or mark as reviewed: update `groundeval.config_status` to `reviewed` and `groundeval.reviewed` to `true`."
        )
        lines.append("")

        return "\n".join(lines)

    def generate_observe_report(self) -> str:
        lines = [
            "# GroundEval Observation Report",
            "",
            f"Run ID: `{self._observed.run_id}`",
            f"Framework: {self._observed.framework}",
            f"Crew class: {self._observed.agent_class}",
            f"Total latency: {self._observed.total_latency_ms:.0f}ms",
            f"Tool calls recorded: {len(self._observed.tool_calls)}",
            "",
            "## Observed Tool Calls",
            "",
        ]

        for i, tc in enumerate(self._observed.tool_calls):
            lines.append(f"### {i + 1}. {tc.tool_name}")
            lines.append(f"- Arguments: {json.dumps(tc.arguments, default=str)}")
            lines.append(f"- Latency: {tc.latency_ms:.0f}ms")
            return_preview = json.dumps(tc.return_value, default=str)
            if len(return_preview) > 500:
                return_preview = return_preview[:500] + "..."
            lines.append(f"- Return value: {return_preview}")
            lines.append("")

        lines.append("## Final Answer")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self._observed.final_answer, indent=2, default=str))
        lines.append("```")
        lines.append("")

        lines.append("## Generated Draft Config")
        lines.append("")
        lines.append(
            "GroundEval drafted a config from this observed behavior. "
            "Review is required before deterministic scoring because observed "
            "behavior is not ground truth. The agent may have called the wrong "
            "tool, retrieved the wrong artifact, skipped a required check, or "
            "relied on a field that should not define correctness."
        )
        lines.append("")
        lines.append("See `draft_config/REVIEW.md` for the review checklist.")
        lines.append("")

        return "\n".join(lines)


def write_draft_output(
    output_dir: str | Path,
    observed: ObservedRun,
    generator: DraftGenerator,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    draft_dir = out / "draft_config"
    draft_dir.mkdir(parents=True, exist_ok=True)

    config = generator.generate()

    config_path = draft_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    tool_map = config.get("agent", {}).get("tool_map", {})
    tool_map_path = draft_dir / "tool_map.yaml"
    with open(tool_map_path, "w") as f:
        yaml.dump(
            {"tool_map": tool_map},
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    task_contracts_dir = draft_dir / "task_contracts"
    task_contracts_dir.mkdir(parents=True, exist_ok=True)
    contracts = config.get("task_contracts", [])
    for tc in contracts:
        tc_name = tc.get("name", "inferred_task")
        tc_path = task_contracts_dir / f"{tc_name}.yaml"
        with open(tc_path, "w") as f:
            yaml.dump(
                tc, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

    artifacts_dir = draft_dir / "artifacts" / "observed"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for i, tc in enumerate(observed.tool_calls, start=1):
        if isinstance(tc.return_value, dict):
            artifact = dict(tc.return_value)
            artifact_id = artifact.get("id") or artifact.get("_id") or tc.tool_name
            artifact.setdefault("id", artifact_id)
            artifact.setdefault("source", "observed")
            artifact.setdefault("observed_run_id", observed.run_id)
            safe_name = tc.tool_name.replace("/", "_").replace(" ", "_")
            unique_id = artifact_id.replace("/", "_").replace(" ", "_")
            artifact_path = artifacts_dir / f"{i:03d}_{safe_name}_{unique_id}.json"
            with open(artifact_path, "w") as f:
                json.dump(artifact, f, indent=2, default=str)

    review_path = draft_dir / "REVIEW.md"
    with open(review_path, "w") as f:
        f.write(generator.generate_review_checklist())

    observed_path = out / "observed_run.json"
    with open(observed_path, "w") as f:
        json.dump(observed.to_dict(), f, indent=2, default=str)

    report_path = out / "observe_report.md"
    with open(report_path, "w") as f:
        f.write(generator.generate_observe_report())

    logger.info(f"Observed run written to {observed_path}")
    logger.info(f"Observation report written to {report_path}")
    logger.info(f"Draft config written to {draft_dir}/")
    logger.info(f"Review checklist written to {review_path}")

    return out
