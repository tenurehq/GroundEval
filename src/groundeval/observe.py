from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from .adapters import YamlAccessPolicy
from .core import AgentTrajectory, TaskContract, ToolCall
from .scorers import aggregate_task_results, score_framework_observed_run

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
    agent_id: str | None = None
    agent_name: str | None = None
    node_name: str | None = None
    workflow_run_id: str | None = None
    branch_id: str | None = None
    parent_event_id: str | None = None


@dataclass
class ObservedRun:
    run_id: str
    framework: str
    agent_class: str
    tool_calls: list[ObservedToolCall] = field(default_factory=list)
    final_answer: Any = None
    total_latency_ms: float = 0.0
    framework_extra: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservedRun":
        return cls(
            run_id=d["run_id"],
            framework=d["framework"],
            agent_class=d["agent_class"],
            tool_calls=[ObservedToolCall(**tc) for tc in d.get("tool_calls", [])],
            final_answer=d.get("final_answer"),
            total_latency_ms=d.get("total_latency_ms", 0.0),
            framework_extra=d.get("framework_extra"),
        )


class RecordingRuntime:
    def __init__(self):
        self._call_log: list[ObservedToolCall] = []

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        return_value: Any,
        latency_ms: float,
        agent_id: str | None = None,
        agent_name: str | None = None,
        node_name: str | None = None,
        workflow_run_id: str | None = None,
        branch_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> None:
        self._call_log.append(
            ObservedToolCall(
                tool_name=tool_name,
                arguments=arguments,
                return_value=return_value,
                latency_ms=latency_ms,
                agent_id=agent_id,
                agent_name=agent_name,
                node_name=node_name,
                workflow_run_id=workflow_run_id,
                branch_id=branch_id,
                parent_event_id=parent_event_id,
            )
        )

    @property
    def call_log(self) -> list[ObservedToolCall]:
        return list(self._call_log)


@runtime_checkable
class AgentObserver(Protocol):
    def load_agent(self, class_path: str) -> Any: ...

    def instrument_agent(
        self,
        agent: Any,
        recording: RecordingRuntime,
    ) -> Any: ...

    def execute_agent(self, agent: Any) -> Any: ...

    def set_max_steps(self, agent: Any, max_steps: int) -> None: ...


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
    elif framework == "maf":
        try:
            from .framework_adapters.maf_adapter import MafObserver

            register_observer("maf", MafObserver())
        except ImportError:
            pass
    elif framework == "langgraph":
        try:
            from .framework_adapters.langgraph_adapter import LangGraphObserver

            register_observer("langgraph", LangGraphObserver())
        except ImportError:
            pass


def _parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in '[{"':
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        if _json_repair is not None:
            try:
                repaired = _json_repair.repair_json(value)
                if isinstance(repaired, str):
                    return json.loads(repaired)
                return repaired
            except Exception:
                return value
        return value


def _parse_observed_answer(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, str):
        return _parse_json_string(result)
    if isinstance(result, (dict, list, int, float, bool)):
        return result

    raw = getattr(result, "raw", None)
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        return _parse_json_string(raw)

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

    if hasattr(result, "model_dump") and callable(result.model_dump):
        try:
            return result.model_dump()
        except Exception:
            pass

    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            return result.to_dict()
        except Exception:
            pass

    return str(result)


def observe_agent(
    framework: str,
    class_path: str,
    max_steps: int = 10,
) -> ObservedRun:
    observer = _get_observer(framework)
    agent = observer.load_agent(class_path)
    recording = RecordingRuntime()
    run_id = f"observed_{class_path.replace('.', '_')}_{int(time.time())}"

    instrumented = observer.instrument_agent(agent, recording)
    observer.set_max_steps(instrumented, max_steps)

    t_start = time.time()
    result = observer.execute_agent(instrumented)
    total_latency = (time.time() - t_start) * 1000
    final_answer = _parse_observed_answer(result)

    framework_extra = None
    rich_run = getattr(instrumented, "_groundeval_framework_observed_run", None)
    if rich_run is not None and hasattr(rich_run, "to_dict"):
        try:
            framework_extra = rich_run.to_dict()
            if getattr(rich_run, "run_id", None):
                run_id = str(rich_run.run_id)
            if getattr(rich_run, "final_output", None) is not None:
                final_answer = rich_run.final_output
            if getattr(rich_run, "total_latency_ms", None) is not None:
                total_latency = float(rich_run.total_latency_ms)
        except Exception:
            framework_extra = None

    return ObservedRun(
        run_id=run_id,
        framework=framework,
        agent_class=class_path,
        tool_calls=recording.call_log,
        final_answer=final_answer,
        total_latency_ms=total_latency,
        framework_extra=framework_extra,
    )


def observe_crew(
    agent_class_path: str,
    max_steps: int = 10,
) -> ObservedRun:
    return observe_agent(
        framework="crewai",
        class_path=agent_class_path,
        max_steps=max_steps,
    )


def _is_empty_return_value(value: Any) -> bool:
    return value is None or value == {} or value == []


def _observed_run_to_trajectory(observed: ObservedRun, task_id: str) -> AgentTrajectory:
    tool_calls = [
        ToolCall(
            tool_name=tc.tool_name,
            arguments=dict(tc.arguments or {}),
            result_ids=[],
            timestamp_applied=None,
            horizon_violation=False,
            actor_gate_violation=False,
            subsystem_violation=False,
            returned_empty=_is_empty_return_value(tc.return_value),
            latency_ms=float(tc.latency_ms),
            agent_name=tc.agent_name,
            node_name=tc.node_name,
            workflow_run_id=tc.workflow_run_id,
            branch_id=tc.branch_id,
            parent_event_id=tc.parent_event_id,
        )
        for tc in observed.tool_calls
    ]
    final_answer = (
        observed.final_answer if isinstance(observed.final_answer, dict) else {}
    )
    return AgentTrajectory(
        task_id=task_id,
        tool_calls=tool_calls,
        final_answer=final_answer,
        total_latency_ms=float(observed.total_latency_ms),
    )


def score_observed_run(
    observed: ObservedRun,
    cfg: dict[str, Any],
    config_path: str | Path | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    contracts = [TaskContract.from_dict(tc) for tc in cfg.get("task_contracts", [])]
    actors = cfg.get("actors", {})
    roles = cfg.get("roles", {})
    policy = YamlAccessPolicy({"actors": actors, "roles": roles})

    results = []
    trajectories = []

    for contract in contracts:
        trajectory = _observed_run_to_trajectory(observed, task_id=contract.name)
        final_answer = (
            observed.final_answer if isinstance(observed.final_answer, dict) else {}
        )
        result = score_framework_observed_run(
            trajectory=trajectory,
            final_answer=final_answer,
            contract=contract,
            policy=policy,
            actor=contract.actor,
            role=contract.role,
        )
        results.append(result)
        trajectories.append(trajectory.to_dict())

    summary = aggregate_task_results(results)
    payload = {
        "meta": {
            "framework": observed.framework,
            "agent_class": observed.agent_class,
            "run_id": observed.run_id,
            "framework_native_scoring": True,
            "config_path": str(config_path) if config_path is not None else None,
        },
        "summary": summary,
        "results": [r.to_dict() for r in results],
        "trajectories": trajectories,
    }
    return results, payload


class DraftGenerator:
    def __init__(self, observed: ObservedRun, mode: str = "standard"):
        self._observed = observed
        self._mode = mode

    def generate(self) -> dict[str, Any]:
        preconditions = self._infer_preconditions()
        tool_expectations = self._infer_tool_expectations()

        decision_field = "should_act"
        if isinstance(self._observed.final_answer, dict):
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

        if tool_expectations:
            task_contract["tool_expectations"] = tool_expectations

        return {
            "output_dir": "./eval_output",
            "agent": {
                "framework": self._observed.framework,
                "agent_class": self._observed.agent_class,
            },
            "task_contracts": [task_contract],
            "groundeval": {
                "config_status": "draft",
                "generated_from_observation": True,
                "reviewed": False,
                "draft_mode": self._mode,
                "observed_run_id": self._observed.run_id,
            },
        }

    def _infer_preconditions(self) -> list[dict[str, Any]]:
        answer = (
            self._observed.final_answer
            if isinstance(self._observed.final_answer, dict)
            else {}
        )
        preconditions_verified = answer.get("preconditions_verified", [])

        if preconditions_verified:
            return self._infer_from_structured_answer(preconditions_verified)

        if self._mode == "conservative":
            return []

        return self._infer_from_tool_calls()

    def _infer_from_structured_answer(
        self, preconditions_verified: list[dict]
    ) -> list[dict[str, Any]]:
        preconditions = []
        observed_tool_names = [tc.tool_name for tc in self._observed.tool_calls]
        default_required_tool = observed_tool_names[0] if observed_tool_names else ""

        for pc in preconditions_verified:
            check = pc.get("check", "unknown_check")
            facts = pc.get("facts_found", {})
            fact_keys = list(facts.keys()) if isinstance(facts, dict) else []
            precondition = {
                "check": check,
                "description": f"Agent must verify: {check}",
                "required_facts": fact_keys,
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "source": "structured_answer",
                    "reason": f"Observed check '{check}' in agent answer.",
                },
            }
            if default_required_tool:
                precondition["required_tool"] = default_required_tool
            if fact_keys:
                precondition["expected_field"] = fact_keys[0]
            preconditions.append(precondition)

        return preconditions

    def _infer_from_tool_calls(self) -> list[dict[str, Any]]:
        preconditions = []
        for tc in self._observed.tool_calls:
            precondition = {
                "check": f"{tc.tool_name}_observed",
                "description": f"Observed native tool call '{tc.tool_name}'.",
                "required_facts": list(tc.return_value.keys())
                if isinstance(tc.return_value, dict)
                else [],
                "required_tool": tc.tool_name,
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "source": "tool_call",
                    "tool_name": tc.tool_name,
                    "reason": f"Inferred directly from observed native tool call '{tc.tool_name}'.",
                },
            }
            if isinstance(tc.return_value, dict) and tc.return_value:
                precondition["expected_field"] = next(iter(tc.return_value.keys()))
            preconditions.append(precondition)
        return preconditions

    def _infer_tool_expectations(self) -> list[dict[str, Any]]:
        expectations = []
        seen = set()
        for tc in self._observed.tool_calls:
            key = (tc.tool_name, json.dumps(tc.arguments, default=str, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            expectation = {
                "tool": tc.tool_name,
                "match_args": tc.arguments,
                "review_required": True,
                "inferred_from": {
                    "run_id": self._observed.run_id,
                    "reason": f"Observed native tool call '{tc.tool_name}'.",
                },
            }
            if isinstance(tc.return_value, dict):
                expectation["expected_return"] = tc.return_value
            expectations.append(expectation)
        return expectations

    def generate_review_checklist(self) -> str:
        lines = [
            "# GroundEval Draft Config Review Checklist",
            "",
            f"Config generated from observed run: `{self._observed.run_id}`",
            f"Framework: {self._observed.framework}",
            f"Agent class: {self._observed.agent_class}",
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
                if pc.get("required_tool"):
                    lines.append(f"  - Required native tool: `{pc['required_tool']}`")
                if pc.get("required_facts"):
                    lines.append(f"  - Required facts: {pc['required_facts']}")
            lines.append("")

        expectations = self._infer_tool_expectations()
        if expectations:
            lines.append("### Observed Native Tools")
            lines.append("")
            for item in expectations:
                lines.append(
                    f"- [ ] **{item['tool']}**: verify exact native tool name, arguments, and return behavior."
                )
            lines.append("")

        lines.append("### Decision Field")
        decision_field = "should_act"
        answer = (
            self._observed.final_answer
            if isinstance(self._observed.final_answer, dict)
            else {}
        )
        for candidate in ("should_act", "all_preconditions_pass", "should_escalate"):
            if candidate in answer:
                decision_field = candidate
                break
        lines.append(f"- [ ] `decision_field: {decision_field}`")
        lines.append("")
        lines.append("### General")
        lines.append("")
        lines.append("- [ ] Task description updated from inferred placeholder.")
        lines.append("- [ ] Required native tools reflect the reviewed contract.")
        lines.append("- [ ] Tool expectations only include behavior you want scored.")
        lines.append("")
        lines.append("## After review")
        lines.append("")
        lines.append("Run: `groundeval validate --config draft_config/config.yaml`")
        lines.append(
            "Then: `groundeval observe --framework <framework> --agent-class <path> --config draft_config/config.yaml --score`"
        )
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
            f"Agent class: {self._observed.agent_class}",
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
            if tc.agent_name or tc.agent_id:
                lines.append(f"- Agent: {tc.agent_name or ''} ({tc.agent_id or ''})")
            if tc.node_name:
                lines.append(f"- Node: {tc.node_name}")
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
            "GroundEval drafted a config from this observed framework behavior. Review is required before deterministic scoring."
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
