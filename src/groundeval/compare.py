from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_json_files(old_path: str | Path, new_path: str | Path) -> str:
    old_file = Path(old_path)
    new_file = Path(new_path)

    if not old_file.exists():
        raise FileNotFoundError(f"Compare file not found: {old_file}")
    if not new_file.exists():
        raise FileNotFoundError(f"Compare file not found: {new_file}")

    with open(old_file) as f:
        old_data = json.load(f)
    with open(new_file) as f:
        new_data = json.load(f)

    old_kind = _detect_payload_kind(old_data)
    new_kind = _detect_payload_kind(new_data)

    if old_kind != new_kind:
        return "\n".join([
            "GroundEval Compare",
            "",
            f"Old file: {old_file}",
            f"New file: {new_file}",
            "",
            f"File types differ:",
            f"- old: {old_kind}",
            f"- new: {new_kind}",
            "",
            "Comparison aborted because the two JSON files are not the same payload type.",
        ])

    if old_kind == "observed_scores":
        return _compare_observed_scores(old_data, new_data, old_file, new_file)

    if old_kind == "observed_run":
        return _compare_observed_runs(old_data, new_data, old_file, new_file)

    if old_kind == "task_results":
        return _compare_task_results(old_data, new_data, old_file, new_file)

    return "\n".join([
        "GroundEval Compare",
        "",
        f"Old file: {old_file}",
        f"New file: {new_file}",
        "",
        "Unsupported JSON shape.",
    ])


def _detect_payload_kind(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return "unknown"

    if "summary" in data and "results" in data and "trajectories" in data:
        return "observed_scores"

    if "run_id" in data and "framework" in data and "tool_calls" in data:
        return "observed_run"

    if "meta" in data and "summary" in data:
        return "task_results"

    return "unknown"


def _compare_observed_scores(
    old_data: dict[str, Any],
    new_data: dict[str, Any],
    old_file: Path,
    new_file: Path,
) -> str:
    lines: list[str] = [
        "GroundEval Compare",
        "",
        f"Old file: {old_file}",
        f"New file: {new_file}",
        "",
    ]

    old_summary = old_data.get("summary", {}) or {}
    new_summary = new_data.get("summary", {}) or {}

    score_lines = _diff_score_block(old_summary, new_summary)
    if score_lines:
        lines.append("Scores changed:")
        lines.extend(score_lines)
        lines.append("")

    old_result_map = _map_results_by_task(old_data.get("results", []))
    new_result_map = _map_results_by_task(new_data.get("results", []))

    per_task_lines = _diff_per_task_scores(old_result_map, new_result_map)
    if per_task_lines:
        lines.append("Per-task changes:")
        lines.extend(per_task_lines)
        lines.append("")

    old_violations = _collect_violation_strings_from_scored_payload(old_data)
    new_violations = _collect_violation_strings_from_scored_payload(new_data)

    new_only = sorted(new_violations - old_violations)
    fixed = sorted(old_violations - new_violations)

    if new_only:
        lines.append("New violations:")
        for item in new_only:
            lines.append(f"- {item}")
        lines.append("")

    if fixed:
        lines.append("Fixed violations:")
        for item in fixed:
            lines.append(f"- {item}")
        lines.append("")

    trajectory_lines = _diff_trajectories(
        old_data.get("trajectories", []),
        new_data.get("trajectories", []),
    )
    if trajectory_lines:
        lines.append("Trajectory diff:")
        lines.extend(trajectory_lines)
        lines.append("")

    if len(lines) == 5:
        lines.append("No meaningful differences found.")

    return "\n".join(lines).rstrip()


def _compare_observed_runs(
    old_data: dict[str, Any],
    new_data: dict[str, Any],
    old_file: Path,
    new_file: Path,
) -> str:
    lines: list[str] = [
        "GroundEval Compare",
        "",
        f"Old file: {old_file}",
        f"New file: {new_file}",
        "",
    ]

    old_calls = old_data.get("tool_calls", []) or []
    new_calls = new_data.get("tool_calls", []) or []

    if len(old_calls) != len(new_calls):
        lines.append("Tool call count changed:")
        lines.append(f"- old: {len(old_calls)}")
        lines.append(f"- new: {len(new_calls)}")
        lines.append("")

    old_seq = _tool_sequence_from_observed_run(old_data)
    new_seq = _tool_sequence_from_observed_run(new_data)
    if old_seq != new_seq:
        lines.append("Trajectory diff:")
        lines.append(f"- old: {_format_sequence(old_seq)}")
        lines.append(f"- new: {_format_sequence(new_seq)}")
        lines.append("")

    old_answer = old_data.get("final_answer", {})
    new_answer = new_data.get("final_answer", {})

    answer_lines = _compare_final_answer_shape(old_answer, new_answer)
    if answer_lines:
        lines.append("Final answer diff:")
        lines.extend(answer_lines)
        lines.append("")

    if len(lines) == 5:
        lines.append("No meaningful differences found.")

    return "\n".join(lines).rstrip()


def _compare_task_results(
    old_data: dict[str, Any],
    new_data: dict[str, Any],
    old_file: Path,
    new_file: Path,
) -> str:
    lines: list[str] = [
        "GroundEval Compare",
        "",
        f"Old file: {old_file}",
        f"New file: {new_file}",
        "",
    ]

    old_summary = old_data.get("summary", {}) or {}
    new_summary = new_data.get("summary", {}) or {}

    score_lines = _diff_score_block(old_summary, new_summary)
    if score_lines:
        lines.append("Scores changed:")
        lines.extend(score_lines)
        lines.append("")

    old_per_task = _map_results_by_task(old_summary.get("per_task", []))
    new_per_task = _map_results_by_task(new_summary.get("per_task", []))
    per_task_lines = _diff_per_task_scores(old_per_task, new_per_task)
    if per_task_lines:
        lines.append("Per-task changes:")
        lines.extend(per_task_lines)
        lines.append("")

    if len(lines) == 5:
        lines.append("No meaningful differences found.")

    return "\n".join(lines).rstrip()


def _diff_score_block(
    old_summary: dict[str, Any], new_summary: dict[str, Any]
) -> list[str]:
    lines: list[str] = []
    for key in (
        "counterfactual_score",
        "silence_score",
        "perspective_score",
        "overall_score",
        "accuracy",
        "total_violations",
    ):
        old_val = old_summary.get(key)
        new_val = new_summary.get(key)
        if old_val != new_val and old_val is not None and new_val is not None:
            lines.append(f"- {key}: {_fmt_number(old_val)} → {_fmt_number(new_val)}")
    return lines


def _diff_per_task_scores(
    old_map: dict[str, dict[str, Any]],
    new_map: dict[str, dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    all_tasks = sorted(set(old_map.keys()) | set(new_map.keys()))
    for task_name in all_tasks:
        old_item = old_map.get(task_name)
        new_item = new_map.get(task_name)

        if old_item is None:
            lines.append(f"- {task_name}: added in new file")
            continue
        if new_item is None:
            lines.append(f"- {task_name}: missing in new file")
            continue

        old_score = old_item.get("overall_score")
        new_score = new_item.get("overall_score")
        if old_score != new_score and old_score is not None and new_score is not None:
            lines.append(
                f"- {task_name}: overall_score {_fmt_number(old_score)} → {_fmt_number(new_score)}"
            )
    return lines


def _map_results_by_task(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        task_name = item.get("task_name") or item.get("task_id") or f"index_{i}"
        result[str(task_name)] = item
    return result


def _collect_violation_strings_from_scored_payload(data: dict[str, Any]) -> set[str]:
    findings: set[str] = set()

    for result in data.get("results", []) or []:
        task_name = str(result.get("task_name", "unknown_task"))

        for key in (
            "horizon_violations",
            "actor_gate_violations",
            "subsystem_violations",
            "dead_ends_hit",
        ):
            value = result.get(key, 0)
            if isinstance(value, int) and value > 0:
                findings.add(f"{task_name}: {key}={value}")

        for pc in result.get("precondition_results", []) or []:
            check = str(pc.get("check", "unknown_check"))
            if pc.get("error"):
                findings.add(
                    f"{task_name}: precondition '{check}' error: {pc['error']}"
                )
            if pc.get("evidence_supported") is False:
                findings.add(
                    f"{task_name}: precondition '{check}' evidence unsupported"
                )
            if pc.get("verified") is False:
                findings.add(f"{task_name}: precondition '{check}' not verified")
            reasons = pc.get("reasons", [])
            if isinstance(reasons, list):
                for reason in reasons:
                    findings.add(
                        f"{task_name}: precondition '{check}' reason: {reason}"
                    )

        meta = result.get("meta", {}) or {}
        multi_agent = meta.get("multi_agent", {}) or {}

        for item in multi_agent.get("required_agents", []) or []:
            if item.get("observed") is False:
                req = item.get("requirement", {}) or {}
                label = req.get("agent_name") or req.get("agent_id") or "unknown_agent"
                findings.add(f"{task_name}: required agent not observed: {label}")

        for item in multi_agent.get("required_handoffs", []) or []:
            if item.get("observed") is False:
                req = item.get("requirement", {}) or {}
                src = req.get("from_agent") or req.get("from_executor_id") or "unknown"
                dst = req.get("to_agent") or req.get("to_executor_id") or "unknown"
                findings.add(
                    f"{task_name}: required handoff not observed: {src} → {dst}"
                )

        for item in multi_agent.get("required_agent_tool_expectations", []) or []:
            if item.get("observed") is False:
                req = item.get("requirement", {}) or {}
                tool = req.get("tool") or "unknown_tool"
                agent = req.get("agent_name") or req.get("agent_id") or "unknown_agent"
                findings.add(
                    f"{task_name}: required agent tool not observed: {agent} -> {tool}"
                )

    for traj in data.get("trajectories", []) or []:
        task_id = str(traj.get("task_id", "unknown_task"))
        for i, call in enumerate(traj.get("tool_calls", []) or [], start=1):
            tool_name = str(call.get("tool_name", "unknown_tool"))
            agent_name = call.get("agent_name")
            prefix = f"{task_id}: "
            if agent_name:
                prefix += f"{agent_name} "
            prefix += f"{tool_name} call {i}"

            if call.get("horizon_violation") is True:
                findings.add(f"{prefix} horizon violation")
            if call.get("actor_gate_violation") is True:
                findings.add(f"{prefix} actor gate violation")
            if call.get("subsystem_violation") is True:
                findings.add(f"{prefix} subsystem violation")
            if call.get("returned_empty") is True:
                findings.add(f"{prefix} returned empty")

    return findings


def _diff_trajectories(
    old_trajectories: list[dict[str, Any]],
    new_trajectories: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    old_map = _map_trajectories_by_task(old_trajectories)
    new_map = _map_trajectories_by_task(new_trajectories)
    all_tasks = sorted(set(old_map.keys()) | set(new_map.keys()))

    for task_name in all_tasks:
        old_traj = old_map.get(task_name)
        new_traj = new_map.get(task_name)
        if old_traj is None or new_traj is None:
            continue

        old_seq = _tool_sequence_from_trajectory(old_traj)
        new_seq = _tool_sequence_from_trajectory(new_traj)

        if old_seq != new_seq:
            if len(all_tasks) > 1:
                lines.append(f"- task: {task_name}")
                lines.append(f"  old: {_format_sequence(old_seq)}")
                lines.append(f"  new: {_format_sequence(new_seq)}")
            else:
                lines.append(f"- old: {_format_sequence(old_seq)}")
                lines.append(f"- new: {_format_sequence(new_seq)}")

    return lines


def _map_trajectories_by_task(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        task_id = item.get("task_id") or item.get("task_name") or f"index_{i}"
        result[str(task_id)] = item
    return result


def _tool_sequence_from_trajectory(item: dict[str, Any]) -> list[str]:
    seq: list[str] = []
    for call in item.get("tool_calls", []) or []:
        tool_name = call.get("tool_name")
        if tool_name:
            seq.append(str(tool_name))
    return seq


def _tool_sequence_from_observed_run(data: dict[str, Any]) -> list[str]:
    seq: list[str] = []
    for call in data.get("tool_calls", []) or []:
        tool_name = call.get("tool_name")
        if tool_name:
            seq.append(str(tool_name))
    return seq


def _compare_final_answer_shape(old_answer: Any, new_answer: Any) -> list[str]:
    lines: list[str] = []

    if isinstance(old_answer, dict) and isinstance(new_answer, dict):
        old_keys = set(old_answer.keys())
        new_keys = set(new_answer.keys())

        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)

        if added:
            lines.append(f"- added keys: {', '.join(added)}")
        if removed:
            lines.append(f"- removed keys: {', '.join(removed)}")

        if (
            "preconditions_verified" in old_answer
            or "preconditions_verified" in new_answer
        ):
            old_len = len(old_answer.get("preconditions_verified", []) or [])
            new_len = len(new_answer.get("preconditions_verified", []) or [])
            if old_len != new_len:
                lines.append(f"- preconditions_verified count: {old_len} → {new_len}")

    elif old_answer != new_answer:
        lines.append("- final_answer content changed")

    return lines


def _format_sequence(seq: list[str]) -> str:
    if not seq:
        return "(no tool calls)"
    return " → ".join(seq)


def _fmt_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
