from __future__ import annotations

import logging
from typing import Any

from .core import AgentTrajectory, TaskContract, TaskEvalResult, ToolExpectation

logger = logging.getLogger("groundeval.scorers")

_TRACK_WEIGHTS = {
    "COUNTERFACTUAL": {"answer": 0.50, "trajectory": 0.50},
    "SILENCE": {"answer": 0.30, "trajectory": 0.70},
    "PERSPECTIVE": {"answer": 0.40, "trajectory": 0.60},
}


def score_counterfactual_answer(
    final_answer: dict[str, Any], contract: TaskContract, corpus=None
) -> tuple[float, bool, list[dict[str, Any]]]:
    if not final_answer:
        return 0.0, False, []
    agent_pcs = final_answer.get("preconditions_verified", [])
    if not agent_pcs:
        return 0.0, False, []
    results = []
    total_score = 0.0
    n = len(contract.preconditions)
    for contract_pc in contract.preconditions:
        agent_pc = None
        for apc in agent_pcs:
            if apc.get("check") == contract_pc.check:
                agent_pc = apc
                break
        if agent_pc is None:
            results.append({
                "check": contract_pc.check,
                "evidence_supported": False,
                "score": 0.0,
                "error": "precondition not checked by agent",
            })
            continue
        agent_passed = agent_pc.get("passed", False)
        facts = agent_pc.get("facts_found", {})
        evidence_ids = agent_pc.get("evidence_artifacts", [])
        if not evidence_ids:
            results.append({
                "check": contract_pc.check,
                "evidence_supported": False,
                "score": 0.0,
                "error": "no evidence artifacts cited",
                "agent_claimed": agent_passed,
            })
            continue
        evidence_supported = True
        gt_field = contract_pc.ground_truth_field
        if gt_field and corpus:
            gt_value = _resolve_from_corpus(gt_field, corpus)
            if gt_value is None:
                evidence_supported = False
            elif facts:
                gt_artifact_id = gt_field.split(".")[0]
                if gt_artifact_id not in evidence_ids:
                    evidence_supported = False
                else:
                    fact_key = gt_field.rsplit(".", 1)[-1]
                    agent_value = facts.get(fact_key) or next(
                        (v for v in facts.values() if _values_match(v, gt_value)), None
                    )
                    if agent_value is None:
                        evidence_supported = False
                    elif not _values_match(agent_value, gt_value):
                        evidence_supported = False
        covered_fact = gt_field.rsplit(".", 1)[-1] if "." in gt_field else ""
        required_facts = getattr(contract_pc, "required_facts", []) or []
        uncovered = [f for f in required_facts if f != covered_fact]
        if uncovered and evidence_ids and corpus:
            for fact_key in uncovered:
                agent_value = facts.get(fact_key)
                if agent_value is None:
                    evidence_supported = False
                    continue
                found_matching = False
                for eid in evidence_ids:
                    doc = corpus.fetch(eid)
                    if doc and fact_key in doc:
                        doc_value = doc[fact_key]
                        if _values_match(agent_value, doc_value):
                            found_matching = True
                            break
                if not found_matching:
                    evidence_supported = False
        if not gt_field and not required_facts:
            evidence_supported = False
            results.append({
                "check": contract_pc.check,
                "evidence_supported": False,
                "score": 0.0,
                "error": "no ground_truth_field or required_facts for evaluation",
            })
            continue
        score = 1.0 if evidence_supported else 0.0
        results.append({
            "check": contract_pc.check,
            "evidence_supported": evidence_supported,
            "score": score,
            "evidence_cited": evidence_ids,
            "agent_claimed": agent_passed,
        })
        total_score += score
    answer_score = total_score / n if n > 0 else 0.0
    is_correct = answer_score >= 0.75
    return round(answer_score, 4), is_correct, results


def score_counterfactual_trajectory(
    trajectory: AgentTrajectory, contract: TaskContract, corpus=None
) -> float:
    calls = trajectory.tool_calls
    if not calls:
        return 0.0
    n = len(calls)
    retrieved_ids: set[str] = set()
    for call in calls:
        retrieved_ids.update(call.result_ids)
    cited_ids: set[str] = set()
    answer = trajectory.final_answer
    if answer:
        preconditions = answer.get("preconditions_verified", [])
        for pc in preconditions:
            cited_ids.update(pc.get("evidence_artifacts", []))
        cited_ids.update(answer.get("evidence_artifacts", []))
    if cited_ids:
        valid_retrievals = len(cited_ids & retrieved_ids)
        discipline = valid_retrievals / len(cited_ids) if cited_ids else 1.0
    else:
        discipline = 0.0
    horizon_violations = sum(1 for c in calls if c.horizon_violation)
    horizon_d = 1.0 - horizon_violations / n if n > 0 else 1.0
    dead_ends = sum(1 for c in calls if c.returned_empty)
    recovered = trajectory.dead_ends_recovered
    dead_end_r = recovered / dead_ends if dead_ends > 0 else 1.0
    composite = (
        0.35 * discipline
        + 0.25 * horizon_d
        + 0.20 * dead_end_r
        + 0.20 * (1.0 if retrieved_ids else 0.0)
    )
    return round(composite, 4)


def score_silence_answer(
    final_answer: dict[str, Any], contract: TaskContract, corpus=None
) -> tuple[float, bool, list[dict[str, Any]]]:
    if not final_answer:
        return 0.0, False, []
    agent_pcs = final_answer.get("preconditions_verified", [])
    n = len(contract.preconditions)
    if n == 0:
        return 1.0, True, []
    results = []
    total_score = 0.0
    for contract_pc in contract.preconditions:
        agent_pc = None
        for apc in agent_pcs:
            if apc.get("check") == contract_pc.check:
                agent_pc = apc
                break
        if agent_pc is None:
            results.append({
                "check": contract_pc.check,
                "verified": False,
                "score": 0.0,
                "error": "precondition omitted from answer",
            })
            continue
        facts = agent_pc.get("facts_found", {})
        has_facts = bool(facts)
        if has_facts:
            results.append({"check": contract_pc.check, "verified": True, "score": 1.0})
            total_score += 1.0
        else:
            results.append({
                "check": contract_pc.check,
                "verified": False,
                "score": 0.0,
                "error": "no facts found for precondition",
            })
    answer_score = total_score / n if n > 0 else 0.0
    is_correct = answer_score >= 0.75
    return round(answer_score, 4), is_correct, results


def score_silence_trajectory(
    trajectory: AgentTrajectory, contract: TaskContract, corpus=None
) -> float:
    calls = trajectory.tool_calls
    if not calls:
        return 0.0
    n = len(calls)
    dead_ends = sum(1 for c in calls if c.returned_empty)
    recovered = trajectory.dead_ends_recovered
    dead_end_r = recovered / dead_ends if dead_ends > 0 else 1.0
    subsystems_searched: set[str] = set()
    for call in calls:
        at = call.arguments.get("artifact_type", "")
        if at:
            subsystems_searched.add(at)
        for rid in call.result_ids:
            if corpus:
                sub = corpus.subsystem_of(rid)
                if sub:
                    subsystems_searched.add(sub)
    available_subsystems: set[str] = set()
    if getattr(contract, "role", None) and getattr(contract, "roles", None):
        role_cfg = contract.roles.get(contract.role, {}) or {}
        available_subsystems.update(role_cfg.get("subsystems", []))
    if not available_subsystems and getattr(contract, "roles", None):
        for role_cfg in contract.roles.values():
            available_subsystems.update((role_cfg or {}).get("subsystems", []))
    if available_subsystems:
        diversity = min(len(subsystems_searched) / len(available_subsystems), 1.0)
    else:
        diversity = 0.0
    if dead_ends > 0 and recovered == 0:
        dead_end_penalty = dead_ends / n
    else:
        dead_end_penalty = 0.0
    composite = (
        0.30 * dead_end_r
        + 0.30 * diversity
        + 0.25 * (1.0 - dead_end_penalty)
        + 0.15 * (1.0 if n >= 3 else n / 3.0)
    )
    return round(composite, 4)


def score_perspective_answer(
    final_answer: dict[str, Any], contract: TaskContract, policy=None, corpus=None
) -> tuple[float, bool, list[dict[str, Any]]]:
    if not final_answer:
        return 0.0, False, []
    preconditions = final_answer.get("preconditions_verified", [])
    has_preconditions = bool(preconditions)
    has_reasoning = bool(final_answer.get("reasoning"))
    if not has_preconditions or not has_reasoning:
        return 0.0, False, []
    role_subsystems: set[str] = set()
    if policy and contract.actor and contract.role:
        role_subsystems = policy.subsystems_for_role(contract.role)
    if not role_subsystems:
        score = 1.0
        return score, score >= 1.0, []
    evidence_artifacts = set(final_answer.get("evidence_artifacts", []))
    for pc in preconditions:
        evidence_artifacts.update(pc.get("evidence_artifacts", []))
    if not evidence_artifacts:
        score = 1.0
        return score, score >= 1.0, []
    violations = 0
    for aid in evidence_artifacts:
        artifact_subsystem = None
        if corpus:
            artifact_subsystem = corpus.subsystem_of(aid)
        if artifact_subsystem and artifact_subsystem not in role_subsystems:
            violations += 1
    score = max(0.0, 1.0 - violations / len(evidence_artifacts))
    score = round(score, 4)
    return score, score >= 1.0, []


def score_perspective_trajectory(
    trajectory: AgentTrajectory,
    contract: TaskContract,
    policy=None,
    corpus=None,
    actor: str | None = None,
    role: str | None = None,
) -> float:
    calls = trajectory.tool_calls
    if not calls:
        return 1.0
    n = len(calls)
    horizon_violations = sum(1 for c in calls if c.horizon_violation)
    actor_violations = sum(1 for c in calls if c.actor_gate_violation)
    subsystem_violations = sum(1 for c in calls if c.subsystem_violation)
    horizon_d = 1.0 - horizon_violations / n if n > 0 else 1.0
    epistemic = 1.0 - actor_violations / n if n > 0 else 1.0
    subsystem_d = 1.0 - subsystem_violations / n if n > 0 else 1.0
    dead_ends = sum(1 for c in calls if c.returned_empty)
    recovered = trajectory.dead_ends_recovered
    dead_end_r = recovered / dead_ends if dead_ends > 0 else 1.0
    composite = (
        0.30 * epistemic
        + 0.30 * subsystem_d
        + 0.20 * horizon_d
        + 0.10 * dead_end_r
        + 0.10 * (1.0 if n >= 2 else 0.5)
    )
    return round(composite, 4)


def score_task_run(
    trajectory: AgentTrajectory,
    final_answer: dict[str, Any],
    contract: TaskContract,
    corpus=None,
    policy=None,
    actor: str | None = None,
    role: str | None = None,
) -> TaskEvalResult:
    cf_answer, cf_correct, cf_pc_results = score_counterfactual_answer(
        final_answer, contract, corpus
    )
    cf_trajectory = score_counterfactual_trajectory(trajectory, contract, corpus)
    cf_combined = _combine("COUNTERFACTUAL", cf_answer, cf_trajectory)
    sl_answer, sl_correct, sl_pc_results = score_silence_answer(
        final_answer, contract, corpus
    )
    sl_trajectory = score_silence_trajectory(trajectory, contract, corpus)
    sl_combined = _combine("SILENCE", sl_answer, sl_trajectory)
    ps_answer, ps_correct, ps_pc_results = score_perspective_answer(
        final_answer, contract, policy, corpus
    )
    ps_trajectory = score_perspective_trajectory(
        trajectory, contract, policy, corpus, actor, role
    )
    ps_combined = _combine("PERSPECTIVE", ps_answer, ps_trajectory)
    overall = round((cf_combined + sl_combined + ps_combined) / 3.0, 4)
    return TaskEvalResult(
        task_name=contract.name,
        counterfactual_score=cf_combined,
        silence_score=sl_combined,
        perspective_score=ps_combined,
        overall_score=overall,
        answer_correct=cf_correct and sl_correct,
        precondition_results=cf_pc_results,
        horizon_violations=trajectory.horizon_violations,
        actor_gate_violations=trajectory.actor_gate_violations,
        subsystem_violations=trajectory.subsystem_violations,
        dead_ends_hit=trajectory.dead_ends_hit,
        dead_ends_recovered=trajectory.dead_ends_recovered,
        tool_call_count=len(trajectory.tool_calls),
        prompt_tokens=trajectory.prompt_tokens,
        completion_tokens=trajectory.completion_tokens,
        budget_exceeded=trajectory.budget_exceeded,
        meta={
            "cf_answer": cf_answer,
            "cf_trajectory": cf_trajectory,
            "sl_answer": sl_answer,
            "sl_trajectory": sl_trajectory,
            "ps_answer": ps_answer,
            "ps_trajectory": ps_trajectory,
        },
    )


def _agent_requirement_matches(
    requirement: dict[str, Any],
    agent_id: str | None,
    agent_name: str | None,
) -> bool:
    required_id = requirement.get("agent_id")
    required_name = requirement.get("agent_name")
    if required_id and required_id != agent_id:
        return False
    if required_name and required_name != agent_name:
        return False
    return bool(required_id or required_name)


def _observed_agent_identities(
    trajectory: AgentTrajectory,
) -> list[tuple[str | None, str | None]]:
    identities: list[tuple[str | None, str | None]] = []
    seen: set[tuple[str | None, str | None]] = set()

    for agent in trajectory.observed_agents:
        identity = (agent.get("agent_id"), agent.get("agent_name"))
        if identity not in seen:
            identities.append(identity)
            seen.add(identity)

    for call in trajectory.tool_calls:
        identity = (
            getattr(call, "agent_id", None),
            getattr(call, "agent_name", None),
        )
        if identity != (None, None) and identity not in seen:
            identities.append(identity)
            seen.add(identity)

    return identities


def _handoff_endpoint_matches(
    expected: str | None,
    observed: str | None,
    identities: list[tuple[str | None, str | None]],
) -> bool:
    if not expected or not observed:
        return False
    if expected == observed:
        return True
    for agent_id, agent_name in identities:
        aliases = {value for value in (agent_id, agent_name) if value}
        if expected in aliases and observed in aliases:
            return True
    return False


def _score_multi_agent_requirements(
    trajectory: AgentTrajectory,
    contract: TaskContract,
) -> tuple[float, dict[str, Any]]:
    identities = _observed_agent_identities(trajectory)
    agent_results: list[dict[str, Any]] = []
    handoff_results: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for requirement in contract.required_agents:
        matched = any(
            _agent_requirement_matches(requirement, agent_id, agent_name)
            for agent_id, agent_name in identities
        )
        agent_results.append({
            "requirement": requirement,
            "observed": matched,
        })

    dynamic_handoffs = [
        handoff
        for handoff in trajectory.observed_handoffs
        if handoff.get("payload_type") != "langgraph.static_edge"
    ]

    for requirement in contract.required_handoffs:
        expected_from = requirement.get("from_executor_id") or requirement.get(
            "from_agent"
        )
        expected_to = requirement.get("to_executor_id") or requirement.get("to_agent")
        matched = any(
            _handoff_endpoint_matches(
                expected_from,
                handoff.get("from_executor_id"),
                identities,
            )
            and _handoff_endpoint_matches(
                expected_to,
                handoff.get("to_executor_id"),
                identities,
            )
            for handoff in dynamic_handoffs
        )
        handoff_results.append({
            "requirement": requirement,
            "observed": matched,
        })

    for requirement in contract.required_agent_tool_expectations:
        matched = False
        for call in trajectory.tool_calls:
            if call.tool_name != requirement.get("tool"):
                continue
            if not _agent_requirement_matches(
                requirement,
                getattr(call, "agent_id", None),
                call.agent_name,
            ):
                continue
            match_args = requirement.get("match_args", {})
            if match_args and not _call_matches_args(call.arguments, match_args):
                continue
            expected_return = requirement.get("expected_return", {})
            if expected_return and not _return_contains_expected(
                _framework_call_return_value(call),
                expected_return,
            ):
                continue
            matched = True
            break

        tool_results.append({
            "requirement": requirement,
            "observed": matched,
        })

    all_results = agent_results + handoff_results + tool_results
    score = (
        sum(1 for item in all_results if item["observed"]) / len(all_results)
        if all_results
        else 1.0
    )
    details = {
        "score": round(score, 4),
        "required_agents": agent_results,
        "required_handoffs": handoff_results,
        "required_agent_tool_expectations": tool_results,
    }
    return round(score, 4), details


def score_framework_observed_run(
    trajectory: AgentTrajectory,
    final_answer: dict[str, Any],
    contract: TaskContract,
    policy=None,
    actor: str | None = None,
    role: str | None = None,
) -> TaskEvalResult:
    cf_answer, cf_correct, cf_pc_results = _score_framework_counterfactual_answer(
        final_answer=final_answer,
        contract=contract,
        trajectory=trajectory,
    )
    cf_trajectory = _score_framework_counterfactual_trajectory(
        trajectory=trajectory,
        contract=contract,
    )
    cf_combined = _combine("COUNTERFACTUAL", cf_answer, cf_trajectory)

    sl_answer, sl_correct, sl_pc_results = _score_framework_silence_answer(
        final_answer=final_answer,
        contract=contract,
        trajectory=trajectory,
    )
    sl_trajectory = _score_framework_silence_trajectory(
        trajectory=trajectory,
        contract=contract,
    )
    sl_combined = _combine("SILENCE", sl_answer, sl_trajectory)

    ps_answer, ps_correct, ps_pc_results = _score_framework_perspective_answer(
        final_answer=final_answer,
        contract=contract,
        policy=policy,
        actor=actor,
        role=role,
        trajectory=trajectory,
    )
    ps_trajectory = _score_framework_perspective_trajectory(
        trajectory=trajectory,
        contract=contract,
        policy=policy,
        role=role,
    )
    ps_combined = _combine("PERSPECTIVE", ps_answer, ps_trajectory)

    multi_agent_score, multi_agent_details = _score_multi_agent_requirements(
        trajectory=trajectory,
        contract=contract,
    )

    base_overall = (cf_combined + sl_combined + ps_combined) / 3.0
    overall = round(base_overall * multi_agent_score, 4)
    merged_preconditions = _merge_precondition_details(
        cf_pc_results,
        sl_pc_results,
        ps_pc_results,
    )

    return TaskEvalResult(
        task_name=contract.name,
        counterfactual_score=cf_combined,
        silence_score=sl_combined,
        perspective_score=ps_combined,
        overall_score=overall,
        answer_correct=(cf_correct and sl_correct and multi_agent_score == 1.0),
        precondition_results=merged_preconditions,
        horizon_violations=trajectory.horizon_violations,
        actor_gate_violations=trajectory.actor_gate_violations,
        subsystem_violations=trajectory.subsystem_violations,
        dead_ends_hit=trajectory.dead_ends_hit,
        dead_ends_recovered=trajectory.dead_ends_recovered,
        tool_call_count=len(trajectory.tool_calls),
        prompt_tokens=trajectory.prompt_tokens,
        completion_tokens=trajectory.completion_tokens,
        budget_exceeded=trajectory.budget_exceeded,
        meta={
            "framework_mode": True,
            "cf_answer": cf_answer,
            "cf_trajectory": cf_trajectory,
            "sl_answer": sl_answer,
            "sl_trajectory": sl_trajectory,
            "ps_answer": ps_answer,
            "ps_trajectory": ps_trajectory,
            "multi_agent": multi_agent_details,
        },
    )


def aggregate_task_results(results: list[TaskEvalResult]) -> dict[str, Any]:
    if not results:
        return {}
    n = len(results)
    counterfactual_scores = [r.counterfactual_score for r in results]
    silence_scores = [r.silence_score for r in results]
    perspective_scores = [r.perspective_score for r in results]
    overall_scores = [r.overall_score for r in results]
    correct = sum(1 for r in results if r.answer_correct)
    total_violations = sum(
        r.horizon_violations + r.actor_gate_violations + r.subsystem_violations
        for r in results
    )
    return {
        "n_tasks": n,
        "counterfactual_score": round(sum(counterfactual_scores) / n, 4),
        "silence_score": round(sum(silence_scores) / n, 4),
        "perspective_score": round(sum(perspective_scores) / n, 4),
        "overall_score": round(sum(overall_scores) / n, 4),
        "accuracy": round(correct / n, 4) if n > 0 else 0.0,
        "total_violations": total_violations,
        "per_task": [r.to_dict() for r in results],
    }


def _combine(track: str, answer_score: float, trajectory_score: float) -> float:
    w = _TRACK_WEIGHTS.get(track, {"answer": 0.5, "trajectory": 0.5})
    return round(w["answer"] * answer_score + w["trajectory"] * trajectory_score, 4)


def _resolve_from_corpus(dotted_path: str, corpus) -> Any:
    if not dotted_path or "." not in dotted_path:
        return None
    artifact_id, field_path = dotted_path.split(".", 1)
    doc = corpus.fetch(artifact_id)
    if doc is None:
        return None
    return _get_nested(doc, field_path)


def _get_nested(d: dict, dotted_path: str) -> Any:
    parts = dotted_path.split(".")
    cur = d
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list):
            try:
                idx = int(p)
                cur = cur[idx] if 0 <= idx < len(cur) else None
            except ValueError:
                return None
        else:
            return None
    return cur


def _values_match(agent_value: Any, gt_value: Any) -> bool:
    if agent_value == gt_value:
        return True
    if str(agent_value).strip().lower() == str(gt_value).strip().lower():
        return True
    return False


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _call_matches_args(arguments: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in arguments:
            return False
        if _normalize_value(arguments.get(key)) != _normalize_value(value):
            return False
    return True


def _return_contains_expected(return_value: Any, expected: dict[str, Any]) -> bool:
    if not expected:
        return True
    if not isinstance(return_value, dict):
        return False
    for key, value in expected.items():
        observed = (
            _get_nested(return_value, key) if "." in key else return_value.get(key)
        )
        if _normalize_value(observed) != _normalize_value(value):
            return False
    return True


def _find_matching_calls(
    trajectory: AgentTrajectory, expectation: ToolExpectation
) -> list[Any]:
    matches = []
    for call in trajectory.tool_calls:
        if call.tool_name != expectation.tool:
            continue
        if expectation.match_args and not _call_matches_args(
            call.arguments, expectation.match_args
        ):
            continue
        matches.append(call)
    return matches


def _answer_preconditions(final_answer: dict[str, Any]) -> list[dict[str, Any]]:
    pcs = final_answer.get("preconditions_verified", [])
    if isinstance(pcs, list):
        return [pc for pc in pcs if isinstance(pc, dict)]
    return []


def _framework_call_return_value(call: Any) -> Any:
    if hasattr(call, "return_value"):
        return call.return_value
    return getattr(call, "observed_return_value", None)


def _framework_call_subsystems(trajectory: AgentTrajectory) -> set[str]:
    subsystems: set[str] = set()
    for call in trajectory.tool_calls:
        for key in ("artifact_type", "subsystem", "resource_type", "collection"):
            value = call.arguments.get(key)
            if isinstance(value, str) and value.strip():
                subsystems.add(value.strip())
        observed_return = _framework_call_return_value(call)
        if isinstance(observed_return, dict):
            value = observed_return.get("subsystem")
            if isinstance(value, str) and value.strip():
                subsystems.add(value.strip())
        if isinstance(observed_return, list):
            for item in observed_return:
                if isinstance(item, dict):
                    value = item.get("subsystem")
                    if isinstance(value, str) and value.strip():
                        subsystems.add(value.strip())
    return subsystems


def _score_framework_counterfactual_answer(
    final_answer: dict[str, Any], contract: TaskContract, trajectory: AgentTrajectory
) -> tuple[float, bool, list[dict[str, Any]]]:
    results = []
    total = 0.0
    preconditions = contract.preconditions
    if not preconditions:
        return 1.0, True, []
    answer_pcs = _answer_preconditions(final_answer)
    call_names = {c.tool_name for c in trajectory.tool_calls}
    for pc in preconditions:
        answer_pc = next(
            (apc for apc in answer_pcs if apc.get("check") == pc.check), None
        )
        evidence_supported = True
        reasons = []
        if pc.required_tool and pc.required_tool not in call_names:
            evidence_supported = False
            reasons.append("required_tool_not_observed")
        if pc.expected_field:
            if answer_pc is None:
                evidence_supported = False
                reasons.append("precondition_not_in_answer")
            else:
                facts_found = answer_pc.get("facts_found", {}) or {}
                if pc.expected_field not in facts_found:
                    evidence_supported = False
                    reasons.append("expected_field_missing_from_facts_found")
        if answer_pc is None:
            evidence_supported = False
            reasons.append("precondition_not_reported")
        score = 1.0 if evidence_supported else 0.0
        total += score
        results.append({
            "check": pc.check,
            "evidence_supported": evidence_supported,
            "score": score,
            "required_tool": pc.required_tool or None,
            "expected_field": pc.expected_field or None,
            "reasons": reasons,
        })
    answer_score = round(total / len(preconditions), 4)
    is_correct = answer_score >= 0.75
    return answer_score, is_correct, results


def _score_framework_counterfactual_trajectory(
    trajectory: AgentTrajectory, contract: TaskContract
) -> float:
    expectations = contract.tool_expectations
    if not expectations:
        return round(1.0 if trajectory.tool_calls else 0.0, 4)
    total = 0.0
    for exp in expectations:
        calls = _find_matching_calls(trajectory, exp)
        if not calls:
            total += 0.0
            continue
        if exp.expected_return:
            any_return_match = any(
                _return_contains_expected(
                    _framework_call_return_value(call), exp.expected_return
                )
                for call in calls
            )
            total += 1.0 if any_return_match else 0.5
        else:
            total += 1.0
    return round(total / len(expectations), 4)


def _score_framework_silence_answer(
    final_answer: dict[str, Any], contract: TaskContract, trajectory: AgentTrajectory
) -> tuple[float, bool, list[dict[str, Any]]]:
    preconditions = contract.preconditions
    if not preconditions:
        return 1.0, True, []
    answer_pcs = _answer_preconditions(final_answer)
    results = []
    total = 0.0
    for pc in preconditions:
        answer_pc = next(
            (apc for apc in answer_pcs if apc.get("check") == pc.check), None
        )
        verified = True
        reasons = []
        if answer_pc is None:
            verified = False
            reasons.append("precondition_omitted_from_answer")
        else:
            facts_found = answer_pc.get("facts_found", {}) or {}
            if pc.required_facts:
                missing = [
                    fact for fact in pc.required_facts if fact not in facts_found
                ]
                if missing:
                    verified = False
                    reasons.append(f"missing_required_facts:{','.join(missing)}")
            elif not facts_found and pc.expected_field:
                verified = False
                reasons.append("no_facts_found_for_expected_field_precondition")
        if pc.required_tool:
            tool_seen = any(
                call.tool_name == pc.required_tool for call in trajectory.tool_calls
            )
            if not tool_seen:
                verified = False
                reasons.append("required_tool_not_called")
        score = 1.0 if verified else 0.0
        total += score
        results.append({
            "check": pc.check,
            "verified": verified,
            "score": score,
            "reasons": reasons,
        })
    answer_score = round(total / len(preconditions), 4)
    is_correct = answer_score >= 0.75
    return answer_score, is_correct, results


def _score_framework_silence_trajectory(
    trajectory: AgentTrajectory, contract: TaskContract
) -> float:
    calls = trajectory.tool_calls
    if not calls:
        return 0.0
    required_tools = {
        pc.required_tool for pc in contract.preconditions if pc.required_tool
    }
    expected_tools = {exp.tool for exp in contract.tool_expectations}
    all_required = required_tools | expected_tools
    if all_required:
        covered = {call.tool_name for call in calls if call.tool_name in all_required}
        coverage = len(covered) / len(all_required)
    else:
        coverage = 1.0 if calls else 0.0
    distinct_tools = len({call.tool_name for call in calls})
    diversity = min(distinct_tools / max(len(all_required), 3), 1.0)
    composite = 0.70 * coverage + 0.30 * diversity
    return round(composite, 4)


def _score_framework_perspective_answer(
    final_answer: dict[str, Any],
    contract: TaskContract,
    policy=None,
    actor: str | None = None,
    role: str | None = None,
    trajectory: AgentTrajectory | None = None,
) -> tuple[float, bool, list[dict[str, Any]]]:
    resolved_role = role or contract.role
    if not policy or not resolved_role:
        return 1.0, True, []
    allowed_subsystems = policy.subsystems_for_role(resolved_role)
    if not allowed_subsystems:
        return 1.0, True, []
    if trajectory is None:
        return (
            0.0,
            False,
            [{"error": "missing_trajectory_for_framework_perspective_scoring"}],
        )
    observed_subsystems = _framework_call_subsystems(trajectory)
    if not observed_subsystems:
        return 1.0, True, []
    violations = sorted(
        sub for sub in observed_subsystems if sub not in allowed_subsystems
    )
    score = round(max(0.0, 1.0 - len(violations) / len(observed_subsystems)), 4)
    return (
        score,
        score >= 1.0,
        [{"checked_subsystems": sorted(observed_subsystems), "violations": violations}],
    )


def _score_framework_perspective_trajectory(
    trajectory: AgentTrajectory,
    contract: TaskContract,
    policy=None,
    role: str | None = None,
) -> float:
    calls = trajectory.tool_calls
    if not calls:
        return 1.0
    n = len(calls)
    horizon_violations = sum(1 for c in calls if c.horizon_violation)
    actor_violations = sum(1 for c in calls if c.actor_gate_violation)
    subsystem_violations = sum(1 for c in calls if c.subsystem_violation)
    base = (
        0.35 * (1.0 - actor_violations / n)
        + 0.35 * (1.0 - subsystem_violations / n)
        + 0.30 * (1.0 - horizon_violations / n)
    )
    resolved_role = role or contract.role
    if policy and resolved_role:
        allowed_subsystems = policy.subsystems_for_role(resolved_role)
        observed_subsystems = _framework_call_subsystems(trajectory)
        if allowed_subsystems and observed_subsystems:
            violations = sum(
                1 for sub in observed_subsystems if sub not in allowed_subsystems
            )
            base = 0.7 * base + 0.3 * (1.0 - violations / len(observed_subsystems))
    return round(max(0.0, base), 4)


def _merge_precondition_details(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            check = item.get("check", "unknown")
            current = merged.setdefault(f"{check}", {"check": check})
            current.update(item)
    return list(merged.values())
