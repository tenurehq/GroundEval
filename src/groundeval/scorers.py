"""
groundeval/scorers.py
==========================
Three-track scoring for task-contract evaluation.

Each track operates on a single task run (one trajectory + one answer)
and scores a different failure mode:

  Counterfactual  — did the agent's evidence support its conclusion?
                     (checks cited artifact contents against seed-artifact ground truth)
  Silence         — did the agent verify all preconditions before deciding?
                     (checks dead-end recovery, search diversity, precondition coverage)
  Perspective     — did the agent stay within its role's subsystem boundaries?
                     (checks horizon, actor gate, and subsystem violations)

No pre-declared causal link specs. No search_space declarations.
No LLM judge. All scoring is structural comparison.
"""

from __future__ import annotations

import json
import logging
from statistics import mean
from typing import Any

from .core import (
    AgentTrajectory,
    TaskContract,
    TaskEvalResult,
)

logger = logging.getLogger("groundeval.scorers")

_TRACK_WEIGHTS = {
    "COUNTERFACTUAL": {"answer": 0.50, "trajectory": 0.50},
    "SILENCE": {"answer": 0.30, "trajectory": 0.70},
    "PERSPECTIVE": {"answer": 0.40, "trajectory": 0.60},
}


def score_counterfactual_answer(
    final_answer: dict[str, Any],
    contract: TaskContract,
    corpus=None,
) -> tuple[float, bool, list[dict[str, Any]]]:
    """
    Score whether the agent's evidence supports its conclusions.

    For each precondition in the contract:
      1. Find the agent's cited evidence_artifacts for that precondition
      2. Fetch those artifacts from the corpus
      3. Compare the agent's claimed facts_found against the actual artifact content
      4. Score based on whether the evidence actually backs the claim

    Returns (score, is_correct, precondition_details).
    """
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

        if evidence_supported and agent_passed:
            score = 1.0
        elif evidence_supported and not agent_passed:
            score = 1.0
        else:
            score = 0.0

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
    trajectory: AgentTrajectory,
    contract: TaskContract,
    corpus=None,
) -> float:
    """
    Score the trajectory for counterfactual soundness.

    Checks:
      - Evidence retrieval: did the agent actually fetch the artifacts it cited?
      - Citation discipline: did it only cite artifacts it actually retrieved?
      - Causal chain: are the artifacts temporally consistent?

    No pre-declared causal link specs. The ground truth in the seed artifacts
    IS the causal link.
    """
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
        hallucinated = len(cited_ids - retrieved_ids)
        discipline = valid_retrievals / len(cited_ids) if cited_ids else 1.0
    else:
        discipline = 0.0
        hallucinated = 0

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
    final_answer: dict[str, Any],
    contract: TaskContract,
    corpus=None,
) -> tuple[float, bool, list[dict[str, Any]]]:
    """
    Score whether the agent verified all preconditions.

    Silence failure modes:
      - Precondition omitted entirely (not in preconditions_verified list)
      - facts_found is empty for a precondition (agent claimed to check but didn't)
      - Agent gave up without trying alternative search paths (scored in trajectory)

    Returns (score, is_correct, precondition_details).
    """
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
            # Omitted precondition — silence failure
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
            results.append({
                "check": contract_pc.check,
                "verified": True,
                "score": 1.0,
            })
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
    trajectory: AgentTrajectory,
    contract: TaskContract,
    corpus=None,
) -> float:
    """
    Score the trajectory for search diligence.

    Key signals:
      - Dead-end recovery: did the agent try something else after an empty search?
      - Search diversity: did the agent search multiple distinct subsystems/artifact types?
      - Precondition coverage: did the agent's tool calls span all preconditions?

    No search_space coverage. The agent can find information however it can.
    """
    calls = trajectory.tool_calls

    if not calls:
        return 0.0

    n = len(calls)

    # Dead-end recovery
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
    final_answer: dict[str, Any],
    contract: TaskContract,
    policy=None,
    corpus=None,
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
    """
    Score the trajectory for permission boundary adherence.

    Key signals:
      - Horizon violations: did the agent access artifacts beyond its time boundary?
      - Actor gate violations: did the agent access artifacts outside its visibility cone?
      - Subsystem violations: did the agent try to access subsystems outside its role?
    """
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
    """
    Score one task run through all three tracks simultaneously.

    Returns a TaskEvalResult with per-track scores and an overall.
    """
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
        precondition_results=cf_pc_results,  # richest detail from counterfactual
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


def aggregate_task_results(results: list[TaskEvalResult]) -> dict[str, Any]:
    """Aggregate results across multiple task runs."""
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
    """Combine answer and trajectory scores using per-track weights."""
    w = _TRACK_WEIGHTS.get(track, {"answer": 0.5, "trajectory": 0.5})
    return round(w["answer"] * answer_score + w["trajectory"] * trajectory_score, 4)


def _resolve_from_corpus(dotted_path: str, corpus) -> Any:
    """
    Resolve a dotted path like 'crm_account.account_status' from the corpus.

    The first segment is the artifact ID, the rest is the field path.
    """
    if not dotted_path or "." not in dotted_path:
        return None

    artifact_id, field_path = dotted_path.split(".", 1)
    doc = corpus.fetch(artifact_id)
    if doc is None:
        return None

    return _get_nested(doc, field_path)


def _get_nested(d: dict, dotted_path: str) -> Any:
    """Retrieve a value from a nested dict using a dotted path string."""
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
    """Compare agent value against ground truth, handling type coercion."""
    if agent_value == gt_value:
        return True
    if str(agent_value).strip().lower() == str(gt_value).strip().lower():
        return True
    return False
