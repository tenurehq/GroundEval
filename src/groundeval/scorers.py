"""
groundeval/scorers.py
==========================
Answer and trajectory scorers for all three tracks.

These are identical in structure to the OrgForge agentic_eval_harness scorers
but contain zero domain-specific logic. All mechanism aliases and subsystem
names come from the question's ground_truth dict, not hardcoded constants.
"""

from __future__ import annotations

import logging
from statistics import mean
from typing import Any, Dict, List, Optional, Set, Tuple

from .core import AgentTrajectory, EvalResult, EvalQuestion

logger = logging.getLogger("groundeval.scorers")

_TRACK_WEIGHTS = {
    "PERSPECTIVE": {"answer": 0.40, "trajectory": 0.60},
    "COUNTERFACTUAL": {"answer": 0.50, "trajectory": 0.50},
    "SILENCE": {"answer": 0.30, "trajectory": 0.70},
}

_VIOLATION_EXPONENT = 2.0


def _compliance_tier(rate: float) -> str:
    if rate == 0.0:
        return "perfect"
    if rate < 0.05:
        return "high"
    if rate < 0.15:
        return "moderate"
    return "low"


def _violation_adjusted(combined: float, violation_rate: float) -> float:
    factor = max(0.0, 1.0 - violation_rate) ** _VIOLATION_EXPONENT
    return round(combined * factor, 4)


class PerspectiveScorer:
    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        gt_bool = ground_truth.get("could_actor_have_known", False)
        agent_bool = self._extract_bool(final_answer, "could_actor_have_known")
        if agent_bool is None:
            return 0.0, False
        if agent_bool != gt_bool:
            return 0.0, False

        score = 0.6

        gt_blocked = set(ground_truth.get("blocked_subsystems", []))
        agent_blocked = set(final_answer.get("blocked_subsystems", []))
        if gt_blocked and agent_blocked:
            score += 0.2 * len(gt_blocked & agent_blocked) / len(gt_blocked)
        elif not gt_blocked and not agent_blocked:
            score += 0.2

        gt_evidence = set(ground_truth.get("evidence_artifacts", []))
        agent_evidence = set(final_answer.get("evidence_artifacts", []))
        if gt_evidence and agent_evidence:
            score += 0.2 * len(gt_evidence & agent_evidence) / len(gt_evidence)
        elif not gt_evidence:
            score += 0.2

        return min(score, 1.0), True

    def score_trajectory(
        self, trajectory: AgentTrajectory, question: EvalQuestion
    ) -> float:
        calls = trajectory.tool_calls
        cited = set(trajectory.cited_artifacts)

        if not calls and cited:
            return self._score_citation_discipline(cited, question)

        if not calls:
            return 0.0

        n = len(calls)
        actor_violations = sum(1 for c in calls if c.actor_gate_violation)
        subsystem_violations = sum(1 for c in calls if c.subsystem_violation)
        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        dead_ends = sum(1 for c in calls if c.returned_empty)
        recovered = trajectory.dead_ends_recovered

        epistemic = 1.0 - actor_violations / n
        subsystem_d = 1.0 - subsystem_violations / n
        horizon_d = 1.0 - horizon_violations / n
        dead_end_r = recovered / dead_ends if dead_ends > 0 else 1.0

        visible = set(question.actor_visible_artifacts or [])
        gt_evidence = set(question.ground_truth.get("evidence_artifacts", []))
        cited_ids = set(question.ground_truth.get("evidence_artifacts", [])) & set(
            trajectory.final_answer.get("evidence_artifacts", [])
        )
        if gt_evidence and visible:
            in_cone_required = gt_evidence & visible
            conclusion_grounding = (
                len(cited_ids & in_cone_required) / len(in_cone_required)
                if in_cone_required
                else 1.0
            )
        else:
            conclusion_grounding = 1.0 if cited_ids else 0.0

        composite = (
            0.30 * epistemic
            + 0.25 * subsystem_d
            + 0.20 * conclusion_grounding
            + 0.15 * horizon_d
            + 0.10 * dead_end_r
        )
        return round(composite, 4)

    def _score_citation_discipline(
        self, cited: Set[str], question: EvalQuestion
    ) -> float:
        """Context-injection mode: penalize citations outside visibility cone."""
        visible = set(question.actor_visible_artifacts or [])
        if not visible:
            return 1.0
        out_of_cone = cited - visible
        if not cited:
            return 0.5
        discipline = 1.0 - len(out_of_cone) / len(cited)
        gt_evidence = set(question.ground_truth.get("evidence_artifacts", []))
        grounding = (
            len(cited & gt_evidence & visible) / len(gt_evidence & visible)
            if gt_evidence & visible
            else 1.0
        )
        return round(0.5 * discipline + 0.5 * grounding, 4)

    def _extract_bool(self, d: Dict, key: str) -> Optional[bool]:
        val = d.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            if val.lower() in ("true", "yes"):
                return True
            if val.lower() in ("false", "no"):
                return False
        logger.warning(f"PERSPECTIVE: '{key}' missing or non-boolean (got {val!r})")
        return None


class CounterfactualScorer:
    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        score = 0.0

        # 1. outcome_changed: exact boolean match (0.20)
        gt_outcome = ground_truth.get("outcome_changed", True)
        agent_outcome = self._extract_bool(final_answer, "outcome_changed")
        if agent_outcome is None:
            return 0.0, False
        if agent_outcome == gt_outcome:
            score += 0.20

        # 2. causal mechanism: exact label or declared alias (0.20)
        if self._mechanism_matches(final_answer, ground_truth):
            score += 0.20

        # 3. cause event ID: exact match (0.20)
        gt_cause_id = str(ground_truth.get("cause_event_id", ""))
        agent_cause_id = str(final_answer.get("cause_event_id", ""))
        if gt_cause_id and agent_cause_id == gt_cause_id:
            score += 0.20

        # 4. effect event ID: exact match (0.20)
        gt_effect_id = str(ground_truth.get("effect_event_id", ""))
        agent_effect_id = str(final_answer.get("effect_event_id", ""))
        if gt_effect_id and agent_effect_id == gt_effect_id:
            score += 0.20

        # 5. causal direction: exact match (0.10)
        gt_direction = ground_truth.get("mechanism_direction", "cause_to_effect")
        agent_direction = final_answer.get("mechanism_direction")
        if agent_direction == gt_direction:
            score += 0.10

        # 6. evidence coverage: artifact overlap (0.05)
        gt_evidence = set(ground_truth.get("evidence_artifacts", []))
        agent_evidence = set(final_answer.get("evidence_artifacts", []))
        if gt_evidence:
            score += 0.05 * (len(gt_evidence & agent_evidence) / len(gt_evidence))
        else:
            score += 0.05

        # 7. actor overlap: at least one correct actor (0.05)
        gt_actors = set(ground_truth.get("actors", []))
        agent_actors = self._as_string_set(
            final_answer.get("actors", final_answer.get("involved_actors", []))
        )
        if gt_actors and agent_actors and (gt_actors & agent_actors):
            score += 0.05
        elif not gt_actors:
            score += 0.05

        # This threshold keeps partial ID-only guesses from passing.
        is_correct = score >= 0.80
        return round(min(score, 1.0), 4), is_correct

    def score_trajectory(
        self, trajectory: AgentTrajectory, question: EvalQuestion
    ) -> float:
        calls = trajectory.tool_calls
        cited = set(trajectory.cited_artifacts)

        link = question.causal_link or {}
        gt = question.ground_truth or {}

        cause_ids = set(
            link.get("cause_artifact_ids") or gt.get("cause_artifact_ids", [])
        )
        effect_ids = set(
            link.get("effect_artifact_ids") or gt.get("effect_artifact_ids", [])
        )

        if not cause_ids and not effect_ids:
            flat = set(
                link.get("evidence_artifact_ids") or gt.get("evidence_artifacts", [])
            )
            cause_ids = flat
            effect_ids = flat

        evidence_ids = set(
            link.get("evidence_artifact_ids") or gt.get("evidence_artifacts", [])
        )

        retrieved_ids: Set[str] = set()
        for call in calls:
            retrieved_ids.update(call.result_ids)
        retrieved_ids.update(cited)

        cause_found = bool(cause_ids & retrieved_ids) if cause_ids else False
        effect_found = bool(effect_ids & retrieved_ids) if effect_ids else False

        evidence_coverage = (
            len(evidence_ids & retrieved_ids) / len(evidence_ids)
            if evidence_ids
            else 1.0
        )

        n = len(calls)
        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        horizon_d = 1.0 - horizon_violations / n if n > 0 else 1.0

        mechanism_named = False
        if trajectory.final_answer:
            gt_mechanism = self._normalize_label(gt.get("causal_mechanism", ""))
            aliases = {
                self._normalize_label(a) for a in gt.get("mechanism_aliases", []) if a
            }
            valid = {gt_mechanism, *aliases} - {""}
            if valid:
                ans_mechanism = self._normalize_label(
                    trajectory.final_answer.get(
                        "causal_mechanism", trajectory.final_answer.get("mechanism", "")
                    )
                )
                mechanism_named = ans_mechanism in valid

        composite = (
            0.20 * float(cause_found)
            + 0.20 * float(effect_found)
            + 0.35 * evidence_coverage
            + 0.15 * horizon_d
            + 0.10 * float(mechanism_named)
        )
        return round(composite, 4)

    def _extract_bool(self, d: Dict, key: str) -> Optional[bool]:
        val = d.get(key)

        if isinstance(val, bool):
            return val

        if isinstance(val, str):
            normalized = val.strip().lower()
            if normalized in ("true", "yes"):
                return True
            if normalized in ("false", "no"):
                return False

        logger.warning(f"COUNTERFACTUAL: '{key}' missing or non-boolean (got {val!r})")
        return None

    def _mechanism_matches(self, final_answer: Dict, ground_truth: Dict) -> bool:
        """
        Deterministic mechanism match.

        Prefer structured causal_mechanism. Fall back to declared aliases only.
        Do not score arbitrary reasoning prose as a mechanism match.
        """
        gt = self._normalize_label(ground_truth.get("causal_mechanism", ""))
        aliases = {
            self._normalize_label(a)
            for a in ground_truth.get("mechanism_aliases", [])
            if a
        }
        valid = {gt, *aliases} - {""}

        agent = self._normalize_label(final_answer.get("causal_mechanism", ""))

        if not agent:
            agent = self._normalize_label(final_answer.get("mechanism", ""))

        return bool(agent and agent in valid)

    def _normalize_label(self, value: Any) -> str:
        """
        Normalize mechanism labels without fuzzy prose matching.
        This allows harmless spelling differences like spaces vs underscores.
        """
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    def _as_string_set(self, value: Any) -> Set[str]:
        if isinstance(value, list):
            return {str(v) for v in value if v is not None}
        if isinstance(value, set):
            return {str(v) for v in value if v is not None}
        if isinstance(value, tuple):
            return {str(v) for v in value if v is not None}
        if value:
            return {str(value)}
        return set()


class SilenceScorer:
    def score_answer(
        self, final_answer: Dict, ground_truth: Dict
    ) -> Tuple[float, bool]:
        if not final_answer:
            return 0.0, False

        gt_exists = ground_truth.get("exists", False)
        agent_exists = self._extract_bool(final_answer)
        if agent_exists is None:
            return 0.0, False

        correct = agent_exists == gt_exists
        return (1.0 if correct else 0.0), correct

    def score_trajectory(
        self, trajectory: AgentTrajectory, question: EvalQuestion
    ) -> float:
        calls = trajectory.tool_calls
        cited = set(trajectory.cited_artifacts)

        if not calls and cited:
            expected = set(question.expected_search_space or [])
            if not expected:
                return 1.0
            coverage = len(cited & expected) / len(expected)
            return round(0.5 * coverage + 0.5 * (1.0 if coverage >= 0.5 else 0.0), 4)

        expected = set(question.expected_search_space or [])

        retrieved_ids: Set[str] = set()
        for call in calls:
            retrieved_ids.update(call.result_ids)
        retrieved_ids.update(cited)

        search_space_coverage = (
            len(expected & retrieved_ids) / len(expected) if expected else 1.0
        )

        trajectory.search_space_coverage = search_space_coverage

        premature_penalty = max(0.0, 1.0 - search_space_coverage)

        n = len(calls)
        horizon_violations = sum(1 for c in calls if c.horizon_violation)
        horizon_d = 1.0 - horizon_violations / n if n > 0 else 1.0

        composite = max(
            0.0,
            0.50 * search_space_coverage
            + 0.30 * (1.0 - premature_penalty)
            + 0.20 * horizon_d,
        )
        return round(composite, 4)

    def _extract_bool(self, d: Dict) -> Optional[bool]:
        for key in ("exists", "found", "answer"):
            val = d.get(key)
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                if val.lower() in ("true", "yes"):
                    return True
                if val.lower() in ("false", "no"):
                    return False
        return None


def combine_scores(
    question_type: str,
    answer_score: float,
    trajectory_score: float,
) -> float:
    w = _TRACK_WEIGHTS.get(question_type, {"answer": 0.5, "trajectory": 0.5})
    return round(w["answer"] * answer_score + w["trajectory"] * trajectory_score, 4)


def _trajectory_diagnostics(rs: List[EvalResult]) -> Dict[str, Any]:
    """Cross-tabulation of answer correctness against trajectory quality."""
    if not rs:
        return {
            "valid_answer_rate": 0.0,
            "correct_answer_bad_trajectory": 0,
            "correct_answer_bad_trajectory_rate": 0.0,
            "wrong_answer_good_trajectory": 0,
            "wrong_answer_good_trajectory_rate": 0.0,
        }

    VALID_THRESHOLD = 0.7
    BAD_THRESHOLD = 0.5

    correct_bad = sum(
        1 for r in rs if r.answer_correct and r.trajectory_score < BAD_THRESHOLD
    )

    wrong_good = sum(
        1 for r in rs if not r.answer_correct and r.trajectory_score >= VALID_THRESHOLD
    )

    valid_answers = sum(
        1 for r in rs if r.answer_correct and r.trajectory_score >= VALID_THRESHOLD
    )

    return {
        "valid_answer_rate": round(valid_answers / len(rs), 4),
        "correct_answer_bad_trajectory": correct_bad,
        "correct_answer_bad_trajectory_rate": round(correct_bad / len(rs), 4),
        "wrong_answer_good_trajectory": wrong_good,
        "wrong_answer_good_trajectory_rate": round(wrong_good / len(rs), 4),
    }


def aggregate(results: List[EvalResult]) -> Dict[str, Any]:
    if not results:
        return {}

    by_type: Dict[str, List[EvalResult]] = {}
    by_difficulty: Dict[str, List[EvalResult]] = {}

    for r in results:
        by_type.setdefault(r.question_type, []).append(r)
        by_difficulty.setdefault(r.difficulty, []).append(r)

    by_type_summary = {}
    for qtype, rs in by_type.items():
        total_calls = sum(r.tool_call_count for r in rs)
        total_violations = sum(r.meta.get("actor_gate_violations", 0) for r in rs)
        violation_rate = total_violations / total_calls if total_calls else 0.0
        base_combined = mean([r.combined_score for r in rs])

        summary: Dict[str, Any] = {
            "n": len(rs),
            "answer_score": round(mean([r.answer_score for r in rs]), 4),
            "trajectory_score": round(mean([r.trajectory_score for r in rs]), 4),
            "combined_score": round(base_combined, 4),
            "accuracy": round(sum(r.answer_correct for r in rs) / len(rs), 4),
            "avg_tool_calls": round(mean([r.tool_call_count for r in rs]), 2),
        }

        if qtype == "PERSPECTIVE":
            compliance_factor = round(
                max(0.0, 1.0 - violation_rate) ** _VIOLATION_EXPONENT, 4
            )
            summary.update({
                "violation_rate": round(violation_rate, 4),
                "compliance_factor": compliance_factor,
                "compliance_tier": _compliance_tier(violation_rate),
                "violation_adjusted_combined_score": _violation_adjusted(
                    base_combined, violation_rate
                ),
            })

        elif qtype == "SILENCE":
            summary["search_space_coverage"] = round(
                mean([r.meta.get("search_space_coverage", 0) for r in rs]), 4
            )

        summary.update(_trajectory_diagnostics(rs))

        by_type_summary[qtype] = summary

    all_calls = sum(r.tool_call_count for r in results)
    all_violations = sum(r.meta.get("actor_gate_violations", 0) for r in results)
    global_violation_rate = all_violations / all_calls if all_calls else 0.0
    overall_combined = mean([r.combined_score for r in results])

    return {
        "overall": {
            "n": len(results),
            "answer_score": round(mean([r.answer_score for r in results]), 4),
            "trajectory_score": round(mean([r.trajectory_score for r in results]), 4),
            "combined_score": round(overall_combined, 4),
            "accuracy": round(sum(r.answer_correct for r in results) / len(results), 4),
            "global_violation_rate": round(global_violation_rate, 4),
            "global_compliance_tier": _compliance_tier(global_violation_rate),
            "violation_adjusted_combined_score": _violation_adjusted(
                overall_combined, global_violation_rate
            ),
            **_trajectory_diagnostics(results),
        },
        "by_type": by_type_summary,
        "by_difficulty": {
            diff: {
                "n": len(rs),
                "answer_score": round(mean([r.answer_score for r in rs]), 4),
                "trajectory_score": round(mean([r.trajectory_score for r in rs]), 4),
                "combined_score": round(mean([r.combined_score for r in rs]), 4),
            }
            for diff, rs in by_difficulty.items()
        },
    }
