import pytest

from groundeval.core import AgentTrajectory, EvalQuestion, ToolCall
from groundeval.scorers import (
    PerspectiveScorer,
    CounterfactualScorer,
    SilenceScorer,
    combine_scores,
    aggregate,
)


def test_perspective_answer_correct():
    scorer = PerspectiveScorer()
    gt = {
        "could_actor_have_known": True,
        "blocked_subsystems": [],
        "evidence_artifacts": ["A1"],
    }
    ans = {
        "could_actor_have_known": True,
        "blocked_subsystems": [],
        "evidence_artifacts": ["A1"],
    }
    score, correct = scorer.score_answer(ans, gt)
    assert correct is True
    assert score == pytest.approx(1.0)


def test_perspective_answer_wrong_boolean():
    scorer = PerspectiveScorer()
    gt = {"could_actor_have_known": True}
    ans = {"could_actor_have_known": False}
    score, correct = scorer.score_answer(ans, gt)
    assert correct is False
    assert score == 0.0


def test_perspective_trajectory_with_violations():
    scorer = PerspectiveScorer()
    traj = AgentTrajectory(
        question_id="q1",
        question_type="PERSPECTIVE",
        tool_calls=[
            ToolCall(
                tool_name="fetch",
                arguments={},
                result_ids=[],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=True,
                subsystem_violation=False,
                returned_empty=True,
                latency_ms=1.0,
            ),
            ToolCall(
                tool_name="fetch",
                arguments={},
                result_ids=["A1"],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=False,
                returned_empty=False,
                latency_ms=1.0,
            ),
        ],
    )
    q = EvalQuestion(
        question_id="q1",
        question_type="PERSPECTIVE",
        question_text="?",
        difficulty="easy",
        ground_truth={"evidence_artifacts": ["A1"]},
        actor_visible_artifacts=["A1"],
    )
    score = scorer.score_trajectory(traj, q)
    assert score < 1.0
    assert score > 0.0


def test_counterfactual_answer_full_match():
    scorer = CounterfactualScorer()
    gt = {
        "outcome_changed": True,
        "causal_mechanism": "esc_to_post",
        "mechanism_aliases": ["esc to post"],
        "cause_event_id": "c1",
        "effect_event_id": "e1",
        "mechanism_direction": "cause_to_effect",
        "evidence_artifacts": ["A1"],
        "actors": ["alice"],
    }
    ans = {
        "outcome_changed": True,
        "causal_mechanism": "esc_to_post",
        "cause_event_id": "c1",
        "effect_event_id": "e1",
        "mechanism_direction": "cause_to_effect",
        "evidence_artifacts": ["A1"],
        "actors": ["alice"],
    }
    score, correct = scorer.score_answer(ans, gt)
    assert score == pytest.approx(1.0)
    assert correct is True


def test_counterfactual_mechanism_by_alias():
    scorer = CounterfactualScorer()
    gt = {
        "outcome_changed": False,
        "causal_mechanism": "esc_to_post",
        "mechanism_aliases": ["escalation follow-up"],
        "cause_event_id": "c1",
        "effect_event_id": "e1",
        "mechanism_direction": "cause_to_effect",
        "evidence_artifacts": [],
        "actors": [],
    }
    ans = {
        "outcome_changed": False,
        "causal_mechanism": "escalation follow-up",
        "cause_event_id": "c1",
        "effect_event_id": "e1",
        "mechanism_direction": "cause_to_effect",
        "evidence_artifacts": [],
        "actors": [],
    }
    score, correct = scorer.score_answer(ans, gt)
    assert score >= 0.95
    assert correct is True


def test_counterfactual_trajectory_evidence_coverage():
    scorer = CounterfactualScorer()
    traj = AgentTrajectory(
        question_id="q1",
        question_type="COUNTERFACTUAL",
        tool_calls=[
            ToolCall(
                tool_name="fetch",
                arguments={},
                result_ids=["A1", "A2"],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=False,
                returned_empty=False,
                latency_ms=1.0,
            )
        ],
    )
    q = EvalQuestion(
        question_id="q1",
        question_type="COUNTERFACTUAL",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        causal_link={
            "evidence_artifact_ids": ["A1", "A2", "A3"],
        },
    )
    score = scorer.score_trajectory(traj, q)
    assert score == pytest.approx(0.7833, abs=1e-3)


def test_silence_answer_correct_exists_true():
    scorer = SilenceScorer()
    gt = {"exists": True}
    ans = {"exists": True}
    score, correct = scorer.score_answer(ans, gt)
    assert score == 1.0
    assert correct is True


def test_silence_trajectory_search_coverage():
    scorer = SilenceScorer()
    traj = AgentTrajectory(
        question_id="q1",
        question_type="SILENCE",
        tool_calls=[
            ToolCall(
                tool_name="search",
                arguments={},
                result_ids=["S1", "S2"],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=False,
                returned_empty=False,
                latency_ms=1.0,
            )
        ],
        final_answer={"exists": False},
    )
    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={"exists": False},
        expected_search_space=["S1", "S2", "S3"],
    )
    score = scorer.score_trajectory(traj, q)
    assert score == pytest.approx(0.7333, abs=1e-3)


def test_combine_scores_weights():
    s = combine_scores("PERSPECTIVE", 1.0, 0.0)
    assert s == pytest.approx(0.40)
    s = combine_scores("SILENCE", 0.0, 1.0)
    assert s == pytest.approx(0.70)
    s = combine_scores("UNKNOWN", 1.0, 1.0)
    assert s == 1.0


def test_aggregate_groups_by_type():
    from groundeval.core import EvalResult

    results = [
        EvalResult("q1", "PERSPECTIVE", "easy", 1.0, True, 1.0, 1.0, None, 2),
        EvalResult("q2", "PERSPECTIVE", "easy", 0.0, False, 0.0, 0.0, None, 2),
        EvalResult("q3", "SILENCE", "easy", 1.0, True, 1.0, 1.0, None, 5),
    ]
    summary = aggregate(results)
    assert summary["overall"]["n"] == 3
    assert summary["by_type"]["PERSPECTIVE"]["n"] == 2
    assert summary["by_type"]["SILENCE"]["n"] == 1


def test_perspective_empty_answer_zero_score():
    scorer = PerspectiveScorer()
    score, ok = scorer.score_answer({}, {"could_actor_have_known": True})
    assert score == 0.0 and ok is False


def test_perspective_fabricated_blocked_on_positive_does_not_get_bonus():
    scorer = PerspectiveScorer()
    gt = {
        "could_actor_have_known": True,
        "blocked_subsystems": [],
        "evidence_artifacts": [],
    }
    ans = {
        "could_actor_have_known": True,
        "blocked_subsystems": ["fake"],
        "evidence_artifacts": [],
    }
    score, ok = scorer.score_answer(ans, gt)
    assert ok is True
    assert score == pytest.approx(0.8)


def test_counterfactual_partial_ids_low_score():
    scorer = CounterfactualScorer()
    gt = {
        "outcome_changed": True,
        "causal_mechanism": "m",
        "cause_event_id": "c1",
        "effect_event_id": "e1",
        "mechanism_direction": "cause_to_effect",
        "evidence_artifacts": [],
        "actors": [],
    }
    ans = {"outcome_changed": True, "causal_mechanism": "m"}
    score, ok = scorer.score_answer(ans, gt)
    assert score == pytest.approx(0.5)
    assert ok is False


def test_counterfactual_trajectory_no_calls():
    scorer = CounterfactualScorer()
    traj = AgentTrajectory(
        question_id="q1", question_type="COUNTERFACTUAL", tool_calls=[]
    )
    q = EvalQuestion(
        question_id="q1",
        question_type="COUNTERFACTUAL",
        question_text="?",
        difficulty="easy",
        ground_truth={},
        causal_link={"evidence_artifact_ids": ["A1"]},
    )
    score = scorer.score_trajectory(traj, q)
    assert score == pytest.approx(0.15, abs=1e-4)


def test_silence_trajectory_no_search_space_full_credit():
    scorer = SilenceScorer()
    traj = AgentTrajectory(question_id="q1", question_type="SILENCE", tool_calls=[])
    q = EvalQuestion(
        question_id="q1",
        question_type="SILENCE",
        question_text="?",
        difficulty="easy",
        ground_truth={"exists": False},
        expected_search_space=None,
    )
    assert scorer.score_trajectory(traj, q) == pytest.approx(1.0, abs=1e-4)


def test_aggregate_empty_results():
    assert aggregate([]) == {}


def test_combine_scores_unknown_type_defaults_to_fifty_fifty():
    assert combine_scores("NONEXISTENT", 1.0, 0.0) == 0.5
    assert combine_scores("NONEXISTENT", 0.0, 1.0) == 0.5
