from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from groundeval.scorers import (
    _TRACK_WEIGHTS,
    _combine,
    _get_nested,
    _resolve_from_corpus,
    _values_match,
    aggregate_task_results,
    score_counterfactual_answer,
    score_counterfactual_trajectory,
    score_perspective_answer,
    score_perspective_trajectory,
    score_silence_answer,
    score_silence_trajectory,
    score_task_run,
)


def _make_contract(name="task1", preconditions=None, roles=None, role=None):
    if preconditions is None:
        preconditions = []
    return SimpleNamespace(
        name=name, preconditions=preconditions, roles=roles, role=role
    )


def _make_precondition(check="pc1", ground_truth_field="art1.status"):
    return SimpleNamespace(check=check, ground_truth_field=ground_truth_field)


def _make_tool_call(
    tool_name="search",
    arguments=None,
    result_ids=None,
    horizon_violation=False,
    actor_gate_violation=False,
    subsystem_violation=False,
    returned_empty=False,
):
    if arguments is None:
        arguments = {"artifact_type": "crm"}
    if result_ids is None:
        result_ids = ["art1"]
    return SimpleNamespace(
        tool_name=tool_name,
        arguments=arguments,
        result_ids=result_ids,
        horizon_violation=horizon_violation,
        actor_gate_violation=actor_gate_violation,
        subsystem_violation=subsystem_violation,
        returned_empty=returned_empty,
    )


def _make_trajectory(
    tool_calls=None,
    final_answer=None,
    dead_ends_hit=0,
    dead_ends_recovered=0,
    horizon_violations=0,
    actor_gate_violations=0,
    subsystem_violations=0,
    prompt_tokens=100,
    completion_tokens=50,
    budget_exceeded=False,
):
    if tool_calls is None:
        tool_calls = []
    return SimpleNamespace(
        tool_calls=tool_calls,
        final_answer=final_answer or {},
        dead_ends_hit=dead_ends_hit,
        dead_ends_recovered=dead_ends_recovered,
        horizon_violations=horizon_violations,
        actor_gate_violations=actor_gate_violations,
        subsystem_violations=subsystem_violations,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        budget_exceeded=budget_exceeded,
    )


def _make_corpus(fetch_map=None, subsystem_map=None):
    corpus = MagicMock()
    corpus.fetch = MagicMock(
        side_effect=lambda eid: fetch_map.get(eid) if fetch_map else None
    )
    corpus.subsystem_of = MagicMock(
        side_effect=lambda eid: subsystem_map.get(eid) if subsystem_map else None
    )
    return corpus


class TestValuesMatch:
    def test_exact_match(self):
        assert _values_match("active", "active") is True

    def test_exact_match_int(self):
        assert _values_match(42, 42) is True

    def test_string_int_coercion(self):
        assert _values_match("42", 42) is True

    def test_case_insensitive(self):
        assert _values_match("Active", "active") is True

    def test_whitespace_insensitive(self):
        assert _values_match("  active  ", "active") is True

    def test_mismatch(self):
        assert _values_match("active", "inactive") is False

    def test_none_equals_none(self):
        assert _values_match(None, None) is True

    def test_none_does_not_equal_string(self):
        assert _values_match(None, "active") is False

    def test_bool_int_coercion(self):
        assert _values_match("True", True) is True

    def test_bool_int_coercion_false(self):
        assert _values_match("False", False) is True


class TestGetNested:
    def test_simple_key(self):
        assert _get_nested({"a": 1}, "a") == 1

    def test_dotted_path(self):
        assert _get_nested({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_missing_key(self):
        assert _get_nested({"a": 1}, "z") is None

    def test_missing_nested_key(self):
        assert _get_nested({"a": {"b": 2}}, "a.z") is None

    def test_list_index(self):
        assert _get_nested({"a": [10, 20, 30]}, "a.1") == 20

    def test_list_index_out_of_range(self):
        assert _get_nested({"a": [10]}, "a.5") is None

    def test_list_index_non_numeric(self):
        assert _get_nested({"a": [10]}, "a.abc") is None

    def test_midpath_non_dict_non_list(self):
        assert _get_nested({"a": "string"}, "a.b") is None

    def test_empty_path(self):
        assert _get_nested({"a": 1}, "") is None

    def test_empty_dict(self):
        assert _get_nested({}, "a") is None

    def test_deeply_nested(self):
        doc = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
        assert _get_nested(doc, "l1.l2.l3.l4.l5") == "deep"


class TestResolveFromCorpus:
    def test_resolves_artifact_and_field(self):
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        assert _resolve_from_corpus("art1.status", corpus) == "active"

    def test_no_dot_returns_none(self):
        corpus = _make_corpus()
        assert _resolve_from_corpus("art1", corpus) is None

    def test_artifact_not_in_corpus(self):
        corpus = _make_corpus(fetch_map={})
        assert _resolve_from_corpus("art1.status", corpus) is None

    def test_none_corpus(self):
        with pytest.raises(AttributeError):
            _resolve_from_corpus("art1.status", None)

    def test_empty_string(self):
        corpus = _make_corpus()
        assert _resolve_from_corpus("", corpus) is None


class TestCombine:
    def test_counterfactual_weights(self):
        result = _combine("COUNTERFACTUAL", 0.8, 0.6)
        expected = round(0.50 * 0.8 + 0.50 * 0.6, 4)
        assert result == expected

    def test_silence_weights(self):
        result = _combine("SILENCE", 0.8, 0.6)
        expected = round(0.30 * 0.8 + 0.70 * 0.6, 4)
        assert result == expected

    def test_perspective_weights(self):
        result = _combine("PERSPECTIVE", 0.8, 0.6)
        expected = round(0.40 * 0.8 + 0.60 * 0.6, 4)
        assert result == expected

    def test_unknown_track_defaults_to_fifty_fifty(self):
        result = _combine("UNKNOWN", 0.8, 0.6)
        expected = round(0.5 * 0.8 + 0.5 * 0.6, 4)
        assert result == expected

    def test_track_weights_structure(self):
        assert "COUNTERFACTUAL" in _TRACK_WEIGHTS
        assert "SILENCE" in _TRACK_WEIGHTS
        assert "PERSPECTIVE" in _TRACK_WEIGHTS
        for track in _TRACK_WEIGHTS:
            w = _TRACK_WEIGHTS[track]
            assert abs(w["answer"] + w["trajectory"] - 1.0) < 1e-9


class TestCounterfactualAnswer:
    def test_empty_answer_returns_zero(self):
        score, correct, details = score_counterfactual_answer({}, _make_contract())
        assert score == 0.0
        assert correct is False
        assert details == []

    def test_no_preconditions_in_answer(self):
        answer = {"preconditions_verified": []}
        score, correct, details = score_counterfactual_answer(answer, _make_contract())
        assert score == 0.0

    def test_precondition_not_in_contract(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": [],
                }
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc2")])
        score, correct, details = score_counterfactual_answer(answer, contract)
        assert score == 0.0
        assert details[0]["error"] == "precondition not checked by agent"

    def test_no_evidence_cited(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": [],
                }
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        score, correct, details = score_counterfactual_answer(answer, contract)
        assert score == 0.0
        assert details[0]["error"] == "no evidence artifacts cited"

    def test_evidence_supported_agent_passed(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                }
            ]
        }
        contract = _make_contract(
            preconditions=[_make_precondition("pc1", "art1.status")]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 1.0
        assert correct is True

    def test_evidence_mismatch_agent_passed(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                }
            ]
        }
        contract = _make_contract(
            preconditions=[_make_precondition("pc1", "art1.status")]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "inactive"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 0.0

    def test_evidence_supported_agent_failed(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": False,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                }
            ]
        }
        contract = _make_contract(
            preconditions=[_make_precondition("pc1", "art1.status")]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 1.0

    def test_fact_key_missing_from_facts(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"other": "value"},
                    "evidence_artifacts": ["art1"],
                }
            ]
        }
        contract = _make_contract(
            preconditions=[_make_precondition("pc1", "art1.status")]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 0.0

    def test_cited_artifact_not_in_corpus(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["missing_art"],
                }
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        corpus = _make_corpus(fetch_map={})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 0.0

    def test_no_ground_truth_field_no_corpus(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": ["art1"],
                }
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc1", "")])
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 0.0
        assert (
            details[0]["error"]
            == "no ground_truth_field or required_facts for evaluation"
        )

    def test_multiple_preconditions_partial_pass(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                },
                {
                    "check": "pc2",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": [],
                },
            ]
        }
        contract = _make_contract(
            preconditions=[
                _make_precondition("pc1", "art1.status"),
                _make_precondition("pc2", "art2.status"),
            ]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 0.5

    def test_threshold_75_percent(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": f"pc{i}",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                }
                for i in range(4)
            ]
        }
        contract = _make_contract(
            preconditions=[
                _make_precondition(f"pc{i}", "art1.status") for i in range(4)
            ]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 1.0
        assert correct is True

    def test_mismatched_agent_pc_found_during_match_loop(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {},
                    "evidence_artifacts": [],
                },
                {
                    "check": "pc2",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                },
            ]
        }
        contract = _make_contract(
            preconditions=[
                _make_precondition("pc2", "art1.status"),
            ]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 1.0


class TestCounterfactualTrajectory:
    def test_empty_calls_returns_zero(self):
        traj = _make_trajectory(tool_calls=[])
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert score == 0.0

    def test_all_citations_retrieved(self):
        calls = [_make_tool_call(result_ids=["art1", "art2"])]
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "evidence_artifacts": ["art1", "art2"],
                    "passed": True,
                    "facts_found": {},
                }
            ]
        }
        traj = _make_trajectory(tool_calls=calls, final_answer=answer)
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert score > 0.8

    def test_hallucinated_citation(self):
        calls = [_make_tool_call(result_ids=["art1"])]
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "evidence_artifacts": ["art1", "art2"],
                    "passed": True,
                    "facts_found": {},
                }
            ]
        }
        traj = _make_trajectory(tool_calls=calls, final_answer=answer)
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert score < 0.85

    def test_no_citations_at_all(self):
        calls = [_make_tool_call(result_ids=["art1"])]
        traj = _make_trajectory(
            tool_calls=calls, final_answer={"preconditions_verified": []}
        )
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert score < 0.7

    def test_horizon_violations_penalized(self):
        calls = [
            _make_tool_call(result_ids=["art1"]),
            _make_tool_call(result_ids=["art2"], horizon_violation=True),
        ]
        traj = _make_trajectory(tool_calls=calls, final_answer={})
        score1 = score_counterfactual_trajectory(traj, _make_contract())
        calls_clean = [
            _make_tool_call(result_ids=["art1"]),
            _make_tool_call(result_ids=["art2"]),
        ]
        traj_clean = _make_trajectory(tool_calls=calls_clean, final_answer={})
        score2 = score_counterfactual_trajectory(traj_clean, _make_contract())
        assert score1 < score2

    def test_dead_end_recovery_improves_score(self):
        calls = [_make_tool_call(result_ids=["art1"], returned_empty=True)]
        traj_bad = _make_trajectory(tool_calls=calls, dead_ends_recovered=0)
        traj_good = _make_trajectory(tool_calls=calls, dead_ends_recovered=1)
        score_bad = score_counterfactual_trajectory(traj_bad, _make_contract())
        score_good = score_counterfactual_trajectory(traj_good, _make_contract())
        assert score_good > score_bad

    def test_no_dead_ends_gives_full_dead_end_score(self):
        calls = [_make_tool_call(result_ids=["art1"], returned_empty=False)]
        traj = _make_trajectory(tool_calls=calls)
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert score >= 0.6

    def test_evidence_artifacts_at_top_level(self):
        calls = [_make_tool_call(result_ids=["art1", "art2"])]
        answer = {"evidence_artifacts": ["art1", "art2"]}
        traj = _make_trajectory(tool_calls=calls, final_answer=answer)
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert score > 0.8


class TestSilenceAnswer:
    def test_empty_answer(self):
        score, correct, details = score_silence_answer({}, _make_contract())
        assert score == 0.0
        assert correct is False

    def test_no_preconditions_in_contract(self):
        score, correct, details = score_silence_answer(
            {"preconditions_verified": []}, _make_contract(preconditions=[])
        )
        assert score == 1.0
        assert correct is True

    def test_all_preconditions_checked_with_facts(self):
        answer = {
            "preconditions_verified": [
                {"check": "pc1", "facts_found": {"status": "active"}, "passed": True}
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        score, correct, details = score_silence_answer(answer, contract)
        assert score == 1.0
        assert correct is True

    def test_precondition_omitted(self):
        answer = {"preconditions_verified": []}
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        score, correct, details = score_silence_answer(answer, contract)
        assert score == 0.0
        assert details[0]["error"] == "precondition omitted from answer"

    def test_facts_found_empty(self):
        answer = {
            "preconditions_verified": [
                {"check": "pc1", "facts_found": {}, "passed": True}
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        score, correct, details = score_silence_answer(answer, contract)
        assert score == 0.0
        assert details[0]["error"] == "no facts found for precondition"

    def test_mixed_preconditions(self):
        answer = {
            "preconditions_verified": [
                {"check": "pc1", "facts_found": {"a": 1}, "passed": True},
                {"check": "pc2", "facts_found": {}, "passed": True},
            ]
        }
        contract = _make_contract(
            preconditions=[
                _make_precondition("pc1"),
                _make_precondition("pc2"),
            ]
        )
        score, correct, details = score_silence_answer(answer, contract)
        assert score == 0.5

    def test_threshold_75_percent(self):
        answer = {
            "preconditions_verified": [
                {"check": "pc1", "facts_found": {"a": 1}, "passed": True},
                {"check": "pc2", "facts_found": {"a": 1}, "passed": True},
                {"check": "pc3", "facts_found": {"a": 1}, "passed": True},
                {"check": "pc4", "facts_found": {}, "passed": True},
            ]
        }
        contract = _make_contract(
            preconditions=[
                _make_precondition("pc1"),
                _make_precondition("pc2"),
                _make_precondition("pc3"),
                _make_precondition("pc4"),
            ]
        )
        score, correct, details = score_silence_answer(answer, contract)
        assert score == 0.75
        assert correct is True

    def test_answer_is_none(self):
        score, correct, details = score_silence_answer(None, _make_contract())
        assert score == 0.0


class TestSilenceTrajectory:
    def test_empty_trajectory(self):
        traj = _make_trajectory(tool_calls=[])
        score = score_silence_trajectory(traj, _make_contract())
        assert score == 0.0

    def test_perfect_diligence(self):
        calls = [
            _make_tool_call(arguments={"artifact_type": "crm"}, result_ids=["art1"]),
            _make_tool_call(
                arguments={"artifact_type": "billing"}, result_ids=["art2"]
            ),
            _make_tool_call(
                arguments={"artifact_type": "support"}, result_ids=["art3"]
            ),
        ]
        traj = _make_trajectory(tool_calls=calls)
        contract = _make_contract(
            roles={"agent": {"subsystems": ["crm", "billing", "support"]}},
            role="agent",
        )
        score = score_silence_trajectory(traj, contract)
        assert score >= 0.85

    def test_no_diversity_penalized(self):
        calls = [
            _make_tool_call(arguments={"artifact_type": "crm"}, result_ids=["art1"]),
            _make_tool_call(arguments={"artifact_type": "crm"}, result_ids=["art2"]),
        ]
        traj = _make_trajectory(tool_calls=calls)
        contract = _make_contract(
            roles={"agent": {"subsystems": ["crm", "email", "slack"]}},
            role="agent",
        )
        score = score_silence_trajectory(traj, contract)
        assert score < 0.85

    def test_few_calls_penalized(self):
        calls = [_make_tool_call()]
        traj = _make_trajectory(tool_calls=calls)
        contract = _make_contract(
            roles={"agent": {"subsystems": ["crm", "email", "slack"]}},
            role="agent",
        )
        score = score_silence_trajectory(traj, contract)
        assert score < 0.75

    def test_dead_ends_without_recovery(self):
        calls = [
            _make_tool_call(returned_empty=True),
            _make_tool_call(returned_empty=True),
            _make_tool_call(),
        ]
        traj = _make_trajectory(tool_calls=calls, dead_ends_recovered=0)
        contract = _make_contract(
            roles={"agent": {"subsystems": ["crm", "email", "slack"]}},
            role="agent",
        )
        score = score_silence_trajectory(traj, contract)
        assert score < 0.85

    def test_dead_ends_with_recovery(self):
        calls = [
            _make_tool_call(returned_empty=True),
            _make_tool_call(returned_empty=True),
            _make_tool_call(),
        ]
        traj = _make_trajectory(tool_calls=calls, dead_ends_recovered=2)
        contract = _make_contract(
            roles={"agent": {"subsystems": ["crm", "email", "slack"]}},
            role="agent",
        )
        score = score_silence_trajectory(traj, contract)
        assert score >= 0.8

    def test_subsystem_of_corpus_enriches_diversity(self):
        calls = [_make_tool_call(arguments={}, result_ids=["art1", "art2"])]
        corpus = _make_corpus(subsystem_map={"art1": "crm", "art2": "billing"})
        traj = _make_trajectory(tool_calls=calls)
        contract = _make_contract(
            roles={"agent": {"subsystems": ["crm", "billing", "support"]}},
            role="agent",
        )
        score = score_silence_trajectory(traj, contract, corpus)
        assert score > 0.6


class TestPerspectiveAnswer:
    def test_empty_answer(self):
        score, correct, details = score_perspective_answer({}, _make_contract())
        assert score == 0.0

    def test_full_answer(self):
        answer = {
            "preconditions_verified": [{"check": "pc1"}],
            "reasoning": "some reasoning",
        }
        score, correct, details = score_perspective_answer(answer, _make_contract())
        assert score == 1.0
        assert correct is True

    def test_missing_preconditions(self):
        answer = {"reasoning": "some reasoning"}
        score, correct, details = score_perspective_answer(answer, _make_contract())
        assert score == 0.0
        assert correct is False

    def test_missing_reasoning(self):
        answer = {"preconditions_verified": [{"check": "pc1"}]}
        score, correct, details = score_perspective_answer(answer, _make_contract())
        assert score == 0.0
        assert correct is False

    def test_empty_preconditions_list(self):
        answer = {"preconditions_verified": [], "reasoning": "some reasoning"}
        score, correct, details = score_perspective_answer(answer, _make_contract())
        assert score == 0.0


class TestPerspectiveTrajectory:
    def test_empty_trajectory(self):
        traj = _make_trajectory(tool_calls=[])
        score = score_perspective_trajectory(traj, _make_contract())
        assert score == 1.0

    def test_perfect_trajectory(self):
        calls = [
            _make_tool_call(),
            _make_tool_call(),
        ]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score >= 0.85

    def test_horizon_violations(self):
        calls = [
            _make_tool_call(),
            _make_tool_call(horizon_violation=True),
        ]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score < 0.95

    def test_actor_gate_violations(self):
        calls = [
            _make_tool_call(),
            _make_tool_call(actor_gate_violation=True),
        ]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score <= 0.85

    def test_subsystem_violations(self):
        calls = [
            _make_tool_call(),
            _make_tool_call(subsystem_violation=True),
        ]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score <= 0.85

    def test_all_violations(self):
        calls = [
            _make_tool_call(
                horizon_violation=True,
                actor_gate_violation=True,
                subsystem_violation=True,
            ),
        ]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score < 0.5

    def test_dead_end_recovery_helps(self):
        calls = [_make_tool_call(returned_empty=True)]
        traj_bad = _make_trajectory(tool_calls=calls, dead_ends_recovered=0)
        traj_good = _make_trajectory(tool_calls=calls, dead_ends_recovered=1)
        assert score_perspective_trajectory(
            traj_good, _make_contract()
        ) > score_perspective_trajectory(traj_bad, _make_contract())

    def test_few_calls_penalized(self):
        calls = [_make_tool_call()]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score <= 0.95


class TestScoreTaskRun:
    def test_perfect_run(self):
        calls = [
            _make_tool_call(result_ids=["art1"]),
            _make_tool_call(result_ids=["art2"]),
            _make_tool_call(
                result_ids=["art3"], arguments={"artifact_type": "billing"}
            ),
        ]
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["art1"],
                }
            ],
            "reasoning": "thorough analysis",
        }
        traj = _make_trajectory(tool_calls=calls, final_answer=answer)
        contract = _make_contract(
            preconditions=[_make_precondition("pc1", "art1.status")]
        )
        corpus = _make_corpus(fetch_map={"art1": {"status": "active"}})
        result = score_task_run(traj, answer, contract, corpus)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.task_name == "task1"

    def test_empty_run(self):
        traj = _make_trajectory(tool_calls=[])
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        result = score_task_run(traj, {}, contract)
        assert result.overall_score < 0.5

    def test_result_fields_populated(self):
        traj = _make_trajectory(
            tool_calls=[_make_tool_call(horizon_violation=True)],
            dead_ends_hit=2,
            dead_ends_recovered=1,
            horizon_violations=1,
            actor_gate_violations=0,
            subsystem_violations=0,
            prompt_tokens=200,
            completion_tokens=100,
            budget_exceeded=False,
        )
        result = score_task_run(traj, {"preconditions_verified": []}, _make_contract())
        assert result.dead_ends_hit == 2
        assert result.dead_ends_recovered == 1
        assert result.horizon_violations == 1
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100
        assert result.budget_exceeded is False

    def test_meta_keys_present(self):
        traj = _make_trajectory()
        answer = {"preconditions_verified": [], "reasoning": "ok"}
        contract = _make_contract()
        result = score_task_run(traj, answer, contract)
        assert "cf_answer" in result.meta
        assert "sl_answer" in result.meta
        assert "ps_answer" in result.meta
        assert "cf_trajectory" in result.meta
        assert "sl_trajectory" in result.meta
        assert "ps_trajectory" in result.meta


class TestAggregateTaskResults:
    def test_empty_list(self):
        assert aggregate_task_results([]) == {}

    def test_single_result(self):
        from groundeval.core import TaskEvalResult

        r = TaskEvalResult(
            task_name="t1",
            counterfactual_score=0.9,
            silence_score=0.8,
            perspective_score=0.85,
            overall_score=0.85,
            answer_correct=True,
            precondition_results=[],
            horizon_violations=0,
            actor_gate_violations=0,
            subsystem_violations=0,
            dead_ends_hit=0,
            dead_ends_recovered=0,
            tool_call_count=3,
            prompt_tokens=100,
            completion_tokens=50,
            budget_exceeded=False,
            meta={},
        )
        agg = aggregate_task_results([r])
        assert agg["n_tasks"] == 1
        assert agg["accuracy"] == 1.0

    def test_multiple_results(self):
        from groundeval.core import TaskEvalResult

        r1 = TaskEvalResult(
            task_name="t1",
            counterfactual_score=1.0,
            silence_score=1.0,
            perspective_score=1.0,
            overall_score=1.0,
            answer_correct=True,
            precondition_results=[],
            horizon_violations=0,
            actor_gate_violations=0,
            subsystem_violations=0,
            dead_ends_hit=0,
            dead_ends_recovered=0,
            tool_call_count=2,
            prompt_tokens=50,
            completion_tokens=25,
            budget_exceeded=False,
            meta={},
        )
        r2 = TaskEvalResult(
            task_name="t2",
            counterfactual_score=0.0,
            silence_score=0.0,
            perspective_score=0.0,
            overall_score=0.0,
            answer_correct=False,
            precondition_results=[],
            horizon_violations=1,
            actor_gate_violations=1,
            subsystem_violations=1,
            dead_ends_hit=2,
            dead_ends_recovered=0,
            tool_call_count=4,
            prompt_tokens=80,
            completion_tokens=40,
            budget_exceeded=True,
            meta={},
        )
        agg = aggregate_task_results([r1, r2])
        assert agg["n_tasks"] == 2
        assert agg["counterfactual_score"] == 0.5
        assert agg["overall_score"] == 0.5
        assert agg["accuracy"] == 0.5
        assert agg["total_violations"] == 3
        assert len(agg["per_task"]) == 2

    def test_all_correct(self):
        from groundeval.core import TaskEvalResult

        results = []
        for i in range(4):
            results.append(
                TaskEvalResult(
                    task_name=f"t{i}",
                    counterfactual_score=1.0,
                    silence_score=1.0,
                    perspective_score=1.0,
                    overall_score=1.0,
                    answer_correct=True,
                    precondition_results=[],
                    horizon_violations=0,
                    actor_gate_violations=0,
                    subsystem_violations=0,
                    dead_ends_hit=0,
                    dead_ends_recovered=0,
                    tool_call_count=1,
                    prompt_tokens=10,
                    completion_tokens=5,
                    budget_exceeded=False,
                    meta={},
                )
            )
        agg = aggregate_task_results(results)
        assert agg["accuracy"] == 1.0
        assert agg["counterfactual_score"] == 1.0


class TestEdgeCases:
    def test_counterfactual_answer_with_none(self):
        score, correct, details = score_counterfactual_answer(None, _make_contract())
        assert score == 0.0

    def test_silence_answer_with_none(self):
        score, correct, details = score_silence_answer(None, _make_contract())
        assert score == 0.0

    def test_perspective_answer_with_none(self):
        score, correct, details = score_perspective_answer(None, _make_contract())
        assert score == 0.0

    def test_counterfactual_no_preconditions_in_contract(self):
        answer = {"preconditions_verified": [{"check": "pc1"}]}
        contract = _make_contract(preconditions=[])
        score, correct, details = score_counterfactual_answer(answer, contract)
        assert score == 0.0

    def test_silence_trajectory_with_none_corpus(self):
        calls = [_make_tool_call(arguments={}, result_ids=["art1"])]
        traj = _make_trajectory(tool_calls=calls)
        score = score_silence_trajectory(traj, _make_contract(), corpus=None)
        assert 0.0 <= score <= 1.0

    def test_get_nested_starts_with_list(self):
        assert _get_nested([{"a": 1}, {"a": 2}], "1.a") == 2

    def test_get_nested_list_of_non_dicts(self):
        assert _get_nested({"items": ["a", "b", "c"]}, "items.1") == "b"

    def test_values_match_both_none_case(self):
        assert _values_match(None, None) is True

    def test_values_match_edge_whitespace(self):
        assert _values_match("", "") is True

    def test_resolve_from_corpus_dotted_path_deeply_nested(self):
        corpus = _make_corpus(fetch_map={"art1": {"a": {"b": {"c": "deep"}}}})
        assert _resolve_from_corpus("art1.a.b.c", corpus) == "deep"

    def test_resolve_from_corpus_field_not_found(self):
        corpus = _make_corpus(fetch_map={"art1": {"status": "ok"}})
        assert _resolve_from_corpus("art1.missing.deeply", corpus) is None

    def test_counterfactual_trajectory_dead_ends_zero(self):
        calls = [_make_tool_call(returned_empty=False)]
        traj = _make_trajectory(tool_calls=calls)
        score = score_counterfactual_trajectory(traj, _make_contract())
        assert 0.6 <= score <= 1.0

    def test_silence_trajectory_no_artifact_type_in_args(self):
        calls = [
            _make_tool_call(arguments={}, result_ids=["art1"]),
            _make_tool_call(arguments={}, result_ids=["art2"]),
        ]
        traj = _make_trajectory(tool_calls=calls)
        score = score_silence_trajectory(traj, _make_contract())
        assert 0.0 <= score <= 1.0

    def test_perspective_trajectory_single_call_penalty(self):
        calls = [_make_tool_call()]
        traj = _make_trajectory(tool_calls=calls)
        score = score_perspective_trajectory(traj, _make_contract())
        assert score == 0.95

    def test_counterfactual_answer_cited_artifact_not_in_corpus_with_gt_field(self):
        answer = {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["nonexistent"],
                }
            ]
        }
        contract = _make_contract(
            preconditions=[_make_precondition("pc1", "art1.status")]
        )
        corpus = _make_corpus(fetch_map={})
        score, correct, details = score_counterfactual_answer(answer, contract, corpus)
        assert score == 0.0
        assert details[0]["evidence_supported"] is False

    def test_facts_found_truthy_empty(self):
        answer = {
            "preconditions_verified": [
                {"check": "pc1", "facts_found": {"status": ""}, "passed": True}
            ]
        }
        contract = _make_contract(preconditions=[_make_precondition("pc1")])
        score, correct, details = score_silence_answer(answer, contract)
        assert score == 1.0
