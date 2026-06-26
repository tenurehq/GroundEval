import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from groundeval.core import TaskContract, TaskPrecondition, ANSWER_SCHEMA_TASK
from groundeval.task_eval import (
    build_task_question_text,
    run_task,
    run_all_tasks,
    _TaskEvalQuestion,
)


def test_build_task_question_text_includes_all_preconditions():
    contract = TaskContract(
        name="test",
        task_description="Verify the thing.",
        preconditions=[
            TaskPrecondition(
                check="check_a",
                description="First thing to verify.",
                ground_truth_field="a.status",
            ),
            TaskPrecondition(
                check="check_b",
                description="Second thing to verify.",
                ground_truth_field="b.status",
            ),
        ],
        decision_field="should_act",
    )
    text = build_task_question_text(contract)
    assert "Verify the thing." in text
    assert "check_a: First thing to verify." in text
    assert "check_b: Second thing to verify." in text
    assert "should_act" in text
    assert "Before acting, verify each of the following:" in text


def test_build_task_question_text_no_preconditions():
    contract = TaskContract(
        name="simple",
        task_description="Just do it.",
        preconditions=[],
    )
    text = build_task_question_text(contract)
    assert "Just do it." in text
    # Should still include the structure, just no precondition list


def test_build_task_question_text_uses_decision_field():
    contract = TaskContract(
        name="t",
        task_description="Check.",
        preconditions=[TaskPrecondition(check="c", description="d")],
        decision_field="should_send_email",
    )
    text = build_task_question_text(contract)
    assert "should_send_email" in text


# ── _TaskEvalQuestion ───────────────────────────────────────


def test_task_eval_question_stores_all_fields():
    q = _TaskEvalQuestion(
        question_id="q1",
        question_type="TASK",
        question_text="What to do?",
        difficulty="hard",
        ground_truth={"key": "value"},
        actor="alice",
        actor_role="engineer",
        as_of_time="2026-01-15",
        actor_visible_artifacts=["a1", "a2"],
        actor_subsystem_access=["crm", "email"],
        expected_answer_schema={"type": "object"},
    )
    assert q.question_id == "q1"
    assert q.question_type == "TASK"
    assert q.question_text == "What to do?"
    assert q.difficulty == "hard"
    assert q.ground_truth == {"key": "value"}
    assert q.actor == "alice"
    assert q.actor_role == "engineer"
    assert q.as_of_time == "2026-01-15"
    assert q.actor_visible_artifacts == ["a1", "a2"]
    assert q.actor_subsystem_access == ["crm", "email"]
    assert q.expected_answer_schema == {"type": "object"}


def test_task_eval_question_defaults_none():
    q = _TaskEvalQuestion(
        question_id="q1",
        question_type="TASK",
        question_text="text",
        difficulty="medium",
        ground_truth={},
    )
    assert q.actor is None
    assert q.actor_role is None
    assert q.as_of_time is None
    assert q.actor_visible_artifacts is None
    assert q.actor_subsystem_access is None
    assert q.expected_answer_schema is None


# ── run_task ────────────────────────────────────────────────


def test_run_task_returns_result():
    """run_task calls agent, merges runtime trajectory, and returns scored result."""
    contract = TaskContract(
        name="test_task",
        task_description="Verify customer status.",
        preconditions=[
            TaskPrecondition(
                check="customer_active",
                description="Customer must be active.",
                ground_truth_field="crm.account_status",
            )
        ],
        actor="agent",
        role="sales_rep",
    )

    # Build a minimal in-memory corpus
    seed = {"id": "crm", "subsystem": "crm", "account_status": "active"}
    from groundeval.adapters import InMemoryCorpusAdapter

    corpus = InMemoryCorpusAdapter([seed])

    mock_traj = MagicMock()
    mock_traj.horizon_violations = 0
    mock_traj.actor_gate_violations = 0
    mock_traj.subsystem_violations = 0
    mock_traj.dead_ends_hit = 1
    mock_traj.dead_ends_recovered = 1
    mock_traj.tool_calls = []
    mock_traj.prompt_tokens = 100
    mock_traj.completion_tokens = 50
    mock_traj.budget_exceeded = False
    mock_traj.task_id = "test_task"

    mock_answer = {
        "preconditions_verified": [
            {
                "check": "customer_active",
                "passed": True,
                "facts_found": {"account_status": "active"},
                "evidence_artifacts": ["crm"],
            }
        ],
        "all_preconditions_pass": True,
        "should_act": True,
        "reasoning": "CRM shows active account.",
        "evidence_artifacts": ["crm"],
    }

    def fake_agent(question, context, tools, max_steps, runtime=None):
        return mock_traj, mock_answer

    from groundeval.adapters import YamlAccessPolicy

    policy = YamlAccessPolicy({
        "actors": {"agent": "sales_rep"},
        "roles": {"sales_rep": {"subsystems": ["crm"]}},
    })

    result = run_task(
        contract=contract,
        agent_fn=fake_agent,
        corpus=corpus,
        policy=policy,
        max_steps=5,
    )

    assert result.task_name == "test_task"
    assert 0.0 <= result.counterfactual_score <= 1.0
    assert 0.0 <= result.silence_score <= 1.0
    assert 0.0 <= result.perspective_score <= 1.0
    assert 0.0 <= result.overall_score <= 1.0
    assert result.tool_call_count >= 0


def test_run_task_no_policy_defaults():
    """No policy passed: creates empty YamlAccessPolicy."""
    contract = TaskContract(
        name="t",
        task_description="Do.",
        preconditions=[TaskPrecondition(check="pc", description="d")],
    )
    corpus = MagicMock()
    corpus.list_ids.return_value = []
    corpus.subsystem_of.return_value = None

    def fake_agent(*args, **kwargs):
        traj = MagicMock()
        traj.horizon_violations = 0
        traj.actor_gate_violations = 0
        traj.subsystem_violations = 0
        traj.dead_ends_hit = 0
        traj.dead_ends_recovered = 0
        traj.tool_calls = []
        traj.prompt_tokens = 0
        traj.completion_tokens = 0
        traj.budget_exceeded = False
        traj.task_id = "t"
        return traj, {}

    result = run_task(contract=contract, agent_fn=fake_agent, corpus=corpus)
    assert result is not None
    assert result.task_name == "t"


def test_run_task_uses_first_actor_from_contract():
    """When contract.actor is None but contract.actors has entries, uses first."""
    contract = TaskContract(
        name="multi",
        task_description="Multi-agent task.",
        preconditions=[TaskPrecondition(check="pc", description="d")],
        actors={"agent1": "role_a", "agent2": "role_b"},
    )
    corpus = MagicMock()
    corpus.list_ids.return_value = []
    corpus.subsystem_of.return_value = None

    policy = MagicMock()
    policy.subsystems_for_role.return_value = {"crm"}
    policy.visible_artifacts.return_value = set()

    def fake_agent(*args, **kwargs):
        traj = MagicMock()
        traj.horizon_violations = 0
        traj.actor_gate_violations = 0
        traj.subsystem_violations = 0
        traj.dead_ends_hit = 0
        traj.dead_ends_recovered = 0
        traj.tool_calls = []
        traj.prompt_tokens = 0
        traj.completion_tokens = 0
        traj.budget_exceeded = False
        traj.task_id = "multi"
        return traj, {}

    result = run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=policy
    )
    assert result.task_name == "multi"


def test_run_task_empty_answer():
    """Agent returns empty answer: scoring still works."""
    contract = TaskContract(
        name="empty_answer_test",
        task_description="Test.",
        preconditions=[TaskPrecondition(check="pc", description="d")],
    )
    corpus = MagicMock()
    corpus.list_ids.return_value = []
    corpus.subsystem_of.return_value = None

    def fake_agent(*args, **kwargs):
        traj = MagicMock()
        traj.horizon_violations = 0
        traj.actor_gate_violations = 0
        traj.subsystem_violations = 0
        traj.dead_ends_hit = 0
        traj.dead_ends_recovered = 0
        traj.tool_calls = []
        traj.prompt_tokens = 0
        traj.completion_tokens = 0
        traj.budget_exceeded = False
        traj.task_id = "empty_answer_test"
        return traj, {}

    result = run_task(contract=contract, agent_fn=fake_agent, corpus=corpus)
    assert result is not None
    assert result.counterfactual_score == 0.0
    assert result.silence_score == 0.0
    assert not result.answer_correct


def test_run_all_tasks_empty_contracts():
    """Empty contracts list returns empty results."""
    results = run_all_tasks(
        contracts=[],
        agent_fn=MagicMock(),
        artifacts_dir="/nonexistent",
        max_steps=3,
    )
    assert results == []


def test_run_all_tasks_policy_from_first_contract():
    """Policy derived from first contract's actor/role declarations."""
    with tempfile.TemporaryDirectory() as tmp:
        art_dir = Path(tmp) / "artifacts"
        art_dir.mkdir()
        (art_dir / "a.json").write_text(
            '{"id": "a", "subsystem": "crm", "status": "ok"}'
        )

        contract = TaskContract(
            name="t1",
            task_description="Do.",
            preconditions=[TaskPrecondition(check="pc", description="d")],
            actors={"agent1": "sales_rep"},
            roles={"sales_rep": {"subsystems": ["crm"]}},
        )

        def fake_agent(*args, **kwargs):
            traj = MagicMock()
            traj.horizon_violations = 0
            traj.actor_gate_violations = 0
            traj.subsystem_violations = 0
            traj.dead_ends_hit = 0
            traj.dead_ends_recovered = 0
            traj.tool_calls = []
            traj.prompt_tokens = 0
            traj.completion_tokens = 0
            traj.budget_exceeded = False
            traj.task_id = "t1"
            return traj, {}

        results = run_all_tasks(
            contracts=[contract],
            agent_fn=fake_agent,
            artifacts_dir=str(art_dir),
            max_steps=3,
        )
        assert len(results) == 1
