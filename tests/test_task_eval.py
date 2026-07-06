from unittest.mock import MagicMock, patch

from groundeval.adapters import InMemoryCorpusAdapter, YamlAccessPolicy
from groundeval.core import ANSWER_SCHEMA_TASK, TaskContract, TaskPrecondition, ToolCall
from groundeval.task_eval import (
    _TaskEvalQuestion,
    build_task_question_text,
    run_all_tasks,
    run_task,
)


def _fake_traj(task_id="t1"):
    traj = MagicMock()
    traj.horizon_violations = 0
    traj.actor_gate_violations = 0
    traj.subsystem_violations = 0
    traj.dead_ends_hit = 0
    traj.dead_ends_recovered = 0
    traj.tool_calls = []
    traj.prompt_tokens = 1
    traj.completion_tokens = 2
    traj.budget_exceeded = False
    traj.task_id = task_id
    traj.final_answer = {}
    return traj


def test_build_task_question_text_with_multiple_preconditions_and_custom_decision_field():
    contract = TaskContract(
        name="t1",
        task_description="Verify customer before action",
        preconditions=[
            TaskPrecondition(check="pc1", description="First check"),
            TaskPrecondition(check="pc2", description="Second check"),
        ],
        decision_field="should_escalate",
    )
    text = build_task_question_text(contract)
    assert "Verify customer before action" in text
    assert "pc1: First check" in text
    assert "pc2: Second check" in text
    assert "should_escalate" in text


def test_build_task_question_text_with_no_preconditions_still_formats():
    contract = TaskContract(
        name="t1",
        task_description="Do work",
        preconditions=[],
    )
    text = build_task_question_text(contract)
    assert "Do work" in text
    assert "Before acting, verify each of the following:" in text
    assert "Submit your findings" in text


def test_task_eval_question_stores_all_fields():
    q = _TaskEvalQuestion(
        question_id="q1",
        question_type="TASK",
        question_text="Question?",
        difficulty="medium",
        ground_truth={"a": 1},
        actor="alice",
        actor_role="sales",
        as_of_time="2026-01-01T00:00:00",
        actor_visible_artifacts=["a1"],
        actor_subsystem_access=["crm"],
        expected_answer_schema={"type": "object"},
    )
    assert q.question_id == "q1"
    assert q.question_type == "TASK"
    assert q.question_text == "Question?"
    assert q.difficulty == "medium"
    assert q.ground_truth == {"a": 1}
    assert q.actor == "alice"
    assert q.actor_role == "sales"
    assert q.as_of_time == "2026-01-01T00:00:00"
    assert q.actor_visible_artifacts == ["a1"]
    assert q.actor_subsystem_access == ["crm"]
    assert q.expected_answer_schema == {"type": "object"}


def test_run_task_corpus_mode_uses_runtime_and_scores():
    contract = TaskContract.from_dict({
        "name": "t1",
        "task_description": "Verify customer",
        "preconditions": [
            {
                "check": "pc1",
                "description": "status active",
                "ground_truth_field": "a1.status",
                "required_facts": ["status"],
            }
        ],
        "actor": "alice",
        "role": "sales",
    })
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm", "status": "active"}
    ])
    policy = YamlAccessPolicy({
        "actors": {"alice": "sales"},
        "roles": {"sales": {"subsystems": ["crm"]}},
    })

    def fake_agent(question, context, tools, max_steps, runtime=None):
        assert question.expected_answer_schema == ANSWER_SCHEMA_TASK
        assert question.actor == "alice"
        assert question.actor_role == "sales"
        assert runtime is not None
        runtime.fetch("a1")
        traj = _fake_traj("t1")
        return traj, {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["a1"],
                }
            ],
            "all_preconditions_pass": True,
            "reasoning": "ok",
        }

    result = run_task(
        contract=contract,
        agent_fn=fake_agent,
        corpus=corpus,
        policy=policy,
        max_steps=5,
    )
    assert result.task_name == "t1"
    assert result.tool_call_count >= 1
    assert result.counterfactual_score >= 0.0
    assert result.silence_score >= 0.0
    assert result.perspective_score >= 0.0


def test_run_task_fixture_mode_uses_fixture_backend_and_policy_visibility():
    contract = TaskContract.from_dict({
        "name": "t1",
        "task_description": "Verify fixture customer",
        "preconditions": [
            {
                "check": "pc1",
                "description": "status active",
                "ground_truth_field": "crm-1.status",
                "required_facts": ["status"],
            }
        ],
        "actor": "alice",
        "role": "sales",
        "allowed_tools": {
            "fetch_customer": {
                "artifact_id": "crm-1",
                "returns": {"status": "active"},
                "subsystem": "crm",
            }
        },
    })
    corpus = InMemoryCorpusAdapter([])
    policy = YamlAccessPolicy({
        "actors": {"alice": "sales"},
        "roles": {"sales": {"subsystems": ["crm"]}},
    })

    def fake_agent(question, context, tools, max_steps, runtime=None):
        doc = runtime.fetch("crm-1")
        assert doc["status"] == "active"
        assert question.actor == "alice"
        assert question.actor_role == "sales"
        traj = _fake_traj("t1")
        return traj, {
            "preconditions_verified": [
                {
                    "check": "pc1",
                    "passed": True,
                    "facts_found": {"status": "active"},
                    "evidence_artifacts": ["crm-1"],
                }
            ],
            "all_preconditions_pass": True,
            "reasoning": "ok",
        }

    result = run_task(
        contract=contract,
        agent_fn=fake_agent,
        corpus=corpus,
        policy=policy,
        max_steps=5,
    )
    assert result.task_name == "t1"
    assert result.tool_call_count >= 1


def test_run_task_policy_defaults_when_none():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    corpus = InMemoryCorpusAdapter([])

    def fake_agent(question, context, tools, max_steps, runtime=None):
        assert question.actor is None
        assert question.actor_role is None
        traj = _fake_traj("t1")
        return traj, {}

    result = run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert result.task_name == "t1"


def test_run_task_uses_first_contract_actor_when_actor_missing():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
        "actors": {"alice": "sales"},
        "roles": {"sales": {"subsystems": ["crm"]}},
    })
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "crm"}])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        captured["actor"] = question.actor
        captured["role"] = question.actor_role
        traj = _fake_traj("t1")
        return traj, {}

    result = run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert captured["actor"] == "alice"
    assert captured["role"] == "sales"
    assert result.task_name == "t1"


def test_run_task_preserves_explicit_actor_even_if_contract_actors_exist():
    contract = TaskContract.from_dict({
        "name": "t1",
        "actor": "bob",
        "role": "eng",
        "preconditions": [{"check": "pc1"}],
        "actors": {"alice": "sales"},
        "roles": {"sales": {"subsystems": ["crm"]}, "eng": {"subsystems": ["jira"]}},
    })
    corpus = InMemoryCorpusAdapter([])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        captured["actor"] = question.actor
        captured["role"] = question.actor_role
        traj = _fake_traj("t1")
        return traj, {}

    run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert captured["actor"] == "bob"
    assert captured["role"] == "eng"


def test_run_task_runtime_trajectory_replaces_agent_empty_tool_calls():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    corpus = InMemoryCorpusAdapter([{"id": "a1", "subsystem": "crm"}])

    def fake_agent(question, context, tools, max_steps, runtime=None):
        runtime.fetch("a1")
        runtime.fetch("missing")
        traj = _fake_traj("t1")
        traj.horizon_violations = 999
        traj.actor_gate_violations = 999
        traj.subsystem_violations = 999
        traj.dead_ends_hit = 999
        traj.dead_ends_recovered = 999
        traj.tool_calls = []
        return traj, {}

    result = run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert result.tool_call_count == 2
    assert result.dead_ends_hit == 1
    assert result.dead_ends_recovered == 0
    assert result.horizon_violations == 0


def test_run_task_keeps_agent_trajectory_if_runtime_has_no_calls():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    corpus = InMemoryCorpusAdapter([])

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj("t1")
        traj.tool_calls = [
            ToolCall(
                tool_name="fetch_artifact",
                arguments={"artifact_id": "a1"},
                result_ids=["a1"],
                timestamp_applied=None,
                horizon_violation=False,
                actor_gate_violation=False,
                subsystem_violation=False,
                returned_empty=False,
                latency_ms=1.0,
            )
        ]
        traj.horizon_violations = 4
        traj.actor_gate_violations = 3
        traj.subsystem_violations = 2
        traj.dead_ends_hit = 1
        traj.dead_ends_recovered = 0
        return traj, {}

    result = run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert result.tool_call_count == 1
    assert result.horizon_violations == 4
    assert result.actor_gate_violations == 3
    assert result.subsystem_violations == 2


def test_run_task_sets_final_answer_on_trajectory_before_scoring():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    corpus = InMemoryCorpusAdapter([])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj("t1")
        return traj, {"reasoning": "hello"}

    def fake_score_task_run(**kwargs):
        captured["trajectory_final_answer"] = kwargs["trajectory"].final_answer
        result = MagicMock()
        result.task_name = "t1"
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.tool_call_count = 0
        return result

    with patch("groundeval.task_eval.score_task_run", side_effect=fake_score_task_run):
        run_task(
            contract=contract,
            agent_fn=fake_agent,
            corpus=corpus,
            policy=None,
            max_steps=5,
        )

    assert captured["trajectory_final_answer"] == {"reasoning": "hello"}


def test_run_task_fixture_mode_sets_runtime_all_subsystems():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
        "allowed_tools": {
            "fetch_customer": {
                "artifact_id": "crm-1",
                "returns": {"status": "active"},
                "subsystem": "crm",
            },
            "fetch_email": {
                "artifact_id": "email-1",
                "returns": {"subject": "Hi"},
                "subsystem": "email",
            },
        },
    })
    corpus = InMemoryCorpusAdapter([])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        captured["subsystems"] = runtime.all_subsystems
        traj = _fake_traj("t1")
        return traj, {}

    run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert captured["subsystems"] == ["crm", "email"]


def test_run_task_corpus_mode_sets_runtime_all_subsystems():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    corpus = InMemoryCorpusAdapter([
        {"id": "a1", "subsystem": "crm"},
        {"id": "a2", "subsystem": "email"},
        {"id": "a3", "subsystem": "crm"},
    ])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        captured["subsystems"] = runtime.all_subsystems
        traj = _fake_traj("t1")
        return traj, {}

    run_task(
        contract=contract, agent_fn=fake_agent, corpus=corpus, policy=None, max_steps=5
    )
    assert captured["subsystems"] == ["crm", "email"]


def test_run_task_passes_fixture_backend_to_scorer():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
        "allowed_tools": {
            "fetch_customer": {
                "artifact_id": "crm-1",
                "returns": {"status": "active"},
                "subsystem": "crm",
            }
        },
    })
    corpus = InMemoryCorpusAdapter([])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj("t1")
        return traj, {}

    def fake_score_task_run(**kwargs):
        captured["corpus_type"] = type(kwargs["corpus"]).__name__
        result = MagicMock()
        result.task_name = "t1"
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.tool_call_count = 0
        return result

    with patch("groundeval.task_eval.score_task_run", side_effect=fake_score_task_run):
        run_task(
            contract=contract,
            agent_fn=fake_agent,
            corpus=corpus,
            policy=None,
            max_steps=5,
        )

    assert captured["corpus_type"] == "FixtureBackend"


def test_run_task_passes_original_corpus_to_scorer_in_corpus_mode():
    contract = TaskContract.from_dict({
        "name": "t1",
        "preconditions": [{"check": "pc1"}],
    })
    corpus = InMemoryCorpusAdapter([])
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj("t1")
        return traj, {}

    def fake_score_task_run(**kwargs):
        captured["corpus_is_same"] = kwargs["corpus"] is corpus
        result = MagicMock()
        result.task_name = "t1"
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.tool_call_count = 0
        return result

    with patch("groundeval.task_eval.score_task_run", side_effect=fake_score_task_run):
        run_task(
            contract=contract,
            agent_fn=fake_agent,
            corpus=corpus,
            policy=None,
            max_steps=5,
        )

    assert captured["corpus_is_same"] is True


def test_run_all_tasks_fixture_only_mode_builds_shared_fixture_backend():
    contracts = [
        TaskContract.from_dict({
            "name": "t1",
            "preconditions": [{"check": "pc1"}],
            "allowed_tools": {
                "fetch_customer": {
                    "artifact_id": "crm-1",
                    "returns": {"status": "active"},
                    "subsystem": "crm",
                }
            },
        }),
        TaskContract.from_dict({
            "name": "t2",
            "preconditions": [{"check": "pc2"}],
            "allowed_tools": {
                "fetch_email": {
                    "artifact_id": "email-1",
                    "returns": {"subject": "Hi"},
                    "subsystem": "email",
                }
            },
        }),
    ]
    seen = []

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj(question.question_id)
        return traj, {}

    def fake_run_task(**kwargs):
        seen.append(type(kwargs["corpus"]).__name__)
        result = MagicMock()
        result.task_name = kwargs["contract"].name
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.horizon_violations = 0
        result.actor_gate_violations = 0
        result.subsystem_violations = 0
        return result

    with patch("groundeval.task_eval.run_task", side_effect=fake_run_task):
        results = run_all_tasks(
            contracts=contracts,
            agent_fn=fake_agent,
            artifacts_dir="unused",
            policy=None,
            max_steps=3,
        )

    assert len(results) == 2
    assert seen == ["FixtureBackend", "FixtureBackend"]


def test_run_all_tasks_corpus_mode_uses_file_corpus_adapter(tmp_path):
    (tmp_path / "a1.json").write_text(
        '{"id": "a1", "subsystem": "crm", "status": "active"}'
    )

    contracts = [
        TaskContract.from_dict({
            "name": "t1",
            "preconditions": [{"check": "pc1"}],
        })
    ]
    seen = []

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj(question.question_id)
        return traj, {}

    def fake_run_task(**kwargs):
        seen.append(type(kwargs["corpus"]).__name__)
        result = MagicMock()
        result.task_name = kwargs["contract"].name
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.horizon_violations = 0
        result.actor_gate_violations = 0
        result.subsystem_violations = 0
        return result

    with patch("groundeval.task_eval.run_task", side_effect=fake_run_task):
        results = run_all_tasks(
            contracts=contracts,
            agent_fn=fake_agent,
            artifacts_dir=str(tmp_path),
            policy=None,
            max_steps=3,
        )

    assert len(results) == 1
    assert seen == ["FileCorpusAdapter"]


def test_run_all_tasks_derives_policy_from_first_contract_when_missing():
    contracts = [
        TaskContract.from_dict({
            "name": "t1",
            "preconditions": [{"check": "pc1"}],
            "actors": {"alice": "sales"},
            "roles": {"sales": {"subsystems": ["crm"]}},
            "allowed_tools": {
                "fetch_customer": {
                    "artifact_id": "crm-1",
                    "returns": {"status": "active"},
                    "subsystem": "crm",
                }
            },
        })
    ]
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj(question.question_id)
        return traj, {}

    def fake_run_task(**kwargs):
        captured["policy_type"] = type(kwargs["policy"]).__name__
        captured["role"] = kwargs["policy"].role_for_actor("alice")
        result = MagicMock()
        result.task_name = kwargs["contract"].name
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.horizon_violations = 0
        result.actor_gate_violations = 0
        result.subsystem_violations = 0
        return result

    with patch("groundeval.task_eval.run_task", side_effect=fake_run_task):
        run_all_tasks(
            contracts=contracts,
            agent_fn=fake_agent,
            artifacts_dir="unused",
            policy=None,
            max_steps=3,
        )

    assert captured["policy_type"] == "YamlAccessPolicy"
    assert captured["role"] == "sales"


def test_run_all_tasks_keeps_explicit_policy_when_provided():
    contracts = [
        TaskContract.from_dict({
            "name": "t1",
            "preconditions": [{"check": "pc1"}],
        })
    ]
    explicit_policy = YamlAccessPolicy({
        "actors": {"bob": "eng"},
        "roles": {"eng": {"subsystems": ["jira"]}},
    })
    captured = {}

    def fake_agent(question, context, tools, max_steps, runtime=None):
        traj = _fake_traj(question.question_id)
        return traj, {}

    def fake_run_task(**kwargs):
        captured["same_policy"] = kwargs["policy"] is explicit_policy
        result = MagicMock()
        result.task_name = kwargs["contract"].name
        result.counterfactual_score = 0.0
        result.silence_score = 0.0
        result.perspective_score = 0.0
        result.overall_score = 0.0
        result.horizon_violations = 0
        result.actor_gate_violations = 0
        result.subsystem_violations = 0
        return result

    with patch("groundeval.task_eval.run_task", side_effect=fake_run_task):
        run_all_tasks(
            contracts=contracts,
            agent_fn=fake_agent,
            artifacts_dir="unused",
            policy=explicit_policy,
            max_steps=3,
        )

    assert captured["same_policy"] is True
