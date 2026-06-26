"""
groundeval/task_eval.py
========================
Task-contract evaluation path.

The simpler alternative to event-log-based evaluation:
  1. Define a task contract with preconditions
  2. Seed artifacts as ground truth
  3. Run the agent against the contract
  4. Score through all three tracks (Counterfactual, Silence, Perspective)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .core import (
    AgentTrajectory,
    GatedRuntime,
    TaskContract,
    ANSWER_SCHEMA_TASK,
)
from .core import AccessPolicy, CorpusAdapter
from .adapters import FileCorpusAdapter, NullCorpusAdapter, YamlAccessPolicy
from .scorers import score_task_run, aggregate_task_results, TaskEvalResult


logger = logging.getLogger("groundeval.task_eval")


def build_task_question_text(contract: TaskContract) -> str:
    pc_descriptions = []
    for pc in contract.preconditions:
        pc_descriptions.append(f"  - {pc.check}: {pc.description}")

    return (
        f"{contract.task_description}\n\n"
        f"Before acting, verify each of the following:\n"
        + "\n".join(pc_descriptions)
        + f"\n\nAfter verifying all preconditions, determine whether "
        f"the action is valid (all preconditions must pass). "
        f"Submit your findings with the preconditions_verified list "
        f"and the final decision in '{contract.decision_field}'."
    )


def run_task(
    contract: TaskContract,
    agent_fn,
    corpus: CorpusAdapter,
    policy: AccessPolicy | None = None,
    max_steps: int = 10,
) -> TaskEvalResult:
    if policy is None:
        policy = YamlAccessPolicy({"actors": {}, "roles": {}})

    question_text = build_task_question_text(contract)

    actor = contract.actor
    role = contract.role
    if not actor and contract.actors:
        actor = next(iter(contract.actors.keys()))
        role = contract.actors[actor]

    from .core import FixtureBackend

    fixture_mode = contract.is_fixture_mode

    actor_subsystems: set[str] | None = None
    if actor and role:
        actor_subsystems = policy.subsystems_for_role(role)

    backend = None
    if fixture_mode:
        backend = FixtureBackend(contract.allowed_tools)
        all_ids = backend.list_ids()
        actor_visible: set[str] | None = None
        if actor and role:
            actor_visible = policy.visible_artifacts(
                actor_id=actor,
                all_artifact_ids=all_ids,
                as_of=None,
                corpus=backend,
            )

        runtime = GatedRuntime(
            corpus=backend,
            policy=policy,
            task_id=contract.name,
            actor=actor,
            as_of=None,
            actor_visible_artifacts=actor_visible,
            actor_subsystem_access=actor_subsystems,
        )

        all_subsystems = sorted(
            set(
                backend.subsystem_of(aid)
                for aid in all_ids
                if backend.subsystem_of(aid)
            )
        )
        runtime._all_subsystems = all_subsystems
    else:
        actor_visible = None
        if actor and role:
            all_ids = corpus.list_ids()
            actor_visible = policy.visible_artifacts(
                actor_id=actor,
                all_artifact_ids=all_ids,
                as_of=None,
                corpus=corpus,
            )

        runtime = GatedRuntime(
            corpus=corpus,
            policy=policy,
            task_id=contract.name,
            actor=actor,
            as_of=None,
            actor_visible_artifacts=actor_visible,
            actor_subsystem_access=actor_subsystems,
        )

        all_subsystems = sorted(
            set(
                corpus.subsystem_of(aid)
                for aid in corpus.list_ids()
                if corpus.subsystem_of(aid)
            )
        )
        runtime._all_subsystems = all_subsystems

    question = _TaskEvalQuestion(
        question_id=f"TASK_{hashlib.md5(contract.name.encode()).hexdigest()[:8]}",
        question_type="TASK",
        question_text=question_text,
        difficulty="medium",
        ground_truth={},
        actor=actor,
        actor_role=role,
        as_of_time=None,
        actor_visible_artifacts=None,
        actor_subsystem_access=None,
        expected_answer_schema=ANSWER_SCHEMA_TASK,
    )

    trajectory, final_answer = agent_fn(
        question=question,
        context=None,
        tools=None,
        max_steps=max_steps,
        runtime=runtime,
    )

    if runtime is not None:
        runtime_traj = runtime.trajectory()
        if runtime_traj.tool_calls:
            trajectory.tool_calls = runtime_traj.tool_calls
            trajectory.horizon_violations = runtime_traj.horizon_violations
            trajectory.actor_gate_violations = runtime_traj.actor_gate_violations
            trajectory.subsystem_violations = runtime_traj.subsystem_violations
            trajectory.dead_ends_hit = runtime_traj.dead_ends_hit
            trajectory.dead_ends_recovered = runtime_traj.dead_ends_recovered

    trajectory.final_answer = final_answer or {}

    result = score_task_run(
        trajectory=trajectory,
        final_answer=final_answer or {},
        contract=contract,
        corpus=backend if fixture_mode else corpus,
        policy=policy,
        actor=actor,
        role=role,
    )

    return result


def run_all_tasks(
    contracts: list[TaskContract],
    agent_fn,
    artifacts_dir: str,
    policy: AccessPolicy | None = None,
    max_steps: int = 10,
) -> list[TaskEvalResult]:
    fixture_only = all(c.is_fixture_mode for c in contracts)

    fixture_only = all(c.is_fixture_mode for c in contracts)

    if fixture_only:
        from .core import FixtureBackend

        all_tools: list = []
        for c in contracts:
            all_tools.extend(c.allowed_tools)
        corpus = FixtureBackend(all_tools)
        logger.info(
            f"  Corpus: fixture mode ({len(corpus.list_ids())} virtual artifacts)"
        )
    else:
        corpus = FileCorpusAdapter(artifacts_dir)
        logger.info(
            f"  Corpus: {len(corpus.list_ids())} artifacts loaded from {artifacts_dir}"
        )

    if policy is None and contracts:
        first = contracts[0]
        if first.actors and first.roles:
            policy = YamlAccessPolicy({
                "actors": first.actors,
                "roles": first.roles,
            })

    results = []
    for contract in contracts:
        logger.info(f"Running task: {contract.name}")
        result = run_task(
            contract=contract,
            agent_fn=agent_fn,
            corpus=corpus,
            policy=policy,
            max_steps=max_steps,
        )
        results.append(result)
        logger.info(
            f"  counterfactual={result.counterfactual_score:.3f} "
            f"silence={result.silence_score:.3f} "
            f"perspective={result.perspective_score:.3f} "
            f"overall={result.overall_score:.3f} "
            f"violations={result.horizon_violations + result.actor_gate_violations + result.subsystem_violations}"
        )

    return results


class _TaskEvalQuestion:
    def __init__(
        self,
        question_id: str,
        question_type: str,
        question_text: str,
        difficulty: str,
        ground_truth: dict,
        actor: str | None = None,
        actor_role: str | None = None,
        as_of_time: str | None = None,
        actor_visible_artifacts: list[str] | None = None,
        actor_subsystem_access: list[str] | None = None,
        expected_answer_schema: dict | None = None,
    ):
        self.question_id = question_id
        self.question_type = question_type
        self.question_text = question_text
        self.difficulty = difficulty
        self.ground_truth = ground_truth
        self.actor = actor
        self.actor_role = actor_role
        self.as_of_time = as_of_time
        self.actor_visible_artifacts = actor_visible_artifacts
        self.actor_subsystem_access = actor_subsystem_access
        self.expected_answer_schema = expected_answer_schema
