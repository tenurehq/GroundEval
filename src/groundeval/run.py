"""
groundeval/run.py
======================
CLI entrypoint. Two commands:

    python -m groundeval generate --config config.yaml --events events.jsonl
        Produces eval_questions.json in the output directory.

    python -m groundeval eval --config config.yaml --questions eval_questions.json
        Runs an agent against the questions and writes results.json.

Minimal viable config.yaml:

    output_dir: ./eval_output

    actors:
      alice: engineer
      bob: sales

    roles:
      engineer:
        subsystems: [jira, git, slack, confluence, email]
        broadcast_event_types: [incident_opened, incident_resolved]
      sales:
        subsystems: [salesforce, email, slack]

    # optional: point at a directory of JSON artifact files
    artifacts_dir: ./artifacts

    # Optional — fine-tune perspective question balance
    perspective:
      positive_ratio: 0.5
      negative_permission_ratio: 0.25
      negative_temporal_ratio: 0.25
      require_cross_subsystem_cases: true

    causal_links:
      - name: ticket_closed_after_incident
        cause_event_type: incident_opened
        effect_event_type: ticket_closed
        premise_template: "the incident had been caught earlier"
        outcome_template: "the ticket would have closed sooner"
        outcome_changed: true
        join:
          - cause: artifact_ids.jira
            effect: artifact_ids.jira

    silence_pairs:
      - trigger_event_type: escalation_opened
        response_event_type: postmortem_created
        search_space_subsystems: [confluence, jira]
        search_space:
          - subsystem: confluence
            query_template: "postmortem {artifact_ids.jira}"
          - subsystem: jira
            id_template: "{artifact_ids.jira}"
        max_gap_days: 7
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import logging
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .core import (
    AgentTrajectory,
    CausalLinkSpec,
    EvalQuestion,
    EvalResult,
    GatedRuntime,
    PerspectiveConfig,
    SilencePairSpec,
    load_events,
)
from .adapters import (
    EventLogPolicy,
    FileCorpusAdapter,
    NullCorpusAdapter,
    YamlAccessPolicy,
)
from .question_gen import AbsenceCatalogBuilder, CausalLinkIndexer, QuestionGenerator
from .scorers import (
    PerspectiveScorer,
    CounterfactualScorer,
    SilenceScorer,
    aggregate,
    combine_scores,
)

logger = logging.getLogger("groundeval")


def cmd_generate(args) -> None:
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Expected YAML mapping, got {type(raw_cfg).__name__}")
        cfg: dict = raw_cfg

    out_dir = Path(cfg.get("output_dir", "./eval_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading events from {args.events}")
    events = load_events(Path(args.events))
    logger.info(f"  {len(events)} events loaded")

    artifacts_dir = cfg.get("artifacts_dir")
    if artifacts_dir:
        corpus = FileCorpusAdapter(artifacts_dir)
        logger.info(f"  Corpus: FileCorpusAdapter ({artifacts_dir})")
    else:
        corpus = NullCorpusAdapter()
        logger.info("  Corpus: NullCorpusAdapter (context-injection mode)")

    use_event_log_policy = cfg.get("use_event_log_policy", True)
    if use_event_log_policy:
        policy = EventLogPolicy(cfg, events)
        logger.info("  Access policy: EventLogPolicy (event-log derived visibility)")
    else:
        policy = YamlAccessPolicy(cfg)
        logger.info("  Access policy: YamlAccessPolicy (subsystem-role based)")

    link_specs = [CausalLinkSpec.from_dict(d) for d in cfg.get("causal_links", [])]
    logger.info(f"  {len(link_specs)} causal link spec(s)")

    silence_specs = [SilencePairSpec.from_dict(d) for d in cfg.get("silence_pairs", [])]
    logger.info(f"  {len(silence_specs)} silence pair spec(s)")

    logger.info("Building causal link index...")
    link_indexer = CausalLinkIndexer(events, link_specs)
    causal_links = link_indexer.build()
    logger.info(f"  {len(causal_links)} causal links found")

    links_path = out_dir / "causal_links.json"
    with open(links_path, "w") as f:
        json.dump([l.to_dict() for l in causal_links], f, indent=2)

    logger.info("Building absence catalog...")
    absence_builder = AbsenceCatalogBuilder(events, silence_specs, corpus)
    absences, confirmed = absence_builder.build()
    logger.info(f"  {len(absences)} absences, {len(confirmed)} confirmed")

    absence_path = out_dir / "absence_catalog.json"
    with open(absence_path, "w") as f:
        json.dump([a.to_dict() for a in absences], f, indent=2)

    llm_fn = None
    if cfg.get("llm_question_prose"):
        llm_fn = _build_llm_fn(cfg)
        logger.info("  LLM question prose: enabled")

    perspective_actors = cfg.get("perspective_actors")
    perspective_config = None
    if "perspective" in cfg:
        perspective_config = PerspectiveConfig.from_dict(cfg["perspective"])
        logger.info(f"  Perspective config: {perspective_config}")

    logger.info("Generating questions...")
    generator = QuestionGenerator(
        events=events,
        causal_links=causal_links,
        absence_records=absences,
        confirmed_records=confirmed,
        policy=policy,
        corpus=corpus,
        llm_fn=llm_fn,
        perspective_actors=perspective_actors,
        perspective_config=perspective_config,
    )
    questions = generator.generate()
    logger.info(f"  {len(questions)} questions generated")

    by_type = {}
    for q in questions:
        by_type[q.question_type] = by_type.get(q.question_type, 0) + 1

    questions_path = out_dir / "eval_questions.json"
    with open(questions_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "events_file": str(args.events),
                    "total_questions": len(questions),
                    "by_type": by_type,
                    "causal_links": len(causal_links),
                    "absences": len(absences),
                },
                "questions": [q.to_dict() for q in questions],
            },
            f,
            indent=2,
        )

    logger.info(f"Questions written to {questions_path}")
    logger.info(f"  By type: {by_type}")


def cmd_eval(args) -> None:
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Expected YAML mapping, got {type(raw_cfg).__name__}")
        cfg: dict = raw_cfg

    out_dir = Path(cfg.get("output_dir", "./eval_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.questions) as f:
        data = json.load(f)

    questions = [
        EvalQuestion(**{
            k: v for k, v in q.items() if k in EvalQuestion.__dataclass_fields__
        })
        for q in data["questions"]
    ]

    if args.types:
        questions = [q for q in questions if q.question_type in args.types]
    if args.max_questions:
        questions = questions[: args.max_questions]

    logger.info(f"Running eval on {len(questions)} questions")

    artifacts_dir = cfg.get("artifacts_dir")
    corpus = FileCorpusAdapter(artifacts_dir) if artifacts_dir else NullCorpusAdapter()
    events = load_events(Path(args.events)) if args.events else []
    use_event_log_policy = cfg.get("use_event_log_policy", True)
    if use_event_log_policy and not args.events:
        logger.warning(
            "--events not provided but use_event_log_policy=true in config. "
            "EventLogPolicy will have empty event-derived visibility; "
            "falling back to role-based subsystems only."
        )
    policy = (
        EventLogPolicy(cfg, events) if use_event_log_policy else YamlAccessPolicy(cfg)
    )
    all_ids = corpus.list_ids()
    for q in questions:
        if q.actor_visible_artifacts is None and q.actor:
            q.actor_visible_artifacts = sorted(
                policy.visible_artifacts(
                    actor_id=q.actor,
                    all_artifact_ids=all_ids,
                    as_of=q.as_of_time,
                    corpus=corpus,
                )
            )

    agent_fn = _build_agent_fn(cfg, args)

    perspective_scorer = PerspectiveScorer()
    counterfactual_scorer = CounterfactualScorer()
    silence_scorer = SilenceScorer()

    results: List[EvalResult] = []
    per_question: List[dict] = []

    for i, question in enumerate(questions):
        logger.info(
            f"[{i + 1}/{len(questions)}] {question.question_type} — {question.question_id}"
        )

        try:
            result = _run_one(
                question=question,
                agent_fn=agent_fn,
                corpus=corpus,
                policy=policy,
                perspective_scorer=perspective_scorer,
                counterfactual_scorer=counterfactual_scorer,
                silence_scorer=silence_scorer,
                max_steps=args.max_steps,
                context_injection=args.context_injection,
                zero_shot=args.zero_shot,
            )
        except Exception as exc:
            logger.error(f"  Failed: {exc}")
            result = EvalResult(
                question_id=question.question_id,
                question_type=question.question_type,
                difficulty=question.difficulty,
                answer_score=0.0,
                answer_correct=False,
                trajectory_score=0.0,
                combined_score=0.0,
                failure_reason=str(exc),
                tool_call_count=0,
            )

        results.append(result)
        per_question.append(result.to_dict())
        logger.info(
            f"  answer={result.answer_score:.3f} "
            f"trajectory={result.trajectory_score:.3f} "
            f"combined={result.combined_score:.3f}"
        )

    summary = aggregate(results)
    model_safe = args.model.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"results_{model_safe}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "model": args.model,
                    "n_questions": len(results),
                    "context_injection": args.context_injection,
                },
                "summary": summary,
                "per_question": per_question,
            },
            f,
            indent=2,
        )

    logger.info(f"Results written to {out_path}")
    logger.info(
        f"Overall — "
        f"answer: {summary['overall']['answer_score']:.3f}  "
        f"trajectory: {summary['overall']['trajectory_score']:.3f}  "
        f"combined: {summary['overall']['combined_score']:.3f}"
    )


def _run_one(
    question: EvalQuestion,
    agent_fn,
    corpus,
    policy,
    perspective_scorer: PerspectiveScorer,
    counterfactual_scorer: CounterfactualScorer,
    silence_scorer: SilenceScorer,
    max_steps: int,
    context_injection: bool,
    zero_shot=False,
) -> EvalResult:
    """
    Run a single question through the agent and score it.

    If context_injection is True: the runtime packs visible artifacts into
    a context string and the agent runs without tool access.

    If False: a ``GatedRuntime`` is created with the corpus + policy.
    The agent receives the runtime's tool surface and the runtime
    records every call, auto-enforcing gates.
    """

    qtype = question.question_type

    steps = max_steps
    if qtype == "SILENCE" and question.expected_search_space:
        steps = max(steps, len(question.expected_search_space) + 5)

    if zero_shot:
        trajectory, final_answer = agent_fn(
            question=question,
            context="(no artifacts or tools available)",
            tools=None,
            max_steps=1,
        )
    elif context_injection:
        context = _build_context(question, corpus)
        trajectory, final_answer = agent_fn(
            question=question,
            context=context,
            tools=None,
            max_steps=steps,
        )
    else:
        all_ids = corpus.list_ids()
        actor_visible = (
            policy.visible_artifacts(
                actor_id=question.actor or "",
                all_artifact_ids=all_ids,
                as_of=question.as_of_time,
                corpus=corpus,
            )
            if qtype == "PERSPECTIVE"
            else None
        )
        actor_subsystems = (
            policy.subsystems_for_role(question.actor_role or "")
            if qtype == "PERSPECTIVE" and question.actor_role
            else None
        )

        runtime = GatedRuntime(
            corpus=corpus,
            policy=policy,
            question=question,
            actor_visible_artifacts=actor_visible,
            actor_subsystem_access=actor_subsystems,
        )

        tool_specs = _build_tool_specs(cfg={})
        trajectory, final_answer = agent_fn(
            question=question,
            context=None,
            tools=tool_specs,
            max_steps=steps,
            runtime=runtime,
        )
        runtime_traj = runtime.trajectory()
        trajectory.tool_calls = runtime_traj.tool_calls
        trajectory.horizon_violations = runtime_traj.horizon_violations
        trajectory.actor_gate_violations = runtime_traj.actor_gate_violations
        trajectory.subsystem_violations = runtime_traj.subsystem_violations
        trajectory.dead_ends_hit = runtime_traj.dead_ends_hit
        trajectory.dead_ends_recovered = runtime_traj.dead_ends_recovered

    trajectory.final_answer = final_answer or {}

    if qtype == "PERSPECTIVE":
        answer_score, answer_correct = perspective_scorer.score_answer(
            final_answer, question.ground_truth
        )
        trajectory_score = perspective_scorer.score_trajectory(trajectory, question)
    elif qtype == "COUNTERFACTUAL":
        answer_score, answer_correct = counterfactual_scorer.score_answer(
            final_answer, question.ground_truth
        )
        trajectory_score = counterfactual_scorer.score_trajectory(trajectory, question)
    else:
        answer_score, answer_correct = silence_scorer.score_answer(
            final_answer, question.ground_truth
        )
        trajectory_score = silence_scorer.score_trajectory(trajectory, question)

    combined = combine_scores(qtype, answer_score, trajectory_score)

    n_calls = len(trajectory.tool_calls)
    actor_violations = sum(1 for c in trajectory.tool_calls if c.actor_gate_violation)
    search_coverage = trajectory.search_space_coverage

    return EvalResult(
        question_id=question.question_id,
        question_type=qtype,
        difficulty=question.difficulty,
        answer_score=answer_score,
        answer_correct=answer_correct,
        trajectory_score=trajectory_score,
        combined_score=combined,
        failure_reason=None,
        tool_call_count=n_calls,
        meta={
            "actor_gate_violations": actor_violations,
            "search_space_coverage": search_coverage,
            "prompt_tokens": trajectory.prompt_tokens,
            "completion_tokens": trajectory.completion_tokens,
        },
    )


def _build_context(
    question: EvalQuestion,
    corpus,
    max_tokens: Optional[int] = None,
) -> str:
    """
    For context-injection mode: fetch relevant artifacts and pack into a string.
    In PERSPECTIVE mode, only inject visible artifacts.
    """
    max_tokens = max_tokens or 16000

    artifact_ids = question.actor_visible_artifacts or []
    if question.question_type == "SILENCE":
        artifact_ids = question.expected_search_space or []

    seen = set()
    ranked = []
    for aid in artifact_ids:
        if aid and not aid.startswith("[") and aid not in seen:
            seen.add(aid)
            ranked.append(aid)

    if question.question_type == "PERSPECTIVE" and question.as_of_time:

        def _recency(aid: str) -> str:
            return corpus.timestamp_of(aid) or ""

        ranked.sort(key=_recency, reverse=True)

    chunks: List[str] = []
    tokens_used = 0

    for aid in ranked:
        doc = corpus.fetch(aid)
        if not doc:
            continue
        text = f"--- {aid} ---\n{json.dumps(doc, indent=2)}"
        est_tokens = len(text) // 4 + 4
        if tokens_used + est_tokens > max_tokens:
            break
        chunks.append(text)
        tokens_used += est_tokens

    return "\n\n".join(chunks) if chunks else "(no artifacts available)"


def _build_tool_specs(cfg: dict) -> List[dict]:
    """
    Returns a minimal tool spec list for the agent.
    Users can extend this in their own agent_fn.
    """
    return [
        {"name": "fetch_artifact", "description": "Retrieve an artifact by ID"},
        {"name": "search_artifacts", "description": "Search artifacts by keyword"},
    ]


def _build_llm_fn(cfg: dict):
    """
    Build an LLM callable for question prose generation by reusing
    ModelProvider.from_config so key / base_url / retries / temperature
    are handled in one place.
    """
    from .providers import ModelProvider, build_prose_fn

    provider = ModelProvider.from_config(cfg)
    return build_prose_fn(provider)


def _build_agent_fn(cfg: dict, args) -> Callable[..., tuple[AgentTrajectory, dict]]:
    from .providers import ModelProvider, build_agent_fn

    provider = ModelProvider.from_config(cfg)
    return build_agent_fn(provider)


def main():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_file), mode="a"),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Logging to {log_file}")

    parser = argparse.ArgumentParser(
        prog="groundeval",
        description="Deterministic agentic evaluation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate eval questions from an event log")
    gen.add_argument("--config", required=True, help="Path to config.yaml")
    gen.add_argument("--events", required=True, help="Path to events.jsonl")

    ev = sub.add_parser("eval", help="Run an agent on generated questions")
    ev.add_argument("--config", required=True, help="Path to config.yaml")
    ev.add_argument("--questions", required=True, help="Path to eval_questions.json")
    ev.add_argument(
        "--events",
        required=False,
        help="Path to events.jsonl (needed for EventLogPolicy)",
    )
    ev.add_argument("--model", default="claude-sonnet-4-6")
    ev.add_argument("--max-steps", type=int, default=5)
    ev.add_argument(
        "--context-injection",
        action="store_true",
        help="Pre-inject context instead of using tool calls",
    )
    ev.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run with no corpus access and no tools. Parametric knowledge only",
    )
    ev.add_argument(
        "--types",
        nargs="+",
        choices=["PERSPECTIVE", "COUNTERFACTUAL", "SILENCE"],
    )
    ev.add_argument("--max-questions", type=int, default=None)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
