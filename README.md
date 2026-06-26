# GroundEval

**Deterministic evaluation for agents that reason over state.**

GroundEval is a deterministic evaluation harness for agents that act on state. You define what an agent must verify before acting, what evidence it is allowed to use, and what decision it must return. GroundEval runs the agent through a gated runtime, records the trajectory, and scores whether the action was justified by valid evidence.

## The problem with LLM judges

An LLM judge can tell you if an answer looks reasonable. It cannot tell you if the agent:

- cited a document it should not have had access to
- used information from after the question's cutoff time
- skipped verifying a precondition before declaring it safe to act
- claimed to find evidence that does not exist in the artifacts

GroundEval answers these questions deterministically. The ground truth comes from state, artifacts, access rules, and tool traces. Not from another model's judgment.

## Three tracks, one run

GroundEval scores every task run through three tracks simultaneously. Each track tests a different failure mode against the same trajectory and answer.

**Counterfactual: Did the evidence support the causal claim?** Tests whether the agent's conclusion depends on a valid artifact-grounded relationship, not mere proximity or unsupported inference. If the agent claims an action was safe because all prerequisites passed, the scorer checks whether that causal or decision chain is supported by the evidence.

**Silence: Did the agent verify all preconditions?** Tests whether the agent resolved every required condition before deciding or acting. A correct conclusion is not enough if a required precondition remained unresolved. The diagnostic trace can show shallow search, empty-result handling, and dead-end recovery, but the core Silence failure is acting with unresolved state.

**Perspective: Did the agent stay within permission boundaries?** Tests whether the agent accessed only what its role allows. If a sales rep role only has `crm`, `email`, and `outreach_log` access, and the agent tried to access `audit_trail`, that is a violation. Horizon gates and actor visibility cones apply too.

## How it works

1. **Define the task.** Declare what the agent is trying to decide or do, and what must be verified before it may act.
2. **Provide the evidence world.** Use artifacts or an adapter-backed corpus to represent the state the agent should reason over.
3. **Run the agent.** The agent searches, retrieves evidence, and submits a structured answer.
4. **Get deterministic scores.** GroundEval scores the same run across Counterfactual, Silence, and Perspective, then writes a structured report.

A correct answer through an invalid trajectory is a failure. GroundEval penalizes agents that reach the right conclusion through the wrong evidence, skipped verification, or out-of-bounds access.

## What gets scored

Each task run produces four scores:

| Score | What it measures |
|---|---|
| `counterfactual_score` | Did cited evidence support the agent's conclusions? |
| `silence_score` | Did the agent verify all preconditions before deciding? |
| `perspective_score` | Did the agent stay within permission boundaries? |
| `overall_score` | Mean of the three track scores |

Each track reports an answer verdict and trajectory diagnostics. GroundEval ships with default scoring profiles. Advanced users can override track weights and penalties.

Every evaluated task also includes a structured diagnostic trace. The trace records the agent run as data: tool calls, tool results, submitted answers, and errors. Diagnostics are not used to award credit. The scorer still relies on the deterministic trajectory. The diagnostic trace exists so you can debug why the score happened without reconstructing it from interleaved logs.

## Quick start: run the bundled demo

GroundEval ships with a complete demo so you can see the CLI, agent trajectory, deterministic scoring, and generated report without writing your own config or artifacts first.

Before running the demo, configure a model provider.

For Anthropic:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

For OpenAI:

```bash
export OPENAI_API_KEY=your_key_here
```

If you are using an OpenAI-compatible endpoint, also set:

```bash
export OPENAI_BASE_URL=your_url_here
```

Then run:

```bash
uv sync --group dev
uv run python -m groundeval task --config config/evaluation.yaml
```

By default, the bundled demo uses the provider and model declared in `config/evaluation.yaml`.

To override the model from the CLI:

```bash
uv run python -m groundeval task --config config/evaluation.yaml --model gpt-4o
```

The demo runs a sales-outreach verification task. The agent searches the bundled evidence corpus, retrieves artifacts, submits a structured answer, and GroundEval scores the run across Counterfactual, Silence, and Perspective.

Results are written to:

```bash
eval_output/
```

You should see terminal output similar to:

```text
Task results written to eval_output/task_results_<model>.json
Overall -- counterfactual=0.833  silence=0.750  perspective=1.000  overall=0.861  accuracy=1.000
Total violations: 0
```

The JSON report includes per-task scores, precondition-level results, violation counts, tool trajectory, submitted answer, and diagnostic details.

### Validate the demo config without running the agent

```bash
uv run python -m groundeval validate --config config/config.yaml
```

This checks that the config is well-formed, artifacts are present, and task contracts are valid before spending API credits.

## What the demo includes

The bundled demo includes everything needed for a first run:

- a sample task contract
- seed artifacts
- role and subsystem permissions
- provider settings
- a built-in agent loop
- deterministic scoring across all three tracks

The demo is intentionally small. It exists to show how the CLI works, what a trajectory looks like, and what kind of report GroundEval generates.

You do not need to copy the demo structure exactly for your own evaluations.

## Configuring your own evaluation

When you are ready to evaluate your own agent or domain, use the guides in `docs/`.

Start here:

- [`docs/crewai.md`](docs/crewai.md) for CrewAI integration
- `docs/` for task contracts, artifacts, access policy, providers, and custom runners

At a high level, a real evaluation defines:

- what task the agent is trying to complete
- what preconditions must be verified before acting
- what evidence the agent is allowed to access
- where ground truth comes from
- how the agent runner is invoked
- what structured answer the scorer should expect

GroundEval does not require your artifacts to look like the demo artifacts. They can represent tickets, claims, alerts, contracts, medical orders, Slack messages, audit records, database rows, GitHub issues, notes, game state, home-lab logs, or any other state your agent reasons over.

The important part is that the task contract points to the fields that define correctness, and the runtime records the evidence the agent actually searched, retrieved, and used.

## Bringing your own agent

GroundEval includes a built-in Anthropic/OpenAI agent loop so the demo can run immediately.

For production-style evaluation, wire GroundEval to your own agent or framework. The scoring pipeline stays the same: GroundEval records the trajectory, validates access and evidence, and scores the result deterministically.

See the integration guides in `docs/` for framework-specific setup.

## Supported agent frameworks

### CrewAI

GroundEval ships with a first-class CrewAI adapter. Point GroundEval at your existing `@CrewBase` class, and the adapter loads your crew, wraps its tools through the gated runtime, records the trajectory, and scores Counterfactual, Silence, and Perspective from the same run.

```bash
uv sync --group dev --group crewai
```

[See the CrewAI integration guide →](docs/crewai.md)

## Extension points

The framework is designed so you can swap parts without rebuilding the engine.

| What you might replace | When you would do it |
|---|---|
| Artifacts | You already know your domain's preconditions and want to evaluate against your own state |
| `CorpusAdapter` | Your artifacts live in MongoDB, Elasticsearch, Postgres, S3, or a proprietary backend |
| `AccessPolicy` | Your visibility rules are tenant-scoped, project-scoped, account-scoped, or time-scoped |
| Agent runner | You are using a specific model provider or agent framework |
| Task contracts | Your domain has different precondition verification requirements |

## Design principles

- **Ground truth comes from state, not an LLM judge.** Answer keys are derived from artifacts, access policy, and recorded trajectories. No LLM judge evaluates the agent's output.
- **A correct answer through an invalid trajectory is a failure.** The framework penalizes agents that reach the right conclusion through wrong evidence, skipped verification, or out-of-bounds access.
- **The config declares what to verify; the framework handles mechanics.** You describe the task and required checks. GroundEval handles gating, recording, and scoring.
- **Local-first, adapter-ready.** GroundEval starts with a local demo so anyone can run an eval quickly. The same artifact interface can later point at a database, object store, or production retrieval backend.
- **Start with the demo, graduate to your own evals.** Run the bundled demo to see the framework in action. When ready, swap in your own contracts, artifacts, policies, and agent.

## What this is not

GroundEval does not replace human or model judgment for subjective quality: tone, style, persuasiveness, conversational fluency. It is for cases where correctness can be verified from state, evidence, permissions, and tool traces.

## Citation

If you use this work, please cite:

```bibtex
@article{Jeffrey_Flynt_GroundEval_A_Deterministic_2026,
author = {Jeffrey Flynt},
journal = {arXiv preprint},
title = {{GroundEval: A Deterministic Replacement for LLM-as-Judge in Stateful Agent Evaluation}},
url = {https://arxiv.org/abs/2606.22737},
year = {2026}
}
```
