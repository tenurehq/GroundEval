# GroundEval

**A debugging loop for AI agents. See what your agent checked, what it skipped, what evidence it used, and whether each action stayed inside the right permissions.**


## The problem with LLM judges

An LLM judge tells you if the answer looks right. It cannot tell you if the agent:

- cited a document it should not have had access to
- used information from after the question's cutoff time
- skipped verifying a precondition before declaring it safe to act
- claimed to find evidence that does not exist in the artifacts

GroundEval answers these questions deterministically. The ground truth comes from state, artifacts, access rules, and tool traces. Not from another model's judgment.

## What GroundEval actually checks

GroundEval scores every task run through three tracks simultaneously. Each track tests a different failure mode against the same trajectory and answer.

**Counterfactual: Did the evidence actually support the decision?** If your agent said it was safe to act because all checks passed, GroundEval verifies whether the evidence it cited actually supports that conclusion or whether it just happened to be nearby.

**Silence: Did the agent skip anything it was supposed to check??** A correct final answer is not enough if the agent never verified a required condition before acting. This catches agents that get lucky, not agents that are reliable.

**Perspective: Did the agent stay in its lane?** If a role only has access to CRM, email, and outreach logs, and the agent touched the audit trail, that is a violation, even if the answer was correct.

## How it works

1. **Observe an agent.** Record the tools it called, the evidence it returned, and the final answer it produced.
2. **Generate a draft config.** GroundEval drafts a task contract, tool map, fixture artifacts, and review checklist from the observed run.
3. **Review the draft.** Confirm the required preconditions, allowed tools, ground truth fields, roles, and decision field.
4. **Run the evaluation.** GroundEval reruns the agent through a gated runtime and scores the trajectory across evidence use, skipped checks, and access boundaries.

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

## Quick start: observe an existing agent

Observe Mode is the fastest path from an existing agent to a deterministic evaluation.

```bash
uv sync --group dev --group crewai
uv run python -m groundeval observe \
  --framework crewai \
  --crew-class your_package.your_module.YourCrew \
  --output eval_output
```

GroundEval writes:

```text
eval_output/
  observed_run.json
  observe_report.md
  draft_config/
    config.yaml
    tool_map.yaml
    REVIEW.md
    task_contracts/
    artifacts/
```

Observation is not evaluation. The draft config is marked as unreviewed because observed behavior may not be correct behavior. Review the inferred preconditions, allowed tools, fixture returns, roles, and decision field before scoring.

Validate the draft:

```bash
uv run python -m groundeval validate --config eval_output/draft_config/config.yaml
```

After review, mark it reviewed:

```bash
uv run python -m groundeval validate \
  --config eval_output/draft_config/config.yaml \
  --mark-reviewed
```

Then run the evaluation:

```bash
uv run python -m groundeval task --config eval_output/draft_config/config.yaml
```

### Draft modes

Use `--draft-mode conservative`, `standard`, or `aggressive` to control how much GroundEval infers from the observed run. All inferred checks still require review before scoring.

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
uv run python -m groundeval task --config config.yaml
```

By default, the bundled demo uses the provider and model declared in `config.yaml`.

To override the model from the CLI:

```bash
uv run python -m groundeval task --config config.yaml --model gpt-4o
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
uv run python -m groundeval validate --config config.yaml
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

Use Observe Mode to draft the first config, then move deeper configuration into `docs/`.

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

## Bringing your own agent

GroundEval includes a built-in Anthropic/OpenAI agent loop so the demo can run immediately.

For production-style evaluation, use Observe Mode or wire GroundEval to your own runner. The scoring pipeline stays the same: GroundEval records the trajectory, validates access and evidence, and scores the result deterministically.

See the integration guides in `docs/` for framework-specific setup.

## Supported agent frameworks

GroundEval's observer interface is designed for framework adapters. Each adapter loads the agent, instruments its tools, records the observed run, and hands the result to the same draft and scoring pipeline.

### CrewAI

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


## What this is not

GroundEval is not for grading whether your agent sounds good. If you want to evaluate tone, style, or conversational quality, use an LLM judge for that. GroundEval is for the parts where there is a right answer: did it check the right things, use the right evidence, and stay within its permissions.

## Preprint implementation

The implementation used for the initial GroundEval preprint is preserved at:

- Branch: `paper/preprint-2026`
- Tag: `groundeval-preprint-v1`

The `main` branch contains the simplified public version intended for easier adoption.

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
