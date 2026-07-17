# GroundEval

**A debugging loop for AI agents. See what your agent checked, what it skipped, what evidence it actually used, and whether every action stayed inside the permissions it was supposed to have.**

## Framework support

| Framework | Observe Mode | Deterministic Scoring | Integration Guide |
|---|:---:|:---:|:---:|
| CrewAI | ✅ | ✅ | [docs/crewai.md](docs/crewai.md) |
| LangGraph | ✅ | ✅ | [docs/langgraph.md](docs/langgraph.md) |
| Microsoft Agent Framework | ✅ | ✅ | [docs/maf.md](docs/maf.md) |
| OpenAI Agents SDK | ✅ | ✅ | [docs/openai-agents.md](docs/openai-agents.md) |
| LlamaIndex | ⏳ | ⏳ | Planned |
| Pydantic AI | ⏳ | ⏳ | Planned |

> More adapters are on the way. If there's a framework you need, open an issue or check [CONTRIBUTING.md](CONTRIBUTING.md), we'd love the help.

## Why LLM judges aren't enough on their own

An LLM judge can tell you whether an answer looks right. What it can't tell you is whether your agent:

- cited a document it should never have had access to in the first place
- pulled in information from after the question's cutoff time
- skipped a precondition it was supposed to verify before calling something safe
- claimed evidence exists that simply isn't in the artifacts

GroundEval is built to answer exactly these questions, and it does it deterministically. The ground truth here comes from actual state, artifacts, access rules, and tool traces, not from another model guessing at what looks plausible.

## What GroundEval actually checks

Every task run gets scored across three tracks at once, each one probing a different way an agent can quietly go wrong, even while looking correct on the surface.

**Counterfactual: Did the evidence actually support the decision?** If your agent says it's safe to act because "all checks passed," GroundEval goes back and checks whether the cited evidence really backs that up, or whether it just happened to be sitting nearby.

**Silence: Did the agent skip something it was supposed to check?** Getting the right answer isn't enough if the agent never verified a required condition along the way. This is what separates an agent that got lucky from one you can actually trust.

**Perspective: Did the agent stay in its lane?** If a role is only supposed to touch CRM, email, and outreach logs, and the agent reaches into the audit trail anyway, that's a violation. Doesn't matter if the final answer was still correct.

## How it works

1. **Observe an agent.** GroundEval records the tools it called, the evidence it pulled back, and the answer it landed on.
2. **Generate a draft config.** From that observed run, GroundEval drafts a task contract, an observed tool summary, and a review checklist.
3. **Review the draft.** You confirm the required preconditions, tool expectations or allowed tools, ground truth fields, roles, and the decision field. This step matters, don't skip it.
4. **Run the evaluation.** For framework-backed runs, GroundEval observes a fresh run and scores it against your reviewed contract. In task mode, it runs the configured agent loop directly against the task contracts.

Here's the core idea worth internalizing: a correct answer reached through an invalid trajectory still counts as a failure. GroundEval penalizes agents that arrive at the right conclusion by the wrong path, whether that's using the wrong evidence, skipping verification, or wandering outside where they're allowed to look.

## What gets scored

Each task run produces four scores:

| Score | What it measures |
|---|---|
| `counterfactual_score` | Did the cited evidence actually support the agent's conclusions? |
| `silence_score` | Did the agent verify every precondition before deciding? |
| `perspective_score` | Did the agent stay within its permission boundaries? |
| `overall_score` | The mean of the three track scores |

Each track also comes back with an answer verdict and trajectory diagnostics. GroundEval ships with sensible default scoring profiles, but if you need more control, you can override track weights and penalties.

Every task also produces a structured diagnostic trace, essentially the agent run recorded as data: tool calls, tool results, submitted answers, errors, all of it. That trace isn't used to award credit; scoring still comes from the deterministic trajectory alone. It's there so you can actually understand why a score happened without having to piece it back together from a jumble of interleaved logs.

## Quick start: observe an existing agent

Observe Mode is the fastest way to get from an agent you already have to a deterministic evaluation.

```bash
uv sync --group dev --group crewai
uv run python -m groundeval observe \
  --framework crewai \
  --agent-class your_package.your_module.YourCrew \
  --output eval_output
```

GroundEval writes:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  draft_config/
    config.yaml
    observed_tools.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

One thing worth being clear on: observation is not the same as evaluation. The draft config comes back marked unreviewed on purpose, because how the agent actually behaved isn't necessarily how it should behave. Take the time to review the inferred preconditions, tool expectations, roles, and decision field before you score anything against it.

Validate the draft:

```bash
uv run python -m groundeval validate --config eval_output/draft_config/config.yaml
```

Once you've reviewed it, mark it reviewed:

```bash
uv run python -m groundeval validate \
  --config eval_output/draft_config/config.yaml \
  --mark-reviewed
```

Then score a fresh observed run against that reviewed config:

```bash
uv run python -m groundeval observe \
  --framework crewai \
  --agent-class your_package.your_module.YourCrew \
  --config eval_output/draft_config/config.yaml \
  --score \
  --output eval_output/scored_run
```

### Observe artifacts

Observe mode writes the standard observation artifacts below:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
```

When draft generation is enabled, GroundEval also writes:

```text
eval_output/
  draft_config/
    config.yaml
    observed_tools.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

When there is framework-rich observation data available, GroundEval also writes framework-specific artifacts.

CrewAI can also write:

```text
eval_output/
  observed_run_crewai.json
```

MAF can also write:

```text
eval_output/
  observed_run_maf.json
  observe_report_maf.md
```

LangGraph can also write:

```text
eval_output/
  observed_run_langgraph.json
  observe_report_langgraph.md
```

With scoring, GroundEval writes observation outputs plus scored results into whatever directory you pass to `--output`. For example:

```text
eval_output/scored_run/
  observed_run_<timestamp>.json
  observe_report_<timestamp>.md
  observe_diagram_<timestamp>.pdf
  observed_scores_<timestamp>.json
```

### Artifact filename behavior

Artifact filenames are intentionally not identical in every observe flow.

- `groundeval observe` with draft generation writes stable filenames like `observed_run.json`, `observe_report.md`, and `observe_diagram.pdf`.
- `groundeval observe --no-draft` writes timestamped filenames so multiple runs can coexist and be compared later.
- `groundeval observe --score` also writes timestamped filenames so you can keep multiple scored observation runs side by side and use `compare` on them.

### Draft modes

Use `--draft-mode conservative`, `standard`, or `aggressive` to control how much GroundEval is willing to infer from the observed run. No matter which mode you pick, every inferred check still needs your review before it's used for scoring.

## Comparing GroundEval outputs

There's also a JSON comparison workflow built in, useful for reviewing what changed between two runs, two observed score outputs, or two task result files.

See `compare.md` for the compare command itself, the file shapes it supports, and what the report looks like.

## The observe diagram

`observe_diagram.pdf` renders the observed run purely in terms of behavior, nothing else.

It's meant to give you a fast scan of:

- tool calls in the order they happened
- compact summaries of arguments and return values
- evidence tags, when normalized framework evidence is available
- agent swimlanes, if multiple agents or sub-agents were observed
- handoff arrows, but only when normalized handoff records actually exist
- the final answer

Right now the renderer only goes top to bottom. It uses a single global execution timeline across every lane, so vertical position reflects the overall run order rather than each lane progressing independently.

## Quick start: run the bundled demo

GroundEval ships with a full demo, so you can see the CLI, an agent trajectory, deterministic scoring, and a generated report without writing your own config or artifacts first.

Before running it, set up a model provider.

For Anthropic:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

For OpenAI:

```bash
export OPENAI_API_KEY=your_key_here
```

If you're using an OpenAI-compatible endpoint, also set:

```bash
export OPENAI_BASE_URL=your_url_here
```

Then run:

```bash
uv sync --group dev
uv run python -m groundeval task --config config.yaml
```

By default, the demo uses whatever provider and model are declared in `config.yaml`.

To override the model from the command line:

```bash
uv run python -m groundeval task --config config.yaml --model gpt-4o
```

The demo itself runs a sales-outreach verification task. The agent searches the bundled evidence corpus, retrieves artifacts, submits a structured answer, and GroundEval scores the whole run across Counterfactual, Silence, and Perspective.

Results land in:

```bash
eval_output/
```

You should see terminal output that looks something like this:

```text
Task results written to eval_output/task_results_<model>.json
Overall - counterfactual=0.833  silence=0.750  perspective=1.000  overall=0.861  accuracy=1.000
Total violations: 0
```

The JSON report gives you per-task scores, precondition-level results, violation counts, the full tool trajectory, the submitted answer, and the diagnostic details behind all of it.

### Validate the demo config without running the agent

```bash
uv run python -m groundeval validate --config config.yaml
```

This checks that the config is well-formed, that any required artifacts are actually present, and that the task contracts are valid, all before you spend a single API credit.

## What the demo includes

Everything you need for a first run is already bundled in:

- a sample task contract
- seed artifacts
- role and subsystem permissions
- provider settings
- a built-in agent loop
- deterministic scoring across all three tracks

The demo is small on purpose. It's there to show you how the CLI behaves, what a trajectory actually looks like, and what kind of report comes out the other end. You don't need to mirror its structure exactly when you build your own evaluations.

## Configuring your own evaluation

Use Observe Mode to draft your first config, then move into `docs/` for deeper configuration as you need it.

Good places to start:

- [`docs/crewai.md`](docs/crewai.md) for CrewAI integration
- [`docs/maf.md`](docs/maf.md) for Microsoft Agent Framework integration
- `docs/` more broadly, for task contracts, artifacts, access policy, providers, and custom runners

At a high level, a real evaluation needs to define:

- what task the agent is actually trying to complete
- what preconditions have to be verified before it acts
- what evidence it's allowed to access
- where ground truth comes from
- how the agent runner gets invoked
- what structured answer the scorer should expect back

Your artifacts don't need to look anything like the demo's. They can represent tickets, claims, alerts, contracts, medical orders, Slack messages, audit records, database rows, GitHub issues, notes, game state, home-lab logs, or really any other state your agent has to reason over.

## Bringing your own agent

GroundEval comes with a built-in Anthropic/OpenAI agent loop so the demo can run out of the box.

For anything closer to production, use Observe Mode or wire GroundEval up to your own runner. Either way, the scoring pipeline underneath stays the same: GroundEval records the trajectory, checks access and evidence, and scores the result deterministically.

The integration guides in `docs/` walk through framework-specific setup.

## Supported agent frameworks

GroundEval's observer interface is built around framework adapters. Each adapter loads the agent, records the observed run, and hands the result off to the same draft and scoring pipeline everything else uses.

### CrewAI

```bash
uv sync --group dev --group crewai
```

[See the CrewAI integration guide →](docs/crewai.md)

### LangGraph

LangGraph observation and deterministic scoring are implemented in code. Framework-specific artifact details should live in the framework docs.

### Microsoft Agent Framework

```bash
uv sync --group dev --group maf
```

MAF observe mode runs the agent as it normally would, captures native OpenTelemetry spans in-process, normalizes what it observed, and supports both observing and scoring against a reviewed contract.

[See the MAF integration guide →](docs/maf.md)

## Extension points

GroundEval is built so you can swap out individual parts without having to rebuild the whole engine.

| What you might replace | When you'd do it |
|---|---|
| Artifacts | You already know your domain's preconditions and want to evaluate against your own state |
| `CorpusAdapter` | Your artifacts live in MongoDB, Elasticsearch, Postgres, S3, or some proprietary backend |
| `AccessPolicy` | Your visibility rules are scoped by tenant, project, account, or time |
| Agent runner | You're using a specific model provider or agent framework |
| Task contracts | Your domain has its own precondition verification requirements |

## What this is not

GroundEval isn't built for grading whether your agent sounds good. If you care about tone, style, or how natural a conversation feels, an LLM judge is the right tool for that job. GroundEval is for the parts where there's an actual right answer: did the agent check the right things, use the right evidence, and stay inside the permissions it was given.

## Preprint implementation

The implementation behind the initial GroundEval preprint is preserved at:

- Branch: `paper/preprint-2026`
- Tag: `groundeval-preprint-v1`

The `main` branch holds the simplified public version, meant to be easier to adopt.

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
