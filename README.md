# GroundEval

**Deterministic evaluation for agents that reason over state.**

You built an agent that searches your docs, respects permissions, and reasons about what happened and when. How do you know it's actually doing those things and not just sounding convincing?

That's the problem GroundEval solves. It scores LLM agents not just on whether the answer is right, but on whether they reached it through valid evidence. If your agent retrieves from memory, calls tools over documents, respects permissions, or reasons about time, you need to know whether it followed a valid path, not just whether the final answer sounded plausible.

## The problem with LLM judges

An LLM judge can tell you if an answer looks reasonable. It cannot tell you if the agent:

* cited a document it shouldn't have had access to
* used information from after the question's cutoff time
* skipped the search required to prove something didn't happen
* mistook correlation for causation because two events happened close together

GroundEval answers these questions deterministically. The ground truth comes from state: event logs, artifact corpora, and access rules. Not from another model's judgment.

## Three tracks, three questions

GroundEval ships with three evaluation tracks. Each tests a different kind of reasoning about state.

**PERSPECTIVE: Could the actor have known?** Tests whether the agent respects what a specific actor could see at a specific time. Catches failures like using artifacts outside the actor's visibility cone, using a subsystem their role cannot access, or using information created after the question's cutoff.

> *Based only on what Morgan could access as of March 5, could Morgan have known that Acme was at churn risk?*

**COUNTERFACTUAL: Did X cause Y?** Tests whether the agent identifies the correct cause-and-effect relationship from the event log. Causal links are declared in the config with join conditions (shared ticket IDs, account IDs) so they don't rely on temporal adjacency alone. Two things happening close together isn't causation, and the scorer knows it.

> *If the incident on March 5 had been resolved earlier, would the postmortem still have been created by the same person?*

**SILENCE: Did the agent prove absence?** Tests whether the agent searched the required places before concluding something did not happen. A correct "no" isn't enough. The trajectory must show the expected search space was covered. If your team writes postmortems in Confluence and the agent only checked Jira, the answer is invalid even if no postmortem actually exists.

> *Was a postmortem documented for incident ESC-42?*

## Start with the walkthroughs

The README gives you the framework. These short essays show how to think about each failure mode before you wire GroundEval into your own agent.

* **[How to test whether an agent checked before saying "no"](https://tenureai.dev/writing/how-to-test-agent-checked-before-saying-no/)**
  Start here if your agent answers absence questions: no postmortem, no follow-up, no escalation, no record found. This walks through the Silence track and why a correct "no" still fails if the agent did not search the required places.

* **[How to test what an AI agent was allowed to know](https://tenureai.dev/writing/how-to-test-agent-perspective/)**
  Read this if your agent works across tools with role boundaries. This walks through the Perspective track and the difference between relevant evidence and permissible evidence.

* **How to test whether an agent found the real cause**
  Read this if your agent explains why something happened. This walks through the Counterfactual track and why temporal adjacency is not enough to prove causation.


## What gets scored

Each question produces two scores:

| Score | What it measures |
|---|---|
| `answer_score` | Did the final structured answer match ground truth? |
| `trajectory_score` | Did the agent follow a valid evidence path? |

The trajectory score checks different things depending on the track: subsystem coverage for SILENCE, visibility-cone discipline for PERSPECTIVE, causal mechanism identification for COUNTERFACTUAL. The combined score weights trajectory more heavily for tracks where the path is the point. A correct answer through an invalid trajectory is a failure.

Every evaluated question also includes a structured diagnostic trace. The trace records the agent run as data: tool calls, tool results, submitted answers, errors, and optional agent messages emitted between steps. Diagnostics are not used to award credit. The scorer still relies only on the deterministic trajectory. The diagnostic trace exists so you can debug why the score happened, without reconstructing it from interleaved logs.

## What you bring, and what the framework handles

GroundEval separates what you bring from what the framework does.

**You bring:**

* An **event log**: a JSONL file of timestamped events with actors, artifact IDs, and facts. Generate one with an LLM in minutes for a synthetic scenario, or bring your own operational log.
* An **artifact corpus**: the documents, tickets, messages, or records your agent retrieves from. Drop JSON files in a directory, or implement a thin adapter for your existing backend.
* A **config**: around fifty lines of YAML declaring a few synthetic actors, their roles, and a handful of causal links or silence pairs that define correctness for your domain.

**The framework handles the rest:** question generation, gated tool access, trajectory recording, deterministic scoring, and per-question failure diagnostics.

## Quick start

Generate some events and artifacts, or use your own. The event log format is simple JSONL. Here's a minimal example:

```jsonl
{"id":"evt-001","type":"incident_opened","timestamp":"2026-03-05T10:00:00","actors":["alice"],"artifact_ids":{"jira":"ESC-42"},"facts":{"severity":"high"}}
{"id":"evt-002","type":"postmortem_created","timestamp":"2026-03-07T14:30:00","actors":["alice","bob"],"artifact_ids":{"jira":"ESC-42","confluence":"CONF-9"},"facts":{}}
```

Artifacts are JSON files in a directory:

```json
{
  "id": "ESC-42",
  "timestamp": "2026-03-05T10:00:00",
  "subsystem": "jira",
  "title": "Customer escalation for Acme",
  "body": "..."
}
```

If you don't have your own data yet, ask an LLM: "Generate a 40-event log and matching artifacts for a customer-support domain with three actors over two weeks in March 2026." Drop the outputs into `events.jsonl` and `artifacts/`. You're ready.

Then generate questions and run evaluation:

```bash
uv sync --group dev
uv run python -m groundeval generate --config config/config.yaml --events events.jsonl
uv run python -m groundeval eval --config config/config.yaml --questions eval_output/eval_questions.json --events events.jsonl
```

You'll get per-question scores with failure reasons, an aggregate summary, and trajectory diagnostics. Fix your agent, run again, compare.

## Tool mode and context mode

GroundEval supports two agent architectures.

**Tool mode**: the framework creates a gated runtime. Your agent calls `runtime.fetch()` and `runtime.search()`. The runtime records every call, enforces visibility and temporal gates, and the trajectory scorer checks whether the trace was valid.

**Context mode** (`--context-injection`): the framework packs relevant artifacts into the context window. Your agent answers from context without tool calls. The scorer checks citation discipline: did the agent cite the right artifacts and avoid citing irrelevant ones?

Wire your agent by replacing `_build_agent_fn` in `groundeval/run.py`. The expected signature takes a question, context, tools, max_steps, and optional runtime, and returns a trajectory plus answer dict.

## Example domains

The `examples/` directory contains five ready-to-run evaluation scenarios. Each one is a complete domain: a config, an event log, and an artifact corpus. Copy any folder and you can generate questions and run an eval in under ten minutes.

| Domain | Actors | What it tests |
|---|---|---|
| **enterprise-support** | 5 (engineer, sales, support lead, support agent) | Role-based access across Zendesk, Jira, Confluence, Slack, Salesforce, and Git. Escalation causality, postmortem silence gaps, churn detection chains |
| **cybersecurity** | 5 (L1 analyst, L2 analyst, incident responder, security engineer, threat hunter) | Tiered SOC access control. Attack chain causality (phishing, credential theft, lateral movement, ransomware). Search discipline across Splunk, CrowdStrike, Jira, and Confluence |
| **healthcare** | 6 (physician, nurse, pharmacist, billing specialist, care coordinator, patient advocate) | HIPAA-style access boundaries: billing can't see clinical notes, pharmacist can't see imaging, advocate can't see billing claims. Medication error causality, lab-to-treatment chains, discharge follow-up gaps |
| **finance** | 5 (applicant, loan officer, underwriter, fraud analyst, compliance reviewer) | Temporal cutoff discipline (what was known at decision time). Role-based access to credit reports, fraud alerts, and underwriting notes. Regulatory silence checks for adverse action notices |
| **legal** | 6 (associate, partner, client, opposing counsel, paralegal, compliance reviewer) | Privilege boundaries: opposing counsel cannot see matter notes or privileged docs. Citation discipline. Version-tracking across contract redlines. DPA and filing compliance gaps |

Each domain exercises all three tracks.

### Using an example

```bash
# 1. Copy a domain
cp -r examples/healthcare my-eval/
cd my-eval/

# 2. Generate questions
uv run python -m groundeval generate --config config.yaml --events events.jsonl

# 3. Run evaluation
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --model claude-sonnet-4-6
```

### Creating your own

Each domain needs exactly three things, and the examples show you the pattern:

1. **config.yaml** (around 50 lines) declaring actors, roles, subsystems, causal links, and silence pairs
2. **events.jsonl** (around 40 timestamped events with actor and artifact references)
3. **artifacts/** (JSON files your agent retrieves, one per artifact ID)

If you don't have your own data yet, ask an LLM:

> "Generate a 40-event JSONL log and matching artifacts for a [your domain] with 4 actors over 2 weeks. Include a mix of causal chains and silent gaps."

Drop the output in your folder and you're ready. The framework doesn't care whether the data came from an LLM prompt or a production event stream. The evaluation pipeline is the same.

## Extension points

The framework is designed so you swap parts without rebuilding the engine.

| What you might replace | When you'd do it |
|---|---|
| Event log and artifacts | You're testing your own domain instead of a synthetic scenario |
| `CorpusAdapter` | Your artifacts live in MongoDB, Elasticsearch, Postgres, S3, or a proprietary backend |
| `AccessPolicy` | Your visibility rules are tenant-scoped, project-scoped, or account-scoped |
| Agent runner | You're using a specific model provider or agent framework |
| Eval questions | You already have a benchmark and want deterministic trajectory scoring |
| Causal links and silence pairs | Your domain has different correctness contracts |

## Design principles

* **Ground truth comes from state, not an LLM judge.** Answer keys are derived from the event log, artifact corpus, and access policy. Question prose may be generated by an LLM, but the scoring path is deterministic.
* **A correct answer through an invalid trajectory is a failure.** The framework penalizes agents that reach the right conclusion through the wrong evidence.
* **The config declares domain truth; the framework handles mechanics.** You write YAML describing what correctness looks like in your domain. The engine handles generation, gating, recording, and scoring.
* **Start synthetic, graduate to production.** You can run a full evaluation against generated data in under 15 minutes. When you're ready, swap in your real event log, artifacts, and access policy. The evaluation pipeline doesn't change.

## What this is not

GroundEval does not replace human or model judgment for subjective quality: tone, style, persuasiveness, conversational fluency. It is for cases where correctness can be verified from state, evidence, permissions, and tool traces.

## Citation

If you use this work, please cite:

```bibtex
@article{Jeffrey_Flynt_GroundEval_A_Deterministic_2026,
author = {Jeffrey Flynt, Jeffrey Flynt},
journal = {arXiv preprint},
title = {{GroundEval: A Deterministic Replacement for LLM-as-Judge in Stateful Agent Evaluation}},
url = {https://arxiv.org/abs/2606.22737},
year = {2026}
}
```
