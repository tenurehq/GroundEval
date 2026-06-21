# GroundEval

**Deterministic evaluation for agents that reason over state.**

GroundEval scores LLM agents not just on whether the answer is right, but on whether they reached it through valid evidence. If your agent retrieves from memory, calls tools over documents, respects permissions, or reasons about time, you need to know whether it followed a valid path, not just whether the final answer sounded plausible.

## Why not an LLM judge?

LLM judges can tell you if an answer looks reasonable. They cannot tell you if the agent cited a document it shouldn't have had access to, used information from after the question's cutoff time, skipped the search required to prove absence, or mistook correlation for causation.

GroundEval answers these questions deterministically. The ground truth comes from state: event logs, artifact corpora, and access rules, not from another model's judgment.

## What you need

GroundEval separates what you bring from what the framework does.

**You bring:**

- An **event log**: a JSONL file of timestamped events with actors, artifact IDs, and facts. Generate one with an LLM in minutes for a synthetic scenario, or use your own operational log.
- An **artifact corpus**: the documents, tickets, messages, or records your agent retrieves from. Drop JSON files in a directory, or implement a thin adapter for your existing backend.
- A **config**: 40–60 lines of YAML declaring a few synthetic actors, their roles, and 2–4 causal links or silence pairs that define correctness for your domain.

**The framework handles the rest:** question generation, gated tool access, trajectory recording, deterministic scoring, and per-question failure diagnostics.

## Quick start

### 1. Generate some events and artifacts (or use your own)

The event log format is simple JSONL. Here's a minimal example:

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

### 2. Write a config

#### Example 1: Solo dev evaluating a memory agent

This is for someone who built a memory agent over a weekend and wants to know if it actually retrieves correctly.

They start with nothing. They ask an LLM for synthetic data:

> "Generate a 30-event JSONL log for two actors (maya and dev) over one week of a personal knowledge base app. Events should include `note_created`, `note_updated`, `note_tagged`, and `memory_queried`. Include artifact IDs pointing into a markdown notes directory. Make the timeline realistic."

The LLM produces `events.jsonl` and a handful of markdown artifacts. They drop them in `artifacts/`.

Their config is 30 lines:

```yaml
output_dir: ./eval_output
artifacts_dir: ./artifacts
use_event_log_policy: true

actors:
  maya: user
  dev: maintainer

roles:
  user:
    subsystems: [notes]
  maintainer:
    subsystems: [notes, audit]
    broadcast_event_types: [note_tagged]

causal_links:
  - name: tagging_improves_retrieval
    cause_event_type: note_tagged
    effect_event_type: memory_queried
    premise_template: "the note was tagged correctly"
    outcome_template: "the tagged note would appear in query results"
    outcome_changed: true
    max_gap_days: 3
    join:
      - cause: artifact_ids.note_id
        effect: artifact_ids.note_id

silence_pairs:
  - trigger_event_type: note_updated
    response_event_type: memory_queried
    max_gap_days: 2
    search_space_subsystems: [notes]
    search_space:
      - subsystem: notes
        query_template: "{artifact_ids.note_id}"
    join:
      - cause: artifact_ids.note_id
        effect: artifact_ids.note_id
```

They run two commands. In 10 minutes they have per-question scores telling them exactly which retrievals failed and why. They fix their memory agent's tag resolution, re-run, and see the trajectory score improve. No enterprise infrastructure. No org chart. Two synthetic users and one subsystem.

#### Example 2: Enterprise team evaluating a support agent

This is the same framework, different scale. The team already has an operational event log (Zendesk, Jira, Slack, Confluence), an artifact corpus, and role-based permissions.

Their config declares their actual actors, roles, subsystems, and the correctness contracts for their support workflow. The structure is identical to Example 1 — actors, roles, causal links, silence pairs — but populated from production data rather than synthetic prompts.

```yaml
output_dir: ./eval_output
artifacts_dir: ./artifacts
use_event_log_policy: true

actors:
  alice: engineer
  bob: sales
  carol: support_lead
  dave: support_agent

roles:
  engineer:
    subsystems: [jira, git, slack, confluence]
    broadcast_event_types: [incident_opened, incident_resolved]
  sales:
    subsystems: [salesforce, slack]
  support_lead:
    subsystems: [zendesk, slack, confluence, jira]
    broadcast_event_types: [zd_ticket_opened, zd_tickets_escalated]
  support_agent:
    subsystems: [zendesk, confluence]
    broadcast_event_types: [zd_ticket_opened]

perspective:
  positive_ratio: 0.5
  negative_permission_ratio: 0.25
  negative_temporal_ratio: 0.25

causal_links:
  - name: escalation_caused_postmortem
    cause_event_type: escalation_opened
    effect_event_type: postmortem_created
    premise_template: "the escalation had been investigated"
    outcome_template: "the postmortem would have been written"
    outcome_changed: true
    max_gap_days: 7
    join:
      - cause: artifact_ids.jira
        effect: artifact_ids.jira

silence_pairs:
  - trigger_event_type: escalation_opened
    response_event_type: postmortem_created
    max_gap_days: 7
    search_space_subsystems: [confluence, jira, slack]
    search_space:
      - subsystem: confluence
        query_template: "postmortem {artifact_ids.jira}"
      - subsystem: jira
        id_template: "{artifact_ids.jira}"
      - subsystem: slack
        query_template: "{artifact_ids.jira} escalation"
    join:
      - cause: artifact_ids.jira
        effect: artifact_ids.jira
```

The framework doesn't care whether the data came from an LLM prompt or from a production Kafka stream. The evaluation pipeline is the same: generate questions, run the agent, score answers and trajectories, inspect failure reasons, fix the agent, repeat.

The enterprise team runs the same two commands the solo dev ran. They just have a bigger config and a real artifact backend.

### 3. Generate questions and run evaluation

```bash
uv sync --group dev
uv run python -m groundeval generate --config config.yaml --events events.jsonl
uv run python -m groundeval eval --config config.yaml --questions eval_output/eval_questions.json --events events.jsonl
```

You'll get per-question scores with failure reasons, an aggregate summary, and trajectory diagnostics. Fix your agent, run again, compare.

## Three evaluation tracks

GroundEval ships with three tracks. Each tests a different kind of reasoning about state.

### PERSPECTIVE: Could the actor have known?

Tests whether the agent respects what a specific actor could see at a specific time. Catches failures like using artifacts outside the actor's visibility cone, using a subsystem their role cannot access, or using information created after the question's cutoff.

> *Based only on what Morgan could access as of March 5, could Morgan have known that Acme was at churn risk?*

### COUNTERFACTUAL: Did X cause Y?

Tests whether the agent identifies the correct cause-and-effect relationship from the event log. Causal links are declared in the config with join conditions (shared ticket IDs, account IDs) so they don't rely on temporal adjacency alone.

> *If the incident on March 5 had been resolved earlier, would the postmortem still have been created by the same person?*

### SILENCE: Did the agent prove absence?

Tests whether the agent searched the required places before concluding something did not happen. A correct "no" isn't enough — the trajectory must show the expected search space was covered.

> *Was a postmortem documented for incident ESC-42?*

## What gets scored

Each question produces two scores:

| Score | What it measures |
|---|---|
| `answer_score` | Did the final structured answer match ground truth? |
| `trajectory_score` | Did the agent follow a valid evidence path? |

The trajectory score checks different things depending on the track: subsystem coverage for SILENCE, visibility-cone discipline for PERSPECTIVE, causal mechanism identification for COUNTERFACTUAL. The combined score weights trajectory more heavily for tracks where the path is the point.

## Tool mode and context mode

GroundEval supports two agent architectures.

**Tool mode**: the framework creates a gated runtime. Your agent calls `runtime.fetch()` and `runtime.search()`. The runtime records every call, enforces visibility and temporal gates, and the trajectory scorer checks whether the trace was valid.

**Context mode** (`--context-injection`): the framework packs relevant artifacts into the context window. Your agent answers from context without tool calls. The scorer checks citation discipline — did the agent cite the right artifacts and avoid citing irrelevant ones?

Wire your agent by replacing `_build_agent_fn` in `groundeval/run.py`. The expected signature takes a question, context, tools, max_steps, and optional runtime, and returns a trajectory plus answer dict.

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

- **Ground truth comes from state, not an LLM judge.** Answer keys are derived from the event log, artifact corpus, and access policy. Question prose may be generated by an LLM, but the scoring path is deterministic.
- **A correct answer through an invalid trajectory is a failure.** The framework penalizes agents that reach the right conclusion through the wrong evidence.
- **The config declares domain truth; the framework handles mechanics.** You write YAML describing what correctness looks like in your domain. The engine handles generation, gating, recording, and scoring.
- **Start synthetic, graduate to production.** You can run a full evaluation against generated data in under 15 minutes. When you're ready, swap in your real event log, artifacts, and access policy. The evaluation pipeline doesn't change.

## What this is not

GroundEval does not replace human or model judgment for subjective quality — tone, style, persuasiveness, conversational fluency. It is for cases where correctness can be verified from state, evidence, permissions, and tool traces. Use it alongside, not instead of, qualitative evaluation.

## Installation

Python 3.13+ with `uv`:

```bash
uv sync --group dev
```

## Status

This is an early framework. The core abstractions are in place. Production use should add provider-backed agent runners, generation diagnostics, reproducible seeds, and richer examples.