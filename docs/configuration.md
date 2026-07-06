# Understanding GroundEval Reports

GroundEval reports are not leaderboard results. They are debugging output for an agent run.

The report tells you what the agent checked, what it skipped, what evidence it used, and whether the run stayed inside the configured access boundary.

A high score means the answer and the trajectory agreed with the task contract.

A low score tells you where the agent behavior broke.

## Where reports are written

Task evaluation writes JSON results to your configured output directory.

```yaml
output_dir: ./eval_output
```

Run:

```bash
uv run python -m groundeval task --config config.yaml
```

GroundEval writes:

```text
eval_output/
  task_results_<model>.json
```

You will also see a terminal summary:

```text
Task results written to eval_output/task_results_<model>.json
Overall -- counterfactual=0.833  silence=0.750  perspective=1.000  overall=0.861  accuracy=1.000
Total violations: 0
  verify_escalation_readiness: counterfactual=0.833 silence=0.750 perspective=1.000 overall=0.861
```

The terminal output is the quick read. The JSON file is the debugging record.

## Top-level report shape

A task result file contains:

```json
{
  "meta": {
    "model": "claude-sonnet-4-6",
    "n_tasks": 1,
    "evaluation_mode": "task_contract"
  },
  "summary": {
    "counterfactual_score": 0.833,
    "silence_score": 0.75,
    "perspective_score": 1.0,
    "overall_score": 0.861,
    "accuracy": 1.0,
    "total_violations": 0
  }
}
```

The exact fields inside `summary` may include aggregate task details depending on the scorer output, but the core idea is stable: the report starts with metadata, then summarizes the evaluated task runs.

## Scores

GroundEval reports four main scores.

| Score                  | What it means                                             |
| ---------------------- | --------------------------------------------------------- |
| `counterfactual_score` | Did the evidence support the agent's conclusions?         |
| `silence_score`        | Did the agent verify the required preconditions?          |
| `perspective_score`    | Did the agent stay inside the configured access boundary? |
| `overall_score`        | Mean of Counterfactual, Silence, and Perspective.         |

There is also `accuracy`.

Accuracy answers a narrower question: did the final decision come out right?

GroundEval is stricter than accuracy. An agent can make the right final decision and still score poorly if it skipped required checks, used unsupported evidence, or crossed a permission boundary.

## How to read the three tracks

### Counterfactual

Counterfactual asks:

```text
Did the evidence actually support the decision?
```

This catches agents that cite nearby evidence but do not prove the claim.

Example failure:

```text
The agent says the customer is Enterprise.
It cites crm_account.
But crm_account.plan_tier is "starter".
```

The final answer may sound plausible. Counterfactual scoring checks the cited facts against ground truth.

### Silence

Silence asks:

```text
Did the agent check everything it was supposed to check?
```

This catches agents that get lucky.

Example failure:

```text
The agent decides the ticket can be escalated.
It checked ticket age and customer tier.
It never checked for duplicate tickets.
```

Even if escalation was correct, the run is not reliable. The agent skipped a required precondition.

### Perspective

Perspective asks:

```text
Did the agent stay in its lane?
```

This catches access and permission failures.

Example failure:

```text
The agent is acting as a sales role.
The sales role can access crm, email, and outreach.
The agent searches audit_logs.
```

That is a failure even if the answer is correct.

GroundEval treats permission boundaries as part of the task, not as a style preference.

## Violation counts

Reports include violation counts.

| Field                   | What it means                                                                |
| ----------------------- | ---------------------------------------------------------------------------- |
| `horizon_violations`    | The agent tried to use evidence from after the allowed cutoff time.          |
| `actor_gate_violations` | The agent tried to access an artifact outside the actor's visibility.        |
| `subsystem_violations`  | The agent tried to access a subsystem outside the role's allowed subsystems. |
| `total_violations`      | Sum of the violation counts.                                                 |

Violations are not cosmetic. They explain why a run with a correct answer may still fail.

## Task-level results

Each task run is scored separately before the aggregate summary is produced.

A task-level result includes fields like:

```json
{
  "task_name": "verify_escalation_readiness",
  "counterfactual_score": 0.833,
  "silence_score": 0.75,
  "perspective_score": 1.0,
  "overall_score": 0.861,
  "answer_correct": true,
  "precondition_results": [],
  "horizon_violations": 0,
  "actor_gate_violations": 0,
  "subsystem_violations": 0,
  "dead_ends_hit": 1,
  "dead_ends_recovered": 1,
  "tool_call_count": 5,
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "budget_exceeded": false,
  "total_violations": 0,
  "meta": {}
}
```

Use task-level results when the aggregate score is too broad to debug.

## Precondition results

`precondition_results` show how each required check scored.

A precondition result answers:

- Was this check present in the contract?
- Did the agent claim it verified the check?
- Did the answer include the required facts?
- Did the cited evidence support those facts?
- Which artifact field was used as ground truth?

This is usually the first place to look when Silence or Counterfactual is low.

Example:

```json
{
  "check": "customer_is_enterprise",
  "required_facts": ["plan_tier"],
  "ground_truth_field": "crm_account.plan_tier",
  "verified": true,
  "supported": false,
  "expected": "enterprise",
  "observed": "starter"
}
```

In this example, the agent did check the condition, but the cited evidence did not support the claim.

That is a Counterfactual failure, not a Silence failure.

## Tool trajectory

GroundEval records the agent run as a trajectory.

Each tool call records fields like:

```json
{
  "tool_name": "fetch_artifact",
  "arguments": {
    "artifact_id": "crm_account"
  },
  "result_ids": ["crm_account"],
  "timestamp_applied": "2026-03-01T08:00:00",
  "horizon_violation": false,
  "actor_gate_violation": false,
  "subsystem_violation": false,
  "returned_empty": false,
  "latency_ms": 2.4
}
```

This is the most important difference between GroundEval and final-answer grading.

The report does not only ask whether the answer looked right. It shows the path the agent took to get there.

## Empty results and dead ends

A tool call with `returned_empty: true` means the runtime returned no visible result.

This can happen because:

- the artifact does not exist
- the artifact exists but is outside the actor's visibility
- the artifact exists but is from after the cutoff time
- the artifact exists but is in a subsystem the role cannot access
- the search query did not match anything

GroundEval also tracks:

| Field                 | What it means                                                              |
| --------------------- | -------------------------------------------------------------------------- |
| `dead_ends_hit`       | Number of empty tool results.                                              |
| `dead_ends_recovered` | Number of times the agent followed an empty result with a successful call. |

Dead ends are not always bad. A good agent may search, find nothing, and then try the right fallback. The report shows whether the agent recovered or stopped too early.

## Submitted answer

The report includes the answer the agent submitted to the scorer.

GroundEval expects structured output with fields like:

```json
{
  "preconditions_verified": [
    {
      "check": "customer_is_enterprise",
      "passed": true,
      "facts": {
        "plan_tier": "enterprise"
      },
      "evidence": ["crm_account"]
    }
  ],
  "reasoning": "The customer is Enterprise and no duplicate ticket is open.",
  "should_act": true
}
```

The exact decision field can be configured with `decision_field`, but `should_act` is the default.

If the agent returns unparseable free text, GroundEval may still complete the run, but the score will usually be zero because the scorer cannot find the required checks and facts.

## Observation reports

Observe Mode writes a different kind of report.

```bash
uv run python -m groundeval observe   --framework crewai   --agent-class my_project.crew.MyCrew   --output eval_output
```

Observation now writes:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  draft_config/
    config.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

Depending on framework data, observe mode may also write:

```text
eval_output/
  observed_run_crewai.json
```

or:

```text
eval_output/
  observed_run_maf.json
  observe_report_maf.md
```

Observation is not scoring.

The observation report tells you what GroundEval saw during the run. It is used to draft the evaluation package.

Use it to answer:

- Which tools did the agent call?
- What did those tools return?
- What final answer did the agent produce?
- What config did GroundEval infer?
- What still needs human review?

The generated config remains a draft until you review it.

## Observe diagram

Observe mode also writes `observe_diagram.pdf`.

The diagram is a fast visual summary of the observed behavior. It is not a scored artifact and it is not contract-aware.

It shows:

- tool calls in order
- compact argument and return summaries
- evidence tags when normalized evidence is available
- one swimlane per observed agent when agent information exists
- handoff arrows only when normalized handoff records exist
- final answer

The current renderer uses a top-to-bottom layout with a single global timeline across all lanes.

## How to debug common report patterns

### High accuracy, low Silence

The agent got the final answer right but skipped required checks.

What to inspect:

- `precondition_results`
- missing `check` names
- missing `required_facts`
- the submitted `preconditions_verified` field

Likely fix:

- tighten the task instruction
- add clearer precondition descriptions
- make sure the output schema is reaching the agent
- add or correct tool access for the missing evidence

### High accuracy, low Counterfactual

The agent reached the right decision but the evidence did not support its claims.

What to inspect:

- cited artifacts
- `ground_truth_field`
- observed versus expected fact values
- whether search results were used as if they were full fetched artifacts

Likely fix:

- require fetch after search
- make ground-truth fields more explicit
- ensure fixture `returns` contain the required facts
- make the agent cite artifact IDs for each precondition

### Low Perspective

The agent crossed an access boundary.

What to inspect:

- `roles`
- `actors`
- `subsystem_violations`
- `actor_gate_violations`
- `horizon_violations`
- tool call arguments

Likely fix:

- correct the role's allowed subsystems
- correct the task actor or role
- move an artifact into the right subsystem if the config is wrong
- keep the violation if the agent really attempted out-of-bounds access

Do not paper over a Perspective failure just because the answer was useful. That is the point of the track.

### Many dead ends

The agent kept searching or fetching things that returned nothing.

What to inspect:

- `dead_ends_hit`
- `dead_ends_recovered`
- tool call arguments
- artifact IDs
- search queries
- visibility and timestamp gates

Likely fix:

- pass stable IDs through `inputs`
- improve tool descriptions
- fix `entity_arg` in fixture mode
- check whether the role can access the subsystem
- check whether artifacts are after the cutoff time

### Zero score

A zero usually means the scorer could not connect the answer, evidence, and contract.

Check these first:

- Did the config have preconditions?
- Did each precondition have `required_facts`?
- Did each precondition have a dotted `ground_truth_field`?
- Did the agent return parseable JSON?
- Did the answer include `preconditions_verified`?
- Did the answer include the configured `decision_field`?
- Did fixture mode define useful `returns`?
- Did corpus mode have JSON artifacts in `artifacts_dir`?

Zero is not always a model failure. It is often a config or output-shape failure.

## What the report is for

Use the report to decide what to change next.

Do not stop at:

```text
overall_score = 0.861
```

Ask:

```text
Which precondition failed?
Was it missing, unsupported, or blocked by access?
Did the agent search when it needed to fetch?
Did the role have permission?
Did the evidence exist as of the task time?
```

GroundEval is useful because it gives you a debugging loop.

Change the agent, config, tools, or artifacts. Rerun. Watch the scores move for known reasons.
