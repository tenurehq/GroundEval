# Understanding GroundEval Reports

GroundEval reports are not leaderboard results. They are debugging output for an agent run.

The report tells you what GroundEval scored, which track failed, which preconditions were supported, and whether the run crossed access boundaries.

A high score means the answer and the trajectory agreed with the task contract.

A low score tells you where the agent behavior broke.

## Where task reports are written

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

The terminal output is the quick read. The JSON file is the structured result.

## Top-level task report shape

A task result file contains:

```json
{
  "meta": {
    "model": "claude-sonnet-4-6",
    "n_tasks": 1,
    "evaluation_mode": "task_contract"
  },
  "summary": {
    "n_tasks": 1,
    "counterfactual_score": 0.833,
    "silence_score": 0.75,
    "perspective_score": 1.0,
    "overall_score": 0.861,
    "accuracy": 1.0,
    "total_violations": 0,
    "per_task": []
  }
}
```

`meta` describes the run.

`summary` contains aggregate scores and the per-task results.

## Scores

GroundEval reports four main scores.

| Score                  | What it means                                             |
| ---------------------- | --------------------------------------------------------- |
| `counterfactual_score` | Did the evidence support the agent's conclusions?         |
| `silence_score`        | Did the agent verify the required preconditions?          |
| `perspective_score`    | Did the agent stay inside the configured access boundary? |
| `overall_score`        | Mean of Counterfactual, Silence, and Perspective.         |

There is also `accuracy`.

Accuracy answers a narrower question: did the answer meet the scorer's correctness threshold?

GroundEval is stricter than accuracy. An agent can make the right final decision and still score poorly if it skipped required checks, used unsupported evidence, or crossed a permission boundary.

## How the three tracks are combined

Each track has an answer component and a trajectory component.

| Track          | Answer weight | Trajectory weight |
| -------------- | ------------: | ----------------: |
| Counterfactual |          0.50 |              0.50 |
| Silence        |          0.30 |              0.70 |
| Perspective    |          0.40 |              0.60 |

This is why a correct-looking answer can still score poorly. GroundEval does not only inspect what the agent said. It also scores the path it took.

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

## Aggregate summary fields

The aggregate `summary` contains:

| Field                  | What it means                                         |
| ---------------------- | ----------------------------------------------------- |
| `n_tasks`              | Number of task contracts evaluated.                   |
| `counterfactual_score` | Average Counterfactual score across tasks.            |
| `silence_score`        | Average Silence score across tasks.                   |
| `perspective_score`    | Average Perspective score across tasks.               |
| `overall_score`        | Average overall score across tasks.                   |
| `accuracy`             | Share of tasks marked answer-correct by the scorer.   |
| `total_violations`     | Sum of horizon, actor gate, and subsystem violations. |
| `per_task`             | Task-level result objects.                            |

Use the aggregate summary to spot the broad failure mode.

Use `per_task` to debug a specific task.

## Task-level result fields

Each task result can include:

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
  "meta": {
    "cf_answer": 0.75,
    "cf_trajectory": 0.9167,
    "sl_answer": 0.6667,
    "sl_trajectory": 0.7857,
    "ps_answer": 1.0,
    "ps_trajectory": 1.0
  },
  "total_violations": 0
}
```

The exact numbers will depend on your task and trajectory.

## Precondition results

`precondition_results` show whether each required check was supported by evidence.

These results currently come from the Counterfactual scorer because they are the richest precondition-level details.

Example supported precondition:

```json
{
  "check": "customer_is_enterprise",
  "evidence_supported": true,
  "score": 1.0,
  "evidence_cited": ["crm_account"],
  "agent_claimed": true
}
```

Example unsupported precondition:

```json
{
  "check": "customer_is_enterprise",
  "evidence_supported": false,
  "score": 0.0,
  "error": "no evidence artifacts cited",
  "agent_claimed": true
}
```

Example omitted precondition:

```json
{
  "check": "no_open_duplicate",
  "evidence_supported": false,
  "score": 0.0,
  "error": "precondition not checked by agent"
}
```

A low precondition score usually means one of four things:

* the agent omitted the check
* the agent did not cite evidence
* the cited artifact did not match the ground-truth field
* the agent's `facts_found` did not match the artifact value

## Violation counts

Reports include violation counts.

| Field                   | What it means                                                                |
| ----------------------- | ---------------------------------------------------------------------------- |
| `horizon_violations`    | The agent tried to use evidence from after the allowed cutoff time.          |
| `actor_gate_violations` | The agent tried to access an artifact outside the actor's visibility.        |
| `subsystem_violations`  | The agent tried to access a subsystem outside the role's allowed subsystems. |
| `total_violations`      | Sum of the violation counts.                                                 |

Violations are not cosmetic. They explain why a run with a correct answer may still fail.

## Dead ends

A dead end is a tool call that returned no visible result.

Reports include:

| Field                 | What it means                                                              |
| --------------------- | -------------------------------------------------------------------------- |
| `dead_ends_hit`       | Number of empty tool results.                                              |
| `dead_ends_recovered` | Number of times the agent followed an empty result with a successful call. |

Empty results can happen because:

* the artifact does not exist
* the artifact exists but is outside the actor's visibility
* the artifact exists but is from after the cutoff time
* the artifact exists but is in a subsystem the role cannot access
* the search query did not match anything

Dead ends are not always bad. A good agent may search, find nothing, and then try the right fallback. The report shows whether the agent recovered or stopped too early.

## Component scores in `meta`

Each task result includes component scores in `meta`.

```json
{
  "meta": {
    "cf_answer": 0.75,
    "cf_trajectory": 0.9167,
    "sl_answer": 0.6667,
    "sl_trajectory": 0.7857,
    "ps_answer": 1.0,
    "ps_trajectory": 1.0
  }
}
```

Use these to see whether the failure came from the answer or the trajectory.

For example:

```text
cf_answer low, cf_trajectory high
```

The agent retrieved reasonable evidence, but the final answer did not correctly use it.

```text
sl_answer high, sl_trajectory low
```

The agent claimed to verify the checks, but the path was weak. It may have failed to recover from dead ends or searched too narrowly.

```text
ps_answer high, ps_trajectory low
```

The answer cited acceptable evidence, but the tool path crossed an access boundary.

## Expected submitted answer shape

GroundEval expects the agent to produce structured output.

The key fields are:

```json
{
  "preconditions_verified": [
    {
      "check": "customer_is_enterprise",
      "passed": true,
      "facts_found": {
        "plan_tier": "enterprise"
      },
      "evidence_artifacts": ["crm_account"]
    }
  ],
  "reasoning": "The customer is Enterprise and no duplicate ticket is open.",
  "should_act": true
}
```

The scorer reads:

| Field                    | What it does                                      |
| ------------------------ | ------------------------------------------------- |
| `preconditions_verified` | List of checks the agent claims it verified.      |
| `check`                  | Must match the contract precondition name.        |
| `passed`                 | Whether the agent claims the precondition passed. |
| `facts_found`            | Facts the agent found for this precondition.      |
| `evidence_artifacts`     | Artifact IDs the agent used as evidence.          |
| `reasoning`              | Used by Perspective answer scoring.               |
| `should_act`             | Default decision field.                           |

If the agent returns unparseable free text, GroundEval may still complete the run, but the score will usually be zero because the scorer cannot find the required checks and facts.

## Observation reports

Observe Mode writes a different kind of report.

```bash
uv run python -m groundeval observe \
  --framework crewai \
  --agent-class my_project.crew.MyCrew \
  --output eval_output
```

Observation writes:

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

Observation is not scoring.

The observation report tells you what GroundEval saw during the run. It is used to draft the evaluation package.

Use it to answer:

* Which tools did the agent call?
* What arguments did the tools receive?
* What did those tools return?
* How long did each tool call take?
* What final answer did the agent produce?
* What config did GroundEval infer?
* What still needs human review?

The generated config remains a draft until you review it.

## `observed_run.json`

`observed_run.json` contains the raw observed run.

Its shape is:

```json
{
  "run_id": "observed_my_project_crew_MyCrew_1234567890",
  "framework": "crewai",
  "agent_class": "my_project.crew.MyCrew",
  "tool_calls": [
    {
      "tool_name": "fetch_customer",
      "arguments": {
        "customer_id": "crm_account"
      },
      "return_value": {
        "id": "crm_account",
        "subsystem": "crm",
        "plan_tier": "enterprise"
      },
      "latency_ms": 12.4
    }
  ],
  "final_answer": {
    "preconditions_verified": [],
    "reasoning": "...",
    "should_act": true
  },
  "total_latency_ms": 1400.0
}
```

This is useful when you want to inspect exactly what happened during observation before deciding whether the draft config is correct.

## `observe_report.md`

`observe_report.md` is the readable version of the observed run.

It includes:

* run ID
* framework
* agent class
* total latency
* tool calls recorded
* observed tool calls
* tool arguments
* tool latency
* return value previews
* final answer
* pointer to the generated draft config

This report is for review, not scoring.

## Draft config output

Observe Mode also writes:

```text
draft_config/
  config.yaml
  tool_map.yaml
  REVIEW.md
  task_contracts/
    inferred_task.yaml
  artifacts/
    observed/
      001_fetch_customer_crm_account.json
```

The generated draft may include fields like:

```yaml
review_required: true
inferred_from:
  run_id: observed_my_project_crew_MyCrew_1234567890
  source: structured_answer
  reason: Observed check in agent answer.
```

These fields are review aids. They help you understand why GroundEval inferred something. They are not proof that the inferred check is correct.

Observed behavior is not ground truth.

## How to debug common report patterns

### High accuracy, low Silence

The agent got the final answer right but skipped required checks.

What to inspect:

* `precondition_results`
* missing `check` names
* missing `facts_found`
* the submitted `preconditions_verified` field
* `sl_answer` and `sl_trajectory` in `meta`

Likely fix:

* tighten the task instruction
* add clearer precondition descriptions
* make sure the output schema is reaching the agent
* add or correct tool access for the missing evidence

### High accuracy, low Counterfactual

The agent reached the right decision but the evidence did not support its claims.

What to inspect:

* `precondition_results`
* `evidence_cited`
* `ground_truth_field`
* `facts_found`
* whether the cited artifact ID matches the ground-truth artifact ID
* `cf_answer` and `cf_trajectory` in `meta`

Likely fix:

* require fetch after search
* make ground-truth fields more explicit
* ensure fixture `returns` contain the required facts
* make the agent cite artifact IDs for each precondition

### Low Perspective

The agent crossed an access boundary.

What to inspect:

* `roles`
* `actors`
* `subsystem_violations`
* `actor_gate_violations`
* `horizon_violations`
* `ps_answer` and `ps_trajectory` in `meta`

Likely fix:

* correct the role's allowed subsystems
* correct the task actor or role
* move an artifact into the right subsystem if the config is wrong
* keep the violation if the agent really attempted out-of-bounds access

Do not paper over a Perspective failure just because the answer was useful. That is the point of the track.

### Many dead ends

The agent kept searching or fetching things that returned nothing.

What to inspect:

* `dead_ends_hit`
* `dead_ends_recovered`
* fixture `entity_arg`
* artifact IDs
* search queries from observation output
* visibility and timestamp gates

Likely fix:

* pass stable IDs through `inputs`
* improve tool descriptions
* fix `entity_arg` in fixture mode
* check whether the role can access the subsystem
* check whether artifacts are after the cutoff time

### Zero score

A zero usually means the scorer could not connect the answer, evidence, and contract.

Check these first:

* Did the config have preconditions?
* Did each precondition have `required_facts`?
* Did each precondition have a dotted `ground_truth_field`?
* Did the agent return parseable JSON?
* Did the answer include `preconditions_verified`?
* Did each precondition include `facts_found`?
* Did each precondition include `evidence_artifacts`?
* Did fixture mode define useful `returns`?
* Did corpus mode have JSON artifacts in `artifacts_dir`?

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
Did the answer use facts_found and evidence_artifacts correctly?
```

GroundEval is useful because it gives you a debugging loop.

Change the agent, config, tools, or artifacts. Rerun. Watch the scores move for known reasons.
