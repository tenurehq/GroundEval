# Evaluating CrewAI Crews with GroundEval

GroundEval plugs into your existing CrewAI crew without touching your agent code. Point it at your crew, run it, and get back a score that tells you what your agent checked, what it skipped, what evidence it used, and whether it stayed inside its permissions.

No LLM judge. Scores come from task contracts, declared fixtures or seeded artifacts, and access rules.

## 1. Prerequisites

* **Python 3.13+**
* **CrewAI**, installed per the [CrewAI installation guide][crewai-install]
* **GroundEval**, cloned and installed with CrewAI support

GroundEval works with existing `crewai create crew my-project` and `crewai create flow my-flow` projects.

### Install GroundEval with CrewAI support

```bash
git clone https://github.com/tenurehq/groundeval.git
cd groundeval
uv sync --group dev --group crewai
```

If you already have GroundEval installed:

```bash
uv add crewai
```

## 2. Observe an existing crew first

If you already have a CrewAI crew, start with Observe Mode. GroundEval runs the crew, records tool calls and outputs, then generates a draft evaluation package you can review.

```bash
uv run python -m groundeval observe \
  --framework crewai \
  --crew-class my_project.crew.MyCrew \
  --output eval_output
```

## 3. How the adapter works

GroundEval lets you observe and deterministically evaluate existing CrewAI crews without rewriting your agent code. In Observe Mode, your crew runs with recording hooks so GroundEval can draft an evaluation config. In evaluation mode, GroundEval wraps tools through the gated runtime, records the trajectory, and scores each run across Counterfactual, Silence, and Perspective.
It:

1. **Loads your crew class** by dotted path, such as `my_project.crew.MyCrew`.
   The adapter supports `@CrewBase` classes with a `.crew()` method and plain `Crew` instances.

2. **Wraps each agent's tools.**
   Every tool in `agent.tools` is deep-copied, and its `_run` method is replaced with a version that routes through GroundEval's `GatedRuntime`. The runtime records every call, checks access policy, enforces subsystem and visibility gates, and builds a trajectory. The adapter does not distinguish between evaluation modes — it always wraps tools the same way. What changes is the backend providing the data.

   Tool mapping is automatic:

   * names containing `fetch`, `get`, `retrieve`, `read`, or `lookup` map to `fetch`
   * names containing `search`, `query`, `find`, `list`, or `discover` map to `search`

   You can override this behavior with `tool_map`.

3. **Injects the task contract.**
   The task description and contract `inputs` (if any) are prepended to the first CrewAI task. The expected output schema is appended to the last task's `expected_output`, so your crew's LLM sees the required response shape.

4. **Calls `crew.kickoff()`** and captures the result.

5. **Parses the output.**
   The adapter first tries `json.loads` on `result.raw`, then `json_repair` for malformed JSON, then falls back to wrapping free text. If your crew uses Pydantic output, set `output_mode: pydantic` and the adapter reads `result.pydantic` directly.

6. **Scores the run** from the recorded trajectory and structured answer.

Your crew code does not change. GroundEval patches at the tool boundary.

### Evaluation modes

There are two ways to give GroundEval the data it scores against.

Using real artifact files (**Corpus mode**) -- the default:
The `GatedRuntime` is backed by a file corpus in `./data/`. When your agent calls a tool, the runtime fetches or searches real artifact JSON files. This mode evaluates evidence-path correctness, retrieval quality, and access-boundary adherence against a realistic artifact set.

Using inline fixture data (**Fixture mode**) -- no artifact files needed
The `GatedRuntime` is backed by a `FixtureBackend` that synthesizes data from the contract's `allowed_tools` declarations. When your agent calls a tool, the runtime returns deterministic data from the declared `returns` dict merged with schema-compatible defaults for any missing fields. This mode lets you evaluate whether your agent uses the required facts and respects permission boundaries without building a full artifact corpus.

Fixture mode does not remove GroundEval's enforcement model. The runtime still records every call, checks subsystem access, enforces actor visibility, and applies temporal gates (when `timestamp` is declared on an `AllowedTool`). Perspective scoring works the same in both modes.

## 4. Tell GroundEval what your agent is supposed to do

This is where you define the checks your agent should complete before taking action. GroundEval calls this a task contract -- a YAML block that lists the preconditions, the data sources, and what a passing run looks like.

```yaml
# config.yaml
output_dir: ./eval_output

actors:
  verifier: analyst

roles:
  analyst:
    subsystems: [crm, email, jira, confluence]

task_contracts:
  - name: verify_escalation_readiness
    task_description: >
      Determine whether this support ticket should be escalated to
      engineering.
    preconditions:
      - check: ticket_age_exceeds_sla
        description: "The ticket must be older than 24 hours."
        required_facts: [created_at, sla_hours]
        ground_truth_field: "ticket_42.created_at"

      - check: customer_impact_is_high
        description: "Customer must be on Enterprise plan with active usage."
        required_facts: [plan_tier, account_status]
        ground_truth_field: "crm_account.plan_tier"

      - check: no_open_duplicate
        description: "No other open ticket from this customer describes the same issue."
        required_facts: [duplicate_count]
        ground_truth_field: "duplicate_check.duplicate_count"

    valid_action: "all_preconditions_pass"
    decision_field: "should_escalate"
```

### Fixture mode example

To use fixture mode, add `allowed_tools` to your task contract:

```yaml
task_contracts:
  - name: verify_escalation_readiness
    task_description: >
      Determine whether this support ticket should be escalated.
    preconditions:
      # ... same as above ...
    valid_action: "all_preconditions_pass"
    decision_field: "should_escalate"
    allowed_tools:
      fetch_ticket:
        entity_arg: ticket_id
        artifact_id: ticket_42
        subsystem: jira
        returns:
          created_at: "2026-03-01T08:00:00"
          sla_hours: 36
          status: open
      fetch_customer:
        entity_arg: customer_id
        artifact_id: crm_account
        subsystem: crm
        returns:
          plan_tier: enterprise
          account_status: active
      search_duplicates:
        artifact_id: duplicate_check
        subsystem: jira
        returns:
          duplicate_count: 0
```

When `allowed_tools` are declared, no artifact files are needed. Each declared tool becomes a virtual artifact in the `FixtureBackend`. The `returns` dict provides the ground truth values. Fields not in `returns` are populated with schema-compatible defaults from the tool's return type annotations.

## 5. Wire your crew

Add an `agent` block to `config.yaml`:

```yaml
agent:
  framework: crewai
  agent_class: my_project.crew.MyCrew
  tool_map:                         # optional
    fetch_customer: fetch
    search_tickets: search
  output_mode: auto                 # "auto", "pydantic", or "raw"
```

| Field         | Required | What it does                                                               |
| ------------- | -------: | -------------------------------------------------------------------------- |
| `framework`   |      Yes | Must be `"crewai"`.                                                        |
| `agent_class` |      Yes | Dotted Python path to your `@CrewBase` class or Crew factory.              |
| `tool_map`    |       No | Maps your tool names to `"fetch"` or `"search"`. Auto-detected if omitted. |
| `answer_key`  |       No | Extracts a nested key from your crew output.                               |
| `output_mode` |       No | `"auto"` by default. Use `"pydantic"` or `"raw"` when needed.              |

## 6. Output format

You don't need to write an output schema. GroundEval appends the expected format to your crew's last task automatically.

The scorers consume three fields:

* **`preconditions_verified`**: an array of objects. Each object states which precondition was checked, whether it passed, what facts were found, and which artifact IDs were used as evidence.
* **`reasoning`**: a short explanation of the agent's decision.
* **`should_act`** or your configured `decision_field`: a boolean indicating whether the action is valid.

If your crew produces output that does not parse as JSON with those fields, the adapter wraps the raw text with empty preconditions. The run will not crash, but the score will be zero.

## 7. Provide seed artifacts (corpus mode)

When running in corpus mode (no `allowed_tools` declared), GroundEval needs ground-truth artifacts to validate the crew's answer.

```json
[
  {
    "id": "ticket_42",
    "subsystem": "jira",
    "created_at": "2026-03-01T08:00:00",
    "sla_hours": 36,
    "status": "open"
  },
  {
    "id": "crm_account",
    "subsystem": "crm",
    "plan_tier": "enterprise",
    "account_status": "active"
  },
  {
    "id": "duplicate_check",
    "subsystem": "jira",
    "duplicate_count": 0
  }
]
```

Put these files in `./data/`.

In fixture mode, skip this step. The ground truth comes from the `returns` dict in your `allowed_tools` declarations.

## 8. Run the evaluation

```bash
uv run python -m groundeval task --config config.yaml
```

Example output:

```text
Overall -- counterfactual=0.833  silence=0.750  perspective=1.000  overall=0.861  accuracy=0.500
  verify_escalation_readiness: counterfactual=0.833  silence=0.750 perspective=1.000 overall=0.861
```

counterfactual = evidence used correctly, silence = checks completed, perspective = permissions respected. overall is the mean of the three.

## 9. Known constraints and edge cases

* **GroundEval does not configure your crew's LLM.**
  The `provider` and `model` fields in `config.yaml` only control GroundEval's built-in agent loop. When `framework: crewai` is set, your crew uses its own configured LLM.

* **`max_steps` maps to `crew.max_iter`** when that attribute exists.
  Some CrewAI versions may not respect it.

* **Tool wrapping happens at the `_run` level.**
  The adapter deep-copies each tool and replaces `_run`. This is the
  standard CrewAI execution entry point. If a custom tool performs I/O
  in `__init__`, a property, or a helper called outside `_run`, refactor
  that work into `_run` so the gated runtime can record it.

* **`@CrewBase` classes work best.**
  Plain `Crew` instances also work, but they lose the caching behavior of `_load_crew()`.

## 10. Full example layout

```text
my-eval/
├── config.yaml
├── task_artifacts/          # only needed in corpus mode
│   ├── ticket_42.json
│   ├── crm_account.json
│   └── duplicate_check.json
└── my_project/
    └── src/
        └── my_project/
            ├── crew.py
            └── config/
                ├── agents.yaml
                └── tasks.yaml
```

Run:

```bash
uv run python -m groundeval task --config config.yaml
```

[crewai-install]: https://docs.crewai.com/v1.14.7/en/installation
[crewai-llms]: https://docs.crewai.com/v1.14.7/en/llms