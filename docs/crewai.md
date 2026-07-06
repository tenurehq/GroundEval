# Evaluating CrewAI Crews with GroundEval

GroundEval plugs into an existing CrewAI crew without requiring you to rewrite your agent code. It runs the crew, records the tool calls and outputs, normalizes the CrewAI execution trace, and scores the observed run against a reviewed task contract.

There is one CrewAI path: observe the crew, review the generated contract, then score the same framework-native run structure. GroundEval does not require an artifact mode or a copied artifact corpus for CrewAI evaluation.

No LLM judge. Scores come from the reviewed task contract, the observed CrewAI tool trajectory, the final answer, and the access rules in the config.

## 1. Prerequisites

- **Python 3.13+**
- **CrewAI**, installed per the [CrewAI installation guide][crewai-install]
- **GroundEval**, cloned and installed with CrewAI support

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

## 2. Observe an existing crew

Start by observing a real CrewAI run. GroundEval loads your crew, runs it normally, records the execution, and writes a draft evaluation package you can review.

```bash
uv run python -m groundeval observe \
  --framework crewai \
  --agent-class my_project.crew.MyCrew \
  --output eval_output
```

Observe output includes the normalized observed run, a Markdown report, a behavior diagram, and a CrewAI-rich run file when framework-rich data is available:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  observed_run_crewai.json
  draft_config/
    config.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

## 3. How the CrewAI adapter works

GroundEval lets you observe and deterministically evaluate existing CrewAI crews without changing the crew implementation.

The CrewAI adapter:

1. Loads the configured crew class.
2. Installs a CrewAI event collector.
3. Runs the crew normally.
4. Converts observed CrewAI events into GroundEval's normalized run format.
5. Stores CrewAI-specific details under `framework_extra`.

It captures:

- tool calls
- tool arguments and return summaries
- evidence extracted from tool outputs
- agents
- workflow nodes
- handoffs when delegation-like events are present
- model events when surfaced by CrewAI
- final output
- errors

The normalized run is written to `observed_run.json`. CrewAI-rich data is attached to `framework_extra` and also written to `observed_run_crewai.json` when observe outputs are written.

## 4. The observe diagram for CrewAI

Each observe run writes `observe_diagram.pdf`.

For CrewAI, the diagram is behavior-only. It can show:

- ordered tool calls
- compact argument and return summaries
- evidence tags when normalized evidence exists
- handoffs between agents when normalized handoff records exist
- multiple agents and sub-agents as swimlanes
- final answer
- errors when present

The diagram uses a top-down global execution timeline across lanes, so sequence stays visually aligned.

The PDF is not contract-aware and does not show scoring overlays.

## 5. Define what the crew is supposed to verify

GroundEval scores CrewAI runs against a task contract. A task contract describes the checks the crew must complete before it takes action, the facts each check depends on, and the final decision field the scorer should read.

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
        ground_truth_field: "ticket.created_at"

      - check: customer_impact_is_high
        description: "Customer must be on Enterprise plan with active usage."
        required_facts: [plan_tier, account_status]
        ground_truth_field: "customer.plan_tier"

      - check: no_open_duplicate
        description: "No other open ticket from this customer describes the same issue."
        required_facts: [duplicate_count]
        ground_truth_field: "duplicate_check.duplicate_count"

    valid_action: "all_preconditions_pass"
    decision_field: "should_escalate"
```

For CrewAI, the contract is evaluated against the observed CrewAI trajectory and final answer. The evidence comes from the tools your crew actually called and the outputs those tools returned during the observed run.

## 6. Review the draft config

Observe can draft a config from the observed run. The generated config is intentionally treated as a draft. Review the task contract, access rules, tool mapping, and expected checks before scoring.

```bash
uv run python -m groundeval validate \
  --config eval_output/draft_config/config.yaml \
  --mark-reviewed
```

GroundEval blocks scoring with unreviewed draft configs unless you explicitly allow drafts.

## 7. Observe and score a CrewAI run

After the config is reviewed, run observe with scoring enabled:

```bash
uv run python -m groundeval observe \
  --framework crewai \
  --agent-class my_project.crew.MyCrew \
  --config eval_output/draft_config/config.yaml \
  --score \
  --output eval_output/scored_run
```

This writes:

```text
eval_output/scored_run/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  observed_scores.json
  observed_run_crewai.json
```

GroundEval scores the observed CrewAI trajectory and final answer against the reviewed contract. It checks whether the crew performed the required verification steps, whether the final decision matches the contract, whether the cited or extracted evidence supports the decision, and whether the run stayed within the configured access rules.

## 8. Output format

You do not need to hand-write an output schema. GroundEval appends the expected response format to the crew's last task automatically.

The scorer consumes three fields:

- **`preconditions_verified`**: an array of objects. Each object states which precondition was checked, whether it passed, what facts were found, and which evidence IDs were used.
- **`reasoning`**: a short explanation of the crew's decision.
- **`should_act`** or your configured `decision_field`: a boolean indicating whether the action is valid.

If the crew produces output that does not parse as JSON with those fields, the adapter wraps the raw text with empty preconditions. The run will not crash, but the score will be zero because GroundEval cannot verify the final answer structure.

## 9. Comparing CrewAI evaluation outputs

After observing or scoring CrewAI runs, you can compare two GroundEval JSON outputs to see changes in tool trajectories, scores, evidence use, and final answer structure.

See `compare.md` for the compare workflow and supported output types.

## 10. Known constraints and edge cases

- **GroundEval does not configure your crew's LLM.**
  The `provider` and `model` fields in `config.yaml` only control GroundEval's built-in agent loop. When `framework: crewai` is set, your crew uses its own configured LLM.

- **`max_steps` maps to `crew.max_iter` when that attribute exists.**
  Some CrewAI versions may not respect it.

- **Scoring requires a reviewed config.**
  Draft configs are blocked unless you explicitly allow them.

- **CrewAI sub-agents appear in the same single PDF** when they are surfaced as distinct observed agents and handoffs.

- **The CrewAI path is framework-native.**
  GroundEval scores the observed CrewAI run structure. It does not require a separate artifact mode for CrewAI.

## 11. Full example layout

```text
my-eval/
├── config.yaml
└── my_project/
    └── src/
        └── my_project/
            ├── crew.py
            └── config/
                ├── agents.yaml
                └── tasks.yaml
```

Run an observed CrewAI evaluation:

```bash
uv run python -m groundeval observe \
  --framework crewai \
  --agent-class my_project.crew.MyCrew \
  --config config.yaml \
  --score \
  --output eval_output
```

[crewai-install]: https://docs.crewai.com/v1.14.7/en/installation
[crewai-llms]: https://docs.crewai.com/v1.14.7/en/llms
