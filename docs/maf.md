# Microsoft Agent Framework integration

GroundEval evaluates Microsoft Agent Framework (MAF) agents through a single **observe + score** workflow using MAF's native OpenTelemetry observability.

MAF already records what happened during an agent run. GroundEval runs the agent, captures those spans in-process, normalizes the observed trajectory, and scores it deterministically against a reviewed task contract.

> Observability shows the path. GroundEval grades the path.

## Mental model

The MAF adapter does not pass middleware into your agent, replace tool results, deny tool calls.

The flow is:

```text
Run 1: observe
  GroundEval loads the MAF agent
  GroundEval enables MAF OpenTelemetry instrumentation for the run
  The MAF agent runs normally
  GroundEval captures emitted spans in memory
  GroundEval normalizes spans into tool calls, model calls, workflow events, and final output
  GroundEval drafts a config

Human review
  You review the generated contract
  You decide which tools were required
  You decide which observed return fields define correctness
  You decide the expected final decision

Run 2: observe --score
  GroundEval runs the MAF agent normally again
  GroundEval captures a fresh MAF OpenTelemetry trace in memory
  GroundEval compares the observed run to the reviewed contract
  GroundEval writes deterministic scores and a diagnostic report
```

No separate trace export step is required.

## Install

Install GroundEval with the MAF adapter dependencies for your project environment.

```bash
uv sync --group dev
```

If your package manager does not define a `maf` group yet, install the Microsoft Agent Framework package your project uses alongside GroundEval. Microsoft Agent Framework includes the OpenTelemetry API and SDK packages used for observability.

## First run: observe and draft

```bash
uv run python -m groundeval observe \
  --framework maf \
  --agent-class your_package.your_module.YourAgent \
  --output eval_output
```

GroundEval writes:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  observed_run_maf.json
  observe_report_maf.md
  draft_config/
    config.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

The draft config is not ground truth. It is a starting point generated from one observed run. The first observation may be wrong, incomplete, lucky, or overfit to the agent's current behavior.

## Review the contract

A MAF contract is based on the tools the agent should call, the arguments those calls should use, the fields those calls should return, and the final decision the agent should make.

Example:

```yaml
agent:
  framework: maf
  agent_class: your_package.your_module.YourAgent

task_contracts:
  - name: verify_customer_outreach
    task_description: Verify whether the customer can be contacted.
    decision_field: should_act
    expected_decision: true

    tool_expectations:
      - tool: fetch_customer
        match_args:
          customer_id: CUST-123
        expected_return:
          account_status: active
          consent_status: opted_in
        required: true
        satisfies:
          - customer_is_active
          - customer_has_consented

      - tool: fetch_email_history
        match_args:
          customer_id: CUST-123
        expected_return:
          last_contact_days: 45
        required: true
        satisfies:
          - customer_not_recently_contacted

    preconditions:
      - check: customer_is_active
        description: Customer account must be active.
        required_tool: fetch_customer
        expected_field: account_status
        expected_value: active

      - check: customer_has_consented
        description: Customer must have opted into contact.
        required_tool: fetch_customer
        expected_field: consent_status
        expected_value: opted_in

      - check: customer_not_recently_contacted
        description: Customer must not have been contacted in the recent cooldown window.
        required_tool: fetch_email_history
        expected_field: last_contact_days
        expected_min: 30
```

The contract defines the expected behavior. The MAF run supplies the observed behavior. GroundEval scores the gap between the two.

Review the expected fields in `config.yaml` before scoring. Do not treat the first observation as automatically correct just because the agent produced it.

## Mark the config reviewed

```bash
uv run python -m groundeval validate \
  --config eval_output/draft_config/config.yaml \
  --mark-reviewed
```

## Second run: observe and score

```bash
uv run python -m groundeval observe \
  --framework maf \
  --agent-class your_package.your_module.YourAgent \
  --config eval_output/draft_config/config.yaml \
  --score \
  --output eval_output/scored_run
```

GroundEval writes:

```text
eval_output/scored_run/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  observed_scores.json
  observed_run_maf.json
  observe_report_maf.md
```

`observed_scores.json` includes:

- required tool call coverage
- argument matching
- expected return matching
- precondition verification
- final decision matching
- optional tool-boundary diagnostics when the contract declares allowed or forbidden tool names

## What the MAF adapter records

The adapter records whatever MAF emits through OpenTelemetry for the run, then normalizes what it understands:

- agent invocation spans
- subagent spans when MAF emits them
- workflow and executor spans when MAF emits them
- model and chat spans
- input and output token usage when present
- completed function or tool calls
- tool arguments when sensitive telemetry is enabled
- tool return values when sensitive telemetry is enabled
- final output when present in the run result or spans
- span timing and errors
- raw span attributes in the diagnostic event timeline

Tool calls and final answers are used for deterministic scoring. The richer MAF metadata is preserved for the report so users can debug the run without reading raw logs.

## The observe diagram for MAF

Each observe flow also writes `observe_diagram.pdf`.

For MAF runs, the PDF can reflect normalized:

- agents
- tool calls
- return summaries
- handoffs
- sub-agent activity when emitted by MAF as separate observed agents
- final answer

The diagram is behavior-only and contract-agnostic.

It uses a single top-down timeline across lanes, so ordering is preserved even in multi-agent runs.

## Sensitive telemetry

MAF only includes prompts, responses, function arguments, and tool results when sensitive telemetry is enabled. GroundEval enables sensitive telemetry for the local observe run because observe + score is a development and evaluation workflow, not production monitoring.

Do not run this against production traffic unless the trace content is acceptable for your environment.

## What this integration does not do

The MAF adapter does not:

- require users to export a trace manually
- pass middleware into the MAF agent
- deny a tool call
- replace a tool result
- treat the first observation as canonical truth
