# OpenAI Agents SDK integration

GroundEval evaluates OpenAI Agents SDK agents through the same **observe + score** workflow as its other framework adapters. The adapter uses the SDK's lifecycle hooks and native tracing processor interface to capture what happened during a run.

GroundEval runs the agent normally, records agents, function tools, handoffs, model activity, traces, spans, errors, and final output, then normalizes that data into an observed trajectory for deterministic scoring against a reviewed task contract.

> Observability shows the path. GroundEval grades the path.

## Mental model

The OpenAI Agents adapter does not replace your tools, rewrite your agent instructions, alter handoff behavior, or judge output with another model. It observes the SDK run.

The flow is:

```text
Run 1: observe
  GroundEval loads your OpenAI Agents entry point
  GroundEval registers a native tracing processor
  GroundEval passes lifecycle hooks to Runner.run_sync
  The agent runs normally
  GroundEval records agents, tools, handoffs, model calls, traces, spans, and errors
  GroundEval normalizes the run and drafts a config

Human review
  You review the generated contract
  You decide which tools and arguments are required
  You decide which return fields define correctness
  You decide which agents and handoffs are required
  You decide the expected final decision

Run 2: observe --score
  GroundEval runs the agent normally again
  GroundEval captures a fresh trajectory
  GroundEval compares the observed run to the reviewed contract
  GroundEval writes deterministic scores and diagnostic reports
```

No manual wrapping of individual tools is required. The adapter receives local tool activity through OpenAI Agents lifecycle hooks and supplements it with native SDK spans.

## Requirements

The integration requires the OpenAI Agents SDK package and an environment supported by that SDK.

Install the adapter dependency group:

```bash
uv sync --group dev --group openai-agents
```

The corresponding dependency group in `pyproject.toml` is:

```toml
[dependency-groups]
openai-agents = ["openai-agents"]
```

Set the credentials required by your configured model provider. For the default OpenAI provider:

```bash
export OPENAI_API_KEY=your_key_here
```

GroundEval uses `openai_agents` as the framework identifier even though the Python package is installed as `openai-agents` and imported as `agents`.

## What you point GroundEval at

The `--agent-class` option is a dotted Python path to an importable entry point. It can resolve to:

- an OpenAI Agents SDK `Agent`
- a factory function returning an `Agent`
- an `OpenAIAgentsEntry`
- a factory returning an `OpenAIAgentsEntry`
- an `(agent, input)` tuple
- a dictionary containing `agent`, `input`, `context`, and `run_config`
- an object with an `agent` attribute and optional `input`, `context`, and `run_config` attributes
- an object implementing `run_groundeval(hooks=..., max_turns=...)`

Using `OpenAIAgentsEntry` is the clearest option because it keeps the agent, input, context, and run configuration together.

## Recommended entry point

Create a factory that returns `OpenAIAgentsEntry`:

```python
from agents import Agent, function_tool

from groundeval.framework_adapters.openai_agents_adapter import OpenAIAgentsEntry


@function_tool
def fetch_customer(customer_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "account_status": "active",
        "consent_status": "opted_in",
        "subsystem": "crm",
    }


def build_customer_agent() -> OpenAIAgentsEntry:
    agent = Agent(
        name="Customer Review",
        instructions="Review the customer and return a structured decision.",
        tools=[fetch_customer],
    )
    return OpenAIAgentsEntry(
        agent=agent,
        input="Review customer CUST-123 and determine whether outreach is allowed.",
    )
```

The dotted entry path for this example is:

```text
your_package.your_module.build_customer_agent
```

The adapter sends the entry's `input`, `context`, and `run_config` to `Runner.run_sync`. The CLI `--max-steps` value is passed to the SDK as `max_turns`.

## Direct Agent input

If the entry point resolves directly to an `Agent`, the adapter looks for input in this order:

1. The agent's `groundeval_input` attribute
2. The `GROUNDEVAL_AGENT_INPUT` environment variable
3. An empty string

For example:

```bash
export GROUNDEVAL_AGENT_INPUT="Review customer CUST-123 and return a structured decision."
```

Using `OpenAIAgentsEntry` is preferable when the input is part of a repeatable evaluation scenario.

## Custom execution

For advanced setups, expose an object with `run_groundeval`:

```python
from agents import Agent, Runner


class CustomerEvaluationEntry:
    def __init__(self) -> None:
        self.agent = Agent(
            name="Customer Review",
            instructions="Review the customer and return a structured decision.",
        )

    def run_groundeval(self, hooks, max_turns):
        return Runner.run_sync(
            self.agent,
            "Review customer CUST-123.",
            hooks=hooks,
            max_turns=max_turns,
        )


def build_evaluation_entry() -> CustomerEvaluationEntry:
    return CustomerEvaluationEntry()
```

A custom entry must pass the supplied hooks into its SDK run. Otherwise GroundEval may still receive native trace spans, but lifecycle details such as agent identity on tool calls can be incomplete.

## First run: observe and draft

```bash
uv run python -m groundeval observe \
  --framework openai_agents \
  --agent-class your_package.your_module.build_customer_agent \
  --output eval_output
```

GroundEval writes the standard observation artifacts and OpenAI Agents-specific rich artifacts:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  observed_run_openai_agents.json
  observe_report_openai_agents.md
  draft_config/
    config.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

The draft config is not ground truth. It reflects one observed run, which may be incomplete, incorrect, lucky, or overfit to current behavior. Review all inferred tools, arguments, return values, preconditions, agents, handoffs, and decision fields before scoring.

## Review the contract

An OpenAI Agents contract can specify expected tool behavior and multi-agent behavior.

```yaml
agent:
  framework: openai_agents
  agent_class: your_package.your_module.build_customer_agent

task_contracts:
  - name: verify_customer_outreach
    task_description: Verify whether the customer can be contacted.
    decision_field: should_act

    tool_expectations:
      - tool: fetch_customer
        match_args:
          customer_id: CUST-123
        expected_return:
          account_status: active
          consent_status: opted_in

    preconditions:
      - check: customer_is_active
        description: Customer account must be active.
        required_facts:
          - account_status
        required_tool: fetch_customer
        expected_field: account_status

      - check: customer_has_consented
        description: Customer must have opted into contact.
        required_facts:
          - consent_status
        required_tool: fetch_customer
        expected_field: consent_status

    required_agents:
      - agent_name: Customer Review
```

For a multi-agent workflow, you can also require handoffs and agent-specific tools:

```yaml
    required_agents:
      - agent_name: Triage Agent
      - agent_name: Account Specialist

    required_handoffs:
      - from_agent: Triage Agent
        to_agent: Account Specialist

    required_agent_tool_expectations:
      - agent_name: Account Specialist
        tool: fetch_customer
        match_args:
          customer_id: CUST-123
        expected_return:
          account_status: active
```

The contract defines expected behavior. The observed OpenAI Agents run supplies actual behavior. GroundEval scores the gap between them.

Tool names must match the names exposed by the SDK. Agent names must match the `Agent.name` values observed during the run. Review IDs in `observed_run_openai_agents.json` if you prefer to contract against `agent_id` or executor IDs.

## Structured final output

Framework scoring expects the final output to be a dictionary with the fields used by the task contract. A typical output is:

```json
{
  "preconditions_verified": [
    {
      "check": "customer_is_active",
      "passed": true,
      "facts_found": {
        "account_status": "active"
      },
      "evidence_artifacts": []
    },
    {
      "check": "customer_has_consented",
      "passed": true,
      "facts_found": {
        "consent_status": "opted_in"
      },
      "evidence_artifacts": []
    }
  ],
  "all_preconditions_pass": true,
  "should_act": true,
  "reasoning": "The account is active and the customer opted in."
}
```

The adapter accepts an SDK result's `final_output` and parses JSON strings when possible. For reliable deterministic scoring, configure the agent to return structured output rather than free-form text.

## Mark the config reviewed

```bash
uv run python -m groundeval validate \
  --config eval_output/draft_config/config.yaml \
  --mark-reviewed
```

## Second run: observe and score

```bash
uv run python -m groundeval observe \
  --framework openai_agents \
  --agent-class your_package.your_module.build_customer_agent \
  --config eval_output/draft_config/config.yaml \
  --score \
  --output eval_output/scored_run
```

Scored and no-draft runs use timestamped artifact names so multiple runs can coexist. The output directory includes files shaped like:

```text
eval_output/scored_run/
  observed_run_<timestamp>.json
  observe_report_<timestamp>.md
  observe_diagram_<timestamp>.pdf
  observed_scores_<timestamp>.json
  observed_run_openai_agents_<timestamp>.json
  observe_report_openai_agents_<timestamp>.md
```

The score output includes:

- required tool coverage
- tool argument matching
- expected return matching
- precondition verification
- agent requirements
- handoff requirements
- per-agent tool requirements
- final answer and trajectory diagnostics

## What the adapter records

The lifecycle hooks record:

- agent start and end events
- local tool start and end events
- tool names, arguments, return values, and latency
- the agent responsible for each local tool call
- handoffs between agents
- model call completion data when available
- final agent output

The native tracing processor records:

- trace start and end events
- agent spans
- function spans
- handoff spans
- generation spans
- span timing and parent relationships
- span errors
- raw normalized span payloads in the event timeline

The rich observed run contains:

- agent inventory
- workflow nodes
- handoffs
- tool calls
- model events
- errors
- final output
- a capability map indicating which signals were observed

Tool calls, final output, agents, and handoffs participate in deterministic scoring. The fuller event timeline is retained for debugging and reporting.

## Native tracing behavior

The OpenAI Agents SDK tracing surface must remain enabled for native trace and span capture. If tracing is globally disabled or disabled in the supplied `run_config`, lifecycle hooks can still capture local agent, tool, handoff, and model events, but native span data will be absent.

The adapter adds its processor alongside the SDK's existing tracing processors. It does not replace the default OpenAI trace exporter. Configure or disable the SDK's own trace export separately if your environment requires it.

Trace payloads and tool hooks may contain prompts, tool arguments, tool results, and model output. Treat observation artifacts as sensitive evaluation data and do not run the adapter against production traffic unless storing that content is acceptable.

## Tool-call normalization

Local function tools are primarily recorded through lifecycle hooks because those hooks provide the calling agent and tool-call context. Function spans are used as a fallback when a corresponding hook-recorded tool call is not found.

Hosted tools and provider-side operations may appear in native spans or raw events without being normalized as local `ObservedToolCall` records. Verify `observed_run_openai_agents.json` before writing a contract that depends on hosted-tool behavior.

## Multi-agent workflows

The adapter records each observed SDK agent using its configured name and a normalized ID. Handoffs are captured through both lifecycle hooks and native handoff spans, with duplicate span handoffs suppressed when an equivalent hook handoff already exists.

For the strongest multi-agent scoring:

- give every agent a stable, unique `name`
- use SDK handoffs rather than hiding delegation inside a custom tool
- pass GroundEval's supplied hooks into custom runner implementations
- review observed agent IDs and handoff endpoints before finalizing the contract

## The observe diagram

Each observe flow writes `observe_diagram.pdf`.

For OpenAI Agents runs, the diagram can show:

- agent swimlanes
- local tool calls
- tool argument and return summaries
- handoff arrows
- final output

The diagram is behavior-only and contract-agnostic. It reflects normalized records, so an SDK event that is retained only in the raw event timeline may not appear as a diagram node.

## Troubleshooting

### No observer registered

Confirm that the OpenAI Agents adapter registration is present in `observe.py` and that the framework name is exactly:

```text
openai_agents
```

### OpenAI Agents package is missing

Install the dependency group:

```bash
uv sync --group openai-agents
```

The runtime package is imported as `agents`.

### The run receives an empty input

Return an `OpenAIAgentsEntry` with an explicit `input`, set `groundeval_input` on a direct agent, or define `GROUNDEVAL_AGENT_INPUT`.

### Tool calls have no agent identity

Use the standard `OpenAIAgentsEntry` execution path or ensure a custom `run_groundeval` method passes the provided hooks to `Runner.run_sync` or `Runner.run`.

### Native spans are missing

Check that SDK tracing is enabled globally and that the supplied `run_config` does not disable tracing.

### Structured scoring is zero

Confirm that:

- the final output is a dictionary or valid JSON object
- `preconditions_verified` uses the same `check` names as the reviewed contract
- `facts_found` contains the required fields
- required tool names exactly match observed tool names
- expected arguments and return fields match the normalized observed values

### Synchronous runner error

The built-in entry path uses `Runner.run_sync`, which expects a normal synchronous CLI process. If your integration must run inside an existing event loop, provide a custom `run_groundeval` entry that manages execution appropriately, or invoke GroundEval from its normal CLI process.

## What this integration does not do

The OpenAI Agents adapter does not:

- rewrite agent instructions
- replace or mock tool results
- deny a tool call
- change handoff routing
- treat the first observation as canonical truth
- guarantee normalization of every hosted or provider-side tool operation
- replace the SDK's default trace exporter configuration
- use an LLM judge for deterministic scoring
