# LangGraph integration

GroundEval evaluates LangGraph agents through the same **observe + score** workflow as its other adapters, using LangGraph's own streaming API and LangChain's callback system to see what happened during a run.

GroundEval loads your compiled graph, runs it the way it would normally run, watches the nodes and tool calls as they happen, normalizes that into an observed trajectory, and scores it deterministically against a reviewed task contract.

> Observability shows the path. GroundEval grades the path.

## Mental model

The LangGraph adapter does not rewrite your graph, replace tool results, or block a node from running. It watches.

The flow is:

```text
Run 1: observe
  GroundEval loads your compiled graph
  GroundEval inspects the graph's static structure (nodes and edges)
  GroundEval streams the graph run, attaching a callback handler to catch tool and model activity
  GroundEval normalizes what it saw into nodes, tool calls, model calls, and a final output
  GroundEval drafts a config

Human review
  You review the generated contract
  You decide which nodes or tools were required
  You decide which observed return fields define correctness
  You decide the expected final decision

Run 2: observe --score
  GroundEval runs the graph normally again
  GroundEval captures a fresh trajectory the same way
  GroundEval compares the observed run to the reviewed contract
  GroundEval writes deterministic scores and a diagnostic report
```

No manual instrumentation is required. GroundEval attaches to the graph's existing `stream`/`astream` interface and LangChain's callback system, both of which LangGraph already supports.

## Requirements

LangGraph observation requires **Python 3.11 or later**. Earlier versions don't propagate context variables reliably across asyncio tasks, and both LangGraph and LangChain lean on that propagation for callbacks to reach nested and async node execution correctly.

## Install

```bash
uv sync --group dev
```

There's no separate `langgraph` extra to install. The adapter doesn't import `langgraph` or `langchain` itself — it observes your compiled graph through the `stream`/`astream`/`get_graph`/`get_subgraphs` interface it already exposes, and taps into LangChain's callback system by matching its method names rather than subclassing anything from it. The only real requirement is that `langgraph` (and whatever produces your graph) is already installed in your own project.

## What you point GroundEval at

The `--agent-class` path can point to a few different things, and GroundEval will do the right thing with any of them:

- a compiled graph object
- a class or factory function that returns a graph when called
- an object with a `.compile()` method, which GroundEval will call for you

Either way, GroundEval expects to end up with a compiled graph that exposes `stream(...)` or `astream(...)`. If neither is available, it raises a clear error rather than guessing.

## First run: observe and draft

```bash
uv run python -m groundeval observe \
  --framework langgraph \
  --agent-class your_package.your_module.your_graph \
  --output eval_output
```

GroundEval writes:

```text
eval_output/
  observed_run.json
  observe_report.md
  observe_diagram.pdf
  observed_run_langgraph.json
  observe_report_langgraph.md
  draft_config/
    config.yaml
    tool_map.yaml
    REVIEW.md
    task_contracts/
      inferred_task.yaml
```

The draft config is not ground truth. It's a starting point generated from one observed run. The first observation may be wrong, incomplete, lucky, or overfit to the graph's current behavior.

## Review the contract

A LangGraph contract is based on the nodes and tools the graph should exercise, the arguments those calls should use, the fields those calls should return, and the final decision the graph should reach.

Example:

```yaml
agent:
  framework: langgraph
  agent_class: your_package.your_module.your_graph

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

The contract defines the expected behavior. The LangGraph run supplies the observed behavior. GroundEval scores the gap between the two.

Review the expected fields in `config.yaml` before scoring. Do not treat the first observation as automatically correct just because the graph produced it.

### A note on nodes vs. tools

LangGraph runs are made of nodes, and nodes don't always call a distinct "tool." GroundEval handles both cases:

- If a node calls a LangChain tool (or retriever), that call is recorded under the tool's own name — e.g. `fetch_customer`.
- If a node does its own work without invoking a separate tool, the node itself is recorded as the observed operation, named after the node — e.g. `verify_customer`.

When you write `tool_expectations`, you can reference either kind by name. GroundEval doesn't require every contract to be built strictly around tool-calling nodes.

## Mark the config reviewed

```bash
uv run python -m groundeval validate \
  --config eval_output/draft_config/config.yaml \
  --mark-reviewed
```

## Second run: observe and score

```bash
uv run python -m groundeval observe \
  --framework langgraph \
  --agent-class your_package.your_module.your_graph \
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
  observed_run_langgraph.json
  observe_report_langgraph.md
```

`observed_scores.json` includes:

- required tool call coverage
- argument matching
- expected return matching
- precondition verification
- final decision matching
- optional tool-boundary diagnostics when the contract declares allowed or forbidden tool names

## What the LangGraph adapter records

Before the run starts, GroundEval inspects the compiled graph itself to learn its static shape — the node names and the edges between them. Those edges become the handoffs shown in reporting; they describe how the graph is wired, not just what happened to occur in one run.

During the run, GroundEval streams the graph the same way you would, and attaches a callback handler to catch what LangChain already exposes. Together, this captures:

- every node the graph entered, including ones that ran inside a subgraph or branch
- tool calls and retriever calls made from within a node
- model/chat calls, including token usage and finish reason when available
- errors raised by a node, a tool, or the run itself
- the final output the graph produced
- the raw event timeline, preserved for the diagnostic report

Tool calls and the final answer are used for deterministic scoring. The rest of what's captured is preserved in the report so you can debug a run without digging through raw LangGraph traces.

## The observe diagram for LangGraph

Each observe flow also writes `observe_diagram.pdf`.

For LangGraph runs, the PDF reflects normalized:

- agent lanes, one per graph node
- tool calls and node-level operations
- return summaries
- handoffs, drawn from the graph's static edges
- the final answer

The diagram is behavior-only and contract-agnostic.

## What this integration does not do

The LangGraph adapter does not:

- require you to manually wire GroundEval into your graph
- rewrite your graph's nodes or edges
- deny a tool call
- replace a tool result
- treat the first observation as canonical truth