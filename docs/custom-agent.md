# Bringing your own agent

GroundEval is agent-agnostic. The framework doesn't import your agent: you subclass `ModelProvider` and point your config at it. Everything else (question generation, gated runtime, trajectory recording, deterministic scoring) is handled by the framework.

## Step 1: Subclass `ModelProvider`

Create a file anywhere in your project:

```python
# my_eval/my_provider.py
from groundeval.providers import ModelProvider
from groundeval.core import AgentTrajectory, EvalQuestion, GatedRuntime


class MyAgentProvider(ModelProvider):
    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        # Initialize your model or agent framework here.

    def complete(self, prompt: str, max_tokens: int = 2056) -> str:
        # Used for question prose generation only (llm_question_prose: true).
        # If you're using a built-in provider for generation and this
        # provider only for evaluation, return an empty string.
        return your_model.generate(prompt, max_tokens)

    def run_agent(
        self,
        question: EvalQuestion,
        context: str | None,
        runtime: GatedRuntime | None,
        max_steps: int = 5,
    ) -> tuple[AgentTrajectory, dict[str, Any]]:
        # Your agent loop. Called once per question.
        # - context: pre-built artifact string (context-injection mode)
        # - runtime: GatedRuntime providing gated fetch/search (tool mode)
        # Exactly one of context or runtime will be non-None.
        ...
        return trajectory, final_answer
```

You must implement two methods:

**`complete(prompt, max_tokens)`**: single-turn completion. Only called if you enable `llm_question_prose: true` in your config for question text generation. If you are using the built-in Anthropic or OpenAI provider for generation and your custom provider only for evaluation, this method can return an empty string: it won't be called during `eval`.

**`run_agent(question, context, runtime, max_steps)`**: the agent loop. Called once per question. When `runtime` is not `None` (tool mode), call `runtime.fetch("artifact-id")` and `runtime.search("query")` to retrieve gated artifacts. When `context` is not `None` (context-injection mode), answer from the pre-built context string. Your agent must return both a populated `AgentTrajectory` and a final answer dict matching the track's schema.

## Step 2: Point your config at it

Add `provider_path` to your config, using a dotted Python import path:

```yaml
provider_path: my_eval.my_provider.MyAgentProvider
model: my-model-name
```

That's it. The framework's `ModelProvider.from_config` resolves the path, imports your class, calls its own `from_config` classmethod, and wires it into the eval loop. No shim, no monkey-patching, no modifying framework code.

You can still use the standard CLI:

```bash
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --model my-model-name
```

## The agent function contract

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `question` | `EvalQuestion` | The structured eval question. `question.question_type` is one of `"PERSPECTIVE"`, `"COUNTERFACTUAL"`, or `"SILENCE"`. `question.question_text` is the prose. `question.expected_answer_schema` carries the JSON schema your answer must conform to. Ground truth fields are stripped before the question reaches your agent. |
| `context` | `str` or `None` | Pre-built artifact string in context-injection mode. Non-`None` only when `--context-injection` is passed. |
| `runtime` | `GatedRuntime` or `None` | Gated corpus interface. Non-`None` in tool mode (the default). Call `runtime.fetch()` and `runtime.search()`. Never call the raw corpus directly when a runtime is provided. |
| `max_steps` | `int` | Maximum allowed turns. The framework sets this per question. SILENCE questions get extra steps proportional to their expected search space. Your agent must respect it. |

### Return value

`(trajectory, final_answer)`:

- `trajectory` is an `AgentTrajectory`. If you used a `GatedRuntime`, the framework merges tool call records, violation counts, and dead-end stats from the runtime into your trajectory automatically after `run_agent` returns. You only need to set `cited_artifacts` (a list of artifact ID strings the agent cited) and `final_answer`.
- `final_answer` is a dict conforming to `question.expected_answer_schema` (which is one of the `ANSWER_SCHEMAS` [4]).

## The two modes

### Tool mode (default)

The agent receives a `GatedRuntime` and calls:

```python
doc = runtime.fetch("ENG-001")  # returns None if gated
results = runtime.search("postmortem")  # results already gated by actor/time
```

The runtime records every call, enforces visibility and temporal gates automatically, and builds the trajectory that gets scored. Your agent must never call the raw corpus directly when a runtime is provided.

### Context-injection mode (`--context-injection`)

The framework packs relevant artifacts into a context string and passes it to the agent. `runtime` is `None`, `context` is populated. The agent answers from context without tool calls. The trajectory scorer checks citation discipline: did the agent cite the right artifacts from the injected context and avoid citing artifacts outside its visibility cone?

## Answer schemas by track

Your `final_answer` dict must conform to the schema for the question type [4]:

**PERSPECTIVE**: `could_actor_have_known` (boolean), `reasoning` (string), `evidence_artifacts` (list of artifact ID strings), `blocked_subsystems` (list of subsystem name strings).

**COUNTERFACTUAL**: `outcome_changed` (boolean), `causal_mechanism` (string matching the link type or an alias), `cause_event_id` (string), `effect_event_id` (string), `mechanism_direction` (one of `"cause_to_effect"`, `"effect_to_cause"`, `"no_causal_link"`), `evidence_artifacts` (list of artifact ID strings), `actors` (list of actor strings), `reasoning` (string).

**SILENCE**: `exists` (boolean), `answer` (one of `"yes"` or `"no"`), `reasoning` (string).

The schema is available at `question.expected_answer_schema`. If you are using structured output enforcement (tool use, function calling, or constrained decoding), use this schema directly.

## What not to do

- Don't call the raw corpus (`corpus.fetch()`, `corpus.search()`) when a `runtime` is provided. The `GatedRuntime` is the gated interface and the only thing the trajectory scorer examines.
- Don't ignore `max_steps`. The framework sets a per-question budget based on question type, and `budget_exceeded` on the trajectory will be set to `True` if your loop exhausts it without submitting an answer.
- Don't modify the question object. Treat it as read-only. Ground truth fields are already stripped, but your agent should not mutate any question field.
- Don't assume the question type. Your agent should be question-type-agnostic: just call tools, gather evidence, and submit a structured answer matching the schema. The scorer handles the rest.

## Testing your agent

```bash
cd examples/enterprise-support
uv run python -m groundeval generate --config config.yaml --events events.jsonl
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --model my-model-name
```

The per-question scores will tell you exactly where your agent's trajectory failed: wrong evidence cited, missed search spaces, visibility violations. The aggregate summary gives you a single number to track over iterations.