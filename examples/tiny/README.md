# Tiny Example

A minimal 2-actor, 4-event scenario for learning the config format.

## What is happening
- 4 events in `events.jsonl` create causal links, perspective boundaries, and one silence candidate.
- 2 artifacts in `artifacts/` provide the corpus.
- `config.yaml` declares roles/subsystems, causal link specs with **join conditions**, and silence search-space **selectors/templates**.

## Quick Start

```bash
cd examples/tiny
uv run python -m groundeval generate --config config.yaml --events events.jsonl
uv run python -m groundeval eval --config config.yaml --questions eval_output/eval_questions.json --events events.jsonl
```
