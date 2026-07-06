# Comparing GroundEval Runs

Here's the question GroundEval compare exists to answer:

> Did this agent actually get better, or did it just get different?

A single score tells you how one run did. It doesn't tell you what changed. Compare takes two GroundEval JSON outputs and shows you the difference: scores, task-level results, violations, tool trajectories, and the shape of the final answer.

You'll reach for this any time you touch something and want to know if it mattered. Changed a prompt. Swapped a model. Edited a task contract. Tweaked a tool. Updated an adapter. Refactored the agent workflow. Compare tells you whether that change actually shifted behavior, or if things are basically the same underneath.

## What compare can read

`compare.py` compares two JSON files, as long as they're the same kind of GroundEval payload. It understands three:

1. `observed_scores.json`
2. `observed_run.json`
3. task result JSON from GroundEval task runs

Both files need to match. If you hand it an `observed_run.json` and an `observed_scores.json`, it won't try to force a diff out of mismatched data. It'll just stop and tell you what it detected, so you're not staring at a comparison that's quietly meaningless.

## How you'd typically run it

Comparing an earlier scored run against a newer one:

```bash
uv run python -m groundeval compare \
  eval_output/baseline/observed_scores.json \
  eval_output/new_run/observed_scores.json
```

Or if you want to check behavior before you even look at scores:

```bash
uv run python -m groundeval compare \
  eval_output/baseline/observed_run.json \
  eval_output/new_run/observed_run.json
```

One habit worth keeping: put the older or trusted run first, and the newer run second.

```text
old → new
```

This isn't just a convention. Compare uses that order to decide what counts as a regression versus a fix versus something new that showed up. Flip the order and the labels flip with it.

## Why this actually matters

Here's the thing that trips people up: an agent's behavior can change in ways that never show up in the final answer.

Say a new run still lands on the right answer, but it quietly skips the tool that was supposed to check a precondition first. Or a model swap keeps the same final decision, but changes the order tools get called in, drops a handoff that was supposed to happen, or introduces a permission violation nobody asked for. None of that is visible if you're only looking at whether the answer was "correct."

Compare surfaces that stuff. Use it to check things like:

- is the agent still calling the same tools it used to
- did that prompt change actually help, or make things worse
- is a precondition that used to fail now getting verified
- did any new permission, horizon, or subsystem violations sneak in
- did a required multi-agent behavior quietly disappear
- did the final answer's schema shift
- did a framework adapter refactor change the shape of the normalized run

## Comparing scored runs

When you feed it two `observed_scores.json` files, compare reports on both scores and diagnostics. It looks at summary-level fields like:

- `counterfactual_score`
- `silence_score`
- `perspective_score`
- `overall_score`
- `accuracy`
- `total_violations`

Here's what that output looks like in practice:

```text
GroundEval Compare

Old file: eval_output/baseline/observed_scores.json
New file: eval_output/new_run/observed_scores.json

Scores changed:
- overall_score: 0.67 → 0.92
- total_violations: 2 → 0

Per-task changes:
- verify_customer_outreach: overall_score 0.67 → 0.92

Fixed violations:
- verify_customer_outreach: precondition 'customer_has_consented' not verified
- verify_customer_outreach: fetch_email_history call 2 returned empty
```

If you only have time to run one kind of comparison, this is the one to reach for. It's the clearest signal on whether a change actually improved things.

### What happens with violations

For scored payloads, compare doesn't just lump everything together. It separates what got fixed from what newly broke. That includes things like:

- horizon violations
- actor-gate violations
- subsystem violations
- dead ends
- precondition errors
- unsupported evidence
- unverified preconditions
- precondition reason strings
- missing required agents
- missing required handoffs
- missing required agent-tool expectations
- tool calls that came back empty

The score alone won't tell you why a run changed. This is where you actually find out.

## Comparing observed runs

When both files are `observed_run.json`, you're looking at behavior before scoring even enters the picture. Compare checks:

- tool call count
- tool sequence
- final answer shape
- `preconditions_verified` count, if it's present

Example output:

```text
GroundEval Compare

Old file: eval_output/baseline/observed_run.json
New file: eval_output/new_run/observed_run.json

Tool call count changed:
- old: 3
- new: 2

Trajectory diff:
- old: fetch_customer → fetch_email_history → send_outreach
- new: fetch_customer → send_outreach

Final answer diff:
- preconditions_verified count: 3 → 1
```

This is handy when you don't have a reviewed contract yet, or when you're trying to figure out if your adapter is still watching the same execution path after a dependency or framework change.

## Comparing task results

When both files are GroundEval task result payloads, compare gives you summary score changes plus per-task changes. This is the one to use for batch evaluations, when you want a quick before-and-after across a whole set of tasks instead of digging through each one.

```text
GroundEval Compare

Old file: results/baseline.json
New file: results/new.json

Scores changed:
- accuracy: 0.78 → 0.84
- overall_score: 0.74 → 0.81

Per-task changes:
- verify_escalation_readiness: overall_score 0.50 → 1.00
- check_missing_postmortem: overall_score 1.00 → 0.75
```

## What compare won't do

It's worth being upfront about this: compare is not a general-purpose JSON diff tool, and that's on purpose.

It won't print out every field that changed. Instead, it pulls out the parts of GroundEval's output that actually matter for reviewing an evaluation: scores, violations, trajectories, and the shape of the final answer. Run JSON tends to be full of timestamps, IDs, framework metadata, and raw spans that shift between runs for reasons that have nothing to do with behavior. Compare filters that noise out so what's left is actually worth reading.

## When you should run it

Basically, any time you make a change that could plausibly affect how the agent behaves:

- prompt edits
- model changes
- tool implementation changes
- framework adapter changes
- contract edits
- dependency upgrades
- multi-agent workflow changes
- CI regression checks

A workflow that works well:

```text
1. Save a baseline run.
2. Make one change.
3. Run the same evaluation again.
4. Compare the old output against the new one.
5. Look at the score, violation, and trajectory changes before you decide to accept the change.
```

## What "no differences" actually means

Sometimes you'll get this:

```text
No meaningful differences found.
```

That doesn't mean the two files are identical byte for byte, they almost certainly aren't. It means that within the fields compare actually pays attention to, nothing meaningful moved. That's a genuinely useful signal, not just an absence of output.

## Using it in CI

Compare works well in CI when you want something a human can actually read in a regression check.

A typical setup: a pull request runs GroundEval against a baseline task set, produces a new `observed_scores.json`, and the compare output gets attached to the build log or dropped into a PR comment. That turns a vague pass/fail into a concrete question anyone reviewing the PR can answer at a glance:

```text
Did this change alter the agent's score, evidence path, permissions behavior, or required tool sequence?
```

That's a much easier thing to review than raw JSON, and it tells you a lot more than a single pass/fail ever could.