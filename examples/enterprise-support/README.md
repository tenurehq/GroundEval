# Enterprise Support Evaluation Example

A mid-size SaaS customer-support organization with five actors across engineering, sales, and support. Models a realistic multi-system workflow: Zendesk tickets escalate through Jira incidents into engineering hotfixes, postmortems in Confluence, Slack coordination, and Salesforce churn tracking.

## Domain Overview

| System     | Role in this domain                                    |
|------------|-------------------------------------------------------|
| Zendesk    | Customer-facing tickets                               |
| Jira       | Engineering incidents and escalations                 |
| Confluence | Postmortem documents                                  |
| Slack      | Cross-team coordination (incidents, engineering, sales, support) |
| Salesforce | Account health, deals, churn alerts                   |
| Git        | Hotfix deployment commits                             |

## Actors & Roles

| Actor | Role          | Can see                                                    |
|-------|---------------|------------------------------------------------------------|
| maria | engineer      | jira, git, confluence, slack (+ broadcast: incidents)      |
| james | engineer      | jira, git, confluence, slack (+ broadcast: incidents)      |
| priya | support_lead  | zendesk, jira, confluence, slack (+ broadcast: tickets, escalations) |
| david | support_agent | zendesk, confluence, slack (+ broadcast: tickets)          |
| lisa  | sales         | salesforce, slack, email (+ broadcast: deals, churn)       |

## What This Tests

**PERSPECTIVE**: Can an agent respect that Lisa (sales) cannot see Jira incidents? That David (support agent) cannot access git commits? That a postmortem written on March 5 isn't visible to someone querying "as of March 3"?

**COUNTERFACTUAL**: Three causal chains — escalation drives postmortem, churn follows unresolved escalation, hotfix follows incident. Each enforced with shared ticket IDs, not just temporal adjacency.

**SILENCE**: Four search-space pairs:
- An incident (ESC-102) that auto-recovered with no postmortem
- An escalation (ZD-501) that DID get a postmortem — agent must find it, not claim absence
- A deal closure (SF-201) with no subsequent churn alert
- An escalation (ZD-501) that triggered churn — agent must find the alert

## File Inventory

```
enterprise-support/
  config.yaml         # 5 actors, 3 causal links, 3 silence pairs
  events.jsonl        # 44 events across 2026-03-01 to 2026-03-20
  artifacts/
    jira/             # ESC-101 through ESC-107
    confluence/       # CONF-201 through CONF-204
    slack/            # SL-300 through SL-310
    zendesk/          # ZD-501 through ZD-540
    salesforce/       # SF-101 through SF-402
    git/              # GIT-801 through GIT-803
  README.md
```

## Quick Start

```bash
cd examples/enterprise-support

# Generate evaluation questions
uv run python -m groundeval generate --config config.yaml --events events.jsonl

# Run evaluation (replace with your model)
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --model claude-sonnet-4-6

# Or run only a specific track
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --types COUNTERFACTUAL
```

## Expected Evaluation Shape

- ~30-40 questions across all three tracks
- PERSPECTIVE: both positive (actor could know) and negative (blocked by role or time)
- COUNTERFACTUAL: questions probing each causal link
- SILENCE: questions requiring the agent to search before concluding "no"
- Difficulty mixes: easy ~34%, medium ~33%, hard ~33%
