# GroundEval Domain Examples

Ready-to-run evaluation scenarios for GroundEval. Each domain includes a complete config, event log, and artifact corpus — copy the folder, run two commands, and get scored results in under 10 minutes.

## Available Domains

| Domain | What it models | Why it's useful |
|---|---|---|
| [enterprise-support/](enterprise-support/) | SaaS customer support org (5 actors, 6 systems, 3-week timeline) | Tests role-based access across Zendesk to Jira to Confluence to Salesforce chains, causal links through escalation workflows, and silence detection for missing postmortems |
| [cybersecurity/](cybersecurity/) | SOC with tiered analysts (5 actors, 6 systems, 2-week timeline) | Tests tiered access control (L1 vs L2 vs responder vs engineer), attack chain causality (phishing to credential theft to lateral movement to ransomware), and search discipline across SIEM/EDR/ticketing |
| [healthcare/](healthcare/) | Hospital clinical operations (6 actors, 6 systems, 2-week timeline) | Tests HIPAA-style access boundaries — billing can't see clinical notes, pharmacist can't see imaging, advocate can't see billing. Medication error causality, lab result chains, and discharge follow-up gaps |
| [finance/](finance/) | Consumer lending pipeline (5 actors, 6 systems, 2-week timeline) | Tests temporal cutoff discipline (what was known at decision time), role-based access to credit/fraud/underwriting data, and regulatory compliance (adverse action notices). Five distinct visibility cones |
| [legal/](legal/) | Law firm contract review (6 actors, 6 systems, 10-day timeline) | Tests privilege boundaries (opposing counsel cannot see matter notes or privileged docs), citation discipline, version-tracking across redlines, and DPA/filing compliance gaps |
| [tiny/](tiny/) | Minimal 2-actor 4-event scenario | Quickstart for learning the config format before building your own domain |

## How to Use

### 1. Pick a domain and copy it

```bash
cp -r examples/enterprise-support my-eval/
cd my-eval/
```

### 2. Generate questions

```bash
uv run python -m groundeval generate --config config.yaml --events events.jsonl
```

### 3. Run evaluation

```bash
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --model claude-sonnet-4-6
```

### 4. Read the results

```bash
cat eval_output/results_claude-sonnet-4-6.json
```

## Creating Your Own Domain

Each domain needs exactly three things:

1. **config.yaml** — declares actors, roles, subsystems, causal links, and silence pairs
2. **events.jsonl** — timestamped events with actors, artifact IDs, and facts
3. **artifacts/** — JSON files your agent will retrieve (one per artifact ID)

Use these examples as templates. The config schema is documented in the [main README](../README.md). If you don't have your own data yet, ask an LLM:

> "Generate a 40-event JSONL log and matching artifacts for a <your-domain> with 4 actors over 2 weeks. Include a mix of causal chains and silent gaps."

Drop the output into your folder and you're ready.

## Evaluation Tracks in These Examples

Every domain exercises all three tracks:

- **PERSPECTIVE**: Could actor X have known about event Y at time Z? Tests access boundaries and temporal gates.
- **COUNTERFACTUAL**: If event A had not happened, would event B have occurred? Tests causal reasoning against the event log.
- **SILENCE**: Did event B happen after event A? Tests whether the agent searches the right places before concluding "no".
