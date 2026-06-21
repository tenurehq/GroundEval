# Cybersecurity Incident Response Evaluation Example

A Security Operations Center (SOC) with five actors in tiered roles: L1 analyst, L2 analyst, incident responder, security engineer, and threat hunter. Models a realistic SOC workflow: Splunk alerting, Jira ticketing, CrowdStrike endpoint detection, Slack war rooms, Confluence postmortems, and threat intel distribution.

## Domain Overview

| System      | Role in this domain                                          |
|-------------|-------------------------------------------------------------|
| Splunk      | SIEM: alert generation, log aggregation, detection rules    |
| CrowdStrike | EDR: endpoint detection, host isolation, IOC matching       |
| Jira        | Incident ticketing and tracking                             |
| Confluence  | Postmortem documentation                                    |
| Slack       | SOC war room, threat intel, and cross-team coordination     |
| Email       | Threat intel distribution (ISAC bulletins)                  |

## Actors & Roles

| Actor  | Role              | Can see                                                         |
|--------|-------------------|-----------------------------------------------------------------|
| chen   | soc_analyst_l1    | splunk, jira, slack (+ broadcast: alerts)                       |
| fatima | soc_analyst_l2    | splunk, jira, slack, crowdstrike (+ broadcast: alerts, escalations) |
| marcus | incident_responder| splunk, jira, crowdstrike, email, slack (+ broadcast: incidents, containment) |
| naomi  | security_engineer | splunk, jira, crowdstrike, confluence, slack (+ broadcast: incidents, postmortems, rules) |
| rick   | threat_hunter     | splunk, crowdstrike, jira, confluence, slack (+ broadcast: alerts, IOCs) |

## What This Tests

**PERSPECTIVE**: Can an agent respect that an L1 analyst (chen) cannot see CrowdStrike detections? That an incident responder (marcus) cannot access Confluence postmortems directly? That a detection on March 4 isn't visible to someone querying "as of March 2"?

**COUNTERFACTUAL**: Four causal chains modeling real attack sequences:
1. Phishing alert → credential theft (within 2 days)
2. Credential theft → lateral movement (within 3 days)
3. Containment → blocked exfiltration (within 1 day)
4. Initial access → ransomware encryption (within 5 days)

Each enforced with shared Splunk alert IDs and Jira ticket IDs.

**SILENCE**: Four search-space pairs:
- A phishing alert (SPLK-1005) caught early — agent must confirm no credential theft followed
- A low-severity incident (SEC-106) with no postmortem — agent must search Confluence and Jira before concluding "no"
- A low-priority alert (SPLK-1003) never escalated — agent must verify the alert stayed at L1
- Postmortems that DO exist — agent must find them, not claim absence

## File Inventory

```
cybersecurity/
  config.yaml         # 5 actors, 4 causal links, 4 silence pairs
  events.jsonl        # 42 events across 2026-03-02 to 2026-03-14
  artifacts/
    splunk/           # SPLK-1001 through SPLK-1006
    jira/             # SEC-101 through SEC-106
    confluence/       # CONF-301 through CONF-303
    crowdstrike/      # CS-501 through CS-801
    slack/            # SL-401 through SL-404
    email/            # EML-201
  README.md
```

## Quick Start

```bash
cd examples/cybersecurity

# Generate evaluation questions
uv run python -m groundeval generate --config config.yaml --events events.jsonl

# Run evaluation
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --model claude-sonnet-4-6

# Run only SILENCE track to test search discipline
uv run python -m groundeval eval \
  --config config.yaml \
  --questions eval_output/eval_questions.json \
  --events events.jsonl \
  --types SILENCE
```

## Expected Evaluation Shape

- ~30-35 questions across all three tracks
- PERSPECTIVE: tiered access control — L1 vs L2 vs responder vs engineer visibility
- COUNTERFACTUAL: questions probing each attack chain link
- SILENCE: questions requiring thorough search across Splunk, Jira, CrowdStrike, and Confluence
- Difficulty mixes: easy ~34%, medium ~33%, hard ~33%
