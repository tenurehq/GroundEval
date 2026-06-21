# Legal / Contract Review Evaluation Example

A mid-size law firm with six actors: associate, partner, client, opposing counsel, paralegal, and compliance reviewer. Tests citation discipline (privilege boundaries), temporal reasoning (which draft version was current at review time), and regulatory compliance checks around DPAs and filings.

## Domain Overview

| System         | Role in this domain                                         |
|----------------|------------------------------------------------------------|
| Contracts      | Draft agreements, MSA, SLA, merger agreements              |
| Redlines       | Version-tracked revisions from opposing counsel            |
| Email          | Attorney-client and attorney-opposing counsel communications|
| Matter Notes   | Internal firm notes, strategy memos                        |
| Privilege Docs | Attorney-client privileged communications                  |
| Filing System  | Executed contract repository with document inventory        |

## Actors & Roles

| Actor             | Role              | Can see                                                       |
|-------------------|-------------------|---------------------------------------------------------------|
| associate_ramirez | associate         | contracts, redlines, email, matter_notes, filing_system       |
| partner_kim       | partner           | contracts, redlines, email, matter_notes, privilege_docs, filing_system |
| client_acme_rep   | client            | contracts, redlines, email                                    |
| opp_counsel_jones | opposing_counsel  | contracts, redlines, email                                    |
| paralegal_chen    | paralegal         | contracts, filing_system, email                               |
| compliance_wong   | compliance_reviewer| contracts, redlines, email, matter_notes                      |

## What This Tests

**PERSPECTIVE**: The signature test here is privilege boundaries. Opposing counsel cannot see matter notes or privilege documents. The paralegal cannot see redlines or matter notes. Only the partner can see privilege documents. An agent that cites PRIV-601 when answering from the associate's perspective has committed a privilege violation — exactly the kind of error an LLM judge would miss.

**COUNTERFACTUAL**: Three causal chains:
1. Redline with revised indemnity clause (RED-201) enables partner approval (CON-101)
2. Compliance hold (CON-101) delays execution until DPA is resolved
3. Client approval triggers contract execution within 48 hours (CON-101, CON-103)

**SILENCE**: Three search-space pairs:
- A contract (CON-102) executed without a DPA — procedural gap the agent must find
- Contracts (CON-101, CON-103) that DO have DPAs — agent must not claim absence
- Filings that exist vs. missing filings

## File Inventory

```
legal/
  config.yaml         # 6 actors, 3 causal links, 3 silence pairs
  events.jsonl        # 35 events across 2026-03-01 to 2026-03-10
  artifacts/
    contracts/        # CON-101 through CON-103
    redlines/         # RED-201 through RED-205
    email/            # EML-301 through EML-306
    matter_notes/     # MN-501 through MN-503
    privilege_docs/   # PRIV-601
    filing_system/    # FILE-401 through FILE-403
  README.md
```

## Quick Start

```bash
cd examples/legal
uv run python -m groundeval generate --config config.yaml --events events.jsonl
uv run python -m groundeval eval --config config.yaml --questions eval_output/eval_questions.json --events events.jsonl --model claude-sonnet-4-6
```
