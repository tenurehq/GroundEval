# Finance / Loan Underwriting Evaluation Example

A consumer lending pipeline with five actors: applicant, loan officer, underwriter, compliance reviewer, and fraud analyst. Tests temporal cutoff discipline (what was known at decision time), role-based access to sensitive financial data, and regulatory compliance checks.

## Domain Overview

| System            | Role in this domain                                         |
|-------------------|------------------------------------------------------------|
| CRM               | Loan applications, status tracking, decision records       |
| Credit Report     | FICO scores, DTI ratios, delinquency history               |
| Bank Statements   | Account balances, income verification                      |
| Underwriting Notes| Internal assessments, conditions, rationale                 |
| Fraud Alerts      | Suspicious activity flags, investigation outcomes          |
| Approval Letters  | Approval, denial, and adverse action notices               |

## Actors & Roles

| Actor              | Role               | Can see                                                       |
|--------------------|--------------------|---------------------------------------------------------------|
| loan_officer_smith | loan_officer       | crm, credit_report, bank_statements, approval_letters         |
| underwriter_garcia | underwriter        | crm, credit_report, bank_statements, underwriting_notes, approval_letters |
| fraud_analyst_lee  | fraud_analyst      | crm, fraud_alerts, credit_report                              |
| compliance_kim     | compliance_reviewer| crm, underwriting_notes, fraud_alerts, approval_letters       |
| applicant_johnson  | applicant          | crm                                                           |

## What This Tests

**PERSPECTIVE**: Can an agent respect that the loan officer cannot see fraud alerts or underwriting notes? That the applicant can only see CRM updates? That a fraud analyst cannot see underwriting notes or approval letters? Five distinct visibility cones — this is the richest perspective test of any domain pack.

**COUNTERFACTUAL**: Three causal chains:
1. Fraud flag (FA-601) causes application denial (CRM-1002)
2. Income verification failure causes denial (CRM-1003)
3. Underwriting approval triggers offer letter generation (CRM-1001)

**SILENCE**: Three search-space pairs:
- An approval (CRM-1004) with no letter generated — procedural gap
- A denial (CRM-1005) with no adverse action notice — compliance gap
- A fraud flag that DID block an application — agent must find it, not claim absence

## File Inventory

```
finance/
  config.yaml         # 5 actors, 3 causal links, 3 silence pairs
  events.jsonl        # 37 events across 2026-03-01 to 2026-03-14
  artifacts/
    crm/                # CRM-1001 through CRM-1005
    credit_report/      # CR-201 through CR-205
    bank_statements/    # BS-301 through BS-303
    underwriting_notes/ # UN-401 through UN-405
    fraud_alerts/       # FA-601, FA-602
    approval_letters/   # AL-501 through AL-503
  README.md
```

## Quick Start

```bash
cd examples/finance
uv run python -m groundeval generate --config config.yaml --events events.jsonl
uv run python -m groundeval eval --config config.yaml --questions eval_output/eval_questions.json --events events.jsonl --model claude-sonnet-4-6
```
