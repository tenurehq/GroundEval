# Healthcare Clinical Operations Evaluation Example

A mid-size hospital/clinic with six actors across clinical, administrative, billing, and patient-advocacy roles. Models HIPAA-style access boundaries, lab-result release timing, medication workflows, and clinical decision chains.

## Domain Overview

| System          | Role in this domain                                        |
|-----------------|-----------------------------------------------------------|
| EHR             | Electronic health record: encounters, orders, diagnoses   |
| Lab             | Lab orders and results (CBC, CMP, troponin, etc.)        |
| Imaging         | Radiology: CT scans, findings, critical flags             |
| Pharmacy        | Medication orders, dose adjustments, administration       |
| Billing         | Insurance claims, billing codes, submission tracking      |
| Secure Messages | HIPAA-compliant clinician messaging                      |

## Actors & Roles

| Actor             | Role              | Can see                                                       |
|-------------------|-------------------|---------------------------------------------------------------|
| dr_chen           | physician         | ehr, lab, imaging, pharmacy, secure_messages                  |
| nurse_patel       | nurse             | ehr, lab, pharmacy, secure_messages                           |
| dr_williams       | pharmacist        | pharmacy, ehr, secure_messages                                |
| biller_torres     | billing_specialist| billing, ehr                                                  |
| coordinator_nguyen| care_coordinator  | ehr, secure_messages, billing                                 |
| advocate_brooks   | patient_advocate  | secure_messages, ehr                                          |

## What This Tests

**PERSPECTIVE**: Can an agent respect that a billing specialist cannot see clinical notes, lab results, or pharmacy orders? That a patient advocate cannot see billing claims? That a care coordinator sees a broader set than a billing specialist but not as much as a physician? Four actors, four different visibility cones.

**COUNTERFACTUAL**: Three causal chains:
1. Abnormal lab result (LAB-301, K+ 6.2) leads to medication adjustment (PHARM-501 dose change)
2. Abnormal imaging finding (IMG-201, pericardial effusion) leads to surgical consult
3. Medication administration error (PHARM-502, wrong dose) causes adverse event report

**SILENCE**: Three search-space pairs:
- A normal lab result (LAB-303) with no medication change needed — agent must confirm absence
- A discharge (EHR-1003) with no timely billing submission — agent must search billing
- Discharges that DO have follow-ups — agent must find them, not claim absence

## File Inventory

```
healthcare/
  config.yaml         # 6 actors, 3 causal links, 3 silence pairs
  events.jsonl        # 40 events across 2026-03-02 to 2026-03-14
  artifacts/
    ehr/              # EHR-1001 through EHR-1004
    lab/              # LAB-301 through LAB-303
    imaging/          # IMG-201, IMG-202
    pharmacy/         # PHARM-501 through PHARM-503
    secure_messages/  # MSG-401 through MSG-407
    billing/          # BILL-601 through BILL-603
  README.md
```

## Quick Start

```bash
cd examples/healthcare
uv run python -m groundeval generate --config config.yaml --events events.jsonl
uv run python -m groundeval eval --config config.yaml --questions eval_output/eval_questions.json --events events.jsonl --model claude-sonnet-4-6
```
