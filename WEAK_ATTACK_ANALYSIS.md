# Weak attack category analysis

Audited candidate: `model/candidates/ids_netflow14_full_20261110T142838157565Z.joblib`.
SHA-256: `9e1a47538aa5952e5189ee3d3cf5bd004be38b9024e04d3f8d1b605439673495`.
The audit reconstructed the notebook's cleaned data and original partitions,
verified the artifact checksum, and reproduced its saved test confusion matrix
exactly. No model fitting, threshold changes, database writes, or deployments
were performed.

## Two distinct failure modes

| Category | Test examples | Stage A rejected as benign | Correct final type | Main wrong final label |
| --- | ---: | ---: | ---: | --- |
| Bot | 376 | 104 | 272 | Benign: 104 |
| Web Brute Force | 273 | 28 | 157 | XSS: 85 |
| SQL Injection | 3 | 1 | 1 | Benign: 1; DoS GoldenEye: 1 |
| XSS | 125 | 18 | 51 | Web Brute Force: 55 |

When evaluated independently of Stage A, Stage B identifies all 376 Bot test
examples correctly. Bot's current failure is therefore primarily the binary
detector. In contrast, Stage B alone correctly identifies only 53/125 XSS and
177/273 Brute Force examples. Changing the binary threshold cannot repair those
attack-family errors.

## Exact feature collisions in the raw dataset

These counts are measured BEFORE the notebook removes conflicting rows. They
explain information lost by the reduced schema; they are not remaining duplicate
leakage in the cleaned train/test split.

| Category | Raw rows | Rows with identical 14-feature values but another label | Such rows with all 78 features |
| --- | ---: | ---: | ---: |
| Bot | 1,966 | 57 | 0 |
| Web Brute Force | 1,507 | 123 | 0 |
| SQL Injection | 21 | 6 | 0 |
| XSS | 652 | 29 | 0 |

A classifier cannot distinguish two identical input vectors based on their labels
alone. Having no exact 78-feature collisions does NOT establish separability,
generalization, or which additional fields would be sufficient.

After cleanup, just 15 distinct unambiguous SQL Injection rows remain in the
14-feature dataset, of which 3 are in the test split. This is insufficient to
establish reliable SQL Injection performance.

A descriptive nearest-training-neighbor check also found that 64 of 125 XSS test
examples were closest to Brute Force references, versus 47 closest to XSS. The
check uses imputed, signed-log standardized features with a sampled reference set
fitted on training data only. Distance depends on representation and sampling;
this supports feature overlap but is not proof that another model cannot improve.

## Threshold tradeoff on validation data only

| Binary threshold | Benign false-positive rate | Bot detection recall |
| --- | ---: | ---: |
| Current: 0.34229858575544503 | 0.9263% | 70.48% |
| 0.20 | 1.2339% | 81.65% |
| 0.10 | 1.5902% | 94.15% |
| 0.05 | 2.0602% | 97.34% |

The configured benign false-positive limit is 1%. These lower thresholds violate
that limit. SQL Injection validation detection stays at 1/3 for all these
thresholds. Arbitrarily lowering the threshold is therefore not a sufficient fix.

## Recommended next experiments

1. For the 14-feature production path, compare Stage A training with capped
   per-attack-family sample weights against the existing binary-balanced model.
   Fit weights using training counts only and compare recall at the SAME benign
   false-positive limit using validation data. Do not promote merely because Bot
   recall rises; review the complete tradeoff.
2. Gather additional independent labeled samples for SQL Injection, Heartbleed,
   Infiltration, and the real deployment network. Do not duplicate rare examples
   into validation or test data. Synthetic training examples are not independent
   evaluation evidence.
3. For web attack types, compare feature groups available from the actual
   collector. The 78-feature notebook mode can serve as an offline comparison,
   but all chosen inputs must be measured reliably in production. Supplying zeros
   for unavailable fields does not reproduce CICFlowMeter features. If NetFlow
   alone cannot meet the targets, add a suitable richer traffic sensor rather
   than promise all 14 labels from the current inputs.
4. Evaluate Stage B alternatives on validation data with macro/per-family recall,
   while retaining the current independent preprocessing. Stage B already uses
   balanced class weights; more weighting alone is not established as a solution.
5. Freeze the chosen configuration and use fresh independent captures for final
   evaluation. This test partition has now been inspected for diagnosis; it must
   not become a repeatedly tuned benchmark presented as untouched evidence.
6. Promote a matching API/model pair only after it passes agreed gates. Rebuild
   demo samples against BOTH stages afterward, without treating selected
   successful demos as accuracy measurements.

## Reproduce

From the parent Laravel project directory:

```powershell
.venv-ids/Scripts/python.exe tools/analyze_weak_attack_categories.py
```

This reads the original CSV files, the notebook's loading/splitting cells, and
the most recent full candidate. It requires the full netflow14 configuration
used for the audited candidate. Full machine-readable output is written to
`storage/app/weak-attack-analysis.json` in the parent project.

The model remains a diagnostic candidate and has not become deployment-ready
through this analysis. Block and Recover remain unchanged.
