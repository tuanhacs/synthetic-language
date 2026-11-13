# UD Path-QA Results — Five Training Seeds

## Experiment

- Task: cross-region path QA on a 6×6 grid; two training regions sharing the middle two rows.
- Queries: 2–6 waypoints; cross-region test queries start and end in opposite outer regions.
- UD codewords, disjoint assignment, 1–4 codewords per vertex; repeated vertices allowed within segments.
- No CoT; no training or test noise.
- Model: base, `d_model=512`, `n_layers=10`, `n_heads=8`.
- Batch size 48; AdamW
- Maximum 50,000 steps; validation every 500 steps over up to 100 batches; early-stopping patience 8 checks, minimum delta 0.0001 bits/token.
- Training seeds: 0–4. Data seed and encoding seed: 0.
- Training budgets: 1.6M, 1.8M, 2.0M, 2.2M, 2.4M query-pair occurrences. An L-waypoint query contributes L−1 occurrences; these are not unique-pair counts or example counts.
- Shared 2.4M latent training pool; fixed validation/test budgets of 100k query-pair occurrences each.
- Evaluation: 1,000 samples per run, temperature 1.0, evaluation seed 0, generation batch size 512.

Entries are percentages reported as **mean ± sample standard deviation across training seeds (ddof=1)**. Means are unweighted across seeds, not pooled-sample conditional percentages. Conditional tables state their own availability and denominators.

Four older seed-0 reports (1.6M–2.2M) do not contain detailed error analysis. Their overall metrics are included; unavailable joint/conditional metrics and row-pair locations are not inferred from the ten saved examples. The newer plateau experiments are excluded from the five-seed cosine aggregates.

## 1. Overall test results

Denominator: all 1,000 evaluated samples per run. Every row includes five training seeds.

| Training pairs | Decodable | Valid walk | Semantic success | Full success + EOS |
|---|---:|---:|---:|---:|
| 1.6M | 84.00 ± 12.39 | 83.48 ± 12.30 | 21.66 ± 7.56 | 21.66 ± 7.56 |
| 1.8M | 74.58 ± 11.49 | 73.88 ± 11.30 | 26.54 ± 6.18 | 26.54 ± 6.18 |
| 2.0M | 81.48 ± 14.53 | 81.24 ± 14.57 | 26.02 ± 17.59 | 26.02 ± 17.59 |
| 2.2M | 74.28 ± 15.78 | 73.90 ± 15.85 | 42.06 ± 16.16 | 42.06 ± 16.16 |
| 2.4M | 75.54 ± 9.87 | 75.12 ± 10.02 | 31.60 ± 14.34 | 31.60 ± 14.34 |

## 2. Other overall checks

Denominator: all evaluated samples; five seeds per row. Repeated vertices are allowed, so simple-path compliance is not a success requirement.

| Training pairs | Correct start | Correct end | Waypoints in order | Terminated with EOS |
|---|---:|---:|---:|---:|
| 1.6M | 83.32 ± 12.30 | 40.02 ± 14.15 | 21.66 ± 7.56 | 86.00 ± 13.47 |
| 1.8M | 73.70 ± 11.41 | 43.02 ± 8.23 | 26.54 ± 6.18 | 79.90 ± 13.54 |
| 2.0M | 81.12 ± 14.51 | 40.44 ± 21.75 | 26.02 ± 17.59 | 85.16 ± 16.20 |
| 2.2M | 73.80 ± 15.91 | 56.60 ± 16.33 | 42.06 ± 16.16 | 77.94 ± 16.97 |
| 2.4M | 75.04 ± 9.92 | 47.42 ± 17.16 | 31.60 ± 14.34 | 79.02 ± 7.92 |

## 3. Errors among valid walks

Denominator: valid-walk samples in each available report. Wrong-start and wrong-end errors can overlap. Intermediate-waypoint failure requires correct endpoints but failure to satisfy the waypoint sequence in order and with sufficient occurrences. It is not simply “any waypoint error.”

The first four rows cover seeds 1–4 only; seed 0 lacks the saved joint counts. The 2.4M row covers seeds 0–4. These are not five-seed aggregates where only four reports are available.

| Training pairs | Seeds with detailed stats | Wrong start | Wrong end | Intermediate-waypoint failure, correct endpoints | Semantic success among valid walks |
|---|---|---:|---:|---:|---:|
| 1.6M | 1, 2, 3, 4 | 0.12 ± 0.09 | 51.93 ± 16.12 | 22.26 ± 7.48 | 25.74 ± 9.46 |
| 1.8M | 1, 2, 3, 4 | 0.33 ± 0.46 | 39.70 ± 12.17 | 22.73 ± 6.74 | 37.26 ± 6.69 |
| 2.0M | 1, 2, 3, 4 | 0.17 ± 0.27 | 42.08 ± 28.13 | 19.59 ± 7.19 | 38.18 ± 21.86 |
| 2.2M | 1, 2, 3, 4 | 0.14 ± 0.22 | 21.18 ± 7.62 | 19.48 ± 6.26 | 59.20 ± 9.61 |
| 2.4M | 0, 1, 2, 3, 4 | 0.09 ± 0.13 | 38.03 ± 15.71 | 20.84 ± 3.63 | 41.04 ± 14.50 |

## 4. Decodable answers with invalid graph edges

Denominator: all evaluated samples. All five seeds are included. For the four older reports without detailed analysis, this percentage is derived exactly from `decodable_pct − valid_walk_pct`; no location information is inferred.

| Training pairs | Invalid-walk percentage |
|---|---:|
| 1.6M | 0.52 ± 0.27 |
| 1.8M | 0.70 ± 0.29 |
| 2.0M | 0.24 ± 0.19 |
| 2.2M | 0.38 ± 0.18 |
| 2.4M | 0.42 ± 0.49 |

## 5. Location of the first invalid edge

Each localized invalid walk contributes its first non-edge, scanning left to right. `a → b` denotes source row a and destination row b, numbered 1–6, not vertex IDs. All 36 ordered row pairs are included, including same-row transitions.

Percentages are conditional on the localized invalid-walk count **within each run**, then averaged across eligible seeds. A run with zero invalid walks has an undefined conditional distribution and is excluded, not filled with zeros. Missing older statistics are also excluded. All available invalid walks were localized; there were no unlocalized cases.

| Training pairs | Seeds with a defined row-pair distribution | Localized cases across these seeds |
|---|---|---:|
| 1.6M | 1, 2, 3, 4 | 22 |
| 1.8M | 1, 2, 3, 4 | 30 |
| 2.0M | 1, 2, 3 | 11 |
| 2.2M | 1, 2, 3, 4 | 13 |
| 2.4M | 0, 1, 3 | 21 |

The following cells are mean ± sample standard deviation over the seeds listed above. Each column’s mean percentages sum to 100% within rounding. These locations are based on small counts and should be interpreted accordingly.

| Source row → destination row | 1.6M | 1.8M | 2.0M | 2.2M | 2.4M |
|---|---:|---:|---:|---:|---:|
| 1→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 1→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 1→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 1→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 1→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 1→6 | 12.50 ± 25.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 2→5 | 56.25 ± 42.70 | 44.95 ± 38.21 | 35.56 ± 33.55 | 48.75 ± 46.61 | 71.11 ± 34.21 |
| 2→6 | 0.00 ± 0.00 | 19.63 ± 11.55 | 22.22 ± 19.25 | 6.25 ± 12.50 | 5.56 ± 9.62 |
| 3→1 | 0.00 ± 0.00 | 6.25 ± 12.50 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 3→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 3→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 3→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 3→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 3→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 4→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 12.50 ± 25.00 | 0.00 ± 0.00 |
| 4→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 12.50 ± 25.00 | 0.00 ± 0.00 |
| 4→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.78 ± 4.81 |
| 4→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 4→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 4→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 2.78 ± 4.81 |
| 5→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 5→2 | 6.25 ± 12.50 | 8.33 ± 16.67 | 0.00 ± 0.00 | 0.00 ± 0.00 | 8.33 ± 14.43 |
| 5→3 | 0.00 ± 0.00 | 6.25 ± 12.50 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 5→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 6.67 ± 11.55 |
| 5→5 | 6.25 ± 12.50 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 5→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 6→1 | 0.00 ± 0.00 | 14.58 ± 17.18 | 20.00 ± 34.64 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 6→2 | 12.50 ± 25.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 15.00 ± 30.00 | 2.78 ± 4.81 |
| 6→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 11.11 ± 19.25 | 5.00 ± 10.00 | 0.00 ± 0.00 |
| 6→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 6→5 | 6.25 ± 12.50 | 0.00 ± 0.00 | 11.11 ± 19.25 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| 6→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## 6. Results by individual seed

Denominator: all 1,000 evaluated samples. Values are percentages without averaging.

### Semantic success

| Training pairs | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| 1.6M | 23.5 | 17.7 | 13.3 | 20.4 | 33.4 |
| 1.8M | 19.3 | 32.9 | 23.2 | 33.1 | 24.2 |
| 2.0M | 11.1 | 30.0 | 25.9 | 9.9 | 53.2 |
| 2.2M | 19.9 | 47.5 | 53.7 | 58.3 | 30.9 |
| 2.4M | 47.5 | 19.0 | 29.6 | 45.2 | 16.7 |

### Decodable

| Training pairs | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| 1.6M | 85.2 | 62.3 | 89.9 | 91.9 | 90.7 |
| 1.8M | 63.1 | 78.4 | 85.1 | 84.7 | 61.6 |
| 2.0M | 83.7 | 56.0 | 87.7 | 88.2 | 91.8 |
| 2.2M | 51.7 | 85.1 | 78.2 | 90.9 | 65.5 |
| 2.4M | 85.0 | 63.6 | 68.3 | 85.8 | 75.0 |


