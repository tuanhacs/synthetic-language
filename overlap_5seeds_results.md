# Prefix and Suffix Overlap Path-QA Results — Five Training Seeds

## Experiment

- Cross-region path QA on a 6×6 grid; upper/lower training regions share the middle two rows. Test queries start and end in opposite outer regions.
- Prefix-free or suffix-free encoding with **arbitrary codebook overlap**, distinct from the overlap between spatial training regions. Codebooks use 1–4 codewords per vertex.
- Query length 2–6; repeated vertices allowed within path segments; no CoT or training/test noise.
- Base model: `d_model=512`, `n_layers=10`, `n_heads=8`.
- Batch size 64; AdamW
- Validation every 500 steps, up to 100 batches; early-stopping patience 8 checks and minimum delta 0.0001 bits/token.
- Training seeds 0–4; data/encoding seed 0. Training budgets are 1.2M–2.0M **query-pair occurrences**, not unique pairs or example counts. An L-waypoint query contributes L−1 occurrences.
- Evaluation: 1,000 samples per run, temperature 1.0. All **50/50 training and evaluation runs completed successfully**, with detailed error statistics available for every seed.
- Saved semantic training-edge counts are identical across prefix/suffix and all seeds at each budget. No generation was rerun to produce this report.

Entries are percentages expressed as **mean ± sample standard deviation across five training seeds (ddof=1)**, unless a conditional table explicitly lists fewer eligible seeds. Conditional percentages are calculated within each seed before averaging, not by pooling their denominators.

## 1. Overall test results

Denominator: all 1,000 evaluated samples per run; five seeds per row. “Success” means semantic success, not merely successful completion of the evaluation process.

| Code type | Training pairs | Decodable | Valid walk | Semantic success | Full success + EOS |
|---|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 64.56 ± 10.47 | 59.36 ± 8.59 | 11.24 ± 2.65 | 11.24 ± 2.65 |
| Prefix | 1.4M | 69.98 ± 8.34 | 65.92 ± 7.17 | 13.84 ± 8.29 | 13.84 ± 8.29 |
| Prefix | 1.6M | 68.24 ± 11.56 | 63.40 ± 13.90 | 20.32 ± 5.61 | 20.32 ± 5.61 |
| Prefix | 1.8M | 71.30 ± 15.11 | 68.86 ± 15.43 | 10.92 ± 5.22 | 10.92 ± 5.22 |
| Prefix | 2.0M | 68.50 ± 11.76 | 63.98 ± 11.88 | 13.62 ± 4.44 | 13.62 ± 4.44 |
| Suffix | 1.2M | 77.06 ± 13.38 | 71.10 ± 11.66 | 12.44 ± 7.55 | 12.44 ± 7.55 |
| Suffix | 1.4M | 77.82 ± 10.62 | 74.34 ± 10.97 | 13.62 ± 6.74 | 13.62 ± 6.74 |
| Suffix | 1.6M | 79.88 ± 7.51 | 75.26 ± 8.14 | 11.36 ± 3.67 | 11.36 ± 3.67 |
| Suffix | 1.8M | 85.28 ± 10.30 | 82.94 ± 9.70 | 12.42 ± 7.47 | 12.42 ± 7.47 |
| Suffix | 2.0M | 79.34 ± 7.80 | 74.48 ± 8.68 | 10.46 ± 2.33 | 10.46 ± 2.33 |

## 2. Other overall checks

Denominator: all evaluated samples; five seeds per row. Repeats are permitted, so simple-path compliance is not a success requirement.

| Code type | Training pairs | Correct start | Correct end | Waypoints in order | Terminated with EOS |
|---|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 58.26 ± 7.86 | 18.80 ± 4.53 | 11.24 ± 2.65 | 71.36 ± 13.65 |
| Prefix | 1.4M | 65.18 ± 6.95 | 21.32 ± 10.81 | 13.84 ± 8.29 | 76.00 ± 9.22 |
| Prefix | 1.6M | 62.56 ± 14.20 | 30.66 ± 7.51 | 20.32 ± 5.61 | 72.32 ± 13.66 |
| Prefix | 1.8M | 68.32 ± 15.52 | 18.24 ± 8.59 | 10.92 ± 5.22 | 73.76 ± 18.25 |
| Prefix | 2.0M | 62.76 ± 12.09 | 22.48 ± 6.97 | 13.62 ± 4.44 | 75.10 ± 16.33 |
| Suffix | 1.2M | 65.56 ± 10.16 | 27.38 ± 11.97 | 12.44 ± 7.55 | 76.60 ± 16.03 |
| Suffix | 1.4M | 68.86 ± 8.45 | 29.10 ± 5.41 | 13.62 ± 6.74 | 78.06 ± 12.54 |
| Suffix | 1.6M | 69.14 ± 7.98 | 26.66 ± 7.48 | 11.36 ± 3.67 | 80.10 ± 8.99 |
| Suffix | 1.8M | 77.20 ± 9.41 | 28.46 ± 13.95 | 12.42 ± 7.47 | 84.92 ± 12.69 |
| Suffix | 2.0M | 69.08 ± 8.43 | 24.74 ± 4.43 | 10.46 ± 2.33 | 79.40 ± 9.10 |

## 3. Errors among valid walks

Denominator: valid-walk samples within each run; **all five seeds contribute to every row**. Wrong-start and wrong-end errors can overlap. Intermediate-waypoint failure requires correct endpoints but failure to complete the required waypoint sequence in order and with sufficient occurrences.

| Code type | Training pairs | Wrong start | Wrong end | Intermediate-waypoint failure, correct endpoints | Semantic success among valid walks |
|---|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 1.76 ± 1.28 | 68.47 ± 5.91 | 11.00 ± 3.02 | 18.81 ± 2.94 |
| Prefix | 1.4M | 1.10 ± 0.67 | 67.13 ± 17.44 | 10.35 ± 4.23 | 21.48 ± 13.49 |
| Prefix | 1.6M | 1.47 ± 0.94 | 48.57 ± 20.80 | 15.51 ± 6.02 | 34.45 ± 15.49 |
| Prefix | 1.8M | 0.83 ± 0.93 | 70.65 ± 17.32 | 10.85 ± 6.53 | 17.67 ± 10.68 |
| Prefix | 2.0M | 1.98 ± 2.14 | 63.36 ± 13.86 | 12.26 ± 3.72 | 22.40 ± 9.36 |
| Suffix | 1.2M | 7.67 ± 2.19 | 62.17 ± 13.14 | 13.54 ± 4.67 | 16.90 ± 8.81 |
| Suffix | 1.4M | 7.07 ± 3.00 | 59.82 ± 12.07 | 13.86 ± 2.65 | 19.40 ± 12.44 |
| Suffix | 1.6M | 8.16 ± 2.12 | 64.26 ± 11.01 | 12.47 ± 3.88 | 15.33 ± 5.57 |
| Suffix | 1.8M | 6.92 ± 2.60 | 66.37 ± 13.74 | 12.22 ± 6.62 | 14.66 ± 7.48 |
| Suffix | 2.0M | 7.29 ± 1.85 | 66.61 ± 5.65 | 12.10 ± 3.49 | 14.14 ± 2.96 |

## 4. Decodable answers with invalid walks

Percentage denominator: all evaluated samples per run; five seeds per row. Counts are pooled across the five seeds (5,000 samples per row).

| Code type | Training pairs | Invalid-walk percentage | Invalid walks | First edge localized | Unlocalized |
|---|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 5.20 ± 3.12 | 260 | 2 | 258 |
| Prefix | 1.4M | 4.06 ± 2.10 | 203 | 3 | 200 |
| Prefix | 1.6M | 4.84 ± 3.21 | 242 | 1 | 241 |
| Prefix | 1.8M | 2.44 ± 0.92 | 122 | 1 | 121 |
| Prefix | 2.0M | 4.52 ± 2.74 | 226 | 4 | 222 |
| Suffix | 1.2M | 5.96 ± 2.28 | 298 | 6 | 292 |
| Suffix | 1.4M | 3.48 ± 2.31 | 174 | 1 | 173 |
| Suffix | 1.6M | 4.62 ± 4.68 | 231 | 3 | 228 |
| Suffix | 1.8M | 2.34 ± 1.10 | 117 | 1 | 116 |
| Suffix | 2.0M | 4.86 ± 1.12 | 243 | 10 | 233 |

Across all 50,000 evaluated samples, **32 of 2,116 decodable invalid walks were localized; 2,084 were unlocalized**. Every unlocalized case is labeled `shared_codeword_ownership` in the saved diagnostic.

The current graph-independent localizer declines to assign a unique vertex sequence when it encounters a shared codeword. These are not missing reports or necessarily undecodable bitstrings. This is a limitation of the diagnostic, not proof that a richer contextual decoder could never resolve such a sequence. Valid walks are decoded using the graph-aware validator and still contribute to the valid-walk statistics.

## 5. Location of the first invalid edge — limited coverage

Only the **32 localized cases** contribute. Each contributes the first non-edge encountered from left to right. `a→b` denotes source row a and destination row b (1-based rows 1–6), not vertex IDs; all 36 row pairs, including same-row transitions, are shown.

Denominator: localized invalid walks within each seed. Seeds with zero localized cases have an undefined distribution and are excluded, not filled with zeros. For a single eligible seed, the mean is shown but the sample SD is `n/a`. The tables **do not describe the 2,084 unlocalized errors** and have too few cases for strong location-based conclusions.

| Code type | Training pairs | Eligible training seeds | Localized cases |
|---|---:|---|---:|
| Prefix | 1.2M | 2, 4 | 2 |
| Prefix | 1.4M | 0, 3 | 3 |
| Prefix | 1.6M | 0 | 1 |
| Prefix | 1.8M | 4 | 1 |
| Prefix | 2.0M | 1, 2 | 4 |
| Suffix | 1.2M | 0, 2, 3, 4 | 6 |
| Suffix | 1.4M | 4 | 1 |
| Suffix | 1.6M | 1, 3 | 3 |
| Suffix | 1.8M | 1 | 1 |
| Suffix | 2.0M | 0, 1, 2, 4 | 10 |

### Prefix

| Source row → destination row | 1.2M | 1.4M | 1.6M | 1.8M | 2.0M |
|---|---:|---:|---:|---:|---:|
| 1→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→5 | 50.00 ± 70.71 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 50.00 ± 70.71 |
| 2→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→4 | 50.00 ± 70.71 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 25.00 ± 35.36 |
| 5→2 | 0.00 ± 0.00 | 50.00 ± 70.71 | 100.00 ± n/a | 100.00 ± n/a | 0.00 ± 0.00 |
| 5→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→4 | 0.00 ± 0.00 | 50.00 ± 70.71 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 25.00 ± 35.36 |
| 6→2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→4 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→5 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→6 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± n/a | 0.00 ± 0.00 |

### Suffix

| Source row → destination row | 1.2M | 1.4M | 1.6M | 1.8M | 2.0M |
|---|---:|---:|---:|---:|---:|
| 1→1 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→2 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→3 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→4 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→5 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 1→6 | 0.00 ± 0.00 | 0.00 ± n/a | 50.00 ± 70.71 | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→1 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→2 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→3 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→4 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 2→5 | 12.50 ± 25.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 17.50 ± 23.63 |
| 2→6 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 17.50 ± 23.63 |
| 3→1 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→2 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→3 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→4 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→5 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 3→6 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→1 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→2 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→3 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→4 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→5 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 4→6 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→1 | 12.50 ± 25.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→2 | 37.50 ± 47.87 | 0.00 ± n/a | 25.00 ± 35.36 | 0.00 ± n/a | 30.00 ± 24.49 |
| 5→3 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→4 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→5 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 5→6 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→1 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 100.00 ± n/a | 5.00 ± 10.00 |
| 6→2 | 37.50 ± 47.87 | 100.00 ± n/a | 25.00 ± 35.36 | 0.00 ± n/a | 5.00 ± 10.00 |
| 6→3 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→4 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 25.00 ± 50.00 |
| 6→5 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |
| 6→6 | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 | 0.00 ± n/a | 0.00 ± 0.00 |

## 6. Results by individual seed

Denominator: all 1,000 evaluated samples. Values below are percentages, without averaging.

### Semantic success

| Code type | Training pairs | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 7.1 | 12.3 | 14.3 | 10.8 | 11.7 |
| Prefix | 1.4M | 24.7 | 7.5 | 20.9 | 7.8 | 8.3 |
| Prefix | 1.6M | 26.2 | 14.3 | 24.2 | 22.5 | 14.4 |
| Prefix | 1.8M | 15.9 | 13.6 | 3.0 | 8.4 | 13.7 |
| Prefix | 2.0M | 8.6 | 16.7 | 18.6 | 9.4 | 14.8 |
| Suffix | 1.2M | 19.0 | 6.1 | 5.9 | 22.0 | 9.2 |
| Suffix | 1.4M | 11.0 | 25.5 | 8.7 | 11.6 | 11.3 |
| Suffix | 1.6M | 8.7 | 17.5 | 9.6 | 12.0 | 9.0 |
| Suffix | 1.8M | 11.1 | 9.4 | 5.6 | 25.2 | 10.8 |
| Suffix | 2.0M | 10.8 | 9.8 | 8.4 | 14.3 | 9.0 |

### Decodable

| Code type | Training pairs | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 53.8 | 63.1 | 82.0 | 60.7 | 63.2 |
| Prefix | 1.4M | 73.1 | 71.2 | 59.3 | 65.0 | 81.3 |
| Prefix | 1.6M | 60.7 | 66.5 | 54.6 | 76.1 | 83.3 |
| Prefix | 1.8M | 63.0 | 77.6 | 87.7 | 78.8 | 49.4 |
| Prefix | 2.0M | 72.7 | 63.2 | 77.4 | 78.7 | 50.5 |
| Suffix | 1.2M | 83.7 | 57.7 | 68.6 | 86.7 | 88.6 |
| Suffix | 1.4M | 71.6 | 63.6 | 81.6 | 80.8 | 91.5 |
| Suffix | 1.6M | 68.0 | 76.8 | 85.1 | 85.2 | 84.3 |
| Suffix | 1.8M | 85.3 | 94.5 | 79.2 | 95.9 | 71.5 |
| Suffix | 2.0M | 77.7 | 74.5 | 90.3 | 83.7 | 70.5 |

## 7. Conditional cross-region success

Mean ± sample SD of semantic success within each indicated test subgroup, across five seeds. Counts and subgroup denominators are retained in each full report and `summary.json`. The five-cross subgroup is small; zero observed success does not establish impossibility.

| Code type | Training pairs | Bottom → top | Top → bottom | 1 cross segment | 3 cross segments | 5 cross segments |
|---|---:|---:|---:|---:|---:|---:|
| Prefix | 1.2M | 11.48 ± 4.48 | 11.00 ± 3.08 | 15.98 ± 3.73 | 0.07 ± 0.16 | 0.00 ± 0.00 |
| Prefix | 1.4M | 15.04 ± 10.09 | 12.64 ± 7.14 | 18.92 ± 10.83 | 1.97 ± 2.57 | 0.00 ± 0.00 |
| Prefix | 1.6M | 19.92 ± 9.52 | 20.72 ± 4.04 | 27.69 ± 7.04 | 3.03 ± 3.33 | 1.43 ± 3.19 |
| Prefix | 1.8M | 13.36 ± 6.91 | 8.48 ± 4.16 | 15.36 ± 7.25 | 0.49 ± 0.53 | 0.00 ± 0.00 |
| Prefix | 2.0M | 13.52 ± 6.35 | 13.72 ± 3.35 | 19.17 ± 6.03 | 0.56 ± 0.92 | 0.00 ± 0.00 |
| Suffix | 1.2M | 14.64 ± 7.72 | 10.24 ± 7.53 | 17.66 ± 10.66 | 0.14 ± 0.31 | 0.00 ± 0.00 |
| Suffix | 1.4M | 12.12 ± 9.02 | 15.12 ± 5.06 | 19.06 ± 9.07 | 0.85 ± 1.33 | 0.00 ± 0.00 |
| Suffix | 1.6M | 11.64 ± 3.99 | 11.08 ± 4.22 | 16.13 ± 5.28 | 0.14 ± 0.31 | 0.00 ± 0.00 |
| Suffix | 1.8M | 11.08 ± 5.54 | 13.76 ± 9.59 | 17.52 ± 10.35 | 0.42 ± 0.76 | 0.00 ± 0.00 |
| Suffix | 2.0M | 8.72 ± 2.27 | 12.20 ± 3.06 | 14.87 ± 3.26 | 0.07 ± 0.16 | 0.00 ± 0.00 |


