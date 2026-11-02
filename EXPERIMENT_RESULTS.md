# Experiment Results

## Task and sample format

Each question is an ordered waypoint sequence `x_1, ..., x_L`. Every vertex is replaced by one of its assigned binary codewords, and the query codewords are concatenated directly. The answer is a piecewise-simple walk that starts at `x_1`, ends at `x_L`, and visits all waypoints in query order.

- Without CoT: `BOS + query_bits + _ + answer_bits + EOS`.
- Segment CoT: `BOS + query_bits + _ + segment_1 + _ + ... + segment_(L-1) + _ + final_answer + EOS`.
- `segment_i` is the simple path for consecutive waypoints `(x_i, x_(i+1))`. `final_answer` merges all segments and removes repeated boundary waypoints.
- During evaluation, the model receives `BOS + query_bits + _` and autoregressively generates everything after the first `_`.

Held-out-pairs experiments use unordered pairs. The pairs `{3,16}`, `{7,21}`, and `{10,35}` are excluded in both orientations from train and validation queries. Every held-out test query contains at least one of these consecutive waypoint pairs.

## Segment CoT

### Common settings

- CoT: segments
- `d_model`: 512
- `n_heads`: 8
- Evaluation samples: 1000

### IID

| Code | Assignment | Query | Train pairs | Layers | Temperature | Decodable | Success |
|---|---|---|---:|---:|---:|---:|---:|
| UD | disjoint | 2–4 | 1.2M | 10 | 1.0 | 98.7% | 87.1% |
| suffix | disjoint | 2–4 | 1.2M | 10 | 1.0 | 98.4% | 96.4% |
| prefix | disjoint | 2–4 | 1.2M | 10 | 1.0 | 99.9% | 98.8% |
| UD | overlap | 2–6 | 2.4M | 10 | 1.0 | 99.6% | 95.2% |

### Held-out pairs

| Code | Assignment | Query | Train pairs | Layers | Temperature | Decodable | Success |
|---|---|---|---:|---:|---:|---:|---:|
| UD | disjoint | 2–4 | 1.2M | 10 | 1.0 | 99.6% | 95.4% |

### Cross-region

| Code | Assignment | Query | Train pairs | Layers | Temperature | Decodable | Success |
|---|---|---|---:|---:|---:|---:|---:|
| UD | disjoint | 2–4 | 1.6M | 10 | 1.0 | 55.7% | 1.0% |
| UD | overlap | 2–4 | 1.6M | 10 | 1.0 | 70.5% | 8.8% |
| UD | disjoint | 2–4 | 2.0M | 10 | 1.0 | 84.5% | 27.9% |
| UD | overlap | 2–4 | 2.0M | 10 | 0.5 | 79.0% | 26.7% |
| UD | overlap | 2–4 | 2.0M | 10 | 1.0 | 77.2% | 25.7% |
| UD | overlap | 2–4 | 2.0M | 10 | 0.5 | 94.5% | 58.5% |
| UD | overlap | 2–4 | 2.0M | 10 | 1.0 | 92.9% | 55.5% |
| UD | overlap | 2–4 | 2.0M | 12 | 1.0 | 67.4% | 14.3% |
| UD | overlap | 2–4 | 2.4M | 12 | 0.5 | 57.0% | 5.3% |
| UD | overlap | 2–4 | 2.4M | 12 | 1.0 | 54.6% | 5.5% |
| UD | overlap | 2–5 | 2.0M | 10 | 1.0 | 50.7% | 1.0% |
| UD | overlap | 2–5 | 2.4M | 10 | 1.0 | 50.6% | 2.3% |
| UD | overlap | 2–5 | 2.4M | 10 | 1.0 | 41.6% | 4.8% |
| UD | disjoint | 2–5 | 2.5M | 10 | 1.0 | 44.0% | 2.0% |
| suffix | overlap | 2–5 | 2.5M | 10 | 1.0 | 50.1% | 8.1% |
| prefix | overlap | 2–5 | 2.5M | 10 | 1.0 | 81.7% | 50.8% |
| UD | overlap | 2–6 | 2.4M | 10 | 1.0 | 50.4% | 4.2% |
| prefix | overlap | 2–6 | 2.5M | 10 | 1.0 | 25.1% | 2.4% |
| prefix | overlap | 2–6 | 3.0M | 10 | 1.0 | 31.7% | 4.1% |
| prefix | overlap | 2–6 | 3.0M | 12 | 1.0 | 64.5% | 15.6% |

## Without CoT

### Common settings

- Model: `d512-L10-H8`
- Temperature: 1.0
- Evaluation samples: 1000

### IID

| Language | Assignment | Query | Train data | Decodable | Success |
|---|---|---|---:|---:|---:|
| Prefix | disjoint | 2–4 | 1M | 99.6% | 83.8% |
| Prefix | disjoint | 2–4 | 1.5M | 99.9% | 96.8% |
| Suffix | disjoint | 2–4 | 1M | 99.6% | 81.9% |
| Suffix | disjoint | 2–4 | 1.5M | 99.8% | 91.3% |
| UD | disjoint | 2–4 | 600k | 99.6% | 79.1% |
| UD | overlap | 2–4 | 600k | 99.7% | 82.7% |
| UD | overlap | 2–4 | 900k | 99.9% | 90.8% |

### Held-out pairs

| Language | Assignment | Query | Train data | Decodable | Success |
|---|---|---|---:|---:|---:|
| UD | disjoint | 2–4 | 900k | 100.0% | 92.7% |
| UD | disjoint | 2–4 | 900k | 100.0% | 80.7% |
| UD | disjoint | 2–5 | 1.2M | 100.0% | 79.4% |

### Cross-region

| Language | Assignment | Query | Train data | Decodable | Success |
|---|---|---|---:|---:|---:|
| UD | overlap | 2–4 | 900k | 99.2% | 16.1% |
| UD | overlap | 2–4 | 1.2M | 99.3% | 23.4% |
| UD | overlap | 2–4 | 1.5M | 98.6% | 35.9% |
| UD | overlap | 2–5 | 2.4M | 99.8% | 22.7% |
| UD | overlap | 2–6 | 3.6M | 98.8% | 25.8% |
| UD | disjoint | 2–4 | 900k | 96.8% | 18.7% |
| UD | disjoint | 2–4 | 1.2M | 99.8% | 40.0% |

## Standard prefix/suffix sweep — Without CoT (5 training seeds)

### Experiment setup

This section records the ordinary IID, disjoint-assignment prefix/suffix data-size sweep, separately from the paired-reversal experiment below. The model is `d512-L10-H8`, query length is 2–4, and CoT is disabled: `BOS + query_bits + _ + answer_bits + EOS`. Evaluation uses the ordinary test split, with 1000 generated samples at temperature 1.0.

The latent pool and encoding settings are fixed, and train subsets are nested across requested sizes. `Train pairs` counts query-pair occurrences, not unique pairs or examples.

All results are **mean ± sample standard deviation** across the five training seeds (`n=5`, `ddof=1`). Means are percentages; standard deviations are in percentage points.

### Prefix — Ordinary test

| Train pairs | Decodable (%) | Success (%) |
|---:|---:|---:|
| 600k | 99.82 ± 0.16 | 93.20 ± 2.71 |
| 800k | 99.92 ± 0.13 | 93.58 ± 3.59 |
| 1.0M | 99.98 ± 0.04 | 95.86 ± 1.17 |
| 1.2M | 99.90 ± 0.14 | 95.70 ± 1.76 |
| 1.4M | 99.94 ± 0.05 | 96.78 ± 0.81 |
| 1.6M | 99.94 ± 0.05 | 96.10 ± 1.64 |
| 1.8M | 99.98 ± 0.04 | 94.80 ± 1.68 |
| 2.0M | 99.96 ± 0.05 | 96.08 ± 1.08 |

### Suffix — Ordinary test

| Train pairs | Decodable (%) | Success (%) |
|---:|---:|---:|
| 600k | 99.82 ± 0.15 | 86.96 ± 3.64 |
| 800k | 99.86 ± 0.11 | 88.44 ± 2.89 |
| 1.0M | 99.94 ± 0.09 | 91.20 ± 2.20 |
| 1.2M | 99.88 ± 0.13 | 90.98 ± 3.88 |
| 1.4M | 99.96 ± 0.05 | 91.98 ± 2.84 |
| 1.6M | 99.90 ± 0.10 | 91.10 ± 4.97 |
| 1.8M | 99.86 ± 0.09 | 92.64 ± 1.64 |
| 2.0M | 99.96 ± 0.05 | 92.14 ± 1.46 |

### Prefix vs suffix — Success comparison

| Train pairs | Prefix success (%) | Suffix success (%) |
|---:|---:|---:|
| 600k | 93.20 ± 2.71 | 86.96 ± 3.64 |
| 800k | 93.58 ± 3.59 | 88.44 ± 2.89 |
| 1.0M | 95.86 ± 1.17 | 91.20 ± 2.20 |
| 1.2M | 95.70 ± 1.76 | 90.98 ± 3.88 |
| 1.4M | 96.78 ± 0.81 | 91.98 ± 2.84 |
| 1.6M | 96.10 ± 1.64 | 91.10 ± 4.97 |
| 1.8M | 94.80 ± 1.68 | 92.64 ± 1.64 |
| 2.0M | 96.08 ± 1.08 | 92.14 ± 1.46 |

## Paired prefix/suffix reversal — Without CoT

### Experiment setup

This is a separate IID, disjoint-assignment experiment that compares prefix and suffix under an exact reversal transformation, rather than independently generated codebooks. The model is `d512-L10-H8`, query length is 2–4, and the sample format is `BOS + query_bits + _ + answer_bits + EOS` (no CoT).

- Each suffix codeword is the bit reversal of the corresponding prefix codeword at the same vertex and codeword slot. Vertex assignments and the number of codewords per vertex are preserved.
- Suffix training and validation samples reverse both the waypoint order and the answer walk of the corresponding prefix sample. In encoded form, `Q_A` becomes `reverse(Q)_reverse(A)`; the query remains before the answer.
- **Natural test:** prefix uses its ordinary test set. Suffix uses the original latent test queries/walks in their original direction, encoded with the suffix codebook. This is not an independently sampled test pool.
- **Reversed test:** suffix is evaluated on the exact reversed prefix test samples, retaining their order and source identities. Natural suffix test samples that collide semantically with reversed full-pool train/validation are filtered by the data pipeline.
- The latent data/codebooks are fixed across training seeds: `data_seed = encoding_seed = eval_seed = 0`. Train subsets are nested; only the training RNG seed changes. Each evaluation generates 1000 samples at temperature 1.0.

### Aggregation

All cells below are **mean ± sample standard deviation** across the listed training seeds (`ddof=1`). Means are percentages; standard deviations are in percentage points. These are not confidence intervals or variability across independently generated datasets.

The corrected results contain 48 distinct training results: both prefix and suffix have training seeds `0, 1, 2` at every size (`n=3`). The previously supplied seed-3 results are excluded, and repeated seed-1 results are counted only once.

`Train pairs` below is the requested number of query-pair occurrences, not unique pairs or examples. The actual count is one higher at 600k, 1.2M, and 1.4M because selection retains whole samples; other sizes match the requested count.

### Prefix — Natural test

| Train pairs | Training seeds (n) | Decodable (%) | Success (%) |
|---:|---|---:|---:|
| 600k | 0, 1, 2 (3) | 99.83 ± 0.21 | 93.97 ± 2.08 |
| 800k | 0, 1, 2 (3) | 99.87 ± 0.15 | 94.90 ± 0.87 |
| 1.0M | 0, 1, 2 (3) | 99.97 ± 0.06 | 95.77 ± 1.63 |
| 1.2M | 0, 1, 2 (3) | 99.93 ± 0.12 | 96.70 ± 0.89 |
| 1.4M | 0, 1, 2 (3) | 99.93 ± 0.06 | 96.47 ± 0.90 |
| 1.6M | 0, 1, 2 (3) | 99.97 ± 0.06 | 96.33 ± 2.01 |
| 1.8M | 0, 1, 2 (3) | 99.97 ± 0.06 | 94.60 ± 1.61 |
| 2.0M | 0, 1, 2 (3) | 99.93 ± 0.06 | 95.77 ± 0.35 |

### Suffix — Natural and reversed tests

Both test views use the same trained suffix model for each seed and train size; they are not additional training runs.

| Train pairs | Training seeds (n) | Natural decodable (%) | Natural success (%) | Reversed decodable (%) | Reversed success (%) |
|---:|---|---:|---:|---:|---:|
| 600k | 0, 1, 2 (3) | 99.70 ± 0.10 | 86.10 ± 4.56 | 99.87 ± 0.15 | 86.10 ± 4.50 |
| 800k | 0, 1, 2 (3) | 99.73 ± 0.25 | 85.90 ± 7.08 | 99.77 ± 0.21 | 86.20 ± 6.68 |
| 1.0M | 0, 1, 2 (3) | 99.77 ± 0.15 | 88.17 ± 3.86 | 99.93 ± 0.06 | 87.73 ± 3.62 |
| 1.2M | 0, 1, 2 (3) | 99.80 ± 0.00 | 84.63 ± 2.35 | 99.80 ± 0.17 | 86.07 ± 2.57 |
| 1.4M | 0, 1, 2 (3) | 99.90 ± 0.10 | 92.03 ± 1.70 | 99.97 ± 0.06 | 92.63 ± 1.59 |
| 1.6M | 0, 1, 2 (3) | 99.93 ± 0.12 | 92.73 ± 4.32 | 99.97 ± 0.06 | 93.87 ± 3.00 |
| 1.8M | 0, 1, 2 (3) | 99.83 ± 0.06 | 93.03 ± 1.45 | 99.97 ± 0.06 | 92.97 ± 2.08 |
| 2.0M | 0, 1, 2 (3) | 99.87 ± 0.12 | 92.93 ± 4.05 | 99.87 ± 0.06 | 92.90 ± 2.89 |

### Prefix vs suffix — Success comparison

Success is reported as **mean ± sample standard deviation** across training seeds `0, 1, 2` (`n=3`).

| Train pairs | Prefix natural success (%) | Suffix natural success (%) | Suffix reversed success (%) |
|---:|---:|---:|---:|
| 600k | 93.97 ± 2.08 | 86.10 ± 4.56 | 86.10 ± 4.50 |
| 800k | 94.90 ± 0.87 | 85.90 ± 7.08 | 86.20 ± 6.68 |
| 1.0M | 95.77 ± 1.63 | 88.17 ± 3.86 | 87.73 ± 3.62 |
| 1.2M | 96.70 ± 0.89 | 84.63 ± 2.35 | 86.07 ± 2.57 |
| 1.4M | 96.47 ± 0.90 | 92.03 ± 1.70 | 92.63 ± 1.59 |
| 1.6M | 96.33 ± 2.01 | 92.73 ± 4.32 | 93.87 ± 3.00 |
| 1.8M | 94.60 ± 1.61 | 93.03 ± 1.45 | 92.97 ± 2.08 |
| 2.0M | 95.77 ± 0.35 | 92.93 ± 4.05 | 92.90 ± 2.89 |

## Notes

- `Train pairs` is the number of query-pair occurrences in the training split, not the number of unique vertex pairs. A query containing `L` waypoints contributes `L - 1` query pairs, and the same pair may occur in many samples.
- In the Without CoT tables, `Train data` preserves the original bit-based training-data amounts supplied for those experiments.
- `Decodable` is the percentage of generated final answers that can be completely segmented into codewords.
- `Success` is `semantic_success_pct`: the decoded answer starts and ends correctly, visits all requested waypoints in order, and every pairwise path segment is simple.
- Repeated rows with the same displayed configuration are retained when results were supplied from different trained model artifacts.
- `—` means that the value was not available in the recorded output and should not be inferred.
