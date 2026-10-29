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

## Notes

- `Train pairs` is the number of query-pair occurrences in the training split, not the number of unique vertex pairs. A query containing `L` waypoints contributes `L - 1` query pairs, and the same pair may occur in many samples.
- In the Without CoT tables, `Train data` preserves the original bit-based training-data amounts supplied for those experiments.
- `Decodable` is the percentage of generated final answers that can be completely segmented into codewords.
- `Success` is `semantic_success_pct`: the decoded answer starts and ends correctly, visits all requested waypoints in order, and every pairwise path segment is simple.
- Repeated rows with the same displayed configuration are retained when results were supplied from different trained model artifacts.
- `—` means that the value was not available in the recorded output and should not be inferred.
