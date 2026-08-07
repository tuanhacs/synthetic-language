# Project Context: Synthetic Language and Thinking

## 1. Research Goal

Study the internal mechanisms of LLMs ("physics of language models" / interpretability) on **small models** by replacing natural language with a **fully controlled synthetic language**. Because we design the language, we know exactly: the hidden "world" that generates every sentence, the grammar, and the ground-truth answer for every string. Every model behavior is therefore measurable against an exact reference.

The approach is in the same spirit as *Physics of Language Models, Part 1* (Allen-Zhu & Li, arXiv:2305.13673 — CFG-based), but uses a language we designed ourselves: **graph walks encoded through binary codebooks**. Detailed reproduction notes for the CFG paper's experiments (generation accuracy, diversity, robustness) exist in the repo and serve as a methodological template.

There is **no supervised task**. The model is pre-trained purely autoregressively on sentences of the language. At evaluation, it generates freely (from BOS, or completing a valid prefix), and the output is scored by an exact decoder. The graph never appears in the input/output — it is the hidden world the model must discover from bit-stream statistics alone.

## 2. The Language (from the theory draft)

Source: `draft_synthetic_language_and_thinking.pdf` (top-level of the repo).

- A finite simple undirected graph `G = (V, E)`. The "meaning" of a sentence is a **walk** on `G` (vertex sequence, consecutive vertices adjacent, length ≥ 2).
- Each vertex `v` carries a **codebook** `B_v ⊆ {0,1}+` — the binary codewords that vertex may "speak".
- A sentence encodes a walk by choosing one codeword per step and **concatenating with no separators**: `enc(W) = b_1 b_2 … b_m`.
- The language = the set of **good strings** (strings that encode at least one walk). The grammar IS the adjacency structure of `G`.
- The global codebook is `C = ⋃_v B_v`.

Difficulty lies in **comprehension**, not production, and splits into two independent failure modes:

1. **Segmentation ambiguity** — a bit string may admit several decompositions into codewords. Property of `C` alone. Ruled out by three increasingly general conditions: prefix-free ⊂ suffix-free-analogue ⊂ **uniquely decipherable (UD)**. UD is decidable by the **Sardinas–Patterson algorithm**.
2. **Vertex ambiguity** — a codeword may belong to several vertices; resolving it requires the graph. **Theorem D.1**: with `C` UD, every good string uniquely determines its walk iff for every two distinct directed edges `(u,v) ≠ (u',v')`: `B_u ∩ B_u' = ∅` or `B_v ∩ B_v' = ∅`. Corollary: adjacent vertices must have disjoint codebooks — overlap is only permitted between non-adjacent vertices.

Key facts used in the design:

- If `C` is UD, then `C^n` (concatenations of exactly n codewords) is UD (draft Prop E.2). The same preservation holds for prefix-free and suffix-free (easy induction on the first block).
- The draft's running realization uses base code `C = {0, 01, 110}` — deliberately UD but neither prefix-free nor suffix-free — powered to `C^10` on an 8×8 grid.
- The draft treats prefix-free only as a theoretical condition; it never constructs prefix-free datasets. The prefix/suffix/UD comparison as three difficulty levels of segmentation is **our extension**.
- Beyond-UD regime (graph repairing segmentation ambiguity, draft §G): **out of scope for v1** (no controlled generation procedure yet). The general decoder already handles it, so adding it later is just a new code generator.

## 3. Language Design Space (experiment dials)

A language is determined by 4 independent axes:

| Axis | Values (v1) | Controls |
|---|---|---|
| Graph `G` | grid 4×4, 6×6, 8×8 | world complexity |
| Code type of base `C` | prefix-free, suffix-free, UD (not prefix/suffix-free) | segmentation difficulty |
| Codebook assignment `B_v ⊆ C^x` | disjoint-random (v1); overlap constructions later | vertex ambiguity |
| Walk distribution | uniform random walk, length uniform in [min, max] | sentence length / entropy |

**Design decision (option A):** all three code types go through the same mold — generate a small **base code** (e.g., 3 codewords) with the desired property, certify it, then take `C^x` as the codeword pool. This keeps codeword length distribution and internal block structure identical across code types, so comparisons isolate exactly one variable: the code property. (Prefix/suffix-free could be generated directly at any size by tree pruning, but that would confound length/structure with code type when compared against UD, which *must* use the power construction.)

Number of codewords: grid n×n has n² vertices; each vertex gets **k** codewords (k > 1 forces the model to learn "vertex" as an abstraction rather than memorizing surface strings). Requires `|C|^x ≥ n² · k`, with generous slack (subset sampled randomly from the pool; room needed later for overlap).

## 4. Per-Code-Type Generation and Certification

The ONLY part of the pipeline that branches by code type:

- **prefix-free**: prune a random binary tree — repeatedly pick a leaf and either keep it as a codeword or expand it into two children, until `base_size` leaves within `depth_range`. Leaves of a pruned tree form an antichain ⇒ prefix-free by construction. Certify: pairwise prefix check.
- **suffix-free**: generate prefix-free as above, then reverse every codeword. Reject and retry if the result is *still* prefix-free (symmetric codes) — the two types must be genuinely distinct. Certify: pairwise suffix check + must fail prefix-free check.
- **UD (not prefix-free, not suffix-free)**: rejection sampling — random codewords with lengths in `len_range`, accept when Sardinas–Patterson passes AND prefix-free fails AND suffix-free fails. Fallback after `max_tries`: the known code `{0, 01, 110}`. Certify: Sardinas–Patterson.

Everything downstream (power `C^x`, codebook assignment, sampling, decoding, entropy oracle, noise, storage) is a single shared pipeline that never inspects the code type.

**Certifier vs decoder** (distinct roles): the certifier checks the *design* once at dataset build time ("does this code/codebook have the property it claims?") and its report is stored in the dataset manifest — the dataset carries its own proof of regime. The decoder checks *one string* at eval time ("is this valid? which walk?") and is the scoring instrument for all evaluations. Both are kept even though generators are correct-by-construction: the certifier catches implementation bugs (the proof is about the algorithm on paper, the certifier checks the actual artifact), is required inside some generators anyway (rejection conditions), and provides provenance independent of code version.

## 5. Core Algorithms

**Sampler**: random walk on `G` (uniform neighbor choice), walk length uniform in `[min, max]`; at each vertex choose a codeword uniformly from `B_v`; concatenate. Ground truth (walk + cut positions) recorded for free.

**Decoder / validator** (one general DP covering all regimes):

```
states = { (0, v) : some codeword of B_v is a prefix of x }     # via trie
from state (i, u): for every w adjacent to u, every b ∈ B_w
    with x[i : i+len(b)] == b  →  add state (i+len(b), w)
valid  ⇔  some state (len(x), ·) is reachable
```

Handles segmentation ambiguity and vertex ambiguity simultaneously. Backtracking recovers the **set of all consistent walks**. Complexity O(len · |V| · k · maxlen). Prefix-free codes would admit a simpler greedy decoder, but we use the single general DP everywhere — one instrument, one scoring standard. The greedy-vs-lookahead difference is the *experimental variable* (what the model must learn), not an infrastructure branch.

**Entropy oracle**: the language is Markov over states (vertex, offset within codeword). Running the same DP state set forward with sampler-exact probabilities gives the **exact optimal next-bit distribution**, hence the cross-entropy floor of each language. This matters for scaling laws: model loss is compared to the Bayes-optimal floor per language ("distance to perfect understanding"), not raw loss — floors differ across code types (with UD-not-prefix-free codes, multiple candidate segmentations can be temporarily alive mid-sentence, blurring the next-bit distribution; with prefix-free exactly one cut is alive at all times).

**Certification suite**: `is_prefix_free`, `is_suffix_free`, `sardinas_patterson` (on codes); Theorem D.1 check over all directed-edge pairs (on codebook assignments). All reports go into the manifest.

## 6. Dataset Design

**Two modes:**
- **Frozen pool (default)**: generator fully deterministic from `(config, seed)`. Generate a large pool, dedup at string level, split train/valid/test **disjoint at string level**, freeze to disk with config hash. For memorization/generalization studies.
- **Streaming (infinite-data)**: fresh samples from the seeded generator, nothing stored. For scaling laws at large D.

For scaling in D: **nested subsets** (the 10M-token set is a prefix of the 100M set, etc.) so runs are comparable.

**Tokenization is bit-level**: tokens are `0`, `1`, plus `BOS/EOS/PAD` (~6-token vocab). Codeword-level tokens would give segmentation away for free. Training format: `BOS + bits + EOS`, sentences concatenated and cut into fixed context windows (as in the CFG paper).

**Sample record** stores full ground truth: `{bits (clean), noised_bits|null, walk, cuts}` — no re-decoding needed for later analyses.

**Noise (future experiments, hooks built in now)**: a transform layer after the sampler, config `noise: {type, gamma, rho}` where `gamma` = fraction of corrupted samples, `rho` = per-bit corruption rate within a corrupted sample. Types: `bit-flip` (surface), `bit-delete` (breaks segmentation frame, shortens), `vertex-noise` (replace a walk vertex before encoding — semantic-level noise, clean encoding of an invalid walk). Mirrors Experiment 6 of the CFG paper (robust pre-training, corrupted-prefix eval, mode-switch phenomenon, temperature sweeps). Default: off.

**Config schema** (one dataset = one YAML + seed; any change to the `language` block = a different language = a new dataset):

```yaml
language:
  graph:        grid 4x4            # |V| = 16
  code:
    type:       prefix-free         # prefix-free | suffix-free | ud
    base_size:  3
    depth_range: [1, 4]             # (len_range for ud)
    power_x:    6
  k:            4                   # codewords per vertex
  assignment:   disjoint-random
data:
  walk_len:     [8, 32]             # vertices per walk
  pool_tokens:  10_000_000
  split:        [98, 1, 1]
  seed:         42
  noise:        null                # {type, gamma, rho}
```

Constraints: `|C|^x ≥ |V| · k` with slack; codeword bit-length in `[x·min_len, x·max_len]` (e.g., x=6 with base lengths 1–3 → 6–18 bits); sentence length ≈ walk_len × avg codeword length must fit the training context (512).

**Core language matrix (planned)**: 3 grids × 3 code types = 9 core languages (overlap axis multiplies later), each with nested pools. Start with a single smoke config (4×4, prefix-free) to validate the pipeline end-to-end.

## 7. Model Side (later phase, folder `model/`)

A family of modern decoder-only transformers (Llama-style: RoPE, RMSNorm, SwiGLU, no biases, weight tying), **one architecture scaled purely by config** (~0.5M → 100M params) to search for scaling laws in (N params, D tokens, language difficulty). Vocab is ~6 tokens, so tiny models are meaningful. No GQA/MoE — keep the architecture clean so results are attributable.

## 8. Evaluation Plan (later phase)

- **Validity**: generate from scratch (cut=0) and complete valid prefixes (cut=k), multinomial sampling τ=1, score with the exact decoder — analogous to CFG-paper generation accuracy.
- **Loss vs entropy floor**: per-language gap to Bayes-optimal.
- **Diversity**: does the model cover the language or repeat a small set (birthday-paradox collision counting).
- **Memorization vs rule learning**: n-gram overlap with train, copying analyses (spirit inherited from the old routing-code diagnostics).
- **Robustness** (with noise datasets): robust accuracy on corrupted prefixes, mode-switch, temperature sweeps.
- **Scaling experiments**: difficulty sweep across the 4 dials; scaling laws of gap-to-floor.
- (Extension) **Probing**: where inside the model are segmentation boundaries and the current vertex represented.

## 9. Repo Layout

```
synthetic-language-and-thinking/
├── docs/
│   └── context.md        # this file
├── data/                 # data codebase: language engine + dataset generation
└── model/                # model codebase (later): architecture, training, eval
```

## 10. Decisions Log

- Drop the old supervised routing task entirely; pure autoregressive LM only.
- v1 code types: prefix-free, suffix-free, UD. Non-UD (regime G) postponed — no controlled generation procedure yet.
- Option A: all code types via base → `C^x` for clean cross-type comparison.
- v1 assignment: disjoint-random only; overlap constructions (harmonious colouring, sparse pairing, greedy maximal — draft §F) are a later axis. All are graph-conditions only (Theorem D.1) and work identically for any UD code, so no per-type branching is ever needed.
- One general DP decoder for all regimes; no per-type decoding paths.
- Certifier reports stored in dataset manifests (self-certifying datasets).
- Noise hooks (schema + record fields) built into v1; noise experiments themselves are a later phase.
- No unit tests requested for the data codebase.
- Everything written in English.
