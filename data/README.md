# `data/` — synthetic language engine and dataset generation

This package defines and generates the **synthetic language** the models are pre-trained on:
sentences are **walks on a hidden graph**, each step encoded by one **binary codeword** and
concatenated with no separators. Because we design the language, every string has an exact
ground truth (is it valid? which walk does it mean? what is the optimal next-bit distribution?),
so every model behaviour can be measured against an exact reference.

Research context, theory and design decisions: [`../docs/context.md`](../docs/context.md).

---

## 1. The language

* A finite undirected graph `G = (V, E)` — the hidden world. v1: square grids (`grid-4x4`,
  `grid-6x6`, `grid-8x8`), vertices `0 .. n²-1` in row-major order, 4-neighbourhood.
* Each vertex `v` owns a **codebook** `B_v ⊆ {0,1}⁺` of `k` binary codewords — what that vertex
  may "speak". The global code is `C = ⋃_v B_v`.
* The **meaning** of a sentence is a walk `W = v_1 v_2 … v_m` (consecutive vertices adjacent,
  `m ≥ 2`). A sentence encodes it by choosing one codeword per step and concatenating:
  `enc(W) = b_1 b_2 … b_m` with `b_i ∈ B_{v_i}`.
* The **language** is the set of *good strings*: strings that encode at least one walk.
  The grammar *is* the adjacency structure of `G`. The graph never appears in the data — the
  model must discover it from bit statistics alone.

### Worked example

Graph `grid-2x2`: vertices `0 1 / 2 3`, edges `0–1, 0–2, 1–3, 2–3` (note `0` and `3` are **not**
adjacent). Base code `C = {0, 10, 11}` (prefix-free), powered to `C² = {00, 010, 011, 100, 1010,
1011, 110, 1110, 1111}`, with `k = 2` disjoint codewords per vertex:

| vertex | `B_v` |
|---|---|
| 0 | `00`, `010` |
| 1 | `011`, `100` |
| 2 | `1010`, `1011` |
| 3 | `110`, `1110` |

Encode the walk `0 → 1 → 3 → 2` by picking `00` at 0, `100` at 1, `1110` at 3, `1011` at 2:

```
00 · 100 · 1110 · 1011   ->   sentence "0010011101011"   (13 bits, no separators)
```

Decoding recovers exactly one walk, `[0, 1, 3, 2]`, and one segmentation,
`00 | 100 | 1110 | 1011`. By contrast `00 · 110` ("go from 0 to 3") decodes to **nothing**: the
string is invalid because `0` and `3` are not adjacent. That is the only kind of grammaticality
this language has.

### The two difficulties (and why the code type matters)

Production is easy; **comprehension** is the hard part, and it splits into two independent
failure modes:

1. **Segmentation ambiguity** — where do the codeword boundaries lie? A property of `C` alone.
2. **Vertex ambiguity** — a codeword may belong to several vertices; resolving it needs the graph.
   Ruled out by **Theorem D.1**: with `C` uniquely decipherable, every good string determines its
   walk uniquely iff for every two distinct directed edges `(u,v) ≠ (u',v')`,
   `B_u ∩ B_u' = ∅` or `B_v ∩ B_v' = ∅`. Pairwise-disjoint codebooks (v1 `disjoint-random`)
   satisfy this trivially.

The three **code types** are three increasingly hard segmentation regimes
(`prefix-free ⊂ suffix-free-analogue ⊂ UD`):

| `code.type` | Property | What a reader must do | Difficulty |
|---|---|---|---|
| `prefix-free` | no codeword is a prefix of another | greedy, left-to-right: exactly one cut is alive at every position | easiest |
| `suffix-free` | no codeword is a *suffix* of another (and *not* prefix-free) | must look ahead: several cuts may be alive mid-sentence, resolved later | medium |
| `ud` | uniquely decipherable, but neither prefix- nor suffix-free | full lookahead / backtracking; ambiguity can persist for many bits | hardest |

All three are **uniquely decipherable**, so every sentence still has exactly one meaning — the
difficulty is purely in the work needed to find it. This shows up quantitatively in the
**entropy floor**: with a prefix-free code the optimal next-bit distribution is sharper than with
a UD code, where several candidate segmentations stay alive and blur the prediction.

All three code types go through the same mould (design option A): generate a small **base code**
of `base_size` words with the requested property, certify it, then use `C^x` (all concatenations
of exactly `x` base codewords) as the codeword pool. Prefix-freeness, suffix-freeness and unique
decipherability are all inherited by `C^x`, and the mould keeps codeword length distribution and
internal block structure identical across code types — so a comparison isolates exactly one
variable: the code property.

---

## 2. Config schema

The generator supports the original language-modeling task and a conditional
path-QA task. In path-QA, the question is a sequence of waypoint vertices and
the answer concatenates one simple graph path for every consecutive waypoint
pair. Vertices may repeat across different segments; only each individual
segment must be simple. By default the serialized training text is
`query_bits_answer_bits`. With `cot_mode: segments`, it is
`query_bits_segment_1_bits_..._segment_(L-1)_bits_answer_bits`: each rationale
field is the simple route for one consecutive waypoint pair, and the final
answer is their merge with duplicate boundary vertices removed. The exact same
sampled surface codewords are reused between rationale fields and the final
answer. `_` is the only separator; no extra token type is introduced.

```yaml
task:
  type: path-qa
  cot_mode: none                # none | segments
  query_len: [2, 4]
  segment_len: [2, 14]
  path_trials: 1000
  split_mode: iid                # iid | held-out-pairs | cross-region
  pair_direction: unordered      # [u,v] and [v,u] are the same held-out pair
  held_out_pairs: []             # required in test, excluded from train/valid
```

With arbitrary-overlap codebooks, query waypoints use private codewords so the
question is unambiguous. Answer paths may use every assigned codeword, including
shared ones. See `configs/path_qa_iid.yaml` and
`configs/path_qa_heldout.yaml`.

For compositional generalisation across two overlapping row regions, use:

```yaml
task:
  type: path-qa
  split_mode: cross-region
  region_overlap_rows: 2
  query_len: [2, 4]
  segment_len: [2, 14]
```

On a 6x6 grid this defines an upper region spanning human rows 1--4 and a
lower region spanning rows 3--6. Every train/valid example lives wholly inside
one region, including its answer paths. Test waypoints use only rows 1--2 and
5--6, have opposite start/end bands, and therefore contain at least one direct
cross-region query segment. Validation remains in-distribution and is suitable
for checkpoint selection; only test measures cross-region generalisation. See
`configs/path_qa_cross_region.yaml`.

One dataset = one YAML file + a seed. **Any change inside the `language` block is a different
language, hence a different dataset.** See [`configs/smoke_4x4_prefix.yaml`](configs/smoke_4x4_prefix.yaml).

```yaml
language:
  graph: grid-4x4                # hidden world; |V| = 16. World complexity.
  code:
    type: prefix-free            # prefix-free | suffix-free | ud. Segmentation difficulty.
    base_size: 3                 # |C| of the base code.
    depth_range: [1, 4]          # base codeword length range (use len_range for `ud`).
    power_x: 6                   # pool = C^x; codeword lengths are x * base lengths.
  k: 4                           # codewords per vertex. k > 1 forces the model to learn
                                 #   "vertex" as an abstraction instead of memorising strings.
  assignment: disjoint-random    # how B_v are drawn from the pool. Vertex ambiguity.
data:
  walk_len: [8, 32]              # vertices per walk, uniform in this range. Sentence length.
  pool_tokens: 2_000_000         # total bits in the frozen pool (dataset size D).
  split: [98, 1, 1]              # train/valid/test ratios, disjoint at string level.
  seed: 42                       # the only source of randomness.
  reverse_walks: false           # true: b1|...|bm -> bm|...|b1; bits inside bi unchanged.
  noise: null                    # {type, gamma, rho}; see section 6.
  context_len: 512               # training context: used for packing and length warnings.
```

Path-QA also supports a semantic dataset budget. Specify exactly one of:

```yaml
data:
  pool_tokens: 300_000_000       # encoded-bit budget
# or
data:
  pool_query_pairs: 1_000_000    # sum of L-1 over accepted QA samples
```

These fields are mutually exclusive. Both modes retain exact-sample
deduplication by complete `sample.bits`. A pair-budget split may exceed its
target by at most `query_len.max - 2`, because the last sample is kept whole.

For path-QA, `context_len` may be derived from the generated dataset:

```yaml
data:
  context_len: auto
```

The generator measures `BOS + query + separators + CoT + final answer + EOS`
for every sample, reports its min/mean/p95/p99/max, and recommends the observed
maximum rounded up to the nearest multiple of 64. The Modal path-QA launcher
uses this recommendation automatically; it no longer needs a context-length
CLI argument. Its checkpoint directory includes the derived value (for example
`..._ctx2688`), so an auto-context run cannot accidentally resume a checkpoint
created earlier with a manually selected context length.

For a paired directionality control, copy a config without changing its language
or seed and set `data.reverse_walks: true`. The generator first samples the same
walk and the same codeword at every step as the normal run, then reverses the
walk and codeword order. Individual codewords are not bit-reversed. Thus
`b1|b2|...|bm` becomes `bm|...|b2|b1`, with `cuts` recomputed accordingly.

`language.k` may also be an inclusive range. In that form, each vertex receives
an independently sampled, uniformly random number of codewords:

```yaml
language:
  k: [1, 4]                     # each |B_v| is sampled independently from {1,2,3,4}
```

The assignment is deterministic for a fixed config and `data.seed`. The pool
must contain at least `|V| * k_max` words so every possible draw is feasible.

To create genuine vertex ambiguity while retaining a uniquely recoverable
walk, use the draft's general support-set construction:

```yaml
language:
  k: [1, 4]                      # exact target |B_v|, sampled per vertex
  assignment: arbitrary-overlap
  overlap:
    support_size: [2, 4]         # a shared word belongs to 2..4 vertices
    shared_codewords: 12         # exact target; failure aborts rather than lowering it
    max_restarts: 100
    candidate_trials: 10_000
```

The constructor starts with one private singleton support per vertex, greedily
adds the requested shared supports, and fills unused vertex capacity with
private words. Every candidate support `A_j` is accepted only when
`|E_dir ∩ (A_j × A_k)| <= 1` in both orders for every existing support `A_k`.
The final codebooks are independently checked by the directed-edge-pair
certificate. `support_size`, `shared_codewords`, and `k` therefore control
different quantities: ambiguity per word, number of ambiguous words, and
lexical capacity per vertex respectively.

To replace the generated `C^x` pool with an existing ordered JSON array of
codewords, set `language.codeword_pool_file`:

```yaml
language:
  graph: grid-4x4
  codeword_pool_file: ../codeword_pools/my_prefix_pool.json
  code:
    type: prefix-free             # the complete external pool is certified
  k: 4
  assignment: disjoint-random     # assignment still happens normally
data:
  walk_len: [8, 32]               # the new dataset's walk distribution
  pool_tokens: 2_000_000
  split: [98, 1, 1]
  seed: 42
  noise: null
  context_len: 512
```

The canonical JSON format is just `["001", "1010", ...]`;
`{"codewords": [...]}` is also accepted. The loader preserves array order,
rejects duplicates/non-binary words and checks that the pool contains at least
`|V| * k_max` words. It then runs the normal assignment strategy and certifies the
assigned subset against `code.type`. Thus the file replaces only
`base -> C^x`; graph
construction, random assignment to vertices, language certification, walk
sampling, noise, splitting and output storage are unchanged. Relative paths are
resolved against the YAML file. `base_size`, length range and `power_x` do not
apply when `codeword_pool_file` is present.

Cross-constraints checked at load time (hard errors): `|C|^x ≥ |V| · k_max` (disjoint assignment must
be feasible), `base_size ≥ 2`, `power_x ≥ 1`, `walk_len` min `≥ 2`, valid graph spec, positive
split. Soft **warnings** are printed for: little pool slack (`< 4 ×` the required number of
codewords, i.e. no room for later overlap constructions) and a worst-case sentence length
exceeding `context_len` (only a warning because the actual codeword lengths depend on the sampled
base code — the build report prints the real range).

---

## 3. Generating a dataset

Requires Python ≥ 3.10 and `pyyaml` (`pip install -e .` or just `pip install pyyaml`).

```bash
cd data
python scripts/gen_dataset.py configs/smoke_4x4_prefix.yaml --out outputs/smoke
```

Pipeline: load config → generate base code → **certify** → power to `C^x` → assign codebooks →
**certify the language** → build the deduplicated pool → apply noise (if configured) → split →
save. Certification failure **aborts** the run: a dataset must never claim a regime it does not
have.

Useful flags:

```bash
--pool-tokens 200000     # override data.pool_tokens for a quick run (config value is kept in the manifest)
--pool-query-pairs 1000  # override data.pool_query_pairs for a quick QA run
--entropy-samples 64     # held-out sentences used to measure the entropy floor (0 = skip)
```

The script prints a human-readable report: the base code and its certification, `|C^x|` and the
codeword length range, the language certification (code regime, Theorem D.1, unique decoding),
pool and split sizes, and the entropy floor in bits/token.

### Output layout

```
outputs/smoke/
├── manifest.json    # config + config_hash + full certification report + stats (self-certifying dataset)
├── codebook.json    # graph + every B_v + walk_len + base code — enough to rebuild the Language alone
├── train.jsonl      # one record per line
├── valid.jsonl
└── test.jsonl
```

Each record carries the full ground truth, so later analyses never need to re-decode:

```json
{"bits":"1010000111...","noised_bits":null,"walk":[7,11,7,3,...],"cuts":[9,19,30,...]}
```

* `bits` — the clean sentence.
* `noised_bits` — the corrupted string, or `null` when noise is off. `bits` is **always** clean.
* `walk` — the ground-truth vertex sequence.
* `cuts` — bit offset just *after* each codeword, so `len(cuts) == len(walk)` and
  `cuts[-1] == len(bits)`.

For path-QA, each JSONL record stores `query_bits`, `answer_bits`,
`query_vertices`, `answer_walk`, `query_cuts`, `answer_cuts`, and
`segment_cuts`. Segment-CoT datasets additionally populate
`cot_segments_bits`; older records without the field load as non-CoT samples.
No textual QUESTION/ANSWER/COT tokens are used.

Reload everything with:

```python
from synthdata.storage import load_dataset
ds = load_dataset("outputs/smoke")
ds.config, ds.language, ds.splits.train, ds.manifest
```

---

## 4. Inspecting and decoding

```bash
# distributions, vertex frequencies, split sizes, split-disjointness check
python scripts/inspect.py outputs/smoke --validate 25

# decode an arbitrary bit string with the dataset's language
# (the string must belong to *this* dataset's language — every dataset has its own codebooks)
python scripts/decode.py --dataset outputs/smoke --bits "$(head -1 outputs/smoke/test.jsonl \
    | python -c 'import json,sys; print(json.load(sys.stdin)["bits"])')"

# decode dataset samples and compare against ground truth, with the oracle's next-token distribution
python scripts/decode.py --dataset outputs/smoke --sample 3 --oracle
```

`decode.py` prints every consistent walk (capped at 100), the segmentation of the first one, and
`INVALID` (exit code 1) for a string that encodes no walk. This decoder is the **exact scoring
instrument** for all model evaluations.

> Note: `scripts/inspect.py` shadows the stdlib `inspect` module for anything run *from inside*
> `scripts/`. The scripts remove their own directory from `sys.path` to protect themselves; run
> them from `data/` as shown above.

### The API in three calls

```python
import random
from synthdata.storage import load_dataset

ds = load_dataset("outputs/smoke")
lang = ds.language

lang.sample(None, random.Random(0))   # -> Sample(bits, walk, cuts): draw a sentence
lang.decode(bits)                     # -> [[v1, v2, ...], ...]: all consistent walks ([] = invalid)
lang.is_valid(bits)                   # -> bool, with early exit
lang.next_bit_dist(prefix)            # -> {'0': p, '1': p, 'EOS': p}: the exact optimal oracle
lang.entropy_floor(samples)           # -> Bayes-optimal cross-entropy in bits/token
lang.certify()                        # -> LanguageReport (the same one stored in the manifest)
```

**Certifier vs decoder** — two distinct roles. The *certifier* checks the design once at build
time ("does this code/codebook actually have the property it claims?") and its report is stored in
the manifest, so each dataset carries its own proof of regime. The *decoder* checks one string at
eval time ("valid? which walk?") and is the scoring instrument. One general DP decoder is used
for every regime, so all regimes are scored by the same standard.

**Entropy floor.** The language is Markov over states (step index, vertex, offset inside the
current codeword). Running the parse forward with the sampler's own probabilities (walk length
uniform in `walk_len`, first vertex uniform, next vertex uniform among neighbours, codeword
uniform in `B_v`) gives the **exact** Bayes-optimal next-token distribution — verified to agree
with the analytic sentence probability to floating-point precision. Model loss should be compared
to this floor ("distance to perfect understanding"), not read raw, because the floor differs
across languages. The only assumption is that the graph has no dead ends (true for grids), which
makes every continuation of a live parse state completable.

---

## 5. Tokenisation and the two data modes

Tokenisation is **bit-level**: vocabulary `{0, 1, BOS, EOS, PAD}` (5 tokens) for
language modeling, or `{0, 1, BOS, EOS, PAD, _}` (6 tokens) for path-QA. Codeword-level
tokens would hand the segmentation to the model for free — exactly the difficulty under study.
Training format is `BOS + bits + EOS`, sentences concatenated and cut into fixed context windows
(sentences are *not* aligned to window boundaries):

```python
from synthdata.tokenizer import BitTokenizer
tok = BitTokenizer()
windows = list(tok.pack(ds.splits.train, context_len=512))
```

**Frozen pool (default).** A deduplicated pool controlled by `pool_tokens`, or for
path-QA by `pool_query_pairs`, split disjointly at string level and frozen to disk
with a config hash. For memorisation / generalisation studies. For
scaling in dataset size `D`, use **nested subsets** so that the small run's data is a prefix of
the large run's:

```python
from synthdata.dataset import nested_subsets
subsets = nested_subsets(ds.splits.train, [10_000, 100_000, 1_000_000])  # in bits
```

**Streaming (infinite data).** Fresh samples from the seeded generator, nothing stored — for
scaling laws at large `D`:

```python
from synthdata.dataset import stream
for sample in stream(ds.language, ds.config.data, seed=123):
    ...
```

---

## 6. Noise (hooks in place, experiments later)

A transform layer after the sampler. `gamma` = fraction of corrupted samples, `rho` = per-unit
corruption rate inside a corrupted sample. The clean `bits`, `walk` and `cuts` are always
preserved; the corrupted string goes to `noised_bits`.

| `noise.type` | Effect |
|---|---|
| `bit-flip` | surface noise; keeps the length and the segmentation frame |
| `bit-delete` | drops bits; breaks the frame and shortens the sentence |
| `vertex-noise` | resamples walk vertices *before* re-encoding: a clean encoding of an invalid walk (semantic-level noise) |

```yaml
data:
  noise: {type: bit-flip, gamma: 0.5, rho: 0.1}
```

Default is `null`. This mirrors Experiment 6 of the CFG paper (robust pre-training,
corrupted-prefix evaluation, mode switching, temperature sweeps).

---

## 7. Determinism

Everything derives from `(config, seed)`. There is **no global random state**: a
`random.Random(seed)` is threaded explicitly through every stage, and `gen_dataset.py` derives an
independent sub-stream per stage (`code`, `assignment`, `pool`, `noise`, `split`) from the single
config seed. The same config and seed reproduce the base code, the codebooks and the pool
**bit-exactly**; `config_hash` (sha256 of the canonical JSON of the config, 16 hex chars) is
stored in the manifest so a dataset can always be traced back to its config.

Two caveats worth knowing:

* `--pool-tokens` changes the pool size but not the sampling stream, so a smaller pool is a
  **prefix** of a larger one (before splitting). The override is recorded in the manifest.
* Changing the code *generator implementation* changes what a given seed produces. That is why
  `codebook.json` stores the resulting codebooks explicitly: reloading a dataset never re-runs a
  generator.

---

## 8. Package map

| Module | Role |
|---|---|
| `synthdata/config.py` | dataclass schema, YAML loading, cross-constraint validation, `config_hash` |
| `synthdata/graphs.py` | `GridGraph`, `make_grid` |
| `synthdata/codes/model.py` | `Code`: word set, `power(x)`, `sample_subset` |
| `synthdata/codes/generate.py` | the *only* code-type branch: tree pruning, reversal, UD rejection sampling |
| `synthdata/codes/certify.py` | `is_prefix_free`, `is_suffix_free`, `sardinas_patterson`, `CodeReport` |
| `synthdata/codebooks.py` | disjoint/arbitrary-overlap assignments + support/Theorem D.1 certification |
| `synthdata/language.py` | sampler, general DP decoder, exact next-bit oracle, entropy floor, certifier |
| `synthdata/noise.py` | `bit-flip`, `bit-delete`, `vertex-noise` transforms |
| `synthdata/tokenizer.py` | bit-level tokenizer and context-window packing |
| `synthdata/dataset.py` | pool building, splitting, nested subsets, streaming |
| `synthdata/storage.py` | `save_dataset` / `load_dataset`, on-disk format |

Adding a new **code type**, **assignment strategy** or **graph family** touches exactly one of
these modules; everything downstream never inspects the code type.

## Shared latent path-QA pools and data-size comparisons

Use a codebook-independent pool when comparing prefix/suffix/UD or tuning train
size. Query and path sampling retain the existing uniform waypoint proposals and
randomized DFS; no pair-frequency scoring or forced histogram balancing is added.
Rejections and semantic deduplication mean the accepted distribution is not
necessarily uniform. Histograms measure the actual distribution.

From `data/`, first generate the largest semantic pool:

```bash
python scripts/gen_qa_latent.py configs/path_qa_cross_region.yaml \
  --out outputs/latent_cross_q2-4_seed0 \
  --train-query-pairs 3_000_000 \
  --valid-query-pairs 100_000 \
  --test-query-pairs 100_000 \
  --checkpoints 500_000 1_000_000 2_000_000 3_000_000
```

The pool contains `train.jsonl`, `valid.jsonl`, `test.jsonl` and `manifest.json`.
Rows store `sample_id`, `query_vertices`, `answer_walk`, `segment_cuts`. Segment
boundaries reconstruct every simple segment, including its repeated boundary
vertex. There are no codewords. Graph and task settings come from the YAML;
encoded budgets, assignment and code type do not enter semantic sampling.

Validation is generated first, then test, then train, with independent deterministic
RNG streams. Semantic query/final-walk duplicates are rejected within and across
splits. For a fixed seed, graph, task and fixed validation/test budgets, generating
500k train pairs directly produces exactly the prefix selected from a 3M pool.
Changing validation/test budgets defines a different experiment and may change
train because reserved eval samples are excluded. Python's process-randomized
`hash()` is not used. This is a new versioned stream, not a recreation of old datasets.

Encode a smaller train prefix and all of the fixed validation/test sets:

```bash
python scripts/gen_dataset.py configs/path_qa_cross_region.yaml \
  --out outputs/runtime_ud_cross_500k \
  --latent-pool outputs/latent_cross_q2-4_seed0 \
  --train-query-pairs 500_000 \
  --encoding-seed 0
```

Here `--train-query-pairs` counts **train only**; legacy `--pool-query-pairs` counts
the total pool before the configured split. Do not combine them. Encoded-bit
budgets and the YAML split percentages are ignored when reading a latent pool.
Whole samples are retained: a query of length L contributes L-1 pair occurrences,
so the selected train budget may exceed the target by at most `query_max - 2`.

Each sample's encoding RNG is derived from `encoding_seed` and its stable ID.
Increasing train size keeps shared samples' bits unchanged. CoT may be enabled or
disabled without changing semantic samples or codeword choices. Changing graph,
query length, segment constraints or split mode requires a different latent pool;
the encoder checks these settings and source checksums.

To compare prefix/suffix/UD, make YAMLs with identical graph/task settings and
encode the same pool and prefix. They must only differ in language/encoding.
Use the first encoded dataset's `codebook.json` as `--assignment-template` on the
other runs to preserve every vertex's ordered codeword slots and overlap supports:

```bash
python scripts/gen_dataset.py YOUR_MATCHING_SUFFIX_CONFIG.yaml \
  --out outputs/runtime_suffix_cross_500k \
  --latent-pool outputs/latent_cross_q2-4_seed0 \
  --train-query-pairs 500_000 \
  --assignment-template outputs/runtime_ud_cross_500k/codebook.json
```

The template relabels words using a certified subset of the new codeword pool;
it does not import the old bit strings. Assignment type and k range must match.

Reports are saved in:

- Latent `manifest.json`: `stats` for each split and `train_checkpoints` for requested prefixes.
- Encoded `manifest.json`: `stats.semantic_splits`, plus `stats.latent_pool` provenance.

Reports include exact query/segment/final-path vertex-length histograms, unique
queries, pair occurrences, vertex/endpoint frequencies, and directed/undirected
query-pair and graph-edge coverage. Frequency histograms include zero counts;
unseen pairs/edges and individual frequencies are listed. Query-pair denominators
follow split endpoint restrictions, rather than asserting simple-path feasibility
for every endpoint pair. Held-out pair coverage and cross-region subgroups are
reported separately. Graph edges are counted once on the final walk, without
double-counting the CoT fields. Encoded bits and full-sequence lengths retain their
existing reports.

Each pair/edge coverage report also includes `total_occurrences`, per-item
`relative_frequencies` (fractions summing to 1 within the report's eligible set
when occurrences are nonzero), and count summaries both including zero-frequency
items and for observed items only: min, median, mean, p95, max, population standard
deviation and coefficient of variation (`cv = std / mean`). The exact
`frequency_histogram_including_zero` maps occurrence counts to the number of
pairs/edges having that count. `binned_frequency_histogram` groups positive counts
into at most 10 ranges, keeping zero separate; this compact version is printed in
generation logs for both directed and undirected pairs/edges. Full details are
stored for every split and requested latent train checkpoint. Existing manifests
are not updated automatically; regenerate the report/data to get these fields.

### Modal

The local, git-ignored `app_path_qa.py` accepts the same workflow. For example:

```bash
python -m modal run --detach app_path_qa.py \
  --code-type ud --assignment overlap --qa-mode cross-region \
  --cot-mode segments --query-min 2 --query-max 4 \
  --train-query-pairs 500_000 --latent-train-pairs 3_000_000 \
  --valid-query-pairs 100_000 --test-query-pairs 100_000
```

Keep `--latent-train-pairs 3_000_000` fixed and change only `--train-query-pairs`
to 1M/2M/3M or `--code-type` to reuse the same cached latent pool. Its automatic
Volume path depends on semantic settings/seed/eval budgets/maximum pool size,
not code type, assignment or CoT. Alternatively, pass `--latent-pool latent_pools/NAME`
to select a Volume pool explicitly. Create it in one job before launching parallel
encodings against the same named pool. `--assignment-template` accepts a Volume
`codebook.json` path to preserve slot/support structure across languages.

`--data-only` skips train/eval and persists histograms/coverage and codebooks under
`/mnt/synthetic-language/data_reports/`. Ordinary training runs persist them beside
the checkpoint. The latent pool itself is stored on the Volume. Removing
`--data-only` enables the normal training/evaluation workflow. Push the data-package
changes before running the local Modal launcher, since its image clones GitHub.
The launcher pins the local Git commit so a new pushed revision refreshes the
cached source automatically; uncommitted source changes are not uploaded.

### Server sweep: prefix/suffix, 0.6M to 2M train pairs

From the repository root, use `bash scripts/sweep_path_qa.sh`. It defaults to
cross-region, overlap, segment CoT, query length 2–4, seed 0, and base model
512/10/8 on CUDA. It reserves 100k pairs each for fixed validation and test,
generates a latent pool of **3M train pairs**, and trains at 600k, 800k, 1M, 1.2M,
1.4M, 1.6M, 1.8M and 2M train pairs. These budgets count occurrences, not unique
pairs. Actual counts may exceed a boundary by a few pairs to retain whole samples.

Install the packages in the server environment once:

```bash
python -m pip install -e ./data -e ./model
```

First inspect the plan without starting work:

```bash
bash scripts/sweep_path_qa.sh --dry-run
```

Generate only the shared pool first:

```bash
bash scripts/sweep_path_qa.sh --stage pool
```

Then start all 16 train/evaluate runs using that pool:

```bash
bash scripts/sweep_path_qa.sh --stage all
```

Keep the same experiment arguments in both invocations. For IID without CoT and
disjoint assignment, for example, pass `--qa-mode iid --assignment disjoint
--cot-mode none` to both commands. Use `--n-layers 12` to change model depth.
`--train-min`, `--train-max`, `--train-step`, `--pool-train-pairs`,
`--valid-query-pairs` and `--test-query-pairs` accept integer pair counts.

An explicit output directory makes server management convenient:

```bash
bash scripts/sweep_path_qa.sh --stage pool --output-root /srv/experiments/qa_prefix_suffix
bash scripts/sweep_path_qa.sh --stage all --output-root /srv/experiments/qa_prefix_suffix
```

Without `--output-root`, outputs live under `model/outputs/path_qa_sweeps/` in a
directory derived from the settings. It contains `latent_pool/`, runtime YAMLs in
`configs/`, encoded subsets in `datasets/`, checkpoints/evaluation reports in
`runs/`, per-stage output in `logs/`, and consolidated results in `results.json`.
Each result includes requested/actual train pairs, model settings, temperature,
decodability and semantic success. Training defaults to 20k optimizer steps per
run; runs execute sequentially (all prefix sizes, then all suffix sizes).

Within each code type, all train sizes use the same codebooks and bit-identical
validation/test files; the script verifies their checksums. Suffix assignments
reuse the first prefix dataset's vertex slots/supports. Both encodings consume
the same latent sample IDs. Context length comes from each encoded dataset's
observed maximum. Existing complete datasets/pools are reused, interrupted
training resumes from `latest.pt`, and completed training/evaluation stages are
skipped on rerun. A conflicting configuration in an existing output directory
raises an error instead of overwriting it. To start a distinct experiment, choose
a new output root (or let the default settings-derived directory change).

Use `--code-types prefix` or `--code-types suffix` to train only one code type;
suffix-only still prepares the small prefix reference needed for slot matching.
Alternatively invoke `python scripts/sweep_path_qa.py` with the same options;
the shell wrapper uses `${PYTHON:-python}` from the active environment.

To let each run train longer until validation stops improving, increase
`--max-steps` and enable validation early stopping:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/sweep_path_qa.sh --stage all \
  --qa-mode iid --assignment disjoint --cot-mode none \
  --latent-pool ./data/outputs/latent_iid_q2-4_seed0_3m \
  --max-steps 100000 --batch-size 64 \
  --eval-every 250 --eval-batches 20 \
  --early-stopping-patience 20 --early-stopping-min-delta 0.0001 \
  --output-root ./model/outputs/path_qa_sweeps/iid_disjoint_none_bs64_earlystop
```

This uses the existing shared pool and the default 0.6M–2M sweep. The sampler and
step-based training remain unchanged. Patience counts validation checks; here,
20 failed checks normally span 5000 steps. The minimum required improvement is
0.0001 bits/token. `best.pt` always tracks raw lowest validation loss. The test
split is not used to decide when to stop. `results.json` includes actual `steps`,
`max_steps`, `stopped_early`, `stop_reason` and best validation loss along with
decodability/success. Use a new output root when changing batch/stopping settings;
the same latent pool remains reusable. The cosine LR schedule still decays over
the configured `max_steps` horizon.

### Adding UD and W&B to the server sweep

`--code-types prefix suffix ud` runs all three encodings; the default remains
prefix/suffix for compatibility. UD imports `data/codeword_pools/hard_ud_codebook_200.json`
by default; override it with `--ud-pool /absolute/path/to/another_pool.json`. The
same prefix assignment template preserves per-vertex slots and overlap supports
for suffix and UD. `--code-types ud` trains only UD, preparing one small prefix
dataset solely as the assignment reference. Existing latent pools are reusable.

For optional W&B tracking, install `wandb` and authenticate once with `wandb login`
before starting an online sweep:

```bash
nohup env CUDA_VISIBLE_DEVICES=2 bash scripts/sweep_path_qa.sh \
  --stage all --code-types prefix suffix ud \
  --qa-mode iid --assignment disjoint --cot-mode none \
  --latent-pool ./data/outputs/latent_iid_q2-4_seed0_3m \
  --max-steps 100000 --batch-size 64 \
  --eval-every 250 --early-stopping-patience 20 \
  --early-stopping-min-delta 0.0001 \
  --wandb --wandb-project synthetic-language --wandb-group iid-disjoint-none \
  --output-root ./model/outputs/path_qa_sweeps/iid_disjoint_none_ud_wandb \
  > sweep_iid_disjoint_none_ud_wandb.log 2>&1 < /dev/null &
```

With the default budgets this launches 24 sequential training/evaluation runs
(8 sizes for each code type). Each size/code type has its own W&B run, grouped by
`--wandb-group` (defaults to the output-root directory name); names are e.g.
`ud_600000pairs`, and code type/task/assignment/CoT are attached as tags. Optional
flags include `--wandb-entity`, `--wandb-tags` and `--wandb-mode offline`.
Training/early-stopping curves and final test decodability/success are logged;
local `results.json`, checkpoints, histograms and reports are still generated.
Resume uses the saved W&B ID in each run directory. Do not put an API key on the
command line; configure credentials in the server environment or via `wandb login`.
