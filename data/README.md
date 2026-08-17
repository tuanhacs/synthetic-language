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
  noise: null                    # {type, gamma, rho}; see section 6.
  context_len: 512               # training context: used for packing and length warnings.
```

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

Tokenisation is **bit-level**: vocabulary `{0, 1, BOS, EOS, PAD}` (5 tokens). Codeword-level
tokens would hand the segmentation to the model for free — exactly the difficulty under study.
Training format is `BOS + bits + EOS`, sentences concatenated and cut into fixed context windows
(sentences are *not* aligned to window boundaries):

```python
from synthdata.tokenizer import BitTokenizer
tok = BitTokenizer()
windows = list(tok.pack(ds.splits.train, context_len=512))
```

**Frozen pool (default).** A deduplicated pool of `pool_tokens` bits, split disjointly at string
level and frozen to disk with a config hash. For memorisation / generalisation studies. For
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
