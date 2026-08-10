# `model/` — transformer family, training, evaluation

This package pre-trains **decoder-only transformers** purely autoregressively on the
synthetic graph-walk language produced by [`../data`](../data/README.md), and scores
what they generate against the language's **exact decoder** and **entropy floor**.

There is no supervised task. The model sees only `BOS + bits + EOS` streams; the
graph, the codebooks and the segmentation never appear in the input. Research
context and the full experiment plan: [`../docs/context.md`](../docs/context.md)
(§7 model, §8 evaluation).

---

## 1. The model family

One architecture, scaled purely by config (context.md §7 — keep it clean so results
are attributable to `(N params, D tokens, language difficulty)` and nothing else):

* pre-norm **RMSNorm**
* **RoPE** attention (`F.scaled_dot_product_attention`, `is_causal=True`)
* **SwiGLU** MLP, hidden width `4 * d_model` (gate style: three matrices)
* **no biases** anywhere
* **weight tying**: `lm_head.weight is embed.weight`
* no GQA or MoE; generation uses a per-layer KV cache

Vocabulary is `{0, 1, BOS, EOS, PAD}` = **5 tokens**, taken from `synthdata`'s
`BitTokenizer`; `build_model` asserts the two agree. Because the vocabulary is so
small, the embedding is negligible and parameter counts are essentially all
non-embedding — which is exactly what makes sub-million-parameter models meaningful
here.

`SIZE_PRESETS` (in `config.py`) is the named family for scaling sweeps.
Measured at `context_len = 512`, `vocab = 5`:

| size | `d_model` | layers | heads | params | non-embedding |
|---|---|---|---|---|---|
| `nano`  | 64  | 2  | 2  | 131,712 | 131,392 |
| `micro` | 128 | 4  | 4  | 1,050,368 | 1,049,728 |
| `tiny`  | 256 | 6  | 4  | 6,296,064 | 6,294,784 |
| `small` | 384 | 8  | 6  | 18,882,816 | 18,880,896 |
| `base`  | 512 | 10 | 8  | 41,956,352 | 41,953,792 |
| `large` | 768 | 12 | 12 | 113,269,248 | 113,265,408 |

RoPE means `context_len` costs no parameters; it only sizes the (non-persistent)
`cos`/`sin` buffers. A `size` preset can be overridden field by field in the config
(explicit `d_model` etc. win over the preset).

---

## 2. Integration with the data side

`synthdata` lives in the sibling `../data` folder and is **not** listed as a
dependency. `_paths.py` contains a one-function **path shim**: if
`synthdata` is not already importable, it prepends `<repo>/data` to `sys.path`. A
real install (`pip install -e ../data`) takes precedence and the shim does nothing.
This is the only place in the package that touches `sys.path`; every other module
just does `import synthdata...`.

Requirements: Python ≥ 3.10, `torch`, `pyyaml`. From `model/`:

```bash
pip install -e .          # or simply: pip install torch pyyaml
```

Relative paths inside a config (`data.dataset_dir`, `train.out_dir`) are resolved
against **this `model/` directory**, not the shell's cwd, so a config means the same
thing wherever it is invoked from.

---

## 3. Config schema

One run = one YAML file, four blocks (`configs/smoke.yaml`):

```yaml
model:
  size: micro            # SIZE_PRESETS key; or give d_model / n_layers / n_heads directly
  context_len: 512       # packing window and RoPE table length
  # vocab_size: 5        # must match BitTokenizer (asserted)
  # dropout: 0.0

data:
  dataset_dir: ../data/outputs/smoke   # a synthdata dataset directory
  mode: frozen                         # frozen | streaming
  # stream_seed: null                  # streaming only; defaults to train.seed

train:
  max_steps: 2000
  batch_size: 32
  lr: 3.0e-4
  # min_lr_frac: 0.1     # cosine floor as a fraction of lr
  warmup: 100            # linear warmup steps
  # weight_decay: 0.1
  # betas: [0.9, 0.95]
  # grad_clip: 1.0
  log_every: 50
  eval_every: 250
  # eval_batches: 20     # validation batches per eval pass
  seed: 42
  out_dir: outputs/smoke_micro
  # device: null         # null = auto-detect (cuda > mps > cpu)

eval:
  n_samples: 200
  temperature: 1.0
  cuts: [0, 50]          # 0 = generate from scratch; k = complete a k-bit test prefix
  # max_len: null        # generation cap in tokens; null = model context_len
  # gen_batch_size: 64
  # seed: 0
```

Unknown keys are hard errors — a typo never silently trains the wrong thing.

---

## 4. Commands

```bash
# train (writes metrics.jsonl, best.pt, last.pt, train_summary.json into out_dir)
python scripts/train.py configs/smoke.yaml
python scripts/train.py configs/smoke.yaml --size tiny --max-steps 5000 --out-dir outputs/smoke_tiny

# evaluate a checkpoint: loss vs floor, validity at each cut, diversity/memorisation
python scripts/evaluate.py outputs/smoke_micro/best.pt
python scripts/evaluate.py outputs/smoke_micro/best.pt --n-samples 500 --cuts 0 20 50 --temperature 1.0

# sample sentences and decode them with the exact decoder
python scripts/generate.py outputs/smoke_micro/best.pt -n 5 --score
python scripts/generate.py outputs/smoke_micro/best.pt -n 5 --cut 50 --temperature 0.8
```

A checkpoint stores `{model: state_dict, config: full run config, step,
valid_bits_per_token}`, so `evaluate.py` / `generate.py` need nothing but the file
(the dataset directory comes from the stored config, overridable with `--dataset`).

---

## 5. Data pipeline (`data.py`)

**Frozen mode** — `synthdata.storage.load_dataset` → `BitTokenizer.pack` (synthdata's
own packer) into windows of `context_len + 1` tokens, so a batch yields inputs
`w[:, :-1]` and targets `w[:, 1:]` covering exactly `context_len` positions. Sentences
are concatenated and *not* aligned to window boundaries, as in the CFG paper.
`BatchSampler` draws uniform random windows from its own seeded `torch.Generator`.

**Streaming mode** — `stream_batches` wraps `synthdata.dataset.stream` into an
infinite batch iterator, packed identically. For later infinite-data scaling runs;
the two modes differ only in whether the *training* data was stored. The dataset
directory is still required (and still loaded): it supplies the language, the
manifest's entropy floor, and the frozen valid/test splits used for evaluation.

**Loss masking.** Targets equal to `PAD` or `BOS` are excluded (`ignore_index`).
`PAD` never occurs in packed windows (`drop_last=True`) but does in sentence-aligned
batches. `BOS` is excluded because it is *fully predictable* (it always follows
`EOS`): counting it would push the reported bits/token **below** the manifest's
entropy floor, which is averaged over `bits + EOS` tokens only. Excluding it makes
every number in this package directly comparable to the floor. No masking happens
across sentence boundaries — predicting across a boundary is a real part of the task.

---

## 6. Training (`train.py`)

AdamW (`betas 0.9/0.95`, `weight_decay 0.1`), decay applied only to 2-D matrices —
no decay on RMSNorm weights or on the tied embedding. Linear warmup then cosine decay
to `min_lr_frac * lr` (10 %). Gradient clipping at 1.0.

Device auto-detection **cuda > mps > cpu**. AMP (bf16) is enabled on cuda only; mps
and cpu run fp32 — deliberately simple and robust rather than fast.

Loss is always reported in **bits/token** (nats / ln 2) next to the dataset's
`entropy_floor_bits_per_token` from the manifest and the gap between them. During
training, validation uses packed windows (fast); at the end, `train_summary.json`
also records a **sentence-aligned** valid loss (whole sentences, PAD-filled), which is
the number to quote against the floor and the one `evaluate.py` reports.

Outputs in `out_dir`:

| file | content |
|---|---|
| `metrics.jsonl` | one JSON record per logged step: `step, lr, train/valid bits per token, gap_to_floor, elapsed_s` |
| `best.pt` | best checkpoint by validation loss |
| `last.pt` | final checkpoint |
| `train_summary.json` | start/end losses, best valid, floor, gap, wall time |

---

## 7. Evaluation (`evals.py`, context.md §8 v1 scope)

* `eval_loss` — valid/test loss in bits/token, sentence-aligned, plus the gap to the
  entropy floor.
* `eval_validity(cut=k)` — `cut=0` generates from `BOS`; `cut=k` takes fresh **test**
  sentences (never trained on), feeds their first `k` bits and lets the model finish.
  Multinomial sampling at temperature τ, `PAD`/`BOS` masked out of the distribution.
  Every string is scored with `Language.is_valid` (the exact DP decoder). Reported:
  * `validity_pct` — the produced string encodes at least one walk;
  * `terminated_pct` — the model emitted `EOS` itself instead of hitting `max_len`;
  * `valid_and_terminated_pct` — the model's **own EOS placement** yields a valid
    sentence (the strict reading of "did it finish a real sentence?");
  * `walk_len_in_range_pct` — of the valid strings, those whose decoded walk length
    lies inside the language's `walk_len` range (`is_valid` imposes no length bound).
* `eval_diversity` — distinct-sentence fraction, collision count, and the fraction of
  generated sentences appearing **verbatim in train** (memorisation rate).
* `run_all` — everything above, saved as `eval_report.json` (with a handful of example
  generations) and printed by `scripts/evaluate.py`.

Not yet implemented (later phases, per context.md §8): robustness on noised datasets,
temperature sweeps / mode-switch, scaling-law fits, probing.

---

## 8. Determinism

`seed_everything(seed)` seeds Python's `random` and torch from `train.seed`;
`BatchSampler` and generation carry their own `torch.Generator`, so batch order and
sampling are reproducible independently of any global state. **MPS and CUDA kernels
may still be nondeterministic**, so two runs on the same seed can differ slightly in
the last digits — acceptable for research. The *data* side is bit-exactly
reproducible from `(config, seed)`; the optimisation is not.
