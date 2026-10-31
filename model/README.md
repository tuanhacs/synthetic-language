# `model/` — transformer family, training, evaluation

This package trains **decoder-only transformers** autoregressively on either the
synthetic graph-walk language or its conditional path-QA task produced by
[`../data`](../data/README.md), and scores generations with the exact decoder.

Language modeling uses `BOS + bits + EOS`. Standard path-QA uses
`BOS + query_bits + _ + answer_bits + EOS`; segment-CoT path-QA uses
`BOS + query_bits + _ + segment_1 + _ + ... + segment_(L-1) + _ + answer_bits + EOS`.
Loss is masked only through the first `_`, so every CoT field, later separator,
final answer, and EOS is supervised. The vocabulary remains
`{0, 1, _, BOS, EOS, PAD}`. The graph, codebooks and segmentation never appear
in the input. Research
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
* no GQA or MoE; generation uses a preallocated per-layer KV cache

Vocabulary is 5 tokens for language modeling and adds `_` for 6 tokens in
path-QA, taken from `synthdata`'s `BitTokenizer`; `build_model` asserts the two agree. Because the vocabulary is so
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

### Validation early stopping

Training remains step-based with random window sampling with replacement.
`max_steps` is the hard upper bound; validation early stopping can stop it sooner.

```bash
python model/scripts/train.py YOUR_MODEL_CONFIG.yaml \
  --device cuda --max-steps 100000 --batch-size 64 \
  --eval-every 250 --eval-batches 20 \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 0.0001
```

Validation runs every `eval_every` steps, after step 1 and at the final step, as
before. Validation remains a
deterministic subset of up to `eval_batches` batches, not a full final validation
pass. The test split is not used for early stopping. Set the validation batch
budget consistently across comparisons.

`early_stopping_patience` counts consecutive validation checks without an
improvement larger than `early_stopping_min_delta` (absolute bits/token).
It defaults to 0 (disabled). With validation every 250 steps, patience 20 means
20 consecutive checks (normally 5000 steps) without sufficient improvement.
The early-stopping reference tracks significant improvements; `best.pt` still
tracks the lowest raw validation loss, including smaller improvements.

`latest.pt` saves optimizer/RNG/sampler state and early-stopping state.
Resuming restores all of them; an already early-stopped
run does not start training again. Logs and `train_summary.json` report actual
steps and `stop_reason` (`max_steps` or `early_stopping`).
Changing the stopping policy requires a new output directory
or an explicit fresh run. Existing default step-mode checkpoints remain compatible.

By default cosine decay uses `max_steps` as its horizon. Set `lr_decay_steps`
to decouple the LR schedule from the training limit. For example:

```bash
python scripts/train.py configs/smoke.yaml --device cuda \
  --max-steps 50000 --lr-decay-steps 20000 \
  --eval-every 250 --early-stopping-patience 20 \
  --early-stopping-min-delta 0.0001 --out-dir outputs/decay20k
```

LR reaches `lr * min_lr_frac` at step 20000 and stays there afterward.
Early stopping is active throughout training, including before step 20000.
`lr_decay_steps` is an absolute step (including warmup), must exceed warmup,
and is saved in config/checkpoints/W&B. Changing it requires a fresh run/output
directory; omitting it preserves the old schedule and checkpoint compatibility.

### Optional Weights & Biases

Install the extra once (`python -m pip install -e './model[tracking]'` from the
repository root, or `python -m pip install wandb`), then run `wandb login` before
starting an online job. Alternatively provide `WANDB_API_KEY` through the server
environment. Keys are not included in configs or logs by this integration.

```bash
python model/scripts/train.py YOUR_MODEL_CONFIG.yaml \
  --max-steps 100000 --early-stopping-patience 20 \
  --wandb --wandb-project synthetic-language \
  --wandb-group iid-disjoint --wandb-name ud-600k
```

YAML can set the optional top-level `wandb` block with `mode` (default `disabled`),
`project`, `entity`, `name`, `group`, and a `tags` list. Tracking flags also accept
`--wandb-mode offline` for a server without network access. Disabled mode neither
imports the SDK nor creates a W&B run. Existing model configs/checkpoints without
this block continue to load. The server requirements file includes the SDK, while
the base model package keeps it optional.

Training logs learning rate, train/validation bits per token, elapsed time and
early-stopping patience counters at the existing logging/validation intervals.
Run configuration records model dimensions, language/task, actual train query-pair
count, per-split sizes and dataset fingerprint. No weight watching or model-artifact
uploads are enabled. Training summaries report actual steps, best validation loss
and stopping reason. A custom `trainer/step` axis uses optimizer steps rather than
W&B's internal append counter.

`wandb_run.json` in each checkpoint directory persists the run ID. Online restarts
and the evaluation script reuse this ID, so train and test metrics belong to the
same run. After a crash, steps after the last durable checkpoint may be logged
again; local checkpoint/JSON records remain authoritative. Offline mode writes
local W&B segments (cloud history resume is not supported offline); use `wandb sync`
to upload them later. Explicit `train.resume: never` creates a new tracking ID.

`model/scripts/evaluate.py` inherits tracking from the checkpoint, logging
`test/decodable_pct`, `test/success_pct` and other scalar test metrics from
the generated test report. Use `--wandb-mode disabled` on evaluation to opt out.
Checkpoint weights themselves remain on the local filesystem. When adding tracking
to a previously configured server sweep, select a new output root because runtime
YAMLs are immutable once written.

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
  checkpoint_every: 250 # atomic full-state latest.pt interval
  resume: auto           # auto | never | require
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
# train (writes metrics.jsonl, best.pt, latest.pt, train_summary.json)
python scripts/train.py configs/smoke.yaml
python scripts/train.py configs/smoke.yaml --size tiny --max-steps 5000 --out-dir outputs/smoke_tiny
python scripts/train.py configs/smoke.yaml --d-model 192 --n-layers 6 --n-heads 6

# evaluate a checkpoint: generation success on test examples only
python scripts/evaluate.py outputs/smoke_micro/best.pt
python scripts/evaluate.py outputs/smoke_micro/best.pt --n-samples 500 --cuts 0 20 50 --temperature 1.0

# repeated-rollout transfer/confidence analysis for cross-region path-QA
python scripts/analyze_cross_region.py outputs/path_qa_cross_region/best.pt \
  --dataset ../data/outputs/path_qa_cross_region \
  --n-queries 500 --rollouts-per-query 32

# sample sentences and decode them with the exact decoder
python scripts/generate.py outputs/smoke_micro/best.pt -n 5 --score
python scripts/generate.py outputs/smoke_micro/best.pt -n 5 --cut 50 --temperature 0.8
```

`best.pt` is the validation-selected inference checkpoint containing the model,
config, step and validation score. `latest.pt` is written atomically every
`train.checkpoint_every` steps and additionally stores optimizer state, Python/
PyTorch/CUDA RNG state and the frozen-data sampler state. With `resume: auto`,
restarting the same run continues from `latest.pt`; incompatible model, training,
or dataset configuration is rejected instead of silently starting from mixed state.

For a path-QA checkpoint, `evaluate.py` automatically supplies each test prompt
as `query_bits + "_"`. It reports decodability, graph-walk validity, endpoints,
waypoint order, per-segment simplicity, semantic success and exact-reference
rates. Semantic success accepts any route satisfying the question, not only the
stored reference route. For `cot_mode: segments`, it also reports output-format
validity, per-segment route validity/endpoints/simplicity, and whether the final
answer exactly represents the merge of the generated CoT routes.

For a dataset whose `task.split_mode` is `cross-region`,
`analyze_cross_region.py` estimates a nested transfer funnel (`correct start ->`
`enter overlap -> reach target region -> hit/end at destination`), stratifies it
by direction, number of crossing segments, query length and Manhattan distance,
and reports per-query endpoint entropy. It also measures teacher-forced token
probability/entropy/margin at the two region gates, and performs a destination
intervention by replacing a cross-region destination with a same-region one.
The outputs are `cross_region_analysis.json` and
`cross_region_queries.jsonl`.

---

## 5. Data pipeline (`data.py`)

**Frozen mode** — `synthdata.storage.load_dataset` → `BitTokenizer.pack` (synthdata's
own packer) into windows of `context_len + 1` tokens, so a batch yields inputs
`w[:, :-1]` and targets `w[:, 1:]` covering exactly `context_len` positions. Sentences
are concatenated and *not* aligned to window boundaries, as in the CFG paper.
`BatchSampler` draws uniform random windows from its own seeded `torch.Generator`.

For path-QA, model inputs remain ordinary six-token IDs. Targets through the
first `_` are ignored; all later tokens and EOS contribute to the loss. The target mask is
preserved even when packed windows cross example boundaries. Streaming mode is
currently rejected for path-QA.

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

Training loss and periodic validation loss are reported in **bits/token**
(nats / ln 2). Validation uses a configured number of packed-window batches and
selects `best.pt`; training does not run an extra full-validation sweep at the end.
`evaluate.py` does not recompute valid/test loss or an entropy gap.

Outputs in `out_dir`:

| file | content |
|---|---|
| `metrics.jsonl` | one JSON record per logged step: `step, lr, train/valid bits per token, elapsed_s` |
| `best.pt` | best checkpoint by validation loss |
| `latest.pt` | atomic resumable state: model, optimizer, step, RNG and data sampler |
| `train_summary.json` | start/end losses, best/last periodic valid loss, floor, wall time |

---

## 7. Evaluation (`evals.py`, context.md §8 v1 scope)

* `eval_validity(cut=k)` — `cut=0` generates from `BOS`; `cut=k` takes fresh **test**
  sentences (never trained on), feeds their first `k` bits and lets the model finish.
  Multinomial sampling at temperature τ, `PAD`/`BOS` masked out of the distribution.
  Every string is scored with `Language.is_valid` (the exact DP decoder). Reported:
  * `decodable_pct` — the bit string can be segmented completely into words from the
    global code, before checking vertex ownership or graph adjacency;
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

`seed_everything(seed)` seeds Python's `random` and every torch RNG;
`BatchSampler` and generation carry their own `torch.Generator`, so batch order and
sampling are reproducible independently of any global state. PyTorch deterministic
algorithms are mandatory, cuDNN benchmarking and TF32 are disabled, and cuBLAS uses
the deterministic `CUBLAS_WORKSPACE_CONFIG=:4096:8` workspace. An unsupported
nondeterministic operation raises an error instead of silently weakening the
guarantee. Reproduction still requires the same checkpoint/data, software stack,
GPU architecture, batch size and other run parameters.
