"""Generate a path-QA dataset on Modal and evaluate an existing checkpoint.

Examples:
    # Upload ./best.pt, generate the requested dataset, then evaluate it.
    python -m modal run --detach app_path_qa_evaluate.py \
        --checkpoint best.pt \
        --code-type ud --assignment overlap --qa-mode cross-region \
        --cot-mode segments --query-min 2 --query-max 5 \
        --pool-query-pairs 3_000_000

    # Reuse a checkpoint that is already on the persistent Modal Volume.
    python -m modal run --detach app_path_qa_evaluate.py \
        --volume-checkpoint model/outputs/my_run/best.pt \
        --code-type ud --assignment overlap --qa-mode iid \
        --cot-mode segments --query-min 2 --query-max 6 \
        --pool-query-pairs 100_000

The checkpoint supplies the model architecture and context length. Dataset
arguments only control the newly generated evaluation distribution.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal
import yaml


APP_NAME = "synthetic-language-path-qa-evaluate"
REPO_URL = "https://github.com/tuanhacs/synthetic-language.git"
REPO_DIR = PurePosixPath("/root/synthetic-language")
DATA_DIR = REPO_DIR / "data"
MODEL_DIR = REPO_DIR / "model"

VOLUME_MOUNT = "/mnt/synthetic-language"
volume = modal.Volume.from_name("synthetic-language", create_if_missing=True)
GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tqdm", "pyyaml")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        "python -m pip install --upgrade pip",
        f"python -m pip install -e {MODEL_DIR}",
        f"python -m pip install -e {DATA_DIR}",
        force_build=True,
    )
)

app = modal.App(APP_NAME, image=image)


def run(cmd: list[str], cwd: PurePosixPath) -> None:
    print(f"\n>>> cd {cwd}")
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def safe_name(raw: str, max_length: int = 200) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw.strip()).strip(".-")
    if not name:
        raise ValueError("name must contain at least one letter or digit")
    if len(name) <= max_length:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"{name[: max_length - 11]}-{digest}"


def file_digest(path: Path) -> str:
    """Short content hash without loading a large checkpoint fully into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def parse_heldout_pairs(raw: str) -> list[list[int]]:
    if not raw.strip():
        return []
    pairs: list[list[int]] = []
    for item in raw.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*[-:]\s*(\d+)\s*", item)
        if match is None:
            raise ValueError(
                f"invalid held-out pair {item!r}; expected e.g. '3-16,7-21'"
            )
        pairs.append([int(match.group(1)), int(match.group(2))])
    return pairs


def write_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def volume_checkpoint_path(raw: str) -> str:
    raw = raw.replace("\\", "/")
    if raw.startswith(f"{VOLUME_MOUNT}/"):
        return raw
    return f"{VOLUME_MOUNT}/{raw.lstrip('/')}"


@app.function(
    gpu=GPU,
    cpu=4.0,
    memory=32_768,
    timeout=24 * 60 * 60,
    ephemeral_disk=1000 * 1024,
    volumes={VOLUME_MOUNT: volume},
)
def generate_and_evaluate(
    checkpoint_path: str,
    checkpoint_label: str,
    code_type: str,
    assignment: str,
    pool_tokens: int,
    pool_query_pairs: int,
    qa_mode: str,
    cot_mode: str,
    heldout_pairs: str,
    pair_direction: str,
    region_overlap_rows: int,
    query_min: int,
    query_max: int,
    segment_min: int,
    segment_max: int,
    seed: int,
    base_size: int,
    depth_min: int,
    depth_max: int,
    power_x: int,
    ud_pool: str,
    k_min: int,
    k_max: int,
    overlap_support_min: int,
    overlap_support_max: int,
    shared_codewords: int,
    eval_samples: int,
    gen_batch_size: int,
    temperature: float,
    cuts: str,
    eval_name: str,
) -> None:
    # A warm Modal container may have mounted an older Volume snapshot before
    # the local entrypoint uploaded this checkpoint.
    volume.reload()
    code_types = {"prefix": "prefix-free", "suffix": "suffix-free", "ud": "ud"}
    assignments = {"disjoint": "disjoint-random", "overlap": "arbitrary-overlap"}
    qa_modes = {
        "iid": "iid",
        "heldout": "held-out-pairs",
        "cross-region": "cross-region",
    }
    if code_type not in code_types:
        raise ValueError(f"code_type must be one of {sorted(code_types)}")
    if assignment not in assignments:
        raise ValueError(f"assignment must be one of {sorted(assignments)}")
    if qa_mode not in qa_modes:
        raise ValueError(f"qa_mode must be one of {sorted(qa_modes)}")
    if cot_mode not in ("none", "segments"):
        raise ValueError("cot_mode must be 'none' or 'segments'")
    if pair_direction not in ("ordered", "unordered"):
        raise ValueError("pair_direction must be 'ordered' or 'unordered'")
    if pool_query_pairs < 0:
        raise ValueError("pool_query_pairs must be non-negative")
    use_pair_budget = pool_query_pairs > 0
    if not use_pair_budget and pool_tokens < 1:
        raise ValueError("pool_tokens must be positive when no pair budget is selected")
    if not (2 <= query_min <= query_max <= 36):
        raise ValueError("query range must satisfy 2 <= min <= max <= 36")
    if not (2 <= segment_min <= segment_max <= 36):
        raise ValueError("segment range must satisfy 2 <= min <= max <= 36")
    if not (1 <= k_min <= k_max):
        raise ValueError("k range must satisfy 1 <= min <= max")
    if assignment == "overlap" and k_max < 2:
        raise ValueError("overlap assignment requires k_max >= 2")
    if not (2 <= overlap_support_min <= overlap_support_max <= 36):
        raise ValueError("overlap support range must lie in [2, 36]")
    if shared_codewords < 1:
        raise ValueError("shared_codewords must be positive")
    if eval_samples < 1 or gen_batch_size < 1 or temperature <= 0:
        raise ValueError("evaluation counts and temperature must be positive")
    if qa_mode == "cross-region" and (
        not (1 <= region_overlap_rows < 6) or (6 - region_overlap_rows) % 2
    ):
        raise ValueError(
            "for grid-6x6, region_overlap_rows must leave an even number of outer rows; "
            "valid values are 2 or 4"
        )

    pairs = parse_heldout_pairs(heldout_pairs)
    if qa_mode == "heldout" and not pairs:
        raise ValueError("qa_mode='heldout' requires --heldout-pairs")
    if qa_mode != "heldout":
        pairs = []
    try:
        cut_values = [int(value) for value in re.split(r"[,\s]+", cuts.strip()) if value]
    except ValueError as exc:
        raise ValueError("cuts must be comma/space-separated integers, e.g. '0,20,50'") from exc
    if not cut_values or any(value < 0 for value in cut_values):
        raise ValueError("cuts must contain at least one non-negative integer")

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found on Modal: {checkpoint}")

    import torch

    checkpoint_doc = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        model_context = int(checkpoint_doc["config"]["model"]["context_len"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint does not contain config.model.context_len") from exc

    budget_label = (
        f"{pool_query_pairs}pairs" if use_pair_budget else f"{pool_tokens}bits"
    )
    data_label = safe_name(
        f"{code_type}-{assignment}-{qa_mode}-{budget_label}-"
        f"q{query_min}-{query_max}-seed{seed}-{'cot' if cot_mode == 'segments' else 'nocot'}"
    )
    run_label = safe_name(
        eval_name or f"{checkpoint_label}__{data_label}"
    )
    dataset_dir = Path(DATA_DIR) / "outputs" / f"eval_runtime_{run_label}"
    output_dir = Path(VOLUME_MOUNT) / "model" / "evaluations" / run_label
    runtime_dir = output_dir / "runtime"
    data_config_path = runtime_dir / "data.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)

    language: dict = {
        "graph": "grid-6x6",
        "code": {"type": code_types[code_type]},
        "k": [k_min, k_max],
        "assignment": assignments[assignment],
    }
    if code_type == "ud":
        pool_path = Path(ud_pool)
        if not pool_path.is_absolute():
            pool_path = Path(DATA_DIR) / "codeword_pools" / pool_path
        language["codeword_pool_file"] = str(pool_path)
    else:
        language["code"].update(
            {
                "base_size": base_size,
                "depth_range": [depth_min, depth_max],
                "power_x": power_x,
            }
        )
    if assignment == "overlap":
        language["overlap"] = {
            "support_size": [overlap_support_min, overlap_support_max],
            "shared_codewords": shared_codewords,
            "max_restarts": 100,
            "candidate_trials": 10_000,
        }

    budget_document = (
        {"pool_query_pairs": pool_query_pairs}
        if use_pair_budget
        else {"pool_tokens": pool_tokens}
    )
    data_document = {
        "language": language,
        "data": {
            **budget_document,
            "split": [80, 10, 10],
            "seed": seed,
            "noise": None,
            "context_len": "auto",
        },
        "task": {
            "type": "path-qa",
            "cot_mode": cot_mode,
            "query_len": [query_min, query_max],
            "segment_len": [segment_min, segment_max],
            "path_trials": 1_000,
            "split_mode": qa_modes[qa_mode],
            "pair_direction": pair_direction,
            "held_out_pairs": pairs,
            "region_overlap_rows": region_overlap_rows,
        },
    }
    write_yaml(data_config_path, data_document)

    print("Starting path-QA data generation + checkpoint evaluation")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint context_len: {model_context}")
    print(f"Dataset: {data_label}")
    print(f"Persistent evaluation output: {output_dir}")
    run(["git", "log", "-1", "--oneline"], cwd=REPO_DIR)
    run(
        [
            "python",
            "scripts/gen_dataset.py",
            str(data_config_path),
            "--out",
            str(dataset_dir),
        ],
        cwd=DATA_DIR,
    )

    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_max = int(manifest["stats"]["observed_max_sequence_tokens"])
    recommended_context = int(manifest["stats"]["recommended_context_len"])
    print(
        f"Dataset observed max={observed_max}; recommended context="
        f"{recommended_context}; checkpoint context={model_context}",
        flush=True,
    )
    if observed_max > model_context:
        print(
            f"[warn] {observed_max - model_context} tokens exceed the checkpoint context "
            "for the longest reference sample; generation remains capped at the "
            "checkpoint context and long-reference metrics may be harder.",
            flush=True,
        )

    shutil.copy2(manifest_path, output_dir / "dataset_manifest.json")
    run(
        [
            "python",
            "scripts/evaluate.py",
            str(checkpoint),
            "--dataset",
            str(dataset_dir),
            "--n-samples",
            str(eval_samples),
            "--gen-batch-size",
            str(gen_batch_size),
            "--temperature",
            str(temperature),
            "--max-len",
            str(model_context),
            "--cuts",
            *[str(value) for value in cut_values],
            "--seed",
            str(seed),
            "--out-dir",
            str(output_dir),
        ],
        cwd=MODEL_DIR,
    )
    volume.commit()
    print(f"Completed successfully: {output_dir}")


@app.local_entrypoint()
def main(
    checkpoint: str = "best.pt",
    volume_checkpoint: str = "",
    code_type: str = "ud",
    assignment: str = "disjoint",
    pool_tokens: int = 300_000_000,
    pool_query_pairs: int = 0,
    qa_mode: str = "iid",
    cot_mode: str = "none",
    heldout_pairs: str = "3-16,7-21,10-35",
    pair_direction: str = "unordered",
    region_overlap_rows: int = 2,
    query_min: int = 2,
    query_max: int = 4,
    segment_min: int = 2,
    segment_max: int = 14,
    seed: int = 0,
    base_size: int = 3,
    depth_min: int = 1,
    depth_max: int = 4,
    power_x: int = 5,
    ud_pool: str = "hard_ud_codebook_200.json",
    k_min: int = 1,
    k_max: int = 4,
    overlap_support_min: int = 2,
    overlap_support_max: int = 4,
    shared_codewords: int = 12,
    eval_samples: int = 1_000,
    gen_batch_size: int = 512,
    temperature: float = 1.0,
    cuts: str = "0",
    eval_name: str = "",
) -> None:
    if volume_checkpoint:
        checkpoint_path = volume_checkpoint_path(volume_checkpoint)
        checkpoint_label = safe_name(
            f"{Path(volume_checkpoint).parent.name}-{Path(volume_checkpoint).stem}"
        )
        print(f"Using checkpoint already on Modal Volume: {checkpoint_path}")
    else:
        local_checkpoint = Path(checkpoint).expanduser().resolve()
        if not local_checkpoint.is_file():
            raise FileNotFoundError(f"local checkpoint not found: {local_checkpoint}")
        digest = file_digest(local_checkpoint)
        checkpoint_label = safe_name(f"{local_checkpoint.stem}-{digest}")
        remote_relative = f"/uploaded_checkpoints/{checkpoint_label}.pt"
        print(
            f"Uploading {local_checkpoint} ({local_checkpoint.stat().st_size:,} bytes) "
            f"to Modal Volume as {remote_relative}",
            flush=True,
        )
        with volume.batch_upload(force=True) as batch:
            batch.put_file(local_checkpoint, remote_relative)
        checkpoint_path = volume_checkpoint_path(remote_relative)

    call = generate_and_evaluate.spawn(
        checkpoint_path,
        checkpoint_label,
        code_type,
        assignment,
        pool_tokens,
        pool_query_pairs,
        qa_mode,
        cot_mode,
        heldout_pairs,
        pair_direction,
        region_overlap_rows,
        query_min,
        query_max,
        segment_min,
        segment_max,
        seed,
        base_size,
        depth_min,
        depth_max,
        power_x,
        ud_pool,
        k_min,
        k_max,
        overlap_support_min,
        overlap_support_max,
        shared_codewords,
        eval_samples,
        gen_batch_size,
        temperature,
        cuts,
        eval_name,
    )
    print("Submitted path-QA evaluation job")
    print("Call ID:", call.object_id)
