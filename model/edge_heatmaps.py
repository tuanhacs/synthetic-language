"""Directed graph-edge usage: semantic training paths and generated final answers."""
import json
from collections import Counter
from pathlib import Path


def edge_usage(graph, counts):
    counts = {f"{a},{b}": counts.get((a, b), 0) for a, b in graph.directed_edges()}
    total = sum(counts.values())
    return {"counts": counts, "total_occurrences": total,
            "frequencies": {key: value / total if total else 0. for key, value in counts.items()},
            "normalization": "each directed edge count / total valid edge occurrences; repetitions counted"}


def training_edge_usage(dataset_dir, graph):
    """Use cached semantic counts, otherwise stream JSONL without loading/packing train."""
    directory = Path(dataset_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {"available": False, "dataset_dir": str(directory), "reason": "training manifest unavailable"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if config and (config.get("task", {}).get("type") != "path-qa"
                   or config.get("language", {}).get("graph") != f"grid-{graph.n}x{graph.n}"):
        return {"available": False, "dataset_dir": str(directory),
                "reason": "training manifest task/graph does not match evaluated path-QA graph"}
    cached = manifest.get("stats", {}).get("semantic_splits", {}).get("train", {}).get("graph_edges_directed")
    counts = Counter()
    if cached is not None:
        for key, count in cached["frequencies"].items():
            counts[tuple(map(int, key.split(",")))] = count
        source = "manifest"
    else:
        path = directory / manifest.get("files", {}).get("train", "train.jsonl")
        if not path.exists():
            return {"available": False, "dataset_dir": str(directory), "reason": "training JSONL unavailable"}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                if "answer_walk" not in sample:
                    return {"available": False, "dataset_dir": str(directory),
                            "reason": "training records do not contain semantic answer_walk"}
                walk = sample["answer_walk"]
                counts.update(zip(walk, walk[1:]))
        source = "streamed train JSONL"
    return {**edge_usage(graph, counts), "available": True, "source": source,
            "dataset_dir": str(directory), "paths": "semantic final answer_walk (not duplicated CoT segments)"}


def save_edge_heatmaps(graph, training, generated, out_dir):
    """Two local PNGs sharing one frequency normalization; no difference map."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    from matplotlib import colormaps
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import PercentFormatter

    directory = Path(out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    vmax = max((value for usage in (training, generated)
                for value in usage.get("frequencies", {}).values()), default=0.)
    norm = Normalize(0., vmax or 1.)
    cmap = colormaps["YlOrRd"]
    paths = {}
    for name, usage in (("training", training), ("generated", generated)):
        fig = Figure(figsize=(8, 7), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for a, b in graph.directed_edges():
            x, y = a % graph.n, graph.n - 1 - a // graph.n
            u, v = b % graph.n, graph.n - 1 - b // graph.n
            # Offset right of travel: opposite directed edges stay separate.
            dx, dy = u - x, v - y
            offset_x, offset_y = .07 * dy, -.07 * dx
            frequency = usage.get("frequencies", {}).get(f"{a},{b}", 0.)
            color = cmap(norm(frequency)) if frequency else "#cccccc"
            ax.add_patch(FancyArrowPatch((x + offset_x, y + offset_y),
                (u + offset_x, v + offset_y), arrowstyle="-|>", mutation_scale=12,
                shrinkA=15, shrinkB=15, linewidth=2.5, color=color, zorder=1))
        for vertex in range(graph.n ** 2):
            x, y = vertex % graph.n, graph.n - 1 - vertex // graph.n
            ax.scatter(x, y, s=260, facecolor="white", edgecolor="#444444", zorder=2)
            ax.text(x, y, str(vertex), ha="center", va="center", fontsize=9, zorder=3)
        ax.set(xlim=(-.5, graph.n - .5), ylim=(-.5, graph.n - .5), aspect="equal")
        ax.set_axis_off()
        detail = (f"{usage.get('total_occurrences', 0):,} valid edge occurrences"
                  if usage.get("available", True) else "Training source unavailable")
        ax.set_title(f"{name.title()} edge usage\n{detail}")
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     format=PercentFormatter(1.), label="Share of valid directed-edge occurrences")
        fig.text(.03, .01, "Gray = unused; arrows = direction; vertex labels = 0-based", fontsize=9)
        target = directory / f"{name}_edges.png"
        fig.savefig(target, dpi=160)
        paths[name] = str(target)
    return {"plots": paths, "vmin": 0., "vmax": norm.vmax, "colormap": "YlOrRd",
            "zero_color": "#cccccc", "scale": "shared normalized directed-edge frequency"}
