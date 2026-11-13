"""Directed graph-edge usage: semantic training paths and generated final answers."""
import json
import math
from collections import Counter
from pathlib import Path


LEVEL_COLORS = ('#2563eb', '#16a34a', '#ca8a04', '#ea580c', '#dc2626')
LEVEL_NAMES = ('Very low', 'Low', 'Medium', 'High', 'Very high')


def level_colormap():
    """Distinct hues per frequency band; opacity increases within each band."""
    from matplotlib.colors import ListedColormap, to_rgb
    return ListedColormap([
        (*to_rgb(color), .4 + .6 * step / 127)
        for color in LEVEL_COLORS for step in range(128)
    ], name='frequency_levels_alpha')


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


def save_edge_heatmaps(graph, training, generated, out_dir, *, vmax=None):
    """Two local PNGs sharing one frequency normalization; no difference map."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import PercentFormatter

    directory = Path(out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    observed_max = max((value for usage in (training, generated)
                for value in usage.get("frequencies", {}).values()), default=0.)
    if vmax is None:
        vmax = observed_max or 1.
    if not math.isfinite(vmax) or vmax <= 0 or vmax < observed_max:
        raise ValueError('vmax must be finite, positive, and at least the largest frequency')
    norm = Normalize(0., vmax)
    cmap = level_colormap()
    boundaries = [vmax * index / 5 for index in range(6)]
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
                     ticks=boundaries, format=PercentFormatter(1.),
                     label="Share of valid directed-edge occurrences (five levels)")
        fig.text(.03, .035, "Blue → green → yellow → orange → red: increasing frequency", fontsize=8)
        fig.text(.03, .012, "Within each band: faint → opaque. Gray = unused; vertices = 0-based", fontsize=8)
        target = directory / f"{name}_edges.png"
        fig.savefig(target, dpi=160)
        paths[name] = str(target)
    return {"plots": paths, "vmin": 0., "vmax": norm.vmax, "colormap": cmap.name,
            "levels": [{"name": name, "color": color, "lower_frequency": boundaries[i],
                        "upper_frequency": boundaries[i+1]}
                       for i, (name, color) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS))],
            "opacity_range_within_each_level": [.4, 1.],
            "zero_color": "#cccccc", "scale": "shared normalized directed-edge frequency"}
