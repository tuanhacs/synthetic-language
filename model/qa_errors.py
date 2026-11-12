"""Conditional QA errors; localize the first non-edge without graph-constrained decoding."""
from collections import Counter


def _pct(count, total):
    return 100.0 * count / total if total else None


class QAErrorStats:
    def __init__(self, language):
        self.language = language
        self.total = self.undecodable = self.valid = self.invalid = 0
        self.valid_counts = Counter()
        self.edges = Counter()
        self.unlocalized = Counter()
        self.generated_edges = Counter()
        self.invalid_transitions = Counter()
        self.included_samples = 0
        owners = {}
        for vertex, book in enumerate(language.codebooks.books):
            for word in book:
                owners.setdefault(word, set()).add(vertex)
        self.words = tuple(tuple((word, next(iter(vertices)) if len(vertices) == 1 else None)
                                 for word, vertices in owners.items() if word[0] == bit)
                           for bit in "01")

    def _decode_vertices(self, bits):
        # Full codeword segmentation, ignoring graph adjacency. Counts capped
        # at two prevent arbitrary attribution when a non-UD input is ambiguous.
        ways = bytearray(len(bits) + 1)
        ways[0] = 1
        parents = [None] * (len(bits) + 1)
        for start in range(len(bits)):
            if not ways[start]:
                continue
            for word, owner in self.words[int(bits[start])]:
                end = start + len(word)
                if end <= len(bits) and bits.startswith(word, start):
                    if not ways[end]:
                        parents[end] = (start, owner)
                    ways[end] = min(2, ways[end] + ways[start])
        if ways[-1] != 1:
            return None, "ambiguous_codeword_segmentation"
        walk = []
        end = len(bits)
        while end:
            start, owner = parents[end]
            if owner is None:
                return None, "shared_codeword_ownership"
            walk.append(owner)
            end = start
        return tuple(reversed(walk)), None

    def observe(self, answer_bits, result):
        self.total += 1
        if not result["decodable"]:
            self.undecodable += 1
            return
        if result["valid_walk"]:
            self.valid += 1
            self._count_edges(result["walk"])
            both = result["correct_start"] and result["correct_end"]
            self.valid_counts["wrong_start"] += not result["correct_start"]
            self.valid_counts["wrong_end"] += not result["correct_end"]
            self.valid_counts["correct_endpoints"] += both
            self.valid_counts["middle_waypoints_failed"] += both and not result["waypoints_in_order"]
            self.valid_counts["semantic_success"] += result["semantic_success"]
            return
        self.invalid += 1
        walk, reason = self._decode_vertices(answer_bits)
        if reason:
            self.unlocalized[reason] += 1
            return
        self._count_edges(walk)
        for a, b in zip(walk, walk[1:]):
            if not self.language.graph.is_edge(a, b):
                self.edges[(a // self.language.graph.n + 1, b // self.language.graph.n + 1)] += 1
                return  # Exactly one contribution: the leftmost invalid edge.
        self.unlocalized["no_non_edge_found"] += 1

    def _count_edges(self, walk):
        self.included_samples += 1
        for a, b in zip(walk, walk[1:]):
            counter = self.generated_edges if self.language.graph.is_edge(a, b) else self.invalid_transitions
            counter[a, b] += 1

    def edge_usage_report(self):
        from edge_heatmaps import edge_usage
        return {
            **edge_usage(self.language.graph, self.generated_edges),
            "n_samples": self.total,
            "included_samples": self.included_samples,
            "excluded_undecodable": self.undecodable,
            "excluded_ambiguous": self.total - self.undecodable - self.included_samples,
            "exclusion_reasons": dict(self.unlocalized),
            "invalid_occurrences": sum(self.invalid_transitions.values()),
            "invalid_transitions": {f"{a},{b}": count for (a, b), count in sorted(self.invalid_transitions.items())},
        }

    def report(self):
        localized = sum(self.edges.values())
        return {
            "total_n": self.total,
            "undecodable": {"n": self.undecodable, "pct_of_total": _pct(self.undecodable, self.total)},
            "valid_walk": {
                "n": self.valid, "pct_of_total": _pct(self.valid, self.total),
                "conditional_denominator_n": self.valid,
                **{f"{key}_{suffix}": (self.valid_counts[key] if suffix == "n"
                                       else _pct(self.valid_counts[key], self.valid))
                   for key in ("wrong_start", "wrong_end", "correct_endpoints",
                               "middle_waypoints_failed", "semantic_success")
                   for suffix in ("n", "pct")},
            },
            "invalid_walk": {
                "n": self.invalid, "pct_of_total": _pct(self.invalid, self.total),
                "localized_n": localized, "unlocalized_n": sum(self.unlocalized.values()),
                "unlocalized_reasons": dict(self.unlocalized),
                "matrix_denominator_n": localized, "row_numbering": "1-based; source row -> destination row",
                "first_invalid_edge_by_rows": {
                    f"{a}->{b}": {"count": self.edges[(a, b)], "pct": _pct(self.edges[(a, b)], localized)}
                    for a in range(1, self.language.graph.n + 1)
                    for b in range(1, self.language.graph.n + 1)
                },
            },
        }


def format_qa_errors(report):
    def percent(value):
        return "n/a" if value is None else f"{value:.1f}%"

    valid, invalid = report["valid_walk"], report["invalid_walk"]
    lines = [f"         QA errors: undecodable n={report['undecodable']['n']} "
             f"({percent(report['undecodable']['pct_of_total'])} of all); "
             f"valid-walk n={valid['n']} ({percent(valid['pct_of_total'])} of all)",
             f"         among valid-walk (denominator={valid['n']}): "
             f"wrong-start {percent(valid['wrong_start_pct'])}; wrong-end {percent(valid['wrong_end_pct'])}; "
             f"correct-ends/middle-failed {percent(valid['middle_waypoints_failed_pct'])}; "
             f"semantic {percent(valid['semantic_success_pct'])}",
             f"         decodable/invalid-walk n={invalid['n']} ({percent(invalid['pct_of_total'])} of all); "
             f"first-edge localized={invalid['localized_n']} unlocalized={invalid['unlocalized_n']}; "
             f"row-pair denominator={invalid['matrix_denominator_n']}"]
    pairs = invalid["first_invalid_edge_by_rows"]
    rows = sorted({int(key.split("->")[0]) for key in pairs})
    for a in rows:
        cells = [f"{a}->{b} {pairs[f'{a}->{b}']['count']}/{percent(pairs[f'{a}->{b}']['pct'])}" for b in rows]
        lines.append("         first-invalid-edge " + "  ".join(cells))
    if invalid["unlocalized_reasons"]:
        lines.append(f"         unlocalized reasons: {invalid['unlocalized_reasons']}")
    return lines
