"""Replay CPU Expert access traces without loading a model."""

import argparse
from collections import Counter, OrderedDict
import heapq
import json
import math
from pathlib import Path
import re


def read_trace(path):
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header["version"] != 1 or header["dropped"]:
            raise ValueError(f"unsupported or incomplete trace (dropped={header['dropped']})")
        rows = []
        sizes = {}
        for index, line in enumerate(stream):
            seq, source, expert, size, slot, decode, hit = json.loads(line)
            key = (source, expert)
            if (seq != index or str(source) not in header["sources"] or expert < 0
                    or not 0 < size <= header["max_expert_bytes"] or not 0 <= slot < header["slots"]
                    or sizes.setdefault(key, size) != size or decode not in (0, 1) or hit not in (0, 1)
                    or (rows and decode < rows[-1][2])):
                raise ValueError(f"invalid access event {index}")
            rows.append((key, size, decode, hit))
    if len(rows) != header["events"]:
        raise ValueError("trace length differs from header")
    return header, rows


def distances(rows):
    """Count active last-access positions with a Fenwick tree."""
    tree = [0] * (len(rows) + 1)
    last = {}

    def add(position, delta):
        while position < len(tree):
            tree[position] += delta
            position += position & -position

    for index, (key, *_) in enumerate(rows, 1):
        previous = last.get(key)
        distance = None
        if previous is not None:
            prefix = 0
            position = previous
            while position:
                prefix += tree[position]
                position -= position & -position
            distance = len(last) - prefix
            add(previous, -1)
        last[key] = index
        add(index, 1)
        yield distance


def distribution(values):
    reused = sorted(value for value in values if value is not None)
    result = {
        "accesses": len(values), "cold": len(values) - len(reused), "reused": len(reused),
        "histogram": dict(sorted(Counter(reused).items())),
    }
    for name, percentile in (("min", 0), ("median", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1)):
        result[name] = reused[max(0, math.ceil(len(reused) * percentile) - 1)] if reused else None
    return result


def allocate(capacity, counts, weighted):
    weights = counts if weighted else dict.fromkeys(counts, 1)
    total = sum(weights.values())
    if not total:
        return {}
    quotas = {layer: divmod(capacity * weight, total) for layer, weight in weights.items()}
    limits = {layer: quotient for layer, (quotient, _) in quotas.items()}
    remainder = capacity - sum(limits.values())
    for layer in sorted(weights, key=lambda x: (-quotas[x][1], x))[:remainder]:
        limits[layer] += 1
    return limits


def replay(rows, capacity, layers, limits=None, keep=None, future=None):
    results = {name: dict(hits=0, misses=0, evictions=0, reads=0, bytes_read=0)
               for name in ("prefill", "decode", "total")}
    caches = {}
    heap = []
    for index, (key, size, decode, _) in enumerate(rows):
        group = layers[key[0]] if limits is not None else "all"
        limit = limits[group] if limits is not None else capacity
        cache = caches.setdefault(group, OrderedDict())
        hit = key in cache
        evicted = False
        if hit or (limit and (keep is None or key in keep)):
            if not hit and len(cache) == limit:
                if future is None:
                    cache.popitem(last=False)
                else:
                    while heap:
                        negative, victim = heapq.heappop(heap)
                        if cache.get(victim) == -negative:
                            del cache[victim]
                            break
                evicted = True
            if future is None:
                cache[key] = None
                cache.move_to_end(key)
            else:
                cache[key] = future[index]
                heapq.heappush(heap, (-future[index], key))
                if len(heap) > 2 * capacity:
                    heap = [(-position, current) for current, position in cache.items()]
                    heapq.heapify(heap)
        for phase in ("decode" if decode else "prefill", "total"):
            stats = results[phase]
            stats["hits" if hit else "misses"] += 1
            stats["evictions"] += evicted
            stats["reads"] += not hit
            stats["bytes_read"] += 0 if hit else size
    for stats in results.values():
        accesses = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / accesses if accesses else 0
    return results


def next_positions(rows):
    positions = [len(rows)] * len(rows)
    upcoming = {}
    for index in range(len(rows) - 1, -1, -1):
        key = rows[index][0]
        positions[index] = upcoming.get(key, len(rows))
        upcoming[key] = index
    return positions


def analyze(header, rows, slots):
    layers, kinds = {}, {}
    for source, name in header["sources"].items():
        match = re.match(r"blk[.]([0-9]+)[.]", name)
        layers[int(source)] = int(match[1]) if match else -1
        kinds[int(source)] = next((kind for kind in ("gate", "up", "down") if f"ffn_{kind}_" in name), "other")
    reuse = list(distances(rows))
    counts = Counter(row[0] for row in rows)
    by_layer, by_kind = {}, Counter()
    for (source, expert), count in counts.items():
        layer = by_layer.setdefault(layers[source], {"accesses": 0, "unique_keys": 0, "expert_ids": set()})
        layer["accesses"] += count
        layer["unique_keys"] += 1
        layer["expert_ids"].add(expert)
        by_kind[kinds[source]] += count
    for layer in by_layer.values():
        layer["unique_expert_ids"] = len(layer.pop("expert_ids"))
    layer_counts = {layer: stats["accesses"] for layer, stats in by_layer.items()}
    next_use = next_positions(rows)
    policies = {}
    # Retrospective frequency ranking is an offline comparator, not a deployable selector.
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    decode_rows = [row for row in rows if row[2]]
    for count in sorted(set(slots) | ({header["slots"]} if header["mode"] == "lru" else set())):
        global_lru = replay(rows, count, layers)
        if global_lru["total"]["hits"] != sum(d is not None and d < count for d in reuse):
            raise ValueError("LRU replay and stack-distance result disagree")
        policies[count] = {
            "slot_payload_bytes": count * header["max_expert_bytes"],
            "global_lru": global_lru,
            "lru_decode_cold_start": replay(decode_rows, count, layers)["decode"],
            "layer_equal": replay(rows, count, layers, allocate(count, layer_counts, False)),
            "layer_weighted": replay(rows, count, layers, allocate(count, layer_counts, True)),
            "selective": replay(rows, count, layers, keep=set(ranked[:count])),
            "belady": replay(rows, count, layers, future=next_use),
        }
    observed = {key: header[key] for key in ("hits", "misses", "evictions", "reads", "bytes_read")}
    if observed["hits"] != sum(row[3] for row in rows) or observed["misses"] != len(rows) - observed["hits"]:
        raise ValueError("recorded outcomes disagree with cache counters")
    actual_lru = policies[header["slots"]]["global_lru"]["total"] if header["mode"] == "lru" else None
    match = ({key: actual_lru[key] == value for key, value in observed.items()} if actual_lru else None)
    if match is not None and header["contexts"] == 1 and not all(match.values()):
        raise ValueError(f"serial LRU replay differs from observed counters: {match}")
    mins = [value for value in reuse if value is not None]
    phases = {}
    for phase, name in enumerate(("prefill", "decode")):
        phase_rows = [(row, distance) for row, distance in zip(rows, reuse) if row[2] == phase]
        phases[name] = {
            **distribution([distance for _, distance in phase_rows]),
            "unique_keys": len({row[0] for row, _ in phase_rows}),
            "requested_bytes": sum(row[1] for row, _ in phase_rows),
            "observed_reads": sum(not row[3] for row, _ in phase_rows),
        }
    first_outcome_difference = next(
        (i for i, (row, distance) in enumerate(zip(rows, reuse))
         if row[3] != (distance is not None and distance < header["slots"])), None
    ) if actual_lru else None
    if first_outcome_difference is not None and header["contexts"] == 1:
        raise ValueError(f"serial LRU hit outcome differs at event {first_outcome_difference}")
    return {"trace": header, "reuse": distribution(reuse), "unique_keys": len(counts),
            "first_lru_hit_slots": min(mins) + 1 if mins else None,
            "phases": phases, "layers": by_layer, "tensor_kinds": by_kind,
            "policies": policies, "observed_matches_lru": match,
            "first_lru_outcome_difference": first_outcome_difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--slots", nargs="+", type=int,
                        default=[1, 16, 64, 128, 256, 512, 1024, 1152, 1536, 2048, 4096, 8192])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.slots) < 1 or args.output.resolve() == args.trace.resolve():
        parser.error("slot counts must be positive and the output must differ from the trace")
    try:
        header, rows = read_trace(args.trace)
        result = analyze(header, rows, args.slots)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, KeyError, StopIteration) as error:
        parser.exit(1, f"Trace analysis failed: {error}\n")


if __name__ == "__main__":
    main()
