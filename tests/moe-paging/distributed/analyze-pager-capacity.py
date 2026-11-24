"""Compare ordered Stage TCP capacity runs or diagnose a bounded Expert trace."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import runpy
import statistics

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_folders(folders):
    compare = runpy.run_path(str(HERE / "analyze-stage-performance.py"))["compare"]
    ordered = []
    grouped = defaultdict(list)
    reference = None
    outputs = None
    conditions = ("revision", "version", "model_identity", "identity_source", "context", "effective_context",
                  "batch", "threads", "io", "split", "layers", "hidden", "vocab", "cache_state_label",
                  "chat", "max_new_tokens", "prompt_token_sha256")
    for index, path in enumerate(folders, 1):
        record = {"order": index, "folder": str(path)}
        ordered.append(record)
        try:
            values, phases = compare([path / f"{role}.metrics.json" for role in ("baseline", "coordinator", "worker")])
            base = values[0]
            measured = {key: base[key] for key in conditions}
            if any(value.get("expert_trace_limit") for value in values):
                raise ValueError("Expert tracing must be disabled during capacity timing")
            if reference is not None and measured != reference:
                raise ValueError("conditions differ across capacities: " + ", ".join(
                    key for key in conditions if measured[key] != reference[key]))
            hashes = {suffix: digest(path / f"baseline.{suffix}") for suffix in ("generated_tokens", "generated.txt")}
            if outputs is not None and hashes != outputs:
                raise ValueError("Baseline output differs across capacities")
            if reference is None:
                reference, outputs = measured, hashes
            capacity = base["pager_budget_mib"]
            record.update(status="ok", pager_policy=base["pager_policy"], budget_mib=capacity, byte_match=True, output_sha256=hashes)
            for data in values:
                pager, memory, generation = data["pager"], data.get("process_memory") or {}, data.get("generation", {})
                grouped[base["pager_policy"], capacity, data["role"]].append({
                    **{key: generation.get(key) for key in
                       ("prefill_ms", "decode_step_wall_ms", "generation_elapsed_ms", "generation_tokens_per_sec")},
                    **{key: pager[key] for key in ("cache_hits", "cache_misses", "cache_evictions", "hit_rate",
                                                   "disk_reads", "bytes_read", "io_wait_ns", "peak_allocated_bytes", "slots")},
                    **{key: pager.get(key) for key in (
                        "resident_slots", "transient_slots", "sketch_bytes", "policy_management_bytes",
                        "admitted_misses", "bypassed_misses", "admission_comparisons", "admission_rejections",
                        "frequency_updates", "frequency_aging_count", "transient_waits", "transient_wait_ns")},
                    **{key: memory.get(key) for key in ("peak_working_set_bytes", "private_commit_bytes", "peak_rss_bytes")},
                    "stage_execution_wall_ms": data["stage"]["stage_execution_wall_ms"],
                    **{phase + "_reads": phases[phase]["worker_reads"] if data["role"] == "worker"
                       else data["phases"][phase]["pager"]["disk_reads"] for phase in ("prefill", "decode")},
                })
        except (OSError, ValueError, KeyError, TypeError) as error:
            record.update(status="excluded", reason=str(error))
    result = {}
    for (policy, capacity, role), samples in sorted(grouped.items()):
        fields = {}
        for key in samples[0]:
            items = [sample[key] for sample in samples if sample[key] is not None]
            fields[key] = {"median": statistics.median(items), "min": min(items), "max": max(items)} if items else None
        result.setdefault(policy, {}).setdefault(capacity, {})[role] = {"runs": len(samples), "metrics": fields}
    def delta(metrics, reference):
        return {key: value["median"] - reference[key]["median"] if value and reference.get(key) else None
                for key, value in metrics.items()}

    lru = result.get("lru", {})
    for budgets in result.values():
        smallest = min(budgets)
        for capacity, roles in budgets.items():
            for role, group in roles.items():
                metrics = group["metrics"]
                group["delta_vs_smallest_budget"] = delta(metrics, budgets[smallest][role]["metrics"])
                for baseline, suffix in ((capacity, "same_budget_lru"), (768, "768_mib_lru")):
                    if baseline in lru and role in lru[baseline]:
                        group["delta_vs_" + suffix] = delta(metrics, lru[baseline][role]["metrics"])
    return {"conditions": reference, "model_identity_note": "Label is not independent verification." if reference and
            reference["identity_source"] != "local-sha256" else None, "ordered_runs": ordered, "by_budget_mib": result.get("lru", {}), "by_policy": result}


def diagnose(path, coordinator_csv, additional_slots=()):
    tools = runpy.run_path(str(HERE.parent / "analyze-expert-trace.py"))
    header, rows = tools["read_trace"](path)
    if header["mode"] != "lru":
        raise ValueError("reuse-distance diagnostics require an LRU trace")
    phases = {}
    if coordinator_csv:
        with coordinator_csv.open(newline="", encoding="utf-8") as file:
            phases = {int(row["request_id"]): (row["phase"], int(row["position"]), int(row["token_count"]))
                      for row in csv.DictReader(file)}
    parts = [dict(part) for part in header["requests"]]
    next_offset = 0
    per_request = []
    for part in parts:
        start, end = part["begin"], part["end"]
        if start != next_offset or end < start or end > len(rows):
            raise ValueError("non-contiguous trace request boundaries")
        next_offset = end
        if phases:
            phase, position, count = phases[part["request_id"]]
            if (part["position"], part["token_count"]) != (position, count):
                raise ValueError("trace and Coordinator request shapes differ")
            part["phase"] = phase
        elif not part["phase"]:
            part["phase"] = "unknown"
        unique = {}
        for i in range(start, end):
            key, size, _, hit = rows[i]
            rows[i] = (key, size, int(part["phase"] == "decode"), hit)
            unique[key] = size
        hits = sum(row[3] for row in rows[start:end])
        per_request.append({**part, "unique_keys": len(unique), "unique_bytes": sum(unique.values()),
                            "hits": hits, "misses": end - start - hits})
    if next_offset != len(rows) or (phases and len(phases) != len(parts)):
        raise ValueError("trace requests do not cover the history")
    distances = list(tools["distances"](rows))
    capacity = header["slots"]
    candidates = sorted({capacity, *additional_slots, 52, 104, 208, 416, 832})
    replay = {slots: sum(distance is not None and distance < slots for distance in distances)
              for slots in candidates}
    if any(bool(hit) != (distance is not None and distance < capacity)
           for (_, _, _, hit), distance in zip(rows, distances)):
        raise ValueError("serial LRU hit/miss differs from replay")
    if replay[capacity] != header["hits"] or len(rows) - replay[capacity] != header["reads"]:
        raise ValueError("observed counters differ from replay")
    source_totals = defaultdict(lambda: Counter(hits=0, misses=0))
    kind_totals = defaultdict(lambda: Counter(hits=0, misses=0))
    layer_totals = defaultdict(lambda: Counter(hits=0, misses=0))
    expert_totals = defaultdict(lambda: Counter(hits=0, misses=0))
    phase_totals = defaultdict(lambda: Counter(hits=0, misses=0, first_use=0, reuse_after_eviction=0))
    source_layers = {}
    accesses_by_key = Counter()
    accesses_by_layer = Counter()
    reuse_by_phase = defaultdict(list)
    for part in per_request:
        for i in range(part["begin"], part["end"]):
            (source, expert), _, _, hit = rows[i]
            name = header["sources"][str(source)]
            match = re.fullmatch(r"blk[.]([0-9]+)[.]ffn_(gate|up|down)_exps[.]weight", name)
            layer, kind = match.groups() if match else ("other", "other")
            source_layers[source] = layer
            accesses_by_key[source, expert] += 1
            accesses_by_layer[layer] += 1
            reuse_by_phase[part["phase"]].append(distances[i])
            outcome = "hits" if hit else "misses"
            for collection, key in ((source_totals, name), (kind_totals, kind), (layer_totals, layer),
                                    (expert_totals, f"{name}:{expert}"), (phase_totals, part["phase"])):
                collection[key][outcome] += 1
            if distances[i] is None:
                phase_totals[part["phase"]]["first_use"] += 1
            elif not hit:
                phase_totals[part["phase"]]["reuse_after_eviction"] += 1
    reused = [distance for distance in distances if distance is not None]
    ranked = sorted(accesses_by_key, key=lambda key: (-accesses_by_key[key], key))
    policies = {}
    for slots in candidates:
        partitions = tools["allocate"](slots, accesses_by_layer, False)
        policies[slots] = {
            "global_lru": tools["replay"](rows, slots, source_layers),
            "equal_layer_lru": tools["replay"](rows, slots, source_layers, limits=partitions),
            "frequency_oracle": tools["replay"](rows, slots, source_layers, keep=set(ranked[:slots])),
        }
    return {"trace_metadata": header, "requests": per_request, "hits_by_simulated_slots": replay,
            "policies": policies, "phase_reuse": {phase: tools["distribution"](values)
                                                    for phase, values in reuse_by_phase.items()},
            "reuse": tools["distribution"](distances),
            "first_hit_slots": min(reused) + 1 if reused else None,
            "phase": phase_totals, "layer": layer_totals, "weight_kind": kind_totals,
            "tensor_source": source_totals, "expert_by_source": expert_totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="append", type=Path, help="run folder in original execution order; repeat")
    group.add_argument("--trace", type=Path)
    parser.add_argument("--coordinator-csv", type=Path, help="assign Worker request phases from Coordinator")
    parser.add_argument("--slots", action="append", type=int, default=[], help="additional replay slot count; repeat")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run and (args.coordinator_csv or args.slots):
        parser.error("--coordinator-csv and --slots require --trace")
    if any(slots <= 0 for slots in args.slots):
        parser.error("--slots must be positive")
    try:
        result = compare_folders(args.run) if args.run else diagnose(args.trace, args.coordinator_csv, args.slots)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.exit(1, f"capacity analysis failed: {error}\n")
    if args.run and any(run["status"] != "ok" for run in result["ordered_runs"]):
        parser.exit(1, "one or more runs excluded; see the JSON\n")


if __name__ == "__main__":
    main()

