"""Reconcile and decompose delivered Expert prefetches and model bounded lookahead."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import runpy
import statistics

HERE = Path(__file__).resolve().parent
IO = runpy.run_path(str(HERE / "analyze-expert-io.py"))
POLICY = runpy.run_path(str(HERE / "simulate-pager-policies.py"))


def distribution(values, unit="ns"):
    values = sorted(values)
    if not values:
        return None
    def quantile(percent):
        return values[(percent * len(values) + 99) // 100 - 1]
    return {"count": len(values), f"sum_{unit}": sum(values),
            **{f"p{p}_{unit}": quantile(p) for p in (50, 90, 95, 99)}, f"max_{unit}": values[-1]}


def decompose(row, waited):
    _, _, submit, start, end, ready, demand, wait_begin, wait_end, compute_begin, compute_end = row
    if not (0 < submit <= start <= end <= ready and submit <= demand and
            submit <= compute_begin <= compute_end <= demand):
        raise ValueError("invalid prefetch timestamp order")
    if (waited and not (demand <= wait_begin <= ready <= wait_end)) or (
            not waited and (wait_begin or wait_end)):
        raise ValueError("inconsistent condition-variable interval")
    return {
        "queue": start - submit, "service": end - start, "publish": ready - end,
        "lead": demand - submit, "margin": demand - ready,
        "wait": wait_end - wait_begin if waited else 0,
        "resume": wait_end - ready if waited else None,
        "compute": compute_end - compute_begin,
        "overlap": max(0, min(end, compute_end) - max(start, compute_begin)),
    }


def replay(groups, depth, slots, readers, service, compute, layers, mode,
           service_by_index=None, compute_by_index=None, queue_ns=0, resume_ns=0):
    """Single-context, serial demand, known IDs within one tensor only."""
    policy = POLICY["Policy"](mode, slots, 0, layers)
    result = Counter()
    now = 0
    reader_work = 0
    compute_work = 0
    service_by_index = service_by_index or {}
    compute_by_index = compute_by_index or {}
    base = 0
    for group in groups:
        free = [now] * readers
        queued = {}
        for i, (key, size, layer) in enumerate(group):
            outcome, evictions = policy.access(key, layer)
            result[outcome] += 1
            result["evictions"] += evictions
            finish = queued.pop(i, None)
            if outcome != "hits":
                if finish is None:
                    reader = min(range(readers), key=free.__getitem__)
                    duration = service_by_index.get(base + i, service[size])
                    now = free[reader] = max(now, free[reader]) + duration
                    result["reads"] += 1
                    result["bytes"] += size
                    result["demand_service_ns"] += duration
                    reader_work += duration
                else:
                    result["consumed"] += 1
                    result["ready"] += finish <= now
                    if finish > now:
                        result["wait_ns"] += finish - now + resume_ns
                        now = finish + resume_ns
            elif finish is not None:
                result["discarded"] += 1
                free = [max(t, finish) for t in free]
                now = max(now, finish)
            for j in range(i + 1, min(i + 1 + depth, len(group))):
                future_key, future_size, _ = group[j]
                if j in queued or future_key in policy.main:
                    continue
                reader = min(range(readers), key=free.__getitem__)
                start = max(now + queue_ns, free[reader])
                # Multi-reader remains hypothetical at unchanged per-request service.
                duration = service_by_index.get(base + j, service[future_size])
                queued[j] = free[reader] = start + duration
                result["reads"] += 1
                result["bytes"] += future_size
                result["issued"] += 1
                reader_work += duration
            duration = compute_by_index.get(base + i, compute)
            now += duration
            compute_work += duration
        now = max(now, *free)
        base += len(group)
    return dict(result) | {"depth": depth, "resident_slots": slots, "readers": readers,
                           "modeled_expert_loop_ns": now,
                           "reader_busy_fraction_modeled": reader_work / (now * readers) if now else None,
                           "ideal_compute_only_ns": compute_work}


def analyze(folder, role, gguf):
    def path(suffix):
        return folder / (role + suffix)
    header, events = IO["TRACE"]["read_trace"](path(".experts.jsonl"))
    tensors = IO["inventory"](gguf)
    reconciled = IO["analyze"](path(".experts.jsonl"), tensors)
    metrics = json.loads(path(".metrics.json").read_text(encoding="utf-8"))
    sources = {int(k): v for k, v in header["sources"].items()}
    with path(".expert-io.jsonl").open(encoding="utf-8") as f:
        io_header = json.loads(next(f))
        if not io_header["complete"] or io_header["dropped"] or io_header.get("access_dropped", 0):
            raise ValueError("incomplete reader trace")
        reads = [json.loads(line) for line in f]
    with (folder / ("coordinator.perf.csv" if role == "worker" else role + ".perf.csv")).open(
            encoding="utf-8", newline="") as f:
        phase = {int(row["request_id"]): row["phase"] for row in csv.DictReader(f)}
    with path(".experts.jsonl").open(encoding="utf-8") as f:
        next(f)
        slots_by_index = [json.loads(line)[4] for line in f]
    phases, groups = {}, []
    types = defaultdict(Counter)
    for span in header["requests"]:
        group, previous = [], None
        label = phase[span["request_id"]]
        for idx in range(span["begin"], span["end"]):
            key, size, _, hit = events[idx]
            source = key[0]
            layer, kind = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight",
                                       sources[source]).groups()
            phases[idx] = (label, source, int(layer), kind, size)
            counts = types[phases[idx]]
            counts["hit" if hit else "miss"] += 1
            if not hit:
                counts["bypassed" if slots_by_index[idx] >= header["resident_slots"] else "admitted"] += 1
            if group and previous != source:
                groups.append(group)
                group = []
            group.append((key, size, int(layer)))
            previous = source
        if group:
            groups.append(group)
    subsets = defaultdict(lambda: defaultdict(list))
    service = defaultdict(list)
    for row in reads:
        service[row[5]].append(row[8] - row[6])
        subsets[f"reader_slice/{row[5]}"]["provider_service"].append(row[8] - row[6])
        subsets["all_reader"]["provider_service"].append(row[8] - row[6])
        subsets["all_reader"]["shard_mutex"].append(row[7])
        if row[11] is not None:
            subsets["all_reader"]["file_api"].append(row[11])
    timing = path(".prefetch-timing.jsonl")
    simulation = []
    if timing.exists():
        async_reads = {r[13]: r for r in reads if len(r) == 14 and r[13]}
        service_by_index = {row[0]: row[8] - row[6] for row in reads if row[0] is not None}
        compute_by_index = {}
        links = {}
        with path(".prefetch-delivery.jsonl").open(encoding="utf-8") as f:
            for line in f:
                index, job, waited = json.loads(line)
                links[index] = (job, waited)
        seen = set()
        with timing.open(encoding="utf-8") as f:
            meta = json.loads(next(f))
            if meta["version"] != 1 or not meta["complete"] or meta["dropped_accesses"]:
                raise ValueError("incomplete prefetch timing trace")
            for line in f:
                row = json.loads(line)
                idx, job = row[:2]
                if idx in seen or links[idx][0] != job or job not in async_reads:
                    raise ValueError("invalid prefetch timing identity")
                seen.add(idx)
                sample = decompose(row, links[idx][1])
                service_by_index[idx] = sample["service"] + sample["publish"]
                compute_by_index[idx - 1] = sample["compute"]
                label = phases[idx]
                types[label]["wait" if links[idx][1] else "ready"] += 1
                for dest in ("all_delivered", label[0], "/".join(map(str, label))):
                    for key, value in sample.items():
                        if value is not None:
                            subsets[dest][key].append(value)
        if seen != set(links) or len(seen) != metrics["prefetch"]["consumed"]:
            raise ValueError("missing delivery timestamp")
        compute = round(statistics.median(subsets["all_delivered"]["compute"]))
        queue = round(statistics.median(subsets["all_delivered"]["queue"]))
        resumes = subsets["all_delivered"]["resume"]
        resume = round(statistics.median(resumes)) if resumes else 0
        service_median = {size: round(statistics.median(v)) for size, v in service.items()}
        peak = header["managed_bytes"]
        stride = header["max_expert_bytes"]
        budget = metrics["pager_budget_mib"] * (1 << 20)
        resident = header["resident_slots"]
        layers = set(range(*metrics["layer_range"]))
        for depth, readers in ((0, 1), (1, 1), (2, 1), (4, 1), (8, 1), (2, 2), (4, 2)):
            # Replace resident payloads one-for-one; conservatively retain the
            # observed metadata estimate and reserve 64 bytes per new job.
            slots = resident + 1 - depth
            byte_estimate = peak + max(0, depth - 1) * 64
            while byte_estimate > budget:
                slots -= 1
                byte_estimate -= stride
            if slots <= 0:
                continue
            simulation.append(replay(groups, depth, slots, readers, service_median, compute, layers, header["mode"],
                                     service_by_index, compute_by_index, queue, resume) |
                              {"budget_bytes": budget, "estimated_bytes": byte_estimate})
    miss_runs, consecutive = [], 0
    for _, _, _, hit in events:
        if hit:
            if consecutive:
                miss_runs.append(consecutive)
            consecutive = 0
        else:
            consecutive += 1
    if consecutive:
        miss_runs.append(consecutive)
    stage_ns = metrics["stage"]["stage_execution_wall_ms"] * 1e6
    measured = metrics.get("prefetch", {})
    return {"role": role, "events": len(events), "resident_slots": header["resident_slots"],
            "managed_bytes": header["managed_bytes"], "stage_ns": stage_ns,
            "reader_busy_fraction_stage": measured["reader_ns"] / stage_ns if stage_ns and measured.get("enabled") else None,
            "miss_run_lengths": distribution(miss_runs, "accesses"),
            "reader_summary": {"calls": metrics["pager"]["disk_reads"], "bytes": metrics["pager"]["bytes_read"],
                               "adjacent_same_tensor_reads": reconciled["same_tensor_adjacent_forward_reads"]},
            "timing_available": timing.is_file(), "measured_prefetch": measured,
            "intervals": {group: {name: distribution(vals) for name, vals in data.items()}
                          for group, data in subsets.items()},
            "accesses": [{"phase": key[0], "source": key[1], "layer": key[2], "kind": key[3],
                          "slice_bytes": key[4], **counts} for key, counts in sorted(types.items())],
            "simulation": simulation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline", "coordinator", "worker"), required=True)
    parser.add_argument("--gguf", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.run, args.role, args.gguf)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            keys = ("phase", "source", "layer", "kind", "slice_bytes", "hit", "miss", "admitted", "bypassed", "ready", "wait")
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(result["accesses"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"prefetch analysis failed: {error}\n")


if __name__ == "__main__":
    main()
