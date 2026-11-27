"""Extract actual Reader calls or generate a GGUF slice workload."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy

IO = runpy.run_path(str(Path(__file__).with_name("analyze-expert-io.py")))


def from_trace(run, role, tensors):
    def path(suffix):
        return run / (role + suffix)
    header, events = IO["TRACE"]["read_trace"](path(".experts.jsonl"))
    IO["analyze"](path(".experts.jsonl"), tensors)
    with path(".expert-io.jsonl").open(encoding="utf-8") as stream:
        meta = json.loads(next(stream))
        records = [json.loads(line) for line in stream]
    if (meta["version"] not in (1, 2) or not meta["complete"] or meta["dropped"] or
            meta.get("access_dropped", 0) or len(records) != meta["reads"]):
        raise ValueError("incomplete Reader trace")
    deliveries = {}
    if meta["version"] == 2:
        with path(".prefetch-delivery.jsonl").open(encoding="utf-8") as f:
            deliveries = {job: index for index, job, _ in map(json.loads, f)}
    phase_path = run / ("coordinator.perf.csv" if role == "worker" else role + ".perf.csv")
    with phase_path.open(encoding="utf-8", newline="") as f:
        phases = {int(r["request_id"]): r["phase"] for r in csv.DictReader(f)}
    by_demand = [None] * len(events)
    for request in header["requests"]:
        for index in range(request["begin"], request["end"]):
            by_demand[index] = request["request_id"]
    # File emission is completion order. Replay the observed provider-entry order.
    ordered = sorted(enumerate(records), key=lambda item: (item[1][6], item[0]))
    origin = ordered[0][1][6]
    output = []
    for _, row in ordered:
        index, source, expert, shard, offset, size, start, _, end, *rest = row
        job = row[13] if meta["version"] == 2 else 0
        if not row[12] or not 0 <= start <= end:
            raise ValueError("successful replay requires complete successful Reader calls")
        name = header["sources"][str(source)]
        t = tensors[name]
        if (not 0 <= expert < t["count"] or shard != t["shard"] or size != t["stride"] or
                offset != t["offset"] + expert * size):
            raise ValueError("Reader offset differs from GGUF slice")
        if job:
            index = deliveries.get(job)
        request_id = by_demand[index] if index is not None else None
        output.append(dict(shard=shard, offset=offset, size=size, relative_ns=start-origin,
                           phase=phases.get(request_id, "unknown"), request_id=request_id,
                           demand_index=index, source=source, expert=expert, tensor=name,
                           layer=t["layer"], kind="prefetch" if job else "demand"))
    return output, {role+suffix: hashlib.sha256(path(suffix).read_bytes()).hexdigest()
                    for suffix in (".expert-io.jsonl", ".experts.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", action="append", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--role", choices=("baseline", "coordinator", "worker"), default="worker")
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--pattern", choices=("repeat", "sequential", "scattered"))
    parser.add_argument("--tensor")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if bool(args.run) == bool(args.pattern):
            raise ValueError("select a run or a synthetic pattern")
        tensors = IO["inventory"](args.gguf)
        if args.run:
            rows, hashes = from_trace(args.run, args.role, tensors)
            rows = [row for row in rows if args.phase is None or row["phase"] == args.phase]
        else:
            if not 0 < args.samples <= 200000 or args.phase or not tensors:
                raise ValueError("invalid synthetic workload")
            name = args.tensor or next(iter(tensors))
            t = tensors[name]
            def expert(i):
                if args.pattern == "repeat":
                    return 0
                return (i if args.pattern == "sequential" else i * 47) % t["count"]
            rows = [dict(shard=t["shard"], offset=t["offset"] + expert(i) * t["stride"], size=t["stride"],
                         relative_ns=None, phase="synthetic", tensor=name, expert=expert(i),
                         layer=t["layer"], kind="synthetic") for i in range(args.samples)]
            hashes = {}
        if not rows or len(rows) > 200000:
            raise ValueError("unsupported manifest request count")
        result = dict(version=1, shards=[dict(path=str(p.resolve()), size=p.stat().st_size) for p in args.gguf],
                      inputs=dict(role=args.role if args.run else None, phase=args.phase,
                                  pattern=args.pattern, trace_sha256=hashes,
                                  total_bytes=sum(row["size"] for row in rows), requests=len(rows)),
                      requests=rows)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        parser.exit(1, f"Reader manifest failed: {error}\n")


if __name__ == "__main__":
    main()
