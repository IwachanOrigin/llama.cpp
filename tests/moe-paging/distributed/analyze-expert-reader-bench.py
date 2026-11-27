"""Summarize comparable, successful Expert Reader benchmark repetitions."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def summarize(paths):
    trials = defaultdict(list)
    comparison = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["version"] != 1 or data["status"] != "ok" or data["errors"] or
                data["requested_bytes"] != data["completed_bytes"] or data["requests"] <= 0 or
                data["wall_ns"] <= 0 or data["verified"] or
                not math.isclose(data["mib_per_second"], data["completed_bytes"] * 1e9 / data["wall_ns"] / 1048576)):
            raise ValueError(f"{path}: expected successful, unverified performance measurement")
        key = tuple(data.get(k) for k in ("manifest_sha256", "requests", "requested_bytes", "cache_state",
                                         "warmup_requests", "workload", "revision", "os", "binary_arch",
                                         "windows_native_architecture", "windows_native_machine", "timeout_ms"))
        if comparison is not None and key != comparison:
            raise ValueError(f"{path}: incompatible trace, request count or execution conditions")
        comparison = key
        stats = dict(wall_ns=data["wall_ns"], throughput=data["mib_per_second"],
                     p50=data["service_ns"]["p50"], p95=data["service_ns"]["p95"], p99=data["service_ns"]["p99"],
                     latency_p95=data["latency_ns"]["p95"],
                     peak_ws=data.get("process_memory", {}).get("peak_working_set_bytes"),
                     private_commit=data.get("process_memory", {}).get("private_commit_bytes"))
        cpu = data.get("cpu")
        if isinstance(cpu, dict):
            stats.update(user_cpu_ns=cpu["user_ns"], kernel_cpu_ns=cpu["kernel_ns"])
        trials[(data["mode"], data["readers"], data["queue_depth"])].append(stats)
    output = []
    for (mode, readers, depth), rows in sorted(trials.items()):
        values = {}
        for key in sorted(set().union(*(row.keys() for row in rows))):
            samples = [row[key] for row in rows if row.get(key) is not None]
            values[key] = (dict(median=statistics.median(samples), min=min(samples), max=max(samples),
                                measured=len(samples)) if samples else None)
        output.append(dict(mode=mode, readers=readers, queue_depth=depth, trials=len(rows), **values))
    return {"conditions": comparison, "results": output}


def markdown(result):
    lines = ["| Mode | Readers | Trials | MiB/s p50 | Service p50 / p95 / p99 (us) | Peak WS (MiB) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in result["results"]:
        us = " / ".join(f"{r[k]['median']/1000:.2f}" for k in ("p50", "p95", "p99"))
        ws = "-" if r["peak_ws"] is None else f"{r['peak_ws']['median']/1048576:.1f}"
        lines.append(f"| {r['mode']} | {r['readers']} | {r['trials']} | {r['throughput']['median']:.1f} | {us} | {ws} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.trials)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Reader analysis failed: {error}\n")


if __name__ == "__main__":
    main()
