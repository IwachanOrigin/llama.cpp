"""Run each measurement in a fresh process and keep tracing out of timings."""

import argparse
import array
import filecmp
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def compare_logits(reference, candidate):
    if reference.stat().st_size != candidate.stat().st_size or reference.stat().st_size % 4:
        raise RuntimeError(f"invalid logits length: {candidate}")
    result = dict(values=reference.stat().st_size // 4, mismatches=0, max_abs=0.0, max_rel=0.0)
    if filecmp.cmp(reference, candidate, shallow=False):
        return result
    with reference.open("rb") as original, candidate.open("rb") as current:
        while (expected := original.read(65536)):
            actual = current.read(len(expected))
            if actual == expected:
                continue
            left, right = array.array("f"), array.array("f")
            left.frombytes(expected)
            right.frombytes(actual)
            for i, (a, b) in enumerate(zip(left, right)):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise RuntimeError("non-finite FP32 logits")
                if expected[i*4:i*4+4] != actual[i*4:i*4+4]:
                    result["mismatches"] += 1
                    difference = abs(a - b)
                    result["max_abs"] = max(result["max_abs"], difference)
                    result["max_rel"] = max(result["max_rel"], difference / max(abs(a), 1e-30))
    return result


def compare_contexts(reference, candidate, baseline, sample):
    results = []
    for i, (expected, measured) in enumerate(zip(baseline["context_results"], sample["context_results"])):
        def path(prefix, suffix):
            return prefix.parent / f"{prefix.name}.{i}.{suffix}"
        left, right = path(reference, "ids"), path(candidate, "ids")
        for prefix, report in ((reference, expected), (candidate, measured)):
            for field, suffix in (("expert_ids", "ids"), ("prefill_values", "prefill.f32"),
                                  ("decode_values", "decode.f32")):
                if path(prefix, suffix).stat().st_size != 4 * report[field]:
                    raise RuntimeError(f"trace size differs from the reported {field}: {prefix}, context {i}")
        results.append({
            "expert_ids": right.stat().st_size // 4,
            "expert_ids_exact": filecmp.cmp(left, right, shallow=False),
            "prefill": compare_logits(path(reference, "prefill.f32"), path(candidate, "prefill.f32")),
            "decode": compare_logits(path(reference, "decode.f32"), path(candidate, "decode.f32")),
        })
    return results


def median_tree(rows):
    result = {}
    for key, value in rows[0].items():
        if isinstance(value, dict):
            result[key] = median_tree([row[key] for row in rows])
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = statistics.median(row[key] for row in rows)
    return result


def model_shards(first):
    split = re.fullmatch(r"(.*)-00001-of-([0-9]{5})[.]gguf", first.name, flags=re.IGNORECASE)
    if not split:
        return [first]
    stem, total = split.groups()
    return [first.with_name(f"{stem}-{i:05}-of-{total}.gguf") for i in range(1, int(total) + 1)]


def execute(args, root, case, estimate=False, trace_prefix=None):
    output = root / "sample.json"
    output.unlink(missing_ok=True)
    command = [str(args.bench.resolve()), "--model", str(args.model.resolve()), "--output", str(output),
               "--contexts", str(case["contexts"]), "--pager", case["pager"], "--io", args.io,
               "--cache-state", args.cache_state]
    for option in ("threads", "ctx", "batch", "ubatch", "prefill", "decode", "warmup"):
        command.extend(("--" + option, str(getattr(args, option))))
    if estimate:
        command.append("--plan")
    else:
        command.extend(("--required-available-bytes", str(case["planned_bytes_with_headroom"])))
    if case["pager"] == "lru":
        command.extend(("--slots", str(case["slots"])) if estimate else ("--budget", str(case["budget_bytes"])))
    if trace_prefix is not None:
        command.extend(("--trace-prefix", str(trace_prefix)))
    elif args.sample_decode_memory and not estimate:
        command.append("--sample-decode-memory")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace")[-12000:])
    sample = json.loads(output.read_text(encoding="utf-8"))
    if sample["model_buffer_bytes_including_pager"] != (
            sample["model_buffer_bytes_excluding_pager"] + sample["peak_allocated_bytes"]):
        raise RuntimeError("model buffer accounting does not reconcile")
    if case["pager"] == "lru" and (
            sample["slots"] != case["slots"] or sample["peak_allocated_bytes"] > sample["budget_bytes"]):
        raise RuntimeError("LRU allocation exceeded its budget or did not match the requested slots")
    return sample


def attempt(case, phase, operation):
    try:
        return operation()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        case["status"] = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "error"
        case["failure"] = {"phase": phase, "detail": str(error)}
        return None


def check_allocation(plan, sample):
    keys = ["peak_allocated_bytes", "kv_bytes", "workspace_bytes"]
    # The no_alloc pass estimates ordinary allocations. A mapped backend buffer
    # reports an address span, which can include alignment gaps.
    if sample["io"] == "buffered":
        keys.append("model_buffer_bytes_excluding_pager")
    for key in keys:
        if sample[key] > plan[key]:
            raise RuntimeError(f"{key} exceeded preflight: {sample[key]} > {plan[key]}")
    if sample["workspace_bytes_after_decode"] > plan["workspace_bytes"]:
        raise RuntimeError("scheduler workspace grew beyond the preflight estimate")
    if sample["peak_allocated_bytes"] != plan["peak_allocated_bytes"]:
        raise RuntimeError("Pager allocation differs from preflight")
    return sample


def matrix(args, root):
    cases = []
    for contexts in args.contexts:
        for mode in args.modes:
            case = {
                "contexts": contexts, "mode": mode,
                "pager": mode if mode in ("off", "single") else "lru",
                "slots": 0 if mode in ("off", "single") else int(mode),
                "runs": [],
                "status": "pending",
                "verification_status": "disabled" if args.no_verify else "not_run",
            }
            cases.append(case)
            plan = attempt(case, "preflight", lambda: execute(args, root, case, estimate=True))
            if plan is None:
                continue
            case["plan"] = plan
            case["budget_bytes"] = plan["budget_bytes"]
            rows = min(args.batch, args.prefill) if not args.no_verify else 1
            case["preflight_bytes"] = {
                "ordinary_model": plan["model_buffer_bytes_excluding_pager"],
                "pager": plan["peak_allocated_bytes"],
                "kv": plan["kv_bytes"],
                "workspace": plan["workspace_bytes"],
                "logits_allowance": contexts * rows * plan["vocab_size"] * 4,
                "other_headroom": args.headroom_mib * 1024**2,
            }
            case["planned_bytes_with_headroom"] = sum(case["preflight_bytes"].values())
            available = plan["available_physical_bytes"]
            case["preflight_limit_bytes"] = min(available, args.max_planned_mib * 1024**2) if args.max_planned_mib else available
            if case["planned_bytes_with_headroom"] > case["preflight_limit_bytes"]:
                case["status"] = "skipped"
                case["skipped"] = "conservative allocation estimate including headroom exceeds the preflight limit"

    def measured(case):
        return check_allocation(case["plan"], execute(args, root, case))

    active = [case for case in cases if case["status"] == "pending"]
    if args.cache_state == "warm":
        for case in active:
            attempt(case, "warmup", lambda: measured(case))
    for repeat in range(args.repeats):
        for case in active if repeat % 2 == 0 else reversed(active):
            if case["status"] == "pending":
                sample = attempt(case, f"measurement {repeat + 1}", lambda: measured(case))
                if sample is not None:
                    case["runs"].append(sample)
    for case in active:
        if case["status"] == "pending":
            case["median"] = median_tree(case["runs"])
            case["status"] = "ok"

    if args.no_verify:
        return cases
    for contexts in args.contexts:
        group = [case for case in active if case["contexts"] == contexts and case["status"] == "ok"]
        baseline = next((case for case in group if case["pager"] == "off"), None)
        if baseline is None:
            for case in group:
                case["verification_status"] = "skipped"
                case["verification_skipped"] = "Pager-off reference unavailable in this matrix"
            continue
        reference = root / f"reference-{contexts}"
        baseline_sample = attempt(baseline, "reference trace",
                                  lambda: execute(args, root, baseline, trace_prefix=reference))
        for index, case in enumerate(group):
            if baseline_sample is None:
                case["verification_status"] = "skipped"
                case["verification_skipped"] = "Pager-off reference trace failed"
                continue
            def verify():
                with TemporaryDirectory(dir=root, prefix=f"candidate-{contexts}-{index}-") as directory:
                    candidate = Path(directory) / "trace"
                    sample = execute(args, root, case, trace_prefix=candidate)
                    return compare_contexts(reference, candidate, baseline_sample, sample)
            verification = attempt(case, "verification", verify)
            if verification is not None:
                case["verification"] = verification
                if any(not check["expert_ids_exact"] or check["prefill"]["mismatches"] or check["decode"]["mismatches"]
                       for check in verification):
                    case["status"] = "mismatch"
            case["verification_status"] = case["status"]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--io", choices=("mmap", "buffered"), default="mmap")
    parser.add_argument("--cache-state", choices=("warm", "uncontrolled", "operator-cold"), default="warm")
    parser.add_argument("--modes", nargs="+", default=["off", "single", "1", "16", "64", "128", "256"])
    parser.add_argument("--contexts", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    for name, default in (("threads", 4), ("ctx", 512), ("batch", 128), ("ubatch", 64),
                          ("prefill", 128), ("decode", 32), ("warmup", 0), ("repeats", 5),
                          ("headroom-mib", 1024), ("timeout", 3600)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--sample-decode-memory", action="store_true")
    parser.add_argument("--max-planned-mib", type=int, default=0, help="additional ceiling for each preflight")
    args = parser.parse_args()
    shards = model_shards(args.model)
    if args.output.exists() and any(shard.exists() and args.output.samefile(shard) for shard in shards):
        parser.error("output path refers to a model shard")
    if args.repeats < 1 or args.headroom_mib < 0 or args.max_planned_mib < 0 or args.timeout < 1 or any(
            value not in ("off", "single") and (not value.isascii() or not value.isdecimal() or int(value) < 1)
            for value in args.modes):
        parser.error("modes must be off, single or a positive LRU slot count")
    if args.cache_state == "operator-cold" and (
            args.repeats != 1 or args.warmup or len(args.modes) != 1 or len(args.contexts) != 1):
        parser.error("operator-cold runs require one case, one repeat and zero warmup")
    try:
        file_bytes = sum(shard.stat().st_size for shard in shards)
    except OSError as error:
        args.output.write_text(json.dumps({
            "file_bytes": None, "cases": [], "status": "error",
            "failure": {"phase": "input", "detail": str(error)},
        }, indent=2) + "\n", encoding="utf-8")
        parser.exit(1, "model input unavailable; see the JSON\n")
    with TemporaryDirectory(prefix="pager-bench-") as path:
        cases = matrix(args, Path(path))
    args.output.write_text(json.dumps({
        "file_bytes": file_bytes, "cases": cases,
    }, indent=2) + "\n", encoding="utf-8")
    if any(case["status"] in ("error", "timeout", "mismatch") for case in cases):
        parser.exit(1, "one or more cases failed; see the JSON\n")
    if all(case["status"] == "skipped" for case in cases):
        parser.exit(2, "all cases were skipped; see the JSON for memory estimates\n")


if __name__ == "__main__":
    main()
