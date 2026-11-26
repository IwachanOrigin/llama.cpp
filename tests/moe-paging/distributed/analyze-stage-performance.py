"""Validate and compare Stage TCP schema-v1 generation measurements."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys


def load(path, role):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("status") != "ok" or data.get("role") != role:
        raise ValueError(f"{path}: expected successful schema-v1 {role} metrics")
    data.setdefault("expert_prefetch_depth", 0)
    data.setdefault("pager_policy", "lru")  # Older Stage binaries use Global LRU.
    suffix = ".metrics.json"
    if not path.name.endswith(suffix):
        raise ValueError(f"{path}: expected a .metrics.json filename")
    csv_path = path.with_name(path.name[:-len(suffix)] + ".perf.csv")
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != data["stage"]["stage_execution_calls"] or any(
            row["role"] != role or int(row["request_id"]) != i
            for i, row in enumerate(rows, 1)):
        raise ValueError(f"{csv_path}: invalid role or request sequence")
    if sum(int(row["token_count"]) for row in rows) != data["stage"]["input_tokens"]:
        raise ValueError(f"{csv_path}: input token count differs from JSON")
    if role != "worker":
        generation = data["generation"]
        if any(row["phase"] not in ("prefill", "decode") for row in rows) or (
                sum(row["phase"] == "decode" for row in rows) != generation["decode_steps"]):
            raise ValueError(f"{csv_path}: invalid generation phases")
    symbols = (".generated_tokens", ".generated.txt")
    outputs = tuple(path.with_name(path.name[:-len(suffix)] + ext).read_bytes()
                    for ext in symbols) if role != "worker" else ()
    return data, rows, outputs


def compare(paths):
    baseline, coordinator, worker = (load(path, role) for path, role in zip(paths, (
        "baseline", "coordinator", "worker")))
    values = [item[0] for item in (baseline, coordinator, worker)]
    base, front, back = values
    shared = ("revision", "version", "model_identity", "identity_source", "context",
              "effective_context", "batch", "threads", "pager_budget_mib", "pager_policy", "expert_prefetch_depth", "io",
              "hidden", "vocab", "split", "layers", "cache_state_label")
    for key in shared:
        if any(item[key] != base[key] for item in values):
            raise ValueError(f"run condition differs: {key}")
    for key in ("chat", "max_new_tokens", "prompt_token_sha256"):
        if base[key] != front[key]:
            raise ValueError(f"prompt or generation setting differs: {key}")
    if [item["layer_range"] for item in values] != [[0, base["layers"]], [0, base["split"]],
                                                   [base["split"], base["layers"]]]:
        raise ValueError("inconsistent stage ranges")
    mode = (front.get("protocol_version", 1), front.get("logits_response", "full"),
            front.get("compact_capability", False))
    peer_mode = (back.get("protocol_version", 1), back.get("logits_response", "full"),
                 back.get("compact_capability", False))
    if mode != peer_mode or mode not in ((1, "full", False), (2, "worker-greedy", True)):
        raise ValueError("Stage protocol or negotiated response mode differs")
    if not front["session_id"] or front["session_id"] != back["session_id"]:
        raise ValueError("Coordinator and Worker session IDs differ")
    if front["tcp"]["sent_bytes"] != back["tcp"]["received_bytes"] or (
            front["tcp"]["received_bytes"] != back["tcp"]["sent_bytes"]):
        raise ValueError("TCP totals disagree")
    if baseline[2] != coordinator[2]:
        raise ValueError("generated token IDs or UTF-8 bytes differ")
    for key in ("phase", "position", "token_count"):
        if [row[key] for row in baseline[1]] != [row[key] for row in coordinator[1]]:
            raise ValueError(f"baseline and distributed schedules differ: {key}")
    if any(base["generation"][key] != front["generation"][key]
           for key in ("generated_tokens", "stop", "decode_steps")):
        raise ValueError("generation counters differ")
    if len(coordinator[1]) != len(worker[1]):
        raise ValueError("Coordinator and Worker request counts differ")
    joined = {phase: {"worker_execution_ms": 0.0, "worker_reads": 0, "worker_bytes_read": 0, "worker_io_wait_ms": 0.0,
                      "roundtrip_ms": 0.0, "rtt_minus_worker_execution_ms": 0.0,
                      "send_wall_ms": 0.0, "receive_wall_ms": 0.0,
                      "serialization_ms": 0.0}
              for phase in ("prefill", "decode")}
    for local, remote in zip(coordinator[1], worker[1]):
        if any(local[key] != remote[key] for key in ("request_id", "position", "token_count")):
            raise ValueError("request IDs or shapes differ")
        if any(local[a] != remote[b] for a, b in (("send_bytes", "receive_bytes"), ("receive_bytes", "send_bytes"))):
            raise ValueError("request byte counts disagree")
        phase = joined[local["phase"]]
        wall = float(remote["stage_execution_ms"])
        roundtrip = float(local["request_roundtrip_ms"])
        phase["worker_execution_ms"] += wall
        phase["roundtrip_ms"] += roundtrip
        phase["rtt_minus_worker_execution_ms"] += roundtrip - wall
        phase["worker_reads"] += int(remote["pager_read_count_delta"])
        phase["worker_bytes_read"] += int(remote["pager_bytes_read_delta"])
        phase["worker_io_wait_ms"] += float(remote["pager_io_wait_ms_delta"])
        for key in ("send_wall_ms", "receive_wall_ms"):
            phase[key] += float(local[key])
        phase["serialization_ms"] += sum(float(row[key]) for row in (local, remote)
                                         for key in ("serialize_ms", "deserialize_ms"))
    return values, joined


def render(results):
    print("Role | Prefill ms | Additional decode ms | Generation ms | Stage wall ms | TCP sent / received | Reader calls | Requested bytes | I/O wait ms")
    print("---|---:|---:|---:|---:|---:|---:|---:|---:")
    for run, joined in results:
        print()
        print(f'Stage TCP v{run[1].get("protocol_version", 1)} '
              f'{run[1].get("logits_response", "full")}')
        for item in run:
            gen = item.get("generation", {})
            phase = lambda key: f"{gen[key]:.3f}" if key in gen else "-"
            tcp = item["tcp"]
            pager = item["pager"]
            print(f'{item["role"]} | {phase("prefill_ms")} | {phase("decode_step_wall_ms")} | '
                  f'{phase("generation_elapsed_ms")} | {item["stage"]["stage_execution_wall_ms"]:.3f} | '
                  f'{tcp["sent_bytes"]} / {tcp["received_bytes"]} | {pager["disk_reads"]} | '
                  f'{pager["bytes_read"]} | {pager["io_wait_ns"]/1e6:.3f}')
        if run[0]["identity_source"] != "local-sha256":
            print("\nGGUF content identity was not independently verified by all stages.")
        print("\nRequests joined by session and request ID. Phase durations in ms.")
        for phase, data in joined.items():
            print(phase, json.dumps(data, sort_keys=True))
            front_wall = run[1]["phases"][phase]["stage_execution_wall_ms"]
            back_wall = data["worker_execution_ms"]
            residual = data["rtt_minus_worker_execution_ms"]
            candidates = {"Coordinator Stage wall": front_wall, "Worker Stage wall": back_wall,
                          "round-trip residual": residual}
            name = max(candidates, key=candidates.get)
            if candidates[name] > 0:
                print(f"  Largest observed component: {name} ({candidates[name]:.3f} ms).")
        for item in run:
            if item["role"] != "baseline":
                print(item["role"], "transport", json.dumps(item.get("transport"), sort_keys=True),
                      "" if "transport" in item else "legacy OS default and split send")
            print(item["role"], "memory", json.dumps(item["process_memory"], sort_keys=True),
                  "buffers", json.dumps(item["model_memory"], sort_keys=True))
            print(item["role"], "pager", json.dumps(item["pager"], sort_keys=True))
            if "prefetch" in item:
                print(item["role"], "prefetch", json.dumps(item["prefetch"], sort_keys=True))
        print()
    for field in ("prefill_ms", "decode_step_wall_ms", "generation_elapsed_ms"):
        b = [run[0]["generation"][field] for run, _ in results]
        c = [run[1]["generation"][field] for run, _ in results]
        def describe(samples):
            return f"{statistics.median(samples):.3f} [{min(samples):.3f}, {max(samples):.3f}]"
        ratio = statistics.median(c) / statistics.median(b) if statistics.median(b) else None
        print(f"{field} median [min, max]: baseline {describe(b)}, distributed {describe(c)}, ratio {ratio}")
    print("\nRTT minus Worker execution includes transfer, framing, serialization and scheduling. "
          "It is not network latency. Pager reader counters measure application requests, not physical I/O. "
          "Compare Stage wall times and reader waits to locate candidate bottlenecks before profiling CPU or SSD.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("baseline", "coordinator", "worker"):
        parser.add_argument("--" + role, action="append", type=Path, required=True)
    args = parser.parse_args()
    if len({len(args.baseline), len(args.coordinator), len(args.worker)}) != 1:
        parser.error("provide one baseline, coordinator and worker path per repetition")
    try:
        results = [compare(triple) for triple in zip(args.baseline, args.coordinator, args.worker)]
        reference = results[0][0][0]
        first_coordinator = results[0][0][1]
        for values, _ in results[1:]:
            if (values[1].get("logits_response", "full"), "tcp_detail" in values[1]) != (
                    first_coordinator.get("logits_response", "full"), "tcp_detail" in first_coordinator):
                raise ValueError("repetition differs: response mode or TCP diagnostic setting")
            if any(values[index].get("transport") != results[0][0][index].get("transport") for index in (1, 2)):
                raise ValueError("repetition differs: per-role transport settings")
            for key in ("prompt_token_sha256", "model_identity", "revision", "context", "batch", "threads",
                        "pager_budget_mib", "pager_policy", "expert_prefetch_depth", "io", "max_new_tokens", "cache_state_label"):
                if values[0][key] != reference[key]:
                    raise ValueError(f"repetition differs: {key}")
        if len(results) > 1:
            output = load(args.baseline[0], "baseline")[2]
            if any(load(path, "baseline")[2] != output for path in args.baseline[1:]):
                raise ValueError("generated output differs across repetitions")
        render(results)
    except (ValueError, KeyError, OSError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

