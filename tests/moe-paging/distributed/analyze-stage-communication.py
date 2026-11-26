"""Reconcile Stage TCP generation bytes and model compact response alternatives."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import runpy
import statistics

HEADER = 24
COMPARE = runpy.run_path(str(Path(__file__).with_name("analyze-stage-performance.py")))["compare"]


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"mean": statistics.mean(ordered), "median": statistics.median(ordered),
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99": ordered[math.ceil(.99 * len(ordered)) - 1], "max": ordered[-1], "total": sum(ordered)}


def join(front, worker, hidden, vocab, prompt, generated, compact=False):
    if (len(front) != len(worker) or not front or min(hidden, vocab, prompt, generated) <= 0):
        raise ValueError("invalid request counts or model dimensions")
    position = 0
    requests = []
    for index, (local, remote) in enumerate(zip(front, worker), 1):
        for row in (local, remote):
            if int(row["request_id"]) != index or int(row["position"]) != position:
                raise ValueError("invalid request identity or position")
        count = int(local["token_count"])
        if count <= 0 or int(remote["token_count"]) != count:
            raise ValueError("invalid request token count")
        phase = "prefill" if position < prompt else "decode"
        if local["phase"] != phase or remote["phase"] not in ("", phase):
            raise ValueError("phase mismatch")
        if (phase == "prefill" and position + count > prompt) or (phase == "decode" and count != 1):
            raise ValueError("invalid prefill/decode shape")
        hidden_bytes, logits_bytes = 4 * hidden * count, 4 * vocab * count
        used = int(position + count >= prompt)
        kind = (9 if used else 8) if compact else 4
        sent = HEADER + 16 + 4 * count + hidden_bytes + 4 * compact
        token_bytes = 4 * used if compact else 0
        received = HEADER + 12 + (token_bytes if compact else logits_bytes)
        if sent != int(local["send_bytes"]) or sent != int(remote["receive_bytes"]) or (
                received != int(local["receive_bytes"]) or received != int(remote["send_bytes"])):
            raise ValueError("request bytes do not match wire schema")
        timing = {role + "_" + key: float(row[key]) for role, row in (("coordinator", local), ("worker", remote))
                  for key in ("serialize_ms", "deserialize_ms", "send_wall_ms", "receive_wall_ms")}
        rtt, worker_stage = float(local["request_roundtrip_ms"]), float(remote["stage_execution_ms"])
        if any(not math.isfinite(v) or v < 0 for v in (rtt, worker_stage, *timing.values())):
            raise ValueError("invalid duration")
        # Worker parse, execution, encoding all occur before the result is sent.
        # Blocking send/recv can overlap those intervals, so they are excluded here.
        residual = rtt - worker_stage
        unexplained = residual - timing["worker_deserialize_ms"] - timing["worker_serialize_ms"]
        requests.append({"request": index, "position": position, "phase": phase, "tokens": count,
                         "sent_bytes": sent, "received_bytes": received, "hidden_bytes": hidden_bytes,
                         "logits_bytes": logits_bytes, "logits_rows": count, "sampled_rows": used,
                         "response_kind": kind, "token_id_bytes": token_bytes,
                         "transmitted_logits_bytes": 0 if compact else logits_bytes,
                         "rtt_ms": rtt, "worker_stage_ms": worker_stage,
                         "rtt_minus_worker_ms": residual, "unexplained_ms": unexplained, **timing})
        position += count
    if position != prompt + generated - 1 or sum(r["sampled_rows"] for r in requests) != generated:
        raise ValueError("requests do not match generation schedule")
    return requests


def alternatives(rows, control_sent, control_received, vocab):
    result = {}
    sent = control_sent + sum(row["sent_bytes"] for row in rows)
    baseline = control_received + sum(row["received_bytes"] for row in rows)
    for name, per_row in (("full", None), ("last_or_empty", 4 * vocab), ("worker_greedy_or_empty", 4)):
        payloads = [r["logits_bytes"] if per_row is None else per_row * r["sampled_rows"] for r in rows]
        frames = [HEADER + 12 + size for size in payloads]
        received = control_received + sum(frames)
        result[name] = {"coordinator_bytes": sent, "worker_bytes": received, "total_bytes": sent + received,
                        "worker_saved_bytes": baseline - received,
                        "worker_saved_fraction": (baseline - received) / baseline,
                        "max_stage_result_frame_bytes": max(frames), "max_stage_result_body_bytes": max(frames) - HEADER,
                        "logits_or_token_bytes": sum(payloads)}
    return result


def summarize(rows):
    scalar = ("sent_bytes", "received_bytes", "hidden_bytes", "logits_bytes", "logits_rows", "sampled_rows", "token_id_bytes", "transmitted_logits_bytes")
    return {"requests": len(rows), **{key: sum(row[key] for row in rows) for key in scalar},
            "unused_logits_rows": sum(row["logits_rows"] - row["sampled_rows"] for row in rows),
            "durations": {key: distribution([row[key] for row in rows])
                          for key in rows[0] if key.endswith("_ms")} if rows else {}}


def verify_frame_logs(folder, requests, hello_bytes):
    pattern = re.compile(r"^(tx|rx) type=(\d+) request=(\d+) header=(\d+) payload=(\d+)$", re.M)
    exchanges = [(1, 1, 0, hello_bytes, hello_bytes), (2, 7, 0, 0, 0)]
    exchanges.extend((3, row["response_kind"], row["request"], row["sent_bytes"] - HEADER, row["received_bytes"] - HEADER)
                     for row in requests)
    exchanges.append((5, 7, len(requests) + 1, 0, 0))
    verified = []
    for role in ("coordinator", "worker"):
        path = folder / (role + ".log")
        if not path.is_file():
            continue
        local, remote = ("tx", "rx") if role == "coordinator" else ("rx", "tx")
        expected = [(direction, kind, request, HEADER, size)
                    for sent, received, request, sent_size, received_size in exchanges
                    for direction, kind, size in ((local, sent, sent_size), (remote, received, received_size))]
        observed = [(direction, *map(int, fields)) for direction, *fields in pattern.findall(
            path.read_text(encoding="utf-8", errors="replace"))]
        if observed != expected:
            raise ValueError(f"{role} frame log disagrees with successful metric schedule")
        verified.append(role)
    return verified


def analyze(folder, connection):
    paths = [folder / f"{role}.metrics.json" for role in ("baseline", "coordinator", "worker")]
    reports, _ = COMPARE(paths)
    _, front, worker = reports
    csv_rows = []
    for role in ("coordinator", "worker"):
        with (folder / f"{role}.perf.csv").open(encoding="utf-8", newline="") as stream:
            csv_rows.append(list(csv.DictReader(stream)))
    compact = front.get("protocol_version", 1) == 2
    rows = join(*csv_rows, front["hidden"], front["vocab"], front["prompt_tokens"],
                front["generation"]["generated_tokens"], compact)
    sent = front["tcp"]["sent_bytes"] - sum(row["sent_bytes"] for row in rows)
    received = front["tcp"]["received_bytes"] - sum(row["received_bytes"] for row in rows)
    if sent != received or sent < 3 * HEADER + 44:
        raise ValueError("unexpected session control byte counts")
    verified_logs = verify_frame_logs(folder, rows, sent - 3 * HEADER)
    if compact:
        counters = worker["compact_responses"]
        count = sum(row["logits_rows"] for row in rows)
        selected = sum(row["sampled_rows"] for row in rows)
        expected = {"status_only": len(rows) - selected, "greedy_token": selected,
                    "computed_logits_rows": count, "validated_logits_rows": count,
                    "transmitted_logits_rows": 0, "transmitted_token_ids": selected,
                    "omitted_logits_bytes": count * front["vocab"] * 4}
        if any(counters[key] != value for key, value in expected.items()):
            raise ValueError("compact response counters disagree with schedule")
    diagnostic = {}
    for role, metrics in (("coordinator", front), ("worker", worker)):
        detail = metrics.get("tcp_detail")
        if detail is None:
            continue
        requests = detail["requests"]
        if detail["schema_version"] != 1 or not detail["complete"] or len(requests) != len(rows):
            raise ValueError("incomplete TCP diagnostic")
        for row, event in zip(rows, requests):
            if row["request"] != event["request_id"]:
                raise ValueError("TCP diagnostic request identity differs")
            for kind in ("send", "receive"):
                values = event[kind]
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values()):
                    raise ValueError("invalid diagnostic timing or transfer counter")
                if values["calls"] < 2 or values["partial"] + values["retries"] > values["calls"]:
                    raise ValueError("invalid transfer-call accounting")
            expected_copy = 4 * front["vocab"] if role == "coordinator" and not compact else 0
            if event["last_logits_copy_bytes"] != expected_copy:
                raise ValueError("logits row copy bytes differ")
            row[role + "_tcp_detail"] = event
        diagnostic[role] = {
            "reserved_bytes": detail["reserved_bytes"],
            "last_logits_copy_bytes": sum(item["last_logits_copy_bytes"] for item in requests),
            "last_logits_copy_ms": distribution(
                [item["last_logits_copy_ms"] for item in requests if item["last_logits_copy_ms"] is not None]),
            **{kind: {key: sum(item[kind][key] for item in requests) for key in requests[0][kind]}
               for kind in ("send", "receive")},
        }
    return {
        "version": 1, "connection_label": connection,
        "conditions": {key: front.get(key) for key in (
            "revision", "model_identity", "identity_source", "prompt_token_sha256", "pager_policy",
            "pager_budget_mib", "expert_prefetch_depth", "hidden", "vocab", "prompt_tokens", "binary_arch")},
        "control_bytes_per_direction": sent, "hello_payload_bytes": sent - 3 * HEADER,
        "request_frame_header_bytes": len(rows) * HEADER, "response_frame_header_bytes": len(rows) * HEADER,
        "request_metadata_bytes": sum(16 + 4 * row["tokens"] + 4 * compact for row in rows),
        "response_metadata_bytes": len(rows) * 12, "expert_id_wire_bytes": 0,
        "greedy_ms": front["generation"].get("greedy_ms"),
        "all": summarize(rows), "phases": {phase: summarize([row for row in rows if row["phase"] == phase])
                                             for phase in ("prefill", "decode")},
        "verified_frame_logs": verified_logs,
        "logical_conversion_bytes": {
            "coordinator_hidden_encode": sum(r["hidden_bytes"] for r in rows),
            "worker_hidden_decode": sum(r["hidden_bytes"] for r in rows),
            "worker_logits_encode": sum(r["transmitted_logits_bytes"] for r in rows),
            "coordinator_logits_decode": sum(r["transmitted_logits_bytes"] for r in rows),
            "coordinator_last_row_copy": 0 if compact else len(rows) * 4 * front["vocab"],
        },
        "protocol_version": 2 if compact else 1,
        "logits_response": front.get("logits_response", "full"),
        "worker_compact_responses": worker.get("compact_responses"),
        "diagnostic": diagnostic,
        "alternatives": None if compact else alternatives(rows, sent, received, front["vocab"]),
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--connection", default="unspecified", help="run provenance label")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.run, args.connection)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"communication analysis failed: {error}\n")


if __name__ == "__main__":
    main()
