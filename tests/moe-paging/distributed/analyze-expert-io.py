"""Describe GGUF Expert reads and replay within-tensor coalescing offline."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics
import struct

TRACE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "analyze-expert-trace.py"))
EXPERT = re.compile(r"blk[.]([0-9]+)[.]ffn_(gate|up|down)_exps[.]weight")
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
         30: ("BF16", 1, 2)}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}
U64_MAX = (1 << 64) - 1


def checked_end(start, size, limit):
    if not 0 <= start <= limit <= U64_MAX or not 0 < size <= limit - start:
        raise ValueError("invalid file range")
    return start + size


def inventory(paths):
    tensors = {}
    for shard, path in enumerate(paths):
        limit = path.stat().st_size
        with path.open("rb") as stream:
            def read(size):
                if size > limit - stream.tell():
                    raise ValueError("truncated GGUF metadata")
                return stream.read(size)

            def number(fmt):
                return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

            def skip(size):
                if size > limit - stream.tell():
                    raise ValueError("truncated GGUF value")
                stream.seek(size, 1)

            def string():
                size = number("Q")
                if size > 1 << 20:
                    raise ValueError("unexpectedly long GGUF key or tensor name")
                return read(size).decode("utf-8")

            def skip_value(kind):
                if kind in SCALARS:
                    skip(struct.calcsize("<" + SCALARS[kind]))
                elif kind == 8:
                    skip(number("Q"))
                elif kind == 9:
                    item, count = number("I"), number("Q")
                    if item in SCALARS:
                        skip(count * struct.calcsize("<" + SCALARS[item]))
                    elif item == 8 and count <= (limit - stream.tell()) // 8:
                        for _ in range(count):
                            skip(number("Q"))
                    else:
                        raise ValueError("unsupported GGUF array")
                else:
                    raise ValueError("unsupported GGUF metadata type")

            if read(4) != b"GGUF" or number("I") not in (2, 3):
                raise ValueError("expected little-endian GGUF v2/v3")
            tensor_count, metadata_count = number("Q"), number("Q")
            alignment = 32
            for _ in range(metadata_count):
                name, kind = string(), number("I")
                if name == "general.alignment":
                    if kind != 4:
                        raise ValueError("invalid GGUF alignment type")
                    alignment = number("I")
                else:
                    skip_value(kind)
            if not alignment or alignment & (alignment - 1):
                raise ValueError("GGUF alignment must be a power of two")
            entries = []
            for _ in range(tensor_count):
                name, dimensions = string(), number("I")
                if not 1 <= dimensions <= 4:
                    raise ValueError("invalid tensor rank")
                shape = [number("Q") for _ in range(dimensions)]
                quant, relative = number("I"), number("Q")
                match = EXPERT.fullmatch(name)
                if match:
                    entries.append((name, shape, quant, relative, match))
            data = (stream.tell() + alignment - 1) & -alignment
            for name, shape, quant, relative, match in entries:
                if name in tensors or len(shape) != 3 or not all(shape) or relative % alignment or quant not in TYPES:
                    raise ValueError(f"invalid or unsupported Expert tensor: {name}")
                type_name, block, block_bytes = TYPES[quant]
                if shape[0] % block:
                    raise ValueError(f"unaligned quantization rows: {name}")
                stride = shape[0] // block * block_bytes * shape[1]
                offset = data + relative
                checked_end(offset, stride * shape[2], limit)
                tensors[name] = {"shard": shard, "offset": offset, "tensor_data_offset": relative,
                                 "stride": stride, "count": shape[2], "type": type_name,
                                 "alignment": alignment, "layer": int(match[1]), "kind": match[2]}
    return tensors


def distribution(values):
    samples = sorted(values)
    if not samples:
        return {"n": 0}
    return {"n": len(samples), "sum": sum(samples), "median": statistics.median(samples),
            "p95": samples[math.ceil(.95 * len(samples)) - 1],
            "p99": samples[math.ceil(.99 * len(samples)) - 1], "max": samples[-1]}


def coalesce(ranges, gap, maximum):
    """Merge sorted intervals only within one shard and one known demand group."""
    spans = []
    for shard, offset, size in sorted(ranges):
        end = offset + size
        if spans and spans[-1][0] == shard:
            _, start, stop, count = spans[-1]
            if offset - stop <= gap and max(stop, end) - start <= maximum:
                spans[-1] = (shard, start, max(stop, end), count + 1)
                continue
        spans.append((shard, offset, end, 1))
    return spans


def layout_groups(tensors):
    groups = defaultdict(list)
    for tensor in tensors.values():
        groups[tensor["layer"]].append(tensor)
    ratios = []
    adjacent = 0
    checked = 0
    required, covering = [], []
    for items in groups.values():
        if {t["kind"] for t in items} != {"gate", "up", "down"} or len({t["count"] for t in items}) != 1:
            continue
        for expert in range(items[0]["count"]):
            checked += 1
            if len({t["shard"] for t in items}) > 1:
                continue
            ranges = [(t["shard"], t["offset"] + expert * t["stride"], t["stride"]) for t in items]
            size = sum(t["stride"] for t in items)
            span = max(offset + length for _, offset, length in ranges) - min(offset for _, offset, _ in ranges)
            adjacent += span == size
            required.append(size)
            covering.append(span)
            ratios.append(span / size)
    return {"triplets": checked, "adjacent": adjacent, "same_shard_amplification": distribution(ratios),
            "triplet_required_bytes": distribution(required), "triplet_contiguous_span_bytes": distribution(covering),
            "types": dict(Counter(t["type"] for t in tensors.values())),
            "slice_bytes": dict(Counter(t["stride"] for t in tensors.values()))}


def analyze_prefetch_sidecar(stream, summary, sibling, events, sources, metrics):
    """Reconcile actual Reader calls separately from the demand-policy trace."""
    if not summary["complete"] or summary["dropped"] or summary["access_dropped"]:
        raise ValueError("incomplete Expert prefetch I/O trace")
    demand_misses = {index for index, (_, _, _, hit) in enumerate(events) if not hit}
    demand_reads, async_reads = {}, {}
    timings = defaultdict(list)
    actual_bytes = 0
    api_calls, api_bytes = 0, 0
    api_available = True
    for line in stream:
        item = json.loads(line)
        if len(item) != 14:
            raise ValueError("invalid prefetch Reader event shape")
        index, source, expert, shard, offset, size, begin, wait, end, calls, returned, api_ns, ok, job = item
        name, tensor = sources[source]
        if (not 0 <= expert < tensor["count"] or shard != tensor["shard"] or size != tensor["stride"]
                or offset != tensor["offset"] + expert * size or not 0 <= wait <= end - begin
                or (ok and returned is not None and returned != size)):
            raise ValueError(f"invalid Reader data for {name}:{expert}")
        if calls is None:
            if returned is not None or api_ns is not None:
                raise ValueError("inconsistent Reader API samples")
            api_available = False
        else:
            if not (calls >= int(ok) and 0 <= returned <= size and 0 <= api_ns <= end - begin - wait):
                raise ValueError("invalid Reader API sample")
            api_calls += calls
            api_bytes += returned
        if job:
            if index is not None or job in async_reads:
                raise ValueError("duplicate or mixed prefetch Reader identity")
            async_reads[job] = item
        else:
            if index not in demand_misses or index in demand_reads or not ok:
                raise ValueError("invalid synchronous demand Reader identity")
            demand_reads[index] = item
        timings["prefetch" if job else "demand"].append(end - begin)
        actual_bytes += size if ok else 0

    delivery = {}
    used = set()
    with sibling(".prefetch-delivery.jsonl").open(encoding="utf-8") as links:
        for line in links:
            index, job, waited = json.loads(line)
            if index not in demand_misses or index in delivery or job in used or job not in async_reads:
                raise ValueError("invalid prefetch delivery identity")
            if not async_reads[job][-2] or type(waited) is not bool:
                raise ValueError("consumed failed or malformed prefetch")
            delivery[index] = async_reads[job]
            used.add(job)
    if set(demand_reads) & set(delivery) or set(demand_reads) | set(delivery) != demand_misses:
        raise ValueError("unmatched or duplicated demand miss")
    for index, item in (demand_reads | delivery).items():
        (source, expert), size, _, _ = events[index]
        if (source, expert, size) != (item[1], item[2], item[5]):
            raise ValueError("Reader key differs from demand key")

    measured = metrics["prefetch"]
    pager = metrics["pager"]
    if ((len(async_reads), len(delivery), len(demand_reads)) !=
            (measured["reads_started"], measured["consumed"], measured["demand_reads"])
            or actual_bytes != pager["bytes_read"]
            or len(async_reads) + len(demand_reads) != summary["reads"]
            or summary["reads"] != pager["disk_reads"]):
        raise ValueError("prefetch reader metrics do not reconcile")
    return {"actual_reads": summary["reads"], "actual_successful_bytes": actual_bytes,
            "read_api_calls": api_calls if api_available else None,
            "read_api_bytes": api_bytes if api_available else None,
            "demand_reads": len(demand_reads), "prefetch_reads": len(async_reads),
            "consumed": len(delivery), "reader_ns": {key: distribution(value) for key, value in timings.items()},
            "measurement": measured}


def analyze(path, tensors, peer_csv=None):
    header, events = TRACE["read_trace"](path)
    role = header["role"]
    base = path.with_name(path.name.removesuffix(".experts.jsonl"))
    sibling = lambda suffix: base.with_name(base.name + suffix)
    metrics = json.loads(sibling(".metrics.json").read_text(encoding="utf-8"))
    if metrics["status"] != "ok" or metrics["role"] != role or metrics.get("pager_policy", "lru") != header["mode"]:
        raise ValueError("trace and successful Stage metrics differ")
    prompt_hash = metrics.get("prompt_token_sha256")
    peer = []
    if role == "worker":
        if peer_csv is None:
            peer_csv = path.with_name("coordinator.perf.csv")
        coordinator_path = peer_csv.with_name(peer_csv.name.removesuffix(".perf.csv") + ".metrics.json")
        coordinator = json.loads(coordinator_path.read_text(encoding="utf-8"))
        if any(coordinator[key] != metrics[key] for key in ("session_id", "model_identity")):
            raise ValueError("Worker and Coordinator sessions differ")
        prompt_hash = coordinator["prompt_token_sha256"]
        with peer_csv.open(encoding="utf-8", newline="") as stream:
            peer = list(csv.DictReader(stream))
        if len(peer) != len(header["requests"]):
            raise ValueError("Worker phase source has different request count")
    transient = set()
    if header["mode"] == "tinylfu":
        with path.open(encoding="utf-8") as stream:
            next(stream)
            for line in stream:
                event = json.loads(line)
                if event[4] >= header["resident_slots"]:
                    transient.add(event[0])
    sources = {int(key): (name, tensors[name]) for key, name in header["sources"].items()}
    reads = {}
    prefetch_io = None
    io_path = sibling(".expert-io.jsonl")
    has_io = io_path.is_file()
    if has_io:
        with io_path.open(encoding="utf-8") as stream:
            io = json.loads(next(stream))
            if io["version"] == 2:
                prefetch_io = analyze_prefetch_sidecar(stream, io, sibling, events, sources, metrics)
                has_io = False
            elif io["version"] != 1 or not io["complete"] or io["dropped"] or io["access_dropped"]:
                raise ValueError("incomplete Reader trace")
            last_index, last_end = -1, 0
            for line in stream:
                item = json.loads(line)
                if (len(item) != 13 or item[0] <= last_index or item[6] < last_end or not item[-1]):
                    raise ValueError("invalid Reader event")
                reads[item[0]] = item
                last_index, last_end = item[0], item[8]
            if has_io and len(reads) != io["reads"]:
                raise ValueError("Reader trace size mismatch")
    categories = {group: defaultdict(Counter) for group in ("phase", "tensor", "layer", "kind", "expert", "slot_kind")}
    reader_times, lock_times, api_times = defaultdict(list), [], []
    previous = {}
    previous_read = {}
    residual_times, same_operation_reader_gaps = [], []
    gaps, signed_gaps, slices = [], [], []
    adjacent = same_tensor_adjacent = repeated = offset_index = api_calls = api_bytes = total_bytes = 0
    missing_api = False
    accesses = set()
    grouped = defaultdict(list)
    for index, request in enumerate(header["requests"]):
        start, end = request["begin"], request["end"]
        if start != offset_index or not start <= end <= len(events):
            raise ValueError("request boundaries do not cover the trace")
        offset_index = end
        phase = request["phase"]
        if peer:
            row = peer[index]
            if any(int(row[field]) != request[field] for field in ("request_id", "position", "token_count")):
                raise ValueError("Coordinator and Worker requests differ")
            phase = row["phase"]
        if phase not in ("prefill", "decode"):
            raise ValueError("missing phase")
        for coordinate in range(start, end):
            (source, expert), size, _, hit = events[coordinate]
            name, tensor = sources[source]
            if expert >= tensor["count"] or size != tensor["stride"]:
                raise ValueError(f"trace differs from GGUF slice {name}:{expert}")
            shard, offset = tensor["shard"], tensor["offset"] + expert * size
            item = reads.get(coordinate)
            if has_io:
                if (item is None) != bool(hit):
                    raise ValueError("Reader event does not match cache miss")
                if item and item[1:6] != [source, expert, shard, offset, size]:
                    raise ValueError("Reader event differs from GGUF")
            slot = "transient" if header["mode"] == "tinylfu" and coordinate in transient else "resident"
            for group, key in (("phase", phase), ("tensor", name), ("layer", tensor["layer"]),
                               ("kind", tensor["kind"]), ("expert", f"{name}:{expert}"), ("slot_kind", slot)):
                counter = categories[group][key]
                counter["accesses"] += 1
                counter["hits" if hit else "misses"] += 1
                counter["requested_bytes"] += 0 if hit else size
            if hit:
                continue
            total_bytes += size
            slices.append(size)
            key = (source, expert)
            repeated += key in accesses
            accesses.add(key)
            if shard in previous:
                previous_end, previous_source = previous[shard]
                distance = offset - previous_end
                signed_gaps.append(distance)
                gaps.append(abs(distance))
                adjacent += distance == 0
                same_tensor_adjacent += distance == 0 and previous_source == source
            previous[shard] = (offset + size, source)
            grouped[index, source].append((shard, offset, size))
            if item:
                begin, lock, finish, calls, obtained, api_ns = item[6:12]
                if not 0 <= lock <= finish - begin or (calls is not None and
                        (obtained != size or calls <= 0 or not 0 <= api_ns <= finish - begin - lock)):
                    raise ValueError("Reader timing or successful bytes are invalid")
                for label in ("all", phase, slot):
                    reader_times[label].append(finish - begin)
                lock_times.append(lock)
                group = (index, source)
                if group in previous_read:
                    same_operation_reader_gaps.append(begin - previous_read[group])
                previous_read[group] = finish
                if calls is None:
                    missing_api = True
                else:
                    api_calls += calls
                    api_bytes += obtained
                    api_times.append(api_ns)
                    residual_times.append(finish - begin - lock - api_ns)
    if offset_index != len(events) or len(slices) != header["reads"] or total_bytes != header["bytes_read"]:
        raise ValueError("Reader counters do not reconcile")
    pager = metrics["pager"]
    hits = len(events) - len(slices)
    if (hits, len(slices)) != (header["hits"], header["misses"]):
        raise ValueError("trace cache counters do not reconcile")
    if (hits, len(slices)) != (pager["cache_hits"], pager["cache_misses"]) or (
            not header.get("prefetch_enabled") and (len(slices), total_bytes) != (pager["disk_reads"], pager["bytes_read"])):
        raise ValueError("Reader counters differ from metrics")
    if has_io and len(reads) != len(slices):
        raise ValueError("unexpected Reader records")
    for ranges in grouped.values():
        if len(set(ranges)) != len(ranges):
            raise ValueError("repeated slice within a request/tensor; cannot infer one MUL_MAT_ID invocation")
    simulation = []
    for gap in (0, 4, 16, 64, 256):
        for maximum in (2, 4, 8, 16):
            calls = transferred = scratch = copies = 0
            for ranges in grouped.values():
                spans = coalesce(ranges, gap * 1024, maximum * 1048576)
                calls += len(spans)
                transferred += sum(end - start for _, start, end, _ in spans)
                sizes = {(shard, offset): size for shard, offset, size in ranges}
                for shard, start, end, count in spans:
                    if count > 1:
                        scratch = max(scratch, end - start)
                        copies += sum(size for (file, offset), size in sizes.items()
                                      if file == shard and start <= offset < end)
            simulation.append({"gap_kib": gap, "max_mib": maximum, "virtual_reads": calls,
                               "reads_saved": len(slices) - calls, "transferred_bytes": transferred,
                               "extra_gap_bytes": transferred - total_bytes,
                               "amplification": transferred / total_bytes if total_bytes else None,
                               "peak_scratch_bytes": scratch, "slot_copy_bytes": copies})
    return {"role": role, "model_identity": metrics["model_identity"], "identity_source": metrics["identity_source"],
            "revision": metrics["revision"], "prompt_token_sha256": prompt_hash,
            "trace_path": str(path), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "policy": header["mode"], "budget_mib": metrics["pager_budget_mib"], "trace_complete": True,
            "timed_reader_trace": has_io or prefetch_io is not None, "prefetch_io": prefetch_io,
            "reads": len(slices), "requested_bytes": total_bytes,
            "existing_io_wait_ns": pager["io_wait_ns"], "read_api_calls": api_calls if reads and not missing_api else None,
            "read_api_bytes": api_bytes if reads and not missing_api else None,
            "explicit_expert_copy_bytes": 0 if metrics["io"] in ("mmap", "buffered") and api_times else None,
            "explicit_expert_copy_ns": None, "reader_ns": {key: distribution(value) for key, value in reader_times.items()},
            "shard_lock_ns": distribution(lock_times), "read_api_ns_per_reader": distribution(api_times),
            "reader_residual_ns": distribution(residual_times),
            "same_operation_between_readers_ns": distribution(same_operation_reader_gaps),
            "slice_bytes": distribution(slices), "same_shard_absolute_gap_bytes": distribution(gaps),
            "same_shard_signed_gap_bytes": distribution(signed_gaps),
            "adjacent_forward_reads": adjacent, "same_tensor_adjacent_forward_reads": same_tensor_adjacent,
            "repeated_expert_reads": repeated, "groups": categories, "simulation_same_request_tensor": simulation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", nargs="+", type=Path, required=True, help="shards in loader order")
    parser.add_argument("--trace", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        tensors = inventory(args.gguf)
        if not tensors:
            raise ValueError("no supported Expert tensors found")
        reports = [analyze(trace, tensors) for trace in args.trace]
        args.output.write_text(json.dumps({"version": 1, "layout": layout_groups(tensors),
                                          "tensors": tensors, "runs": reports}, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, struct.error) as error:
        parser.exit(1, f"Expert I/O analysis failed: {error}\n")


if __name__ == "__main__":
    main()
