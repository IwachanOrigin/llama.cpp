"""Compare online Expert-cache policies using complete Stage TCP v1 traces."""

import argparse
from collections import Counter, OrderedDict, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import runpy
import struct
import time

HERE = Path(__file__).resolve().parent
TRACE = runpy.run_path(str(HERE.parent / "analyze-expert-trace.py"))
MIB = 1024 ** 2
DEFAULT_BUDGETS = (64, 128, 256, 512, 768, 1024)
VARIANTS = {"lru": (0,), "layer-lru": (0,), "lfu": (0,),
            "slru": (50, 75, 90), "tinylfu": (0,), "w-tinylfu": (1, 5, 10, 20)}


def aligned(size):
    return (size + 63) // 64 * 64


class Layout:
    """64-bit Stage ABI: sizeof(slot)=80; fixed provider overhead from the observed allocation."""

    def __init__(self, header):
        self.stride = aligned(header["max_expert_bytes"])
        slots = header["slots"]
        self.fixed = header["managed_bytes"] - self.stride * slots - aligned(80 * slots)
        if not self.stride or not slots or self.fixed <= 0:
            raise ValueError("invalid fixed-slot layout")

    def bytes(self, slots):
        return self.fixed + self.stride * slots + aligned(80 * slots)

    def slots(self, budget, policy):
        total = max(0, (budget - self.fixed) // self.stride)
        while total and self.bytes(total) > budget:
            total -= 1
        reserved = int(total > 0 and policy in ("tinylfu", "w-tinylfu"))
        slots = total - reserved
        while slots and self.bytes(slots + reserved) + metadata(policy, slots) > budget:
            slots -= 1
        return slots, reserved


def sketch_width(slots):
    return 1 << (max(16, 4 * slots) - 1).bit_length()


def metadata(policy, slots):
    """Incremental packed C++ array estimate beyond existing slot records."""
    if not slots:
        return 0
    if policy == "layer-lru":
        return 0  # budgeted separately using the configured layer count
    if policy in ("slru", "w-tinylfu", "lfu"):
        per_slot = aligned(slots)
    else:
        per_slot = 0
    if policy in ("tinylfu", "w-tinylfu"):
        return per_slot + aligned(4 * sketch_width(slots)) + (128 if policy == "w-tinylfu" else 64)
    return per_slot + (64 if policy in ("lfu", "slru") else 0)


class Sketch:
    def __init__(self, slots):
        self.width = sketch_width(slots)
        self.cells = bytearray(4 * self.width)
        self.period = max(16, 10 * slots)
        self.count = 0

    def indices(self, key):
        digest = hashlib.blake2b(struct.pack("<QQ", *key), digest_size=16).digest()
        return [row * self.width + (word & (self.width - 1))
                for row, word in enumerate(struct.unpack("<4I", digest))]

    def observe(self, key):
        if self.count == self.period:
            for index, value in enumerate(self.cells):
                self.cells[index] = value // 2
            self.count = 0
        for index in self.indices(key):
            self.cells[index] = min(15, self.cells[index] + 1)
        self.count += 1

    def frequency(self, key):
        return min(self.cells[index] for index in self.indices(key))


class Policy:
    def __init__(self, name, slots, parameter, layers):
        self.name, self.slots = name, slots
        self.main, self.protected, self.window = OrderedDict(), OrderedDict(), OrderedDict()
        self.partition = {}
        if name == "layer-lru":
            for index, layer in enumerate(sorted(layers)):
                self.partition[layer] = (slots // len(layers) + (index < slots % len(layers)), OrderedDict())
        self.protected_limit = min(max(0, slots - 1), slots * parameter // 100)
        self.window_limit = min(max(0, slots - 1), max(1, slots * parameter // 100))
        self.sketch = Sketch(slots) if slots and name in ("tinylfu", "w-tinylfu") else None
        self.count = 0

    def access(self, key, layer):
        """Return outcome and number of resident victims; a bypass still reads."""
        if not self.slots:
            return "bypassed", 0
        if self.sketch:
            self.sketch.observe(key)
        if self.name == "layer-lru":
            limit, entries = self.partition[layer]
        else:
            limit, entries = self.slots, self.main
        if self.name == "w-tinylfu":
            return self.window_access(key)
        if self.name == "lfu":
            if self.count == max(16, 10 * self.slots):
                for resident in entries:
                    entries[resident] = max(1, entries[resident] // 2)
                self.count = 0
            self.count += 1
        if key in self.protected:
            self.protected.move_to_end(key)
            return "hits", 0
        if key in entries:
            if self.name == "slru" and self.protected_limit:
                del entries[key]
                self.protected[key] = None
                if len(self.protected) > self.protected_limit:
                    demoted, _ = self.protected.popitem(last=False)
                    entries[demoted] = None
            else:
                if self.name == "lfu":
                    entries[key] = min(255, entries[key] + 1)
                entries.move_to_end(key)
            return "hits", 0
        if not limit:
            return "bypassed", 0
        evictions = 0
        if len(entries) + len(self.protected) == limit:
            victim = min(entries, key=entries.get) if self.name == "lfu" else next(iter(entries))
            if self.sketch and self.sketch.frequency(key) <= self.sketch.frequency(victim):
                return "bypassed", 0
            del entries[victim]
            evictions = 1
        entries[key] = 1
        return "admitted", evictions

    def window_access(self, key):
        if key in self.window:
            self.window.move_to_end(key)
            return "hits", 0
        if key in self.main:
            self.main.move_to_end(key)
            return "hits", 0
        if not self.window_limit:
            evicted = bool(self.main)
            self.main.clear()
            self.main[key] = None
            return "admitted", int(evicted)
        evicted = 0
        if len(self.window) == self.window_limit:
            candidate, _ = self.window.popitem(last=False)
            if len(self.main) == self.slots - self.window_limit:
                victim = next(iter(self.main))
                if self.sketch.frequency(candidate) > self.sketch.frequency(victim):
                    del self.main[victim]
                    self.main[candidate] = None
                evicted = 1
            else:
                self.main[candidate] = None
        self.window[key] = None
        return "admitted", evicted


def replay(rows, policy, observed=False):
    start = time.perf_counter()
    totals = Counter()
    phases, layers = defaultdict(Counter), defaultdict(Counter)
    mismatches = 0
    for key, size, phase, layer, hit in rows:
        outcome, evictions = policy.access(key, layer)
        missed = outcome != "hits"
        mismatches += observed and missed == bool(hit)
        for stats in (totals, phases[phase], layers[layer]):
            stats["accesses"] += 1
            stats[outcome] += 1
            stats["evictions"] += evictions
            stats["bytes_read"] += size if missed else 0
    def counts(stats):
        stats = {key: stats[key] for key in
                 ("accesses", "hits", "admitted", "bypassed", "evictions", "bytes_read")}
        stats["misses"] = stats["admitted"] + stats["bypassed"]
        stats["hit_rate"] = stats["hits"] / stats["accesses"] if stats["accesses"] else None
        return stats
    return {**counts(totals), "phases": {key: counts(value) for key, value in phases.items()},
            "layers": {key: counts(value) for key, value in layers.items()},
            "elapsed_seconds": time.perf_counter() - start, "observed_mismatches": mismatches if observed else None}


def load_run(folder, role):
    path = folder / f"{role}.experts.jsonl"
    header, raw = TRACE["read_trace"](path)
    metrics = json.loads((folder / f"{role}.metrics.json").read_text(encoding="utf-8"))
    front = json.loads((folder / "coordinator.metrics.json").read_text(encoding="utf-8"))
    if (header["role"] != role or metrics["role"] != role or header["contexts"] != 1 or header["mode"] != "lru"
            or metrics["status"] != "ok" or front["status"] != "ok" or not front["chat"]
            or metrics["binary_arch"] != "x86_64"
            or not metrics["expert_trace_limit"] or header["events"] > metrics["expert_trace_limit"]):
        raise ValueError("expected successful x86-64 single-context Stage LRU chat trace")
    for key in ("model_identity", "identity_source", "context", "batch", "threads", "pager_budget_mib",
                "cache_state_label", "io", "binary_arch", "split", "layers", "max_new_tokens"):
        if key == "max_new_tokens" and role == "worker":
            continue
        if metrics[key] != front[key]:
            raise ValueError(f"mismatched run condition: {key}")
    if role != "worker":
        if metrics["prompt_token_sha256"] != front["prompt_token_sha256"]:
            raise ValueError("Baseline and Coordinator prompt tokens differ")
        if any(metrics["generation"][key] != front["generation"][key]
               for key in ("generated_tokens", "stop", "decode_steps")):
            raise ValueError("Baseline and Coordinator generation metadata differs")
    source_layers = {int(source): int(re.fullmatch(
        r"blk[.]([0-9]+)[.]ffn_(gate|up|down)_exps[.]weight", name).group(1))
        for source, name in header["sources"].items()}
    layer_set = set(range(*metrics["layer_range"]))
    if set(source_layers.values()) != layer_set:
        raise ValueError("trace sources do not cover configured layers")
    peer = []
    if role == "worker":
        if metrics["session_id"] != front["session_id"]:
            raise ValueError("Worker session does not match Coordinator")
        with (folder / "coordinator.perf.csv").open(newline="", encoding="utf-8") as stream:
            peer = list(csv.DictReader(stream))
        if len(peer) != len(header["requests"]):
            raise ValueError("Worker and Coordinator request counts differ")
    rows = []
    position = 0
    phase_tokens = Counter()
    for index, req in enumerate(header["requests"]):
        if (req["request_id"] != index + 1 or req["position"] != position
                or not 0 < req["token_count"] <= metrics["batch"]):
            raise ValueError("invalid request position or token count")
        position += req["token_count"]
        phase = req["phase"]
        if peer:
            match = peer[index]
            if any(int(match[key]) != req[key] for key in ("request_id", "position", "token_count")):
                raise ValueError("Worker and Coordinator request shapes differ")
            phase = match["phase"]
        if (phase not in ("prefill", "decode") or req["begin"] != len(rows)
                or not req["begin"] <= req["end"] <= len(raw)):
            raise ValueError("invalid phase or request boundaries")
        if phase == "prefill" and phase_tokens["decode"]:
            raise ValueError("prefill follows decode")
        phase_tokens[phase] += req["token_count"]
        for key, size, decode, hit in raw[req["begin"]:req["end"]]:
            if role != "worker" and decode != (phase == "decode"):
                raise ValueError("phase marker disagrees with request")
            rows.append((key, size, phase, source_layers[key[0]], hit))
    if len(rows) != len(raw) or position != metrics["stage"]["input_tokens"]:
        raise ValueError("trace request coverage differs")
    if (phase_tokens["prefill"], phase_tokens["decode"]) != (front["prompt_tokens"],
                                                             front["generation"]["decode_steps"]):
        raise ValueError("trace token counts differ from generation metrics")
    lru = replay(rows, Policy("lru", header["slots"], 0, layer_set), observed=True)
    for result, field, metric in (("hits", "hits", "cache_hits"), ("misses", "misses", "cache_misses"),
                                  ("misses", "reads", "disk_reads"), ("evictions", "evictions", "cache_evictions"),
                                  ("bytes_read", "bytes_read", "bytes_read")):
        if lru[result] != header[field] or lru[result] != metrics["pager"][metric]:
            raise ValueError(f"observed and replay counters differ: {result}")
    layout = Layout(header)
    budget = metrics["pager_budget_mib"] * MIB
    if (lru["observed_mismatches"] or layout.slots(budget, "lru")[0] != header["slots"]
            or metrics["pager"]["slots"] != header["slots"]
            or metrics["pager"]["peak_allocated_bytes"] != header["managed_bytes"]):
        raise ValueError("observed LRU or capacity differs from replay")
    hashes = {}
    for suffix in ("generated_tokens", "generated.txt"):
        base = (folder / f"baseline.{suffix}").read_bytes()
        if base != (folder / f"coordinator.{suffix}").read_bytes():
            raise ValueError("generated outputs differ")
        if suffix == "generated_tokens" and len(base) != 4 * front["generation"]["generated_tokens"]:
            raise ValueError("generated token file length differs from metrics")
        hashes[suffix] = hashlib.sha256(base).hexdigest()
    return rows, layer_set, layout, {
        "role": role, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_identity": metrics["model_identity"], "identity_source": metrics["identity_source"],
        "prompt_token_sha256": front["prompt_token_sha256"], "generated_sha256": hashes,
        "conditions": {key: metrics[key] for key in ("context", "threads", "batch", "layer_range",
                                                    "cache_state_label", "binary_arch", "io", "split", "layers")},
        "max_new_tokens": front["max_new_tokens"], "host": metrics["host"],
        "observed_budget_mib": metrics["pager_budget_mib"], "observed_slots": header["slots"],
        "prompt_tokens": front["prompt_tokens"], "generated_tokens": front["generation"]["generated_tokens"],
        "observed_lru": lru,
    }


def compare(folder, role, budgets, variants):
    rows, layers, layout, provenance = load_run(folder, role)
    results = []
    for mib in sorted({*budgets, 768}):
        budget = mib * MIB
        for name, parameter in variants if mib in budgets else [("lru", 0)]:
            slots, reserved = layout.slots(budget, name)
            # 32 bytes/layer for head/tail/count/limit in a prospective 64-bit implementation.
            extra = aligned(32 * len(layers)) if name == "layer-lru" else 0
            while slots and layout.bytes(slots + reserved) + metadata(name, slots) + extra > budget:
                slots -= 1
            report = replay(rows, Policy(name, slots, parameter, layers))
            overhead = metadata(name, slots) + extra
            planned = layout.bytes(slots + reserved) + overhead
            if planned > budget or not slots or (name == "layer-lru" and slots < len(layers)):
                raise ValueError(f"{name}: {mib} MiB does not fit the required resident slots and management overhead")
            results.append({**report, "policy": name, "parameter_percent": parameter,
                            "budget_mib": mib, "slots": slots, "slot_stride_bytes": layout.stride,
                            "resident_payload_bytes": slots * layout.stride,
                            "bypass_reserved_payload_bytes": reserved * layout.stride,
                            "base_metadata_bytes": layout.bytes(slots + reserved) - (slots + reserved) * layout.stride,
                            "additional_metadata_bytes_estimate": overhead, "planned_bytes_estimate": planned})
    reference = {row["budget_mib"]: row for row in results if row["policy"] == "lru"}
    for row in results:
        row["reader_calls_saved_vs_same_budget_lru"] = reference[row["budget_mib"]]["misses"] - row["misses"]
        row["bytes_saved_vs_same_budget_lru"] = reference[row["budget_mib"]]["bytes_read"] - row["bytes_read"]
        hits = reference[768]["hits"]
        row["fraction_of_768_mib_lru_hits"] = row["hits"] / hits if hits else None
        row["bytes_saved_vs_768_mib_lru"] = reference[768]["bytes_read"] - row["bytes_read"]
    return {**provenance, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True, help="Prompt A run directory")
    parser.add_argument("--evaluation", action="append", nargs=2, default=[], metavar=("LABEL", "RUN_DIRECTORY"))
    parser.add_argument("--roles", nargs="+", choices=("baseline", "coordinator", "worker"),
                        default=("baseline", "coordinator", "worker"))
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--policies", nargs="+", choices=VARIANTS, default=tuple(VARIANTS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    labels = [label for label, _ in args.evaluation]
    if any(budget <= 0 for budget in args.budgets) or "A" in labels or len(labels) != len(set(labels)):
        parser.error("use positive budgets and unique evaluation labels other than A")
    variants = [(name, value) for name in dict.fromkeys(["lru", *args.policies]) for value in VARIANTS[name]]
    try:
        development = [compare(args.development, role, args.budgets, variants) for role in args.roles]
        stages = [report for report in development if report["role"] != "baseline"] or development
        locked = []
        for name in dict.fromkeys(name for name, _ in variants):
            choices = [param for policy, param in variants if name == policy]
            def cost(param):
                return sum(row["bytes_read"] for stage in stages for row in stage["results"]
                           if row["policy"] == name and row["parameter_percent"] == param
                           and row["budget_mib"] in args.budgets)
            locked.append((name, min(choices, key=lambda param: (cost(param), param))))
        # Freeze once, before opening any evaluation trace.
        expected = {stage["role"]: stage for stage in development}
        evaluation = {}
        prompt_hashes = {stage["prompt_token_sha256"] for stage in development}
        for label, folder in args.evaluation:
            reports = [compare(Path(folder), role, args.budgets, locked) for role in args.roles]
            for report in reports:
                reference = expected[report["role"]]
                if any(report[key] != reference[key] for key in ("model_identity", "identity_source", "conditions", "max_new_tokens")):
                    raise ValueError(f"{label}: evaluation run conditions differ")
            digest = reports[0]["prompt_token_sha256"]
            if digest in prompt_hashes:
                raise ValueError(f"{label}: evaluation must use a different prompt")
            prompt_hashes.add(digest)
            evaluation[label] = reports
        out = {"version": 1, "kind": "serial_fixed_slot_simulation", "locked_parameters_percent": dict(locked),
               "development": {"A": development}, "evaluation": evaluation}
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as error:
        parser.exit(1, f"policy simulation failed: {error}\n")


if __name__ == "__main__":
    main()
