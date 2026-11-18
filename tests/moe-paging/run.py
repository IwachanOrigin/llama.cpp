import filecmp
import re
import statistics
from collections import Counter
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def run(*args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        sys.stderr.buffer.write(result.stdout)
        result.check_returncode()
    return result.stdout


def metrics(log, prefix):
    line = next((line for line in log.decode(errors="replace").splitlines() if prefix in line), "")
    fields = dict(re.findall(r"([a-z_]+)=([0-9.]+)", line))
    if not fields:
        sys.stderr.buffer.write(log)
        raise RuntimeError(f"missing {prefix} metrics")
    return {name: float(value) if "." in value else int(value) for name, value in fields.items()}


def budget(slots):
    # Metadata fits in the slack, which is smaller than one Expert slice.
    return slots * 557056 + 65536


def check_lru(probe, root, single, split):
    rejected = subprocess.run(
        (probe, single, "external-lru", root / "invalid.ids", root / "invalid.logits", "1", "1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if rejected.returncode != 3 or b"budget does not fit one slot" not in rejected.stdout:
        raise RuntimeError("undersized LRU budget was not rejected during model load")

    for threads in (1, 4):
        run(probe, single, "off", root / f"off-{threads}.ids", root / f"off-{threads}.logits", "0", str(threads))
        for layout, model in (("single", single), ("split", split)):
            for slots in (1, 2, 4):
                for suffix in ("", "-buffered"):
                    mode = "external-lru" + suffix
                    verify_lru(probe, root, model, mode, f"{layout}-{slots}{suffix}-{threads}", slots, threads)
    for layout, model in (("single", single), ("split", split)):
        verify_lru(probe, root, model, "external-lru-parallel", layout + "-parallel", 2, 4)
    verify_lru(probe, root, single, "external-lru-provider", "provider", 1, 4)
    verify_lru(probe, root, single, "external-lru-provider-two", "provider-two", 2, 4)
    verify_lru(probe, root, single, "external-lru-lease-pool", "lease-pool", 1, 4)
    verify_lru(probe, root, single, "external-lru-failure-buffered", "failure", 2, 4)


def verify_lru(probe, root, model, mode, name, slots, threads):
    ids, logits = root / (name + ".ids"), root / (name + ".logits")
    log = run(probe, model, mode, ids, logits, str(budget(slots)), str(threads))
    if b"CPU_ExternalExpert cannot allocate" in log or b"initial_read_bytes=0" not in log:
        raise RuntimeError(f"{name}: unexpected Expert allocation or initial I/O")
    setup = metrics(log, "external Expert pager:")
    stats = metrics(log, "external Expert stats:")
    if setup["slots"] != slots or setup["peak_allocated_bytes"] != stats["peak_allocated_bytes"]:
        raise RuntimeError(f"{name}: invalid slot count or changing allocation")
    if stats["peak_allocated_bytes"] > budget(slots) or stats["cache_evictions"] <= 0:
        raise RuntimeError(f"{name}: budget exceeded or no eviction: {stats}")
    if stats["disk_reads"] != stats["cache_misses"]:
        raise RuntimeError(f"{name}: unexpected reader activity: {stats}")
    if mode == "external-lru-provider" and (
        b"provider: shared pin" not in log or stats["cache_hits"] < 1 or stats["capacity_wait_ns"] == 0
    ):
        raise RuntimeError("model provider did not share a pin or wait for capacity")
    if mode == "external-lru-provider-two" and b"provider: distinct Experts pinned concurrently" not in log:
        raise RuntimeError("model provider did not hold distinct Experts concurrently")
    if mode == "external-lru-lease-pool" and (
        b"provider: 64 lease records exhausted and reused" not in log or stats["lease_wait_ns"] == 0
    ):
        raise RuntimeError("lease pool did not block and resume its 65th request")
    if "-failure-" in mode:
        if b"short read returned GGML_STATUS_FAILED" not in log or stats["bytes_read"] >= stats["disk_reads"] * 557056:
            raise RuntimeError("failed read was not propagated")
    for artifact, size in ((ids, 160 * 4), (logits, 5120 * 4)):
        reference = root / (f"off-{threads}" + artifact.suffix)
        if artifact.stat().st_size != size or not filecmp.cmp(artifact, reference, shallow=False):
            raise RuntimeError(f"{artifact.name}: incorrect size or contents")
        if "-parallel" in mode and not filecmp.cmp(Path(str(artifact) + ".peer"), reference, shallow=False):
            raise RuntimeError(f"{artifact.name}: peer differs")
    print(f"{name}: {stats}")


def benchmark(probe, root, model):
    trace = run(
        probe, model, "external-lru-trace", root / "trace.ids", root / "trace.logits", str(budget(64)), "4",
    )
    distances = Counter()
    recency = []
    cold = 0
    for line in trace.decode(errors="replace").splitlines():
        if not line.startswith("expert_access "):
            continue
        key = tuple(line.split()[1:])
        if key in recency:
            distance = recency.index(key)
            distances[distance] += 1
            recency.pop(distance)
        else:
            cold += 1
        recency.insert(0, key)
    requests = cold + sum(distances.values())
    if requests != metrics(trace, "external Expert stats:")["disk_reads"] + metrics(trace, "external Expert stats:")["cache_hits"]:
        raise RuntimeError("Expert access trace did not match reader and hit counters")
    first_hit = min(distances) + 1 if distances else None
    print(f"requests={requests} cold={cold} first_hit_slots={first_hit} reuse_distances={dict(sorted(distances.items()))}")

    capacities = (0, 1, 2, 4, 8, 16, 32, 64)
    extras = set() if first_hit is None else {first_hit, first_hit - 1} - set(capacities)
    results = {slots: [] for slots in (*capacities, *sorted(extras))}
    budgets = {0: 0}
    for slots in results:
        if slots:
            log = run(probe, model, "external-lru-bench", root / "unused", root / "unused", str(budget(slots)), "4")
            setup = metrics(log, "external Expert pager:")
            if setup["slots"] != slots or setup["peak_allocated_bytes"] > budget(slots):
                raise RuntimeError(f"unable to calibrate exact budget for {slots} slots")
            budgets[slots] = setup["peak_allocated_bytes"]
    for repeat in range(8):
        order = list(results) if repeat % 2 == 0 else list(reversed(results))
        for slots in order:
            mode = "external-lru-bench" if slots else "external-bench"
            log = run(probe, model, mode, root / "unused", root / "unused", str(budgets[slots]), "4")
            values = (
                metrics(log, "timing: load_ms=") | metrics(log, "timing: prefill_ms=")
                | metrics(log, "external Expert stats:")
            )
            expected_hits = sum(count for distance, count in distances.items() if distance < slots)
            if values["cache_hits"] != expected_hits or values["disk_reads"] != requests - expected_hits:
                raise RuntimeError(f"{slots} slots: measured hits disagree with the reuse trace")
            if slots and (
                metrics(log, "external Expert pager:")["slots"] != slots
                or values["peak_allocated_bytes"] != budgets[slots]
            ):
                raise RuntimeError(f"{slots} slots: measured allocation differs from calibrated budget")
            results[slots].append(values)
    for slots, rows in results.items():
        print(f"slots={slots} median: " + " ".join(
            f"{key}={statistics.median(row[key] for row in rows)}" for key in rows[0]
        ))


def main(probe, split, profile="lazy"):
    with TemporaryDirectory(prefix="qwen3moe-lazy-") as tmp:
        root = Path(tmp)
        single = root / "model.gguf"
        prefix = root / "split"
        run(sys.executable, Path(__file__).with_name("make_fixture.py"), single)
        run(split, "--split-max-tensors", "13", single, prefix)
        split_model = root / "split-00001-of-00002.gguf"
        if profile == "stage":
            sys.stdout.buffer.write(run(probe, single, split_model))
            return
        if profile == "stage-tcp":
            sys.stdout.buffer.write(run(sys.executable,
                Path(__file__).with_name("distributed") / "test-stage-tcp.py", probe, single, split_model))
            return
        if profile == "lru":
            return check_lru(probe, root, single, split_model)
        if profile == "bench":
            return benchmark(probe, root, single)
        modes = (("off", 0), ("on", 6), ("auto", 0))
        if profile == "external":
            modes = (("off", 0), ("external", 0), ("external-buffered", 0), ("external-parallel", 0))
            log = run(probe, single, "external-invalid-device", root / "reject.ids", root / "reject.logits")
            if b"external experts require n_gpu_layers = 0" not in log:
                raise RuntimeError("GPU placement was not rejected with the expected diagnostic")
        for layout, model in (("single", single), ("split", root / "split-00001-of-00002.gguf")):
            for mode, expected in modes:
                stem = root / f"{layout}-{mode}"
                ids, logits = stem.with_suffix(".ids"), stem.with_suffix(".logits")
                log = run(probe, model, mode, ids, logits)
                count = log.count(b"lazy read enabled")
                if count != expected:
                    raise RuntimeError(f"{stem.name}: expected {expected} lazy tensors, got {count}")
                if mode.startswith("external"):
                    if b"CPU_ExternalExpert cannot allocate" in log:
                        raise RuntimeError(f"{stem.name}: nonzero allocation attempted for external weights")
                    shards = 1 if layout == "single" else 2
                    for report in (
                        b"tensors=6 logical_bytes=106954752 buffer_bytes=0 slot_bytes=557056",
                        f"shards={shards} skipped_bytes=106954752 initial_read_bytes=0".encode(),
                    ):
                        if report not in log:
                            sys.stderr.buffer.write(log)
                            raise RuntimeError(f"{stem.name}: unexpected Expert allocation or initial I/O")
                        print(f"{stem.name}: {report.decode()}")
                for artifact, size in ((ids, 160 * 4), (logits, 5120 * 4)):
                    base = root / ("single-off" + artifact.suffix)
                    if artifact.stat().st_size != size or not filecmp.cmp(artifact, base, shallow=False):
                        raise RuntimeError(f"{artifact.name}: incorrect size or contents")
                    if mode == "external-parallel":
                        peer = Path(str(artifact) + ".peer")
                        if not filecmp.cmp(peer, base, shallow=False):
                            raise RuntimeError(f"{peer.name}: incorrect contents")
                print(f"{stem.name}: {count} lazy tensors, IDs and logits match")


if __name__ == "__main__":
    main(*sys.argv[1:])
