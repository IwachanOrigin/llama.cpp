"""Generate, verify and compare the two-host Qwen3MoE validation fixture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import shutil
import socket
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
NAMES = ("model.gguf", "model-00001-of-00002.gguf", "model-00002-of-00002.gguf")


def inventory(root):
    manifest = {"layers": 2, "hidden_size": 512, "vocab_size": 128,
                "generation": {"seed": 681, "generator": "make_fixture.py",
                               "max_tensors_per_shard": 13, "experts": 32, "top_k": 2},
                "files": {}, "identities": {}}
    for layout, names in (("single", NAMES[:1]), ("split", NAMES[1:])):
        combined = hashlib.sha256()
        for name in names:
            path = root / name
            size = path.stat().st_size
            combined.update(size.to_bytes(8, "little"))
            individual = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    individual.update(block)
                    combined.update(block)
            manifest["files"][name] = {"bytes": size, "sha256": individual.hexdigest()}
        manifest["identities"][layout] = combined.hexdigest()
    return manifest


def prepare(root, splitter):
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in (*NAMES, "manifest.json")):
        raise ValueError("use a fixture directory without existing fixture files")
    subprocess.run([sys.executable, HERE.parent / "make_fixture.py", root / NAMES[0]], check=True)
    subprocess.run([splitter, "--split-max-tensors", "13", root / NAMES[0], root / "model"], check=True)
    manifest = inventory(root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def compare_f32(left, right, atol, rtol, expected):
    a, b = left.read_bytes(), right.read_bytes()
    if len(a) != len(b) or len(a) != expected * 4:
        raise ValueError(f"invalid lengths: {left} / {right}")
    first, maximum_abs, maximum_rel, outliers = None, 0.0, 0.0, 0
    for index, ((x,), (y,)) in enumerate(zip(struct.iter_unpack("<f", a), struct.iter_unpack("<f", b))):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"non-finite value at {index}")
        if first is None and a[index*4:index*4+4] != b[index*4:index*4+4]:
            first = index
        difference = abs(x - y)
        maximum_abs = max(maximum_abs, difference)
        maximum_rel = max(maximum_rel, difference / max(abs(x), 1e-30))
        outliers += difference > atol + rtol * abs(x)
    return {"values": expected, "bitwise": first is None, "first_difference": first,
            "max_abs": maximum_abs, "max_rel": maximum_rel, "outside_tolerance": outliers}


def compare(prefixes, atol, rtol, coordinator_log, worker_log):
    baseline, coordinator, worker = prefixes
    artifact = lambda prefix, name: Path(str(prefix) + name)
    result = {kind: compare_f32(artifact(baseline, "." + kind), artifact(coordinator, "." + kind),
                                atol, rtol, 5 * width)
              for kind, width in (("logits", 128), ("boundary", 512))}
    result["experts"] = {}
    for layer, prefix in enumerate((coordinator, worker)):
        suffix = f".ffn_moe_topk-{layer}"
        a, b = (artifact(path, suffix).read_bytes() for path in (baseline, prefix))
        result["experts"][layer] = len(a) == 40 and a == b
    positions = [(0, 3), (3, 1), (4, 1)]
    result["stages"] = []
    for layer, path in enumerate((coordinator_log, worker_log)):
        log = path.read_text(errors="replace")
        steps = [(int(pos), int(count)) for pos, count in re.findall(r"compute position=(\d+) tokens=(\d+)", log)]
        owned = set(re.findall(r"^tensor blk\.(\d+)\.", log, re.MULTILINE))
        memory = re.search(r"^stage \[\d+,\d+\) (.*)$", log, re.MULTILINE)
        if not memory:
            raise ValueError(f"missing stage inventory in {path}")
        stats = {key: int(value) for key, value in re.findall(r"([a-z_]+)=(\d+)", memory[1])}
        if steps != positions or owned != {str(layer)} or stats["kv"] != 262144 or (
                stats["experts"] != 3 or not stats["reads"] or stats["pinned"] or "\nCLOSED" not in log):
            raise ValueError(f"unexpected stage ownership, history, or Pager state in {path}")
        result["stages"].append({"positions_and_counts": steps, "owned_layer": layer, **stats})
    print(json.dumps(result, indent=2))
    return all(result[kind]["outside_tolerance"] == 0 for kind in ("logits", "boundary")) and all(result["experts"].values())


def self_test(root, splitter, exe):
    fixture = root / "fixture"
    manifest = prepare(fixture, splitter)
    assert inventory(fixture) == manifest
    tools = runpy.run_path(str(HERE / "test-stage-tcp.py"))
    start_worker, command, run = (tools[name] for name in ("start_worker", "command", "run"))
    # Exercise every named option with finite, independently configured deadlines.
    options = ["--ctx", "128", "--batch", "8", "--threads", "2", "--pager-budget-mib", "8",
               "--frame-timeout-ms", "1200", "--response-timeout-ms", "2500", "--accept-timeout-ms", "5000",
               "--connect-timeout-ms", "800", "--tokens", "1,19,7,42,51", "--prefill", "3"]
    for layout, name in (("single", NAMES[0]), ("split", NAMES[1])):
        model = fixture / name
        tag = manifest["identities"][layout]
        out = root / layout
        out.mkdir()
        ref, front, back = (out / s for s in ("reference", "front", "back"))
        run(command(exe, "baseline", model, 0, "mmap", ref, "auto"), out / "reference.log")
        with start_worker(exe, model, "mmap", back, "auto", extra=options) as (child, port, worker_log):
            log = run(command(exe, "coordinator", model, port, "mmap", front.name, "auto") +
                      options + ["--output-dir", str(out)], out / "front.log")
            assert child.wait(timeout=10) == 0
            assert f"identity={tag} identity_source=local-sha256" in log
            assert f"identity={tag} identity_source=local-sha256" in worker_log.read_text(errors="replace")
            assert compare((ref, front, back), 0, 0, out / "front.log", worker_log)
        other = root / ("changed-" + layout)
        other.mkdir()
        names = NAMES[:1] if layout == "single" else NAMES[1:]
        for filename in names:
            shutil.copyfile(fixture / filename, other / filename)
        with (other / names[-1]).open("r+b") as changed:
            changed.seek(-1, 2)
            value = changed.read(1)
            changed.seek(-1, 2)
            changed.write(bytes([value[0] ^ 1]))
        with start_worker(exe, other / name, "mmap", out / "different", "auto", extra=options) as (child, port, log):
            text = run(command(exe, "coordinator", model, port, "mmap", out / "rejected", "auto") + options,
                       out / "rejected.log", expected=1)
            assert "metadata mismatch" in text and child.wait(timeout=10) == 1 and "compute position=" not in text
            assert "compute position=" not in log.read_text(errors="replace")
    model = fixture / NAMES[0]
    invalid = root / "invalid.gguf"
    invalid.write_bytes(b"not a GGUF")
    for args, expected in ((command(exe, "baseline", invalid, 0, "mmap", root / "invalid", "auto"),
                            "invalid GGUF signature"),
                           (command(exe, "baseline", model, 0, "mmap", root / "large", "-") +
                            ["--batch", "1000000"], "16 MiB frame limit"),
                           (command(exe, "baseline", model, 0, "mmap", root / "option", "-") +
                            ["--threads", "0"], "must be positive")):
        assert expected in run(args, root / "invalid.log", expected=1)
    with start_worker(exe, model, "mmap", root / "ctx", "auto", extra=["--ctx", "129"]) as (child, port, _):
        text = run(command(exe, "coordinator", model, port, "mmap", root / "ctx-front", "auto"),
                   root / "ctx-front.log", expected=1)
        assert "metadata mismatch" in text and child.wait(timeout=10) == 1
    with start_worker(exe, model, "mmap", root / "accept", "-", extra=["--accept-timeout-ms", "250"]) as (child, _, log):
        assert child.wait(timeout=4) == 1 and "TCP timeout" in log.read_text(errors="replace")
    with start_worker(exe, model, "mmap", root / "frame", "-", extra=["--frame-timeout-ms", "250"]) as (child, port, log):
        with socket.create_connection(("127.0.0.1", port), timeout=2) as peer:
            peer.sendall(b"M")
            assert tools["read_frame"](peer)[0] == 6
        assert child.wait(timeout=4) == 1 and "TCP timeout" in log.read_text(errors="replace")
    # Simulate an established Worker that has received EXECUTE but has not returned a byte.
    with socket.socket() as listen, ThreadPoolExecutor(max_workers=1) as pool:
        listen.bind(("127.0.0.1", 0))
        listen.listen(1)
        listen.settimeout(5)
        def silent_worker():
            peer, _ = listen.accept()
            with peer:
                peer.settimeout(4)
                kind, _, _, meta = tools["read_frame"](peer)
                assert kind == 1
                peer.sendall(tools["frame"](1, body=meta))
                assert tools["read_frame"](peer)[0] == 2
                peer.sendall(tools["frame"](7, sid=1))
                assert tools["read_frame"](peer)[0] == 3
                assert tools["read_frame"](peer)[0] == 6
        future = pool.submit(silent_worker)
        text = run(command(exe, "coordinator", model, listen.getsockname()[1], "mmap", root / "response", "-") +
                   ["--response-timeout-ms", "250", "--frame-timeout-ms", "3000"],
                   root / "response.log", expected=1)
        future.result(timeout=5)
        assert "awaiting Worker response: TCP timeout" in text
    print("verified auto identities, different content, malformed GGUF, settings and independent deadlines")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--compare", type=Path, nargs=3, metavar=("BASELINE", "COORDINATOR", "WORKER"))
    action.add_argument("--self-test")
    parser.add_argument("--splitter", default="build/bin/llama-gguf-split.exe" if sys.platform == "win32"
                        else "build/bin/llama-gguf-split")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--coordinator-log", type=Path)
    parser.add_argument("--worker-log", type=Path)
    args = parser.parse_args()
    if args.output:
        print(json.dumps(prepare(args.output, args.splitter), indent=2))
    elif args.verify:
        expected = json.loads((args.verify / "manifest.json").read_text(encoding="utf-8"))
        if inventory(args.verify) != expected:
            parser.exit(1, "fixture manifest mismatch\n")
        print(json.dumps(expected["identities"], indent=2))
    elif args.compare:
        if not args.coordinator_log or not args.worker_log or min(args.atol, args.rtol) < 0 or (
                not math.isfinite(args.atol) or not math.isfinite(args.rtol)):
            parser.error("comparison requires both stage logs and finite nonnegative tolerances")
        sys.exit(0 if compare(args.compare, args.atol, args.rtol, args.coordinator_log, args.worker_log) else 1)
    else:
        with TemporaryDirectory(prefix="qwen3moe-lan-options-") as directory:
            self_test(Path(directory), args.splitter, args.self_test)


if __name__ == "__main__":
    main()

