"""Exercise Reader benchmark validation, split GGUF and read equivalence."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def make_shard(path, layer):
    name = f"blk.{layer}.ffn_gate_exps.weight"
    raw = name.encode()
    header = b"GGUF" + struct.pack("<IQQQ", 3, 1, 0, len(raw)) + raw
    header += struct.pack("<IQQQIQ", 3, 256, 1, 2, 12, 0)
    offset = (len(header) + 31) & ~31
    path.write_bytes(header.ljust(offset, b"\0") + bytes((j + layer) % 256 for j in range(288)))
    return name, offset


def test(exe):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = [root / f"part-{i}.gguf" for i in range(2)]
        tensors = [make_shard(path, i) for i, path in enumerate(files)]
        manifest = {"version": 1, "shards": [{"path": str(path), "size": path.stat().st_size} for path in files],
                    "inputs": {"pattern": "test"}, "requests": []}
        for shard, expert in ((0, 0), (1, 1), (0, 0), (1, 0)):
            name, start = tensors[shard]
            manifest["requests"].append(dict(shard=shard, tensor=name, expert=expert,
                offset=start + 144 * expert, size=144, relative_ns=None, phase="synthetic"))
        path, output = root / "manifest.json", root / "result.json"

        def run(data, mode="shared", lanes=1, expected=True, workload="throughput"):
            expected_bytes = sum(req["size"] for req in data["requests"])
            path.write_text(json.dumps(data), encoding="utf-8")
            if output.exists():
                output.unlink()
            result = subprocess.run([exe, "reader-bench", str(path), mode, str(lanes), workload,
                "0", "0", "uncontrolled", "verify", str(output), "30000", "2"],
                capture_output=True, text=True, errors="replace", timeout=30)
            assert (result.returncode == 0) == expected, result.stderr
            if expected:
                data = json.loads(output.read_text())
                assert data["verified"] and data["completed_bytes"] == expected_bytes and data["detail_dropped"] == max(0, data["requests"] - 2)
                assert 0 < data["peak_active_requests"] <= lanes
            else:
                assert not output.exists()

        modes = ("shared", "independent", "mapping", "overlapped") if sys.platform == "win32" else (
            "shared", "independent", "mapping")
        for mode in modes:
            for lanes in (1, 2, 4):
                run(manifest, mode, lanes)
            for count in (1, 2):
                run(manifest | {"requests": manifest["requests"][:count]}, mode, workload="latency")
        for key, value in (("expert", 2), ("size", 0), ("size", 145), ("offset", -1),
                           ("offset", 2**64-1), ("shard", 2), ("tensor", "missing.weight")):
            corrupted = copy.deepcopy(manifest)
            corrupted["requests"][0][key] = value
            run(corrupted, expected=False)
        run(manifest, mode="unknown", expected=False)
        run(manifest, lanes=2, workload="latency", expected=False)
        missing = copy.deepcopy(manifest)
        missing["shards"][0]["path"] = str(root / "missing.gguf")
        run(missing, expected=False)
        missing["shards"][0]["path"] = str(path)
        run(missing, expected=False)
        files[1].write_bytes(files[1].read_bytes()[:-1])
        run(manifest, expected=False)
        too_short = copy.deepcopy(manifest)
        too_short["shards"][1]["size"] -= 1
        run(too_short, expected=False)
        broken = root / "broken.gguf"
        broken.write_bytes(b"invalid GGUF")
        bad = copy.deepcopy(manifest)
        bad["shards"][0] = dict(path=str(broken), size=broken.stat().st_size)
        run(bad, expected=False)
        # The old readers have all closed by this point; remove both shards.
        for file in files:
            file.unlink()
        analysis = runpy.run_path(str(HERE / "analyze-expert-reader-bench.py"))
        trial = root / "trial.json"
        result = {
            "version": 1, "status": "ok", "errors": 0, "verified": False,
            "manifest_sha256": "sample", "completed_bytes": 1048576,
            "requested_bytes": 1048576, "requests": 1, "wall_ns": 1000000000,
            "mib_per_second": 1, "mode": "shared", "readers": 1, "queue_depth": 1,
            "service_ns": {"p50": 1, "p95": 2, "p99": 3}, "latency_ns": {"p95": 3},
        }
        trial.write_text(json.dumps(result))
        parsed = analysis["summarize"]([trial, trial])
        assert parsed["results"][0]["throughput"]["median"] == 1
        other = root / "other.json"
        for invalid in (result | {"requests": 2}, result | {"completed_bytes": 0},
                        result | {"verified": True}, result | {"errors": 1}):
            other.write_text(json.dumps(invalid))
            try:
                analysis["summarize"]([trial, other])
            except ValueError:
                pass
            else:
                raise AssertionError("mixed or damaged trials accepted")
    print("Reader benchmark: split GGUF, bounded detail, all modes and invalid inputs passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe")
    test(parser.parse_args().exe)
