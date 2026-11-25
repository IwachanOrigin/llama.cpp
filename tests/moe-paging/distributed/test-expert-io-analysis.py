from pathlib import Path
import json
from io import StringIO
import runpy
import struct
import tempfile
import unittest

TOOLS = runpy.run_path(str(Path(__file__).with_name("analyze-expert-io.py")))


def fixture(path, name=b"blk.0.ffn_gate_exps.weight"):
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    tensor = struct.pack("<Q", len(name)) + name + struct.pack("<IQQQIQ", 3, 256, 1, 2, 12, 0)
    metadata = header + tensor
    path.write_bytes(metadata + bytes(-len(metadata) % 32 + 288))


class ExpertIO(unittest.TestCase):
    def test_ranges(self):
        check = TOOLS["checked_end"]
        self.assertEqual(check(12, 16, 28), 28)
        for args in ((-1, 1, 10), (0, 0, 10), (0, -1, 10), (8, 3, 10),
                     (2**64-1, 1, 2**64), (2**64-1, 2, 2**64-1)):
            with self.assertRaises(ValueError):
                check(*args)

    def test_shard_gap_and_cap(self):
        ranges = [(0, 10, 10), (0, 20, 10), (0, 25, 10), (0, 40, 10), (1, 10, 10)]
        merge = TOOLS["coalesce"]
        self.assertEqual(merge(ranges, 0, 50),
                         [(0, 10, 35, 3), (0, 40, 50, 1), (1, 10, 20, 1)])
        self.assertEqual(merge(ranges, 5, 40), [(0, 10, 50, 4), (1, 10, 20, 1)])
        self.assertEqual(merge(ranges, 5, 20),
                         [(0, 10, 30, 2), (0, 25, 35, 1), (0, 40, 50, 1), (1, 10, 20, 1)])
        self.assertEqual(merge([], 0, 20), [])

    def test_metadata_shards_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"part{i}.gguf" for i in range(2)]
            fixture(paths[0])
            fixture(paths[1], b"blk.1.ffn_down_exps.weight")
            tensors = TOOLS["inventory"](paths)
            self.assertEqual([entry["shard"] for entry in tensors.values()], [0, 1])
            self.assertTrue(all(entry["stride"] == 144 and entry["offset"] % 32 == 0 for entry in tensors.values()))
            paths[1].write_bytes(paths[1].read_bytes()[:-1])
            with self.assertRaises(ValueError):
                TOOLS["inventory"](paths)

    def test_grouping_uses_only_one_request_and_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "blk.0.ffn_gate_exps.weight"
            model = root / "model.gguf"
            fixture(model)
            trace = root / "baseline.experts.jsonl"
            requests = [
                {"begin": i, "end": i + 1, "request_id": i + 1, "position": i, "token_count": 1, "phase": "prefill"}
                for i in range(2)]
            header = {"sources": {"0": name}, "version": 1, "dropped": 0, "events": 2, "reads": 2,
                      "bytes_read": 288, "hits": 0, "misses": 2, "requests": requests, "slots": 1, "max_expert_bytes": 144,
                      "role": "baseline", "mode": "lru"}
            records = "\n[0,0,0,144,0,0,0]\n[1,0,1,144,0,0,0]\n"
            trace.write_text(json.dumps(header) + records)
            metrics = {"status": "ok", "role": "baseline", "pager_budget_mib": 1, "revision": "test",
                       "identity_source": "label", "model_identity": "test", "io": "mmap",
                       "pager": {"disk_reads": 2, "bytes_read": 288, "io_wait_ns": 1000, "cache_hits": 0, "cache_misses": 2}}
            (root / "baseline.metrics.json").write_text(json.dumps(metrics))
            tensors = TOOLS["inventory"]([model])
            analysis = TOOLS["analyze"](trace, tensors)
            self.assertEqual(analysis["adjacent_forward_reads"], 1)
            for candidate in analysis["simulation_same_request_tensor"]:
                self.assertEqual(candidate["reads_saved"], 0)
                self.assertEqual(candidate["amplification"], 1)
            requests[0]["end"] = 2
            header["requests"] = requests[:1]
            trace.write_text(json.dumps(header) + records)
            analysis = TOOLS["analyze"](trace, tensors)
            self.assertEqual(analysis["simulation_same_request_tensor"][0]["reads_saved"], 1)
            (root / "baseline.expert-io.jsonl").write_text(
                json.dumps({"version": 1, "complete": True, "dropped": 1, "access_dropped": 0}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                TOOLS["analyze"](trace, tensors)


    def test_prefetch_reader_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            delivery = root / "baseline.prefetch-delivery.jsonl"
            delivery.write_text("[0,5,true]\n")
            sources = {0: ("expert", {"shard": 0, "offset": 32, "count": 1, "stride": 144})}
            demand = [((0, 0), 144, 0, False)]
            row = [None, 0, 0, 0, 32, 144, 10, 1, 30, 1, 144, 10, True, 5]
            summary = {"complete": True, "dropped": 0, "access_dropped": 0, "reads": 1}
            metrics = {"pager": {"disk_reads": 1, "bytes_read": 144},
                       "prefetch": {"reads_started": 1, "consumed": 1, "demand_reads": 0}}
            def analyze(item):
                return TOOLS["analyze_prefetch_sidecar"](
                    StringIO(json.dumps(item) + "\n"), summary,
                    lambda suffix: root / ("baseline" + suffix), demand, sources, metrics)
            self.assertEqual(analyze(row)["consumed"], 1)
            for slot, value in ((4, 33), (13, 6), (12, False)):
                malformed = list(row)
                malformed[slot] = value
                with self.assertRaises(ValueError):
                    analyze(malformed)
            delivery.write_text("[0,5,true]\n[0,5,true]\n")
            with self.assertRaisesRegex(ValueError, "delivery identity"):
                analyze(row)


if __name__ == "__main__":
    unittest.main()
