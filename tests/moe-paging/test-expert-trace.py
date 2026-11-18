"""Small deterministic checks for the offline Expert trace analyzer."""
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest


analyzer = runpy.run_path(str(Path(__file__).with_name("analyze-expert-trace.py")))


def rows(keys):
    return [(key, 16, int(index >= 4), 0) for index, key in enumerate(keys)]


class ExpertTraceTest(unittest.TestCase):
    def test_distance_and_prefill_carry(self):
        keys = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0)]
        accesses = rows(keys)
        self.assertEqual(list(analyzer["distances"](accesses)), [None, None, None, None, 3])
        for slots, hits in ((3, 0), (4, 1)):
            outcome = analyzer["replay"](accesses, slots, {0: 0, 1: 1})
            self.assertEqual(outcome["decode"]["hits"], hits)
            self.assertEqual(outcome["total"]["bytes_read"], (5 - hits) * 16)

    def test_belady_improves_over_lru(self):
        accesses = rows([(0, n) for n in (0, 1, 2, 0, 1, 2)])
        lru = analyzer["replay"](accesses, 2, {0: 0})
        optimal = analyzer["replay"](accesses, 2, {0: 0}, future=analyzer["next_positions"](accesses))
        self.assertEqual(lru["total"]["hits"], 0)
        self.assertEqual(optimal["total"]["hits"], 2)

    def test_empty_and_truncation(self):
        header = dict(version=1, dropped=0, events=0, contexts=1, mode="lru",
                      slots=1, max_expert_bytes=16, sources={}, hits=0, misses=0,
                      evictions=0, reads=0, bytes_read=0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(header) + "\n", encoding="utf-8")
            metadata, accesses = analyzer["read_trace"](path)
            result = analyzer["analyze"](metadata, accesses, [1, 64])
            self.assertIsNone(result["first_lru_hit_slots"])
            self.assertEqual(result["reuse"]["histogram"], {})
            header["dropped"] = 1
            path.write_text(json.dumps(header) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyzer["read_trace"](path)


if __name__ == "__main__":
    unittest.main()
