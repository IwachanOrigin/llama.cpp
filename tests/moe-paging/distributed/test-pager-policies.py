"""Unit tests for online policy decisions and byte accounting."""

from pathlib import Path
import json
import runpy
import tempfile
import unittest

SIM = runpy.run_path(str(Path(__file__).with_name("simulate-pager-policies.py")))
Policy, replay, Layout, Sketch = (SIM[name] for name in ("Policy", "replay", "Layout", "Sketch"))


class Policies(unittest.TestCase):
    def decisions(self, name, slots, values, param=0):
        policy = Policy(name, slots, param, {0, 1})
        outcomes = []
        for value in values:
            outcomes.append(policy.access((0, value), value % 2))
            resident = len(policy.main) + len(policy.protected) + len(policy.window)
            resident += sum(len(entries) for _, entries in policy.partition.values())
            self.assertLessEqual(resident, slots)
        return outcomes

    def test_global(self):
        self.assertEqual(self.decisions("lru", 2, [0, 1, 0, 2, 1]),
                         [("admitted", 0), ("admitted", 0), ("hits", 0), ("admitted", 1), ("admitted", 1)])
        for name in SIM["VARIANTS"]:
            self.assertEqual(self.decisions(name, 2, []), [])
            self.assertEqual(self.decisions(name, 0, range(5)), [("bypassed", 0)] * 5)
            self.assertTrue(all(outcome != "hits" for outcome, _ in self.decisions(name, 2, range(8))))
        self.assertEqual(self.decisions("lru", 1, [0] * 4).count(("hits", 0)), 3)
        self.assertEqual(self.decisions("lru", 3, [0, 1, 2] * 4).count(("hits", 0)), 9)
        self.assertEqual(self.decisions("lru", 2, [0, 1, 2] * 4).count(("hits", 0)), 0)

    def test_lfu(self):
        self.assertEqual(self.decisions("lfu", 2, [0, 1, 2, 1, 3, 1])[-1], ("hits", 0))
        policy = Policy("lfu", 30, 0, {0})
        for _ in range(300):
            policy.access((0, 0), 0)
        self.assertEqual(policy.main[(0, 0)], 255)
        policy.access((0, 1), 0)
        self.assertEqual(policy.main[(0, 0)], 127)

    def test_slru(self):
        self.assertEqual(self.decisions("slru", 2, [0, 0, 1, 1, 2, 1, 0], 50)[-2:],
                         [("hits", 0), ("admitted", 1)])

    def test_admission(self):
        policy = Policy("tinylfu", 2, 0, {0})
        rows = [((0, key), 17, "prefill" if i < 2 else "decode", 0, 0)
                for i, key in enumerate((0, 1, 2, 2, 2))]
        result = replay(rows, policy)
        self.assertEqual((result["admitted"], result["bypassed"], result["hits"], result["bytes_read"]), (3, 1, 1, 68))
        self.assertEqual(result["phases"]["decode"]["bytes_read"], 34)
        self.assertEqual(result["layers"][0]["accesses"], 5)
        sketch = Sketch(4)
        for _ in range(40):
            sketch.observe((0, 0))
        self.assertEqual(sketch.frequency((0, 0)), 15)
        sketch.observe((1, 0))
        self.assertLess(sketch.frequency((0, 0)), 15)

    def test_window_and_prefix(self):
        values = [0, 1, 2, 0, 4, 8, 3, 5, 0, 1] * 5
        for name, params in SIM["VARIANTS"].items():
            for parameter in params:
                for slots in (1, 2, 7):
                    prefix = self.decisions(name, slots, values, parameter)
                    self.assertEqual(prefix, self.decisions(name, slots, values, parameter))
                    self.assertEqual(prefix, self.decisions(name, slots, values + [2, 6, 5], parameter)[:len(values)])
        policy = Policy("w-tinylfu", 6, 20, {0})
        for value in values:
            policy.access((0, value), 0)
            self.assertLessEqual(len(policy.window), policy.window_limit)
            self.assertLessEqual(len(policy.main), policy.slots - policy.window_limit)

    def test_fixed_budget(self):
        layout = Layout({"slots": 52, "max_expert_bytes": 1290240, "managed_bytes": 67106743})
        self.assertEqual(layout.slots(64 * SIM["MIB"], "lru"), (52, 0))
        near_limit = layout.bytes(16)
        self.assertEqual(layout.slots(near_limit, "lru"), (16, 0))
        self.assertEqual(layout.slots(near_limit, "tinylfu"), (14, 1))
        for name in SIM["VARIANTS"]:
            for mib in SIM["DEFAULT_BUDGETS"]:
                budget = mib * SIM["MIB"]
                slots, reserved = layout.slots(budget, name)
                self.assertLessEqual(layout.bytes(slots + reserved) + SIM["metadata"](name, slots), budget)

    def test_observed_hits_and_accounting(self):
        rows = [((0, 3), 80, "prefill", 0, 0), ((1, 3), 64, "prefill", 1, 0),
                ((0, 3), 80, "decode", 0, 1), ((0, 4), 80, "decode", 0, 0),
                ((0, 3), 80, "decode", 0, 1), ((1, 3), 64, "decode", 1, 0)]
        first = replay(rows, Policy("lru", 2, 0, {0, 1}), observed=True)
        second = replay(rows, Policy("lru", 2, 0, {0, 1}), observed=True)
        for item in (first, second):
            del item["elapsed_seconds"]
        self.assertEqual(first, second)
        self.assertEqual(first["observed_mismatches"], 0)
        self.assertEqual(first["hits"] + first["misses"], first["accesses"])
        for group in ("layers", "phases"):
            for metric in ("hits", "misses", "evictions", "bytes_read"):
                self.assertEqual(sum(part[metric] for part in first[group].values()), first[metric])

    def test_trace_validation(self):
        header = {"version": 1, "dropped": 0, "events": 0, "sources": {"0": "test"},
                  "max_expert_bytes": 24, "slots": 2}
        reader = SIM["TRACE"]["read_trace"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.jsonl"
            path.write_text(json.dumps(header) + "\n")
            self.assertEqual(reader(path)[1], [])
            for patch, event in (({"dropped": 1}, ""), ({"events": 1}, ""),
                                 ({"events": 1}, "[0,0,1,25,0,0,0]\n")):
                path.write_text(json.dumps({**header, **patch}) + "\n" + event)
                with self.assertRaises(ValueError):
                    reader(path)


if __name__ == "__main__":
    unittest.main()
