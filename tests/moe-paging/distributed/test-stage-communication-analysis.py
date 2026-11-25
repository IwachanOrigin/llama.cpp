import json
from pathlib import Path
import runpy
import socket
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOLS = runpy.run_path(str(HERE / "analyze-stage-communication.py"))
BENCH = runpy.run_path(str(HERE / "benchmark-stage-transport.py"))


def fixture():
    front = []
    for index, (phase, position, count) in enumerate((("prefill", 0, 2), ("prefill", 2, 2), ("decode", 4, 1)), 1):
        front.append(dict(request_id=str(index), position=str(position), token_count=str(count), phase=phase,
                          send_bytes=str(40 + 12 * count), receive_bytes=str(36 + 16 * count),
                          request_roundtrip_ms="10", stage_execution_ms="3",
                          serialize_ms="1", deserialize_ms="2", send_wall_ms="5", receive_wall_ms="4"))
    worker = [dict(row, phase="", stage_execution_ms="6", send_bytes=row["receive_bytes"],
                   receive_bytes=row["send_bytes"]) for row in front]
    return front, worker


class Communication(unittest.TestCase):
    def test_accounting_and_boundaries(self):
        joined = TOOLS["join"](*fixture(), 2, 4, 4, 2)
        report = TOOLS["summarize"](joined)
        self.assertEqual(report["logits_rows"], 5)
        self.assertEqual(report["sampled_rows"], 2)
        self.assertEqual(report["unused_logits_rows"], 3)
        self.assertEqual(report["durations"]["unexplained_ms"]["median"], 1)
        options = TOOLS["alternatives"](joined, 116, 116, 4)
        self.assertEqual(options["last_or_empty"]["worker_saved_bytes"], 48)
        self.assertEqual(options["worker_greedy_or_empty"]["logits_or_token_bytes"], 8)
        self.assertEqual(options["full"]["coordinator_bytes"], options["last_or_empty"]["coordinator_bytes"])

    def test_rejects_inconsistent_runs(self):
        for side, index, key, value in ((1, 0, "request_id", "2"), (0, 1, "request_id", "1"),
                                       (1, 0, "position", "1"), (1, 0, "phase", "decode"),
                                       (0, 0, "send_bytes", "63"), (1, 2, "send_bytes", "51"),
                                       (0, 0, "serialize_ms", "nan")):
            with self.subTest(key=key, value=value):
                rows = fixture()
                rows[side][index][key] = value
                with self.assertRaises(ValueError):
                    TOOLS["join"](*rows, 2, 4, 4, 2)
        front, worker = fixture()
        with self.assertRaises(ValueError):
            TOOLS["join"](front, worker[:-1], 2, 4, 4, 2)
        with self.assertRaises(ValueError):
            TOOLS["join"](front[:-1], worker[:-1], 2, 4, 4, 2)

    def test_partial_data_and_close(self):
        tx, rx = socket.socketpair()
        with tx, rx:
            tx.sendall(b"12")
            tx.sendall(b"345")
            self.assertEqual(BENCH["exact"](rx, 5), b"12345")
            tx.shutdown(socket.SHUT_WR)
            with self.assertRaises(EOFError):
                BENCH["exact"](rx, 1)

    def test_large_payload_with_fragmented_reads(self):
        class FragmentedPeer:
            def __init__(self):
                self.chunks = []
                self.pending = bytearray()

            def sendall(self, data):
                self.chunks.append(len(data))
                self.pending.extend(data)

            def recv_into(self, output, size):
                count = min(size, len(self.pending), 4093)
                output[:count] = self.pending[:count]
                del self.pending[:count]
                return count

        peer = FragmentedPeer()
        payload = BENCH["payload"](607744)
        BENCH["send"](peer, 4, 1, 1, payload)
        self.assertEqual(peer.chunks[0], 24)
        self.assertLessEqual(max(peer.chunks[1:]), 65536)
        self.assertEqual(BENCH["receive"](peer, 4, 1, 1, len(payload)), payload)

    def test_incomplete_details_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            front, worker = fixture()
            import csv
            for role, rows in (("coordinator", front), ("worker", worker)):
                with (folder / (role + ".perf.csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(rows)
            details = {"hidden": 2, "vocab": 4, "prompt_tokens": 4, "generation": {"generated_tokens": 2},
                       "tcp": {"sent_bytes": sum(int(r["send_bytes"]) for r in front)+116,
                               "received_bytes": sum(int(r["receive_bytes"]) for r in front)+116},
                       "tcp_detail": {"schema_version": 1, "complete": False, "requests": []}}
            analyze = TOOLS["analyze"]
            settings = analyze.__globals__
            original = settings["COMPARE"]
            try:
                settings["COMPARE"] = lambda _: ([{}, details, {}], {})
                with self.assertRaisesRegex(ValueError, "incomplete TCP"):
                    analyze(folder, "test")
            finally:
                settings["COMPARE"] = original

    def test_unexpected_length_and_timeout(self):
        tx, rx = socket.socketpair()
        with tx, rx:
            BENCH["send"](tx, 4, 1, 1, b"abcd")
            with self.assertRaises(ValueError):
                BENCH["receive"](rx, 4, 1, 1, 5)
        tx, rx = socket.socketpair()
        with tx, rx:
            rx.settimeout(.05)
            with self.assertRaises(socket.timeout):
                BENCH["exact"](rx, 1)


if __name__ == "__main__":
    unittest.main()
