"""Run isolated baseline and two-process TCP tests against the paging fixture."""
from contextlib import contextmanager
import hashlib
import csv
import json
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import re
import socket
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

MAGIC = 0x53454F4D
HEADER = struct.Struct("<6I")


def verify(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare_floats(reference, actual, count):
    left, right = reference.read_bytes(), actual.read_bytes()
    verify(len(left) == len(right) == 4 * count, f"unexpected output length: {actual}")
    if left != right:
        a, b = struct.unpack(f"<{count}f", left), struct.unpack(f"<{count}f", right)
        pairs = list(zip(a, b))
        first = next(i for i in range(count) if left[4*i:4*i+4] != right[4*i:4*i+4])
        raise RuntimeError(f"{actual}: first={first}, max_abs={max(abs(x-y) for x,y in pairs)}, "
                           f"max_rel={max(abs(x-y)/max(abs(x), 1e-30) for x,y in pairs)}")


def identity(model):
    shards = [model]
    if model.name.endswith("-00001-of-00002.gguf"):
        shards.append(model.with_name(model.name.replace("-00001-", "-00002-")))
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(shard.stat().st_size.to_bytes(8, "little"))
        with shard.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def command(exe, role, model, port, io, target, label, fault=None):
    args = [exe, role, str(model), "1", "2", str(port), io, str(target), label]
    return args if fault is None else args + [fault]


def run(args, log, expected=0):
    with log.open("wb") as out:
        result = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, timeout=25)
    text = log.read_text(errors="replace")
    verify(result.returncode == expected, f"{args[1]} exited {result.returncode}\n{text[-4000:]}")
    return text


@contextmanager
def start_worker(exe, model, io, prefix, label, fault=None, extra=()):
    logfile = prefix.with_suffix(".log")
    with logfile.open("wb") as log:
        child = subprocess.Popen(command(exe, "worker", model, 0, io, prefix, label, fault) + list(extra),
                                 stdout=subprocess.PIPE, stderr=log, text=True)
        result = queue.Queue()
        reader = Thread(target=lambda: result.put(child.stdout.readline()), daemon=True)
        reader.start()
        try:
            ready = result.get(timeout=15)
            verify(ready.startswith("READY "), f"Worker did not report READY (exit={child.poll()})")
            yield child, int(ready.split()[1]), logfile
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            reader.join(timeout=2)
            child.stdout.close()


def distributed(exe, model, io, root, label):
    baseline, front, back = [root / role for role in ("baseline", "front", "back")]
    run(command(exe, "baseline", model, 0, io, baseline, label), root / "baseline.log")
    with start_worker(exe, model, io, back, label) as (child, port, log):
        with (root / "front.log").open("wb") as out:
            coordinator = subprocess.Popen(command(exe, "coordinator", model, port, io, front, label),
                                           stdout=out, stderr=subprocess.STDOUT)
            verify(coordinator.pid != child.pid, "worker and coordinator must be distinct processes")
            try:
                code = coordinator.wait(timeout=25)
            finally:
                if coordinator.poll() is None:
                    coordinator.kill()
                    coordinator.wait(timeout=5)
        a = (root / "front.log").read_text(errors="replace")
        verify(code == 0, a[-4000:])
        verify(child.wait(timeout=15) == 0, log.read_text(errors="replace")[-4000:])
        b = log.read_text(errors="replace")
        print(f"{root.name}: coordinator PID={coordinator.pid}, worker PID={child.pid}")
    compare_floats(baseline.with_suffix(".boundary"), front.with_suffix(".boundary"), 5 * 512)
    compare_floats(baseline.with_suffix(".logits"), front.with_suffix(".logits"), 5 * 128)
    for layer, stage in enumerate((front, back)):
        name = f".ffn_moe_topk-{layer}"
        expected = Path(str(baseline) + name).read_bytes()
        verify(len(expected) == 5 * 2 * 4 and expected == Path(str(stage) + name).read_bytes(),
               f"Expert IDs differ at layer {layer}")
        verify(not Path(str(stage) + f".ffn_moe_topk-{1-layer}").exists(), "unexpected expert graph")
    counters = []
    for layer, text in enumerate((a, b)):
        verify(re.findall(r"compute position=(\d+) tokens=(\d+)", text) ==
               [("0", "3"), ("3", "1"), ("4", "1")], "invalid position sequence")
        owned = re.findall(r"^tensor blk\.(\d+)\.", text, re.MULTILINE)
        verify(owned and set(owned) == {str(layer)}, "unexpected tensor inventory")
        verify("experts=3 expert_bytes=53477376" in text and "kv=262144" in text and
               re.search(r"pinned=0 reads=[1-9]", text) and "\nCLOSED" in text, "invalid Stage metrics")
        totals = re.search(r"wire sent=(\d+) received=(\d+)", text)
        verify(totals is not None, "missing transport counters")
        counters.append((int(totals[1]), int(totals[2])))
        for line in text.splitlines():
            if re.match(r"(stage \[|wire |working_set=|roundtrip |compute position=)", line):
                print(f"  stage {layer} {line}")
    verify(counters[0] == counters[1][::-1], "send/receive counters disagree")
    print(f"  {sum(counters[0])/5:.1f} socket bytes per token including handshake and close")


def frame(kind, sid=0, rid=0, body=b"", version=1, length=None):
    return HEADER.pack(MAGIC, version, kind, len(body) if length is None else length, sid, rid) + body


def receive(peer, count):
    data = bytearray()
    while len(data) != count:
        new = peer.recv(count - len(data))
        verify(new, "peer closed while reading test response")
        data.extend(new)
    return bytes(data)


def read_frame(peer):
    magic, version, kind, size, sid, rid = HEADER.unpack(receive(peer, HEADER.size))
    verify(magic == MAGIC and version == 1 and size <= (16 << 20), "bad response framing")
    return kind, sid, rid, receive(peer, size)


def execute_payload(hidden=512, position=0):
    return struct.pack("<5I", 1, hidden, 0, 1, position) + struct.pack("<512f", *([0.01] * 512))


def reject_raw(exe, model, root, label, reason, action):
    with start_worker(exe, model, "mmap", root / action, label) as (child, port, log):
        with socket.create_connection(("127.0.0.1", port), timeout=5) as peer:
            if action == "idle":
                peer.settimeout(15)
                peer.sendall(frame(1)[:3])
            elif action == "version":
                peer.sendall(frame(1, version=42))
            elif action == "oversize":
                peer.sendall(frame(1, length=(16 << 20) + 1))
            else:
                candidate = ("0" * 64 if label != "0" * 64 else "1" * 64) if action == "model" else label
                meta = struct.pack("<11I", 1, 2, 1, 1, 2, 512, 128, 1, 256, 8, len(candidate)) + candidate.encode("ascii")
                peer.sendall(frame(1, body=meta))
                if action != "model":
                    verify(read_frame(peer) == (1, 0, 0, meta), "HELLO failed")
                    peer.sendall(frame(2))
                    kind, sid, rid, body = read_frame(peer)
                    verify(kind == 7 and sid != 0 and rid == 0 and body == b"", "session OPEN failed")
                    if action == "duplicate":
                        peer.sendall(frame(3, sid, 1, execute_payload()))
                        verify(read_frame(peer)[0] == 4, "first request failed")
                    payload = execute_payload(hidden=511 if action == "hidden" else 512,
                                              position=3 if action == "position" else 0)
                    if action == "partial":
                        peer.sendall(frame(3, sid, 1, payload)[:-32])
                        peer.shutdown(socket.SHUT_WR)
                    elif action == "unexpected":
                        peer.sendall(frame(4, sid, 1))
                    else:
                        wrong_sid = (sid ^ 1) if action == "session" else sid
                        peer.sendall(frame(3, wrong_sid, 1, payload))
            kind, _, _, detail = read_frame(peer)
            verify(kind == 6 and reason.encode() in detail, f"{action}: unexpected rejection {detail!r}")
        verify(child.wait(timeout=15) != 0, "malformed request succeeded")
        text = log.read_text(errors="replace")
        verify(text.count("compute position=") == (1 if action == "duplicate" else 0) and "FAILED:" in text,
               f"unexpected execution in {action}")
        print(f"rejected {action}")


def faults(exe, model, root, label):
    for reason, scenario in (
        ("unsupported protocol version", "version"), ("invalid frame length", "oversize"),
        ("metadata mismatch", "model"), ("invalid hidden size", "hidden"),
        ("unexpected token position", "position"), ("session or request", "duplicate"),
        ("session or request", "session"), ("unexpected message", "unexpected"),
        ("peer closed during frame", "partial"), ("TCP timeout", "idle"),
    ):
        reject_raw(exe, model, root, label, reason, scenario)
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
        # The port is reserved but has no listener.
        report = run(command(exe, "coordinator", model, port, "mmap", root / "refused", label),
                     root / "refused.log", expected=1)
        verify("TCP connect" in report, "connection refusal not reported")
        collision = run(command(exe, "worker", model, port, "mmap", root / "collision", label),
                        root / "collision.log", expected=1)
        verify("bind/listen failed" in collision, "bind collision not reported")
    print("connection refused and bind collision detected")
    for fault in ("after-receive", "after-decode"):
        with start_worker(exe, model, "mmap", root / fault, label, fault) as (child, port, logfile):
            output = root / f"aborted-{fault}"
            text = run(command(exe, "coordinator", model, port, "mmap", output, label),
                       output.with_suffix(".log"), expected=1)
            verify(child.wait(timeout=15) != 0 and "FAILED" in text and "compute position=0" in text,
                   "disconnected session did not fail")
            verify(not output.with_suffix(".logits").exists() and "CLOSED" not in text,
                   "failed session published output")
            worker_log = logfile.read_text(errors="replace")
            verify(("compute position=0" in worker_log) == (fault == "after-decode"),
                   "fault injected at wrong point")
            print(f"discarded session on disconnect {fault}")


def generation(exe, model, root, label):
    options = ["--generate", "--prompt-tokens", "1,19,7", "--max-new-tokens", "4",
               "--sampling", "greedy", "--ctx", "8", "--batch", "2"]
    worker_options = ["--ctx", "8", "--batch", "2"]
    baseline = root / "gen-baseline"
    front = root / "gen-front"
    expected = [(0, 2), (2, 1), (3, 1), (4, 1), (5, 1)]
    log_ref = run(command(exe, "baseline", model, 0, "mmap", baseline, label) + options,
                  root / "gen-baseline.log")
    with start_worker(exe, model, "mmap", root / "gen-worker", label, extra=worker_options) as (child, port, log):
        log_front = run(command(exe, "coordinator", model, port, "mmap", front, label) + options,
                        root / "gen-front.log")
        verify(child.wait(timeout=15) == 0, "generation Worker failed")
        for text in (log_ref, log_front, log.read_text(errors="replace")):
            history = [(int(position), int(count)) for position, count in
                       re.findall(r"compute position=([0-9]+) tokens=([0-9]+)", text)]
            verify(history == expected and "pinned=0" in text, f"incorrect autoregressive history: {history}")
    a, b = (prefix.with_suffix(".generated_tokens").read_bytes() for prefix in (baseline, front))
    verify(len(a) == 16 and a == b and "stop=limit" in log_front, "generation tokens or limit differ")
    verify(not front.with_suffix(".logits").exists(), "generation persisted full logits")
    verify(front.with_suffix(".generated.txt").read_bytes() == b"", "tokenizer-free fixture emitted text")

    profiled = root / "profile-baseline"
    profiled_front = root / "profile-front"
    profiled_worker = root / "profile-worker"
    run(command(exe, "baseline", model, 0, "mmap", profiled, label) + options + ["--perf-details"],
        root / "profile-baseline.log")
    with start_worker(exe, model, "mmap", profiled_worker, label,
                      extra=worker_options + ["--perf-details"]) as (child, port, _):
        run(command(exe, "coordinator", model, port, "mmap", profiled_front, label) +
            options + ["--perf-details"], root / "profile-front.log")
        verify(child.wait(timeout=15) == 0, "profile Worker failed")
    verify(profiled.with_suffix(".generated_tokens").read_bytes() == a and
           profiled_front.with_suffix(".generated_tokens").read_bytes() == b,
           "profiling changed generation")
    verify(not baseline.with_suffix(".metrics.json").exists(), "profiling was not opt-in")
    analyze = runpy.run_path(str(Path(__file__).with_name("analyze-stage-performance.py")))
    summaries, phases = analyze["compare"]([prefix.with_suffix(".metrics.json") for prefix in (
        profiled, profiled_front, profiled_worker)])
    verify(summaries[0]["generation"]["prefill_batches"] == 2 and
           summaries[0]["generation"]["decode_steps"] == 3 and
           summaries[0]["stage"]["stage_execution_calls"] == 5 and
           summaries[0]["stage"]["stage_execution_min_ms"] <= summaries[0]["stage"]["stage_execution_avg_ms"] <=
           summaries[0]["stage"]["stage_execution_max_ms"], "profile aggregate is inconsistent")
    verify(phases["decode"]["worker_reads"] > 0, "worker Pager counters were not recorded")
    with profiled.with_suffix(".perf.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    verify(len(rows) == 5 and all(row["receive_bytes"] == "" for row in rows) and
           sum(int(row["pager_read_count_delta"]) for row in rows) == summaries[0]["pager"]["disk_reads"],
           "baseline CSV does not reconcile with summary")

    zero = root / "profile-zero"
    run(command(exe, "baseline", model, 0, "mmap", zero, label) +
        options + ["--perf-details", "--max-new-tokens", "1"], root / "profile-zero.log")
    metrics = json.loads(zero.with_suffix(".metrics.json").read_text())
    verify(metrics["generation"]["decode_steps"] == 0 and metrics["generation"]["decode_step_wall_ms"] == 0 and
           metrics["generation"]["decode_tokens_per_sec"] is None and
           metrics["phases"]["decode"]["stage_execution_avg_ms"] is None, "zero-decode schema mismatch")

    context_options = options.copy()
    context_options[context_options.index("--ctx") + 1] = "3"
    context_log = run(command(exe, "baseline", model, 0, "mmap", root / "context", label) + context_options,
                      root / "context.log")
    verify("prompt_tokens=3 generated_tokens=1 stop=context" in context_log, "context stop was missed")
    prompt = root / "prompt.txt"
    prompt.write_text("sample", encoding="utf-8")
    rejected = run(command(exe, "baseline", model, 0, "mmap", root / "no-tokenizer", label) +
                   ["--generate", "--prompt-file", str(prompt), "--max-new-tokens", "1"],
                   root / "no-tokenizer.log", expected=1)
    verify("model has no tokenizer" in rejected, "missing tokenizer was accepted")
    chat_args = ["--generate", "--chat", "--prompt-file", str(prompt), "--max-new-tokens", "1"]
    rejected = run(command(exe, "baseline", model, 0, "mmap", root / "no-chat-template", label) +
                   chat_args, root / "no-chat-template.log", expected=1)
    verify("GGUF has no default chat template" in rejected, "missing chat template was accepted")
    rejected = run(command(exe, "baseline", model, 0, "mmap", root / "wrong-chat-mode", label) +
                   ["--system-file", str(prompt)], root / "wrong-chat-mode.log", expected=1)
    verify("chat file options require --chat" in rejected, "system input was silently ignored")
    rejected = run(command(exe, "baseline", model, 0, "mmap", root / "mixed-mode", label) +
                   options + ["--tokens", "1,2"], root / "mixed-mode.log", expected=1)
    verify("cannot be combined" in rejected, "mixed generation and fixed input were accepted")
    with start_worker(exe, model, "mmap", root / "gen-fault", label,
                      "after-decode", extra=worker_options) as (child, port, _):
        aborted = root / "gen-aborted"
        run(command(exe, "coordinator", model, port, "mmap", aborted, label) + options + ["--perf-details"],
            root / "gen-aborted.log", expected=1)
        verify(child.wait(timeout=15) != 0 and not aborted.with_suffix(".generated_tokens").exists()
               and not aborted.with_suffix(".generated.txt").exists(), "aborted generation published output")
        stats = json.loads(aborted.with_suffix(".metrics.json").read_text())
        verify(stats["status"] == "failed" and stats["tcp"]["transport_errors"] > 0 and
               stats["stage"]["stage_execution_calls"] == 1, "missing failure diagnostics")
    print("autoregressive IDs, chunked prefill, limits, validation and failure publication verified")

    for nonfinite in (None, float("nan"), float("inf")):
        with socket.socket() as listener, ThreadPoolExecutor(max_workers=1) as pool:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(8)

            def scripted_worker():
                conn, _ = listener.accept()
                with conn:
                    conn.settimeout(8)
                    hello = read_frame(conn)
                    conn.sendall(frame(1, body=hello[3]))
                    verify(read_frame(conn)[0] == 2, "missing OPEN")
                    conn.sendall(frame(7, sid=1))
                    verify(read_frame(conn)[:3] == (3, 1, 1), "missing prompt request")
                    logits = [0.0] * 128
                    logits[6] = logits[7] = 1.0
                    if nonfinite is not None:
                        logits[100] = nonfinite
                    conn.sendall(frame(4, 1, 1, struct.pack("<3I128f", 0, 1, 128, *logits)))
                    expected = 5 if nonfinite is None else 6
                    verify(read_frame(conn)[0] == expected, "unexpected generation protocol outcome")
                    if nonfinite is None:
                        conn.sendall(frame(7, 1, 2))

            pending = pool.submit(scripted_worker)
            output = root / ("tie" if nonfinite is None else f"reject-{nonfinite}")
            log = run(command(exe, "coordinator", model, listener.getsockname()[1], "mmap", output, label) +
                      ["--generate", "--prompt-tokens", "1", "--max-new-tokens", "1"],
                      output.with_suffix(".log"), expected=int(nonfinite is not None))
            pending.result(timeout=10)
            if nonfinite is None:
                verify(output.with_suffix(".generated_tokens").read_bytes() == struct.pack("<I", 6),
                       "tie did not select the lowest maximum ID")
            else:
                verify("non-finite" in log and not output.with_suffix(".generated_tokens").exists(),
                       "non-finite logits were published")
    print("greedy maximum, lowest-ID tie and non-finite logits verified")


def main(exe, single, split, logs=None):
    def test(root):
        for index, model in enumerate(map(Path, (single, split))):
            label = identity(model)
            for io in ("mmap", "buffered"):
                path = root / f"layout-{index}-{io}"
                path.mkdir()
                distributed(exe, model, io, path, label)
            if index == 0:
                path = root / "faults"
                path.mkdir()
                faults(exe, model, path, label)
                path = root / "generation"
                path.mkdir()
                generation(exe, model, path, label)
    if logs is not None:
        root = Path(logs)
        root.mkdir()
        test(root)
    else:
        with TemporaryDirectory(prefix="qwen3moe-tcp-") as temp:
            test(Path(temp))


if __name__ == "__main__":
    main(*sys.argv[1:])
