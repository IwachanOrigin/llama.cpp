"""Independent stop-and-wait TCP benchmark using the Stage v1 frame layout."""
import argparse
import json
from pathlib import Path
import socket
import struct
import threading
import time

from importlib.util import spec_from_file_location, module_from_spec

HERE = Path(__file__).resolve().parent
SPEC = spec_from_file_location("stage_communication", HERE / "analyze-stage-communication.py")
ANALYSIS = module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
FRAME = struct.Struct("<6I")
MAGIC = 0x53454F4D
LIMIT = 16 << 20
REQUESTS = (0, 4, 64, 1024, 8192, 16384)
RESPONSES = (0, 4, 12, 16, 64, 1024, 8192, 607744, 1215488)


def exact(peer, size):
    output = bytearray(size)
    view = memoryview(output)
    cursor = 0
    while cursor < size:
        count = peer.recv_into(view[cursor:], min(65536, size - cursor))
        if not count:
            raise EOFError("peer closed mid-frame")
        cursor += count
    return output


def send(peer, kind, session, request, body):
    peer.sendall(FRAME.pack(MAGIC, 1, kind, len(body), session, request))
    view = memoryview(body)
    for offset in range(0, len(body), 65536):
        peer.sendall(view[offset:offset + 65536])


def receive(peer, kind, session, request, size):
    magic, version, actual, length, sid, rid = FRAME.unpack(exact(peer, FRAME.size))
    if (magic, version, actual, sid, rid) != (MAGIC, 1, kind, session, request) or length > LIMIT or length != size:
        raise ValueError("unexpected benchmark frame")
    return exact(peer, length)


def payload(size):
    return (bytes(range(256)) * ((size + 255) // 256))[:size]


def cases(samples, warmup, repeats):
    for repeat in range(repeats):
        for request in REQUESTS:
            for response in RESPONSES:
                yield repeat, request, response, samples + warmup


def serve(listener, samples, warmup, repeats, timeout):
    peer, _ = listener.accept()
    with peer:
        peer.settimeout(timeout)
        for session, (_, request_size, response_size, iterations) in enumerate(cases(samples, warmup, repeats), 1):
            request, response = payload(request_size), payload(response_size)
            for index in range(1, iterations + 1):
                if receive(peer, 3, session, index, request_size) != request:
                    raise ValueError("request data differs")
                send(peer, 4, session, index, response)


def client(port, samples, warmup, repeats, timeout):
    rows = []
    with socket.create_connection(("127.0.0.1", port), timeout) as peer:
        for session, (repetition, request_size, response_size, iterations) in enumerate(cases(samples, warmup, repeats), 1):
            request, response = payload(request_size), payload(response_size)
            during = []
            start_wall = start_cpu = None
            for index in range(1, iterations + 1):
                if index == warmup + 1:
                    start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
                start = time.perf_counter_ns()
                send(peer, 3, session, index, request)
                actual = receive(peer, 4, session, index, response_size)
                end = time.perf_counter_ns()
                if actual != response:
                    raise ValueError("response data differs")
                if index > warmup:
                    during.append((end - start) / 1e6)
            wall = (time.perf_counter_ns() - start_wall) / 1e9
            total = (request_size + response_size + 2 * FRAME.size) * samples
            rows.append({"trial": repetition + 1, "request_bytes": request_size, "response_bytes": response_size,
                         "samples": samples, "succeeded": samples, "failed": 0, "frame_bytes": total,
                         "wall_s": wall, "client_process_cpu_s": (time.process_time_ns()-start_cpu)/1e9,
                         "payload_bytes_per_s": (request_size + response_size) * samples / wall,
                         "rtt_ms": ANALYSIS.distribution(during)})
    return rows


def serialization(samples, warmup):
    rows = []
    for size in (8192, 607744, 1215488):
        floats = (1.25,) * (size // 4)
        codec = struct.Struct("<" + str(len(floats)) + "f")
        encoded_times, decoded_times = [], []
        for index in range(samples + warmup):
            start = time.perf_counter_ns()
            encoded = codec.pack(*floats)
            packed = time.perf_counter_ns()
            decoded = codec.unpack(encoded)
            end = time.perf_counter_ns()
            if decoded != floats:
                raise ValueError("serialization round trip differs")
            if index >= warmup:
                encoded_times.append((packed-start)/1e6)
                decoded_times.append((end-packed)/1e6)
        rows.append({"body_bytes": size, "encode_ms": ANALYSIS.distribution(encoded_times),
                     "decode_ms": ANALYSIS.distribution(decoded_times)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("loopback", "server", "client"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.samples, args.repeats, args.timeout) <= 0 or args.warmup < 0 or not 0 <= args.port <= 65535:
        parser.error("invalid benchmark dimensions")
    if args.mode != "server" and (args.output is None or (args.mode == "client" and args.port == 0)):
        parser.error("client requires a port and both measurement modes require --output")
    settings = (args.samples, args.warmup, args.repeats, args.timeout)
    if args.mode == "client":
        rows = client(args.port, *settings)
    else:
        with socket.socket() as listener:
            listener.settimeout(args.timeout)
            listener.bind(("127.0.0.1", args.port))
            listener.listen(1)
            port = listener.getsockname()[1]
            if args.mode == "server":
                print(f"READY {port}", flush=True)
                serve(listener, *settings)
                return
            errors = []
            def server():
                try:
                    serve(listener, *settings)
                except (EOFError, OSError, ValueError) as error:
                    errors.append(error)
            worker = threading.Thread(target=server)
            worker.start()
            try:
                rows = client(port, *settings)
            finally:
                worker.join()
            if errors:
                raise errors[0]
    result = {"mode": args.mode, "warmup_per_case": args.warmup, "case_results": rows,
              "serialization_python_struct": serialization(args.samples, args.warmup)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
