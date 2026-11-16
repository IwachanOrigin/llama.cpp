import filecmp
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


def main(probe, split, profile="lazy"):
    with TemporaryDirectory(prefix="qwen3moe-lazy-") as tmp:
        root = Path(tmp)
        single = root / "model.gguf"
        prefix = root / "split"
        run(sys.executable, Path(__file__).with_name("make_fixture.py"), single)
        run(split, "--split-max-tensors", "13", single, prefix)
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
