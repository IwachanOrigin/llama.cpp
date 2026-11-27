#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 6 ]]; then
    printf 'usage: %s PC-A|PC-B EXE GGUF TRACE_DIR OUTPUT_DIR verify|screen\n' "$0" >&2
    exit 2
fi
host=$1 exe=$2 model=$3 trace=$4 destination=$5 task=$6
case "$host" in
    PC-A) role=coordinator ;;
    PC-B) role=worker ;;
    *) exit 2 ;;
esac
case "$task" in
    verify) limit=256; check=verify ;;
    screen) limit=0; check=measure ;;
    *) exit 2 ;;
esac
here=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$destination"
out="$destination/$host-$task"
mkdir -- "$out"
exec > >(tee -a "$out/run.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$out/exit-code.txt"' EXIT
printf 'Host=%s Role=%s Model=%s Trace=%s Cache=uncontrolled\n' "$host" "$role" "$model" "$trace"
exe_dir=$(cd -- "$(dirname -- "$exe")" && pwd)
(
    cd -- "$exe_dir"
    shopt -s nullglob
    files=("$(basename -- "$exe")" ./*.dll)
    sha256sum -- "${files[@]}"
) > "$out/binary-sha256.txt"
set -x
python "$here/prepare-expert-reader-bench.py" --gguf "$model" --run "$trace" --role "$role" --output "$out/manifest.json"
for item in shared:1 independent:1 independent:2 overlapped:1 overlapped:2 overlapped:4 mapping:1 mapping:2; do
    mode=${item%:*}
    count=${item#*:}
    "$exe" reader-bench "$out/manifest.json" "$mode" "$count" throughput "$limit" 0 uncontrolled "$check" "$out/$mode-$count.json" 30000
done
