# Phase 2B-3 on two Windows hosts

Use PC-A (Intel x86-64) as Coordinator and PC-B (Surface Pro, Windows ARM64)
as Worker. Both Stage TCP sockets use IPv4 loopback. OpenSSH on PC-B
provides the authenticated, encrypted path. Set up the SSH account, host key
verification and network access on the hosts before the experiment. Model
hashes identify files and are independent of SSH peer authentication.

## Build and distribute the fixture

Use the same checkout revision and CPU-only CMake settings on both hosts.
On Surface use the x86-64 UCRT64 toolchain and run the result under Windows
x64 emulation. Keep the matching x86-64 UCRT64 DLLs on PATH
(e.g. libstdc++-6, libgcc_s_seh-1, libwinpthread-1 and libgomp-1 if enabled).
Keep any shared GGML/llama libraries alongside the executable. For a
portable first pass use a generic x86-64 build with GGML_NATIVE=OFF and
verify that the selected CPU instruction set is supported on both hosts.

~~~powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DGGML_NATIVE=OFF
cmake --build build --target qwen3moe-stage-tcp llama-gguf-split
python tests/moe-paging/distributed/prepare-stage-fixture.py --output C:\moe-stage-test\fixture
~~~

Generate once, then copy the complete fixture directory to the same path
on PC-B. Run the next command on both hosts, from the repository root.

~~~powershell
python tests/moe-paging/distributed/prepare-stage-fixture.py --verify C:\moe-stage-test\fixture
~~~

The manifest contains per-file size and SHA-256 plus one identity for each
layout. Wire identity is SHA-256 of the ordered concatenation of
little-endian u64 file size followed by that file's bytes, once per shard.
Each process computes it independently with IDENTITY set to auto. This
streams the file through a bounded buffer. Hashing time is logged and can
materially affect startup for large models. Keep files immutable during
startup and inference.

## Start Worker, then tunnel, then Coordinator

Run the following helper setup in each PowerShell 5.1 terminal that runs
a Stage process. The argument array uses paths without spaces.

~~~powershell
$stage = (Resolve-Path .\build\bin\qwen3moe-stage-tcp.exe).Path
$fixture = 'C:\moe-stage-test\fixture\model.gguf'
$io = 'mmap'
$runDir = 'C:\moe-stage-test\single-mmap'
New-Item -ItemType Directory -Force $runDir | Out-Null
$options = @('--ctx','128','--batch','8','--threads','2','--pager-budget-mib','8',
    '--connect-timeout-ms','15000','--accept-timeout-ms','90000',
    '--frame-timeout-ms','30000','--response-timeout-ms','300000')
~~~

On PC-B start Worker and read its READY line. Accept time starts after the
local model has loaded and the worker prints READY.

~~~powershell
$workerArgs = @('worker',$fixture,'1','2','50053',$io,"$runDir\worker",'auto') + $options
$worker = Start-Process -FilePath $stage -ArgumentList $workerArgs -NoNewWindow -PassThru -RedirectStandardOutput "$runDir\worker.ready" -RedirectStandardError "$runDir\worker.log"
Get-Content "$runDir\worker.ready"
~~~

On PC-A run the SSH command in a dedicated foreground terminal. Substitute
the Surface username and address, checking the host key against the known
Surface host key. Only the SSH service is reachable over the LAN.

~~~powershell
ssh.exe -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -L 127.0.0.1:50053:127.0.0.1:50053 user@surface-address
~~~

On PC-A run Coordinator. Run Baseline in a separate invocation before or
after the distributed run and wait for it to exit. Do not probe the Stage
port with a separate connection because Worker accepts one session.

~~~powershell
$coordinatorArgs = @('coordinator',$fixture,'1','2','50053',$io,"$runDir\coordinator",'auto') + $options
$coordinator = Start-Process -FilePath $stage -ArgumentList $coordinatorArgs -NoNewWindow -PassThru -RedirectStandardError "$runDir\coordinator.log"
if (!$coordinator.WaitForExit(360000)) { Stop-Process -Id $coordinator.Id; throw 'Coordinator process deadline' }
if ($coordinator.ExitCode -ne 0) { throw 'Coordinator failed. See coordinator.log' }

$baselineArgs = @('baseline',$fixture,'1','2','0',$io,"$runDir\baseline",'auto') + $options
$baseline = Start-Process -FilePath $stage -ArgumentList $baselineArgs -NoNewWindow -PassThru -RedirectStandardError "$runDir\baseline.log"
if (!$baseline.WaitForExit(360000)) { Stop-Process -Id $baseline.Id; throw 'Baseline process deadline' }
if ($baseline.ExitCode -ne 0) { throw 'Baseline failed. See baseline.log' }
~~~

On PC-B check that Worker has exited successfully. Copy worker.log and the
worker.ffn_moe_topk-* files to PC-A's matching run directory.

~~~powershell
if (!$worker.WaitForExit(360000)) { Stop-Process -Id $worker.Id; throw 'Worker process deadline' }
if ($worker.ExitCode -ne 0) { throw 'Worker failed. See worker.log' }
~~~

On PC-A compare both stages with the separate Baseline. The comparison
prints JSON with bitwise match, first differing element, maximum absolute
and relative errors, element tolerance failures and Expert ID agreement.
The tolerances below are the initial fixture acceptance thresholds.
A tolerance pass with changed FP32 bits is recorded separately.

~~~powershell
python tests/moe-paging/distributed/prepare-stage-fixture.py --compare "$runDir\baseline" "$runDir\coordinator" "$runDir\worker" --coordinator-log "$runDir\coordinator.log" --worker-log "$runDir\worker.log" --atol 1e-5 --rtol 1e-5
~~~

For the other three configurations, choose a fresh run directory and change
io to buffered or fixture to model-00001-of-00002.gguf on both hosts. Keep
both shard files together. Restart Worker for each session. The tool accepts
optional --tokens (comma separated), --prefill and --output-dir; the
comparison command above is for the standard five-token fixture.

Record hostname, executable SHA-256 (Get-FileHash), revision, CPU/OS,
identity and hash time, Context/batch, per-step compute and round-trip times,
socket totals, Working Set, Tensor inventory, Pager reads/pins and KV.
Socket totals exclude SSH and TCP/IP overhead. A round trip includes Worker
compute. Retain READY and separate stderr logs. On successful CLOSE/ACK
both Stage processes exit. Stop the tunnel with Ctrl+C.

## Failure checks

Use fresh run directories. Repeat with a missing Worker or absent tunnel,
then stop a live tunnel or Worker while a session is in flight. With this
short fixture, use the Worker fault options after-receive and after-decode
immediately after auto for deterministic cuts through a live tunnel.
Coordinator and Worker must exit nonzero, and a fresh failed coordinator
prefix must have no logits artifact. A mismatched copied fixture or a
different --ctx on one host must fail before OPEN/compute. To test the
response timeout, choose a very short --response-timeout-ms or use the
deterministic silent-peer CTest. The response deadline waits for the first
result byte, then the frame timeout covers the remainder. Timing out a socket
does not stop an already-running llama_decode. Use the outer process deadline
if the Worker does not terminate. Start a fresh process pair after a failure.

The compiled default deadlines remain 10 seconds for compatibility with the
prior loopback tests. The commands above explicitly choose the two-host values.

