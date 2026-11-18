# Phase 2A

- [Architecture and ownership](phase2a-architecture.md)
- [RPC code and experiment results](phase2a-rpc-findings.md)
- [Memory and network budget](phase2a-memory-network.md)
- [Phase 2B implementation plan](phase2b-implementation-plan.md)

Run these commands in PowerShell 5.1 with UCRT64 tools on PATH, from the repository root. The probe uses a separate build and links the existing shared libraries.

```powershell
cmake -S . -B build-rpc-2a -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build-rpc-2a --target ggml-rpc-server
g++ -std=c++17 '-Iggml/include' tests/moe-paging/distributed/rpc-cpu-probe.cpp '-Lbuild-rpc-2a/bin' -lggml-rpc -lggml-cpu -lggml-base -o build-rpc-2a/bin/rpc-cpu-probe.exe
g++ -std=c++17 '-Iggml/include' tests/moe-paging/distributed/stage-weights.cpp '-Lbuild-rpc-2a/bin' -lggml-base -o build-rpc-2a/bin/stage-weights.exe
```

In one terminal run `build-rpc-2a/bin/ggml-rpc-server.exe -H 127.0.0.1 -p 50052`. In another, set `$env:GGML_RPC_DEBUG='2'` and run `build-rpc-2a/bin/rpc-cpu-probe.exe`. Stop the server and repeat the probe to test connection refusal. Expect exit 0 with the server and 2 after shutdown. Keep the bind on loopback.

For inventory, run `build-rpc-2a/bin/stage-weights.exe 24 48 D:/models/Qwen3-30B-A3B-Q4_K_M.gguf`. For a split model pass every shard. This reads metadata and allocates Tensor descriptors only. The probe and inventory are compiled explicitly.
