#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <array>
#include <cstdio>
#include <cstring>

bool run(ggml_backend_t backend, bool indexed, std::array<float, 2> & result) {
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    auto * weights = indexed ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 2)
                             : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    auto * input = indexed ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1)
                           : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    auto * ids = indexed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1) : nullptr;
    auto * output = indexed ? ggml_mul_mat_id(ctx, weights, input, ids)
                            : ggml_mul_mat(ctx, weights, input);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = false;
    if (buffer) {
        const float values[] = {1, 2, 3, 4, 5, 6, 7, 8};
        const float activation[] = {2, 3};
        const int32_t expert = 1;
        ggml_backend_tensor_set(weights, values, 0, ggml_nbytes(weights));
        ggml_backend_tensor_set(input, activation, 0, sizeof(activation));
        if (ids) ggml_backend_tensor_set(ids, &expert, 0, sizeof(expert));
        ok = true;
        for (int step = 0; step < 2 && ok; ++step) {
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            if (ok) ggml_backend_tensor_get(output, result.data(), 0, sizeof(result));
            const std::array<float, 2> expected = indexed ? std::array<float, 2>{28, 38}
                                                          : std::array<float, 2>{8, 18};
            ok = ok && result == expected;
        }
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return ok;
}

int main() {
    constexpr const char * endpoint = "127.0.0.1:50052";
    auto * cpu = ggml_backend_cpu_init();
    std::array<float, 2> local[2]{};
    for (int i = 0; i < 2; ++i) {
        if (!run(cpu, i != 0, local[i])) return 1;
    }
    ggml_backend_free(cpu);
    std::puts("Local CPU: both graphs passed twice");
    std::fflush(stdout);
    auto * remote = ggml_backend_rpc_init(endpoint, 0);
    if (!remote) {
        std::fprintf(stderr, "RPC initialization failed for %s\n", endpoint);
        return 2;
    }
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        std::array<float, 2> result{};
        const bool same = run(remote, i != 0, result) &&
            std::memcmp(result.data(), local[i].data(), sizeof(result)) == 0;
        std::printf("%s: [%.0f, %.0f], %s\n", i ? "MUL_MAT_ID" : "MUL_MAT",
                    result[0], result[1], same ? "bitwise match" : "failed");
        ok = ok && same;
    }
    ggml_backend_free(remote);
    return ok ? 0 : 1;
}
