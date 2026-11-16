#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int width = 32;
constexpr int rows = 64;
constexpr int experts = 64;
constexpr int tokens = 4;
constexpr std::array<int32_t, tokens> first_ids{ 53, 7, 53, -1 };
constexpr std::array<int32_t, tokens> second_ids{ 5, 41, 41, 5 };
using result = std::array<float, rows * tokens>;

static void make_expert(int64_t id, void * destination) {
    std::array<float, width * rows> values;
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int((i * (id + 1) + id * 7) % 251) - 125) / 64.0f;
    }
    ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(values.data(), destination, values.size());
}

struct expert_source {
    expert_source(const char * path, size_t bytes) : file(std::fopen(path, "w+b")), slice(bytes) {
        GGML_ASSERT(file);
        for (int id = 0; id < experts; ++id) {
            make_expert(id, slice.data());
            GGML_ASSERT(std::fwrite(slice.data(), 1, slice.size(), file) == slice.size());
        }
        // Partial slice for the short-read test.
        GGML_ASSERT(std::fwrite(slice.data(), 1, slice.size() / 2, file) == slice.size() / 2);
        GGML_ASSERT(std::fflush(file) == 0);
    }

    FILE * file;
    std::vector<uint8_t> slice;
    std::array<int64_t, experts> requested{};
    size_t requests = 0;
    size_t releases = 0;
    int64_t fail_id = -1;
    bool held = false;

    static ggml_cpu_external_expert_lease acquire(void * context, int64_t id, size_t bytes) {
        auto & source = *static_cast<expert_source *>(context);
        GGML_ASSERT(!source.held && bytes == source.slice.size());
        GGML_ASSERT(source.requests < source.requested.size());
        source.requested[source.requests++] = id;
        source.held = true;
        const size_t offset = (id == source.fail_id ? experts : id) * bytes;
        if (std::fseek(source.file, static_cast<long>(offset), SEEK_SET) ||
                std::fread(source.slice.data(), 1, bytes, source.file) != bytes) {
            source.held = false;
            return {};
        }
        return { source.slice.data(), source.slice.data() };
    }

    static void release(void * context, void * token) {
        auto & source = *static_cast<expert_source *>(context);
        GGML_ASSERT(source.held && token == source.slice.data());
        source.held = false;
        ++source.releases;
    }

    void reset_counts() {
        GGML_ASSERT(!held);
        requests = releases = 0;
    }
};

struct graph_case {
    explicit graph_case(bool external) {
        context = ggml_init({ 10 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true });
        GGML_ASSERT(context);
        weights = ggml_new_tensor_3d(context, GGML_TYPE_Q8_0, width, rows, experts);
        input = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, 1, tokens);
        ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, 1, tokens);
        auto * mul = ggml_mul_mat_id(context, weights, input, ids);
        ggml_set_output(mul);
        output = ggml_scale(context, mul, 2);
        graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, output);
        if (external) {
            weights->flags |= GGML_TENSOR_FLAG_EXTERNAL_EXPERT;
            weights->buffer = ggml_cpu_external_expert_buffer_create();
            GGML_ASSERT(weights->buffer);
        }

        allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        GGML_ASSERT(ggml_gallocr_alloc_graph(allocator, graph));
        std::array<float, width * tokens> activation;
        for (size_t i = 0; i < activation.size(); ++i) {
            activation[i] = float(int(i % 19) - 9) / 16.0f;
        }
        ggml_backend_tensor_set(input, activation.data(), 0, sizeof(activation));
    }

    ~graph_case() {
        if (weights->flags & GGML_TENSOR_FLAG_EXTERNAL_EXPERT) {
            ggml_backend_buffer_free(weights->buffer);
        }
        ggml_gallocr_free(allocator);
        ggml_free(context);
    }

    void set_ids(const std::array<int32_t, tokens> & selected) {
        ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
    }

    result read_output() const {
        result value;
        ggml_backend_tensor_get(output, value.data(), 0, sizeof(value));
        return value;
    }

    ggml_context * context;
    ggml_tensor * weights;
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * output;
    ggml_cgraph * graph;
    ggml_gallocr_t allocator;
};

static bool abort_after_node(void * context) {
    ++*static_cast<int *>(context);
    return true;
}

static bool equal_bytes(const result & a, const result & b) {
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static void test(int threads, const char * path) {
    auto * backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend);
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto pool_params = ggml_threadpool_params_default(threads);
    auto * pool = ggml_threadpool_new(&pool_params);
    GGML_ASSERT(pool);
    ggml_backend_cpu_set_threadpool(backend, pool);

    result first, second;
    size_t resident_graph_bytes;
    size_t logical_bytes;
    {
        graph_case resident(false);
        resident_graph_bytes = ggml_gallocr_get_buffer_size(resident.allocator, 0);
        logical_bytes = ggml_nbytes(resident.weights);
        for (int id = 0; id < experts; ++id) {
            make_expert(id, static_cast<char *>(resident.weights->data) + id * resident.weights->nb[2]);
        }
        resident.set_ids(first_ids);
        GGML_ASSERT(ggml_backend_graph_compute(backend, resident.graph) == GGML_STATUS_SUCCESS);
        first = resident.read_output();
        resident.set_ids(second_ids);
        GGML_ASSERT(ggml_backend_graph_compute(backend, resident.graph) == GGML_STATUS_SUCCESS);
        second = resident.read_output();
        GGML_ASSERT(first != second);
    }

    graph_case paged(true);
    const size_t graph_bytes = ggml_gallocr_get_buffer_size(paged.allocator, 0);
    expert_source source(path, paged.weights->nb[2]);
    struct ggml_cpu_external_expert_source reader{ &source, expert_source::acquire, expert_source::release };
    paged.weights->extra = &reader;
    void * const output_data = paged.output->data;
    GGML_ASSERT(ggml_is_contiguous(paged.weights) && ggml_nbytes(paged.weights) == logical_bytes);
    GGML_ASSERT(paged.weights->ne[0] == width && paged.weights->ne[1] == rows && paged.weights->ne[2] == experts);
    GGML_ASSERT(source.slice.size() == ggml_row_size(GGML_TYPE_Q8_0, width) * rows);
    GGML_ASSERT(ggml_backend_buffer_get_size(paged.weights->buffer) == 0);
    GGML_ASSERT(resident_graph_bytes >= logical_bytes && graph_bytes + source.slice.capacity() < logical_bytes);

    paged.set_ids(first_ids);
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(equal_bytes(paged.read_output(), first));
    GGML_ASSERT(source.requests == 2 && source.releases == 2 && source.requested[0] == 7 && source.requested[1] == 53);

    source.reset_counts();
    GGML_ASSERT(ggml_gallocr_alloc_graph(paged.allocator, paged.graph));
    GGML_ASSERT(paged.weights->data == nullptr && paged.output->data == output_data);
    GGML_ASSERT(ggml_gallocr_get_buffer_size(paged.allocator, 0) == graph_bytes);
    paged.set_ids(second_ids);
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(equal_bytes(paged.read_output(), second));
    GGML_ASSERT(source.requests == 2 && source.releases == 2 && source.requested[0] == 5 && source.requested[1] == 41);

    source.reset_counts();
    source.fail_id = 41;
    int abort_calls = 0;
    ggml_backend_cpu_set_abort_callback(backend, abort_after_node, &abort_calls);
    result sentinel;
    sentinel.fill(123.0f);
    ggml_backend_tensor_set(paged.output, sentinel.data(), 0, sizeof(sentinel));
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_FAILED);
    GGML_ASSERT(!source.held && source.requests == 2 && source.releases == 1);
    GGML_ASSERT(abort_calls == 0 && paged.read_output() == sentinel);

    source.reset_counts();
    source.fail_id = -1;
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_ABORTED);
    GGML_ASSERT(!source.held && source.requests == 2 && source.releases == 2);
    GGML_ASSERT(abort_calls == 1 && paged.read_output() == sentinel);

    ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(equal_bytes(paged.read_output(), second) && paged.weights->data == nullptr);
    auto * device = ggml_backend_get_device(backend);
    const auto * matmul = paged.output->src[0];
    GGML_ASSERT(ggml_backend_dev_supports_op(device, matmul));
    GGML_ASSERT(!ggml_backend_buffer_is_host(paged.weights->buffer));

    source.reset_counts();
    auto * ordinary_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 0);
    auto * external_buffer = paged.weights->buffer;
    paged.weights->buffer = ordinary_buffer;
    GGML_ASSERT(!ggml_backend_dev_supports_op(device, matmul));
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_FAILED);
    paged.weights->buffer = external_buffer;
    ggml_backend_buffer_free(ordinary_buffer);

    paged.weights->flags &= ~GGML_TENSOR_FLAG_EXTERNAL_EXPERT;
    GGML_ASSERT(!ggml_backend_dev_supports_op(device, matmul));
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_FAILED);
    paged.weights->flags |= GGML_TENSOR_FLAG_EXTERNAL_EXPERT;

    const auto extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts"));
    if (extra_bufts) {
        auto * input_buffer = paged.input->buffer;
        for (auto * type = extra_bufts(device); *type; ++type) {
            auto * extra_buffer = ggml_backend_buft_alloc_buffer(*type, 0);
            paged.input->buffer = extra_buffer;
            GGML_ASSERT(!ggml_backend_dev_supports_op(device, matmul));
            GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_FAILED);
            paged.input->buffer = input_buffer;
            std::printf("rejected mixed buffer: %s\n", ggml_backend_buffer_name(extra_buffer));
            ggml_backend_buffer_free(extra_buffer);
        }
    }

    auto * copy = ggml_dup(paged.context, paged.weights);
    GGML_ASSERT(!ggml_backend_dev_supports_op(device, copy));
    ggml_build_forward_expand(paged.graph, copy);
    GGML_ASSERT(ggml_backend_graph_compute(backend, paged.graph) == GGML_STATUS_FAILED);
    GGML_ASSERT(source.requests == 0 && source.releases == 0);
    auto * view = ggml_view_tensor(paged.context, paged.weights);
    GGML_ASSERT(!ggml_backend_dev_supports_op(device, view));
    ggml_build_forward_expand(paged.graph, view);
    GGML_ASSERT(!ggml_gallocr_alloc_graph(paged.allocator, paged.graph));
    std::printf("threads=%d logical_weight=%zu resident_graph=%zu external_graph=%zu weight_slot=%zu bytes\n",
            threads, logical_bytes, resident_graph_bytes, graph_bytes, source.slice.capacity());

    ggml_backend_free(backend);
    ggml_threadpool_free(pool);
    GGML_ASSERT(std::fclose(source.file) == 0);
    GGML_ASSERT(std::remove(path) == 0);
}

} // namespace

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    test(1, argv[1]);
    test(4, argv[1]);
}
