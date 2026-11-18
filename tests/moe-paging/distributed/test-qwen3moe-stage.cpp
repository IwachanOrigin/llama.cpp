#include "ggml-backend.h"
#include "ggml-impl.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct trace {
    std::string boundary_name;
    std::vector<float> boundary;
    std::map<std::string, std::vector<int32_t>> experts;

    static bool evaluate(ggml_tensor * tensor, bool ask, void * data) {
        auto & self = *static_cast<trace *>(data);
        const bool boundary = self.boundary_name == tensor->name;
        const bool expert = std::strncmp(tensor->name, "ffn_moe_topk-", 13) == 0;
        if (ask) return boundary || expert;
        if (boundary) {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
            self.boundary.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, self.boundary.data(), 0, ggml_nbytes(tensor));
        } else if (expert) {
            GGML_ASSERT(tensor->type == GGML_TYPE_I32);
            auto & ids = self.experts[tensor->name];
            for (int64_t row = 0; row < tensor->ne[1]; ++row) {
                for (int64_t col = 0; col < tensor->ne[0]; ++col) {
                    int32_t value;
                    ggml_backend_tensor_get(tensor, &value, row*tensor->nb[1] + col*tensor->nb[0], sizeof(value));
                    ids.push_back(value);
                }
            }
        }
        return true;
    }
};

static size_t inventory(const char * label, const llama_model & model, const llama_context & context,
                        int begin, int end) {
    size_t normal_count = 0, normal_bytes = 0, provider_count = 0, expert_bytes = 0;
    for (const auto & [name, tensor] : model.tensors_by_name) {
        if (name.rfind("blk.", 0) == 0) {
            const int layer = std::stoi(name.substr(4));
            require(layer >= begin && layer < end, "tensor from an unowned layer");
        } else if (end < llama_model_n_layer(&model)) {
            require(name.rfind("output", 0) != 0, "output tensor in prefix");
        }
        if (tensor->flags & GGML_TENSOR_FLAG_EXTERNAL_EXPERT) {
            require(tensor->extra && !tensor->data, "invalid Expert provider");
            ++provider_count;
            expert_bytes += ggml_nbytes(tensor);
        } else {
            require(tensor->data, "ordinary tensor was not loaded");
            ++normal_count;
            normal_bytes += ggml_nbytes(tensor);
        }
    }
    require(provider_count == size_t(3 * (end - begin)), "unexpected Expert source count");
    const auto sources = llama_pager_source_names(&model);
    require(sources.size() == provider_count, "Expert source map differs from inventory");
    size_t kv = 0, workspace = 0;
    for (const auto & entry : llama_get_memory_breakdown(&context)) {
        kv += entry.second.context;
        workspace += entry.second.compute;
    }
    const auto stats = llama_get_pager_metrics(&model);
    require(stats.peak_allocated_bytes > 0 && stats.disk_reads > 0 && stats.pinned_slots == 0,
            "Pager did not serve and release requests");
    std::printf("%s layers=%d ordinary_tensors=%zu ordinary_bytes=%zu expert_sources=%zu expert_bytes=%zu "
                "pager_bytes=%zu kv_bytes=%zu workspace_bytes=%zu reads=%llu\n",
                label, end - begin, normal_count, normal_bytes, provider_count, expert_bytes,
                stats.peak_allocated_bytes, kv, workspace, (unsigned long long) stats.disk_reads);
    return kv;
}

static void equal_floats(const float * left, const float * right, size_t count) {
    require(left && right, "missing outputs");
    require(std::all_of(left, left + count, [](float value) { return std::isfinite(value); }) &&
            std::all_of(right, right + count, [](float value) { return std::isfinite(value); }),
            "non-finite output");
    require(std::memcmp(left, right, count * sizeof(float)) == 0, "FP32 outputs differ");
}

static void check(const char * path, llama_load_mode mode) {
    auto mp = llama_model_default_params();
    ggml_backend_dev_t cpu_only[] = {nullptr};
    mp.devices = cpu_only;
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    mp.external_experts = true;
    mp.external_expert_cache_size = 8 * 1024 * 1024;
    mp.load_mode = mode;
    llama_model_ptr full(llama_model_load_from_file(path, mp));
    require(bool(full), "full model load failed");
    const int layers = llama_model_n_layer(full.get());
    const int split = layers / 2;
    require(split > 0, "fixture must have two or more layers");
    llama_model_ptr prefix(llama_model_load_stage(path, mp, 0, split));
    llama_model_ptr suffix(llama_model_load_stage(path, mp, split, layers));
    llama_model_ptr explicit_full(llama_model_load_stage(path, mp, 0, layers));
    require(prefix && suffix && explicit_full, "stage model load failed");
    require(!llama_model_ptr(llama_model_load_stage(path, mp, split, layers + 1)), "invalid range accepted");

    trace full_trace{"l_out-" + std::to_string(split - 1), {}, {}};
    trace prefix_trace{}, suffix_trace{};
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = trace::evaluate;
    cp.cb_eval_user_data = &full_trace;
    llama_context_ptr baseline(llama_init_from_model(full.get(), cp));
    cp.cb_eval_user_data = &prefix_trace;
    llama_context_ptr first(llama_init_from_model(prefix.get(), cp));
    cp.cb_eval_user_data = &suffix_trace;
    llama_context_ptr second(llama_init_from_model(suffix.get(), cp));
    cp.cb_eval = nullptr;
    llama_context_ptr compatible(llama_init_from_model(explicit_full.get(), cp));
    require(baseline && first && second && compatible, "context initialization failed");

    const size_t hidden = llama_model_n_embd(full.get());
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(full.get()));
    llama_token tokens[] = {1, 19, 7, 42, 51};
    llama_pos positions[] = {0, 1, 2, 3, 4};
    int8_t outputs[] = {1, 1, 1, 1, 1};
    require(llama_decode(second.get(), llama_batch_get_one(tokens, 1)) < 0,
            "suffix accepted token IDs");
    size_t boundary_values = 0;
    for (int start : {0, 3, 4}) {
        const int count = start == 0 ? 3 : 1;
        auto token_batch = llama_batch_get_one(tokens + start, count);
        token_batch.pos = positions + start;
        token_batch.logits = outputs + start;
        require(llama_decode(baseline.get(), token_batch) == 0, "baseline decode failed");
        require(llama_decode(compatible.get(), token_batch) == 0, "full-range decode failed");
        require(llama_decode(first.get(), token_batch) == 0, "prefix decode failed");
        const auto * boundary = llama_get_embeddings(first.get());
        require(full_trace.boundary.size() == hidden * count, "boundary has wrong shape");
        equal_floats(boundary, full_trace.boundary.data(), hidden * count);
        std::vector<float> handoff(boundary, boundary + hidden * count);
        llama_batch activation_batch = token_batch;
        activation_batch.token = nullptr;
        activation_batch.embd = handoff.data();
        require(llama_decode(second.get(), activation_batch) == 0, "suffix decode failed");
        for (int row = 0; row < count; ++row) {
            equal_floats(llama_get_logits_ith(baseline.get(), row), llama_get_logits_ith(second.get(), row), vocab);
            equal_floats(llama_get_logits_ith(baseline.get(), row), llama_get_logits_ith(compatible.get(), row), vocab);
        }
        boundary_values += hidden * count;
    }
    auto expert_ids = prefix_trace.experts;
    expert_ids.insert(suffix_trace.experts.begin(), suffix_trace.experts.end());
    require(prefix_trace.experts.size() == size_t(split) &&
            suffix_trace.experts.size() == size_t(layers - split) &&
            expert_ids == full_trace.experts, "Expert IDs differ");
    const auto full_kv = inventory("full", *full, *baseline, 0, layers);
    const auto first_kv = inventory("prefix", *prefix, *first, 0, split);
    const auto second_kv = inventory("suffix", *suffix, *second, split, layers);
    require(first_kv && second_kv && first_kv < full_kv && second_kv < full_kv &&
            first_kv + second_kv == full_kv, "KV ownership differs from stage ranges");
    std::printf("matched %zu boundary values and %zu logits across five positions\n", boundary_values, 5*vocab);
}

#ifdef _WIN32
static void working_set(const char * point) {
    PROCESS_MEMORY_COUNTERS counters{};
    require(GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)),
            "GetProcessMemoryInfo failed");
    std::printf("%s working_set_bytes=%zu lifetime_peak_working_set_bytes=%zu\n",
                point, size_t(counters.WorkingSetSize), size_t(counters.PeakWorkingSetSize));
}

static void check_working_set(const char * path, const std::string & role, int layers) {
    auto mp = llama_model_default_params();
    mp.external_experts = true;
    mp.external_expert_cache_size = 8 * 1024 * 1024;
    mp.use_extra_bufts = false;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    require(role == "full" || role == "prefix" || role == "suffix", "invalid working-set role");
    const int begin = role == "suffix" ? layers / 2 : 0;
    const int end = role == "prefix" ? layers / 2 : layers;
    std::printf("isolated %s\n", role.c_str());
    working_set("before_load");
    llama_model_ptr model(llama_model_load_stage(path, mp, begin, end));
    require(bool(model), "working-set model load failed");
    working_set("after_load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    require(bool(ctx), "working-set context creation failed");
    working_set("after_context");
    llama_token id = 1;
    auto batch = llama_batch_get_one(&id, 1);
    std::vector<float> input;
    if (begin) {
        input.assign(llama_model_n_embd(model.get()), 0.01f);
        batch.token = nullptr;
        batch.embd = input.data();
    }
    require(llama_decode(ctx.get(), batch) == 0, "working-set decode failed");
    llama_synchronize(ctx.get());
    working_set("after_decode");
    ctx.reset();
    model.reset();
    working_set("after_free");
}
#endif

int main(int argc, char ** argv) {
    try {
        require(argc > 1, "pass one or more GGUF paths (first shard for a split model)");
        ggml_backend_load_all();
#ifdef _WIN32
        if (argc == 5 && std::strcmp(argv[1], "--working-set") == 0) {
            check_working_set(argv[2], argv[3], std::stoi(argv[4]));
            return 0;
        }
#endif
        for (int i = 1; i < argc; ++i) {
            check(argv[i], LLAMA_LOAD_MODE_MMAP);
            check(argv[i], LLAMA_LOAD_MODE_NONE);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
