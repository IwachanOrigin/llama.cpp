#include "llama.h"
#include "ggml-backend.h"
#include "external-expert.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <set>
#include <string>
#include <vector>

struct AccessProbe {
    explicit AccessProbe(ggml_tensor * tensor) :
        tensor(tensor), source(static_cast<ggml_cpu_external_expert_source *>(tensor->extra)) {}

    static ggml_cpu_external_expert_lease acquire(void * context, int64_t expert, size_t bytes) {
        auto & probe = *static_cast<AccessProbe *>(context);
        std::fprintf(stderr, "expert_access %s %" PRId64 "\n", probe.tensor->name, expert);
        return probe.source->acquire(probe.source->context, expert, bytes);
    }

    static void release(void * context, ggml_cpu_external_expert_lease lease) {
        auto & probe = *static_cast<AccessProbe *>(context);
        probe.source->release(probe.source->context, lease);
    }

    ggml_tensor * tensor;
    ggml_cpu_external_expert_source * source;
    ggml_cpu_external_expert_source forwarding{this, acquire, release};
};

struct Trace {
    FILE * out;
    size_t id_tensors = 0;
    bool external = false;
    bool invalid_weights = false;
    size_t external_ops = 0;
    int provider_slots = 0;
    int threads = 4;
    bool trace_accesses = false;
    std::map<ggml_tensor *, AccessProbe> probes{};
};

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static void exercise_provider(const ggml_tensor * weights, int slots) {
    const auto & source = *static_cast<const ggml_cpu_external_expert_source *>(weights->extra);
    if (slots == 64) {
        std::array<ggml_cpu_external_expert_lease, 64> held{};
        std::set<void *> tokens;
        for (auto & lease : held) {
            lease = source.acquire(source.context, 0, weights->nb[2]);
            GGML_ASSERT(lease.data && lease.data == held.front().data && tokens.insert(lease.token).second);
        }
        std::promise<void> started;
        auto next = std::async(std::launch::async, [&] {
            started.set_value();
            return source.acquire(source.context, 0, weights->nb[2]);
        });
        GGML_ASSERT(started.get_future().wait_for(5s) == std::future_status::ready);
        GGML_ASSERT(next.wait_for(100ms) == std::future_status::timeout);
        source.release(source.context, held.back());
        GGML_ASSERT(next.wait_for(5s) == std::future_status::ready);
        const auto resumed = next.get();
        GGML_ASSERT(resumed.data == held.front().data && resumed.token == held.back().token &&
                resumed.generation != held.back().generation);
        source.release(source.context, resumed);
        for (size_t i = 0; i + 1 < held.size(); ++i) {
            source.release(source.context, held[i]);
        }
        std::puts("provider: 64 lease records exhausted and reused");
        return;
    }
    const auto first = source.acquire(source.context, 0, weights->nb[2]);
    GGML_ASSERT(first.data);
    auto peer = std::async(std::launch::async, [&] { return source.acquire(source.context, slots == 1 ? 0 : 1, weights->nb[2]); });
    GGML_ASSERT(peer.wait_for(5s) == std::future_status::ready);
    const auto same = peer.get();
    if (slots == 2) {
        GGML_ASSERT(same.data && same.data != first.data && same.token != first.token);
        GGML_ASSERT(std::memcmp(same.data, first.data, weights->nb[2]) != 0);
        source.release(source.context, same);
        source.release(source.context, first);
        std::puts("provider: distinct Experts pinned concurrently");
        return;
    }
    GGML_ASSERT(same.data == first.data && same.token != first.token);

    std::promise<void> started;
    auto next = std::async(std::launch::async, [&] {
        started.set_value();
        return source.acquire(source.context, 1, weights->nb[2]);
    });
    GGML_ASSERT(started.get_future().wait_for(5s) == std::future_status::ready);
    source.release(source.context, first);
    GGML_ASSERT(next.wait_for(50ms) == std::future_status::timeout);
    source.release(source.context, same);
    GGML_ASSERT(next.wait_for(5s) == std::future_status::ready);
    const auto replacement = next.get();
    GGML_ASSERT(replacement.data == same.data);
    source.release(source.context, replacement);
    const auto reread = source.acquire(source.context, 0, weights->nb[2]);
    GGML_ASSERT(reread.data);
    source.release(source.context, reread);
    std::puts("provider: shared pin, capacity wait and eviction verified");
}

static bool trace_experts(ggml_tensor * tensor, bool ask, void * data) {
    auto & trace = *static_cast<Trace *>(data);
    const bool selected = std::strncmp(tensor->name, "ffn_moe_topk-", 13) == 0 && tensor->type == GGML_TYPE_I32;
    if (ask) {
        if (trace.external && tensor->op == GGML_OP_MUL_MAT_ID) {
            auto * weights = tensor->src[0];
            if (!weights || weights->data || !weights->buffer ||
                    ggml_backend_buffer_get_size(weights->buffer) != 0 ||
                    std::strcmp(ggml_backend_buffer_name(weights->buffer), "CPU_ExternalExpert") != 0) {
                trace.invalid_weights = true;
            }
            ++trace.external_ops;
            if (trace.provider_slots && trace.external_ops == 1 && !trace.invalid_weights) {
                exercise_provider(weights, trace.provider_slots);
            }
            if (trace.trace_accesses && !trace.invalid_weights) {
                // The capacity sweep traces one context with model-owned Providers.
                auto [it, inserted] = trace.probes.try_emplace(weights, weights);
                if (inserted) {
                    weights->extra = &it->second.forwarding;
                }
            }
        }
        return selected;
    }
    if (selected) {
        std::vector<unsigned char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        for (int64_t token = 0; token < tensor->ne[1]; ++token) {
            for (int64_t slot = 0; slot < tensor->ne[0]; ++slot) {
                const auto * start = bytes.data() + token * tensor->nb[1] + slot * tensor->nb[0];
                std::fwrite(start, sizeof(int32_t), 1, trace.out);
            }
        }
        ++trace.id_tensors;
    }
    return true;
}

static llama_context * make_context(llama_model * model, Trace & trace) {
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 32;
    params.n_threads = params.n_threads_batch = trace.threads;
    params.cb_eval = trace.out ? trace_experts : nullptr;
    params.cb_eval_user_data = &trace;
    return llama_init_from_model(model, params);
}

static int decode(llama_model * model, llama_context * context, Trace & trace, const char * output) {
    FILE * logits_out = trace.out ? std::fopen(output, "wb") : nullptr;
    if (trace.out && !logits_out) {
        return 5;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt{1, 19, 7, 42, 51, 33, 91, 3};
    std::vector<int8_t> output_flags(prompt.size(), 1);
    llama_batch batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
    batch.logits = output_flags.data();
    auto collect = [&](int count) {
        if (!logits_out) {
            return true;
        }
        for (int row = 0; row < count; ++row) {
            const float * logits = llama_get_logits_ith(context, row);
            if (!logits) {
                return false;
            }
            for (int i = 0; i < n_vocab; ++i) {
                if (!std::isfinite(logits[i])) {
                    return false;
                }
            }
            if (std::fwrite(logits, sizeof(float), n_vocab, logits_out) != static_cast<size_t>(n_vocab)) {
                return false;
            }
        }
        return true;
    };
    auto start = Clock::now();
    if (llama_decode(context, batch) || !collect(static_cast<int>(prompt.size()))) {
        return 6;
    }
    const double prefill_ms = elapsed_ms(start);
    constexpr int generation = 32;
    start = Clock::now();
    for (int i = 0; i < generation; ++i) {
        llama_token token = 1 + (i * 17 + 13) % (n_vocab - 1);
        if (llama_decode(context, llama_batch_get_one(&token, 1)) || !collect(1)) {
            return 7;
        }
    }
    const double decode_ms = elapsed_ms(start);
    if (!trace.out) {
        std::printf("timing: prefill_ms=%.3f decode_ms=%.3f\n", prefill_ms, decode_ms);
    } else if (std::fclose(trace.out) || std::fclose(logits_out) || trace.id_tensors != 2 * (generation + 1) ||
            (trace.external && (!trace.external_ops || trace.invalid_weights))) {
        return 8;
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 7) {
        return 1;
    }
    const std::string mode = argv[2];
    const bool timing = mode.find("-bench") != std::string::npos;
    Trace trace{timing ? nullptr : std::fopen(argv[3], "wb")};
    if (!timing && !trace.out) {
        return 2;
    }
    trace.threads = argc == 7 ? std::atoi(argv[6]) : 4;
    trace.provider_slots = mode == "external-lru-provider" ? 1 :
        mode == "external-lru-provider-two" ? 2 : mode == "external-lru-lease-pool" ? 64 : 0;
    trace.trace_accesses = mode == "external-lru-trace";

    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    trace.external = std::strncmp(argv[2], "external", 8) == 0;
    model_params.external_experts = trace.external;
    model_params.external_expert_cache_size = argc == 7 ? std::strtoull(argv[5], nullptr, 10) : 0;
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = false;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    if (mode.find("-buffered") != std::string::npos) {
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    const bool invalid_device = std::strcmp(argv[2], "external-invalid-device") == 0;
    if (invalid_device) {
        model_params.n_gpu_layers = 1;
    }
    model_params.lazy_mode = std::strcmp(argv[2], "on") == 0 ? LLAMA_LAZY_MODE_ON :
        std::strcmp(argv[2], "auto") == 0 ? LLAMA_LAZY_MODE_AUTO : LLAMA_LAZY_MODE_OFF;
    const auto load_start = Clock::now();
    auto * model = llama_model_load_from_file(argv[1], model_params);
    if (timing) {
        std::printf("timing: load_ms=%.3f\n", elapsed_ms(load_start));
    }
    if (invalid_device && !model) {
        std::fclose(trace.out);
        llama_backend_free();
        return 0;
    }
    if (!model) {
        return 3;
    }

    auto * context = make_context(model, trace);
    if (!context || invalid_device) {
        return 4;
    }
    if (mode == "external-lru-failure-buffered") {
        const std::filesystem::path path = argv[1];
        const auto backup = path.string() + ".backup";
        std::filesystem::copy_file(path, backup);
        std::filesystem::resize_file(path, 0);
        llama_token token = 1;
        const int failure = llama_decode(context, llama_batch_get_one(&token, 1));
        {
            std::ifstream input(backup, std::ios::binary);
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file << input.rdbuf();
            file.close();
            GGML_ASSERT(file.good());
        }
        std::filesystem::remove(backup);
        GGML_ASSERT(failure == -3);
        llama_memory_clear(llama_get_memory(context), true);
        trace.id_tensors = trace.external_ops = 0;
        GGML_ASSERT(std::fseek(trace.out, 0, SEEK_SET) == 0);
        std::puts("provider: short read returned GGML_STATUS_FAILED; retry follows");
    }
    int result;
    if (mode.find("-parallel") != std::string::npos) {
        const std::string peer_ids = std::string(argv[3]) + ".peer";
        const std::string peer_logits = std::string(argv[4]) + ".peer";
        Trace peer{std::fopen(peer_ids.c_str(), "wb")};
        peer.external = true;
        peer.threads = trace.threads;
        auto * peer_context = make_context(model, peer);
        if (!peer.out || !peer_context) {
            return 4;
        }

        std::promise<void> ready;
        std::promise<void> start;
        auto go = start.get_future();
        auto peer_result = std::async(std::launch::async, [&] {
            ready.set_value();
            go.wait();
            return decode(model, peer_context, peer, peer_logits.c_str());
        });
        ready.get_future().wait();
        start.set_value();
        result = decode(model, context, trace, argv[4]);
        const int other_result = peer_result.get();
        llama_free(peer_context);
        if (result == 0) {
            result = other_result;
        }
    } else {
        result = decode(model, context, trace, argv[4]);
    }
    llama_free(context);
    for (auto & entry : trace.probes) {
        entry.first->extra = entry.second.source;
    }
    llama_model_free(model);
    if (result != 0) {
        return result;
    }
    if (trace.external) {
        const std::string path = argv[1];
        std::vector<std::string> paths{path};
        const auto shard = path.rfind("-00001-of-00002.gguf");
        if (shard != std::string::npos) {
            paths.push_back(path.substr(0, shard) + "-00002-of-00002.gguf");
        }
        for (const auto & file : paths) {
            const std::string moved = file + ".closed";
            if (std::rename(file.c_str(), moved.c_str()) || std::rename(moved.c_str(), file.c_str())) {
                return 9;
            }
        }
    }
    llama_backend_free();
}
