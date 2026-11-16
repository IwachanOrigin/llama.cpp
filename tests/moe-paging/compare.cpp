#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <string>
#include <vector>

struct Trace {
    FILE * out;
    size_t id_tensors = 0;
    bool external = false;
    bool invalid_weights = false;
    size_t external_ops = 0;
};

static bool trace_experts(ggml_tensor * tensor, bool ask, void * data) {
    const bool selected = std::strncmp(tensor->name, "ffn_moe_topk-", 13) == 0 && tensor->type == GGML_TYPE_I32;
    if (ask) {
        auto & trace = *static_cast<Trace *>(data);
        if (trace.external && tensor->op == GGML_OP_MUL_MAT_ID) {
            const auto * weights = tensor->src[0];
            if (!weights || weights->data || !weights->buffer ||
                    ggml_backend_buffer_get_size(weights->buffer) != 0 ||
                    std::strcmp(ggml_backend_buffer_name(weights->buffer), "CPU_ExternalExpert") != 0) {
                trace.invalid_weights = true;
            }
            ++trace.external_ops;
        }
        return selected;
    }
    if (selected) {
        auto & trace = *static_cast<Trace *>(data);
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
    params.n_threads = params.n_threads_batch = 4;
    params.cb_eval = trace_experts;
    params.cb_eval_user_data = &trace;
    return llama_init_from_model(model, params);
}

static int decode(llama_model * model, llama_context * context, Trace & trace, const char * output) {
    FILE * logits_out = std::fopen(output, "wb");
    if (!logits_out) {
        return 5;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt{1, 19, 7, 42, 51, 33, 91, 3};
    std::vector<int8_t> output_flags(prompt.size(), 1);
    llama_batch batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
    batch.logits = output_flags.data();
    auto collect = [&](int count) {
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
    if (llama_decode(context, batch) || !collect(static_cast<int>(prompt.size()))) {
        return 6;
    }
    constexpr int generation = 32;
    for (int i = 0; i < generation; ++i) {
        llama_token token = 1 + (i * 17 + 13) % (n_vocab - 1);
        if (llama_decode(context, llama_batch_get_one(&token, 1)) || !collect(1)) {
            return 7;
        }
    }
    if (std::fclose(trace.out) || std::fclose(logits_out) || trace.id_tensors != 2 * (generation + 1) ||
            (trace.external && (!trace.external_ops || trace.invalid_weights))) {
        return 8;
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        return 1;
    }
    Trace trace{std::fopen(argv[3], "wb")};
    if (!trace.out) {
        return 2;
    }

    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    trace.external = std::strncmp(argv[2], "external", 8) == 0;
    model_params.external_experts = trace.external;
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = false;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    if (std::strcmp(argv[2], "external-buffered") == 0) {
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    const bool invalid_device = std::strcmp(argv[2], "external-invalid-device") == 0;
    if (invalid_device) {
        model_params.n_gpu_layers = 1;
    }
    model_params.lazy_mode = std::strcmp(argv[2], "on") == 0 ? LLAMA_LAZY_MODE_ON :
        std::strcmp(argv[2], "auto") == 0 ? LLAMA_LAZY_MODE_AUTO : LLAMA_LAZY_MODE_OFF;
    auto * model = llama_model_load_from_file(argv[1], model_params);
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
    int result;
    if (std::strcmp(argv[2], "external-parallel") == 0) {
        const std::string peer_ids = std::string(argv[3]) + ".peer";
        const std::string peer_logits = std::string(argv[4]) + ".peer";
        Trace peer{std::fopen(peer_ids.c_str(), "wb")};
        peer.external = true;
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
