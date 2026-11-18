#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cinttypes>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct stage_bytes {
    uint64_t ordinary = 0;
    uint64_t experts = 0;
    uint64_t max_expert = 0;
};

int main(int argc, char ** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: stage-weights SPLIT TOTAL_LAYERS GGUF [GGUF ...]");
        const int split = std::stoi(argv[1]);
        const int layers = std::stoi(argv[2]);
        if (split <= 0 || split >= layers || layers > 10000) throw std::runtime_error("invalid layer range");
        std::array<stage_bytes, 2> stages{};
        std::set<std::string> seen;
        std::vector<bool> covered(layers);
        size_t embedding_bytes = 0;
        for (int i = 3; i < argc; ++i) {
            ggml_context * raw = nullptr;
            std::unique_ptr<gguf_context, decltype(&gguf_free)> file(
                gguf_init_from_file(argv[i], {true, &raw}), gguf_free);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(raw, ggml_free);
            if (!file || !metadata) throw std::runtime_error(std::string("could not read ") + argv[i]);
            for (int64_t index = 0; index < gguf_get_n_tensors(file.get()); ++index) {
                const std::string name = gguf_get_tensor_name(file.get(), index);
                if (!seen.insert(name).second) throw std::runtime_error("duplicate tensor " + name);
                int stage = 0;
                if (name.rfind("blk.", 0) == 0) {
                    size_t length = 0;
                    const auto layer = std::stoi(name.substr(4), &length);
                    if (layer < 0 || layer >= layers || name.at(4 + length) != '.') {
                        throw std::runtime_error("invalid layer for " + name);
                    }
                    covered[layer] = true;
                    stage = layer >= split;
                } else if (name.rfind("output.", 0) == 0 || name.rfind("output_norm.", 0) == 0) {
                    stage = 1;
                } else if (name.rfind("token_embd.", 0) != 0) {
                    throw std::runtime_error("unclassified tensor " + name);
                }
                auto & bytes = stages[stage];
                const auto size = gguf_get_tensor_size(file.get(), index);
                if (name == "token_embd.weight") embedding_bytes = size;
                const ggml_tensor * tensor = ggml_get_tensor(metadata.get(), name.c_str());
                if (name.find("_exps.weight") != std::string::npos) {
                    if (!tensor || tensor->ne[2] < 1 || tensor->ne[3] != 1 || size % tensor->ne[2]) {
                        throw std::runtime_error("invalid Expert layout for " + name);
                    }
                    bytes.experts += size;
                    const auto slice = size / tensor->ne[2];
                    if (slice > bytes.max_expert) bytes.max_expert = slice;
                } else {
                    bytes.ordinary += size;
                }
            }
        }
        for (bool present : covered) {
            if (!present) throw std::runtime_error("incomplete layer coverage");
        }
        if (!embedding_bytes) throw std::runtime_error("missing token embedding");
        if (!seen.count("output.weight")) {
            // Qwen3MoE falls back to the embedding when output.weight is absent.
            stages[1].ordinary += embedding_bytes;
            std::puts("stage 1 includes a local copy of tied token_embd.weight");
        }
        for (size_t i = 0; i < stages.size(); ++i) {
            const auto & bytes = stages[i];
            std::printf("stage %zu ordinary=%" PRIu64 " experts=%" PRIu64 " max_expert=%" PRIu64 "\n",
                        i, bytes.ordinary, bytes.experts, bytes.max_expert);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
