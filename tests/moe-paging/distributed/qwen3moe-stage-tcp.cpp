#include "stage-tcp.h"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "llama-model.h"

extern "C" {
#include "sha256.h"
}

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using bytes = std::vector<uint8_t>;
using steady_clock = stage_tcp::clock;
constexpr uint32_t magic = 0x53454f4d;
constexpr uint32_t version = 1;
constexpr uint32_t cap = 16 * 1024 * 1024;
constexpr uint32_t wire_f32 = 1;
constexpr uint32_t wire_qwen3moe = 1;
enum message : uint32_t { hello = 1, open_session, execute_stage, stage_result, close_session, error_message, ack };
enum class state { created, ready, active, closed, failed };

static void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}

static void put(bytes & out, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) out.push_back(uint8_t(value >> shift));
}

struct reader {
    const bytes & data;
    size_t cursor = 0;
    uint32_t word() {
        require(data.size() - cursor >= 4, "incomplete payload");
        uint32_t value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8) value |= uint32_t(data[cursor++]) << shift;
        return value;
    }
    void done() const { require(cursor == data.size(), "unexpected payload length"); }
};

static void put_float(bytes & out, float value) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    put(out, bits);
}

static std::vector<float> floats(reader & r, uint32_t rows, uint32_t width) {
    const uint64_t count = uint64_t(rows) * width;
    require(count <= cap / 4 && count * 4 == r.data.size() - r.cursor, "invalid float payload length");
    std::vector<float> values(size_t(count), 0);
    for (auto & value : values) {
        const uint32_t bits = r.word();
        std::memcpy(&value, &bits, sizeof(value));
        require(std::isfinite(value), "non-finite activation or logits");
    }
    return values;
}

struct frame {
    uint32_t kind, session, request;
    bytes body;
};

struct channel {
    stage_tcp::socket peer;
    uint64_t sent = 0, received = 0;
    std::chrono::milliseconds frame_limit;
    explicit channel(stage_tcp::socket socket, uint32_t frame_ms)
        : peer(std::move(socket)), frame_limit(frame_ms) {}

    void send(frame packet) {
        require(packet.body.size() <= cap, "outgoing frame too large");
        bytes header;
        for (auto value : {magic, version, packet.kind, uint32_t(packet.body.size()), packet.session, packet.request}) {
            put(header, value);
        }
        const auto deadline = steady_clock::now() + frame_limit;
        peer.transfer(header.data(), header.size(), true, deadline, sent);
        peer.transfer(packet.body.data(), packet.body.size(), true, deadline, sent);
        std::fprintf(stderr, "tx type=%u request=%u header=24 payload=%zu\n",
                packet.kind, packet.request, packet.body.size());
    }

    frame receive(uint32_t response_ms = 0) {
        bytes header(24);
        const size_t starting_bytes = response_ms ? 1 : 0;
        if (response_ms) {
            try {
                peer.transfer(header.data(), 1, false, steady_clock::now() + std::chrono::milliseconds(response_ms), received);
            } catch (const std::runtime_error & error) {
                throw std::runtime_error(std::string("awaiting Worker response: ") + error.what());
            }
        }
        const auto deadline = steady_clock::now() + frame_limit;
        peer.transfer(header.data() + starting_bytes, header.size() - starting_bytes, false, deadline, received);
        reader r{header};
        require(r.word() == magic, "bad protocol magic");
        require(r.word() == version, "unsupported protocol version");
        const uint32_t type = r.word();
        const uint32_t length = r.word();
        require(type >= hello && type <= ack, "unexpected message type");
        require(length <= cap, "invalid frame length");
        frame packet{type, r.word(), r.word(), bytes(length)};
        peer.transfer(packet.body.data(), packet.body.size(), false, deadline, received);
        std::fprintf(stderr, "rx type=%u request=%u header=24 payload=%zu\n",
                packet.kind, packet.request, packet.body.size());
        if (type == error_message) {
            throw std::runtime_error("peer ERROR: " + std::string(packet.body.begin(), packet.body.end()));
        }
        return packet;
    }

    ~channel() {
        std::fprintf(stderr, "wire sent=%llu received=%llu\n",
                (unsigned long long) sent, (unsigned long long) received);
    }
};

static void expect(const frame & packet, message kind, uint32_t session, uint32_t request) {
    require(packet.kind == kind && packet.session == session && packet.request == request,
            "unexpected message, session or request ID");
}

struct options {
    std::string role, model, io, output, identity, fault;
    uint32_t port, split, layers;
    uint32_t ctx = 128, batch = 8, threads = 2, pager_mib = 8, prefill = 3;
    uint32_t connect_ms = 10000, accept_ms = 10000, frame_ms = 10000, response_ms = 10000;
    bool auto_identity = false;
    std::vector<llama_token> tokens{1, 19, 7, 42, 51};
};

static uint32_t number(const std::string & value, uint32_t max) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "expected integer");
    const auto n = std::stoull(value);
    require(n <= max, "integer out of range");
    return uint32_t(n);
}

static std::string model_identity(const std::string & filename) {
    namespace fs = std::filesystem;
    const auto first = fs::u8path(filename);
    const auto basename = first.filename().u8string();
    std::smatch split;
    uint32_t count = 1;
    if (std::regex_match(basename, split, std::regex("(.*)-([0-9]{5})-of-([0-9]{5})[.]gguf"))) {
        count = number(split[3], 99999);
        require(split[2] == "00001" && count >= 2, "identity requires the first GGUF shard");
    }
    sha256_t hash;
    sha256_init(&hash);
    bytes buffer(1024 * 1024);
    for (uint32_t shard = 1; shard <= count; ++shard) {
        auto path = first;
        if (count > 1) {
            std::ostringstream name;
            name << split[1] << '-' << std::setw(5) << std::setfill('0') << shard << "-of-" << split[3] << ".gguf";
            path = first.parent_path() / fs::u8path(name.str());
        }
        require(fs::is_regular_file(path), "missing or non-regular GGUF shard");
        uint64_t remaining = fs::file_size(path);
        uint8_t length[8];
        for (size_t i = 0; i < sizeof(length); ++i) length[i] = uint8_t(remaining >> (8*i));
        sha256_update(&hash, length, sizeof(length));
        std::ifstream file(path, std::ios::binary);
        char signature[4];
        require(bool(file.read(signature, sizeof(signature))) &&
                std::memcmp(signature, "GGUF", sizeof(signature)) == 0, "invalid GGUF signature");
        file.seekg(0);
        while (remaining) {
            const size_t chunk = std::min<uint64_t>(remaining, buffer.size());
            require(bool(file.read(reinterpret_cast<char *>(buffer.data()), chunk)), "incomplete GGUF during hashing");
            sha256_update(&hash, buffer.data(), chunk);
            remaining -= chunk;
        }
        require(file.peek() == EOF && !file.bad(), "GGUF changed size while hashing");
    }
    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    std::ostringstream label;
    for (auto value : digest) label << std::hex << std::setw(2) << std::setfill('0') << unsigned(value);
    return label.str();
}

static void parse_options(options & cfg, int argc, char ** argv) {
    for (int i = 8; i < argc; ++i) {
        const std::string key = argv[i];
        if (i == 8 && key.rfind("--", 0) != 0) {
            cfg.identity = key == "-" ? "" : key;
            continue;
        }
        if (i == 9 && key.rfind("--", 0) != 0) {
            cfg.fault = key;
            continue;
        }
        require(i + 1 < argc, "option requires a value");
        const std::string value = argv[++i];
        if (key == "--output-dir") {
            const auto path = std::filesystem::u8path(value);
            std::filesystem::create_directories(path);
            cfg.output = (path / std::filesystem::u8path(cfg.output).filename()).u8string();
            continue;
        }
        if (key == "--tokens") {
            cfg.tokens.clear();
            size_t begin = 0;
            for (;;) {
                const size_t comma = value.find(',', begin);
                cfg.tokens.push_back(number(value.substr(begin, comma - begin), INT32_MAX));
                if (comma == std::string::npos) break;
                begin = comma + 1;
            }
            continue;
        }
        const uint32_t n = number(value, 1000000);
        require(n != 0, "numeric options must be positive");
        if (key == "--ctx") cfg.ctx = n;
        else if (key == "--batch") cfg.batch = n;
        else if (key == "--threads") cfg.threads = n;
        else if (key == "--pager-budget-mib") cfg.pager_mib = n;
        else if (key == "--prefill") cfg.prefill = n;
        else if (key == "--connect-timeout-ms") cfg.connect_ms = n;
        else if (key == "--accept-timeout-ms") cfg.accept_ms = n;
        else if (key == "--frame-timeout-ms") cfg.frame_ms = n;
        else if (key == "--response-timeout-ms") cfg.response_ms = n;
        else throw std::runtime_error("unknown option " + key);
    }
}

struct artifacts {
    explicit artifacts(const std::string & path) : prefix(path) {}
    std::string prefix;
    bytes logits, boundary;
    std::map<std::string, bytes> expert_ids;
    std::string boundary_name;

    static bool eval(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<artifacts *>(user);
        const bool boundary = self.boundary_name == tensor->name;
        const bool expert = std::strncmp(tensor->name, "ffn_moe_topk-", 13) == 0;
        if (ask) return boundary || expert;
        if (boundary) {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
            std::vector<float> values(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
            for (float value : values) put_float(self.boundary, value);
        } else if (expert) {
            GGML_ASSERT(tensor->type == GGML_TYPE_I32);
            auto & out = self.expert_ids[tensor->name];
            for (int64_t row = 0; row < tensor->ne[1]; ++row) {
                for (int64_t col = 0; col < tensor->ne[0]; ++col) {
                    int32_t value;
                    ggml_backend_tensor_get(tensor, &value, row*tensor->nb[1] + col*tensor->nb[0], sizeof(value));
                    put(out, uint32_t(value));
                }
            }
        }
        return true;
    }

    void save() const {
        auto write = [&](const std::string & suffix, const bytes & data) {
            std::ofstream out(prefix + suffix, std::ios::binary);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            if (!data.empty()) out.write(reinterpret_cast<const char *>(data.data()), data.size());
            out.close();
        };
        write(".logits", logits);
        write(".boundary", boundary);
        for (const auto & entry : expert_ids) write("." + entry.first, entry.second);
    }
};

struct stage {
    artifacts output;
    llama_model_ptr model;
    llama_context_ptr context;
    uint32_t begin, end;
    uint32_t hidden, vocab;

    explicit stage(const options & cfg) : output(cfg.output) {
        const bool baseline = cfg.role == "baseline";
        begin = cfg.role == "worker" ? cfg.split : 0;
        end = cfg.role == "coordinator" ? cfg.split : cfg.layers;
        auto mp = llama_model_default_params();
        mp.load_mode = cfg.io == "mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        mp.external_experts = true;
        require(uint64_t(cfg.pager_mib) * 1048576 <= SIZE_MAX, "Pager budget exceeds address space");
        mp.external_expert_cache_size = size_t(cfg.pager_mib) * 1048576;
        mp.use_extra_bufts = false;
        model.reset(llama_model_load_stage(cfg.model.c_str(), mp, begin, end));
        require(bool(model) && uint32_t(llama_model_n_layer(model.get())) == cfg.layers, "stage model or layer count mismatch");
        hidden = llama_model_n_embd(model.get());
        vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(hidden && vocab && 16 + uint64_t(cfg.batch) * (4 + uint64_t(hidden)*4) <= cap &&
                12 + uint64_t(cfg.batch) * vocab*4 <= cap, "batch/model dimensions exceed 16 MiB frame limit");
        require(cfg.prefill <= cfg.batch && cfg.prefill <= cfg.tokens.size() && cfg.tokens.size() <= cfg.ctx &&
                std::all_of(cfg.tokens.begin(), cfg.tokens.end(), [&](llama_token id) { return uint32_t(id) < vocab; }),
                "invalid token sequence, prefill, batch or context");
        if (baseline) output.boundary_name = "l_out-" + std::to_string(cfg.split - 1);
        auto cp = llama_context_default_params();
        cp.n_ctx = cfg.ctx;
        cp.n_batch = cp.n_ubatch = cfg.batch;
        cp.n_threads = cp.n_threads_batch = cfg.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = artifacts::eval;
        cp.cb_eval_user_data = &output;
        context.reset(llama_init_from_model(model.get(), cp));
        require(bool(context), "cannot create stage context");
        std::fprintf(stderr, "execution layers=[%u,%u) requested_ctx=%u effective_ctx=%u batch=%u threads=%u pager_budget_mib=%u\n",
                begin, end, cfg.ctx, llama_n_ctx(context.get()), cfg.batch, cfg.threads, cfg.pager_mib);
    }

    bytes metadata(const options & cfg) const {
        bytes result;
        for (auto value : {wire_qwen3moe, cfg.layers, cfg.split, cfg.split, cfg.layers, hidden, vocab,
                           wire_f32, llama_n_ctx(context.get()), cfg.batch, uint32_t(cfg.identity.size())}) {
            put(result, value);
        }
        result.insert(result.end(), cfg.identity.begin(), cfg.identity.end());
        if (cfg.auto_identity || cfg.ctx != 128 || cfg.threads != 2 || cfg.pager_mib != 8) {
            for (uint32_t value : {uint32_t(cfg.auto_identity), cfg.ctx, cfg.threads, cfg.pager_mib}) put(result, value);
        }
        return result;
    }

    void compute(llama_batch batch) {
        const auto start = steady_clock::now();
        require(llama_decode(context.get(), batch) == 0, "stage decode failed");
        llama_synchronize(context.get());
        const auto ms = std::chrono::duration<double, std::milli>(steady_clock::now() - start).count();
        std::fprintf(stderr, "compute position=%d tokens=%d ms=%.3f\n", batch.pos[0], batch.n_tokens, ms);
    }

    void logits(bytes & out, uint32_t count) {
        for (uint32_t row = 0; row < count; ++row) {
            const auto * values = llama_get_logits_ith(context.get(), row);
            require(values, "missing logits");
            for (uint32_t col = 0; col < vocab; ++col) {
                require(std::isfinite(values[col]), "non-finite logits");
                put_float(out, values[col]);
            }
        }
    }

    void report() {
        size_t ordinary = 0, normal_bytes = 0, experts = 0, expert_bytes = 0, kv = 0, workspace = 0;
        for (const auto & [name, tensor] : model->tensors_by_name) {
            if (name.rfind("blk.", 0) == 0) {
                const uint32_t layer = std::stoul(name.substr(4));
                require(layer >= begin && layer < end, "unowned tensor");
            }
            std::fprintf(stderr, "tensor %s\n", name.c_str());
            if (tensor->flags & GGML_TENSOR_FLAG_EXTERNAL_EXPERT) {
                ++experts;
                expert_bytes += ggml_nbytes(tensor);
                require(tensor->extra && !tensor->data, "invalid external tensor");
            } else {
                require(tensor->data, "ordinary tensor has no storage");
                ++ordinary;
                normal_bytes += ggml_nbytes(tensor);
            }
        }
        for (const auto & item : llama_get_memory_breakdown(context.get())) {
            kv += item.second.context;
            workspace += item.second.compute;
        }
        const auto stats = llama_get_pager_metrics(model.get());
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS memory{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) {
            std::fprintf(stderr, "working_set=%zu peak_working_set=%zu\n",
                    size_t(memory.WorkingSetSize), size_t(memory.PeakWorkingSetSize));
        }
#endif
        std::fprintf(stderr, "stage [%u,%u) ordinary=%zu normal_bytes=%zu experts=%zu expert_bytes=%zu "
                "pager=%zu kv=%zu workspace=%zu pinned=%zu reads=%llu hits=%llu read_bytes=%llu\n",
                begin, end, ordinary, normal_bytes, experts, expert_bytes, stats.peak_allocated_bytes,
                kv, workspace, stats.pinned_slots, (unsigned long long) stats.disk_reads,
                (unsigned long long) stats.cache_hits, (unsigned long long) stats.bytes_read);
    }
};

static void acknowledge(channel & wire, message type, uint32_t session, uint32_t request) {
    const auto packet = wire.receive();
    expect(packet, type, session, request);
    require(packet.body.empty(), "unexpected acknowledgement payload");
}

static void worker(const options & cfg, stage & local, channel & wire, state & status, uint32_t & session, uint32_t & request) {
    auto announcement = wire.receive();
    expect(announcement, hello, 0, 0);
    require(announcement.body == local.metadata(cfg), "model or stage metadata mismatch");
    wire.send({hello, 0, 0, announcement.body});
    acknowledge(wire, open_session, 0, 0);
    session = std::random_device{}();
    if (!session) session = 1;
    status = state::active;
    wire.send({ack, session, 0, {}});
    uint32_t position = 0;
    for (request = 1; ; ++request) {
        const auto packet = wire.receive();
        if (packet.kind == close_session) {
            expect(packet, close_session, session, request);
            require(packet.body.empty(), "invalid CLOSE");
            wire.send({ack, session, request, {}});
            status = state::closed;
            return;
        }
        expect(packet, execute_stage, session, request);
        require(status == state::active, "session is not active");
        reader r{packet.body};
        const uint32_t count = r.word();
        require(count && count <= cfg.batch && count <= llama_n_ctx(local.context.get()) - position,
                "invalid token count");
        require(r.word() == local.hidden, "invalid hidden size");
        require(r.word() == 0 && r.word() == wire_f32, "invalid sequence or activation type");
        std::vector<llama_pos> positions(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto value = r.word();
            require(value == position + i, "unexpected token position");
            positions[i] = value;
        }
        auto activation = floats(r, count, local.hidden);
        if (cfg.fault == "after-receive") throw std::runtime_error("injected disconnect after receive");
        std::vector<int8_t> outputs(count, 1);
        llama_batch batch{};
        batch.n_tokens = count;
        batch.embd = activation.data();
        batch.pos = positions.data();
        batch.logits = outputs.data();
        local.compute(batch);
        if (cfg.fault == "after-decode") throw std::runtime_error("injected disconnect after decode");
        bytes result;
        put(result, 0);
        put(result, count);
        put(result, local.vocab);
        local.logits(result, count);
        wire.send({stage_result, session, request, std::move(result)});
        position += count;
    }
}

static void coordinator(const options & cfg, stage & local, channel * wire, state & status,
                        uint32_t & session, uint32_t & request) {
    if (wire) {
        const auto metadata = local.metadata(cfg);
        wire->send({hello, 0, 0, metadata});
        const auto response = wire->receive();
        expect(response, hello, 0, 0);
        require(response.body == metadata, "worker metadata mismatch");
        wire->send({open_session, 0, 0, {}});
        const auto ready = wire->receive();
        require(ready.session != 0 && ready.body.empty(), "invalid session acknowledgement");
        expect(ready, ack, ready.session, 0);
        session = ready.session;
    }
    status = state::active;
    auto tokens = cfg.tokens;
    request = 1;
    for (uint32_t offset = 0; offset < tokens.size();) {
        const uint32_t count = offset == 0 ? cfg.prefill : 1;
        std::vector<llama_pos> positions(count);
        for (uint32_t i = 0; i < count; ++i) positions[i] = offset + i;
        std::vector<int8_t> outputs(count, 1);
        auto batch = llama_batch_get_one(tokens.data() + offset, count);
        batch.pos = positions.data();
        batch.logits = outputs.data();
        local.compute(batch);
        if (!wire) {
            local.logits(local.output.logits, count);
        } else {
            bytes activation;
            for (auto field : {count, local.hidden, 0u, wire_f32}) put(activation, field);
            for (auto position : positions) put(activation, position);
            const auto * values = llama_get_embeddings(local.context.get());
            require(values, "missing stage boundary");
            for (size_t i = 0; i < size_t(count) * local.hidden; ++i) put_float(activation, values[i]);
            const size_t data_offset = 16 + 4 * count;
            local.output.boundary.insert(local.output.boundary.end(), activation.begin() + data_offset, activation.end());
            const auto start = steady_clock::now();
            wire->send({execute_stage, session, request, std::move(activation)});
            const auto result = wire->receive(cfg.response_ms);
            const auto ms = std::chrono::duration<double, std::milli>(steady_clock::now() - start).count();
            expect(result, stage_result, session, request);
            reader r{result.body};
            require(r.word() == 0 && r.word() == count && r.word() == local.vocab, "invalid output shape or status");
            floats(r, count, local.vocab);
            local.output.logits.insert(local.output.logits.end(), result.body.begin() + 12, result.body.end());
            std::fprintf(stderr, "roundtrip request=%u tokens=%u ms=%.3f\n", request, count, ms);
        }
        ++request;
        offset += count;
    }
    if (wire) {
        wire->send({close_session, session, request, {}});
        acknowledge(*wire, ack, session, request);
    }
    status = state::closed;
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 8,
                "usage: qwen3moe-stage-tcp ROLE MODEL SPLIT LAYERS PORT mmap|buffered OUTPUT_PREFIX [auto|SHA256|-] [OPTIONS]");
        options cfg{argv[1], argv[2], argv[6], argv[7], "", "",
            number(argv[5], 65535), number(argv[3], 65535), number(argv[4], 65535)};
        parse_options(cfg, argc, argv);
        require(cfg.role == "worker" || cfg.role == "coordinator" || cfg.role == "baseline", "invalid role");
        require(cfg.split && cfg.split < cfg.layers && (cfg.io == "mmap" || cfg.io == "buffered"), "invalid stage configuration");
        cfg.auto_identity = cfg.identity == "auto";
        require(cfg.auto_identity || cfg.identity.empty() || (cfg.identity.size() == 64 &&
                cfg.identity.find_first_not_of("0123456789abcdef") == std::string::npos), "invalid SHA256 label");
        require(cfg.fault.empty() || (cfg.role == "worker" &&
                (cfg.fault == "after-receive" || cfg.fault == "after-decode")), "invalid test fault option");
        ggml_backend_load_all();
        const auto hash_start = steady_clock::now();
        if (cfg.auto_identity) cfg.identity = model_identity(cfg.model);
        const auto hash_ms = std::chrono::duration<double, std::milli>(steady_clock::now() - hash_start).count();
#ifdef _WIN32
        SYSTEM_INFO machine{};
        GetNativeSystemInfo(&machine);
        USHORT process_machine = 0, native_machine = 0;
        using machine_info = BOOL (WINAPI *)(HANDLE, USHORT *, USHORT *);
        const auto query = reinterpret_cast<machine_info>(reinterpret_cast<void (*)()>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "IsWow64Process2")));
        if (query) query(GetCurrentProcess(), &process_machine, &native_machine);
        std::fprintf(stderr, "os=Windows native_machine=0x%x reported_arch=%u pid=%lu\n",
                unsigned(native_machine), unsigned(machine.wProcessorArchitecture), (unsigned long) GetCurrentProcessId());
#else
        std::fprintf(stderr, "os=Linux pid=%ld\n", (long) getpid());
#endif
#if defined(__x86_64__) || defined(_M_X64)
        std::fprintf(stderr, "binary_arch=x86_64\n");
#elif defined(__aarch64__) || defined(_M_ARM64)
        std::fprintf(stderr, "binary_arch=arm64\n");
#endif
        std::fprintf(stderr, "build=%s revision=%s identity=%s identity_source=%s hash_ms=%.3f role=%s\n",
                llama_version(), STAGE_BUILD_COMMIT, cfg.identity.empty() ? "-" : cfg.identity.c_str(),
                cfg.auto_identity ? "local-sha256" : cfg.identity.empty() ? "metadata" : "label",
                hash_ms, cfg.role.c_str());
        stage local(cfg);
        state status = state::created;
        uint32_t session = 0, request = 0;
        if (cfg.role == "baseline") {
            coordinator(cfg, local, nullptr, status, session, request);
        } else {
            stage_tcp::runtime sockets;
            auto peer = [&] {
                if (cfg.role == "coordinator") {
                    require(cfg.port > 0, "coordinator requires a worker port");
                    return stage_tcp::connect(uint16_t(cfg.port), std::chrono::milliseconds(cfg.connect_ms));
                }
                stage_tcp::socket listener;
                const auto port = stage_tcp::listen(listener, uint16_t(cfg.port));
                std::printf("READY %u\n", port);
                std::fflush(stdout);
                return stage_tcp::accept(listener, std::chrono::milliseconds(cfg.accept_ms));
            }();
            channel wire(std::move(peer), cfg.frame_ms);
            status = state::ready;
            try {
                if (cfg.role == "worker") worker(cfg, local, wire, status, session, request);
                else coordinator(cfg, local, &wire, status, session, request);
            } catch (const std::exception & error) {
                status = state::failed;
                if (cfg.fault.empty()) {
                    const std::string what = error.what();
                    try { wire.send({error_message, session, request, bytes(what.begin(), what.end())}); }
                    catch (const std::exception &) {}
                }
                throw;
            }
        }
        require(status == state::closed, "session did not close");
        local.report();
        local.output.save();
        std::fprintf(stderr, "CLOSED\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
