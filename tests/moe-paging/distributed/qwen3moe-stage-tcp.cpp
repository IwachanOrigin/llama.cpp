#include "stage-tcp.h"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "chat.h"
#include "common.h"

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
#include "stage-perf.h"
#include "stage-expert-trace.h"

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
    stage_performance * perf;
    explicit channel(stage_tcp::socket socket, uint32_t frame_ms, stage_performance * details)
        : peer(std::move(socket)), frame_limit(frame_ms), perf(details) {}

    void transfer(uint8_t * data, size_t size, bool sending, steady_clock::time_point deadline) {
        auto & count = sending ? sent : received;
        if (!perf) {
            peer.transfer(data, size, sending, deadline, count);
            return;
        }
        const auto start = steady_clock::now();
        auto record = [&] {
            (sending ? perf->send_ms : perf->receive_ms) += stage_performance::elapsed(start);
            (sending ? perf->sent : perf->received) = count;
        };
        try {
            peer.transfer(data, size, sending, deadline, count);
        } catch (const std::exception & error) {
            record();
            ++perf->transport_errors;
            if (std::strstr(error.what(), "timeout")) ++perf->timeouts;
            throw;
        }
        record();
    }

    void send(frame packet) {
        require(packet.body.size() <= cap, "outgoing frame too large");
        bytes header;
        for (auto value : {magic, version, packet.kind, uint32_t(packet.body.size()), packet.session, packet.request}) {
            put(header, value);
        }
        const auto deadline = steady_clock::now() + frame_limit;
        transfer(header.data(), header.size(), true, deadline);
        transfer(packet.body.data(), packet.body.size(), true, deadline);
        std::fprintf(stderr, "tx type=%u request=%u header=24 payload=%zu\n",
                packet.kind, packet.request, packet.body.size());
    }

    frame receive(uint32_t response_ms = 0) {
        bytes header(24);
        const size_t starting_bytes = response_ms ? 1 : 0;
        if (response_ms) {
            try {
                transfer(header.data(), 1, false, steady_clock::now() + std::chrono::milliseconds(response_ms));
            } catch (const std::runtime_error & error) {
                throw std::runtime_error(std::string("awaiting Worker response: ") + error.what());
            }
        }
        const auto deadline = steady_clock::now() + frame_limit;
        transfer(header.data() + starting_bytes, header.size() - starting_bytes, false, deadline);
        reader r{header};
        require(r.word() == magic, "bad protocol magic");
        require(r.word() == version, "unsupported protocol version");
        const uint32_t type = r.word();
        const uint32_t length = r.word();
        require(type >= hello && type <= ack, "unexpected message type");
        require(length <= cap, "invalid frame length");
        frame packet{type, r.word(), r.word(), bytes(length)};
        transfer(packet.body.data(), packet.body.size(), false, deadline);
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
    bool generate = false, fixed_input_set = false;
    uint32_t max_new_tokens = 0;
    std::string prompt_file{}, system_file{}, rendered_prompt_file{};
    bool chat = false;
    bool perf_details = false;
    uint32_t expert_trace_limit = 0;
    std::string perf_cache_state = "uncontrolled";
    std::vector<llama_token> prompt_tokens{};
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

static std::vector<llama_token> token_list(const std::string & value) {
    std::vector<llama_token> tokens;
    size_t begin = 0;
    for (;;) {
        const size_t comma = value.find(',', begin);
        tokens.push_back(number(value.substr(begin, comma - begin), INT32_MAX));
        if (comma == std::string::npos) return tokens;
        begin = comma + 1;
    }
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
        if (key == "--generate") {
            cfg.generate = true;
            continue;
        }
        if (key == "--chat") {
            cfg.chat = true;
            continue;
        }
        if (key == "--perf-details") {
            cfg.perf_details = true;
            continue;
        }
        require(i + 1 < argc, "option requires a value");
        const std::string value = argv[++i];
        if (key == "--prompt-file") {
            require(!value.empty(), "prompt path cannot be empty");
            cfg.prompt_file = value;
            continue;
        }
        if (key == "--system-file" || key == "--rendered-prompt-file") {
            require(!value.empty(), "chat file path cannot be empty");
            (key == "--system-file" ? cfg.system_file : cfg.rendered_prompt_file) = value;
            continue;
        }
        if (key == "--perf-cache-state") {
            require(value == "warm" || value == "cold" || value == "uncontrolled", "invalid cache-state label");
            cfg.perf_cache_state = value;
            continue;
        }
        if (key == "--sampling") {
            require(value == "greedy", "only greedy sampling is supported");
            continue;
        }
        if (key == "--prompt-tokens") {
            cfg.prompt_tokens = token_list(value);
            continue;
        }
        if (key == "--output-dir") {
            const auto path = std::filesystem::u8path(value);
            std::filesystem::create_directories(path);
            cfg.output = (path / std::filesystem::u8path(cfg.output).filename()).u8string();
            continue;
        }
        if (key == "--tokens") {
            cfg.tokens = token_list(value);
            cfg.fixed_input_set = true;
            continue;
        }
        const uint32_t n = number(value, 1000000);
        require(n != 0, "numeric options must be positive");
        if (key == "--ctx") cfg.ctx = n;
        else if (key == "--batch") cfg.batch = n;
        else if (key == "--threads") cfg.threads = n;
        else if (key == "--pager-budget-mib") cfg.pager_mib = n;
        else if (key == "--expert-trace-limit") cfg.expert_trace_limit = n;
        else if (key == "--prefill") {
            cfg.prefill = n;
            cfg.fixed_input_set = true;
        }
        else if (key == "--max-new-tokens") cfg.max_new_tokens = n;
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
    bool generation = false;
    bytes generated_tokens;
    std::string generated_text;

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
        if (generation) {
            write(".generated.txt", bytes(generated_text.begin(), generated_text.end()));
            write(".generated_tokens", generated_tokens);
            return;
        }
        write(".logits", logits);
        write(".boundary", boundary);
        for (const auto & entry : expert_ids) write("." + entry.first, entry.second);
    }
};

struct stage {
    artifacts output;
    std::unique_ptr<stage_expert_trace> trace;
    llama_model_ptr model;
    llama_context_ptr context;
    uint32_t begin, end;
    uint32_t hidden, vocab;
    stage_performance * perf;

    explicit stage(const options & cfg, stage_performance * details) : output(cfg.output), perf(details) {
        output.generation = cfg.generate;
        const bool baseline = cfg.role == "baseline";
        begin = cfg.role == "worker" ? cfg.split : 0;
        end = cfg.role == "coordinator" ? cfg.split : cfg.layers;
        auto mp = llama_model_default_params();
        mp.load_mode = cfg.io == "mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        mp.external_experts = true;
        require(uint64_t(cfg.pager_mib) * 1048576 <= SIZE_MAX, "Pager budget exceeds address space");
        mp.external_expert_cache_size = size_t(cfg.pager_mib) * 1048576;
        mp.use_extra_bufts = false;
        const auto load_start = perf ? steady_clock::now() : steady_clock::time_point{};
        model.reset(llama_model_load_stage(cfg.model.c_str(), mp, begin, end));
        require(bool(model) && uint32_t(llama_model_n_layer(model.get())) == cfg.layers, "stage model or layer count mismatch");
        if (perf) {
            perf->summary["model_load_wall_ms"] = stage_performance::elapsed(load_start);
            perf->summary["gguf_metadata_ms"] = llama_model_metadata_load_us(model.get()) / 1000.0;
            perf->summary["weights_load_ms"] = llama_model_weights_load_us(model.get()) / 1000.0;
            perf->summary["layer_range"] = {begin, end};
        }
        hidden = llama_model_n_embd(model.get());
        vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(hidden && vocab && 16 + uint64_t(cfg.batch) * (4 + uint64_t(hidden)*4) <= cap &&
                12 + uint64_t(cfg.batch) * vocab*4 <= cap, "batch/model dimensions exceed 16 MiB frame limit");
        require(cfg.generate || cfg.role == "worker" || (cfg.prefill <= cfg.batch && cfg.prefill <= cfg.tokens.size() && cfg.tokens.size() <= cfg.ctx &&
                std::all_of(cfg.tokens.begin(), cfg.tokens.end(), [&](llama_token id) { return uint32_t(id) < vocab; })),
                "invalid token sequence, prefill, batch or context");
        if (baseline) output.boundary_name = "l_out-" + std::to_string(cfg.split - 1);
        auto cp = llama_context_default_params();
        cp.n_ctx = cfg.ctx;
        cp.n_batch = cp.n_ubatch = cfg.batch;
        cp.n_threads = cp.n_threads_batch = cfg.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = cfg.generate ? nullptr : artifacts::eval;
        cp.cb_eval_user_data = &output;
        const auto context_start = perf ? steady_clock::now() : steady_clock::time_point{};
        context.reset(llama_init_from_model(model.get(), cp));
        require(bool(context), "cannot create stage context");
        if (cfg.expert_trace_limit) {
            trace = std::make_unique<stage_expert_trace>();
            trace->events.resize(cfg.expert_trace_limit);
            llama_set_pager_trace(model.get(), stage_expert_trace::capture, trace.get());
        }
        if (perf) {
            perf->summary["context_init_ms"] = stage_performance::elapsed(context_start);
            perf->summary["effective_context"] = llama_n_ctx(context.get());
            perf->summary["vocab"] = vocab;
            perf->summary["hidden"] = hidden;
        }
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

    void compute(llama_batch batch, stage_performance::row * sample, uint32_t request, const char * phase) {
        const size_t trace_begin = trace ? trace->seen : 0;
        if (trace) trace->decoding = std::strcmp(phase, "decode") == 0;
        const auto before = sample ? llama_get_pager_metrics(model.get()) : llama_pager_metrics{};
        const auto start = steady_clock::now();
        require(llama_decode(context.get(), batch) == 0, "stage decode failed");
        llama_synchronize(context.get());
        const auto ms = std::chrono::duration<double, std::milli>(steady_clock::now() - start).count();
        if (sample) {
            sample->execution = ms;
            const auto after = llama_get_pager_metrics(model.get());
            sample->pager = stage_performance::delta(after, before);
            sample->pager_valid = true;
        }
        if (trace) {
            trace->requests.push_back({{"request_id", request}, {"position", batch.pos[0]},
                {"token_count", batch.n_tokens}, {"begin", trace_begin}, {"end", trace->seen}, {"phase", phase}});
        }
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
        if (perf) {
            perf->summary["process_memory"] = stage_performance::process_memory();
            perf->summary["model_memory"] = {{"ordinary_tensor_bytes", normal_bytes}, {"expert_logical_bytes", expert_bytes},
                    {"kv_bytes", kv}, {"workspace_bytes", workspace}, {"expert_provider_count", experts}};
            perf->summary["pager"] = stage_performance::pager_json(stats, perf->stage_stats("")["input_tokens"].get<uint64_t>());
        }
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS memory{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) {
            std::fprintf(stderr, "working_set=%zu peak_working_set=%zu\n",
                    size_t(memory.WorkingSetSize), size_t(memory.PeakWorkingSetSize));
        }
#endif
        std::fprintf(stderr, "stage [%u,%u) ordinary=%zu normal_bytes=%zu experts=%zu expert_bytes=%zu "
                "pager=%zu kv=%zu workspace=%zu pinned=%zu reads=%llu hits=%llu read_bytes=%llu misses=%llu io_wait_ns=%llu\n",
                begin, end, ordinary, normal_bytes, experts, expert_bytes, stats.peak_allocated_bytes,
                kv, workspace, stats.pinned_slots, (unsigned long long) stats.disk_reads,
                (unsigned long long) stats.cache_hits, (unsigned long long) stats.bytes_read,
                (unsigned long long) stats.cache_misses, (unsigned long long) stats.io_wait_ns);
    }
};

static void acknowledge(channel & wire, message type, uint32_t session, uint32_t request) {
    const auto packet = wire.receive();
    expect(packet, type, session, request);
    require(packet.body.empty(), "unexpected acknowledgement payload");
}

static void worker(const options & cfg, stage & local, channel & wire, state & status, uint32_t & session, uint32_t & request) {
    auto * perf = local.perf;
    const auto handshake = perf ? steady_clock::now() : steady_clock::time_point{};
    auto announcement = wire.receive();
    expect(announcement, hello, 0, 0);
    require(announcement.body == local.metadata(cfg), "model or stage metadata mismatch");
    wire.send({hello, 0, 0, announcement.body});
    acknowledge(wire, open_session, 0, 0);
    session = std::random_device{}();
    if (!session) session = 1;
    status = state::active;
    wire.send({ack, session, 0, {}});
    if (perf) {
        perf->session = session;
        perf->summary["handshake_wall_ms"] = stage_performance::elapsed(handshake);
    }
    uint32_t position = 0;
    for (request = 1; ; ++request) {
        const auto receive_before = perf ? perf->receive_ms : 0;
        const auto bytes_before = wire.received;
        const auto packet = wire.receive();
        if (packet.kind == close_session) {
            expect(packet, close_session, session, request);
            require(packet.body.empty(), "invalid CLOSE");
            wire.send({ack, session, request, {}});
            status = state::closed;
            return;
        }
        expect(packet, execute_stage, session, request);
        const auto parse_start = perf ? steady_clock::now() : steady_clock::time_point{};
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
        stage_performance::row * sample = nullptr;
        if (perf) {
            perf->rows.push_back({"", request, position, count});
            sample = &perf->rows.back();
            sample->deserialize = stage_performance::elapsed(parse_start);
            sample->receive = perf->receive_ms - receive_before;
            sample->received = wire.received - bytes_before;
        }
        if (cfg.fault == "after-receive") throw std::runtime_error("injected disconnect after receive");
        std::vector<int8_t> outputs(count, 1);
        llama_batch batch{};
        batch.n_tokens = count;
        batch.embd = activation.data();
        batch.pos = positions.data();
        batch.logits = outputs.data();
        local.compute(batch, sample, request, "");
        if (cfg.fault == "after-decode") throw std::runtime_error("injected disconnect after decode");
        const auto encode_start = sample ? steady_clock::now() : steady_clock::time_point{};
        bytes result;
        put(result, 0);
        put(result, count);
        put(result, local.vocab);
        local.logits(result, count);
        if (sample) sample->serialize = stage_performance::elapsed(encode_start);
        const auto send_before = perf ? perf->send_ms : 0;
        const auto sent_before = wire.sent;
        wire.send({stage_result, session, request, std::move(result)});
        if (sample) {
            sample->send = perf->send_ms - send_before;
            sample->sent = wire.sent - sent_before;
        }
        position += count;
    }
}

static std::string checked_utf8(const std::string & input, bool replace) {
    std::string output;
    for (size_t i = 0; i < input.size();) {
        const uint8_t first = input[i];
        const size_t width = first < 0x80 ? 1 : first >= 0xc2 && first <= 0xdf ? 2 :
            first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
        bool valid = width && width <= input.size() - i;
        for (size_t j = 1; valid && j < width; ++j) {
            const uint8_t next = input[i + j];
            valid = (next & 0xc0) == 0x80 &&
                (j != 1 || ((first != 0xe0 || next >= 0xa0) && (first != 0xed || next < 0xa0) &&
                           (first != 0xf0 || next >= 0x90) && (first != 0xf4 || next < 0x90)));
        }
        if (valid) {
            output.append(input, i, width);
            i += width;
        } else {
            require(replace, "prompt is not valid UTF-8");
            output += "\xef\xbf\xbd";
            ++i;
        }
    }
    return output;
}

static std::string read_prompt_text(const std::string & path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open prompt file");
    const auto length = file.tellg();
    require(length > 0 && length <= 16 * 1024 * 1024, "prompt file must contain 1 to 16 MiB of UTF-8 data");
    std::string text(size_t(length), '\0');
    file.seekg(0);
    require(bool(file.read(text.data(), text.size())), "cannot read prompt file");
    if (text.compare(0, 3, "\xef\xbb\xbf") == 0) text.erase(0, 3);
    require(!text.empty(), "empty prompt");
    checked_utf8(text, false);
    return text;
}

static std::vector<llama_token> chat_prompt(const options & cfg, const llama_model * model,
                                           const llama_vocab * vocab) {
    const char * source = llama_model_chat_template(model, nullptr);
    require(source && *source, "GGUF has no default chat template");
    require(llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE, "model has no tokenizer");
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;
    auto append = [&](const char * role, const std::string & path) {
        common_chat_msg message;
        message.role = role;
        message.content = read_prompt_text(path);
        require(message.content.find_first_not_of(" \t\r\n") != std::string::npos, "empty chat message");
        inputs.messages.push_back(std::move(message));
    };
    if (!cfg.system_file.empty()) append("system", cfg.system_file);
    append("user", cfg.prompt_file);
    common_chat_params rendered;
    try {
        const auto templates = common_chat_templates_init(model, source);
        require(common_chat_templates_source(templates.get()) == source, "unexpected template substitution");
        rendered = common_chat_templates_apply(templates.get(), inputs);
    } catch (const std::exception &) {
        throw std::runtime_error("unsupported or invalid GGUF chat template");
    }
    require(!rendered.generation_prompt.empty(), "GGUF chat template did not emit an Assistant generation prompt");
    std::vector<llama_token> tokens;
    // Preserve full-prompt BPE unless a literal special spelling needs a protected span.
    bool protected_input = false;
    for (const auto & part : rendered.prompt.parts) {
        if (part.is_input && common_tokenize(vocab, part.text, false, true) !=
                common_tokenize(vocab, part.text, false, false)) {
            protected_input = true;
        }
    }
    const auto text = rendered.prompt.str();
    if (protected_input) {
        for (const auto & part : rendered.prompt.parts) {
            const auto chunk = common_tokenize(vocab, part.text, false, !part.is_input);
            tokens.insert(tokens.end(), chunk.begin(), chunk.end());
        }
    } else {
        tokens = common_tokenize(vocab, text, false, true);
    }
    // The Jinja renderer strips configured outer BOS/EOS. Add only the leading BOS for generation.
    const auto bos = llama_vocab_bos(vocab);
    if (llama_vocab_get_add_bos(vocab) && bos != LLAMA_TOKEN_NULL && (tokens.empty() || tokens.front() != bos)) {
        tokens.insert(tokens.begin(), bos);
    }
    if (!cfg.rendered_prompt_file.empty()) {
        const auto path = std::filesystem::u8path(cfg.rendered_prompt_file);
        require(!std::filesystem::exists(path), "use a new rendered prompt path");
        std::ofstream file(path, std::ios::binary);
        require(bool(file.write(text.data(), text.size())), "cannot write rendered prompt");
        file.close();
        require(bool(file), "cannot close rendered prompt file");
    }
    std::fprintf(stderr, "chat_enabled=true chat_template_source=gguf chat_template_supports_thinking=%s "
            "protected_input=%s\n", rendered.supports_thinking ? "true" : "false", protected_input ? "true" : "false");
    return tokens;
}

static std::vector<llama_token> generation_prompt(const options & cfg, const llama_model * model,
                                                  const llama_vocab * vocab) {
    if (cfg.chat) return chat_prompt(cfg, model, vocab);
    std::fprintf(stderr, "chat_enabled=false chat_template_source=none\n");
    if (!cfg.prompt_tokens.empty()) return cfg.prompt_tokens;
    require(llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE, "model has no tokenizer");
    const auto text = read_prompt_text(cfg.prompt_file);
    const int capacity = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
    require(capacity < 0 && capacity != INT32_MIN, "cannot size prompt tokens");
    std::vector<llama_token> tokens(-capacity);
    const int count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, false);
    require(count > 0 && size_t(count) <= tokens.size(), "cannot tokenize prompt");
    tokens.resize(count);
    return tokens;
}

static llama_token greedy(const std::vector<float> & logits) {
    require(!logits.empty(), "missing generation logits");
    llama_token best = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        require(std::isfinite(logits[i]), "non-finite generation logits");
        if (logits[i] > logits[best]) best = llama_token(i);
    }
    return best;
}

static std::string generation_text(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE || tokens.empty()) return {};
    const int capacity = llama_detokenize(vocab, tokens.data(), tokens.size(), nullptr, 0, true, false);
    if (capacity == 0) return {};
    require(capacity < 0 && capacity != INT32_MIN, "cannot size generated text");
    std::string text(-capacity, '\0');
    const int count = llama_detokenize(vocab, tokens.data(), tokens.size(), text.data(), text.size(), true, false);
    require(count >= 0 && size_t(count) <= text.size(), "cannot detokenize generation");
    text.resize(count);
    // A token limit can leave an incomplete UTF-8 sequence at the end.
    return checked_utf8(text, true);
}

static void coordinator(const options & cfg, stage & local, channel * wire, state & status,
                        uint32_t & session, uint32_t & request) {
    auto * perf = local.perf;
    const auto handshake = perf ? steady_clock::now() : steady_clock::time_point{};
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
        if (perf) {
            perf->session = session;
            perf->summary["handshake_wall_ms"] = stage_performance::elapsed(handshake);
        }
    }
    status = state::active;
    const auto * vocab = llama_model_get_vocab(local.model.get());
    const auto prepare_start = perf ? steady_clock::now() : steady_clock::time_point{};
    auto tokens = cfg.generate ? generation_prompt(cfg, local.model.get(), vocab) : cfg.tokens;
    if (perf) {
        perf->summary["prompt_preprocess_ms"] = stage_performance::elapsed(prepare_start);
        perf->summary["prompt_tokens"] = tokens.size();
        bytes encoded;
        for (auto token : tokens) put(encoded, uint32_t(token));
        uint8_t digest[SHA256_DIGEST_SIZE];
        sha256_hash(digest, encoded.data(), encoded.size());
        std::ostringstream fingerprint;
        for (auto value : digest) fingerprint << std::hex << std::setw(2) << std::setfill('0') << unsigned(value);
        perf->summary["prompt_token_sha256"] = fingerprint.str();
    }
    const uint32_t context_limit = std::min(cfg.ctx, llama_n_ctx(local.context.get()));
    require(tokens.size() <= context_limit &&
            std::all_of(tokens.begin(), tokens.end(), [&](llama_token id) { return uint32_t(id) < local.vocab; }),
            "prompt tokens exceed context or vocabulary");
    request = 1;
    double extra_decode_wall_ms = 0;
    uint32_t decode_steps = 0;
    auto evaluate = [&](llama_token * input, uint32_t offset, uint32_t count, const char * phase) {
        const auto evaluation_start = perf ? steady_clock::now() : steady_clock::time_point{};
        stage_performance::row * sample = nullptr;
        if (perf) {
            perf->rows.push_back({phase, request, offset, count});
            sample = &perf->rows.back();
        }
        std::vector<llama_pos> positions(count);
        for (uint32_t i = 0; i < count; ++i) positions[i] = offset + i;
        std::vector<int8_t> outputs(count, 1);
        auto batch = llama_batch_get_one(input, count);
        batch.pos = positions.data();
        batch.logits = outputs.data();
        local.compute(batch, sample, request, phase);
        std::vector<float> last;
        if (!wire) {
            if (cfg.generate) {
                const auto * values = llama_get_logits_ith(local.context.get(), count - 1);
                require(values, "missing generation logits");
                last.assign(values, values + local.vocab);
            } else {
                local.logits(local.output.logits, count);
            }
        } else {
            const auto encode_start = sample ? steady_clock::now() : steady_clock::time_point{};
            bytes activation;
            for (auto field : {count, local.hidden, 0u, wire_f32}) put(activation, field);
            for (auto position : positions) put(activation, position);
            const auto * values = llama_get_embeddings(local.context.get());
            require(values, "missing stage boundary");
            for (size_t i = 0; i < size_t(count) * local.hidden; ++i) put_float(activation, values[i]);
            const size_t data_offset = 16 + 4 * count;
            if (!cfg.generate) {
                local.output.boundary.insert(local.output.boundary.end(), activation.begin() + data_offset, activation.end());
            }
            if (sample) sample->serialize = stage_performance::elapsed(encode_start);
            const auto send_before = perf ? perf->send_ms : 0;
            const auto receive_before = perf ? perf->receive_ms : 0;
            const auto sent_before = wire->sent;
            const auto received_before = wire->received;
            const auto start = steady_clock::now();
            wire->send({execute_stage, session, request, std::move(activation)});
            const auto result = wire->receive(cfg.response_ms);
            const auto ms = std::chrono::duration<double, std::milli>(steady_clock::now() - start).count();
            if (sample) {
                sample->roundtrip = ms;
                sample->send = perf->send_ms - send_before;
                sample->receive = perf->receive_ms - receive_before;
                sample->sent = wire->sent - sent_before;
                sample->received = wire->received - received_before;
            }
            const auto parse_start = sample ? steady_clock::now() : steady_clock::time_point{};
            expect(result, stage_result, session, request);
            reader r{result.body};
            require(r.word() == 0 && r.word() == count && r.word() == local.vocab, "invalid output shape or status");
            const auto remote_logits = floats(r, count, local.vocab);
            if (cfg.generate) {
                last.assign(remote_logits.end() - local.vocab, remote_logits.end());
            } else {
                local.output.logits.insert(local.output.logits.end(), result.body.begin() + 12, result.body.end());
            }
            std::fprintf(stderr, "roundtrip request=%u tokens=%u ms=%.3f\n", request, count, ms);
            if (sample) sample->deserialize = stage_performance::elapsed(parse_start);
        }
        ++request;
        if (perf && std::strcmp(phase, "decode") == 0) extra_decode_wall_ms += stage_performance::elapsed(evaluation_start);
        return last;
    };
    const auto prefill_start = steady_clock::now();
    std::vector<float> last;
    uint32_t prefill_batches = 0;
    for (uint32_t offset = 0; offset < tokens.size();) {
        const uint32_t count = cfg.generate ? std::min<uint32_t>(cfg.batch, tokens.size() - offset) :
            (offset == 0 ? cfg.prefill : 1);
        last = evaluate(tokens.data() + offset, offset, count, cfg.generate ? "prefill" : "fixed");
        offset += count;
        ++prefill_batches;
    }
    if (cfg.generate) {
        const auto decode_start = steady_clock::now();
        const double prefill_ms = std::chrono::duration<double, std::milli>(decode_start - prefill_start).count();
        std::vector<llama_token> generated;
        uint32_t next_position = tokens.size();
        const char * reason = "limit";
        double selection_ms = 0, first_selection_ms = 0, stop_check_ms = 0;
        for (uint32_t step = 0; step < cfg.max_new_tokens; ++step) {
            const auto selection_start = perf ? steady_clock::now() : steady_clock::time_point{};
            auto token = greedy(last);
            if (perf) {
                const double duration = stage_performance::elapsed(selection_start);
                selection_ms += duration;
                if (step == 0) first_selection_ms = duration;
            }
            generated.push_back(token);
            const auto stop_start = perf ? steady_clock::now() : steady_clock::time_point{};
            const char * stop = nullptr;
            if (llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE && llama_vocab_is_eog(vocab, token)) stop = "eog";
            else if (step + 1 == cfg.max_new_tokens) stop = "limit";
            else if (next_position == context_limit) stop = "context";
            if (perf) stop_check_ms += stage_performance::elapsed(stop_start);
            if (stop) {
                reason = stop;
                break;
            }
            last = evaluate(&token, next_position++, 1, "decode");
            ++decode_steps;
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(steady_clock::now() - decode_start).count();
        if (perf) {
            perf->summary["generation"] = {{"generated_tokens", generated.size()}, {"stop", reason},
                {"prefill_batches", prefill_batches}, {"prefill_ms", prefill_ms},
                {"prefill_tokens_per_sec", 1000 * tokens.size() / prefill_ms},
                {"generation_loop_ms", decode_ms}, {"decode_steps", decode_steps},
                {"decode_step_wall_ms", extra_decode_wall_ms},
                {"decode_tokens_per_sec", decode_steps && extra_decode_wall_ms > 0 ?
                        stage_performance::json(1000 * decode_steps / extra_decode_wall_ms) : stage_performance::json(nullptr)},
                {"generation_elapsed_ms", prefill_ms + decode_ms},
                {"generation_tokens_per_sec", 1000 * generated.size() / (prefill_ms + decode_ms)},
                {"greedy_ms", selection_ms}, {"first_selection_ms", first_selection_ms},
                {"stop_check_ms", stop_check_ms}};
        }
        local.output.generated_text = generation_text(vocab, generated);
        for (auto token : generated) put(local.output.generated_tokens, uint32_t(token));
        std::fprintf(stderr, "generation prompt_tokens=%zu generated_tokens=%zu stop=%s prefill_ms=%.3f "
                "decode_ms=%.3f ms_per_token=%.3f tokens_per_sec=%.3f\n",
                tokens.size(), generated.size(), reason, prefill_ms, decode_ms, decode_ms / generated.size(),
                decode_ms > 0 ? 1000 * generated.size() / decode_ms : 0);
    }
    if (wire) {
        wire->send({close_session, session, request, {}});
        acknowledge(*wire, ack, session, request);
    }
    status = state::closed;
}

int main(int argc, char ** argv) {
    std::unique_ptr<stage_performance> perf;
    try {
        require(argc >= 8,
                "usage: qwen3moe-stage-tcp ROLE MODEL SPLIT LAYERS PORT mmap|buffered OUTPUT_PREFIX [auto|SHA256|-] [OPTIONS]");
        options cfg{argv[1], argv[2], argv[6], argv[7], "", "",
            number(argv[5], 65535), number(argv[3], 65535), number(argv[4], 65535)};
        parse_options(cfg, argc, argv);
        require(cfg.chat ? cfg.generate && !cfg.prompt_file.empty() :
                cfg.system_file.empty() && cfg.rendered_prompt_file.empty(),
                "--chat requires --generate and --prompt-file; chat file options require --chat");
        require(cfg.role == "worker" || cfg.role == "coordinator" || cfg.role == "baseline", "invalid role");
        require(!cfg.generate ? (cfg.prompt_file.empty() && cfg.prompt_tokens.empty() && !cfg.max_new_tokens) :
                (cfg.role != "worker" && !cfg.fixed_input_set && cfg.max_new_tokens &&
                 (cfg.prompt_file.empty() != cfg.prompt_tokens.empty())),
                "generation requires baseline/coordinator, --max-new-tokens and exactly one prompt source; "
                "fixed --tokens/--prefill cannot be combined with --generate");
        if (cfg.generate) {
            for (const char * suffix : {".generated_tokens", ".generated.txt"}) {
                require(!std::filesystem::exists(std::filesystem::u8path(cfg.output + suffix)),
                        "use a fresh generation output prefix");
            }
        }
        require(cfg.split && cfg.split < cfg.layers && (cfg.io == "mmap" || cfg.io == "buffered"), "invalid stage configuration");
        cfg.auto_identity = cfg.identity == "auto";
        require(cfg.auto_identity || cfg.identity.empty() || (cfg.identity.size() == 64 &&
                cfg.identity.find_first_not_of("0123456789abcdef") == std::string::npos), "invalid SHA256 label");
        require(cfg.fault.empty() || (cfg.role == "worker" &&
                (cfg.fault == "after-receive" || cfg.fault == "after-decode")), "invalid test fault option");
        if (cfg.perf_details) {
            for (const char * suffix : {".metrics.json", ".perf.csv"}) {
                require(!std::filesystem::exists(std::filesystem::u8path(cfg.output + suffix)),
                        "use a fresh performance output prefix");
            }
            perf = std::make_unique<stage_performance>();
            perf->prefix = cfg.output;
            perf->summary = {{"role", cfg.role}, {"revision", STAGE_BUILD_COMMIT}, {"version", llama_version()},
                {"model_path", cfg.model}, {"identity_source", cfg.auto_identity ? "local-sha256" :
                        cfg.identity.empty() ? "metadata" : "label"},
                {"split", cfg.split}, {"layers", cfg.layers}, {"context", cfg.ctx}, {"batch", cfg.batch},
                {"threads", cfg.threads}, {"pager_budget_mib", cfg.pager_mib}, {"io", cfg.io},
                {"cache_state_label", cfg.perf_cache_state}, {"chat", cfg.role == "worker" ?
                        stage_performance::json(nullptr) : stage_performance::json(cfg.chat)},
                {"max_new_tokens", cfg.role == "worker" ? stage_performance::json(nullptr) :
                        stage_performance::json(cfg.max_new_tokens)}};
        }
        if (cfg.expert_trace_limit) {
            require(!std::filesystem::exists(std::filesystem::u8path(cfg.output + ".experts.jsonl")),
                    "use a fresh Expert trace output prefix");
            if (perf) perf->summary["expert_trace_limit"] = cfg.expert_trace_limit;
        }
        ggml_backend_load_all();
        const auto hash_start = steady_clock::now();
        if (cfg.auto_identity) cfg.identity = model_identity(cfg.model);
        const auto hash_ms = std::chrono::duration<double, std::milli>(steady_clock::now() - hash_start).count();
        if (perf) {
            perf->summary["hash_ms"] = cfg.auto_identity ? stage_performance::json(hash_ms) : stage_performance::json(nullptr);
            perf->summary["model_identity"] = cfg.identity.empty() ? stage_performance::json(nullptr) :
                    stage_performance::json(cfg.identity);
        }
#ifdef _WIN32
        SYSTEM_INFO machine{};
        GetNativeSystemInfo(&machine);
        USHORT process_machine = 0, native_machine = 0;
        using machine_info = BOOL (WINAPI *)(HANDLE, USHORT *, USHORT *);
        const auto query = reinterpret_cast<machine_info>(reinterpret_cast<void (*)()>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "IsWow64Process2")));
        if (query) query(GetCurrentProcess(), &process_machine, &native_machine);
        if (perf) perf->summary["host"] = {{"os", "Windows"}, {"native_machine", native_machine},
                                           {"pid", GetCurrentProcessId()}};
        std::fprintf(stderr, "os=Windows native_machine=0x%x reported_arch=%u pid=%lu\n",
                unsigned(native_machine), unsigned(machine.wProcessorArchitecture), (unsigned long) GetCurrentProcessId());
#else
        if (perf) perf->summary["host"] = {{"os", "Linux"}, {"pid", getpid()}};
        std::fprintf(stderr, "os=Linux pid=%ld\n", (long) getpid());
#endif
#if defined(__x86_64__) || defined(_M_X64)
        if (perf) perf->summary["binary_arch"] = "x86_64";
        std::fprintf(stderr, "binary_arch=x86_64\n");
#elif defined(__aarch64__) || defined(_M_ARM64)
        if (perf) perf->summary["binary_arch"] = "arm64";
        std::fprintf(stderr, "binary_arch=arm64\n");
#endif
        std::fprintf(stderr, "build=%s revision=%s identity=%s identity_source=%s hash_ms=%.3f role=%s\n",
                llama_version(), STAGE_BUILD_COMMIT, cfg.identity.empty() ? "-" : cfg.identity.c_str(),
                cfg.auto_identity ? "local-sha256" : cfg.identity.empty() ? "metadata" : "label",
                hash_ms, cfg.role.c_str());
        stage local(cfg, perf.get());
        state status = state::created;
        uint32_t session = 0, request = 0;
        if (cfg.role == "baseline") {
            coordinator(cfg, local, nullptr, status, session, request);
        } else {
            const auto connect_start = perf ? steady_clock::now() : steady_clock::time_point{};
            stage_tcp::runtime sockets;
            auto peer = [&] {
                try {
                    if (cfg.role == "coordinator") {
                        require(cfg.port > 0, "coordinator requires a worker port");
                        return stage_tcp::connect(uint16_t(cfg.port), std::chrono::milliseconds(cfg.connect_ms));
                    }
                    stage_tcp::socket listener;
                    const auto port = stage_tcp::listen(listener, uint16_t(cfg.port));
                    std::printf("READY %u\n", port);
                    std::fflush(stdout);
                    return stage_tcp::accept(listener, std::chrono::milliseconds(cfg.accept_ms));
                } catch (const std::exception & error) {
                    if (perf) {
                        ++perf->transport_errors;
                        if (std::strstr(error.what(), "timeout")) ++perf->timeouts;
                    }
                    throw;
                }
            }();
            if (perf) perf->summary[cfg.role == "worker" ? "accept_wall_ms" : "connect_wall_ms"] =
                    stage_performance::elapsed(connect_start);
            channel wire(std::move(peer), cfg.frame_ms, perf.get());
            status = state::ready;
            try {
                if (cfg.role == "worker") worker(cfg, local, wire, status, session, request);
                else coordinator(cfg, local, &wire, status, session, request);
            } catch (const std::exception & error) {
                status = state::failed;
                if (perf) ++perf->session_errors;
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
        if (local.trace) local.trace->save(cfg.output, cfg.role, local.model.get());
        if (perf) perf->save(true);
        std::fprintf(stderr, "CLOSED\n");
        return 0;
    } catch (const std::exception & error) {
        if (perf) {
            try { perf->save(false, error.what()); }
            catch (const std::exception & write_error) {
                std::fprintf(stderr, "cannot save performance metrics: %s\n", write_error.what());
            }
        }
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
