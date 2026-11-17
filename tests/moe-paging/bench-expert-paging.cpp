#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama-ext.h"
#include "llama.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#ifndef _WIN32
#include <sys/utsname.h>
#endif

namespace {

using clock_type = std::chrono::steady_clock;
namespace fs = std::filesystem;

double ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

struct options {
    std::string model, output, trace;
    std::string pager = "off", io = "mmap", cache_state = "uncontrolled";
    size_t budget = 0, slots = 0, required_available = 0;
    int threads = 4, context = 512, batch = 128, ubatch = 64, prefill = 128, decode = 32, contexts = 1, warmup = 0;
    bool plan = false, sample_decode_memory = false;
};

size_t number(const std::string & value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected a nonnegative integer: " + value);
    }
    const auto n = std::stoull(value);
    if (n > SIZE_MAX) {
        throw std::out_of_range("number is too large");
    }
    return static_cast<size_t>(n);
}

size_t byte_budget(const std::string & value) {
    const auto end = value.find_first_not_of("0123456789");
    const size_t n = number(value.substr(0, end));
    const std::string unit = end == std::string::npos ? "" : value.substr(end);
    const size_t multiplier = unit == "GiB" ? size_t(1) << 30 : unit == "MiB" ? size_t(1) << 20 :
        unit == "KiB" ? size_t(1) << 10 : unit.empty() || unit == "B" ? 1 : 0;
    if (!multiplier || n > SIZE_MAX / multiplier) {
        throw std::invalid_argument("invalid budget (use B, KiB, MiB or GiB)");
    }
    return n * multiplier;
}

options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--plan") {
            o.plan = true;
            continue;
        }
        if (key == "--sample-decode-memory") {
            o.sample_decode_memory = true;
            continue;
        }
        if (++i == argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[i];
        if (key == "--model") o.model = value;
        else if (key == "--output") o.output = value;
        else if (key == "--trace-prefix") o.trace = value;
        else if (key == "--pager") o.pager = value;
        else if (key == "--io") o.io = value;
        else if (key == "--cache-state") o.cache_state = value;
        else if (key == "--budget") o.budget = byte_budget(value);
        else if (key == "--slots") o.slots = number(value);
        else if (key == "--required-available-bytes") o.required_available = number(value);
        else {
            const size_t n = number(value);
            if (n > INT32_MAX) {
                throw std::invalid_argument("integer option is too large: " + key);
            }
            if (key == "--threads") o.threads = n;
            else if (key == "--ctx") o.context = n;
            else if (key == "--batch") o.batch = n;
            else if (key == "--ubatch") o.ubatch = n;
            else if (key == "--prefill") o.prefill = n;
            else if (key == "--decode") o.decode = n;
            else if (key == "--contexts") o.contexts = n;
            else if (key == "--warmup") o.warmup = n;
            else throw std::invalid_argument("unknown option " + key);
        }
    }
    if (o.model.empty() || o.output.empty() || !o.threads || !o.prefill || !o.decode || !o.ubatch ||
            o.batch < o.ubatch || int64_t(o.prefill) + o.decode > o.context ||
            o.contexts < 1 || o.contexts > 2 ||
            (o.io != "mmap" && o.io != "buffered") ||
            (o.pager != "off" && o.pager != "single" && o.pager != "lru") ||
            (o.pager == "lru" ? (!!o.budget == !!o.slots) : (o.budget || o.slots)) ||
            (o.slots && !o.plan)) {
        throw std::invalid_argument("invalid benchmark configuration; use --help");
    }
    return o;
}

struct process_memory {
    uint64_t resident, lifetime_peak;
};

struct process_io {
    bool available = false;
    uint64_t read_requests = 0, interface_bytes = 0, storage_bytes = 0;
};

process_io io_counters() {
#ifdef _WIN32
    IO_COUNTERS io{};
    if (!GetProcessIoCounters(GetCurrentProcess(), &io)) return {};
    return {true, io.ReadOperationCount, io.ReadTransferCount, 0};
#else
    std::ifstream status("/proc/self/io");
    process_io io{};
    for (std::string label; status >> label;) {
        uint64_t value;
        if (!(status >> value)) break;
        if (label == "syscr:") io.read_requests = value;
        if (label == "rchar:") io.interface_bytes = value;
        if (label == "read_bytes:") {
            io.storage_bytes = value;
            io.available = true;
        }
    }
    return io;
#endif
}

void write_io(std::ostream & out, const char * name, process_io a, process_io b) {
    if (!a.available || !b.available) return;
    out << ",\"" << name << "\":{\"read_requests\":" << b.read_requests - a.read_requests
        << ",\"interface_read_bytes\":" << b.interface_bytes - a.interface_bytes;
#ifndef _WIN32
    out << ",\"kernel_storage_read_bytes\":" << b.storage_bytes - a.storage_bytes;
#endif
    out << "}";
}

process_memory memory() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        throw std::runtime_error("GetProcessMemoryInfo failed");
    }
    return {counters.WorkingSetSize, counters.PeakWorkingSetSize};
#else
    std::ifstream status("/proc/self/status");
    process_memory result{};
    for (std::string line; std::getline(status, line);) {
        if (line.compare(0, 6, "VmRSS:") == 0) result.resident = std::stoull(line.substr(6)) * 1024;
        if (line.compare(0, 6, "VmHWM:") == 0) result.lifetime_peak = std::stoull(line.substr(6)) * 1024;
    }
    if (!result.lifetime_peak) throw std::runtime_error("cannot read process memory from /proc");
    return result;
#endif
}

struct system_memory {
    uint64_t total, available;
};

system_memory physical_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX state{};
    state.dwLength = sizeof(state);
    if (!GlobalMemoryStatusEx(&state)) throw std::runtime_error("GlobalMemoryStatusEx failed");
    return {state.ullTotalPhys, state.ullAvailPhys};
#else
    std::ifstream info("/proc/meminfo");
    system_memory result{};
    for (std::string line; std::getline(info, line);) {
        if (line.compare(0, 9, "MemTotal:") == 0) result.total = std::stoull(line.substr(9)) * 1024;
        if (line.compare(0, 13, "MemAvailable:") == 0) result.available = std::stoull(line.substr(13)) * 1024;
    }
    if (!result.total) throw std::runtime_error("cannot read MemTotal from /proc");
    return result;
#endif
}

void write_memory(std::ostream & out, const char * name, process_memory usage) {
    out << ",\"" << name << "\":{\"resident_bytes\":" << usage.resident
        << ",\"lifetime_peak_bytes\":" << usage.lifetime_peak << "}";
}

struct trace {
    std::ofstream ids, prefill, decode;
    bool enabled = false;
    size_t expert_ids = 0;
    size_t prefill_values = 0, decode_values = 0;
};

bool eval(ggml_tensor * tensor, bool ask, void * data) {
    auto & t = *static_cast<trace *>(data);
    const bool selected = t.enabled && tensor->type == GGML_TYPE_I32 && std::string_view(tensor->name).substr(0, 13) == "ffn_moe_topk-";
    if (!ask && selected) {
        std::vector<char> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        for (int64_t j = 0; j < tensor->ne[1]; ++j) {
            for (int64_t i = 0; i < tensor->ne[0]; ++i) {
                t.ids.write(bytes.data() + j * tensor->nb[1] + i * tensor->nb[0], sizeof(int32_t));
                ++t.expert_ids;
            }
        }
    }
    return ask ? selected : true;
}

struct context_run {
    trace record;
    llama_context_ptr context;
    uint64_t decode_peak = 0;
    double prefill_ms = 0, decode_ms = 0;
};

template <typename F>
double parallel(int contexts, F run) {
    std::promise<void> start;
    auto go = start.get_future();
    std::future<void> peer;
    if (contexts == 2) {
        peer = std::async(std::launch::async, [&] { go.wait(); run(1); });
    }
    const auto began = clock_type::now();
    start.set_value();
    run(0);
    if (peer.valid()) peer.get();
    return ms(began);
}

void check_decode(context_run & run, llama_batch batch, bool prefill, int outputs, int vocab) {
    const auto start = clock_type::now();
    if (llama_decode(run.context.get(), batch) != 0) throw std::runtime_error("llama_decode failed");
    llama_synchronize(run.context.get());
    (prefill ? run.prefill_ms : run.decode_ms) += ms(start);
    if (!run.record.enabled) return;
    auto & file = prefill ? run.record.prefill : run.record.decode;
    for (int i = 0; i < outputs; ++i) {
        const auto * values = llama_get_logits_ith(run.context.get(), i);
        if (!values || !std::all_of(values, values + vocab, [](float v) { return std::isfinite(v); })) {
            throw std::runtime_error("missing or non-finite logits");
        }
        file.write(reinterpret_cast<const char *>(values), size_t(vocab) * sizeof(float));
    }
    (prefill ? run.record.prefill_values : run.record.decode_values) += size_t(outputs) * vocab;
}

void write_stats(std::ostream & out, const char * phase, const llama_pager_metrics & before, const llama_pager_metrics & after) {
    const auto hits = after.cache_hits - before.cache_hits;
    const auto misses = after.cache_misses - before.cache_misses;
    out << ",\"" << phase << "\":{\"cache_hits\":" << hits << ",\"cache_misses\":" << misses
        << ",\"hit_rate\":" << (hits + misses ? double(hits) / (hits + misses) : 0)
        << ",\"cache_evictions\":" << after.cache_evictions - before.cache_evictions
        << ",\"disk_reads\":" << after.disk_reads - before.disk_reads
        << ",\"bytes_read\":" << after.bytes_read - before.bytes_read
        << ",\"io_wait_ns\":" << after.io_wait_ns - before.io_wait_ns
        << ",\"cache_wait_ns\":" << after.cache_wait_ns - before.cache_wait_ns
        << ",\"capacity_wait_ns\":" << after.capacity_wait_ns - before.capacity_wait_ns
        << ",\"lease_wait_ns\":" << after.lease_wait_ns - before.lease_wait_ns
        << ",\"pinned_slots\":" << after.pinned_slots << "}";
}

std::string metadata(const llama_model * model, const char * key) {
    char value[64] = {};
    if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) return "";
    return value;
}

const char * target_architecture() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

void write_architecture(std::ostream & out) {
    std::string host = "unknown";
    const char * translated = "null";
#ifdef _WIN32
    USHORT process = 0, native = 0;
    using query_fn = BOOL (WINAPI *)(HANDLE, USHORT *, USHORT *);
    const auto query = reinterpret_cast<query_fn>(reinterpret_cast<void (*)()>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "IsWow64Process2")));
    if (query && query(GetCurrentProcess(), &process, &native)) {
        if (native == IMAGE_FILE_MACHINE_ARM64) host = "arm64";
        else if (native == IMAGE_FILE_MACHINE_AMD64) host = "x86_64";
        else if (native == IMAGE_FILE_MACHINE_I386) host = "x86";
        // The process machine can be UNKNOWN for an x64 process on ARM64.
        translated = process != IMAGE_FILE_MACHINE_UNKNOWN ||
            (host == "arm64" && std::string_view(target_architecture()) == "x86_64") ? "true" : "false";
    }
#else
    struct utsname info = {};
    if (uname(&info) == 0) host = info.machine;
#endif
    out << ",\"host_architecture\":" << std::quoted(host)
        << ",\"binary_architecture\":" << std::quoted(target_architecture())
        << ",\"process_emulated\":" << translated;
}

void benchmark(options o) {
    if (fs::exists(fs::u8path(o.output)) && fs::equivalent(fs::u8path(o.output), fs::u8path(o.model))) {
        throw std::invalid_argument("output path refers to the model");
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t cpu_only[] = {nullptr};
    mp.devices = cpu_only;
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    // The sizing pass uses unmapped buffers to estimate full resident demand.
    mp.load_mode = !o.plan && o.io == "mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    mp.external_experts = o.pager != "off";
    mp.no_alloc = o.plan;
    if (o.slots) {
        llama_model_ptr sizing(llama_model_load_from_file(o.model.c_str(), mp));
        if (!sizing) throw std::runtime_error("could not size Expert cache");
        o.budget = llama_pager_budget_for_slots(sizing.get(), o.slots);
    }
    mp.external_expert_cache_size = o.budget;
    const auto physical = physical_memory();
    if (o.required_available > physical.available) {
        throw std::runtime_error("available physical memory fell below the preflight requirement");
    }
    const auto before_load = memory();
    const auto before_load_io = io_counters();
    auto start = clock_type::now();
    llama_model_ptr model(llama_model_load_from_file(o.model.c_str(), mp));
    const double load_ms = ms(start);
    if (!model) throw std::runtime_error("model load failed");
    const auto after_load = memory();
    const auto after_load_io = io_counters();
    const auto pager = llama_get_pager_metrics(model.get());
    const size_t total_model = llama_get_model_buffer_bytes(model.get());
    // memory_breakdown includes the Pager's planned or allocated bytes exactly once.
    if (pager.peak_allocated_bytes > total_model) {
        throw std::runtime_error("Pager capacity exceeds model buffer accounting");
    }
    const size_t ordinary_model = total_model - pager.peak_allocated_bytes;

    auto cp = llama_context_default_params();
    cp.n_ctx = o.context;
    cp.n_batch = o.batch;
    cp.n_ubatch = o.ubatch;
    cp.n_threads = cp.n_threads_batch = o.threads;
    std::vector<context_run> runs(o.contexts);
    start = clock_type::now();
    for (int i = 0; i < o.contexts; ++i) {
        auto & run = runs[i];
        if (!o.trace.empty() && !o.plan) {
            const std::string prefix = o.trace + "." + std::to_string(i);
            run.record.ids.open(fs::u8path(prefix + ".ids"), std::ios::binary);
            run.record.prefill.open(fs::u8path(prefix + ".prefill.f32"), std::ios::binary);
            run.record.decode.open(fs::u8path(prefix + ".decode.f32"), std::ios::binary);
            run.record.ids.exceptions(std::ios::badbit | std::ios::failbit);
            run.record.prefill.exceptions(std::ios::badbit | std::ios::failbit);
            run.record.decode.exceptions(std::ios::badbit | std::ios::failbit);
            cp.cb_eval = eval;
            cp.cb_eval_user_data = &run.record;
        }
        run.context.reset(llama_init_from_model(model.get(), cp));
        if (!run.context) throw std::runtime_error("context initialization failed");
    }
    const double context_ms = ms(start);
    const auto after_context = memory();
    auto context_bytes = [&] {
        llama_memory_breakdown_data total;
        for (const auto & run : runs) {
            for (const auto & entry : llama_get_memory_breakdown(run.context.get())) {
                total.context += entry.second.context;
                total.compute += entry.second.compute;
            }
        }
        return total;
    };
    const auto initial_context_bytes = context_bytes();

    std::ofstream out(fs::u8path(o.output));
    out.exceptions(std::ios::badbit | std::ios::failbit);
    char description[128] = {};
    llama_model_desc(model.get(), description, sizeof(description));
    out << std::setprecision(9) << "{\"model\":" << std::quoted(o.model)
        << ",\"architecture\":" << std::quoted(metadata(model.get(), "general.architecture"))
        << ",\"gguf_file_type\":" << std::quoted(metadata(model.get(), "general.file_type"))
        << ",\"description\":" << std::quoted(description)
        << ",\"model_tensor_bytes\":" << llama_model_size(model.get())
        << ",\"vocab_size\":" << llama_vocab_n_tokens(llama_model_get_vocab(model.get()))
        << ",\"pager\":" << std::quoted(o.pager) << ",\"io\":" << std::quoted(o.io)
        << ",\"os_cache_state\":" << std::quoted(o.cache_state)
        << ",\"plan\":" << (o.plan ? "true" : "false")
#ifdef _WIN32
        << ",\"memory_definition\":\"windows_working_set\""
        << ",\"process_io_definition\":\"windows_process_io_counters\""
#else
        << ",\"memory_definition\":\"linux_proc_vmrss_vmhwm\""
        << ",\"process_io_definition\":\"linux_proc_self_io\""
#endif
        << ",\"total_physical_bytes\":" << physical.total << ",\"available_physical_bytes\":" << physical.available
        << ",\"threads_per_context\":" << o.threads << ",\"contexts\":" << o.contexts
        << ",\"requested_context\":" << o.context << ",\"effective_context\":" << llama_n_ctx(runs[0].context.get())
        << ",\"batch\":" << o.batch << ",\"ubatch\":" << o.ubatch << ",\"warmup_runs\":" << o.warmup
        << ",\"prefill_tokens_per_context\":" << o.prefill << ",\"decode_steps_per_context\":" << o.decode
        << ",\"load_ms\":" << load_ms << ",\"metadata_ms\":" << llama_model_metadata_load_us(model.get()) / 1000.0
        << ",\"weights_ms\":" << llama_model_weights_load_us(model.get()) / 1000.0 << ",\"context_ms\":" << context_ms
        << ",\"model_buffer_bytes_including_pager\":" << total_model
        << ",\"model_buffer_bytes_excluding_pager\":" << ordinary_model
        << ",\"ordinary_model_budget_kind\":"
        << std::quoted(o.io == "mmap" ? "full_ordinary_weight_allowance" : "ordinary_buffer_capacity")
        << ",\"kv_bytes\":" << initial_context_bytes.context << ",\"workspace_bytes\":" << initial_context_bytes.compute
        << ",\"budget_bytes\":" << o.budget << ",\"slots\":" << pager.slots
        << ",\"max_expert_bytes\":" << pager.max_expert_bytes
        << ",\"expert_slot_payload_bytes\":" << pager.slots * pager.max_expert_bytes
        << ",\"peak_allocated_bytes\":" << pager.peak_allocated_bytes;
    write_architecture(out);
    out << ",\"registered_tensor_bytes_by_type\":{";
    bool first_type = true;
    for (const auto & entry : llama_model_tensor_type_bytes(model.get())) {
        if (!first_type) out << ',';
        first_type = false;
        out << std::quoted(ggml_type_name(entry.first)) << ':' << entry.second;
    }
    out << '}';
    write_memory(out, "before_load", before_load);
    write_memory(out, "after_load", after_load);
    write_memory(out, "after_context", after_context);
    write_io(out, "load_process_io", before_load_io, after_load_io);

    if (!o.plan) {
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (vocab < 2) throw std::runtime_error("model has insufficient vocabulary");
        const std::vector<llama_token> seed{1, 19, 7, 42, 51, 33, 91, 3};
        std::vector<llama_token> prompt(o.prefill);
        for (int i = 0; i < o.prefill; ++i) prompt[i] = seed[i % seed.size()] % vocab;

        auto prefill = [&](int i) {
            auto & run = runs[i];
            std::vector<int8_t> flags(o.batch);
            for (int first = 0; first < o.prefill;) {
                const int n = std::min(o.batch, o.prefill - first);
                std::fill_n(flags.begin(), n, run.record.enabled ? 1 : 0);
                flags[n - 1] = 1;
                auto batch = llama_batch_get_one(prompt.data() + first, n);
                batch.logits = flags.data();
                check_decode(run, batch, true, run.record.enabled ? n : 1, vocab);
                first += n;
            }
        };
        auto decode = [&](int i, bool sample) {
            auto & run = runs[i];
            for (int step = 0; step < o.decode; ++step) {
                llama_token token = 1 + (17LL * step + 13) % (vocab - 1);
                check_decode(run, llama_batch_get_one(&token, 1), false, 1, vocab);
                if (sample) run.decode_peak = std::max(run.decode_peak, memory().resident);
            }
        };
        for (int i = 0; i < o.warmup; ++i) {
            parallel(o.contexts, prefill);
            parallel(o.contexts, [&](int id) { decode(id, false); });
            for (auto & run : runs) llama_memory_clear(llama_get_memory(run.context.get()), true);
        }
        for (auto & run : runs) {
            run.record.enabled = !o.trace.empty();
            run.prefill_ms = run.decode_ms = 0;
        }
        const auto initial_stats = llama_get_pager_metrics(model.get());
        const auto initial_io = io_counters();
        const double prefill_ms = parallel(o.contexts, prefill);
        const auto prefill_end = memory();
        const auto prefill_stats = llama_get_pager_metrics(model.get());
        const auto prefill_io = io_counters();
        for (auto & run : runs) run.decode_peak = prefill_end.resident;
        const double decode_ms = parallel(o.contexts, [&](int id) { decode(id, o.sample_decode_memory); });
        const auto decode_end = memory();
        const auto decode_stats = llama_get_pager_metrics(model.get());
        const auto decode_io = io_counters();
        const auto final_context_bytes = context_bytes();
        uint64_t decode_peak = decode_end.resident;
        out << ",\"prefill_ms\":" << prefill_ms << ",\"decode_ms\":" << decode_ms
            << ",\"prefill_tokens_per_s\":" << 1000.0 * o.prefill * o.contexts / prefill_ms
            << ",\"decode_tokens_per_s\":" << 1000.0 * o.decode * o.contexts / decode_ms
            << ",\"decode_ms_per_token_aggregate\":" << decode_ms / o.decode / o.contexts
            << ",\"workspace_bytes_after_decode\":" << final_context_bytes.compute;
        out << ",\"context_results\":[";
        for (int i = 0; i < o.contexts; ++i) {
            auto & run = runs[i];
            decode_peak = std::max(decode_peak, run.decode_peak);
            if (run.record.enabled) {
                if (!run.record.expert_ids || run.record.prefill_values != size_t(vocab) * o.prefill ||
                        run.record.decode_values != size_t(vocab) * o.decode) {
                    throw std::runtime_error("incomplete correctness trace");
                }
                run.record.ids.close();
                run.record.prefill.close();
                run.record.decode.close();
            }
            if (i) out << ',';
            out << "{\"expert_ids\":" << run.record.expert_ids << ",\"prefill_values\":" << run.record.prefill_values
                << ",\"decode_values\":" << run.record.decode_values << ",\"prefill_compute_ms\":" << run.prefill_ms
                << ",\"decode_compute_ms\":" << run.decode_ms << "}";
        }
        out << "],\"decode_memory_sampling\":" << (o.sample_decode_memory ? "true" : "false");
        if (o.sample_decode_memory) {
            out << ",\"decode_peak_sampled_resident_bytes\":" << decode_peak;
        }
        write_stats(out, "prefill_cache", initial_stats, prefill_stats);
        write_stats(out, "decode_cache", prefill_stats, decode_stats);
        write_memory(out, "after_prefill", prefill_end);
        write_memory(out, "after_inference", decode_end);
        write_io(out, "prefill_process_io", initial_io, prefill_io);
        write_io(out, "decode_process_io", prefill_io, decode_io);
    }
    runs.clear();
    model.reset();
    llama_backend_free();
    write_memory(out, "after_free", memory());
    out << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "--model PATH --output JSON [--pager off|single|lru] [--budget 64MiB]\n"
            "--threads N --ctx N --batch N --ubatch N --prefill N --decode N --contexts 1|2\n"
            "--io mmap|buffered --warmup N --cache-state LABEL [--trace-prefix PATH] [--sample-decode-memory]\n"
            "--plan estimates the allocation without inference; use --plan --pager lru --slots N for an exact budget.\n";
        return 0;
    }
    try {
        benchmark(parse(argc, argv));
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "bench-expert-paging: " << e.what() << '\n';
        return 1;
    }
}
