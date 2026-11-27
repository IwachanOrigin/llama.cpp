#pragma once

#include "llama-mmap.h"
#include "gguf.h"

#include <atomic>
#include <mutex>

namespace reader_bench {
using json = nlohmann::json;

struct request {
    size_t shard, offset, size;
    uint64_t at;
    std::string phase;
};

struct measurement {
    uint64_t queue = 0, service = 0, end_to_end = 0, api = 0, copy = 0, mutex_wait = 0;
    uint64_t started = 0, finished = 0;
    bool api_measured = false, pending = false;
};

inline uint64_t ns(steady_clock::time_point start, steady_clock::time_point end = steady_clock::now()) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

inline json distribution(std::vector<uint64_t> values) {
    if (values.empty()) return nullptr;
    std::sort(values.begin(), values.end());
    double total = 0, squared = 0;
    for (auto value : values) { total += value; squared += double(value) * value; }
    const auto percentile = [&](size_t p) { return values[(p * values.size() + 99) / 100 - 1]; };
    const auto mean = total / values.size();
    return {{"min", values.front()}, {"p50", percentile(50)}, {"p90", percentile(90)},
        {"p95", percentile(95)}, {"p99", percentile(99)}, {"max", values.back()},
        {"mean", mean}, {"stddev", std::sqrt(std::max(0.0, squared / values.size() - mean * mean))}};
}

#ifdef _WIN32
struct handle {
    HANDLE value;
    explicit handle(HANDLE value) : value(value) {
        require(value && value != INVALID_HANDLE_VALUE, "cannot create Reader handle");
    }
    ~handle() { CloseHandle(value); }
    handle(const handle &) = delete;
    handle & operator=(const handle &) = delete;
};

inline void overlapped_read(HANDLE file, HANDLE event, const request & r, void * buffer,
                            measurement & m, uint32_t timeout_ms) {
    require(ResetEvent(event) != 0, "cannot reset OVERLAPPED event");
    OVERLAPPED op{};
    op.Offset = DWORD(r.offset);
    op.OffsetHigh = DWORD(uint64_t(r.offset) >> 32);
    op.hEvent = event;
    DWORD got = 0;
    const auto start = steady_clock::now();
    if (!ReadFile(file, buffer, DWORD(r.size), nullptr, &op)) {
        const DWORD error = GetLastError();
        require(error == ERROR_IO_PENDING, "OVERLAPPED ReadFile failed or EOF");
        m.pending = true;
        if (WaitForSingleObject(event, timeout_ms) != WAIT_OBJECT_0) {
            // Cancellation can race with ordinary completion. Drain either outcome.
            CancelIoEx(file, &op);
            GetOverlappedResult(file, &op, &got, TRUE);
            throw std::runtime_error("OVERLAPPED timeout or wait failure; request drained");
        }
    }
    require(GetOverlappedResult(file, &op, &got, TRUE) != 0, "OVERLAPPED completion failed");
    require(got == r.size, "OVERLAPPED short read");
    m.api = ns(start);
    m.api_measured = true;
}
#endif

inline int run(int argc, char ** argv) {
    require(argc == 12 || argc == 13,
        "reader-bench MANIFEST shared|independent|overlapped|mapping READERS latency|throughput|paced LIMIT WARMUP uncontrolled|warm-repeat|warm-subset verify|measure OUTPUT TIMEOUT_MS [DETAIL_LIMIT]");
    const std::string mode = argv[3], workload = argv[5], cache = argv[8], verification = argv[9];
    const uint32_t lanes = number(argv[4], 8), limit = number(argv[6], 200000);
    const uint32_t warmup = number(argv[7], 200000), timeout_ms = number(argv[11], 600000);
    const uint32_t detail_limit = argc == 13 ? number(argv[12], 200000) : 0;
    require(lanes && (mode == "shared" || mode == "independent" || mode == "overlapped" || mode == "mapping"),
            "invalid Reader mode or count");
    require(workload == "throughput" || workload == "paced" || (workload == "latency" && lanes == 1), "invalid workload/count");
    require(cache == "uncontrolled" || ((cache == "warm-repeat" || cache == "warm-subset") && warmup), "invalid cache label/warmup");
    require(verification == "verify" || verification == "measure", "invalid verification mode");
#ifndef _WIN32
    require(mode != "overlapped", "OVERLAPPED requires Windows");
#endif

    const auto start_init = steady_clock::now();
    const auto manifest_path = std::filesystem::u8path(argv[2]);
    require(std::filesystem::file_size(manifest_path) <= (64u << 20), "manifest exceeds 64 MiB");
    std::ifstream manifest_stream(manifest_path, std::ios::binary);
    std::string manifest_bytes((std::istreambuf_iterator<char>(manifest_stream)), {});
    const auto manifest = json::parse(manifest_bytes);
    unsigned char digest[32];
    sha256_hash(digest, reinterpret_cast<const uint8_t *>(manifest_bytes.data()), manifest_bytes.size());
    std::ostringstream hash;
    for (auto byte : digest) hash << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    require(manifest.at("version") == 1, "unknown Reader manifest version");
    const auto & shards = manifest.at("shards");
    const auto & rows = manifest.at("requests");
    require(shards.is_array() && !shards.empty() && shards.size() <= 64 &&
            rows.is_array() && !rows.empty() && rows.size() <= 200000, "invalid Reader manifest size");
    std::vector<std::string> paths;
    std::vector<uint64_t> sizes;
    using metadata_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    std::vector<metadata_ptr> metadata;
    for (const auto & shard : shards) {
        paths.push_back(shard.at("path").get<std::string>());
        sizes.push_back(std::filesystem::file_size(std::filesystem::u8path(paths.back())));
        require(shard.at("size") == sizes.back(), "manifest shard size mismatch");
        metadata.emplace_back(gguf_init_from_file(paths.back().c_str(), {true, nullptr}), gguf_free);
        require(bool(metadata.back()), "invalid GGUF metadata");
    }
    const auto unsigned_field = [](const json & object, const char * key) {
        const auto & value = object.at(key);
        require(value.is_number_unsigned(), "expected unsigned Reader manifest field");
        return value.get<uint64_t>();
    };
    std::vector<request> requests;
    size_t maximum = 0;
    uint64_t bytes = 0, previous_ns = 0;
    for (size_t i = 0; i < rows.size() && (!limit || i < limit); ++i) {
        const auto & row = rows[i];
        const auto shard = unsigned_field(row, "shard"), offset = unsigned_field(row, "offset"), size = unsigned_field(row, "size");
        require(shard < sizes.size() && offset <= sizes[shard] && offset <= SIZE_MAX &&
                size && size <= (16u << 20) && size <= sizes[shard] - offset, "Reader request outside shard");
        const auto name = row.at("tensor").get<std::string>();
        const auto * file = metadata[shard].get();
        const auto id = gguf_find_tensor(file, name.c_str());
        require(id >= 0 && (name.find(".ffn_gate_exps.weight") != std::string::npos ||
                           name.find(".ffn_up_exps.weight") != std::string::npos ||
                           name.find(".ffn_down_exps.weight") != std::string::npos), "invalid Expert tensor");
        const auto expert = unsigned_field(row, "expert");
        const auto * shape = gguf_get_tensor_ne(file, id);
        require(shape[2] > 0 && shape[3] == 1 && expert < uint64_t(shape[2]), "invalid Expert ID");
        const uint64_t tensor_bytes = gguf_get_tensor_size(file, id);
        const uint64_t data_start = gguf_get_data_offset(file), relative = gguf_get_tensor_offset(file, id);
        require(data_start <= sizes[shard] && relative <= sizes[shard] - data_start &&
                tensor_bytes <= sizes[shard] - data_start - relative &&
                tensor_bytes % shape[2] == 0 && size == tensor_bytes / shape[2] &&
                offset == data_start + relative + expert * size, "invalid GGUF Expert slice");
        const auto at = row.at("relative_ns").is_null() && workload != "paced" ? 0 : unsigned_field(row, "relative_ns");
        require(workload != "paced" || (at >= previous_ns && at <= 600000000000ull), "invalid paced schedule");
        requests.push_back({size_t(shard), size_t(offset), size_t(size), at, row.at("phase").get<std::string>()});
        previous_ns = at;
        maximum = std::max(maximum, size_t(size));
        bytes += size;
    }
    metadata.clear();
    const auto open_start = steady_clock::now();
    const auto new_files = [&] {
        llama_files result;
        for (const auto & name : paths) result.push_back(std::make_unique<llama_file>(name.c_str(), "rb"));
        return result;
    };
    std::vector<llama_files> files;
    if (mode == "shared" || mode == "mapping") files.push_back(new_files());
    if (mode == "independent") for (size_t i = 0; i < lanes; ++i) files.push_back(new_files());
    std::vector<std::mutex> locks(paths.size());
    std::vector<llama_files> reference;
    if (verification == "verify") for (size_t i = 0; i < lanes; ++i) reference.push_back(new_files());
#ifdef _WIN32
    std::vector<std::unique_ptr<handle>> async_files, events;
    if (mode == "overlapped") {
        for (const auto & name : paths) {
            async_files.push_back(std::make_unique<handle>(CreateFileW(std::filesystem::u8path(name).c_str(),
                GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr)));
        }
        for (size_t i = 0; i < lanes; ++i) events.push_back(std::make_unique<handle>(CreateEventW(nullptr, TRUE, FALSE, nullptr)));
    }
#endif
    const auto open_ns = ns(open_start);
    const auto map_start = steady_clock::now();
    llama_mmaps mappings;
    uint64_t mapped_bytes = 0;
    if (mode == "mapping") {
        require(llama_mmap::SUPPORTED, "mmap unavailable");
        for (auto & file : files.front()) {
            mappings.push_back(std::make_unique<llama_mmap>(file.get(), 0));
            mapped_bytes += mappings.back()->size();
        }
    }
    const auto map_ns = ns(map_start);
    const auto alloc_start = steady_clock::now();
    const size_t stride = (maximum + 63) & ~size_t(63);
    const size_t buffer_count = lanes * (verification == "verify" ? 2 : 1);
    std::vector<uint8_t> storage(stride * buffer_count + 63);
    auto * aligned = storage.data() + ((64 - reinterpret_cast<uintptr_t>(storage.data()) % 64) % 64);
    const auto alloc_ns = ns(alloc_start);
    std::vector<measurement> samples(requests.size());
    uint64_t startup_ns = 0;
    auto execute = [&](size_t count, bool measured) {
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::atomic<uint32_t> active{0}, peak{0};
        std::mutex failure_mutex;
        std::exception_ptr failure;
        std::mutex start_mutex;
        std::condition_variable started;
        bool go = false;
        auto origin = steady_clock::now();
        std::vector<std::thread> threads;
        const auto worker = [&](size_t lane) {
            {
                std::unique_lock<std::mutex> lock(start_mutex);
                started.wait(lock, [&] { return go; });
            }
            auto * dest = aligned + lane * stride;
            try {
                while (!failed.load()) {
                    const auto index = next.fetch_add(1);
                    if (index >= count) break;
                    const auto & r = requests[index % requests.size()];
                    measurement m;
                    auto issued = origin;
                    if (measured && workload == "paced") {
                        issued += std::chrono::nanoseconds(r.at - requests.front().at);
                        std::this_thread::sleep_until(issued);
                    }
                    const auto begin = steady_clock::now();
                    if (workload == "latency") issued = begin;
                    m.queue = ns(issued, begin);
                    const auto outstanding = active.fetch_add(1) + 1;
                    auto max_active = peak.load();
                    while (max_active < outstanding && !peak.compare_exchange_weak(max_active, outstanding)) {}
                    if (mode == "mapping") {
                        std::memcpy(dest, static_cast<const uint8_t *>(mappings[r.shard]->addr()) + r.offset, r.size);
                        m.copy = ns(begin);
                    } else if (mode == "shared" || mode == "independent") {
                        llama_file_read_sample io{};
                        std::unique_lock<std::mutex> lock(locks[r.shard], std::defer_lock);
                        if (mode == "shared") { lock.lock(); m.mutex_wait = ns(begin); }
                        files[mode == "shared" ? 0 : lane][r.shard]->read_exact_at(r.offset, dest, r.size, &io);
                        m.api = io.read_api_ns;
                        m.api_measured = io.read_api_measured;
                    }
#ifdef _WIN32
                    else overlapped_read(async_files[r.shard]->value, events[lane]->value, r, dest, m, timeout_ms);
#endif
                    const auto end = steady_clock::now();
                    active.fetch_sub(1);
                    m.service = ns(begin, end);
                    m.started = ns(origin, begin);
                    m.finished = ns(origin, end);
                    m.end_to_end = ns(issued, end);
                    if (!reference.empty()) {
                        auto * expected = aligned + (lanes + lane) * stride;
                        reference[lane][r.shard]->read_exact_at(r.offset, expected, r.size);
                        require(std::memcmp(expected, dest, r.size) == 0, "Reader data differs from current Reader");
                    }
                    if (measured) samples[index] = m;
                }
            } catch (...) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
            }
        };
        try {
            for (size_t i = 0; i < lanes; ++i) threads.emplace_back(worker, i);
        } catch (...) {
            failed.store(true);
            failure = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(start_mutex);
            startup_ns = ns(origin);
            origin = steady_clock::now();
            go = true;
        }
        started.notify_all();
        for (auto & thread : threads) thread.join();
        if (failure) std::rethrow_exception(failure);
        return std::make_pair(ns(origin), peak.load());
    };
    const auto warmup_ns = warmup ? execute(warmup, false).first : 0;
#ifdef _WIN32
    FILETIME a{}, b{}, kernel_before{}, user_before{}, kernel_after{}, user_after{};
    IO_COUNTERS io_before{}, io_after{};
    const bool process_io_available = GetProcessIoCounters(GetCurrentProcess(), &io_before) != 0;
    const auto ft = [](FILETIME x) { return (uint64_t(x.dwHighDateTime) << 32) | x.dwLowDateTime; };
    const bool cpu_available = GetProcessTimes(GetCurrentProcess(), &a, &b, &kernel_before, &user_before);
#endif
    const auto cpu_start = std::clock();
    const auto [wall, peak] = execute(requests.size(), true);
    const double cpu_ns = double(std::clock() - cpu_start) * (1e9 / CLOCKS_PER_SEC);
    json cpu = nullptr, process_io = nullptr;
#ifndef _WIN32
    cpu = cpu_ns;
#else
    if (cpu_available && GetProcessTimes(GetCurrentProcess(), &a, &b, &kernel_after, &user_after)) {
        cpu = {{"user_ns", (ft(user_after) - ft(user_before)) * 100},
               {"kernel_ns", (ft(kernel_after) - ft(kernel_before)) * 100}};
    }
    if (process_io_available && GetProcessIoCounters(GetCurrentProcess(), &io_after)) {
        process_io = {{"read_operations", io_after.ReadOperationCount - io_before.ReadOperationCount},
            {"read_transfer_bytes", io_after.ReadTransferCount - io_before.ReadTransferCount}};
    }
    (void) cpu_ns; // MS CRT clock measures elapsed time.
#endif
    std::vector<uint64_t> service, latency, queue, api, copy, mutex_wait;
    json detail = json::array();
    size_t pending = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto & m = samples[i];
        service.push_back(m.service); latency.push_back(m.end_to_end); queue.push_back(m.queue);
        if (m.api_measured) api.push_back(m.api);
        if (mode == "mapping") copy.push_back(m.copy);
        if (mode == "shared") mutex_wait.push_back(m.mutex_wait);
        pending += m.pending;
        if (i < detail_limit) detail.push_back({{"index", i}, {"phase", requests[i].phase}, {"size", requests[i].size},
            {"service_ns", m.service}, {"latency_ns", m.end_to_end}, {"queue_ns", m.queue},
            {"started_ns", m.started}, {"finished_ns", m.finished},
            {"file_api_ns", m.api_measured ? json(m.api) : json(nullptr)}, {"pending", m.pending}});
    }
    json report = {{"version", 1}, {"status", "ok"}, {"revision", STAGE_BUILD_COMMIT},
#ifdef _WIN32
        {"os", "windows"},
#else
        {"os", "linux"},
#endif
#if defined(__x86_64__) || defined(_M_X64)
        {"binary_arch", "x86_64"},
#elif defined(__aarch64__) || defined(_M_ARM64)
        {"binary_arch", "arm64"},
#endif
        {"manifest_sha256", hash.str()}, {"inputs", manifest.at("inputs")}, {"mode", mode},
        {"workload", workload}, {"readers", lanes}, {"queue_depth", lanes}, {"peak_active_requests", peak}, {"timeout_ms", timeout_ms},
        {"requests", requests.size()}, {"requested_bytes", bytes}, {"completed_bytes", bytes},
        {"cache_state", cache}, {"warmup_requests", warmup}, {"warmup_ns", warmup_ns},
        {"verified", !reference.empty()}, {"pending_completions", mode == "overlapped" ? json(pending) : json(nullptr)},
        {"errors", 0}, {"buffer_count", buffer_count}, {"buffer_stride", stride}, {"buffer_alignment", 64},
        {"buffer_bytes", storage.size()}, {"mapped_bytes", mapped_bytes}, {"open_ns", open_ns},
        {"map_ns", map_ns}, {"buffer_init_ns", alloc_ns}, {"thread_start_ns", startup_ns},
        {"initialization_and_trial_ns", ns(start_init)}, {"wall_ns", wall},
        {"requests_per_second", requests.size() * 1e9 / wall}, {"mib_per_second", bytes * 1e9 / wall / 1048576},
        {"service_ns", distribution(std::move(service))}, {"latency_ns", distribution(std::move(latency))},
        {"queue_ns", distribution(std::move(queue))}, {"file_api_ns", distribution(std::move(api))},
        {"copy_ns", distribution(std::move(copy))}, {"mutex_wait_ns", distribution(std::move(mutex_wait))},
        {"cpu", cpu}, {"process_memory", stage_performance::process_memory()}, {"detail_limit", detail_limit},
        {"detail_dropped", detail_limit ? samples.size() - detail.size() : 0}, {"detail", std::move(detail)}};
    report["process_io"] = process_io;
    report["sample_metadata_bytes"] = samples.capacity() * sizeof(measurement);
    report["file_handles"] = paths.size() * (mode == "independent" ? lanes : 1);
    report["reference_file_handles"] = reference.size() * paths.size();
    report["mappings"] = mappings.size();
    report["first_service_ns"] = samples.front().service;
#ifdef _WIN32
    SYSTEM_INFO host{};
    GetNativeSystemInfo(&host);
    report["windows_native_architecture"] = host.wProcessorArchitecture;
    USHORT process_machine = 0, native_machine = 0;
    using machine_info = BOOL (WINAPI *)(HANDLE, USHORT *, USHORT *);
    const auto query = reinterpret_cast<machine_info>(reinterpret_cast<void (*)()>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "IsWow64Process2")));
    if (query && query(GetCurrentProcess(), &process_machine, &native_machine)) {
        report["windows_native_machine"] = native_machine;
    }
    PROCESS_MEMORY_COUNTERS memory{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) {
        report["process_page_faults_lifetime"] = memory.PageFaultCount;
    }
#endif
    std::ofstream out(std::filesystem::u8path(argv[10]), std::ios::binary);
    out << report.dump(2) << '\n';
    require(bool(out), "cannot write Reader result");
    return 0;
}
} // namespace reader_bench
