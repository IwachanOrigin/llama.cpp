#include "llama-expert-cache.h"
#include "ggml.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

// The standalone cache allocates one aligned block; observe that allocation in this test executable.
static std::atomic<size_t> aligned_blocks{0};
static std::atomic<size_t> aligned_allocations{0};

void * operator new(size_t bytes, std::align_val_t value) {
    const size_t alignment = static_cast<size_t>(value);
    const size_t overhead = sizeof(void *) + alignment - 1;
    if (bytes > SIZE_MAX - overhead) {
        throw std::bad_alloc();
    }
    void * raw = std::malloc(bytes + overhead);
    if (!raw) {
        throw std::bad_alloc();
    }
    void * aligned = static_cast<unsigned char *>(raw) + sizeof(void *);
    size_t space = bytes + alignment - 1;
    std::align(alignment, bytes, aligned, space);
    std::memcpy(static_cast<unsigned char *>(aligned) - sizeof(void *), &raw, sizeof(raw));
    ++aligned_blocks;
    ++aligned_allocations;
    return aligned;
}

void operator delete(void * aligned, std::align_val_t) noexcept {
    if (aligned) {
        void * raw;
        std::memcpy(&raw, static_cast<unsigned char *>(aligned) - sizeof(void *), sizeof(raw));
        std::free(raw);
        --aligned_blocks;
    }
}

void operator delete(void * aligned, size_t, std::align_val_t value) noexcept {
    ::operator delete(aligned, value);
}

namespace {

using namespace std::chrono_literals;
using cache = llama_expert_cache;
constexpr size_t expert_bytes = 97;
constexpr int experts_per_source = 8;
constexpr auto timeout = 8s;
using expert_data = std::array<unsigned char, expert_bytes>;

expert_data expected(cache::key key) {
    expert_data data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (key.tensor_source_id * 67 + key.expert_id * 19 + i * 3) % 256;
    }
    return data;
}

void verify(const cache::lease & lease, cache::key key) {
    const auto reference = expected(key);
    GGML_ASSERT(lease && lease.size() == reference.size());
    GGML_ASSERT(reinterpret_cast<uintptr_t>(lease.data()) % 64 == 0);
    GGML_ASSERT(std::memcmp(lease.data(), reference.data(), reference.size()) == 0);
}

template <typename T>
T get(std::future<T> & result) {
    GGML_ASSERT(result.wait_for(timeout) == std::future_status::ready);
    return result.get();
}

struct file_source {
    enum class outcome { success, short_read, exception };

    struct gate {
        cache::key expert{};
        bool enabled = false;
        bool entered = false;
        bool proceed = false;
        outcome result = outcome::success;
    };

    explicit file_source(const char * prefix) {
        for (size_t i = 0; i < files.size(); ++i) {
            paths[i] = std::string(prefix) + "-" + std::to_string(i) + ".bin";
            files[i] = std::fopen(paths[i].c_str(), "w+b");
            GGML_ASSERT(files[i]);
        }
        for (uint64_t source = 0; source < 3; ++source) {
            for (int id = 0; id < experts_per_source; ++id) {
                const auto data = expected({source, id});
                GGML_ASSERT(std::fwrite(data.data(), 1, data.size(), files[shard(source)]) == data.size());
            }
        }
        for (auto * file : files) {
            const auto partial = expected({0, 3});
            GGML_ASSERT(std::fwrite(partial.data(), 1, partial.size() / 2, file) == partial.size() / 2);
            GGML_ASSERT(std::fflush(file) == 0);
        }
    }

    ~file_source() {
        for (size_t i = 0; i < files.size(); ++i) {
            GGML_ASSERT(std::fclose(files[i]) == 0);
            GGML_ASSERT(std::remove(paths[i].c_str()) == 0);
        }
    }

    void pause(size_t index, cache::key expert, outcome result = outcome::success) {
        std::lock_guard<std::mutex> lock(gate_mutex);
        gates[index] = {expert, true, false, false, result};
    }

    void wait_for_read(size_t index) {
        std::unique_lock<std::mutex> lock(gate_mutex);
        GGML_ASSERT(gate_changed.wait_for(lock, timeout, [&] { return gates[index].entered; }));
    }

    void resume(size_t index) {
        std::lock_guard<std::mutex> lock(gate_mutex);
        gates[index].proceed = true;
        gate_changed.notify_all();
    }

    static bool read(void * context, cache::key expert, void * dst, size_t bytes) {
        auto & self = *static_cast<file_source *>(context);
        GGML_ASSERT(expert.tensor_source_id < 3 && expert.expert_id < experts_per_source && bytes == expert_bytes);
        outcome result = outcome::success;
        {
            std::unique_lock<std::mutex> lock(self.gate_mutex);
            for (auto & gate : self.gates) {
                if (gate.enabled && gate.expert.tensor_source_id == expert.tensor_source_id &&
                        gate.expert.expert_id == expert.expert_id) {
                    GGML_ASSERT(!gate.entered);
                    gate.entered = true;
                    self.gate_changed.notify_all();
                    GGML_ASSERT(self.gate_changed.wait_for(lock, timeout, [&] { return gate.proceed; }));
                    result = gate.result;
                    gate.enabled = false;
                    break;
                }
            }
        }
        const auto index = shard(expert.tensor_source_id);
        std::lock_guard<std::mutex> lock(self.file_mutexes[index]);
        const size_t row = result == outcome::success ?
            (expert.tensor_source_id == 1 ? experts_per_source : 0) + expert.expert_id :
            (index == 0 ? 2 : 1) * experts_per_source;
        auto * file = self.files[index];
        GGML_ASSERT(std::fseek(file, static_cast<long>(row * expert_bytes), SEEK_SET) == 0);
        const size_t received = std::fread(dst, 1, bytes, file);
        GGML_ASSERT(received == (result == outcome::success ? bytes : bytes / 2));
        if (result == outcome::exception) {
            throw std::runtime_error("injected reader exception");
        }
        return received == bytes;
    }

private:
    static size_t shard(uint64_t source) { return source == 2 ? 1 : 0; }
    std::array<std::string, 2> paths;
    std::array<FILE *, 2> files{};
    std::array<std::mutex, 2> file_mutexes;
    std::mutex gate_mutex;
    std::condition_variable gate_changed;
    std::array<gate, 2> gates{};
};

cache::lease acquire(cache & c, cache::key key) {
    return c.acquire(key, expert_bytes);
}

// Observe the wait registration under the cache mutex before unblocking a reader.
void wait_for_waiters(cache & c, size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (c.stats().waiting_requests < count) {
        GGML_ASSERT(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
}

void test_budget_and_lru(file_source & files) {
    const size_t minimum = cache::required_bytes(expert_bytes, 1);
    try {
        cache invalid(minimum - 1, expert_bytes, file_source::read, &files);
        GGML_ASSERT(false);
    } catch (const std::invalid_argument &) {}
    try {
        cache::required_bytes(SIZE_MAX, 1);
        GGML_ASSERT(false);
    } catch (const std::overflow_error &) {}

    for (size_t count : {1, 2, 4}) {
        const size_t budget = cache::required_bytes(expert_bytes, count);
        cache c(budget, expert_bytes, file_source::read, &files);
        cache just_below(cache::required_bytes(expert_bytes, count + 1) - 1, expert_bytes, file_source::read, &files);
        GGML_ASSERT(c.capacity() == count && just_below.capacity() == count);
        GGML_ASSERT(c.stats().peak_allocated_bytes == budget);
        GGML_ASSERT(just_below.stats().peak_allocated_bytes == budget);
        GGML_ASSERT(!c.acquire({0, 0}, expert_bytes + 1) && !acquire(c, {0, -1}));
        for (size_t i = 0; i < count; ++i) {
            const cache::key key{0, static_cast<int64_t>(i)};
            verify(acquire(c, key), key);
        }
        verify(acquire(c, {0, 0}), {0, 0}); // make Expert 0 newest
        verify(acquire(c, {1, 0}), {1, 0}); // same Expert ID, different tensor
        if (count > 1) {
            verify(acquire(c, {0, 0}), {0, 0});
        }
        const cache::key victim{0, count == 1 ? 0 : 1};
        verify(acquire(c, victim), victim);
        const auto stats = c.stats();
        GGML_ASSERT(stats.cache_hits == (count == 1 ? 1u : 2u));
        GGML_ASSERT(stats.cache_misses == count + 2 && stats.disk_reads == count + 2);
        GGML_ASSERT(stats.cache_evictions == 2 && stats.bytes_read == (count + 2) * expert_bytes);
        GGML_ASSERT(stats.pinned_slots == 0 && stats.peak_allocated_bytes <= budget);
        std::printf("slots=%zu budget=%zu hits=%llu misses=%llu evictions=%llu reads=%llu bytes=%llu\n",
                count, budget, (unsigned long long) stats.cache_hits, (unsigned long long) stats.cache_misses,
                (unsigned long long) stats.cache_evictions, (unsigned long long) stats.disk_reads,
                (unsigned long long) stats.bytes_read);
    }

    const size_t initial_blocks = aligned_blocks.load();
    const size_t initial_allocations = aligned_allocations.load();
    for (int repeat = 0; repeat < 128; ++repeat) {
        cache c(minimum, expert_bytes, file_source::read, &files);
        verify(acquire(c, {2, 7}), {2, 7});
        verify(acquire(c, {2, 7}), {2, 7});
        verify(acquire(c, {2, 6}), {2, 6});
    }
    GGML_ASSERT(aligned_blocks.load() == initial_blocks);
    GGML_ASSERT(aligned_allocations.load() - initial_allocations == 128);
}

void test_same_key_and_pins(file_source & files) {
    static_assert(!std::is_copy_constructible<cache::lease>::value, "leases must own a single pin");
    cache c(cache::required_bytes(expert_bytes, 1), expert_bytes, file_source::read, &files);
    const cache::key first{0, 0};
    const cache::key second{1, 0};
    files.pause(0, first);
    auto leader = std::async(std::launch::async, [&] { return acquire(c, first); });
    files.wait_for_read(0);
    auto follower = std::async(std::launch::async, [&] { return acquire(c, first); });
    wait_for_waiters(c, 1);
    auto displaced = std::async(std::launch::async, [&] { return acquire(c, second); });
    wait_for_waiters(c, 2);
    GGML_ASSERT(c.stats().disk_reads == 1 && c.stats().pinned_slots == 0);
    files.resume(0);
    auto a = get(leader);
    auto b = get(follower);
    verify(a, first);
    verify(b, first);
    GGML_ASSERT(a.data() == b.data() && c.stats().pinned_slots == 1);
    GGML_ASSERT(a.reset() && !a.reset() && !a.data()); // released on a different thread
    GGML_ASSERT(c.stats().pinned_slots == 1 && c.stats().disk_reads == 1);
    auto moved = std::move(b);
    GGML_ASSERT(!b && !b.reset());
    GGML_ASSERT(moved.reset());
    auto replacement = get(displaced);
    verify(replacement, second);
    GGML_ASSERT(!moved.reset() && c.stats().pinned_slots == 1);
    const auto stats = c.stats();
    GGML_ASSERT(stats.cache_hits == 1 && stats.cache_misses == 2 && stats.cache_evictions == 1);
    GGML_ASSERT(stats.disk_reads == 2 && stats.io_wait_ns > 0 && stats.cache_wait_ns > 0);
}

void test_distinct_keys(file_source & files) {
    cache c(cache::required_bytes(expert_bytes, 2), expert_bytes, file_source::read, &files);
    const cache::key first{0, 2};
    const cache::key second{1, 2};
    files.pause(0, first);
    files.pause(1, second);
    auto a = std::async(std::launch::async, [&] { return acquire(c, first); });
    files.wait_for_read(0);
    auto b = std::async(std::launch::async, [&] { return acquire(c, second); });
    files.wait_for_read(1); // both readers entered while neither read is allowed to finish
    GGML_ASSERT(c.stats().disk_reads == 2);
    files.resume(0);
    files.resume(1);
    auto pin_a = get(a);
    auto pin_b = get(b);
    verify(pin_a, first);
    verify(pin_b, second);
    GGML_ASSERT(pin_a.data() != pin_b.data() && c.stats().pinned_slots == 2);
    pin_a.reset();
    verify(acquire(c, {2, 2}), {2, 2});
    verify(pin_b, second); // LRU may only evict an unpinned slot
    GGML_ASSERT(c.stats().cache_evictions == 1);
}

void test_read_failure(file_source & files, file_source::outcome outcome) {
    cache c(cache::required_bytes(expert_bytes, 1), expert_bytes, file_source::read, &files);
    const cache::key key{0, 3};
    files.pause(0, key, outcome);
    auto leader = std::async(std::launch::async, [&] { return acquire(c, key); });
    files.wait_for_read(0);
    auto follower1 = std::async(std::launch::async, [&] { return acquire(c, key); });
    auto follower2 = std::async(std::launch::async, [&] { return acquire(c, key); });
    const cache::key different{2, 0};
    auto waiter_for_slot = std::async(std::launch::async, [&] { return acquire(c, different); });
    wait_for_waiters(c, 3);
    files.resume(0);
    GGML_ASSERT(!get(leader) && !get(follower1) && !get(follower2));
    verify(get(waiter_for_slot), different);
    auto stats = c.stats();
    GGML_ASSERT(stats.cache_hits == 0 && stats.disk_reads == 2 && stats.bytes_read == expert_bytes);
    GGML_ASSERT(stats.waiting_requests == 0 && stats.pinned_slots == 0);
    verify(acquire(c, key), key);
    stats = c.stats();
    GGML_ASSERT(stats.disk_reads == 3 && stats.bytes_read == 2 * expert_bytes && stats.cache_evictions == 1);
}

void test_tinylfu_reference(file_source & files) {
    const auto budget = cache::required_bytes(expert_bytes, 4, true);
    cache c(budget, expert_bytes, file_source::read, &files, true);
    GGML_ASSERT(c.capacity() == 4 && c.stats().peak_allocated_bytes == budget);
    GGML_ASSERT(cache::capacity_for_budget(budget - 1, expert_bytes, true) == 3);
    try {
        cache too_small(cache::required_bytes(expert_bytes, 2, true) - 1,
                        expert_bytes, file_source::read, &files, true);
        GGML_ASSERT(false);
    } catch (const std::invalid_argument &) {}
    struct trace_state {
        cache::decision outcome;
        uint64_t checksum = 14695981039346656037ULL;
    } trace;
    c.set_decision_trace([](void * context, const cache::decision & decision) noexcept {
        static_cast<trace_state *>(context)->outcome = decision;
    }, &trace);
    c.set_trace([](void * context, uint64_t, int64_t, size_t, bool hit, size_t) noexcept {
        auto & state = *static_cast<trace_state *>(context);
        const auto & outcome = state.outcome;
        for (auto value : {uint64_t(outcome.candidate_frequency), uint64_t(outcome.victim_frequency),
                outcome.victim.tensor_source_id, uint64_t(outcome.victim.expert_id),
                uint64_t(outcome.bypassed), uint64_t(outcome.aged), uint64_t(hit)}) {
            state.checksum = (state.checksum ^ value) * 1099511628211ULL;
        }
    }, &trace);

    for (int64_t index = 0; index < 256; ++index) {
        const cache::key initial[] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 0}};
        const cache::key key = index < 5 ? initial[index] : index % 5 == 0 ?
            cache::key{uint64_t((index / 7) % 3), (index * 11 + index / 3) % 8} :
            cache::key{0, index % 3};
        verify(acquire(c, key), key);
    }
    // Generated with Policy("tinylfu", 3, 0, ()) from simulate-pager-policies.py.
    GGML_ASSERT(trace.checksum == 7710516606872388577ULL);
    const auto policy = c.policy_stats();
    const auto stats = c.stats();
    GGML_ASSERT(policy.updates == 256 && policy.aging_count == 8 && policy.sketch_bytes == 64);
    GGML_ASSERT(policy.bypassed_misses && policy.admitted_misses > policy.resident_slots);
    GGML_ASSERT(policy.admitted_misses + policy.bypassed_misses == stats.disk_reads);
    GGML_ASSERT(stats.disk_reads + stats.cache_hits == policy.updates);
    GGML_ASSERT(stats.bytes_read == stats.disk_reads * expert_bytes && !policy.transient_in_use);
}

void test_tinylfu_transient(file_source & files) {
    const auto budget = cache::required_bytes(expert_bytes, 2, true);
    cache c(budget, expert_bytes, file_source::read, &files, true);
    const cache::key resident{0, 0}, rejected{1, 0}, other{2, 0};
    verify(acquire(c, resident), resident);
    auto resident_pin = acquire(c, resident);
    auto pending = std::async(std::launch::async, [&] { return acquire(c, rejected); });
    wait_for_waiters(c, 1);
    GGML_ASSERT(!c.policy_stats().transient_in_use && c.stats().disk_reads == 1);
    resident_pin.reset();
    auto transient = get(pending);
    verify(transient, rejected);
    verify(acquire(c, resident), resident);
    auto competing = std::async(std::launch::async, [&] { return acquire(c, other); });
    wait_for_waiters(c, 1);
    verify(transient, rejected);
    GGML_ASSERT(c.policy_stats().transient_in_use && c.stats().disk_reads == 2);
    transient.reset();
    verify(get(competing), other);
    GGML_ASSERT(c.policy_stats().transient_waits && !c.policy_stats().transient_in_use);
    GGML_ASSERT(c.stats().cache_evictions == 0);

    const cache::key failing{1, 2};
    files.pause(0, failing, file_source::outcome::short_read);
    auto failure = std::async(std::launch::async, [&] { return acquire(c, failing); });
    files.wait_for_read(0);
    auto follower = std::async(std::launch::async, [&] { return acquire(c, {2, 2}); });
    wait_for_waiters(c, 1);
    files.resume(0);
    GGML_ASSERT(!get(failure));
    verify(get(follower), {2, 2});
    verify(acquire(c, resident), resident);
    GGML_ASSERT(!c.policy_stats().transient_in_use && c.stats().pinned_slots == 0);
}

} // namespace

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 2);
    file_source files(argv[1]);
    test_budget_and_lru(files);
    test_tinylfu_reference(files);
    test_tinylfu_transient(files);
    test_same_key_and_pins(files);
    test_distinct_keys(files);
    test_read_failure(files, file_source::outcome::short_read);
    test_read_failure(files, file_source::outcome::exception);
    std::puts("same-key, parallel reads, pinned waits, failed reads and retries passed");
}
