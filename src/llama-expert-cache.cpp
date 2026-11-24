#include "llama-expert-cache.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

constexpr size_t slot_alignment = 64;
using cache_clock = std::chrono::steady_clock;

size_t aligned_size(size_t size) {
    if (size > SIZE_MAX - (slot_alignment - 1)) {
        throw std::overflow_error("expert cache size overflow");
    }
    return (size + slot_alignment - 1) & ~(slot_alignment - 1);
}

uint64_t elapsed_ns(cache_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(cache_clock::now() - start).count();
}

size_t sketch_width(size_t residents) {
    if (residents > SIZE_MAX / 16) {
        throw std::overflow_error("expert cache sketch size overflow");
    }
    size_t width = 16;
    while (width < 4 * residents) {
        if (width > SIZE_MAX / 8) {
            throw std::overflow_error("expert cache sketch size overflow");
        }
        width *= 2;
    }
    return width;
}

uint64_t rotate_right(uint64_t value, unsigned shift) {
    return (value >> shift) | (value << (64 - shift));
}

// BLAKE2b-128 for exactly two little-endian uint64 words, matching Python hashlib.
std::array<uint64_t, 2> expert_hash(llama_expert_cache::key key) {
    constexpr uint64_t iv[] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    };
    constexpr unsigned sigma[10][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
        {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
        {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
        {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
        {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
        {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
        {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
        {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
        {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
    };
    const uint64_t message[16] = {key.tensor_source_id, uint64_t(key.expert_id)};
    uint64_t state[16];
    std::copy_n(iv, 8, state);
    std::copy_n(iv, 8, state + 8);
    const uint64_t initial = iv[0] ^ 0x01010010;
    state[0] = initial;
    state[12] ^= 16;
    state[14] = ~state[14];
    for (unsigned round = 0; round < 12; ++round) {
        const auto & order = sigma[round % 10];
        const auto mix = [&](unsigned a, unsigned b, unsigned c, unsigned d, unsigned index) {
            state[a] += state[b] + message[order[index]];
            state[d] = rotate_right(state[d] ^ state[a], 32);
            state[c] += state[d];
            state[b] = rotate_right(state[b] ^ state[c], 24);
            state[a] += state[b] + message[order[index + 1]];
            state[d] = rotate_right(state[d] ^ state[a], 16);
            state[c] += state[d];
            state[b] = rotate_right(state[b] ^ state[c], 63);
        };
        for (unsigned column = 0; column < 4; ++column) {
            mix(column, 4 + column, 8 + column, 12 + column, 2 * column);
        }
        for (unsigned column = 0; column < 4; ++column) {
            mix(column, 4 + (column + 1) % 4, 8 + (column + 2) % 4,
                12 + (column + 3) % 4, 8 + 2 * column);
        }
    }
    return {initial ^ state[0] ^ state[8], iv[1] ^ state[1] ^ state[9]};
}

} // namespace

struct llama_expert_cache::slot {
    enum class state { empty, loading, ready, failed };

    key expert{};
    state status = state::empty;
    unsigned char * data = nullptr;
    size_t bytes = 0;
    size_t references = 0;
    size_t waiters = 0;
    uint64_t generation = 0;
    slot * older = nullptr;
    slot * newer = nullptr;
};

struct llama_expert_cache::admission {
    policy_statistics stats;
    size_t width, period;
    size_t observations = 0;
    unsigned char * cells;
    decision_fn trace = nullptr;
    void * trace_context = nullptr;

    unsigned frequency(key expert, bool increment) {
        const auto hash = expert_hash(expert);
        unsigned estimate = 15;
        for (size_t row = 0; row < 4; ++row) {
            const uint32_t word = uint32_t(hash[row / 2] >> (32 * (row % 2)));
            auto & cell = cells[row * width + (word & (width - 1))];
            if (increment && cell < 15) {
                ++cell;
            }
            estimate = std::min(estimate, unsigned(cell));
        }
        return estimate;
    }

    unsigned observe(key expert) {
        if (observations == period) {
            for (size_t index = 0; index < 4 * width; ++index) {
                cells[index] /= 2;
            }
            observations = 0;
            ++stats.aging_count;
        }
        ++observations;
        ++stats.updates;
        return frequency(expert, true);
    }
};

size_t llama_expert_cache::required_bytes(size_t max_expert_bytes, size_t count, bool tinylfu) {
    if (!max_expert_bytes || !count) {
        throw std::invalid_argument("expert cache needs nonzero Expert size and slot count");
    }
    const size_t stride = aligned_size(max_expert_bytes);
    if (count > SIZE_MAX / sizeof(slot)) {
        throw std::overflow_error("expert cache metadata size overflow");
    }
    const size_t metadata = aligned_size(count * sizeof(slot));
    if (metadata > SIZE_MAX - sizeof(llama_expert_cache) ||
            count > (SIZE_MAX - sizeof(llama_expert_cache) - metadata) / stride) {
        throw std::overflow_error("expert cache capacity overflow");
    }
    const size_t base = sizeof(llama_expert_cache) + metadata + count * stride;
    if (!tinylfu) {
        return base;
    }
    if (count < 2) {
        throw std::invalid_argument("TinyLFU needs resident and transient Expert slots");
    }
    const size_t sketch = 4 * sketch_width(count - 1);
    if (base > SIZE_MAX - aligned_size(sizeof(admission)) ||
            sketch > SIZE_MAX - base - aligned_size(sizeof(admission))) {
        throw std::overflow_error("expert cache sketch allocation overflow");
    }
    return base + aligned_size(sizeof(admission)) + sketch;
}

size_t llama_expert_cache::capacity_for_budget(size_t budget, size_t max_expert_bytes, bool tinylfu) {
    if (tinylfu) {
        size_t upper = capacity_for_budget(budget, max_expert_bytes, false);
        size_t lower = 2;
        if (upper < lower || required_bytes(max_expert_bytes, lower, true) > budget) {
            throw std::invalid_argument("TinyLFU budget cannot fit resident and transient slots plus sketch");
        }
        while (lower < upper) {
            const size_t count = lower + (upper - lower + 1) / 2;
            bool fits;
            try {
                fits = required_bytes(max_expert_bytes, count, true) <= budget;
            } catch (const std::overflow_error &) {
                fits = false;
            }
            if (fits) {
                lower = count;
            } else {
                upper = count - 1;
            }
        }
        return lower;
    }
    if (budget < required_bytes(max_expert_bytes, 1)) {
        throw std::invalid_argument("expert cache budget must fit at least one slot and its metadata");
    }
    const size_t stride = aligned_size(max_expert_bytes);
    const size_t available = budget - sizeof(llama_expert_cache);
    size_t lower = 1;
    size_t upper = available / stride;
    while (lower < upper) {
        const size_t count = lower + (upper - lower + 1) / 2;
        const size_t metadata_budget = available - count * stride;
        const size_t aligned_budget = metadata_budget & ~(slot_alignment - 1);
        if (count <= aligned_budget / sizeof(slot)) {
            lower = count;
        } else {
            upper = count - 1;
        }
    }
    return lower;
}

llama_expert_cache::llama_expert_cache(size_t budget, size_t max_expert_bytes, read_fn reader, void * reader_context, bool tinylfu) :
    reader(reader), reader_context(reader_context), max_expert_bytes(max_expert_bytes) {
    if (!reader) {
        throw std::invalid_argument("expert cache needs a reader");
    }
    slot_count = capacity_for_budget(budget, max_expert_bytes, tinylfu);
    const size_t stride = aligned_size(max_expert_bytes);
    static_assert(alignof(slot) <= slot_alignment, "slot metadata must fit in the aligned allocation");
    static_assert(std::is_trivially_destructible<slot>::value, "slot destruction must not require a loop");
    counters.peak_allocated_bytes = required_bytes(max_expert_bytes, slot_count, tinylfu);
    auto * allocation = ::operator new(counters.peak_allocated_bytes - sizeof(*this), std::align_val_t(slot_alignment));
    // The standard non-allocating placement array form has zero overhead (CWG 2382).
    slots = ::new (allocation) slot[slot_count]();
    auto * data = static_cast<unsigned char *>(allocation) + aligned_size(slot_count * sizeof(slot));
    for (size_t i = 0; i < slot_count; ++i) {
        slots[i].data = data + i * stride;
    }
    if (tinylfu) {
        auto * storage = data + slot_count * stride;
        frequency = ::new (storage) admission{};
        frequency->width = sketch_width(slot_count - 1);
        frequency->period = std::max(size_t(16), 10 * (slot_count - 1));
        frequency->cells = storage + aligned_size(sizeof(admission));
        auto & stats = frequency->stats;
        stats.resident_slots = slot_count - 1;
        stats.transient_slots = 1;
        stats.sketch_bytes = 4 * frequency->width;
        stats.management_bytes = aligned_size(sizeof(admission));
        stats.slot_stride = stride;
        std::fill_n(frequency->cells, stats.sketch_bytes, 0);
    }
}

llama_expert_cache::~llama_expert_cache() {
    assert(counters.pinned_slots == 0 && counters.waiting_requests == 0);
#ifndef NDEBUG
    for (size_t i = 0; i < slot_count; ++i) {
        assert(slots[i].status != slot::state::loading);
    }
#endif
    ::operator delete(slots, std::align_val_t(slot_alignment));
}

llama_expert_cache::lease::lease(llama_expert_cache & owner, slot & storage) :
    owner(&owner), storage(&storage), bytes(storage.data), length(storage.bytes), generation(storage.generation) {}

llama_expert_cache::lease::~lease() {
    reset();
}

llama_expert_cache::lease::lease(lease && other) noexcept {
    *this = std::move(other);
}

llama_expert_cache::lease & llama_expert_cache::lease::operator=(lease && other) noexcept {
    if (this != &other) {
        reset();
        owner = std::exchange(other.owner, nullptr);
        storage = std::exchange(other.storage, nullptr);
        bytes = std::exchange(other.bytes, nullptr);
        length = std::exchange(other.length, 0);
        generation = std::exchange(other.generation, 0);
    }
    return *this;
}

bool llama_expert_cache::lease::reset() noexcept {
    if (!owner) {
        return false;
    }
    auto * cache = std::exchange(owner, nullptr);
    auto * pinned = std::exchange(storage, nullptr);
    bytes = nullptr;
    length = 0;
    return cache->release(*pinned, std::exchange(generation, 0));
}

void llama_expert_cache::unlink(slot & storage) {
    if (storage.older) {
        storage.older->newer = storage.newer;
    } else if (oldest == &storage) {
        oldest = storage.newer;
    }
    if (storage.newer) {
        storage.newer->older = storage.older;
    } else if (newest == &storage) {
        newest = storage.older;
    }
    storage.older = storage.newer = nullptr;
}

void llama_expert_cache::touch(slot & storage) {
    if (newest == &storage) {
        return;
    }
    unlink(storage);
    storage.older = newest;
    if (newest) {
        newest->newer = &storage;
    } else {
        oldest = &storage;
    }
    newest = &storage;
}

llama_expert_cache::lease llama_expert_cache::pin(slot & storage) {
    if (storage.references == SIZE_MAX) {
        return {};
    }
    if (storage.references++ == 0) {
        ++counters.pinned_slots;
    }
    if (frequency && &storage == slots + slot_count - 1) {
        frequency->stats.peak_transient_in_use = 1;
    } else {
        touch(storage);
        if (frequency) {
            const size_t resident_pins = counters.pinned_slots - (slots[slot_count - 1].references != 0);
            frequency->stats.peak_resident_in_use = std::max(frequency->stats.peak_resident_in_use, resident_pins);
        }
    }
    return lease(*this, storage);
}

bool llama_expert_cache::release(slot & storage, uint64_t generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (storage.status != slot::state::ready || storage.generation != generation || !storage.references) {
        return false;
    }
    if (--storage.references == 0) {
        --counters.pinned_slots;
        if (frequency && &storage == slots + slot_count - 1) {
            storage.status = slot::state::empty;
        }
        changed.notify_all();
    }
    return true;
}

llama_expert_cache::lease llama_expert_cache::acquire(key expert, size_t bytes) {
    if (expert.expert_id < 0 || !bytes || bytes > max_expert_bytes) {
        return {};
    }
    std::unique_lock<std::mutex> lock(mutex);
    const size_t residents = slot_count - (frequency != nullptr);
    const auto wait = [&](bool for_capacity, bool for_transient = false) {
        const auto start = cache_clock::now();
        ++counters.waiting_requests;
        changed.wait(lock);
        --counters.waiting_requests;
        const auto duration = elapsed_ns(start);
        counters.cache_wait_ns += duration;
        if (for_transient) {
            ++frequency->stats.transient_waits;
            frequency->stats.transient_wait_ns += duration;
        }
        if (for_capacity) {
            counters.capacity_wait_ns += duration;
        }
    };

    slot * selected = nullptr;
    decision outcome;
    for (;;) {
        slot * empty = nullptr;
        for (size_t i = 0; i < residents; ++i) {
            auto & current = slots[i];
            if (current.status == slot::state::empty) {
                if (!empty) {
                    empty = &current;
                }
                continue;
            }
            if (current.expert.tensor_source_id != expert.tensor_source_id || current.expert.expert_id != expert.expert_id) {
                continue;
            }
            if (current.bytes != bytes) {
                return {};
            }
            if (current.status == slot::state::loading) {
                ++current.waiters;
                do {
                    wait(false);
                } while (current.status == slot::state::loading);
                --current.waiters;
                if (current.status == slot::state::failed && !current.waiters) {
                    current.status = slot::state::empty;
                    changed.notify_all();
                    return {};
                }
            }
            if (current.status != slot::state::ready) {
                return {};
            }
            if (frequency) {
                outcome.aged = frequency->observations == frequency->period;
                frequency->observe(expert);
            }
            auto result = pin(current);
            if (result) {
                ++counters.cache_hits;
                if (frequency && frequency->trace) {
                    frequency->trace(frequency->trace_context, outcome);
                }
                if (trace) {
                    trace(trace_context, expert.tensor_source_id, expert.expert_id, bytes, true, i);
                }
            }
            return result;
        }
        auto * victim = empty ? nullptr : oldest;
        while (victim && (victim->references || victim->waiters)) {
            victim = victim->newer;
        }
        if (!empty && !victim) {
            wait(true);
            continue;
        }
        // Wait before observing the request so retries never advance the sketch twice.
        // A competing request may change both the resident hit and the eligible victim.
        if (frequency && victim && slots[residents].status != slot::state::empty) {
            wait(true, true);
            continue;
        }
        selected = empty ? empty : victim;
        if (frequency) {
            outcome.aged = frequency->observations == frequency->period;
            const auto candidate_frequency = frequency->observe(expert);
            if (victim) {
                ++frequency->stats.comparisons;
                outcome.victim = victim->expert;
                outcome.candidate_frequency = candidate_frequency;
                outcome.victim_frequency = frequency->frequency(victim->expert, false);
                outcome.bypassed = candidate_frequency <= outcome.victim_frequency;
                if (outcome.bypassed) {
                    ++frequency->stats.rejections;
                    selected = &slots[residents];
                }
            }
            if (outcome.bypassed) {
                ++frequency->stats.bypassed_misses;
            } else {
                ++frequency->stats.admitted_misses;
            }
        }
        if (victim && !outcome.bypassed) {
            ++counters.cache_evictions;
            unlink(*victim);
        }
        break;
    }

    selected->expert = expert;
    selected->bytes = bytes;
    selected->status = slot::state::loading;
    ++selected->generation;
    ++counters.cache_misses;
    ++counters.disk_reads;
    if (frequency && frequency->trace) {
        frequency->trace(frequency->trace_context, outcome);
    }
    if (trace) {
        trace(trace_context, expert.tensor_source_id, expert.expert_id, bytes, false, selected - slots);
    }
    lock.unlock();

    const auto start = cache_clock::now();
    bool success = false;
    try {
        success = reader(reader_context, expert, selected->data, bytes);
    } catch (...) {
        // A reader exception has the same outcome as a short read.
    }
    const auto duration = elapsed_ns(start);

    lock.lock();
    counters.io_wait_ns += duration;
    if (!success) {
        selected->status = selected->waiters ? slot::state::failed : slot::state::empty;
        changed.notify_all();
        return {};
    }
    counters.bytes_read += bytes;
    selected->status = slot::state::ready;
    auto result = pin(*selected);
    changed.notify_all();
    return result;
}

llama_expert_cache::statistics llama_expert_cache::stats() const {
    std::lock_guard<std::mutex> lock(mutex);
    return counters;
}

void llama_expert_cache::set_decision_trace(decision_fn callback, void * context) noexcept {
    if (frequency) {
        frequency->trace = callback;
        frequency->trace_context = context;
    }
}

llama_expert_cache::policy_statistics llama_expert_cache::policy_stats() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (frequency) {
        auto stats = frequency->stats;
        stats.transient_in_use = slots[slot_count - 1].status != slot::state::empty;
        return stats;
    }
    policy_statistics stats;
    stats.resident_slots = slot_count;
    stats.admitted_misses = counters.cache_misses;
    stats.slot_stride = aligned_size(max_expert_bytes);
    return stats;
}
