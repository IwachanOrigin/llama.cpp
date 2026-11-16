#include "llama-expert-cache.h"

#include <cassert>
#include <chrono>
#include <new>
#include <stdexcept>
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

size_t llama_expert_cache::required_bytes(size_t max_expert_bytes, size_t count) {
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
    return sizeof(llama_expert_cache) + metadata + count * stride;
}

llama_expert_cache::llama_expert_cache(size_t budget, size_t max_expert_bytes, read_fn reader, void * reader_context) :
    reader(reader), reader_context(reader_context), max_expert_bytes(max_expert_bytes) {
    if (!reader || budget < required_bytes(max_expert_bytes, 1)) {
        throw std::invalid_argument("expert cache budget must fit at least one slot and its metadata");
    }
    const size_t stride = aligned_size(max_expert_bytes);
    const size_t available = budget - sizeof(*this);
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
    slot_count = lower;
    static_assert(alignof(slot) <= slot_alignment, "slot metadata must fit in the aligned allocation");
    counters.peak_allocated_bytes = required_bytes(max_expert_bytes, slot_count);
    allocation = ::operator new(counters.peak_allocated_bytes - sizeof(*this), std::align_val_t(slot_alignment));
    slots = ::new (allocation) slot[slot_count]();
    auto * data = static_cast<unsigned char *>(allocation) + aligned_size(slot_count * sizeof(slot));
    for (size_t i = 0; i < slot_count; ++i) {
        slots[i].data = data + i * stride;
    }
}

llama_expert_cache::~llama_expert_cache() {
    assert(counters.pinned_slots == 0 && counters.waiting_requests == 0);
#ifndef NDEBUG
    for (size_t i = 0; i < slot_count; ++i) {
        assert(slots[i].status != slot::state::loading);
    }
#endif
    ::operator delete(allocation, std::align_val_t(slot_alignment));
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
    touch(storage);
    return lease(*this, storage);
}

bool llama_expert_cache::release(slot & storage, uint64_t generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (storage.status != slot::state::ready || storage.generation != generation || !storage.references) {
        return false;
    }
    if (--storage.references == 0) {
        --counters.pinned_slots;
        changed.notify_all();
    }
    return true;
}

llama_expert_cache::lease llama_expert_cache::acquire(key expert, size_t bytes) {
    if (expert.expert_id < 0 || !bytes || bytes > max_expert_bytes) {
        return {};
    }
    std::unique_lock<std::mutex> lock(mutex);
    const auto wait = [&] {
        const auto start = cache_clock::now();
        ++counters.waiting_requests;
        changed.wait(lock);
        --counters.waiting_requests;
        counters.cache_wait_ns += elapsed_ns(start);
    };

    slot * selected = nullptr;
    for (;;) {
        slot * empty = nullptr;
        for (size_t i = 0; i < slot_count; ++i) {
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
                    wait();
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
            auto result = pin(current);
            if (result) {
                ++counters.cache_hits;
            }
            return result;
        }
        selected = empty;
        if (!selected) {
            for (auto * candidate = oldest; candidate; candidate = candidate->newer) {
                if (candidate->references == 0 && candidate->waiters == 0) {
                    selected = candidate;
                    ++counters.cache_evictions;
                    unlink(*selected);
                    break;
                }
            }
        }
        if (selected) {
            break;
        }
        wait();
    }

    selected->expert = expert;
    selected->bytes = bytes;
    selected->status = slot::state::loading;
    ++selected->generation;
    ++counters.cache_misses;
    ++counters.disk_reads;
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
