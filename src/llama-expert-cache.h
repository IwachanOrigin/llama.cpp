#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

class llama_expert_cache {
    struct slot;

public:
    struct key {
        uint64_t tensor_source_id;
        int64_t expert_id;
    };

    struct statistics {
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        uint64_t cache_evictions = 0;
        uint64_t disk_reads = 0;
        uint64_t bytes_read = 0;
        uint64_t io_wait_ns = 0;
        uint64_t cache_wait_ns = 0;
        uint64_t capacity_wait_ns = 0;
        size_t pinned_slots = 0;
        size_t peak_allocated_bytes = 0;
        size_t waiting_requests = 0;
    };

    // Return true only after writing all bytes. The reader must be thread-safe.
    using read_fn = bool (*)(void * context, key expert, void * dst, size_t bytes);

    class lease {
    public:
        lease() = default;
        ~lease();
        lease(lease && other) noexcept;
        lease & operator=(lease && other) noexcept;
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;

        explicit operator bool() const { return owner != nullptr; }
        const void * data() const { return bytes; }
        size_t size() const { return length; }
        bool reset() noexcept;

    private:
        friend class llama_expert_cache;
        lease(llama_expert_cache & owner, slot & storage);

        llama_expert_cache * owner = nullptr;
        slot * storage = nullptr;
        const void * bytes = nullptr;
        size_t length = 0;
        uint64_t generation = 0;
    };

    // The cache and reader context must outlive all acquires and leases.
    llama_expert_cache(size_t budget, size_t max_expert_bytes, read_fn reader, void * reader_context);
    ~llama_expert_cache();
    llama_expert_cache(const llama_expert_cache &) = delete;
    llama_expert_cache & operator=(const llama_expert_cache &) = delete;

    static size_t required_bytes(size_t max_expert_bytes, size_t slot_count);
    static size_t capacity_for_budget(size_t budget, size_t max_expert_bytes);
    size_t capacity() const { return slot_count; }
    lease acquire(key expert, size_t bytes);
    statistics stats() const;

private:
    void unlink(slot & storage);
    void touch(slot & storage);
    lease pin(slot & storage);
    bool release(slot & storage, uint64_t generation) noexcept;

    mutable std::mutex mutex;
    std::condition_variable changed;
    statistics counters;
    read_fn reader;
    void * reader_context;
    size_t max_expert_bytes;
    size_t slot_count = 0;
    void * allocation = nullptr;
    slot * slots = nullptr;
    slot * newest = nullptr;
    slot * oldest = nullptr;
};
