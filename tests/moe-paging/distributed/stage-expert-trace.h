#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <limits>
#include <string>
#include <vector>

struct stage_expert_trace {
    struct event {
        uint64_t source;
        int64_t expert;
        size_t bytes, slot;
        bool decode, hit;
        uint64_t prefetch_id = 0;
        bool prefetch_waited = false;
    };

    std::vector<event> events;
    std::vector<llama_expert_cache::delivery_sample> delivery_timing;
    struct reader_event {
        size_t index;
        llama_expert_io_sample sample;
    };
    std::vector<reader_event> reads;
    std::string io_prefix;
    size_t reads_seen = 0;
    bool prefetch_enabled = false;
    mutable std::mutex reader_trace_mutex;

    static void capture_delivery(void * context, uint64_t id, bool waited,
                                 const llama_expert_cache::delivery_sample & sample) noexcept {
        auto & trace = *static_cast<stage_expert_trace *>(context);
        if (trace.seen && trace.seen <= trace.events.size()) {
            trace.events[trace.seen - 1].prefetch_id = id;
            trace.events[trace.seen - 1].prefetch_waited = waited;
            if (trace.seen <= trace.delivery_timing.size()) trace.delivery_timing[trace.seen - 1] = sample;
        }
    }

    static void capture_io(void * context, const llama_expert_io_sample & sample) noexcept {
        auto & trace = *static_cast<stage_expert_trace *>(context);
        std::lock_guard<std::mutex> lock(trace.reader_trace_mutex);
        if (trace.reads_seen < trace.reads.size()) {
            const auto index = sample.prefetch_id ? std::numeric_limits<size_t>::max() : trace.seen - 1;
            trace.reads[trace.reads_seen] = {index, sample};
        }
        ++trace.reads_seen;
    }

    void save_io(const std::string & prefix, bool complete) const {
        if (reads.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(reader_trace_mutex);
        using json = nlohmann::json;
        std::ofstream stream(std::filesystem::u8path(prefix + ".expert-io.jsonl"), std::ios::binary);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        const size_t saved = std::min(reads_seen, reads.size());
        stream << json({{"version", prefetch_enabled ? 2 : 1}, {"complete", complete}, {"reads", reads_seen},
            {"dropped", reads_seen - saved}, {"access_dropped", seen - std::min(seen, events.size())}}).dump() << '\n';
        for (size_t i = 0; i < saved; ++i) {
            const auto & entry = reads[i];
            const auto & s = entry.sample;
            const auto api = [&](uint64_t value) { return s.file.read_api_measured ? json(value) : json(nullptr); };
            const auto index = s.prefetch_id ? json(nullptr) : json(entry.index);
            auto row = json::array({index, s.source, s.expert, s.shard, s.offset, s.bytes,
                s.begin_ns, s.wait_ns, s.end_ns, api(s.file.read_api_calls), api(s.file.read_api_bytes),
                api(s.file.read_api_ns), s.success});
            if (prefetch_enabled) {
                row.push_back(s.prefetch_id);
            }
            stream << row.dump() << '\n';
        }
        stream.close();
        if (prefetch_enabled && complete) {
            std::ofstream delivery(std::filesystem::u8path(prefix + ".prefetch-delivery.jsonl"), std::ios::binary);
            delivery.exceptions(std::ios::badbit | std::ios::failbit);
            for (size_t index = 0; index < std::min(seen, events.size()); ++index) {
                const auto & event = events[index];
                if (event.prefetch_id) {
                    delivery << json::array({index, event.prefetch_id, event.prefetch_waited}).dump() << '\n';
                }
            }
            delivery.close();
            std::ofstream timing(std::filesystem::u8path(prefix + ".prefetch-timing.jsonl"), std::ios::binary);
            timing.exceptions(std::ios::badbit | std::ios::failbit);
            timing << json({{"version", 1}, {"limit", delivery_timing.size()},
                {"dropped_accesses", seen - std::min(seen, delivery_timing.size())},
                {"complete", ! (reads_seen > reads.size() || seen > delivery_timing.size())}}).dump() << '\n';
            for (size_t index = 0; index < std::min(seen, delivery_timing.size()); ++index) {
                if (!events[index].prefetch_id) continue;
                const auto & t = delivery_timing[index];
                timing << json::array({index, events[index].prefetch_id, t.submit, t.reader_start, t.reader_end,
                    t.ready, t.demand_begin, t.wait_begin, t.wait_end, t.previous_compute_begin,
                    t.previous_compute_end}).dump() << '\n';
            }
        }
    }
    std::vector<llama_expert_cache::decision> decisions;

    static void capture_decision(void * context, const llama_expert_cache::decision & decision) noexcept {
        auto & trace = *static_cast<stage_expert_trace *>(context);
        if (trace.seen < trace.decisions.size()) {
            trace.decisions[trace.seen] = decision;
        }
    }
    nlohmann::json requests = nlohmann::json::array();
    size_t seen = 0;
    bool decoding = false;

    static void capture(void * context, uint64_t source, int64_t expert, size_t bytes, bool hit, size_t slot) noexcept {
        auto & trace = *static_cast<stage_expert_trace *>(context);
        if (trace.seen < trace.events.size()) {
            trace.events[trace.seen] = {source, expert, bytes, slot, trace.decoding, hit};
        }
        ++trace.seen;
    }

    void save(const std::string & prefix, const std::string & role, const llama_model * model) const {
        const auto stats = llama_get_pager_metrics(model);
        const auto saved = std::min(seen, events.size());
        uint64_t demand_miss_bytes = 0;
        for (size_t index = 0; index < saved; ++index) {
            if (!events[index].hit) {
                demand_miss_bytes += events[index].bytes;
            }
        }
        nlohmann::json sources = nlohmann::json::object();
        for (const auto & [id, name] : llama_pager_source_names(model)) {
            sources[std::to_string(id)] = name;
        }
        const nlohmann::json header{
            {"version", 1}, {"contexts", 1},
            {"mode", stats.policy.transient_slots ? "tinylfu" : "lru"}, {"role", role},
            {"resident_slots", stats.policy.resident_slots}, {"transient_slots", stats.policy.transient_slots},
            {"slots", stats.slots}, {"max_expert_bytes", stats.max_expert_bytes},
            {"managed_bytes", stats.peak_allocated_bytes}, {"dropped", seen - saved},
            {"events", seen}, {"hits", stats.cache_hits}, {"misses", stats.cache_misses},
            {"evictions", stats.cache_evictions},
            {"reads", prefetch_enabled ? stats.cache_misses : stats.disk_reads},
            {"bytes_read", prefetch_enabled ? demand_miss_bytes : stats.bytes_read},
            {"prefetch_enabled", prefetch_enabled}, {"reader_calls", stats.disk_reads},
            {"reader_bytes", stats.bytes_read},
            {"sources", sources}, {"requests", requests},
        };
        std::ofstream stream(std::filesystem::u8path(prefix + ".experts.jsonl"), std::ios::binary);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream << header.dump() << '\n';
        for (size_t i = 0; i < saved; ++i) {
            const auto & entry = events[i];
            stream << '[' << i << ',' << entry.source << ',' << entry.expert << ',' << entry.bytes
                   << ',' << entry.slot << ',' << entry.decode << ',' << entry.hit << "]\n";
        }
        stream.close();
        if (!decisions.empty()) {
            std::ofstream detail(std::filesystem::u8path(prefix + ".admission.jsonl"), std::ios::binary);
            detail.exceptions(std::ios::badbit | std::ios::failbit);
            for (size_t i = 0; i < saved; ++i) {
                const auto & decision = decisions[i];
                detail << nlohmann::json::array({i, decision.candidate_frequency, decision.victim_frequency,
                    decision.victim.tensor_source_id, decision.victim.expert_id, decision.bypassed, decision.aged}).dump()
                    << '\n';
            }
            detail.close();
        }
    }
};

