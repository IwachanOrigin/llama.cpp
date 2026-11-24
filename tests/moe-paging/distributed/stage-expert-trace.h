#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct stage_expert_trace {
    struct event {
        uint64_t source;
        int64_t expert;
        size_t bytes, slot;
        bool decode, hit;
    };

    std::vector<event> events;
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
            {"evictions", stats.cache_evictions}, {"reads", stats.disk_reads}, {"bytes_read", stats.bytes_read},
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

