#pragma once

#include "llama-ext.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

struct stage_performance {
    using clock = std::chrono::steady_clock;
    using json = nlohmann::json;
    struct row {
        std::string phase;
        uint32_t request = 0, position = 0, tokens = 0;
        double execution = -1, roundtrip = -1, send = -1, receive = -1, serialize = -1, deserialize = -1;
        uint64_t sent = 0, received = 0;
        bool pager_valid = false;
        llama_pager_metrics pager{};
    };
    struct transfer_detail {
        double first_byte_ms = 0, header_ms = 0, payload_ms = 0;
        double combined_ms = 0, coalesce_copy_ms = 0;
        uint64_t calls = 0, partial = 0, retries = 0, combined_bytes = 0, coalesce_copy_bytes = 0;
        json encode() const {
            return {{"first_byte_ms", first_byte_ms}, {"header_ms", combined_bytes ? json(nullptr) : json(header_ms)},
                    {"payload_ms", combined_bytes ? json(nullptr) : json(payload_ms)},
                    {"combined_send_ms", combined_bytes ? json(combined_ms) : json(nullptr)},
                    {"combined_send_bytes", combined_bytes}, {"coalesce_copy_ms", coalesce_copy_ms},
                    {"coalesce_copy_bytes", coalesce_copy_bytes},
                    {"calls", calls}, {"partial", partial}, {"retries", retries}};
        }
    };
    struct tcp_detail {
        uint32_t request;
        transfer_detail sent, received;
        double copy_ms;
        uint64_t copy_bytes;
    };
    bool detail_enabled = false;
    static constexpr size_t detail_limit = 4096;
    std::vector<tcp_detail> communication;
    void add_communication(tcp_detail entry) {
        if (communication.size() < detail_limit) {
            communication.push_back(entry);
        }
    }

    json summary = json::object();
    std::vector<row> rows;
    std::string prefix;
    clock::time_point started = clock::now();
    double send_ms = 0, receive_ms = 0;
    uint64_t sent = 0, received = 0, timeouts = 0, transport_errors = 0, session_errors = 0;
    uint32_t session = 0;

    static double elapsed(clock::time_point start) {
        return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    }

    static llama_pager_metrics delta(const llama_pager_metrics & after, const llama_pager_metrics & before) {
        return {after.cache_hits-before.cache_hits, after.cache_misses-before.cache_misses,
                after.cache_evictions-before.cache_evictions, after.disk_reads-before.disk_reads,
                after.bytes_read-before.bytes_read, after.io_wait_ns-before.io_wait_ns,
                after.cache_wait_ns-before.cache_wait_ns, after.capacity_wait_ns-before.capacity_wait_ns,
                after.lease_wait_ns-before.lease_wait_ns, after.pinned_slots, after.peak_allocated_bytes,
                after.slots, after.max_expert_bytes};
    }

    static json pager_json(const llama_pager_metrics & pager, uint64_t inputs) {
        const auto lookups = pager.cache_hits + pager.cache_misses;
        return {{"cache_hits", pager.cache_hits}, {"cache_misses", pager.cache_misses},
            {"cache_evictions", pager.cache_evictions}, {"disk_reads", pager.disk_reads},
            {"bytes_read", pager.bytes_read}, {"io_wait_ns", pager.io_wait_ns},
            {"cache_wait_ns", pager.cache_wait_ns}, {"capacity_wait_ns", pager.capacity_wait_ns},
            {"lease_wait_ns", pager.lease_wait_ns}, {"peak_allocated_bytes", pager.peak_allocated_bytes},
            {"pinned_slots", pager.pinned_slots}, {"slots", pager.slots},
            {"hit_rate", lookups ? json(double(pager.cache_hits)/lookups) : json(nullptr)},
            {"reads_per_input_token", inputs ? json(double(pager.disk_reads)/inputs) : json(nullptr)},
            {"bytes_per_input_token", inputs ? json(double(pager.bytes_read)/inputs) : json(nullptr)},
            {"io_wait_ns_per_read", pager.disk_reads ? json(double(pager.io_wait_ns)/pager.disk_reads) : json(nullptr)}};
    }

    static json process_memory() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX stats{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&stats), sizeof(stats))) {
            return {{"working_set_bytes", stats.WorkingSetSize}, {"peak_working_set_bytes", stats.PeakWorkingSetSize},
                    {"private_commit_bytes", stats.PrivateUsage}, {"peak_pagefile_usage_bytes", stats.PeakPagefileUsage}};
        }
#else
        std::ifstream status("/proc/self/status");
        json values = json::object();
        for (std::string line; std::getline(status, line);) {
            if (line.rfind("VmRSS:", 0) == 0) values["rss_bytes"] = std::stoull(line.substr(6)) * 1024;
            if (line.rfind("VmHWM:", 0) == 0) values["peak_rss_bytes"] = std::stoull(line.substr(6)) * 1024;
        }
        if (!values.empty()) return values;
#endif
        return nullptr;
    }

    json stage_stats(const std::string & phase) const {
        size_t calls = 0;
        uint64_t inputs = 0;
        double total = 0, minimum = 0, maximum = 0, roundtrip = 0;
        bool roundtrip_measured = false;
        llama_pager_metrics pager{};
        for (const auto & sample : rows) {
            if (sample.execution < 0 || (!phase.empty() && sample.phase != phase)) continue;
            minimum = calls ? std::min(minimum, sample.execution) : sample.execution;
            maximum = std::max(maximum, sample.execution);
            total += sample.execution;
            if (sample.roundtrip >= 0) {
                roundtrip += sample.roundtrip;
                roundtrip_measured = true;
            }
            ++calls;
            inputs += sample.tokens;
            pager.cache_hits += sample.pager.cache_hits;
            pager.cache_misses += sample.pager.cache_misses;
            pager.cache_evictions += sample.pager.cache_evictions;
            pager.disk_reads += sample.pager.disk_reads;
            pager.bytes_read += sample.pager.bytes_read;
            pager.io_wait_ns += sample.pager.io_wait_ns;
            pager.cache_wait_ns += sample.pager.cache_wait_ns;
            pager.capacity_wait_ns += sample.pager.capacity_wait_ns;
            pager.lease_wait_ns += sample.pager.lease_wait_ns;
            pager.peak_allocated_bytes = std::max(pager.peak_allocated_bytes, sample.pager.peak_allocated_bytes);
            pager.pinned_slots = sample.pager.pinned_slots;
            pager.slots = sample.pager.slots;
        }
        return {{"stage_execution_calls", calls}, {"input_tokens", inputs},
                {"stage_execution_wall_ms", total}, {"roundtrip_ms", roundtrip_measured ? json(roundtrip) : json(nullptr)},
                {"stage_execution_avg_ms", calls ? json(total/calls) : json(nullptr)},
                {"stage_execution_min_ms", calls ? json(minimum) : json(nullptr)},
                {"stage_execution_max_ms", calls ? json(maximum) : json(nullptr)},
                {"pager", pager_json(pager, inputs)}};
    }

    void save(bool success, const char * error = nullptr) {
        summary["schema_version"] = 1;
        summary["status"] = success ? "ok" : "failed";
        if (error) summary["error"] = error;
        summary["session_id"] = session ? json(session) : json(nullptr);
        summary["elapsed_before_metrics_write_ms"] = elapsed(started);
        summary["stage"] = stage_stats("");
        if (summary["role"] != "worker") {
            summary["phases"] = {{"prefill", stage_stats("prefill")}, {"decode", stage_stats("decode")}};
        }
        summary["tcp"] = {{"sent_bytes", sent}, {"received_bytes", received}, {"send_wall_ms", send_ms},
                          {"receive_wall_ms", receive_ms}, {"timeouts", timeouts}, {"transport_errors", transport_errors},
                          {"session_errors", session_errors}, {"request_rows", rows.size()}};
        if (detail_enabled) {
            summary["tcp_detail"] = {{"schema_version", 1}, {"limit", detail_limit},
                {"reserved_bytes", communication.capacity() * sizeof(tcp_detail)},
                {"complete", success && (summary["role"] == "baseline" || communication.size() == rows.size())},
                {"requests", json::array()}};
            for (const auto & entry : communication) {
                summary["tcp_detail"]["requests"].push_back({{"request_id", entry.request},
                    {"send", entry.sent.encode()}, {"receive", entry.received.encode()},
                    {"last_logits_copy_ms", entry.copy_ms < 0 ? json(nullptr) : json(entry.copy_ms)},
                    {"last_logits_copy_bytes", entry.copy_bytes}});
            }
        }
        if (!summary.contains("process_memory")) summary["process_memory"] = process_memory();
        std::ofstream csv(std::filesystem::u8path(prefix + ".perf.csv"), std::ios::binary);
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << std::setprecision(9);
        csv << "role,phase,request_id,position,token_count,stage_execution_ms,request_roundtrip_ms,"
               "send_wall_ms,receive_wall_ms,serialize_ms,deserialize_ms,send_bytes,receive_bytes,"
               "pager_read_count_delta,pager_bytes_read_delta,pager_io_wait_ms_delta\n";
        auto optional = [&](double value) { if (value >= 0) csv << value; csv << ','; };
        for (const auto & sample : rows) {
            csv << summary["role"].get<std::string>() << ',' << sample.phase << ',' << sample.request << ','
                << sample.position << ',' << sample.tokens << ',';
            optional(sample.execution);
            optional(sample.roundtrip);
            optional(sample.send);
            optional(sample.receive);
            optional(sample.serialize);
            optional(sample.deserialize);
            if (sample.send >= 0) csv << sample.sent;
            csv << ',';
            if (sample.receive >= 0) csv << sample.received;
            csv << ',';
            if (sample.pager_valid) {
                csv << sample.pager.disk_reads << ',' << sample.pager.bytes_read << ',' << sample.pager.io_wait_ns/1e6;
            } else {
                csv << ",,";
            }
            csv << '\n';
        }
        csv.close();
        std::ofstream json_file(std::filesystem::u8path(prefix + ".metrics.json"), std::ios::binary);
        json_file.exceptions(std::ios::badbit | std::ios::failbit);
        json_file << summary.dump(2, ' ', false, json::error_handler_t::replace) << '\n';
        json_file.close();
    }
};

