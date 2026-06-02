/**
 * Market Replay Engine Implementation
 */

#include "market_replay.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cstring>

namespace quant_core {

MarketReplayEngine::MarketReplayEngine()
    : is_playing_(false)
    , is_paused_(false)
    , should_stop_(false) {
}

void MarketReplayEngine::load_data(uint32_t symbol_id, const std::vector<ReplayEvent>& events) {
    // Sort events by timestamp
    std::vector<ReplayEvent> sorted_events = events;
    std::sort(sorted_events.begin(), sorted_events.end(),
              [](const ReplayEvent& a, const ReplayEvent& b) {
                  return a.timestamp_ns < b.timestamp_ns;
              });
    
    event_data_[symbol_id] = sorted_events;
    current_indices_[symbol_id] = 0;
}

bool MarketReplayEngine::load_from_file(uint32_t symbol_id, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read event count
    size_t event_count;
    file.read(reinterpret_cast<char*>(&event_count), sizeof(size_t));
    
    // Read events
    std::vector<ReplayEvent> events(event_count);
    file.read(reinterpret_cast<char*>(events.data()), 
              event_count * sizeof(ReplayEvent));
    
    if (file.fail()) {
        return false;
    }
    
    load_data(symbol_id, events);
    return true;
}

bool MarketReplayEngine::save_to_file(uint32_t symbol_id, const std::string& filepath) {
    auto it = event_data_.find(symbol_id);
    if (it == event_data_.end()) {
        return false;
    }
    
    const std::vector<ReplayEvent>& events = it->second;
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write event count
    size_t event_count = events.size();
    file.write(reinterpret_cast<const char*>(&event_count), sizeof(size_t));
    
    // Write events
    file.write(reinterpret_cast<const char*>(events.data()),
               event_count * sizeof(ReplayEvent));
    
    return !file.fail();
}

void MarketReplayEngine::start_playback(const PlaybackControl& control) {
    control_ = control;
    should_stop_.store(false, std::memory_order_relaxed);
    is_playing_.store(true, std::memory_order_relaxed);
    is_paused_.store(false, std::memory_order_relaxed);
    
    // Start playback thread
    playback_thread_ = std::thread(&MarketReplayEngine::_playback_loop, this);
}

void MarketReplayEngine::stop_playback() {
    should_stop_.store(true, std::memory_order_relaxed);
    
    if (playback_thread_.joinable()) {
        playback_thread_.join();
    }
    
    is_playing_.store(false, std::memory_order_relaxed);
}

void MarketReplayEngine::pause_playback() {
    is_paused_.store(true, std::memory_order_relaxed);
}

void MarketReplayEngine::resume_playback() {
    is_paused_.store(false, std::memory_order_relaxed);
}

bool MarketReplayEngine::seek_to(uint64_t timestamp_ns) {
    // Update all current indices to point to events >= timestamp
    for (auto& [symbol_id, events] : event_data_) {
        size_t index = _find_event_index(symbol_id, timestamp_ns);
        current_indices_[symbol_id] = index;
    }
    
    control_.current_timestamp_ns = timestamp_ns;
    return true;
}

void MarketReplayEngine::step_forward(size_t n) {
    for (size_t step = 0; step < n; ++step) {
        // Find the next event across all symbols
        uint64_t min_timestamp = UINT64_MAX;
        uint32_t next_symbol = 0;
        
        for (auto& [symbol_id, events] : event_data_) {
            size_t idx = current_indices_[symbol_id];
            if (idx < events.size()) {
                if (events[idx].timestamp_ns < min_timestamp) {
                    min_timestamp = events[idx].timestamp_ns;
                    next_symbol = symbol_id;
                }
            }
        }
        
        if (min_timestamp == UINT64_MAX) {
            // No more events
            break;
        }
        
        // Process the event
        size_t& idx = current_indices_[next_symbol];
        const ReplayEvent& event = event_data_[next_symbol][idx];
        _process_event(event);
        idx++;
        
        control_.current_timestamp_ns = event.timestamp_ns;
    }
}

void MarketReplayEngine::step_backward(size_t n) {
    for (size_t step = 0; step < n; ++step) {
        // Find the previous event across all symbols
        uint64_t max_timestamp = 0;
        uint32_t prev_symbol = 0;
        
        for (auto& [symbol_id, events] : event_data_) {
            size_t idx = current_indices_[symbol_id];
            if (idx > 0 && idx <= events.size()) {
                if (events[idx - 1].timestamp_ns > max_timestamp) {
                    max_timestamp = events[idx - 1].timestamp_ns;
                    prev_symbol = symbol_id;
                }
            }
        }
        
        if (max_timestamp == 0) {
            // No previous events
            break;
        }
        
        // Move back
        current_indices_[prev_symbol]--;
        control_.current_timestamp_ns = max_timestamp;
    }
}

uint64_t MarketReplayEngine::get_current_timestamp() const {
    return control_.current_timestamp_ns;
}

bool MarketReplayEngine::is_playing() const {
    return is_playing_.load(std::memory_order_relaxed);
}

bool MarketReplayEngine::is_paused() const {
    return is_paused_.load(std::memory_order_relaxed);
}

void MarketReplayEngine::set_event_callback(EventCallback callback) {
    event_callback_ = callback;
}

size_t MarketReplayEngine::get_event_count(uint32_t symbol_id) const {
    auto it = event_data_.find(symbol_id);
    if (it == event_data_.end()) return 0;
    return it->second.size();
}

size_t MarketReplayEngine::get_total_event_count() const {
    size_t total = 0;
    for (const auto& [symbol_id, events] : event_data_) {
        total += events.size();
    }
    return total;
}

std::vector<uint32_t> MarketReplayEngine::get_symbols() const {
    std::vector<uint32_t> symbols;
    symbols.reserve(event_data_.size());
    
    for (const auto& [symbol_id, events] : event_data_) {
        symbols.push_back(symbol_id);
    }
    
    return symbols;
}

void MarketReplayEngine::clear() {
    stop_playback();
    event_data_.clear();
    current_indices_.clear();
}

const ReplayEvent* MarketReplayEngine::get_event_at(uint32_t symbol_id, size_t index) const {
    auto it = event_data_.find(symbol_id);
    if (it == event_data_.end()) return nullptr;
    
    const std::vector<ReplayEvent>& events = it->second;
    if (index >= events.size()) return nullptr;
    
    return &events[index];
}

void MarketReplayEngine::_playback_loop() {
    uint64_t last_real_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    while (!should_stop_.load(std::memory_order_relaxed)) {
        if (is_paused_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Process next event
        step_forward(1);
        
        // Check if we've reached the end
        bool has_more_events = false;
        for (const auto& [symbol_id, events] : event_data_) {
            size_t idx = current_indices_[symbol_id];
            if (idx < events.size()) {
                has_more_events = true;
                break;
            }
        }
        
        if (!has_more_events) {
            break;
        }
        
        // Sleep to maintain playback speed
        uint64_t current_real_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        
        uint64_t elapsed_real_ns = current_real_time_ns - last_real_time_ns;
        uint64_t elapsed_replay_ns = static_cast<uint64_t>(
            (1.0 / control_.speed_multiplier) * elapsed_real_ns
        );
        
        if (elapsed_replay_ns > 0) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(elapsed_replay_ns));
        }
        
        last_real_time_ns = current_real_time_ns;
    }
    
    is_playing_.store(false, std::memory_order_relaxed);
}

void MarketReplayEngine::_process_event(const ReplayEvent& event) {
    if (event_callback_) {
        event_callback_(event);
    }
}

size_t MarketReplayEngine::_find_event_index(uint32_t symbol_id, uint64_t timestamp_ns) const {
    auto it = event_data_.find(symbol_id);
    if (it == event_data_.end()) return 0;
    
    const std::vector<ReplayEvent>& events = it->second;
    
    // Binary search for first event >= timestamp
    auto it_event = std::lower_bound(events.begin(), events.end(), timestamp_ns,
                                     [](const ReplayEvent& event, uint64_t ts) {
                                         return event.timestamp_ns < ts;
                                     });
    
    return std::distance(events.begin(), it_event);
}

// ReplayCoordinator implementation

ReplayCoordinator::ReplayCoordinator() {
}

void ReplayCoordinator::add_engine(uint32_t symbol_id, std::shared_ptr<MarketReplayEngine> engine) {
    engines_[symbol_id] = engine;
}

void ReplayCoordinator::remove_engine(uint32_t symbol_id) {
    engines_.erase(symbol_id);
}

void ReplayCoordinator::start_synchronized_playback(const PlaybackControl& control) {
    for (auto& [symbol_id, engine] : engines_) {
        engine->start_playback(control);
    }
}

void ReplayCoordinator::stop_all_playback() {
    for (auto& [symbol_id, engine] : engines_) {
        engine->stop_playback();
    }
}

void ReplayCoordinator::pause_all_playback() {
    for (auto& [symbol_id, engine] : engines_) {
        engine->pause_playback();
    }
}

void ReplayCoordinator::resume_all_playback() {
    for (auto& [symbol_id, engine] : engines_) {
        engine->resume_playback();
    }
}

bool ReplayCoordinator::seek_all_to(uint64_t timestamp_ns) {
    for (auto& [symbol_id, engine] : engines_) {
        if (!engine->seek_to(timestamp_ns)) {
            return false;
        }
    }
    return true;
}

std::vector<std::shared_ptr<MarketReplayEngine>> ReplayCoordinator::get_all_engines() const {
    std::vector<std::shared_ptr<MarketReplayEngine>> result;
    result.reserve(engines_.size());
    
    for (const auto& [symbol_id, engine] : engines_) {
        result.push_back(engine);
    }
    
    return result;
}

} // namespace quant_core
