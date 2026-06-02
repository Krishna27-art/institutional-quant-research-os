/**
 * Position Manager Implementation
 */

#include "position_manager.h"
#include <algorithm>
#include <cmath>

namespace quant_core {

PositionManager::PositionManager()
    : margin_multiplier_(0.5)  // Default 50% margin
    , total_unrealized_pnl_(0.0)
    , total_realized_pnl_(0.0)
    , total_notional_(0.0)
    , total_margin_(0.0) {
}

void PositionManager::update_position(uint32_t symbol_id, int64_t quantity,
                                     double price, uint64_t timestamp_ns) {
    auto it = positions_.find(symbol_id);
    
    if (it == positions_.end()) {
        // New position
        Position pos;
        pos.symbol_id = symbol_id;
        pos.quantity = quantity;
        pos.avg_entry_price = price;
        pos.current_price = price;
        pos.total_cost = quantity * price;
        pos.last_update_ns = timestamp_ns;
        
        _calculate_pnl(pos);
        
        positions_[symbol_id] = pos;
    } else {
        // Update existing position
        Position& pos = it->second;
        
        int64_t old_quantity = pos.quantity;
        double old_cost = pos.total_cost;
        
        // Update quantity
        pos.quantity += quantity;
        
        // Update average entry price
        if ((old_quantity >= 0 && quantity > 0) || (old_quantity <= 0 && quantity < 0)) {
            // Adding to position (same direction)
            pos.total_cost += quantity * price;
            pos.avg_entry_price = pos.total_cost / pos.quantity;
        } else {
            // Reducing or flipping position
            double realized = 0.0;
            
            if (abs(quantity) <= abs(old_quantity)) {
                // Partial close
                realized = quantity * (price - pos.avg_entry_price);
                pos.realized_pnl += realized;
                pos.total_cost += quantity * price;
            } else {
                // Flip position
                realized = -old_quantity * (price - pos.avg_entry_price);
                pos.realized_pnl += realized;
                pos.total_cost = pos.quantity * price;
                pos.avg_entry_price = price;
            }
        }
        
        pos.current_price = price;
        pos.last_update_ns = timestamp_ns;
        
        _calculate_pnl(pos);
    }
    
    _update_cached_totals();
}

void PositionManager::update_price(uint32_t symbol_id, double price, uint64_t timestamp_ns) {
    auto it = positions_.find(symbol_id);
    if (it == positions_.end()) return;
    
    Position& pos = it->second;
    pos.current_price = price;
    pos.last_update_ns = timestamp_ns;
    
    _calculate_pnl(pos);
    _update_cached_totals();
}

const Position* PositionManager::get_position(uint32_t symbol_id) const {
    auto it = positions_.find(symbol_id);
    if (it == positions_.end()) return nullptr;
    return &it->second;
}

std::vector<Position> PositionManager::get_all_positions() const {
    std::vector<Position> result;
    result.reserve(positions_.size());
    
    for (const auto& [symbol_id, pos] : positions_) {
        result.push_back(pos);
    }
    
    return result;
}

double PositionManager::get_total_unrealized_pnl() const {
    return total_unrealized_pnl_.load(std::memory_order_relaxed);
}

double PositionManager::get_total_realized_pnl() const {
    return total_realized_pnl_.load(std::memory_order_relaxed);
}

double PositionManager::get_total_pnl() const {
    return get_total_unrealized_pnl() + get_total_realized_pnl();
}

double PositionManager::get_total_notional() const {
    return total_notional_.load(std::memory_order_relaxed);
}

double PositionManager::get_total_margin_requirement() const {
    return total_margin_.load(std::memory_order_relaxed);
}

size_t PositionManager::get_position_count() const {
    return positions_.size();
}

bool PositionManager::has_position(uint32_t symbol_id) const {
    return positions_.find(symbol_id) != positions_.end();
}

void PositionManager::close_position(uint32_t symbol_id, double price, uint64_t timestamp_ns) {
    auto it = positions_.find(symbol_id);
    if (it == positions_.end()) return;
    
    Position& pos = it->second;
    
    // Realize all PnL
    double realized = -pos.quantity * (price - pos.avg_entry_price);
    pos.realized_pnl += realized;
    pos.unrealized_pnl = 0.0;
    pos.quantity = 0;
    pos.current_price = price;
    pos.last_update_ns = timestamp_ns;
    
    // Update market value and notional
    pos.market_value = 0.0;
    pos.notional = 0.0;
    pos.margin_requirement = 0.0;
    
    _update_cached_totals();
}

void PositionManager::clear() {
    positions_.clear();
    total_unrealized_pnl_.store(0.0, std::memory_order_relaxed);
    total_realized_pnl_.store(0.0, std::memory_order_relaxed);
    total_notional_.store(0.0, std::memory_order_relaxed);
    total_margin_.store(0.0, std::memory_order_relaxed);
}

void PositionManager::set_margin_multiplier(double multiplier) {
    margin_multiplier_ = multiplier;
    
    // Recalculate all margins
    for (auto& [symbol_id, pos] : positions_) {
        pos.margin_requirement = pos.notional * margin_multiplier_;
    }
    
    _update_cached_totals();
}

double PositionManager::get_margin_multiplier() const {
    return margin_multiplier_;
}

void PositionManager::_update_cached_totals() const {
    double total_unrealized = 0.0;
    double total_realized = 0.0;
    double total_notional_val = 0.0;
    double total_margin_val = 0.0;
    
    for (const auto& [symbol_id, pos] : positions_) {
        total_unrealized += pos.unrealized_pnl;
        total_realized += pos.realized_pnl;
        total_notional_val += pos.notional;
        total_margin_val += pos.margin_requirement;
    }
    
    total_unrealized_pnl_.store(total_unrealized, std::memory_order_relaxed);
    total_realized_pnl_.store(total_realized, std::memory_order_relaxed);
    total_notional_.store(total_notional_val, std::memory_order_relaxed);
    total_margin_.store(total_margin_val, std::memory_order_relaxed);
}

void PositionManager::_calculate_pnl(Position& pos) {
    // Calculate unrealized PnL
    if (pos.quantity != 0) {
        pos.unrealized_pnl = pos.quantity * (pos.current_price - pos.avg_entry_price);
    } else {
        pos.unrealized_pnl = 0.0;
    }
    
    // Calculate market value
    pos.market_value = pos.quantity * pos.current_price;
    
    // Calculate notional (absolute value)
    pos.notional = std::abs(pos.market_value);
    
    // Calculate margin requirement
    pos.margin_requirement = pos.notional * margin_multiplier_;
}

// PortfolioManager implementation

PortfolioManager::PortfolioManager() {
}

PositionManager* PortfolioManager::get_position_manager(const std::string& asset_class) {
    auto it = position_managers_.find(asset_class);
    
    if (it == position_managers_.end()) {
        position_managers_[asset_class] = std::make_unique<PositionManager>();
        return position_managers_[asset_class].get();
    }
    
    return it->second.get();
}

PortfolioSummary PortfolioManager::get_portfolio_summary() const {
    PortfolioSummary summary;
    summary.total_pnl = 0.0;
    summary.unrealized_pnl = 0.0;
    summary.realized_pnl = 0.0;
    summary.total_notional = 0.0;
    summary.total_margin = 0.0;
    summary.position_count = 0;
    summary.last_update_ns = 0;
    
    for (const auto& [asset_class, manager] : position_managers_) {
        summary.total_pnl += manager->get_total_pnl();
        summary.unrealized_pnl += manager->get_total_unrealized_pnl();
        summary.realized_pnl += manager->get_total_realized_pnl();
        summary.total_notional += manager->get_total_notional();
        summary.total_margin += manager->get_total_margin_requirement();
        summary.position_count += manager->get_position_count();
    }
    
    summary.last_update_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    return summary;
}

std::vector<std::string> PortfolioManager::get_asset_classes() const {
    std::vector<std::string> result;
    result.reserve(position_managers_.size());
    
    for (const auto& [asset_class, manager] : position_managers_) {
        result.push_back(asset_class);
    }
    
    return result;
}

} // namespace quant_core
