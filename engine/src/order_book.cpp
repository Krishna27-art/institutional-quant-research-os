/**
 * Limit Order Book Implementation
 */

#include "order_book.h"
#include <algorithm>
#include <stdexcept>

namespace quant_core {

LimitOrderBook::LimitOrderBook(uint32_t symbol_id)
    : symbol_id_(symbol_id) {
}

void LimitOrderBook::add_order(uint64_t order_id, bool is_buy,
                               double price, uint64_t quantity,
                               uint64_t timestamp_ns) {
    // Check if order already exists
    if (orders_.find(order_id) != orders_.end()) {
        throw std::runtime_error("Order already exists");
    }
    
    // Create order
    Order order;
    order.order_id = order_id;
    order.symbol_id = symbol_id_;
    order.is_buy = is_buy;
    order.price = price;
    order.quantity = quantity;
    order.timestamp_ns = timestamp_ns;
    order.prev_order = 0;
    order.next_order = 0;
    
    // Store order
    orders_[order_id] = order;
    
    // Add to price level
    if (is_buy) {
        _add_to_price_level(order, bid_levels_);
        bid_prices_.push(price);
    } else {
        _add_to_price_level(order, ask_levels_);
        ask_prices_.push(price);
    }
}

bool LimitOrderBook::cancel_order(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    Order& order = it->second;
    
    // Remove from price level
    if (order.is_buy) {
        _remove_from_price_level(order, bid_levels_);
    } else {
        _remove_from_price_level(order, ask_levels_);
    }
    
    // Remove order
    orders_.erase(it);
    
    return true;
}

bool LimitOrderBook::modify_order(uint64_t order_id, uint64_t new_quantity) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    Order& order = it->second;
    
    // Update quantity
    uint64_t old_quantity = order.quantity;
    order.quantity = new_quantity;
    
    // Update price level total quantity
    auto& levels = order.is_buy ? bid_levels_ : ask_levels_;
    auto level_it = levels.find(order.price);
    if (level_it != levels.end()) {
        level_it->second.total_quantity += (new_quantity - old_quantity);
    }
    
    return true;
}

uint64_t LimitOrderBook::execute_market_order(bool is_buy, uint64_t quantity) {
    uint64_t filled = 0;
    
    if (is_buy) {
        // Buy: consume asks
        while (quantity > 0 && !ask_prices_.empty()) {
            double best_ask = ask_prices_.top();
            auto level_it = ask_levels_.find(best_ask);
            
            if (level_it == ask_levels_.end()) {
                ask_prices_.pop();
                continue;
            }
            
            PriceLevel& level = level_it->second;
            uint64_t available = level.total_quantity;
            uint64_t to_fill = std::min(quantity, available);
            
            filled += to_fill;
            quantity -= to_fill;
            level.total_quantity -= to_fill;
            
            if (level.total_quantity == 0) {
                _cleanup_price_level(best_ask, ask_levels_);
            }
        }
    } else {
        // Sell: consume bids
        while (quantity > 0 && !bid_prices_.empty()) {
            double best_bid = bid_prices_.top();
            auto level_it = bid_levels_.find(best_bid);
            
            if (level_it == bid_levels_.end()) {
                bid_prices_.pop();
                continue;
            }
            
            PriceLevel& level = level_it->second;
            uint64_t available = level.total_quantity;
            uint64_t to_fill = std::min(quantity, available);
            
            filled += to_fill;
            quantity -= to_fill;
            level.total_quantity -= to_fill;
            
            if (level.total_quantity == 0) {
                _cleanup_price_level(best_bid, bid_levels_);
            }
        }
    }
    
    return filled;
}

double LimitOrderBook::get_best_bid() const {
    if (bid_prices_.empty()) return 0.0;
    
    // Clean up stale entries
    while (!bid_prices_.empty()) {
        double price = bid_prices_.top();
        if (bid_levels_.find(price) != bid_levels_.end()) {
            return price;
        }
        bid_prices_.pop();
    }
    
    return 0.0;
}

double LimitOrderBook::get_best_ask() const {
    if (ask_prices_.empty()) return 0.0;
    
    // Clean up stale entries
    while (!ask_prices_.empty()) {
        double price = ask_prices_.top();
        if (ask_levels_.find(price) != ask_levels_.end()) {
            return price;
        }
        ask_prices_.pop();
    }
    
    return 0.0;
}

uint64_t LimitOrderBook::get_best_bid_quantity() const {
    double best_bid = get_best_bid();
    if (best_bid == 0.0) return 0;
    
    auto it = bid_levels_.find(best_bid);
    if (it != bid_levels_.end()) {
        return it->second.total_quantity;
    }
    return 0;
}

uint64_t LimitOrderBook::get_best_ask_quantity() const {
    double best_ask = get_best_ask();
    if (best_ask == 0.0) return 0;
    
    auto it = ask_levels_.find(best_ask);
    if (it != ask_levels_.end()) {
        return it->second.total_quantity;
    }
    return 0;
}

double LimitOrderBook::get_spread() const {
    double bid = get_best_bid();
    double ask = get_best_ask();
    
    if (bid == 0.0 || ask == 0.0) return 0.0;
    return ask - bid;
}

double LimitOrderBook::get_mid_price() const {
    double bid = get_best_bid();
    double ask = get_best_ask();
    
    if (bid == 0.0 || ask == 0.0) return 0.0;
    return (bid + ask) / 2.0;
}

std::vector<std::pair<double, uint64_t>> LimitOrderBook::get_bids(size_t n) const {
    std::vector<std::pair<double, uint64_t>> result;
    
    // Copy prices to avoid modifying heap
    auto temp = bid_prices_;
    
    for (size_t i = 0; i < n && !temp.empty(); ++i) {
        double price = temp.top();
        temp.pop();
        
        auto it = bid_levels_.find(price);
        if (it != bid_levels_.end()) {
            result.emplace_back(price, it->second.total_quantity);
        }
    }
    
    return result;
}

std::vector<std::pair<double, uint64_t>> LimitOrderBook::get_asks(size_t n) const {
    std::vector<std::pair<double, uint64_t>> result;
    
    // Copy prices to avoid modifying heap
    auto temp = ask_prices_;
    
    for (size_t i = 0; i < n && !temp.empty(); ++i) {
        double price = temp.top();
        temp.pop();
        
        auto it = ask_levels_.find(price);
        if (it != ask_levels_.end()) {
            result.emplace_back(price, it->second.total_quantity);
        }
    }
    
    return result;
}

uint64_t LimitOrderBook::get_total_bid_volume() const {
    uint64_t total = 0;
    for (const auto& [price, level] : bid_levels_) {
        total += level.total_quantity;
    }
    return total;
}

uint64_t LimitOrderBook::get_total_ask_volume() const {
    uint64_t total = 0;
    for (const auto& [price, level] : ask_levels_) {
        total += level.total_quantity;
    }
    return total;
}

size_t LimitOrderBook::get_order_count() const {
    return orders_.size();
}

void LimitOrderBook::clear() {
    orders_.clear();
    bid_levels_.clear();
    ask_levels_.clear();
    
    // Clear heaps
    while (!bid_prices_.empty()) bid_prices_.pop();
    while (!ask_prices_.empty()) ask_prices_.pop();
}

void LimitOrderBook::_add_to_price_level(Order& order,
                                          std::unordered_map<double, PriceLevel>& levels) {
    auto it = levels.find(order.price);
    
    if (it == levels.end()) {
        // New price level
        PriceLevel level;
        level.price = order.price;
        level.total_quantity = order.quantity;
        level.order_count = 1;
        level.head_order = order.order_id;
        level.tail_order = order.order_id;
        
        levels[order.price] = level;
    } else {
        // Existing price level
        PriceLevel& level = it->second;
        level.total_quantity += order.quantity;
        level.order_count += 1;
        
        // Add to linked list
        uint64_t old_tail = level.tail_order;
        order.prev_order = old_tail;
        level.tail_order = order.order_id;
        
        // Update old tail
        auto old_tail_it = orders_.find(old_tail);
        if (old_tail_it != orders_.end()) {
            old_tail_it->second.next_order = order.order_id;
        }
    }
}

void LimitOrderBook::_remove_from_price_level(Order& order,
                                               std::unordered_map<double, PriceLevel>& levels) {
    auto level_it = levels.find(order.price);
    if (level_it == levels.end()) return;
    
    PriceLevel& level = level_it->second;
    level.total_quantity -= order.quantity;
    level.order_count -= 1;
    
    // Update linked list
    if (order.prev_order != 0) {
        auto prev_it = orders_.find(order.prev_order);
        if (prev_it != orders_.end()) {
            prev_it->second.next_order = order.next_order;
        }
    }
    
    if (order.next_order != 0) {
        auto next_it = orders_.find(order.next_order);
        if (next_it != orders_.end()) {
            next_it->second.prev_order = order.prev_order;
        }
    }
    
    // Update head/tail
    if (level.head_order == order.order_id) {
        level.head_order = order.next_order;
    }
    
    if (level.tail_order == order.order_id) {
        level.tail_order = order.prev_order;
    }
    
    // Cleanup if empty
    if (level.order_count == 0) {
        _cleanup_price_level(order.price, levels);
    }
}

void LimitOrderBook::_cleanup_price_level(double price,
                                          std::unordered_map<double, PriceLevel>& levels) {
    levels.erase(price);
}

// OrderBookManager implementation

OrderBookManager::OrderBookManager() {
}

LimitOrderBook* OrderBookManager::get_order_book(uint32_t symbol_id) {
    auto it = order_books_.find(symbol_id);
    
    if (it == order_books_.end()) {
        order_books_[symbol_id] = std::make_unique<LimitOrderBook>(symbol_id);
        return order_books_[symbol_id].get();
    }
    
    return it->second.get();
}

void OrderBookManager::remove_order_book(uint32_t symbol_id) {
    order_books_.erase(symbol_id);
}

std::vector<uint32_t> OrderBookManager::get_symbols() const {
    std::vector<uint32_t> symbols;
    symbols.reserve(order_books_.size());
    
    for (const auto& [symbol_id, book] : order_books_) {
        symbols.push_back(symbol_id);
    }
    
    return symbols;
}

} // namespace quant_core
