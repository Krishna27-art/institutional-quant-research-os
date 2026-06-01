#include "order_book.h"
#include <algorithm>
#include <spdlog/spdlog.h>

namespace niftyquant {

OrderBook::OrderBook(const std::string& symbol) 
    : symbol_(symbol) {
    spdlog::info("OrderBook created for symbol: {}", symbol);
}

OrderId OrderBook::add_order(std::shared_ptr<Order> order) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    order->id = next_order_id_++;
    order->timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    
    orders_[order->id] = order;
    
    if (order->side == Side::BUY) {
        bids_[order->price] += order->quantity;
    } else {
        asks_[order->price] += order->quantity;
    }
    
    spdlog::debug("Order added: {} {} {} @ {}", 
                  order->symbol, 
                  order->side == Side::BUY ? "BUY" : "SELL",
                  order->quantity, 
                  order->price / 100.0);
    
    match_orders();
    
    return order->id;
}

bool OrderBook::cancel_order(OrderId order_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    auto order = it->second;
    
    if (order->side == Side::BUY) {
        bids_[order->price] -= order->remaining();
        if (bids_[order->price] == 0) {
            bids_.erase(order->price);
        }
    } else {
        asks_[order->price] -= order->remaining();
        if (asks_[order->price] == 0) {
            asks_.erase(order->price);
        }
    }
    
    order->status = OrderStatus::CANCELLED;
    orders_.erase(it);
    
    spdlog::debug("Order cancelled: {}", order_id);
    return true;
}

MarketDepth OrderBook::get_depth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    MarketDepth depth;
    
    int i = 0;
    for (const auto& [price, qty] : bids_) {
        if (i >= MarketDepth::LEVELS) break;
        depth.bid_prices[i] = price;
        depth.bid_quantities[i] = qty;
        i++;
    }
    
    i = 0;
    for (const auto& [price, qty] : asks_) {
        if (i >= MarketDepth::LEVELS) break;
        depth.ask_prices[i] = price;
        depth.ask_quantities[i] = qty;
        i++;
    }
    
    return depth;
}

Price OrderBook::vwap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (total_volume_ == 0) return 0;
    
    // Simplified VWAP calculation
    // In production, this would track cumulative volume * price
    return last_trade_price_;
}

Price OrderBook::last_trade_price() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_trade_price_;
}

Quantity OrderBook::total_volume() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_volume_;
}

size_t OrderBook::bid_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bids_.size();
}

size_t OrderBook::ask_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asks_.size();
}

void OrderBook::match_orders() {
    // Simple price-time priority matching
    while (!bids_.empty() && !asks_.empty()) {
        auto best_bid = bids_.begin();
        auto best_ask = asks_.begin();
        
        if (best_bid->first < best_ask->first) {
            // No match possible
            break;
        }
        
        // Match orders
        Price match_price = (best_bid->first + best_ask->first) / 2;
        Quantity match_qty = std::min(best_bid->second, best_ask->second);
        
        // Update quantities
        best_bid->second -= match_qty;
        best_ask->second -= match_qty;
        
        // Remove empty levels
        if (best_bid->second == 0) bids_.erase(best_bid);
        if (best_ask->second == 0) asks_.erase(best_ask);
        
        // Update trade info
        last_trade_price_ = match_price;
        total_volume_ += match_qty;
        
        spdlog::debug("Match: {} qty @ {} price", match_qty, match_price / 100.0);
    }
}

} // namespace niftyquant
