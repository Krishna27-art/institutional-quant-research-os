#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>

namespace niftyquant {

using Price = int64_t;      // Price in paise (1/100 INR)
using Quantity = uint64_t;
using OrderId = uint64_t;
using Timestamp = int64_t;

enum class Side {
    BUY = 0,
    SELL = 1
};

enum class OrderType {
    MARKET = 0,
    LIMIT = 1,
    STOP = 2,
    VWAP = 3
};

enum class OrderStatus {
    PENDING = 0,
    PARTIALLY_FILLED = 1,
    FILLED = 2,
    CANCELLED = 3,
    REJECTED = 4
};

struct Order {
    OrderId id;
    std::string symbol;
    Side side;
    OrderType type;
    Price price;
    Quantity quantity;
    Quantity filled_quantity = 0;
    OrderStatus status = OrderStatus::PENDING;
    Timestamp timestamp = 0;
    
    Quantity remaining() const {
        return quantity - filled_quantity;
    }
};

struct MarketDepth {
    static constexpr int LEVELS = 5;
    
    Price bid_prices[LEVELS] = {0};
    Quantity bid_quantities[LEVELS] = {0};
    Price ask_prices[LEVELS] = {0};
    Quantity ask_quantities[LEVELS] = {0};
    
    Price best_bid() const { return bid_prices[0]; }
    Price best_ask() const { return ask_prices[0]; }
    Price mid_price() const { 
        if (best_bid() == 0 || best_ask() == 0) return 0;
        return (best_bid() + best_ask()) / 2; 
    }
    Price bid_ask_spread() const {
        if (best_bid() == 0 || best_ask() == 0) return 0;
        return best_ask() - best_bid();
    }
};

struct Fill {
    OrderId order_id;
    std::string symbol;
    Side side;
    Price fill_price;
    Quantity fill_quantity;
    Timestamp timestamp;
    double slippage_bps = 0.0;
};

class OrderBook {
public:
    explicit OrderBook(const std::string& symbol);
    ~OrderBook() = default;
    
    OrderId add_order(std::shared_ptr<Order> order);
    bool cancel_order(OrderId order_id);
    MarketDepth get_depth() const;
    Price vwap() const;
    Price last_trade_price() const;
    Quantity total_volume() const;
    size_t bid_count() const;
    size_t ask_count() const;
    const std::string& symbol() const { return symbol_; }
    
private:
    std::string symbol_;
    
    // Price levels: price -> total quantity
    std::map<Price, Quantity, std::greater<Price>> bids_;  // Descending for bids
    std::map<Price, Quantity> asks_;                       // Ascending for asks
    
    // Order tracking
    std::map<OrderId, std::shared_ptr<Order>> orders_;
    std::atomic<OrderId> next_order_id_{1};
    
    mutable std::mutex mutex_;
    
    Price last_trade_price_ = 0;
    Quantity total_volume_ = 0;
    
    void match_orders();
};

} // namespace niftyquant
