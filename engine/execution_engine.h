#pragma once

#include "order_book.h"
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>

namespace niftyquant {

struct VWAPParams {
    double participation_rate = 0.10;
    int max_child_orders = 10;
    Quantity min_fill_size = 50;
    int64_t time_horizon_ms = 300000;  // 5 minutes
    double urgency = 0.5;  // 0.0 = passive, 1.0 = aggressive
    double price_limit_bps = 50.0;
};

struct ExecutionMetrics {
    double implementation_shortfall_bps = 0.0;
    double average_slippage_bps = 0.0;
    double participation_rate_achieved = 0.0;
    double vwap_performance_bps = 0.0;
    int total_child_orders = 0;
    int64_t total_execution_time_ms = 0;
    Quantity total_quantity = 0;
    Price average_fill_price = 0;
};

struct ParentOrder {
    OrderId id;
    std::string symbol;
    Side side;
    Quantity total_quantity;
    Quantity remaining_quantity;
    Price signal_price;
    VWAPParams params;
    int64_t start_time;
    bool is_active = true;
    
    Quantity remaining() const { return remaining_quantity; }
};

class ExecutionEngine {
public:
    explicit ExecutionEngine(size_t num_threads = 2);
    ~ExecutionEngine();
    
    void start();
    void stop();
    
    void register_order_book(std::shared_ptr<OrderBook> book);
    
    // Order submission
    OrderId submit_market_order(
        const std::string& symbol,
        Side side,
        Quantity quantity
    );
    
    OrderId submit_limit_order(
        const std::string& symbol,
        Side side,
        Quantity quantity,
        Price price
    );
    
    OrderId submit_vwap_order(
        const std::string& symbol,
        Side side,
        Quantity quantity,
        Price signal_price,
        const VWAPParams& params = VWAPParams()
    );
    
    bool cancel_order(OrderId order_id);
    
    // Metrics
    ExecutionMetrics get_metrics(OrderId order_id) const;
    
    // Callbacks
    using MarketDataCallback = std::function<void(const std::string&, const MarketDepth&)>;
    using FillCallback = std::function<void(const Fill&)>;
    
    void set_market_data_callback(MarketDataCallback cb);
    void set_fill_callback(FillCallback cb);
    
private:
    void worker_thread();
    void execute_vwap_order(std::shared_ptr<ParentOrder> parent);
    Quantity compute_slice_size(
        const ParentOrder& parent,
        const MarketDepth& depth,
        double volume_forecast
    );
    Price compute_limit_price(
        const ParentOrder& parent,
        const MarketDepth& depth,
        double urgency
    );
    
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    
    std::map<std::string, std::shared_ptr<OrderBook>> order_books_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::map<OrderId, std::shared_ptr<ParentOrder>> parent_orders_;
    std::map<OrderId, ExecutionMetrics> metrics_;
    std::atomic<OrderId> next_parent_order_id_{1000000};
    
    MarketDataCallback market_data_cb_;
    FillCallback fill_cb_;
};

} // namespace niftyquant
