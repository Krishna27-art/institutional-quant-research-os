#include "execution_engine.h"
#include <spdlog/spdlog.h>
#include <random>
#include <cmath>

namespace niftyquant {

ExecutionEngine::ExecutionEngine(size_t num_threads) 
    : workers_(num_threads) {
    spdlog::info("ExecutionEngine initialized with {} threads", num_threads);
}

ExecutionEngine::~ExecutionEngine() {
    stop();
}

void ExecutionEngine::start() {
    running_ = true;
    for (auto& worker : workers_) {
        worker = std::thread(&ExecutionEngine::worker_thread, this);
    }
    spdlog::info("ExecutionEngine started");
}

void ExecutionEngine::stop() {
    running_ = false;
    queue_cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    spdlog::info("ExecutionEngine stopped");
}

void ExecutionEngine::register_order_book(std::shared_ptr<OrderBook> book) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    order_books_[book->symbol()] = book;
    spdlog::info("Registered order book for: {}", book->symbol());
}

OrderId ExecutionEngine::submit_market_order(
    const std::string& symbol,
    Side side,
    Quantity quantity
) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto it = order_books_.find(symbol);
    if (it == order_books_.end()) {
        spdlog::error("Order book not found for symbol: {}", symbol);
        return 0;
    }
    
    auto order = std::make_shared<Order>();
    order->symbol = symbol;
    order->side = side;
    order->type = OrderType::MARKET;
    order->quantity = quantity;
    order->price = 0;  // Market orders have no price limit
    
    OrderId order_id = it->second->add_order(order);
    
    // Simulate immediate fill for market order
    if (fill_cb_) {
        Fill fill;
        fill.order_id = order_id;
        fill.symbol = symbol;
        fill.side = side;
        fill.fill_quantity = quantity;
        fill.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        
        // Get current price from order book
        auto depth = it->second->get_depth();
        fill.fill_price = side == Side::BUY ? depth.best_ask() : depth.best_bid();
        
        fill_cb_(fill);
    }
    
    return order_id;
}

OrderId ExecutionEngine::submit_limit_order(
    const std::string& symbol,
    Side side,
    Quantity quantity,
    Price price
) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto it = order_books_.find(symbol);
    if (it == order_books_.end()) {
        spdlog::error("Order book not found for symbol: {}", symbol);
        return 0;
    }
    
    auto order = std::make_shared<Order>();
    order->symbol = symbol;
    order->side = side;
    order->type = OrderType::LIMIT;
    order->quantity = quantity;
    order->price = price;
    
    return it->second->add_order(order);
}

OrderId ExecutionEngine::submit_vwap_order(
    const std::string& symbol,
    Side side,
    Quantity quantity,
    Price signal_price,
    const VWAPParams& params
) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    auto it = order_books_.find(symbol);
    if (it == order_books_.end()) {
        spdlog::error("Order book not found for symbol: {}", symbol);
        return 0;
    }
    
    auto parent = std::make_shared<ParentOrder>();
    parent->id = next_parent_order_id_++;
    parent->symbol = symbol;
    parent->side = side;
    parent->total_quantity = quantity;
    parent->remaining_quantity = quantity;
    parent->signal_price = signal_price;
    parent->params = params;
    parent->start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    
    parent_orders_[parent->id] = parent;
    metrics_[parent->id] = ExecutionMetrics();
    
    // Execute VWAP order in worker thread
    std::thread executor(&ExecutionEngine::execute_vwap_order, this, parent);
    executor.detach();
    
    spdlog::info("VWAP order submitted: {} {} {} @ signal {}",
                 symbol, side == Side::BUY ? "BUY" : "SELL",
                 quantity, signal_price / 100.0);
    
    return parent->id;
}

bool ExecutionEngine::cancel_order(OrderId order_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Check parent orders first
    auto parent_it = parent_orders_.find(order_id);
    if (parent_it != parent_orders_.end()) {
        parent_it->second->is_active = false;
        spdlog::info("Parent order cancelled: {}", order_id);
        return true;
    }
    
    // Check child orders in order books
    for (auto& [symbol, book] : order_books_) {
        if (book->cancel_order(order_id)) {
            return true;
        }
    }
    
    return false;
}

ExecutionMetrics ExecutionEngine::get_metrics(OrderId order_id) const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    auto it = metrics_.find(order_id);
    if (it != metrics_.end()) {
        return it->second;
    }
    return ExecutionMetrics();
}

void ExecutionEngine::set_market_data_callback(MarketDataCallback cb) {
    market_data_cb_ = std::move(cb);
}

void ExecutionEngine::set_fill_callback(FillCallback cb) {
    fill_cb_ = std::move(cb);
}

void ExecutionEngine::worker_thread() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !running_; });
    }
}

void ExecutionEngine::execute_vwap_order(std::shared_ptr<ParentOrder> parent) {
    auto it = order_books_.find(parent->symbol);
    if (it == order_books_.end()) {
        spdlog::error("Order book not found for VWAP order: {}", parent->symbol);
        return;
    }
    
    auto book = it->second;
    auto& m = metrics_[parent->id];
    
    int64_t elapsed = 0;
    int64_t total_time = parent->params.time_horizon_ms;
    int slice_delay_ms = total_time / parent->params.max_child_orders;
    
    Quantity total_filled = 0;
    Price weighted_price_sum = 0;
    
    while (parent->is_active && parent->remaining() > 0 && elapsed < total_time) {
        auto depth = book->get_depth();
        
        // Simulate volume forecast (in production, get from market data)
        double volume_forecast = 100000.0;  // Placeholder
        
        // Compute slice size
        Quantity slice_size = compute_slice_size(*parent, depth, volume_forecast);
        
        // Compute limit price
        double urgency = parent->params.urgency;
        urgency += (elapsed / total_time) * 0.3;  // Increase urgency over time
        urgency = std::min(urgency, 1.0);
        
        Price limit_price = compute_limit_price(*parent, depth, urgency);
        
        // Submit child order
        auto child_order = std::make_shared<Order>();
        child_order->symbol = parent->symbol;
        child_order->side = parent->side;
        child_order->type = OrderType::LIMIT;
        child_order->quantity = slice_size;
        child_order->price = limit_price;
        
        OrderId child_id = book->add_order(child_order);
        m.total_child_orders++;
        
        // Simulate fill (in production, wait for actual fill)
        Quantity filled_this_slice = slice_size;
        Price fill_price = limit_price;
        
        // Update parent
        parent->remaining_quantity -= filled_this_slice;
        total_filled += filled_this_slice;
        weighted_price_sum += fill_price * filled_this_slice;
        
        // Update metrics
        m.total_quantity = total_filled;
        m.average_fill_price = weighted_price_sum / total_filled;
        
        if (filled_this_slice > 0) {
            m.average_fill_price = 
                (m.average_fill_price * (parent->total_quantity - filled_this_slice) + weighted_price_sum) 
                / parent->total_quantity;
        }
        
        // Check completion
        if (parent->remaining() == 0) {
            parent->is_active = false;
            
            // Finalize metrics
            auto end_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();
            m.total_execution_time_ms = (end_time - parent->start_time) / 1'000'000;
            
            // Implementation shortfall
            double ideal_cost = static_cast<double>(parent->signal_price) * parent->total_quantity;
            double actual_cost = static_cast<double>(m.average_fill_price) * parent->total_quantity;
            m.implementation_shortfall_bps = 
                (actual_cost - ideal_cost) / ideal_cost * 10000.0;
            if (parent->side == Side::SELL) m.implementation_shortfall_bps = -m.implementation_shortfall_bps;
            
            spdlog::info(
                "VWAP Order {} completed: {} qty @ {} avg price, IS: {:.2f} bps",
                parent->id, parent->total_quantity, 
                m.average_fill_price / 100.0, m.implementation_shortfall_bps
            );
            break;
        }
        
        // Wait for next slice interval
        std::this_thread::sleep_for(std::chrono::milliseconds(slice_delay_ms));
        elapsed += slice_delay_ms;
    }
}

Quantity ExecutionEngine::compute_slice_size(
    const ParentOrder& parent,
    const MarketDepth& depth,
    double volume_forecast
) {
    // Base slice: even distribution
    Quantity base_slice = parent.remaining() / 5; // Remaining divided by rough slices left
    
    // Participation rate constraint
    Quantity participation_slice = static_cast<Quantity>(
        volume_forecast * parent.params.participation_rate
    );
    
    // Depth constraint (don't eat more than top 3 levels typically)
    Quantity available_depth = 0;
    if (parent.side == Side::BUY) {
        for (int i = 0; i < 3; ++i) {
            available_depth += depth.ask_quantities[i];
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            available_depth += depth.bid_quantities[i];
        }
    }
    
    Quantity depth_slice = static_cast<Quantity>(available_depth * 0.5); // Take max 50% of available depth
    
    // Take the minimum of all constraints
    Quantity slice_size = std::min({base_slice, participation_slice, depth_slice});
    
    // Enforce minimum fill size
    slice_size = std::max(slice_size, parent.params.min_fill_size);
    
    // Don't exceed remaining
    slice_size = std::min(slice_size, parent.remaining());
    
    return slice_size;
}

Price ExecutionEngine::compute_limit_price(
    const ParentOrder& parent,
    const MarketDepth& depth,
    double urgency
) {
    Price best_bid = depth.best_bid();
    Price best_ask = depth.best_ask();
    
    if (best_bid == 0 || best_ask == 0) {
        return parent.signal_price; // Fallback
    }
    
    Price spread = best_ask - best_bid;
    Price mid = (best_bid + best_ask) / 2;
    
    if (parent.side == Side::BUY) {
        // Urgency 0: Join bid (passive)
        // Urgency 0.5: Mid-price
        // Urgency 1: Join ask (aggressive/cross)
        Price passive = best_bid;
        Price aggressive = best_ask;
        
        return static_cast<Price>(
            passive + (aggressive - passive) * urgency
        );
    } else {
        // Selling
        Price passive = best_ask;
        Price aggressive = best_bid;
        
        return static_cast<Price>(
            passive - (passive - aggressive) * urgency
        );
    }
}

} // namespace niftyquant
