/**
 * Execution Engine Implementation
 */

#include "execution_engine.h"
#include <chrono>
#include <algorithm>

namespace quant_core {

ExecutionEngine::ExecutionEngine()
    : next_order_id_(1)
    , next_fill_id_(1) {
}

bool ExecutionEngine::submit_order(Order& order) {
    // Perform risk check
    RiskCheckResult risk_result;
    if (!_perform_risk_check(order, risk_result)) {
        order.status = OrderStatus::REJECTED;
        _trigger_order_callback(order);
        return false;
    }
    
    // Assign order ID if not set
    if (order.order_id == 0) {
        order.order_id = next_order_id_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Set timestamps
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    
    if (order.created_ns == 0) {
        order.created_ns = now_ns;
    }
    order.updated_ns = now_ns;
    
    // Store order
    orders_[order.order_id] = order;
    
    // Add to symbol index
    symbol_orders_[order.symbol_id].push_back(order.order_id);
    
    // Submit to broker if submission function is set
    if (order_submission_) {
        bool submitted = order_submission_(order);
        if (submitted) {
            _update_order_status(orders_[order.order_id], OrderStatus::SUBMITTED);
        } else {
            _update_order_status(orders_[order.order_id], OrderStatus::REJECTED);
        }
    } else {
        // No broker set, mark as pending
        pending_orders_.push(order.order_id);
        _update_order_status(orders_[order.order_id], OrderStatus::PENDING);
    }
    
    _trigger_order_callback(orders_[order.order_id]);
    
    return true;
}

bool ExecutionEngine::cancel_order(uint64_t order_id) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    Order& order = it->second;
    
    // Check if order can be cancelled
    if (order.status == OrderStatus::FILLED || 
        order.status == OrderStatus::CANCELLED ||
        order.status == OrderStatus::REJECTED) {
        return false;
    }
    
    // Cancel at broker if cancellation function is set
    if (order_cancellation_) {
        bool cancelled = order_cancellation_(order_id);
        if (cancelled) {
            _update_order_status(order, OrderStatus::CANCELLED);
        }
    } else {
        // No broker set, mark as cancelled locally
        _update_order_status(order, OrderStatus::CANCELLED);
    }
    
    _trigger_order_callback(order);
    
    return true;
}

bool ExecutionEngine::modify_order(uint64_t order_id, double new_price, uint64_t new_quantity) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }
    
    Order& order = it->second;
    
    // Check if order can be modified
    if (order.status == OrderStatus::FILLED || 
        order.status == OrderStatus::CANCELLED ||
        order.status == OrderStatus::REJECTED) {
        return false;
    }
    
    // Cancel old order and submit new one
    cancel_order(order_id);
    
    // Create modified order
    Order modified_order = order;
    modified_order.order_id = 0;  // Will get new ID
    modified_order.price = new_price;
    modified_order.quantity = new_quantity;
    modified_order.status = OrderStatus::PENDING;
    
    return submit_order(modified_order);
}

void ExecutionEngine::process_fill(const Fill& fill) {
    auto it = orders_.find(fill.order_id);
    if (it == orders_.end()) {
        return;
    }
    
    Order& order = it->second;
    
    // Update order with fill information
    order.filled_quantity += fill.quantity;
    
    // Update average fill price
    double total_value = order.avg_fill_price * (order.filled_quantity - fill.quantity);
    total_value += fill.price * fill.quantity;
    order.avg_fill_price = total_value / order.filled_quantity;
    
    // Update status
    if (order.filled_quantity >= order.quantity) {
        _update_order_status(order, OrderStatus::FILLED);
    } else {
        _update_order_status(order, OrderStatus::PARTIALLY_FILLED);
    }
    
    order.updated_ns = fill.timestamp_ns;
    
    _trigger_order_callback(order);
    _trigger_fill_callback(fill);
}

const Order* ExecutionEngine::get_order(uint64_t order_id) const {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return nullptr;
    return &it->second;
}

std::vector<Order> ExecutionEngine::get_all_orders() const {
    std::vector<Order> result;
    result.reserve(orders_.size());
    
    for (const auto& [order_id, order] : orders_) {
        result.push_back(order);
    }
    
    return result;
}

std::vector<Order> ExecutionEngine::get_orders_by_symbol(uint32_t symbol_id) const {
    std::vector<Order> result;
    
    auto it = symbol_orders_.find(symbol_id);
    if (it == symbol_orders_.end()) return result;
    
    result.reserve(it->second.size());
    for (uint64_t order_id : it->second) {
        auto order_it = orders_.find(order_id);
        if (order_it != orders_.end()) {
            result.push_back(order_it->second);
        }
    }
    
    return result;
}

std::vector<Order> ExecutionEngine::get_orders_by_status(OrderStatus status) const {
    std::vector<Order> result;
    
    for (const auto& [order_id, order] : orders_) {
        if (order.status == status) {
            result.push_back(order);
        }
    }
    
    return result;
}

size_t ExecutionEngine::get_pending_order_count() const {
    size_t count = 0;
    
    for (const auto& [order_id, order] : orders_) {
        if (order.status == OrderStatus::PENDING || 
            order.status == OrderStatus::SUBMITTED ||
            order.status == OrderStatus::PARTIALLY_FILLED) {
            count++;
        }
    }
    
    return count;
}

size_t ExecutionEngine::get_total_order_count() const {
    return orders_.size();
}

void ExecutionEngine::set_order_callback(OrderCallback callback) {
    order_callback_ = callback;
}

void ExecutionEngine::set_fill_callback(FillCallback callback) {
    fill_callback_ = callback;
}

void ExecutionEngine::set_risk_check(std::function<RiskCheckResult(const Order&)> risk_check) {
    risk_check_ = risk_check;
}

void ExecutionEngine::set_order_submission(std::function<bool(const Order&)> submit_func) {
    order_submission_ = submit_func;
}

void ExecutionEngine::set_order_cancellation(std::function<bool(uint64_t)> cancel_func) {
    order_cancellation_ = cancel_func;
}

void ExecutionEngine::clear() {
    orders_.clear();
    symbol_orders_.clear();
    
    while (!pending_orders_.empty()) {
        pending_orders_.pop();
    }
}

bool ExecutionEngine::_perform_risk_check(const Order& order, RiskCheckResult& result) {
    if (risk_check_) {
        result = risk_check_(order);
        return result.passed;
    }
    
    // Default: pass all orders
    result.passed = true;
    return true;
}

void ExecutionEngine::_update_order_status(Order& order, OrderStatus status) {
    order.status = status;
    order.updated_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

void ExecutionEngine::_trigger_order_callback(const Order& order) {
    if (order_callback_) {
        order_callback_(order);
    }
}

void ExecutionEngine::_trigger_fill_callback(const Fill& fill) {
    if (fill_callback_) {
        fill_callback_(fill);
    }
}

// OrderRouter implementation

OrderRouter::OrderRouter() {
}

void OrderRouter::add_broker(const std::string& broker_id,
                            std::function<bool(const Order&)> submit_func,
                            std::function<bool(uint64_t)> cancel_func) {
    BrokerGateway gateway;
    gateway.submit_func = submit_func;
    gateway.cancel_func = cancel_func;
    
    brokers_[broker_id] = gateway;
    
    // Set as default if first broker
    if (default_broker_.empty()) {
        default_broker_ = broker_id;
    }
}

void OrderRouter::remove_broker(const std::string& broker_id) {
    brokers_.erase(broker_id);
    
    // Update default if needed
    if (default_broker_ == broker_id && !brokers_.empty()) {
        default_broker_ = brokers_.begin()->first;
    }
}

bool OrderRouter::route_order(Order& order, const std::string& broker_id) {
    std::string target_broker = broker_id.empty() ? default_broker_ : broker_id;
    
    auto it = brokers_.find(target_broker);
    if (it == brokers_.end()) {
        return false;
    }
    
    const BrokerGateway& gateway = it->second;
    return gateway.submit_func(order);
}

bool OrderRouter::route_cancellation(uint64_t order_id) {
    // In production, would need to track which broker owns the order
    // For now, use default broker
    auto it = brokers_.find(default_broker_);
    if (it == brokers_.end()) {
        return false;
    }
    
    const BrokerGateway& gateway = it->second;
    return gateway.cancel_func(order_id);
}

std::string OrderRouter::get_default_broker() const {
    return default_broker_;
}

void OrderRouter::set_default_broker(const std::string& broker_id) {
    if (brokers_.find(broker_id) != brokers_.end()) {
        default_broker_ = broker_id;
    }
}

} // namespace quant_core
