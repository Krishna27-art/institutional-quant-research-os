/**
 * Pybind11 Bindings for Quant Core C++ Engine
 * 
 * This module provides Python bindings for the high-performance C++ core
 * components using pybind11 for zero-copy crossing between Python and C++.
 * 
 * Key Features:
 * - Zero-copy data transfer using py::array_t
 * - Python-accessible C++ classes
 * - Type-safe conversions
 * - Exception handling
 * 
 * Usage:
 * import quant_core
 * 
 * # Create order book
 * book = quant_core.LimitOrderBook(symbol_id=1)
 * book.add_order(order_id=1, is_buy=True, price=100.0, quantity=1000)
 * 
 * # Get best bid/ask
 * best_bid = book.get_best_bid()
 * best_ask = book.get_best_ask()
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "lock_free_ring_buffer.h"
#include "order_book.h"
#include "position_manager.h"
#include "execution_engine.h"
#include "market_replay.h"

namespace py = pybind11;

PYBIND11_MODULE(quant_core, m) {
    m.doc() = "High-Performance Quant Core C++ Engine - Python Bindings";
    
    // ============================================================
    // Lock-Free Ring Buffer
    // ============================================================
    
    py::class_<quant_core::TickData>(m, "TickData")
        .def(py::init<>())
        .def_readwrite("timestamp_ns", &quant_core::TickData::timestamp_ns)
        .def_readwrite("symbol_id", &quant_core::TickData::symbol_id)
        .def_readwrite("price", &quant_core::TickData::price)
        .def_readwrite("volume", &quant_core::TickData::volume)
        .def_readwrite("bid_price", &quant_core::TickData::bid_price)
        .def_readwrite("ask_price", &quant_core::TickData::ask_price)
        .def_readwrite("bid_size", &quant_core::TickData::bid_size)
        .def_readwrite("ask_size", &quant_core::TickData::ask_size);
    
    py::class_<quant_core::TickRingBuffer>(m, "TickRingBuffer")
        .def(py::init<>())
        .def("push", &quant_core::TickRingBuffer::push, 
             py::arg("item"), "Push tick to buffer (returns False if full)")
        .def("pop", &quant_core::TickRingBuffer::pop,
             py::arg("out"), "Pop tick from buffer (returns False if empty)")
        .def("empty", &quant_core::TickRingBuffer::empty, "Check if buffer is empty")
        .def("full", &quant_core::TickRingBuffer::full, "Check if buffer is full")
        .def("size", &quant_core::TickRingBuffer::size, "Get current size")
        .def_static("capacity", &quant_core::TickRingBuffer::capacity, "Get buffer capacity");
    
    // ============================================================
    // Order Book
    // ============================================================
    
    py::enum_<quant_core::OrderSide>(m, "OrderSide")
        .value("BUY", quant_core::OrderSide::BUY)
        .value("SELL", quant_core::OrderSide::SELL);
    
    py::enum_<quant_core::OrderType>(m, "OrderType")
        .value("MARKET", quant_core::OrderType::MARKET)
        .value("LIMIT", quant_core::OrderType::LIMIT)
        .value("STOP", quant_core::OrderType::STOP)
        .value("STOP_LIMIT", quant_core::OrderType::STOP_LIMIT);
    
    py::enum_<quant_core::OrderStatus>(m, "OrderStatus")
        .value("PENDING", quant_core::OrderStatus::PENDING)
        .value("SUBMITTED", quant_core::OrderStatus::SUBMITTED)
        .value("PARTIALLY_FILLED", quant_core::OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", quant_core::OrderStatus::FILLED)
        .value("CANCELLED", quant_core::OrderStatus::CANCELLED)
        .value("REJECTED", quant_core::OrderStatus::REJECTED)
        .value("EXPIRED", quant_core::OrderStatus::EXPIRED);
    
    py::enum_<quant_core::TimeInForce>(m, "TimeInForce")
        .value("DAY", quant_core::TimeInForce::DAY)
        .value("GTC", quant_core::TimeInForce::GTC)
        .value("IOC", quant_core::TimeInForce::IOC)
        .value("FOK", quant_core::TimeInForce::FOK);
    
    py::class_<quant_core::LimitOrderBook>(m, "LimitOrderBook")
        .def(py::init<uint32_t>(), py::arg("symbol_id"))
        .def("add_order", &quant_core::LimitOrderBook::add_order,
             py::arg("order_id"), py::arg("is_buy"), py::arg("price"),
             py::arg("quantity"), py::arg("timestamp_ns"),
             "Add a limit order to the book")
        .def("cancel_order", &quant_core::LimitOrderBook::cancel_order,
             py::arg("order_id"), "Cancel an existing order")
        .def("modify_order", &quant_core::LimitOrderBook::modify_order,
             py::arg("order_id"), py::arg("new_quantity"),
             "Modify order quantity")
        .def("execute_market_order", &quant_core::LimitOrderBook::execute_market_order,
             py::arg("is_buy"), py::arg("quantity"),
             "Execute a market order (cross the book)")
        .def("get_best_bid", &quant_core::LimitOrderBook::get_best_bid,
             "Get best bid price")
        .def("get_best_ask", &quant_core::LimitOrderBook::get_best_ask,
             "Get best ask price")
        .def("get_best_bid_quantity", &quant_core::LimitOrderBook::get_best_bid_quantity,
             "Get best bid quantity")
        .def("get_best_ask_quantity", &quant_core::LimitOrderBook::get_best_ask_quantity,
             "Get best ask quantity")
        .def("get_spread", &quant_core::LimitOrderBook::get_spread,
             "Get bid-ask spread")
        .def("get_mid_price", &quant_core::LimitOrderBook::get_mid_price,
             "Get mid price")
        .def("get_bids", &quant_core::LimitOrderBook::get_bids,
             py::arg("n"), "Get top N bid levels")
        .def("get_asks", &quant_core::LimitOrderBook::get_asks,
             py::arg("n"), "Get top N ask levels")
        .def("get_total_bid_volume", &quant_core::LimitOrderBook::get_total_bid_volume,
             "Get total bid volume")
        .def("get_total_ask_volume", &quant_core::LimitOrderBook::get_total_ask_volume,
             "Get total ask volume")
        .def("get_order_count", &quant_core::LimitOrderBook::get_order_count,
             "Get order count")
        .def("clear", &quant_core::LimitOrderBook::clear, "Clear the book");
    
    py::class_<quant_core::OrderBookManager>(m, "OrderBookManager")
        .def(py::init<>())
        .def("get_order_book", &quant_core::OrderBookManager::get_order_book,
             py::arg("symbol_id"), py::return_value_policy::reference_internal,
             "Get or create order book for symbol")
        .def("remove_order_book", &quant_core::OrderBookManager::remove_order_book,
             py::arg("symbol_id"), "Remove order book")
        .def("get_symbols", &quant_core::OrderBookManager::get_symbols,
             "Get all symbol IDs");
    
    // ============================================================
    // Position Manager
    // ============================================================
    
    py::class_<quant_core::Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("symbol_id", &quant_core::Position::symbol_id)
        .def_readwrite("quantity", &quant_core::Position::quantity)
        .def_readwrite("avg_entry_price", &quant_core::Position::avg_entry_price)
        .def_readwrite("current_price", &quant_core::Position::current_price)
        .def_readwrite("unrealized_pnl", &quant_core::Position::unrealized_pnl)
        .def_readwrite("realized_pnl", &quant_core::Position::realized_pnl)
        .def_readwrite("total_cost", &quant_core::Position::total_cost)
        .def_readwrite("last_update_ns", &quant_core::Position::last_update_ns)
        .def_readwrite("market_value", &quant_core::Position::market_value)
        .def_readwrite("notional", &quant_core::Position::notional)
        .def_readwrite("margin_requirement", &quant_core::Position::margin_requirement);
    
    py::class_<quant_core::PositionManager>(m, "PositionManager")
        .def(py::init<>())
        .def("update_position", &quant_core::PositionManager::update_position,
             py::arg("symbol_id"), py::arg("quantity"), py::arg("price"),
             py::arg("timestamp_ns"), "Update position after a fill")
        .def("update_price", &quant_core::PositionManager::update_price,
             py::arg("symbol_id"), py::arg("price"), py::arg("timestamp_ns"),
             "Update current price for PnL calculation")
        .def("get_position", &quant_core::PositionManager::get_position,
             py::arg("symbol_id"), py::return_value_policy::reference_internal,
             "Get position for symbol")
        .def("get_all_positions", &quant_core::PositionManager::get_all_positions,
             "Get all positions")
        .def("get_total_unrealized_pnl", &quant_core::PositionManager::get_total_unrealized_pnl,
             "Get total unrealized PnL")
        .def("get_total_realized_pnl", &quant_core::PositionManager::get_total_realized_pnl,
             "Get total realized PnL")
        .def("get_total_pnl", &quant_core::PositionManager::get_total_pnl,
             "Get total PnL")
        .def("get_total_notional", &quant_core::PositionManager::get_total_notional,
             "Get total notional exposure")
        .def("get_total_margin_requirement", &quant_core::PositionManager::get_total_margin_requirement,
             "Get total margin requirement")
        .def("get_position_count", &quant_core::PositionManager::get_position_count,
             "Get position count")
        .def("has_position", &quant_core::PositionManager::has_position,
             py::arg("symbol_id"), "Check if position exists")
        .def("close_position", &quant_core::PositionManager::close_position,
             py::arg("symbol_id"), py::arg("price"), py::arg("timestamp_ns"),
             "Close position for symbol")
        .def("clear", &quant_core::PositionManager::clear, "Clear all positions")
        .def("set_margin_multiplier", &quant_core::PositionManager::set_margin_multiplier,
             py::arg("multiplier"), "Set margin multiplier")
        .def("get_margin_multiplier", &quant_core::PositionManager::get_margin_multiplier,
             "Get margin multiplier");
    
    py::class_<quant_core::PortfolioSummary>(m, "PortfolioSummary")
        .def(py::init<>())
        .def_readwrite("total_pnl", &quant_core::PortfolioSummary::total_pnl)
        .def_readwrite("unrealized_pnl", &quant_core::PortfolioSummary::unrealized_pnl)
        .def_readwrite("realized_pnl", &quant_core::PortfolioSummary::realized_pnl)
        .def_readwrite("total_notional", &quant_core::PortfolioSummary::total_notional)
        .def_readwrite("total_margin", &quant_core::PortfolioSummary::total_margin)
        .def_readwrite("position_count", &quant_core::PortfolioSummary::position_count)
        .def_readwrite("last_update_ns", &quant_core::PortfolioSummary::last_update_ns);
    
    py::class_<quant_core::PortfolioManager>(m, "PortfolioManager")
        .def(py::init<>())
        .def("get_position_manager", &quant_core::PortfolioManager::get_position_manager,
             py::arg("asset_class"), py::return_value_policy::reference_internal,
             "Get position manager for asset class")
        .def("get_portfolio_summary", &quant_core::PortfolioManager::get_portfolio_summary,
             "Get portfolio summary")
        .def("get_asset_classes", &quant_core::PortfolioManager::get_asset_classes,
             "Get all asset classes");
    
    // ============================================================
    // Execution Engine
    // ============================================================
    
    py::class_<quant_core::Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id", &quant_core::Order::order_id)
        .def_readwrite("symbol_id", &quant_core::Order::symbol_id)
        .def_readwrite("side", &quant_core::Order::side)
        .def_readwrite("type", &quant_core::Order::type)
        .def_readwrite("price", &quant_core::Order::price)
        .def_readwrite("quantity", &quant_core::Order::quantity)
        .def_readwrite("filled_quantity", &quant_core::Order::filled_quantity)
        .def_readwrite("avg_fill_price", &quant_core::Order::avg_fill_price)
        .def_readwrite("status", &quant_core::Order::status)
        .def_readwrite("tif", &quant_core::Order::tif)
        .def_readwrite("created_ns", &quant_core::Order::created_ns)
        .def_readwrite("updated_ns", &quant_core::Order::updated_ns)
        .def_readwrite("client_order_id", &quant_core::Order::client_order_id)
        .def_readwrite("strategy_id", &quant_core::Order::strategy_id);
    
    py::class_<quant_core::Fill>(m, "Fill")
        .def(py::init<>())
        .def_readwrite("fill_id", &quant_core::Fill::fill_id)
        .def_readwrite("order_id", &quant_core::Fill::order_id)
        .def_readwrite("symbol_id", &quant_core::Fill::symbol_id)
        .def_readwrite("side", &quant_core::Fill::side)
        .def_readwrite("price", &quant_core::Fill::price)
        .def_readwrite("quantity", &quant_core::Fill::quantity)
        .def_readwrite("timestamp_ns", &quant_core::Fill::timestamp_ns)
        .def_readwrite("execution_id", &quant_core::Fill::execution_id)
        .def_readwrite("broker_id", &quant_core::Fill::broker_id);
    
    py::class_<quant_core::RiskCheckResult>(m, "RiskCheckResult")
        .def(py::init<>())
        .def_readwrite("passed", &quant_core::RiskCheckResult::passed)
        .def_readwrite("reason", &quant_core::RiskCheckResult::reason)
        .def_readwrite("available_margin", &quant_core::RiskCheckResult::available_margin)
        .def_readwrite("required_margin", &quant_core::RiskCheckResult::required_margin)
        .def_readwrite("max_position", &quant_core::RiskCheckResult::max_position)
        .def_readwrite("current_position", &quant_core::RiskCheckResult::current_position);
    
    py::class_<quant_core::ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<>())
        .def("submit_order", &quant_core::ExecutionEngine::submit_order,
             py::arg("order"), "Submit an order")
        .def("cancel_order", &quant_core::ExecutionEngine::cancel_order,
             py::arg("order_id"), "Cancel an order")
        .def("modify_order", &quant_core::ExecutionEngine::modify_order,
             py::arg("order_id"), py::arg("new_price"), py::arg("new_quantity"),
             "Modify an order")
        .def("process_fill", &quant_core::ExecutionEngine::process_fill,
             py::arg("fill"), "Process a fill from broker")
        .def("get_order", &quant_core::ExecutionEngine::get_order,
             py::arg("order_id"), py::return_value_policy::reference_internal,
             "Get order by ID")
        .def("get_all_orders", &quant_core::ExecutionEngine::get_all_orders,
             "Get all orders")
        .def("get_orders_by_symbol", &quant_core::ExecutionEngine::get_orders_by_symbol,
             py::arg("symbol_id"), "Get orders by symbol")
        .def("get_orders_by_status", &quant_core::ExecutionEngine::get_orders_by_status,
             py::arg("status"), "Get orders by status")
        .def("get_pending_order_count", &quant_core::ExecutionEngine::get_pending_order_count,
             "Get pending order count")
        .def("get_total_order_count", &quant_core::ExecutionEngine::get_total_order_count,
             "Get total order count")
        .def("set_order_callback", &quant_core::ExecutionEngine::set_order_callback,
             py::arg("callback"), "Set order callback")
        .def("set_fill_callback", &quant_core::ExecutionEngine::set_fill_callback,
             py::arg("callback"), "Set fill callback")
        .def("set_risk_check", &quant_core::ExecutionEngine::set_risk_check,
             py::arg("risk_check"), "Set risk check function")
        .def("set_order_submission", &quant_core::ExecutionEngine::set_order_submission,
             py::arg("submit_func"), "Set order submission function")
        .def("set_order_cancellation", &quant_core::ExecutionEngine::set_order_cancellation,
             py::arg("cancel_func"), "Set order cancellation function")
        .def("clear", &quant_core::ExecutionEngine::clear, "Clear all orders");
    
    // ============================================================
    // Market Replay
    // ============================================================
    
    py::enum_<quant_core::ReplayEventType>(m, "ReplayEventType")
        .value("TICK", quant_core::ReplayEventType::TICK)
        .value("ORDER_ADD", quant_core::ReplayEventType::ORDER_ADD)
        .value("ORDER_CANCEL", quant_core::ReplayEventType::ORDER_CANCEL)
        .value("ORDER_MODIFY", quant_core::ReplayEventType::ORDER_MODIFY)
        .value("TRADE", quant_core::ReplayEventType::TRADE)
        .value("CORPORATE_ACTION", quant_core::ReplayEventType::CORPORATE_ACTION)
        .value("EARNINGS_ANNOUNCEMENT", quant_core::ReplayEventType::EARNINGS_ANNOUNCEMENT);
    
    py::class_<quant_core::ReplayEvent>(m, "ReplayEvent")
        .def(py::init<>())
        .def_readwrite("type", &quant_core::ReplayEvent::type)
        .def_readwrite("timestamp_ns", &quant_core::ReplayEvent::timestamp_ns)
        .def_readwrite("symbol_id", &quant_core::ReplayEvent::symbol_id);
    
    py::class_<quant_core::PlaybackControl>(m, "PlaybackControl")
        .def(py::init<>())
        .def_readwrite("paused", &quant_core::PlaybackControl::paused)
        .def_readwrite("speed_multiplier", &quant_core::PlaybackControl::speed_multiplier)
        .def_readwrite("start_timestamp_ns", &quant_core::PlaybackControl::start_timestamp_ns)
        .def_readwrite("end_timestamp_ns", &quant_core::PlaybackControl::end_timestamp_ns)
        .def_readwrite("current_timestamp_ns", &quant_core::PlaybackControl::current_timestamp_ns);
    
    py::class_<quant_core::MarketReplayEngine>(m, "MarketReplayEngine")
        .def(py::init<>())
        .def("load_data", &quant_core::MarketReplayEngine::load_data,
             py::arg("symbol_id"), py::arg("events"),
             "Load historical data for replay")
        .def("load_from_file", &quant_core::MarketReplayEngine::load_from_file,
             py::arg("symbol_id"), py::arg("filepath"),
             "Load data from file (binary format)")
        .def("save_to_file", &quant_core::MarketReplayEngine::save_to_file,
             py::arg("symbol_id"), py::arg("filepath"),
             "Save data to file (binary format)")
        .def("start_playback", &quant_core::MarketReplayEngine::start_playback,
             py::arg("control"), "Start playback")
        .def("stop_playback", &quant_core::MarketReplayEngine::stop_playback,
             "Stop playback")
        .def("pause_playback", &quant_core::MarketReplayEngine::pause_playback,
             "Pause playback")
        .def("resume_playback", &quant_core::MarketReplayEngine::resume_playback,
             "Resume playback")
        .def("seek_to", &quant_core::MarketReplayEngine::seek_to,
             py::arg("timestamp_ns"), "Seek to specific timestamp")
        .def("step_forward", &quant_core::MarketReplayEngine::step_forward,
             py::arg("n") = 1, "Step forward by N events")
        .def("step_backward", &quant_core::MarketReplayEngine::step_backward,
             py::arg("n") = 1, "Step backward by N events")
        .def("get_current_timestamp", &quant_core::MarketReplayEngine::get_current_timestamp,
             "Get current playback position")
        .def("is_playing", &quant_core::MarketReplayEngine::is_playing,
             "Get playback status")
        .def("is_paused", &quant_core::MarketReplayEngine::is_paused,
             "Get pause status")
        .def("set_event_callback", &quant_core::MarketReplayEngine::set_event_callback,
             py::arg("callback"), "Set event callback")
        .def("get_event_count", &quant_core::MarketReplayEngine::get_event_count,
             py::arg("symbol_id"), "Get event count for symbol")
        .def("get_total_event_count", &quant_core::MarketReplayEngine::get_total_event_count,
             "Get total event count")
        .def("get_symbols", &quant_core::MarketReplayEngine::get_symbols,
             "Get symbols")
        .def("clear", &quant_core::MarketReplayEngine::clear, "Clear all data")
        .def("get_event_at", &quant_core::MarketReplayEngine::get_event_at,
             py::arg("symbol_id"), py::arg("index"),
             py::return_value_policy::reference_internal,
             "Get event at index for symbol");
    
    py::class_<quant_core::ReplayCoordinator>(m, "ReplayCoordinator")
        .def(py::init<>())
        .def("add_engine", &quant_core::ReplayCoordinator::add_engine,
             py::arg("symbol_id"), py::arg("engine"),
             "Add a replay engine for a symbol")
        .def("remove_engine", &quant_core::ReplayCoordinator::remove_engine,
             py::arg("symbol_id"), "Remove a replay engine")
        .def("start_synchronized_playback", &quant_core::ReplayCoordinator::start_synchronized_playback,
             py::arg("control"), "Start synchronized playback across all engines")
        .def("stop_all_playback", &quant_core::ReplayCoordinator::stop_all_playback,
             "Stop all playback")
        .def("pause_all_playback", &quant_core::ReplayCoordinator::pause_all_playback,
             "Pause all playback")
        .def("resume_all_playback", &quant_core::ReplayCoordinator::resume_all_playback,
             "Resume all playback")
        .def("seek_all_to", &quant_core::ReplayCoordinator::seek_all_to,
             py::arg("timestamp_ns"), "Seek all engines to timestamp")
        .def("get_all_engines", &quant_core::ReplayCoordinator::get_all_engines,
             "Get all engines");
}
