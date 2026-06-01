#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "order_book.h"
#include "execution_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(niftyquant_cpp, m) {
    m.doc() = "NIFTY Quant C++ Execution Engine";
    
    // Enums
    py::enum_<niftyquant::Side>(m, "Side")
        .value("BUY", niftyquant::Side::BUY)
        .value("SELL", niftyquant::Side::SELL)
        .export_values();
        
    py::enum_<niftyquant::OrderType>(m, "OrderType")
        .value("MARKET", niftyquant::OrderType::MARKET)
        .value("LIMIT", niftyquant::OrderType::LIMIT)
        .value("STOP", niftyquant::OrderType::STOP)
        .value("VWAP", niftyquant::OrderType::VWAP)
        .export_values();
        
    py::enum_<niftyquant::OrderStatus>(m, "OrderStatus")
        .value("PENDING", niftyquant::OrderStatus::PENDING)
        .value("PARTIALLY_FILLED", niftyquant::OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", niftyquant::OrderStatus::FILLED)
        .value("CANCELLED", niftyquant::OrderStatus::CANCELLED)
        .value("REJECTED", niftyquant::OrderStatus::REJECTED)
        .export_values();
    
    // Data structures
    py::class_<niftyquant::MarketDepth>(m, "MarketDepth")
        .def(py::init<>())
        .def_readonly("bid_prices", &niftyquant::MarketDepth::bid_prices)
        .def_readonly("bid_quantities", &niftyquant::MarketDepth::bid_quantities)
        .def_readonly("ask_prices", &niftyquant::MarketDepth::ask_prices)
        .def_readonly("ask_quantities", &niftyquant::MarketDepth::ask_quantities)
        .def("best_bid", &niftyquant::MarketDepth::best_bid)
        .def("best_ask", &niftyquant::MarketDepth::best_ask)
        .def("mid_price", &niftyquant::MarketDepth::mid_price)
        .def("spread", &niftyquant::MarketDepth::bid_ask_spread);
        
    py::class_<niftyquant::Fill>(m, "Fill")
        .def(py::init<>())
        .def_readonly("order_id", &niftyquant::Fill::order_id)
        .def_readonly("symbol", &niftyquant::Fill::symbol)
        .def_readonly("side", &niftyquant::Fill::side)
        .def_readonly("fill_price", &niftyquant::Fill::fill_price)
        .def_readonly("fill_quantity", &niftyquant::Fill::fill_quantity)
        .def_readonly("slippage_bps", &niftyquant::Fill::slippage_bps);
        
    py::class_<niftyquant::VWAPParams>(m, "VWAPParams")
        .def(py::init<>())
        .def_readwrite("participation_rate", &niftyquant::VWAPParams::participation_rate)
        .def_readwrite("max_child_orders", &niftyquant::VWAPParams::max_child_orders)
        .def_readwrite("min_fill_size", &niftyquant::VWAPParams::min_fill_size)
        .def_readwrite("time_horizon_ms", &niftyquant::VWAPParams::time_horizon_ms)
        .def_readwrite("urgency", &niftyquant::VWAPParams::urgency)
        .def_readwrite("price_limit_bps", &niftyquant::VWAPParams::price_limit_bps);
        
    py::class_<niftyquant::ExecutionMetrics>(m, "ExecutionMetrics")
        .def(py::init<>())
        .def_readonly("implementation_shortfall_bps", &niftyquant::ExecutionMetrics::implementation_shortfall_bps)
        .def_readonly("average_slippage_bps", &niftyquant::ExecutionMetrics::average_slippage_bps)
        .def_readonly("participation_rate_achieved", &niftyquant::ExecutionMetrics::participation_rate_achieved)
        .def_readonly("vwap_performance_bps", &niftyquant::ExecutionMetrics::vwap_performance_bps)
        .def_readonly("total_child_orders", &niftyquant::ExecutionMetrics::total_child_orders)
        .def_readonly("total_execution_time_ms", &niftyquant::ExecutionMetrics::total_execution_time_ms)
        .def_readonly("total_quantity", &niftyquant::ExecutionMetrics::total_quantity)
        .def_readonly("average_fill_price", &niftyquant::ExecutionMetrics::average_fill_price);
    
    // Order Book
    py::class_<niftyquant::OrderBook, std::shared_ptr<niftyquant::OrderBook>>(m, "OrderBook")
        .def(py::init<const std::string&>())
        .def("add_order", [](niftyquant::OrderBook& book, 
                             const std::string& symbol,
                             niftyquant::Side side, 
                             niftyquant::OrderType type,
                             niftyquant::Price price, 
                             niftyquant::Quantity quantity) {
            auto order = std::make_shared<niftyquant::Order>();
            order->symbol = symbol;
            order->side = side;
            order->type = type;
            order->price = price;
            order->quantity = quantity;
            return book.add_order(order);
        })
        .def("cancel_order", &niftyquant::OrderBook::cancel_order)
        .def("get_depth", &niftyquant::OrderBook::get_depth)
        .def("vwap", &niftyquant::OrderBook::vwap)
        .def("last_trade_price", &niftyquant::OrderBook::last_trade_price)
        .def("total_volume", &niftyquant::OrderBook::total_volume)
        .def("bid_count", &niftyquant::OrderBook::bid_count)
        .def("ask_count", &niftyquant::OrderBook::ask_count)
        .def("symbol", &niftyquant::OrderBook::symbol);
    
    // Execution Engine
    py::class_<niftyquant::ExecutionEngine, std::shared_ptr<niftyquant::ExecutionEngine>>(m, "ExecutionEngine")
        .def(py::init<size_t>(), py::arg("num_threads") = 2)
        .def("start", &niftyquant::ExecutionEngine::start)
        .def("stop", &niftyquant::ExecutionEngine::stop)
        .def("register_order_book", &niftyquant::ExecutionEngine::register_order_book)
        .def("submit_vwap_order", &niftyquant::ExecutionEngine::submit_vwap_order,
             py::arg("symbol"), py::arg("side"), py::arg("quantity"),
             py::arg("signal_price"), py::arg("params") = niftyquant::VWAPParams())
        .def("submit_market_order", &niftyquant::ExecutionEngine::submit_market_order,
             py::arg("symbol"), py::arg("side"), py::arg("quantity"))
        .def("submit_limit_order", &niftyquant::ExecutionEngine::submit_limit_order,
             py::arg("symbol"), py::arg("side"), py::arg("quantity"), py::arg("price"))
        .def("cancel_order", &niftyquant::ExecutionEngine::cancel_order)
        .def("get_metrics", &niftyquant::ExecutionEngine::get_metrics)
        .def("set_fill_callback", [](niftyquant::ExecutionEngine& engine, py::function callback) {
            engine.set_fill_callback([callback](const niftyquant::Fill& fill) {
                try {
                    callback(fill);
                } catch (const py::error_already_set& e) {
                    PyErr_Print();
                }
            });
        });
}
