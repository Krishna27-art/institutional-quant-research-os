#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <vector>
#include <tuple>
#include <stdexcept>

namespace py = pybind11;

// Limit Order Book using std::map for fast O(log N) price level lookups 
// and ordered traversal for L2 depth snapshots
class CPPOrderBook {
private:
    // price -> volume
    std::map<double, int, std::greater<double>> bids; // sorted descending
    std::map<double, int, std::less<double>> asks;    // sorted ascending

public:
    CPPOrderBook() {}

    void add_order(const std::string& side, double price, int volume) {
        if (side == "bid" || side == "buy") {
            bids[price] += volume;
            if (bids[price] <= 0) bids.erase(price);
        } else if (side == "ask" || side == "sell") {
            asks[price] += volume;
            if (asks[price] <= 0) asks.erase(price);
        } else {
            throw std::invalid_argument("side must be bid or ask");
        }
    }

    void remove_order(const std::string& side, double price, int volume) {
        add_order(side, price, -volume);
    }

    std::tuple<double, int> get_best_bid() const {
        if (bids.empty()) return {0.0, 0};
        auto it = bids.begin();
        return {it->first, it->second};
    }

    std::tuple<double, int> get_best_ask() const {
        if (asks.empty()) return {0.0, 0};
        auto it = asks.begin();
        return {it->first, it->second};
    }

    double get_mid_price() const {
        auto best_bid = get_best_bid();
        auto best_ask = get_best_ask();
        if (std::get<0>(best_bid) == 0.0 || std::get<0>(best_ask) == 0.0) return 0.0;
        return (std::get<0>(best_bid) + std::get<0>(best_ask)) / 2.0;
    }

    std::vector<std::tuple<double, int>> get_l2_bids(int levels = 5) const {
        std::vector<std::tuple<double, int>> result;
        int count = 0;
        for (auto const& [price, vol] : bids) {
            result.push_back({price, vol});
            if (++count >= levels) break;
        }
        return result;
    }

    std::vector<std::tuple<double, int>> get_l2_asks(int levels = 5) const {
        std::vector<std::tuple<double, int>> result;
        int count = 0;
        for (auto const& [price, vol] : asks) {
            result.push_back({price, vol});
            if (++count >= levels) break;
        }
        return result;
    }

    void clear() {
        bids.clear();
        asks.clear();
    }
};

PYBIND11_MODULE(cpp_order_book, m) {
    m.doc() = "C++ Limit Order Book for High-Performance Execution";
    
    py::class_<CPPOrderBook>(m, "CPPOrderBook")
        .def(py::init<>())
        .def("add_order", &CPPOrderBook::add_order)
        .def("remove_order", &CPPOrderBook::remove_order)
        .def("get_best_bid", &CPPOrderBook::get_best_bid)
        .def("get_best_ask", &CPPOrderBook::get_best_ask)
        .def("get_mid_price", &CPPOrderBook::get_mid_price)
        .def("get_l2_bids", &CPPOrderBook::get_l2_bids, py::arg("levels") = 5)
        .def("get_l2_asks", &CPPOrderBook::get_l2_asks, py::arg("levels") = 5)
        .def("clear", &CPPOrderBook::clear);
}
