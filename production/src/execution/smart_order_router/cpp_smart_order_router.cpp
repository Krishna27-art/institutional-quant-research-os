#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace py = pybind11;

struct BrokerMetrics {
    double fill_rate = 0.95;
    double latency_ms = 100.0;
    double cost_bps = 5.0;
};

class CPPSmartOrderRouter {
private:
    std::vector<std::string> brokers;
    std::unordered_map<std::string, BrokerMetrics> metrics;

public:
    CPPSmartOrderRouter() {}

    void add_broker(const std::string& broker_name) {
        if (std::find(brokers.begin(), brokers.end(), broker_name) == brokers.end()) {
            brokers.push_back(broker_name);
            metrics[broker_name] = BrokerMetrics();
        }
    }

    void remove_broker(const std::string& broker_name) {
        auto it = std::find(brokers.begin(), brokers.end(), broker_name);
        if (it != brokers.end()) {
            brokers.erase(it);
            metrics.erase(broker_name);
        }
    }

    void update_broker_metrics(const std::string& broker_name, 
                               double fill_rate, 
                               double latency_ms, 
                               double cost_bps) {
        if (metrics.find(broker_name) == metrics.end()) {
            add_broker(broker_name);
        }
        metrics[broker_name].fill_rate = fill_rate;
        metrics[broker_name].latency_ms = latency_ms;
        metrics[broker_name].cost_bps = cost_bps;
    }

    // Scores broker and returns best broker name
    // quotes map is broker_name -> tuple(bid, ask)
    std::string route(const std::string& side, const std::unordered_map<std::string, std::pair<double, double>>& quotes) {
        if (brokers.empty()) {
            throw std::runtime_error("No brokers available");
        }

        std::string best_broker = brokers[0];
        double best_score = -1e9;

        for (const auto& broker : brokers) {
            auto q_it = quotes.find(broker);
            if (q_it == quotes.end()) continue;

            double bid = q_it->second.first;
            double ask = q_it->second.second;

            double score = 0.0;
            if (side == "buy" || side == "bid") {
                if (ask > 0) score += 10.0; // In a real system, you'd compare the actual ask price
            } else {
                if (bid > 0) score += 10.0;
            }

            const auto& m = metrics[broker];
            score += m.fill_rate * 5.0;
            score -= std::min(m.latency_ms / 100.0, 3.0);
            score -= std::min(m.cost_bps / 5.0, 2.0);

            if (score > best_score) {
                best_score = score;
                best_broker = broker;
            }
        }

        return best_broker;
    }
};

PYBIND11_MODULE(cpp_sor, m) {
    m.doc() = "C++ Smart Order Router";

    py::class_<CPPSmartOrderRouter>(m, "CPPSmartOrderRouter")
        .def(py::init<>())
        .def("add_broker", &CPPSmartOrderRouter::add_broker, py::arg("broker_name"))
        .def("remove_broker", &CPPSmartOrderRouter::remove_broker, py::arg("broker_name"))
        .def("update_broker_metrics", &CPPSmartOrderRouter::update_broker_metrics,
             py::arg("broker_name"), py::arg("fill_rate"), py::arg("latency_ms"), py::arg("cost_bps"))
        .def("route", &CPPSmartOrderRouter::route, py::arg("side"), py::arg("quotes"));
}
