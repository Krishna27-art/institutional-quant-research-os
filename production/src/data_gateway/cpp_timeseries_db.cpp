#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

struct Tick {
    double timestamp;
    double price;
    double volume;
};

class CPPTickBuffer {
private:
    std::string symbol;
    std::vector<Tick> buffer;
    size_t head;
    size_t count;
    size_t capacity;

public:
    CPPTickBuffer(const std::string& symbol, size_t max_size = 100000) 
        : symbol(symbol), capacity(max_size), head(0), count(0) {
        buffer.resize(max_size);
    }

    void add_tick(double timestamp, double price, double volume) {
        buffer[head] = {timestamp, price, volume};
        head = (head + 1) % capacity;
        if (count < capacity) {
            count++;
        }
    }

    std::vector<std::tuple<double, double, double>> get_ticks(size_t limit = 1000) const {
        std::vector<std::tuple<double, double, double>> result;
        size_t return_count = std::min(count, limit);
        result.reserve(return_count);
        
        size_t start_idx = (head + capacity - return_count) % capacity;
        
        for (size_t i = 0; i < return_count; ++i) {
            size_t idx = (start_idx + i) % capacity;
            result.push_back({buffer[idx].timestamp, buffer[idx].price, buffer[idx].volume});
        }
        return result;
    }

    // Returns OHLCV: open, high, low, close, volume for the last `duration` seconds
    std::tuple<double, double, double, double, double> aggregate_ohlcv(double duration_sec) const {
        if (count == 0) return {0.0, 0.0, 0.0, 0.0, 0.0};

        size_t latest_idx = (head + capacity - 1) % capacity;
        double current_time = buffer[latest_idx].timestamp;
        double start_time = current_time - duration_sec;

        // Traverse backwards
        double close = buffer[latest_idx].price;
        double open = close;
        double high = close;
        double low = close;
        double volume = 0.0;
        
        size_t i = 0;
        while (i < count) {
            size_t idx = (head + capacity - 1 - i) % capacity;
            if (buffer[idx].timestamp < start_time) {
                break;
            }
            double p = buffer[idx].price;
            open = p; // The oldest tick in the window will end up being the open
            if (p > high) high = p;
            if (p < low) low = p;
            volume += buffer[idx].volume;
            i++;
        }

        return {open, high, low, close, volume};
    }

    size_t size() const { return count; }
    
    void clear() {
        head = 0;
        count = 0;
    }
};

PYBIND11_MODULE(cpp_timeseries_db, m) {
    m.doc() = "C++ Timeseries Tick DB for High-Frequency Data Storage";

    py::class_<CPPTickBuffer>(m, "CPPTickBuffer")
        .def(py::init<std::string, size_t>(), py::arg("symbol"), py::arg("max_size") = 100000)
        .def("add_tick", &CPPTickBuffer::add_tick, py::arg("timestamp"), py::arg("price"), py::arg("volume"))
        .def("get_ticks", &CPPTickBuffer::get_ticks, py::arg("limit") = 1000)
        .def("aggregate_ohlcv", &CPPTickBuffer::aggregate_ohlcv, py::arg("duration_sec"))
        .def("size", &CPPTickBuffer::size)
        .def("clear", &CPPTickBuffer::clear);
}
