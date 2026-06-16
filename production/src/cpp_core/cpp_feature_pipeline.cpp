/*
 * C++ Feature Pipeline for High-Performance Feature Computation
 * Compiled with pybind11 for Python integration
 * 
 * Optimized with Zero-Copy py::array_t
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>

namespace py = pybind11;

// Rolling window statistics
class RollingStatistics {
private:
    std::deque<double> window;
    size_t window_size;
    double sum;
    double sum_sq;

public:
    RollingStatistics(size_t size) : window_size(size), sum(0.0), sum_sq(0.0) {}
    
    void add(double value) {
        if (window.size() == window_size) {
            sum -= window.front();
            sum_sq -= window.front() * window.front();
            window.pop_front();
        }
        window.push_back(value);
        sum += value;
        sum_sq += value * value;
    }
    
    double mean() const {
        return window.empty() ? 0.0 : sum / window.size();
    }
    
    double std() const {
        if (window.size() < 2) return 0.0;
        double mean_val = mean();
        double variance = (sum_sq / window.size()) - (mean_val * mean_val);
        return variance > 0 ? std::sqrt(variance) : 0.0;
    }
    
    double skew() const {
        if (window.size() < 3) return 0.0;
        double mean_val = mean();
        double std_val = std();
        if (std_val == 0.0) return 0.0;
        
        double sum_cubed = 0.0;
        for (double val : window) {
            double z = (val - mean_val) / std_val;
            sum_cubed += z * z * z;
        }
        
        return sum_cubed / window.size();
    }
    
    double kurtosis() const {
        if (window.size() < 4) return 0.0;
        double mean_val = mean();
        double std_val = std();
        if (std_val == 0.0) return 0.0;
        
        double sum_quart = 0.0;
        for (double val : window) {
            double z = (val - mean_val) / std_val;
            sum_quart += z * z * z * z;
        }
        
        return (sum_quart / window.size()) - 3.0;
    }
    
    size_t size() const { return window.size(); }
};

// Technical indicators
class TechnicalIndicators {
public:
    static py::array_t<double> rsi(py::array_t<double> prices_array, size_t period = 14) {
        py::buffer_info buf = prices_array.request();
        double* prices = static_cast<double*>(buf.ptr);
        size_t size = buf.shape[0];
        
        auto result_array = py::array_t<double>(size);
        py::buffer_info res_buf = result_array.request();
        double* result = static_cast<double*>(res_buf.ptr);
        
        if (size < period + 1) {
            for(size_t i=0; i<size; ++i) result[i] = 0.0;
            return result_array;
        }
        
        std::vector<double> gains(size, 0.0);
        std::vector<double> losses(size, 0.0);
        for (size_t i = 1; i < size; i++) {
            double change = prices[i] - prices[i-1];
            if (change > 0) {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }
        
        double avg_gain = 0.0, avg_loss = 0.0;
        for (size_t i = 1; i <= period; i++) {
            avg_gain += gains[i];
            avg_loss += losses[i];
        }
        avg_gain /= period;
        avg_loss /= period;
        
        for(size_t i=0; i<period; ++i) result[i] = 50.0; // Initial
        
        if (avg_loss == 0.0) {
            result[period] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }
        
        for (size_t i = period + 1; i < size; i++) {
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period;
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period;
            
            if (avg_loss == 0.0) {
                result[i] = 100.0;
            } else {
                double rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        
        return result_array;
    }
    
    static py::array_t<double> macd(py::array_t<double> prices_array, 
                                     size_t fast = 12, size_t slow = 26, size_t signal = 9) {
        py::buffer_info buf = prices_array.request();
        double* prices = static_cast<double*>(buf.ptr);
        size_t size = buf.shape[0];
        
        auto result_array = py::array_t<double>(size);
        py::buffer_info res_buf = result_array.request();
        double* result = static_cast<double*>(res_buf.ptr);
        
        if (size < slow + signal) {
            for(size_t i=0; i<size; ++i) result[i] = 0.0;
            return result_array;
        }
        
        auto ema = [](const double* data, size_t len, size_t period) {
            std::vector<double> ema_values(len, 0.0);
            if (len == 0) return ema_values;
            double multiplier = 2.0 / (period + 1);
            ema_values[0] = data[0];
            for (size_t i = 1; i < len; i++) {
                ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1];
            }
            return ema_values;
        };
        
        std::vector<double> ema_fast = ema(prices, size, fast);
        std::vector<double> ema_slow = ema(prices, size, slow);
        
        std::vector<double> macd_line(size, 0.0);
        for (size_t i = 0; i < size; i++) {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }
        
        std::vector<double> signal_line = ema(macd_line.data(), size, signal);
        
        for (size_t i = 0; i < size; i++) {
            result[i] = macd_line[i] - signal_line[i];
        }
        
        return result_array;
    }
    
    static py::tuple bollinger_bands(py::array_t<double> prices_array, size_t period = 20, double std_dev = 2.0) {
        py::buffer_info buf = prices_array.request();
        double* prices = static_cast<double*>(buf.ptr);
        size_t size = buf.shape[0];
        
        auto upper_array = py::array_t<double>(size);
        auto middle_array = py::array_t<double>(size);
        auto lower_array = py::array_t<double>(size);
        
        double* upper = static_cast<double*>(upper_array.request().ptr);
        double* middle = static_cast<double*>(middle_array.request().ptr);
        double* lower = static_cast<double*>(lower_array.request().ptr);
        
        RollingStatistics stats(period);
        
        for (size_t i = 0; i < size; i++) {
            stats.add(prices[i]);
            if (stats.size() >= period) {
                double m = stats.mean();
                double s = stats.std();
                middle[i] = m;
                upper[i] = m + std_dev * s;
                lower[i] = m - std_dev * s;
            } else {
                middle[i] = prices[i];
                upper[i] = prices[i];
                lower[i] = prices[i];
            }
        }
        
        return py::make_tuple(upper_array, middle_array, lower_array);
    }
};

// Microstructure features
class MicrostructureFeatures {
public:
    static py::array_t<double> ofi(py::array_t<double> bid_prices_arr,
                                   py::array_t<double> ask_prices_arr,
                                   py::array_t<int> bid_volumes_arr,
                                   py::array_t<int> ask_volumes_arr) {
        
        size_t size = bid_prices_arr.request().shape[0];
        double* bp = static_cast<double*>(bid_prices_arr.request().ptr);
        double* ap = static_cast<double*>(ask_prices_arr.request().ptr);
        int* bv = static_cast<int*>(bid_volumes_arr.request().ptr);
        int* av = static_cast<int*>(ask_volumes_arr.request().ptr);
        
        auto result_array = py::array_t<double>(size);
        double* result = static_cast<double*>(result_array.request().ptr);
        
        for (size_t i = 0; i < size; i++) {
            double bid_vol = static_cast<double>(bv[i]);
            double ask_vol = static_cast<double>(av[i]);
            
            if (bid_vol + ask_vol == 0.0) {
                result[i] = 0.0;
            } else {
                result[i] = (bid_vol - ask_vol) / (bid_vol + ask_vol);
            }
        }
        
        return result_array;
    }
    
    static py::array_t<double> vpin(py::array_t<double> buy_volumes_arr,
                                    py::array_t<double> sell_volumes_arr,
                                    double bucket_size = 1000000.0) {
        size_t size = buy_volumes_arr.request().shape[0];
        double* bv = static_cast<double*>(buy_volumes_arr.request().ptr);
        double* sv = static_cast<double*>(sell_volumes_arr.request().ptr);
        
        auto result_array = py::array_t<double>(size);
        double* result = static_cast<double*>(result_array.request().ptr);
        
        double current_bucket = 0.0;
        double buy_bucket = 0.0;
        double sell_bucket = 0.0;
        double last_vpin = 0.0;
        
        for (size_t i = 0; i < size; i++) {
            current_bucket += bv[i] + sv[i];
            buy_bucket += bv[i];
            sell_bucket += sv[i];
            
            if (current_bucket >= bucket_size) {
                last_vpin = std::abs(buy_bucket - sell_bucket) / current_bucket;
                current_bucket = 0.0;
                buy_bucket = 0.0;
                sell_bucket = 0.0;
            }
            result[i] = last_vpin;
        }
        
        return result_array;
    }
};

// Cross-sectional features
class CrossSectionalFeatures {
public:
    static py::array_t<double> zscore(py::array_t<double> values_arr) {
        size_t size = values_arr.request().shape[0];
        double* values = static_cast<double*>(values_arr.request().ptr);
        
        auto result_array = py::array_t<double>(size);
        double* result = static_cast<double*>(result_array.request().ptr);
        
        if (size == 0) return result_array;
        
        double sum = 0.0;
        for(size_t i=0; i<size; ++i) sum += values[i];
        double mean = sum / size;
        
        double sq_sum = 0.0;
        for(size_t i=0; i<size; ++i) sq_sum += values[i] * values[i];
        
        double variance = sq_sum / size - mean * mean;
        double std_dev = variance > 0 ? std::sqrt(variance) : 0.0;
        
        for (size_t i=0; i<size; ++i) {
            if (std_dev == 0.0) {
                result[i] = 0.0;
            } else {
                result[i] = (values[i] - mean) / std_dev;
            }
        }
        
        return result_array;
    }
    
    static py::array_t<double> rank(py::array_t<double> values_arr) {
        size_t size = values_arr.request().shape[0];
        double* values = static_cast<double*>(values_arr.request().ptr);
        
        auto result_array = py::array_t<double>(size);
        double* result = static_cast<double*>(result_array.request().ptr);
        
        if (size == 0) return result_array;
        
        std::vector<size_t> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), 
            [values](size_t i, size_t j) { return values[i] < values[j]; });
        
        std::vector<size_t> ranks(size);
        for (size_t i = 0; i < indices.size(); i++) {
            ranks[indices[i]] = i;
        }
        
        for (size_t i=0; i<size; ++i) {
            result[i] = static_cast<double>(ranks[i]) / (size - 1);
        }
        
        return result_array;
    }
};

// Fractional Differencing
class FractionalDifferencing {
private:
    double d;
    double threshold;
    std::vector<double> weights;

public:
    FractionalDifferencing(double d = 0.4, double threshold = 1e-5) 
        : d(d), threshold(threshold) {}

    void compute_weights(size_t max_len) {
        weights.clear();
        weights.push_back(1.0);
        for (size_t k = 1; k < max_len; ++k) {
            double w = -weights.back() * (d - k + 1) / k;
            if (std::abs(w) < threshold) break;
            weights.push_back(w);
        }
    }

    py::array_t<double> frac_diff(py::array_t<double> series_arr) {
        size_t size = series_arr.request().shape[0];
        double* series = static_cast<double*>(series_arr.request().ptr);
        
        compute_weights(size);
        
        auto result_array = py::array_t<double>(size);
        double* result = static_cast<double*>(result_array.request().ptr);

        for (size_t i = 0; i < size; ++i) {
            if (i < weights.size() - 1) {
                result[i] = 0.0;
                continue;
            }
            double val = 0.0;
            size_t max_j = std::min(i + 1, weights.size());
            for (size_t j = 0; j < max_j; ++j) {
                val += series[i - j] * weights[j];
            }
            result[i] = val;
        }
        return result_array;
    }
    
    py::array_t<double> get_weights() const {
        auto result_array = py::array_t<double>(weights.size());
        double* result = static_cast<double*>(result_array.request().ptr);
        for(size_t i=0; i<weights.size(); ++i) result[i] = weights[i];
        return result_array;
    }
};

// Python bindings

extern void init_cpp_dsa(py::module_ &);

PYBIND11_MODULE(cpp_features, m) {
    m.doc() = "C++ Feature Pipeline for High-Performance Feature Computation";
    
    // Initialize Data Structures and Algorithms
    init_cpp_dsa(m);

    py::class_<RollingStatistics>(m, "RollingStatistics")
        .def(py::init<size_t>())
        .def("add", &RollingStatistics::add)
        .def("mean", &RollingStatistics::mean)
        .def("std", &RollingStatistics::std)
        .def("skew", &RollingStatistics::skew)
        .def("kurtosis", &RollingStatistics::kurtosis)
        .def("size", &RollingStatistics::size);
    
    py::class_<TechnicalIndicators>(m, "TechnicalIndicators")
        .def_static("rsi", &TechnicalIndicators::rsi, 
            py::arg("prices"), py::arg("period") = 14)
        .def_static("macd", &TechnicalIndicators::macd,
            py::arg("prices"), py::arg("fast") = 12, py::arg("slow") = 26, py::arg("signal") = 9)
        .def_static("bollinger_bands", &TechnicalIndicators::bollinger_bands,
            py::arg("prices"), py::arg("period") = 20, py::arg("std_dev") = 2.0);
    
    py::class_<MicrostructureFeatures>(m, "MicrostructureFeatures")
        .def_static("ofi", &MicrostructureFeatures::ofi,
            py::arg("bid_prices"), py::arg("ask_prices"), 
            py::arg("bid_volumes"), py::arg("ask_volumes"))
        .def_static("vpin", &MicrostructureFeatures::vpin,
            py::arg("buy_volumes"), py::arg("sell_volumes"), py::arg("bucket_size") = 1000000.0);
    
    py::class_<CrossSectionalFeatures>(m, "CrossSectionalFeatures")
        .def_static("zscore", &CrossSectionalFeatures::zscore)
        .def_static("rank", &CrossSectionalFeatures::rank);
        
    py::class_<FractionalDifferencing>(m, "FractionalDifferencing")
        .def(py::init<double, double>(), py::arg("d") = 0.4, py::arg("threshold") = 1e-5)
        .def("compute_weights", &FractionalDifferencing::compute_weights, py::arg("max_len"))
        .def("frac_diff", &FractionalDifferencing::frac_diff, py::arg("series"))
        .def("get_weights", &FractionalDifferencing::get_weights);
}
