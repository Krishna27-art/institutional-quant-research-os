/*
 * C++ Feature Pipeline for High-Performance Feature Computation
 * Compiled with pybind11 for Python integration
 * 
 * Based on institutional review recommendations:
 * - C++ feature pipeline (via pybind11) for performance
 * - 100-1000x speedup over Python for critical loops
 * - Used by top firms (Renaissance, Two Sigma, Citadel)
 * 
 * Features implemented:
 * - Rolling statistics (mean, std, skew, kurtosis)
 * - Technical indicators (RSI, MACD, Bollinger Bands)
 * - Microstructure features (OFI, VPIN)
 * - Cross-sectional features
 * - Memory-efficient computation
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
        return std::sqrt((sum_sq / window.size()) - (mean_val * mean_val));
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
    // RSI (Relative Strength Index)
    static std::vector<double> rsi(const std::vector<double>& prices, size_t period = 14) {
        std::vector<double> result;
        if (prices.size() < period + 1) return result;
        
        std::vector<double> gains, losses;
        for (size_t i = 1; i < prices.size(); i++) {
            double change = prices[i] - prices[i-1];
            if (change > 0) {
                gains.push_back(change);
                losses.push_back(0.0);
            } else {
                gains.push_back(0.0);
                losses.push_back(-change);
            }
        }
        
        double avg_gain = 0.0, avg_loss = 0.0;
        for (size_t i = 0; i < period; i++) {
            avg_gain += gains[i];
            avg_loss += losses[i];
        }
        avg_gain /= period;
        avg_loss /= period;
        
        result.push_back(50.0); // First value
        
        for (size_t i = period; i < gains.size(); i++) {
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period;
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period;
            
            if (avg_loss == 0.0) {
                result.push_back(100.0);
            } else {
                double rs = avg_gain / avg_loss;
                result.push_back(100.0 - (100.0 / (1.0 + rs)));
            }
        }
        
        return result;
    }
    
    // MACD (Moving Average Convergence Divergence)
    static std::vector<double> macd(const std::vector<double>& prices, 
                                     size_t fast = 12, size_t slow = 26, size_t signal = 9) {
        std::vector<double> result;
        if (prices.size() < slow + signal) return result;
        
        // Calculate EMAs
        auto ema = [](const std::vector<double>& data, size_t period) {
            std::vector<double> ema_values;
            double multiplier = 2.0 / (period + 1);
            
            ema_values.push_back(data[0]);
            for (size_t i = 1; i < data.size(); i++) {
                ema_values.push_back((data[i] - ema_values.back()) * multiplier + ema_values.back());
            }
            return ema_values;
        };
        
        std::vector<double> ema_fast = ema(prices, fast);
        std::vector<double> ema_slow = ema(prices, slow);
        
        // MACD line
        std::vector<double> macd_line;
        for (size_t i = 0; i < ema_slow.size(); i++) {
            macd_line.push_back(ema_fast[i + (fast - slow)] - ema_slow[i]);
        }
        
        // Signal line
        std::vector<double> signal_line = ema(macd_line, signal);
        
        // Histogram
        for (size_t i = 0; i < signal_line.size(); i++) {
            result.push_back(macd_line[i + (macd_line.size() - signal_line.size())] - signal_line[i]);
        }
        
        return result;
    }
    
    // Bollinger Bands
    static std::vector<std::tuple<double, double, double>> bollinger_bands(
        const std::vector<double>& prices, size_t period = 20, double std_dev = 2.0) {
        std::vector<std::tuple<double, double, double>> result;
        
        RollingStatistics stats(period);
        
        for (double price : prices) {
            stats.add(price);
            if (stats.size() >= period) {
                double middle = stats.mean();
                double upper = middle + std_dev * stats.std();
                double lower = middle - std_dev * stats.std();
                result.push_back(std::make_tuple(upper, middle, lower));
            }
        }
        
        return result;
    }
};

// Microstructure features
class MicrostructureFeatures {
public:
    // Order Flow Imbalance (OFI)
    static std::vector<double> ofi(const std::vector<double>& bid_prices,
                                   const std::vector<double>& ask_prices,
                                   const std::vector<int>& bid_volumes,
                                   const std::vector<int>& ask_volumes) {
        std::vector<double> result;
        
        for (size_t i = 0; i < bid_prices.size(); i++) {
            double bid_vol = static_cast<double>(bid_volumes[i]);
            double ask_vol = static_cast<double>(ask_volumes[i]);
            
            if (bid_vol + ask_vol == 0.0) {
                result.push_back(0.0);
            } else {
                result.push_back((bid_vol - ask_vol) / (bid_vol + ask_vol));
            }
        }
        
        return result;
    }
    
    // VPIN (Volume-Synchronized Probability of Informed Trading)
    static std::vector<double> vpin(const std::vector<double>& buy_volumes,
                                    const std::vector<double>& sell_volumes,
                                    double bucket_size = 1000000.0) {
        std::vector<double> result;
        
        double current_bucket = 0.0;
        double buy_bucket = 0.0;
        double sell_bucket = 0.0;
        
        for (size_t i = 0; i < buy_volumes.size(); i++) {
            current_bucket += buy_volumes[i] + sell_volumes[i];
            buy_bucket += buy_volumes[i];
            sell_bucket += sell_volumes[i];
            
            if (current_bucket >= bucket_size) {
                double vpin = std::abs(buy_bucket - sell_bucket) / current_bucket;
                result.push_back(vpin);
                
                current_bucket = 0.0;
                buy_bucket = 0.0;
                sell_bucket = 0.0;
            }
        }
        
        return result;
    }
};

// Cross-sectional features
class CrossSectionalFeatures {
public:
    // Z-score normalization
    static std::vector<double> zscore(const std::vector<double>& values) {
        std::vector<double> result;
        
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
        double std_dev = std::sqrt(sq_sum / values.size() - mean * mean);
        
        for (double val : values) {
            if (std_dev == 0.0) {
                result.push_back(0.0);
            } else {
                result.push_back((val - mean) / std_dev);
            }
        }
        
        return result;
    }
    
    // Rank normalization
    static std::vector<double> rank(const std::vector<double>& values) {
        std::vector<double> result;
        
        std::vector<size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), 
            [&values](size_t i, size_t j) { return values[i] < values[j]; });
        
        std::vector<size_t> ranks(values.size());
        for (size_t i = 0; i < indices.size(); i++) {
            ranks[indices[i]] = i;
        }
        
        for (size_t rank : ranks) {
            result.push_back(static_cast<double>(rank) / (values.size() - 1));
        }
        
        return result;
    }
};

// Python bindings
PYBIND11_MODULE(cpp_features, m) {
    m.doc() = "C++ Feature Pipeline for High-Performance Feature Computation";
    
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
}
