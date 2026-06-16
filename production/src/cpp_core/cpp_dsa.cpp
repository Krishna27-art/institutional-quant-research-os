#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <deque>
#include <unordered_map>
#include <list>
#include <stdexcept>
#include <queue>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// 1. Ring Buffer
class RingBuffer {
private:
    std::vector<double> buffer;
    size_t capacity;
    size_t index;
    size_t count;
    bool is_full;

public:
    RingBuffer(size_t size) : capacity(size), index(0), count(0), is_full(false) {
        buffer.resize(size, 0.0);
    }

    void append(double value) {
        buffer[index] = value;
        index = (index + 1) % capacity;
        if (!is_full) {
            count++;
            if (count == capacity) {
                is_full = true;
            }
        }
    }

    double get(int i) const {
        if (i < 0) i = count + i;
        if (i < 0 || (size_t)i >= count) throw std::out_of_range("Index out of range");
        size_t actual_index = (index + capacity - count + i) % capacity;
        return buffer[actual_index];
    }

    py::array_t<double> to_array() const {
        std::vector<double> result(count);
        for (size_t i = 0; i < count; i++) {
            size_t actual_index = (index + capacity - count + i) % capacity;
            result[i] = buffer[actual_index];
        }
        return py::cast(result);
    }

    double mean() const {
        if (count == 0) return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < count; i++) sum += buffer[i];
        return sum / count;
    }

    double std() const {
        if (count < 2) return 0.0;
        double m = mean();
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; i++) {
            double actual_val = buffer[i]; // sum of squared diffs
            sum_sq += (actual_val - m) * (actual_val - m);
        }
        return std::sqrt(sum_sq / count);
    }

    double sum() const {
        double total = 0.0;
        for(size_t i=0; i<count; ++i) total += buffer[i];
        return total;
    }

    void reset() {
        std::fill(buffer.begin(), buffer.end(), 0.0);
        index = 0;
        count = 0;
        is_full = false;
    }

    size_t len() const { return count; }
};

// 2. LRU Cache
class LRUCache {
private:
    size_t capacity;
    std::list<std::pair<std::string, double>> items;
    std::unordered_map<std::string, decltype(items.begin())> cache;

public:
    LRUCache(size_t capacity) : capacity(capacity) {}

    double get(const std::string& key) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            throw std::out_of_range("Key not found");
        }
        items.splice(items.begin(), items, it->second);
        return it->second->second;
    }

    void put(const std::string& key, double value) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            items.splice(items.begin(), items, it->second);
            it->second->second = value;
            return;
        }
        if (items.size() == capacity) {
            auto last = items.back();
            cache.erase(last.first);
            items.pop_back();
        }
        items.emplace_front(key, value);
        cache[key] = items.begin();
    }
    
    bool contains(const std::string& key) const {
        return cache.find(key) != cache.end();
    }
};

// 3. Segment Tree (Sum and Min/Max)
class SegmentTree {
private:
    size_t n;
    std::vector<double> tree_sum;
    std::vector<double> tree_min;
    std::vector<double> tree_max;

    void build(const std::vector<double>& arr) {
        for (size_t i = 0; i < n; i++) {
            tree_sum[n + i] = arr[i];
            tree_min[n + i] = arr[i];
            tree_max[n + i] = arr[i];
        }
        for (size_t i = n - 1; i > 0; --i) {
            tree_sum[i] = tree_sum[i << 1] + tree_sum[i << 1 | 1];
            tree_min[i] = std::min(tree_min[i << 1], tree_min[i << 1 | 1]);
            tree_max[i] = std::max(tree_max[i << 1], tree_max[i << 1 | 1]);
        }
    }

public:
    SegmentTree(const std::vector<double>& arr) {
        n = arr.size();
        tree_sum.resize(2 * n, 0.0);
        tree_min.resize(2 * n, 0.0);
        tree_max.resize(2 * n, 0.0);
        build(arr);
    }

    void update(size_t p, double value) {
        if (p >= n) throw std::out_of_range("Index out of range");
        p += n;
        tree_sum[p] = value;
        tree_min[p] = value;
        tree_max[p] = value;
        for (size_t i = p; i > 1; i >>= 1) {
            tree_sum[i >> 1] = tree_sum[i] + tree_sum[i ^ 1];
            tree_min[i >> 1] = std::min(tree_min[i], tree_min[i ^ 1]);
            tree_max[i >> 1] = std::max(tree_max[i], tree_max[i ^ 1]);
        }
    }

    double query_sum(size_t l, size_t r) const {
        double res = 0;
        for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res += tree_sum[l++];
            if (r & 1) res += tree_sum[--r];
        }
        return res;
    }
};

// 4. Priority Queue (Min and Max queues)
class MinPriorityQueue {
private:
    std::priority_queue<double, std::vector<double>, std::greater<double>> pq;
public:
    void push(double val) { pq.push(val); }
    double pop() { double val = pq.top(); pq.pop(); return val; }
    double top() const { return pq.top(); }
    size_t size() const { return pq.size(); }
    bool empty() const { return pq.empty(); }
};

void init_cpp_dsa(py::module_ &m) {
    py::class_<RingBuffer>(m, "RingBuffer")
        .def(py::init<size_t>())
        .def("append", &RingBuffer::append)
        .def("get", &RingBuffer::get)
        .def("to_array", &RingBuffer::to_array)
        .def("mean", &RingBuffer::mean)
        .def("std", &RingBuffer::std)
        .def("sum", &RingBuffer::sum)
        .def("reset", &RingBuffer::reset)
        .def("__len__", &RingBuffer::len);

    py::class_<LRUCache>(m, "LRUCache")
        .def(py::init<size_t>())
        .def("get", &LRUCache::get)
        .def("put", &LRUCache::put)
        .def("contains", &LRUCache::contains);

    py::class_<SegmentTree>(m, "SegmentTree")
        .def(py::init<const std::vector<double>&>())
        .def("update", &SegmentTree::update)
        .def("query_sum", &SegmentTree::query_sum);

    py::class_<MinPriorityQueue>(m, "MinPriorityQueue")
        .def(py::init<>())
        .def("push", &MinPriorityQueue::push)
        .def("pop", &MinPriorityQueue::pop)
        .def("top", &MinPriorityQueue::top)
        .def("size", &MinPriorityQueue::size)
        .def("empty", &MinPriorityQueue::empty);
}
