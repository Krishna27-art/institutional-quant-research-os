#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

namespace py = pybind11;

// Lock-free Single-Producer Single-Consumer (SPSC) Ring Buffer
template<typename T, size_t Capacity>
class SPSCRingBuffer {
private:
    T buffer[Capacity];
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};

public:
    bool push(const T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; // Full
        }
        buffer[current_tail] = item;
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t current_head = head.load(std::memory_order_relaxed);
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // Empty
        }
        item = buffer[current_head];
        head.store((current_head + 1) % Capacity, std::memory_order_release);
        return true;
    }
};

class ZMQConsumer {
private:
    std::string endpoint;
    std::atomic<bool> running;
    std::thread worker_thread;
    SPSCRingBuffer<std::string, 1024> queue;

    void worker_loop() {
        std::cout << "[C++] ZMQ Consumer connected to " << endpoint << std::endl;
        
        while (running.load(std::memory_order_relaxed)) {
            // Simulate receiving a tick without blocking
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            // In reality: 
            // zmq::message_t msg;
            // if (socket.recv(msg, zmq::recv_flags::dontwait)) {
            //     queue.push(msg.to_string());
            // }
            
            // Push payload to lock-free queue instead of hitting Python GIL
            queue.push("TICK_DATA_PAYLOAD");
        }
        std::cout << "[C++] ZMQ Consumer disconnected." << std::endl;
    }

public:
    ZMQConsumer(const std::string& endpoint) : endpoint(endpoint), running(false) {}

    ~ZMQConsumer() {
        stop();
    }

    void start() {
        if (running.load()) return;
        running.store(true);
        worker_thread = std::thread(&ZMQConsumer::worker_loop, this);
    }

    void stop() {
        if (!running.load()) return;
        running.store(false);
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
    
    // Polled by Python thread without GIL contention during push
    std::vector<std::string> poll_batch(size_t max_batch = 100) {
        std::vector<std::string> batch;
        batch.reserve(max_batch);
        std::string item;
        while (batch.size() < max_batch && queue.pop(item)) {
            batch.push_back(item);
        }
        return batch;
    }

    bool is_running() const {
        return running.load();
    }
};

PYBIND11_MODULE(cpp_zmq_consumer, m) {
    m.doc() = "C++ ZeroMQ Consumer for High-Performance Feed Handling";
    
    py::class_<ZMQConsumer>(m, "ZMQConsumer")
        .def(py::init<const std::string&>())
        .def("start", &ZMQConsumer::start)
        .def("stop", &ZMQConsumer::stop)
        .def("poll_batch", &ZMQConsumer::poll_batch, py::arg("max_batch") = 100)
        .def("is_running", &ZMQConsumer::is_running);
}
