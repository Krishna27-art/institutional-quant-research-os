import sys
import os

print("--- Testing Phase 1 Components ---")

# 1. Test Data Structures
try:
    import cpp_features
    rb = cpp_features.RingBuffer(5)
    for i in range(1, 6):
        rb.append(i * 1.0)
    print(f"✅ cpp_features loaded. RingBuffer mean: {rb.mean()} (Expected: 3.0)")
except Exception as e:
    print(f"❌ cpp_features failed: {e}")

# 2. Test ZMQ Consumer
try:
    import cpp_zmq_consumer
    consumer = cpp_zmq_consumer.ZMQConsumer("tcp://localhost:5555")
    print(f"✅ cpp_zmq_consumer loaded. Running state: {consumer.is_running()} (Expected: False)")
except Exception as e:
    print(f"❌ cpp_zmq_consumer failed: {e}")

# 3. Test Order Book
try:
    import cpp_order_book
    book = cpp_order_book.CPPOrderBook()
    book.add_order("bid", 100.5, 10)
    book.add_order("ask", 101.0, 5)
    print(f"✅ cpp_order_book loaded. Mid price: {book.get_mid_price()} (Expected: 100.75)")
except Exception as e:
    print(f"❌ cpp_order_book failed: {e}")

# 4. Test Rust Risk Engine
try:
    from src.risk.institutional_risk_engine import InstitutionalRiskEngine
    engine = InstitutionalRiskEngine()
    passed = engine.check_pre_trade_limits_rust(200.0, 50, "NIFTY")
    print(f"✅ rust_risk_engine loaded via FFI. Order passed risk check: {passed} (Expected: True)")
except Exception as e:
    print(f"❌ rust_risk_engine FFI failed: {e}")

print("--- Verification Complete ---")
