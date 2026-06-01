# Go Data Ingestion Module

Lightweight Go module for data ingestion as part of Architecture V2.

## Purpose
- WebSocket → Go routine → Redis Streams
- Low latency, high throughput
- Simpler than Kafka for our scale

## Architecture V2 Data Flow
- Data feed: WebSocket → Go routine → Redis Streams
- Feature calculation: Python + Polars (vectorized) → Redis Hash
- Signal generation: LightGBM (C API via Python) → Redis Pub/Sub
- Risk & order generation: Python (single thread) → Redis Queue
- Execution: FastAPI (order submission to broker) → HTTP/2

## Installation
```bash
cd ingestion
go mod download
go run main.go
```

## Configuration
Edit the Config struct in main.go to configure:
- Redis connection details
- Broker API endpoint
- Symbols to ingest

## Usage
```bash
# Run ingestion
go run main.go

# Build binary
go build -o ingestion main.go
./ingestion
```

## WebSocket Server
Optional WebSocket server on :8081 for testing:
```bash
# Connect to WebSocket
ws://localhost:8081/ws
```

## Dependencies
- go-redis/redis/v8: Redis client
- gorilla/websocket: WebSocket server
