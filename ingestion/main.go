/*
Go Data Ingestion Module
Based on Architecture V2 agent debate consensus

Key findings from research:
- Go routine for lightweight data ingestion
- WebSocket → Go routine → Redis Streams
- Low latency, high throughput
- Simpler than Kafka for our scale

Architecture V2 - Quantitative Trading System for Indian Markets
Phase 1: Go module for data ingestion (optional, for Phase 2 scaling)
*/

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/gorilla/websocket"
)

// Tick represents a market data tick
type Tick struct {
	Symbol    string  `json:"symbol"`
	Timestamp string  `json:"timestamp"`
	Price     float64 `json:"price"`
	Volume    int64   `json:"volume"`
	Bid       float64 `json:"bid"`
	Ask       float64 `json:"ask"`
}

// Config holds configuration
type Config struct {
	RedisHost     string
	RedisPort     int
	RedisDB       int
	StreamPrefix  string
	BrokerAPI     string
	Symbols       []string
}

// Ingestor handles data ingestion
type Ingestor struct {
	config      Config
	redisClient *redis.Client
	wg          sync.WaitGroup
	ctx         context.Context
	cancel      context.CancelFunc
	tickChan    chan Tick
}

// NewIngestor creates a new data ingestor
func NewIngestor(config Config) *Ingestor {
	ctx, cancel := context.WithCancel(context.Background())
	
	redisClient := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", config.RedisHost, config.RedisPort),
		DB:       config.RedisDB,
		Password: "",
	})
	
	return &Ingestor{
		config:      config,
		redisClient: redisClient,
		ctx:         ctx,
		cancel:      cancel,
		tickChan:    make(chan Tick, 10000),
	}
}

// Connect establishes Redis connection
func (i *Ingestor) Connect() error {
	_, err := i.redisClient.Ping(i.ctx).Result()
	if err != nil {
		return fmt.Errorf("failed to connect to Redis: %w", err)
	}
	log.Printf("Connected to Redis at %s:%d", i.config.RedisHost, i.config.RedisPort)
	return nil
}

// Disconnect closes Redis connection
func (i *Ingestor) Disconnect() {
	i.cancel()
	i.redisClient.Close()
	log.Println("Disconnected from Redis")
}

// createStream creates a Redis Stream for a symbol
func (i *Ingestor) createStream(symbol string) error {
	streamName := fmt.Sprintf("%s:%s", i.config.StreamPrefix, symbol)
	
	// Try to create stream (ignore if exists)
	_, err := i.redisClient.XAdd(i.ctx, &redis.XAddArgs{
		Stream: streamName,
		MaxLen: 0,
		Approx: false,
	}, map[string]interface{}{
		"test": "init",
	}).Result()
	
	if err != nil && err.Error() != "ERR Stream already exists" {
		return fmt.Errorf("failed to create stream: %w", err)
	}
	
	// Remove test entry
	_, err = i.redisClient.XDel(i.ctx, streamName, 0).Result()
	if err != nil {
		log.Printf("Warning: failed to remove test entry: %v", err)
	}
	
	return nil
}

// createConsumerGroup creates a consumer group for a stream
func (i *Ingestor) createConsumerGroup(symbol string) error {
	streamName := fmt.Sprintf("%s:%s", i.config.StreamPrefix, symbol)
	consumerGroup := "feature_processor"
	
	// Try to create consumer group (ignore if exists)
	_, err := i.redisClient.XGroupCreateMkStream(i.ctx, streamName, consumerGroup, "0").Result()
	if err != nil && err.Error() != "ERR BUSYGROUP Consumer Group name already exists" {
		return fmt.Errorf("failed to create consumer group: %w", err)
	}
	
	return nil
}

// publishTick publishes a tick to Redis Stream
func (i *Ingestor) publishTick(symbol string, tick Tick) error {
	streamName := fmt.Sprintf("%s:%s", i.config.StreamPrefix, symbol)
	
	// Convert tick to map
	tickMap := map[string]interface{}{
		"symbol":    tick.Symbol,
		"timestamp": tick.Timestamp,
		"price":     tick.Price,
		"volume":    tick.Volume,
		"bid":       tick.Bid,
		"ask":       tick.Ask,
	}
	
	_, err := i.redisClient.XAdd(i.ctx, &redis.XAddArgs{
		Stream: streamName,
		MaxLen: 0,
		Approx: false,
	}, tickMap).Result()
	
	return err
}

// processTicks processes ticks from channel and publishes to Redis
func (i *Ingestor) processTicks() {
	defer i.wg.Done()
	
	tickCount := make(map[string]int)
	lastLog := time.Now()
	
	for {
		select {
		case <-i.ctx.Done():
			return
		case tick := <-i.tickChan:
			err := i.publishTick(tick.Symbol, tick)
			if err != nil {
				log.Printf("Error publishing tick for %s: %v", tick.Symbol, err)
			} else {
				tickCount[tick.Symbol]++
			}
			
			// Log stats every 10 seconds
			if time.Since(lastLog) > 10*time.Second {
				log.Printf("Processed ticks: %v", tickCount)
				tickCount = make(map[string]int)
				lastLog = time.Now()
			}
		}
	}
}

// simulateWebSocket simulates WebSocket connection to broker API
func (i *Ingestor) simulateWebSocket(symbol string) {
	defer i.wg.Done()
	
	ticker := time.NewTicker(100 * time.Millisecond) // 10 ticks per second
	defer ticker.Stop()
	
	for {
		select {
		case <-i.ctx.Done():
			return
		case <-ticker.C:
			// Simulate tick data
			tick := Tick{
				Symbol:    symbol,
				Timestamp: time.Now().Format(time.RFC3339),
				Price:     20000.0 + float64(time.Now().UnixNano()%1000)/100.0,
				Volume:    100000 + int64(time.Now().UnixNano()%50000),
				Bid:       20000.0 + float64(time.Now().UnixNano()%1000)/100.0 - 1.0,
				Ask:       20000.0 + float64(time.Now().UnixNano()%1000)/100.0 + 1.0,
			}
			
			select {
			case i.tickChan <- tick:
			case <-i.ctx.Done():
				return
			}
		}
	}
}

// Start starts the ingestion process
func (i *Ingestor) Start() error {
	// Connect to Redis
	if err := i.Connect(); err != nil {
		return err
	}
	
	// Create streams and consumer groups for each symbol
	for _, symbol := range i.config.Symbols {
		if err := i.createStream(symbol); err != nil {
			log.Printf("Warning: failed to create stream for %s: %v", symbol, err)
		}
		if err := i.createConsumerGroup(symbol); err != nil {
			log.Printf("Warning: failed to create consumer group for %s: %v", symbol, err)
		}
	}
	
	// Start tick processor
	i.wg.Add(1)
	go i.processTicks()
	
	// Start WebSocket simulators for each symbol
	for _, symbol := range i.config.Symbols {
		i.wg.Add(1)
		go i.simulateWebSocket(symbol)
	}
	
	log.Printf("Started ingestion for %d symbols", len(i.config.Symbols))
	return nil
}

// Stop stops the ingestion process
func (i *Ingestor) Stop() {
	log.Println("Stopping ingestion...")
	i.cancel()
	i.wg.Wait()
	log.Println("Ingestion stopped")
}

// serveWebSocket handles WebSocket connections
func (i *Ingestor) serveWebSocket(w http.ResponseWriter, r *http.Request) {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}
	
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()
	
	log.Println("WebSocket connection established")
	
	// Send ticks to client
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Send simulated tick
			tick := Tick{
				Symbol:    "NIFTY",
				Timestamp: time.Now().Format(time.RFC3339),
				Price:     20000.0,
				Volume:    100000,
				Bid:       19999.0,
				Ask:       20001.0,
			}
			
			data, _ := json.Marshal(tick)
			err := conn.WriteMessage(websocket.TextMessage, data)
			if err != nil {
				log.Printf("WebSocket write error: %v", err)
				return
			}
		}
	}
}

func main() {
	config := Config{
		RedisHost:    "localhost",
		RedisPort:    6379,
		RedisDB:      0,
		StreamPrefix: "market_data",
		BrokerAPI:    "http://localhost:8080",
		Symbols:      []string{"NIFTY", "BANKNIFTY", "RELIANCE", "HDFCBANK", "INFY"},
	}
	
	ingestor := NewIngestor(config)
	
	// Start ingestion
	if err := ingestor.Start(); err != nil {
		log.Fatalf("Failed to start ingestion: %v", err)
	}
	defer ingestor.Stop()
	
	// Start WebSocket server (optional)
	http.HandleFunc("/ws", ingestor.serveWebSocket)
	go func() {
		log.Println("WebSocket server started on :8081")
		log.Fatal(http.ListenAndServe(":8081", nil))
	}()
	
	// Keep running
	log.Println("Ingestion running. Press Ctrl+C to stop.")
	
	// Wait for interrupt signal
	select {}
}
