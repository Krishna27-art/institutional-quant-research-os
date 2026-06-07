// Quant Trading Dashboard - Connected API Implementation

let socket = null;
let reconnectTimer = null;
const apiBase = window.location.origin;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    fetchStaticApiData();
    
    // Poll data every 5 seconds for REST-only endpoints
    setInterval(() => {
        fetchStaticApiData();
    }, 5000);
});

// Initialize WebSocket for real-time updates
function initWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/ws`;
    
    updateConnectionStatus('connecting');
    
    if (socket) {
        socket.close();
    }
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log('WebSocket connection established.');
        updateConnectionStatus('connected');
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };
    
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket state received:', data);
            handleStateUpdate(data);
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    };
    
    socket.onclose = (event) => {
        console.warn('WebSocket connection closed. Code:', event.code);
        updateConnectionStatus('disconnected');
        
        // Attempt reconnection in 3 seconds
        if (!reconnectTimer) {
            reconnectTimer = setTimeout(() => {
                reconnectTimer = null;
                initWebSocket();
            }, 3000);
        }
    };
    
    socket.onerror = (err) => {
        console.error('WebSocket encountered an error:', err);
        socket.close();
    };
}

// Update connection status badge in UI
function updateConnectionStatus(status) {
    const badge = document.getElementById('conn-status');
    if (!badge) return;
    
    badge.className = 'connection-badge';
    const label = badge.querySelector('.label-text');
    
    if (status === 'connected') {
        badge.classList.add('connected');
        label.textContent = 'CONNECTED';
    } else if (status === 'connecting') {
        label.textContent = 'CONNECTING...';
    } else {
        badge.classList.add('disconnected');
        label.textContent = 'DISCONNECTED';
    }
}

// Process state updates sent via WebSocket (from API Server's StatePublisher)
function handleStateUpdate(state) {
    // 1. Portfolio Metrics
    const navVal = state.nav !== undefined ? state.nav : 250000000;
    const dailyPnLVal = state.daily_pnl !== undefined ? state.daily_pnl : 0;
    
    document.getElementById('portfolio-nav').textContent = `₹${navVal.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
    
    const pnlEl = document.getElementById('portfolio-daily-pnl');
    pnlEl.textContent = `${dailyPnLVal >= 0 ? '+' : ''}₹${dailyPnLVal.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
    pnlEl.className = `metric-value ${dailyPnLVal >= 0 ? 'positive' : 'negative'}`;
    
    // 2. Regime Status
    const regime = state.regime || 'sideways';
    const conf = state.regime_confidence !== undefined ? state.regime_confidence : 0.5;
    
    const regimeBadge = document.getElementById('current-regime-badge');
    if (regimeBadge) {
        regimeBadge.className = `regime-badge ${regime.toLowerCase()}`;
        regimeBadge.textContent = formatRegimeName(regime);
    }
    
    const regimeConf = document.getElementById('regime-confidence-value');
    if (regimeConf) {
        regimeConf.textContent = `${(conf * 100).toFixed(0)}%`;
    }
    
    // Update individual progress bars if regime probabilities are provided
    if (state.regime_probabilities) {
        updateRegimeBars(state.regime_probabilities);
    } else {
        // Mock distribution based on primary regime
        const mockProbs = { bull: 10, bear: 10, sideways: 10, high_vol: 10 };
        mockProbs[regime.toLowerCase()] = 70;
        updateRegimeBars(mockProbs);
    }
    
    // 3. Risk Metrics
    if (state.risk) {
        document.getElementById('risk-var').textContent = `₹${(state.risk.var || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
        document.getElementById('risk-cvar').textContent = `₹${(state.risk.cvar || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
        document.getElementById('risk-tail').textContent = `₹${(state.risk.tail_risk || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    }
    
    // 4. Positions Table
    const positions = state.positions || [];
    updatePositionsTable(positions);
    
    // 5. Signals Table (WebSocket live signals)
    const signals = state.signals || [];
    updateSignalsTable(signals);
}

// Fetch REST-only API endpoints
function fetchStaticApiData() {
    // A. Fetch Screener data
    fetch(`${apiBase}/api/screener`)
        .then(res => res.json())
        .then(data => {
            const stocks = data.stocks || data.data || [];
            updateScreenerTable(stocks);
        })
        .catch(err => console.error('Error fetching screener:', err));
        
    // B. Fetch System Health data
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            updateSystemHealth(health);
        })
        .catch(err => console.error('Error fetching system health:', err));
        
    // C. Fetch Indices
    fetch(`${apiBase}/api/indices`)
        .then(res => res.json())
        .then(indices => {
            updateIndices(indices);
        })
        .catch(err => console.error('Error fetching indices:', err));
        
    // D. Fetch General Metrics
    fetch(`${apiBase}/api/metrics`)
        .then(res => res.json())
        .then(metrics => {
            updateHistoricalMetrics(metrics);
        })
        .catch(err => console.error('Error fetching metrics:', err));
}

// Helper to format regime name
function formatRegimeName(regime) {
    if (regime === 'high_vol' || regime === 'high-vol') return 'High Volatility';
    return regime.toUpperCase();
}

// Update HMM regime progress bars
function updateRegimeBars(probs) {
    const categories = ['bull', 'bear', 'sideways', 'high_vol'];
    categories.forEach(cat => {
        const val = probs[cat] !== undefined ? probs[cat] : probs[cat === 'high_vol' ? 'highVol' : cat] || 0;
        const progressFill = document.getElementById(`regime-fill-${cat}`);
        const labelVal = document.getElementById(`regime-val-${cat}`);
        if (progressFill) {
            progressFill.style.width = `${val}%`;
        }
        if (labelVal) {
            labelVal.textContent = `${parseFloat(val).toFixed(0)}%`;
        }
    });
}

// Update Active Positions Table
function updatePositionsTable(positions) {
    const tbody = document.getElementById('positions-tbody');
    if (!tbody) return;
    
    if (positions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No active positions</td></tr>`;
        return;
    }
    
    tbody.innerHTML = positions.map(pos => {
        const sideClass = pos.side.toUpperCase() === 'BUY' || pos.side.toUpperCase() === 'LONG' ? 'buy-badge' : 'sell-badge';
        const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';
        return `
            <tr>
                <td style="font-weight: 600;">${pos.symbol}</td>
                <td><span class="${sideClass}">${pos.side.toUpperCase()}</span></td>
                <td>${pos.quantity}</td>
                <td>₹${pos.entry_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</td>
                <td class="${pnlClass}" style="font-weight: 600;">₹${pos.pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</td>
            </tr>
        `;
    }).join('');
}

// Update Screener Table (Real-time stock scanner)
function updateScreenerTable(stocks) {
    const tbody = document.getElementById('screener-tbody');
    if (!tbody) return;
    
    if (stocks.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary);">Scanning active tickers...</td></tr>`;
        return;
    }
    
    tbody.innerHTML = stocks.map(stock => {
        const signalClass = stock.signal === 'BUY' ? 'buy-badge' : stock.signal === 'SHORT' ? 'sell-badge' : 'neutral-badge';
        const changeClass = stock.change >= 0 ? 'positive' : 'negative';
        return `
            <tr>
                <td style="font-weight: 600;">${stock.symbol}</td>
                <td>₹${stock.price.toLocaleString('en-IN')}</td>
                <td class="${changeClass}">${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)}%</td>
                <td><span class="${signalClass}">${stock.signal}</span></td>
                <td>${stock.rsi.toFixed(0)}</td>
                <td>${stock.conf.toFixed(0)}%</td>
                <td>${stock.rr.toFixed(1)}x</td>
            </tr>
        `;
    }).join('');
}

// Update Live Signals List
function updateSignalsTable(signals) {
    const tbody = document.getElementById('signals-tbody');
    if (!tbody) return;
    
    if (signals.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">Waiting for new signals...</td></tr>`;
        return;
    }
    
    tbody.innerHTML = signals.map(sig => {
        const time = sig.timestamp ? new Date(sig.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        const sideClass = sig.direction > 0 ? 'buy-badge' : sig.direction < 0 ? 'sell-badge' : 'neutral-badge';
        const sideText = sig.direction > 0 ? 'LONG' : sig.direction < 0 ? 'SHORT' : 'NEUTRAL';
        return `
            <tr>
                <td style="color: var(--text-secondary);">${time}</td>
                <td style="font-weight: 600;">${sig.symbol}</td>
                <td>${sig.strategy || 'ORB'}</td>
                <td><span class="${sideClass}">${sideText}</span></td>
                <td>${((sig.confidence || 0.5) * 100).toFixed(0)}%</td>
            </tr>
        `;
    }).join('');
}

// Update Market Indices Widgets
function updateIndices(indices) {
    indices.forEach(idx => {
        const idBase = idx.name.replace(/\s+/g, '-').toLowerCase();
        const valEl = document.getElementById(`idx-${idBase}-val`);
        const chgEl = document.getElementById(`idx-${idBase}-chg`);
        
        if (valEl) {
            valEl.textContent = idx.value.toLocaleString('en-IN', { minimumFractionDigits: 2 });
        }
        if (chgEl) {
            chgEl.textContent = `${idx.change >= 0 ? '+' : ''}${idx.change.toFixed(2)}%`;
            chgEl.className = `metric-value ${idx.change >= 0 ? 'positive' : 'negative'}`;
        }
    });
}

// Update System Health card
function updateSystemHealth(health) {
    const dbBadge = document.getElementById('health-db');
    if (dbBadge) {
        const status = health.database_status === 'CONNECTED' ? 'ok' : 'error';
        dbBadge.className = `health-status-badge ${status}`;
        dbBadge.textContent = health.database_status;
    }
    
    const brokerBadge = document.getElementById('health-broker');
    if (brokerBadge) {
        const status = health.broker_status === 'CONNECTED' ? 'ok' : 'warning';
        brokerBadge.className = `health-status-badge ${status}`;
        brokerBadge.textContent = health.broker_status.replace('_', ' ');
    }
    
    const countEl = document.getElementById('health-predictions');
    if (countEl) {
        countEl.textContent = health.predictions ? health.predictions.total_predictions : health.prediction_count || 0;
    }
    
    const loadedEl = document.getElementById('health-loaded');
    if (loadedEl) {
        loadedEl.textContent = health.universe ? health.universe.count : health.stocks_loaded || 0;
    }
    
    const stampEl = document.getElementById('health-timestamp');
    if (stampEl) {
        const t = health.latest_data_timestamp ? new Date(health.latest_data_timestamp) : new Date();
        stampEl.textContent = t.toLocaleTimeString();
    }
    
    const marketBadge = document.getElementById('health-market');
    if (marketBadge) {
        const state = health.market_status ? health.market_status.status || health.market_status : 'CLOSED';
        const statusClass = state === 'OPEN' ? 'ok' : 'warning';
        marketBadge.className = `health-status-badge ${statusClass}`;
        marketBadge.textContent = state;
    }
}

// Update performance stats
function updateHistoricalMetrics(metrics) {
    const winRateEl = document.getElementById('metric-winrate');
    if (winRateEl) {
        winRateEl.textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
    }
    
    const sharpeEl = document.getElementById('metric-sharpe');
    if (sharpeEl) {
        sharpeEl.textContent = metrics.sharpe_ratio.toFixed(2);
    }
    
    const drawdownEl = document.getElementById('metric-drawdown');
    if (drawdownEl) {
        drawdownEl.textContent = `${metrics.max_drawdown.toFixed(2)}%`;
    }
}
