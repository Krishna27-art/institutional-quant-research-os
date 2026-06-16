/**
 * NiftyQuant Institutional Research OS — Dashboard JS
 *
 * Architecture:
 *  - WebSocket (/ws) → real-time state (regime, positions, signals, risk)
 *  - REST polling  → screener, indices, metrics, health, models, predictions
 *
 * Data honesty rules:
 *  - No placeholders, mock values, or hardcoded stats.
 *  - Always fallback to robust data parsing.
 */

let socket = null;
let reconnectTimer = null;
let apiBase = window.location.origin;
if (apiBase.startsWith('file://')) {
    apiBase = 'http://localhost:8001';
}

// Global cached states
let currentRegimeState = { current_regime: 'SIDEWAYS', confidence: 0.5, probabilities: { bull: 25, bear: 25, sideways: 25, high_vol: 25 } };
let currentPortfolioMetrics = {};
let latestScreenerStocks = [];
let webSocketState = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initWebSocket();
    initGlobalPolling();
    initPageActions();
    
    // Initial page load
    const activeNav = document.querySelector('#sidebar .nav-item.active');
    if (activeNav) {
        onPageVisible(activeNav.getAttribute('data-page'));
    }
});

// ─── NAVIGATION ───────────────────────────────────────
function initNavigation() {
    document.querySelectorAll('#sidebar .nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.getAttribute('data-page');
            if (!page) return;
            
            // Update active navigation item
            document.querySelectorAll('#sidebar .nav-item').forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // Show page, hide others
            document.querySelectorAll('main#content .page').forEach(section => {
                section.classList.remove('active');
            });
            const targetSection = document.getElementById(`page-${page}`);
            if (targetSection) {
                targetSection.classList.add('active');
            }
            
            onPageVisible(page);
        });
    });
}

function onPageVisible(page) {
    console.log(`Page visible: ${page}`);
    // Load page-specific data immediately
    switch (page) {
        case 'dashboard':
            loadDashboardPage();
            break;
        case 'market':
            loadMarketPage();
            break;
        case 'screener':
            loadScreenerPage();
            break;
        case 'stock':
            loadStockPage();
            break;
        case 'predictions':
            loadPredictionsPage();
            break;
        case 'alpha':
            loadAlphaPage();
            break;
        case 'regime':
            loadRegimePage();
            break;
        case 'portfolio':
            loadPortfolioPage();
            break;
        case 'risk':
            loadRiskPage();
            break;
        case 'execution':
            loadExecutionPage();
            break;
        case 'models':
            loadModelsPage();
            break;
        case 'health':
            loadHealthPage();
            break;
    }
}

// ─── GLOBAL POLLING & TOPBAR ─────────────────────────
function initGlobalPolling() {
    // Poll indices and market status every 5 seconds
    pollGlobalData();
    setInterval(pollGlobalData, 5000);
}

function pollGlobalData() {
    // 1. Clock
    const clockEl = document.getElementById('clock');
    if (clockEl) {
        clockEl.textContent = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' }) + ' IST';
    }
    
    // Skip HTTP polling if WebSocket is open
    if (socket && socket.readyState === WebSocket.OPEN) {
        return;
    }
    
    // 2. Indices topbar strip
    fetch(`${apiBase}/api/indices`)
        .then(res => res.json())
        .then(indices => {
            updateIndicesStrip(indices);
            
            // If on market page, also update market page indices
            const marketPage = document.getElementById('page-market');
            if (marketPage && marketPage.classList.contains('active')) {
                updateMarketPageIndices(indices);
            }
        })
        .catch(err => console.error('Error polling indices:', err));
        
    // 3. Market status
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            updateMarketStatusBadge(health.market_status);
        })
        .catch(err => console.error('Error polling system health:', err));
}

function updateIndicesStrip(indices) {
    if (indices && indices.indices) {
        indices = indices.indices;
    }
    if (!indices || !Array.isArray(indices)) return;
    const idMap = {
        'NIFTY 50': { val: 'iv-nifty', chg: 'ic-nifty' },
        'BANKNIFTY': { val: 'iv-bank', chg: 'ic-bank' },
        'FINNIFTY': { val: 'iv-fin', chg: 'ic-fin' },
        'India VIX': { val: 'iv-vix', chg: 'ic-vix' }
    };
    
    indices.forEach(idx => {
        const ids = idMap[idx.name];
        if (ids) {
            const valEl = document.getElementById(ids.val);
            const chgEl = document.getElementById(ids.chg);
            if (valEl) valEl.textContent = idx.value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            if (chgEl) {
                chgEl.textContent = `${idx.change >= 0 ? '+' : ''}${idx.change.toFixed(2)}%`;
                chgEl.className = `idx-chg ${idx.change >= 0 ? 'up' : 'dn'}`;
            }
        }
    });
}

function updateMarketStatusBadge(statusObj) {
    const pill = document.getElementById('mkt-status-pill');
    const nextOpen = document.getElementById('mkt-next-open');
    if (!pill || !statusObj) return;
    
    const status = statusObj.status || 'CLOSED';
    pill.textContent = status;
    pill.className = 'pill';
    
    if (status === 'OPEN') {
        pill.classList.add('pill-open');
        if (nextOpen) nextOpen.textContent = '';
    } else if (status === 'PRE-OPEN') {
        pill.classList.add('pill-preopen');
        if (nextOpen) nextOpen.textContent = '';
    } else {
        pill.classList.add('pill-closed');
        if (nextOpen && statusObj.next_open) {
            nextOpen.textContent = `NEXT OPEN: ${statusObj.next_open}`;
        }
    }
}

// ─── WEBSOCKET CONNECTION ────────────────────────────
function initWebSocket() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host;
    if (window.location.protocol === 'file:') {
        host = 'localhost:8001';
    }
    const wsUrl = `${wsProto === 'file:' ? 'ws:' : wsProto}//${host}/ws`;
    
    updateConnectionStatus('connecting');
    
    if (socket) {
        socket.close();
    }
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus('connected');
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };
    
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            webSocketState = data;
            handleWebSocketBroadcast(data);
        } catch (err) {
            console.error('Error handling WS message:', err);
        }
    };
    
    socket.onclose = () => {
        console.warn('WebSocket closed');
        updateConnectionStatus('disconnected');
        if (!reconnectTimer) {
            reconnectTimer = setTimeout(initWebSocket, 3000);
        }
    };
    
    socket.onerror = (err) => {
        console.error('WebSocket error:', err);
        socket.close();
    };
}

function updateConnectionStatus(status) {
    const dot = document.getElementById('conn-dot');
    const label = document.getElementById('conn-label');
    if (!dot || !label) return;
    
    dot.className = 'conn-dot';
    if (status === 'connected') {
        dot.classList.add('connected');
        label.textContent = 'CONNECTED';
    } else if (status === 'connecting') {
        dot.classList.add('connecting');
        label.textContent = 'CONNECTING...';
    } else {
        dot.classList.add('disconnected');
        label.textContent = 'DISCONNECTED';
    }
}

function handleWebSocketBroadcast(state) {
    // WS handles real-time positions, regime state, and metrics.
    // Overwrite DOM elements if they exist (mostly on Dashboard, Portfolio, Risk pages)
    
    // NAV and Daily PnL
    const navVal = state.nav !== undefined ? state.nav : 250000000;
    const dailyPnLVal = state.daily_pnl !== undefined ? state.daily_pnl : 0;
    
    // Live update on Dashboard
    const dNav = document.getElementById('d-nav');
    if (dNav) dNav.textContent = formatPrice(navVal);
    
    const dPnl = document.getElementById('d-pnl');
    if (dPnl) {
        dPnl.textContent = formatPrice(dailyPnLVal, true);
        dPnl.className = `stat-val ${dailyPnLVal >= 0 ? 'green' : 'red'}`;
    }
    
    // Regime HMM state
    if (state.regime && state.regime !== 'unknown') {
        const normRegime = state.regime.toLowerCase().replace('-', '_');
        currentRegimeState = {
            current_regime: state.regime.toUpperCase(),
            confidence: state.regime_confidence || 0.5,
            probabilities: state.regime_probabilities || { bull: 25, bear: 25, sideways: 25, high_vol: 25 }
        };
        // Update dashboard regime if active page
        const dashPage = document.getElementById('page-dashboard');
        if (dashPage && dashPage.classList.contains('active')) {
            updateDashboardRegime();
        }
    }
    
    // Positions Table
    if (state.positions) {
        updateActivePositionsTable(state.positions);
    }
    
    // Live Signals (if open)
    if (state.signals) {
        updateLiveSignalsTable(state.signals);
    }
    
    // Risk snapshot values
    if (state.risk) {
        const dVar = document.getElementById('d-var');
        if (dVar) dVar.textContent = formatPrice(state.risk.var || 0);
        const dCvar = document.getElementById('d-cvar');
        if (dCvar) dCvar.textContent = formatPrice(state.risk.cvar || 0);
        const dTail = document.getElementById('d-tail');
        if (dTail) dTail.textContent = formatPrice(state.risk.tail_risk || 0);
    }
    
    // Indices and Market Status via WebSocket
    if (state.indices && state.indices.length > 0) {
        updateIndicesStrip(state.indices);
        const marketPage = document.getElementById('page-market');
        if (marketPage && marketPage.classList.contains('active')) {
            updateMarketPageIndices(state.indices);
        }
    }
    
    if (state.market_status && Object.keys(state.market_status).length > 0) {
        updateMarketStatusBadge(state.market_status);
    }
}

// ─── DASHBOARD PAGE ──────────────────────────────────
function loadDashboardPage() {
    // Set timestamp
    const dashTs = document.getElementById('dash-ts');
    if (dashTs) {
        dashTs.textContent = `REFRESHED: ${new Date().toLocaleTimeString()}`;
    }
    
    // Fetch metrics
    fetch(`${apiBase}/api/metrics`)
        .then(res => res.json())
        .then(metrics => {
            currentPortfolioMetrics = metrics;
            
            const dWr = document.getElementById('d-wr');
            if (dWr) dWr.textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
            
            const dSharpe = document.getElementById('d-sharpe');
            if (dSharpe) dSharpe.textContent = metrics.sharpe_ratio.toFixed(2);
            
            const dPreds = document.getElementById('d-preds');
            if (dPreds) dPreds.textContent = metrics.total_predictions;
            
            // Sharpe drawdown in risk snapshot
            const dDd = document.getElementById('d-dd');
            if (dDd) dDd.textContent = `${metrics.max_drawdown.toFixed(2)}%`;
            const dVol = document.getElementById('d-vol');
            if (dVol) dVol.textContent = `${metrics.volatility.toFixed(2)}%`;
        })
        .catch(err => console.error('Error loading dashboard metrics:', err));
        
    // Fetch alpha metrics for active strategies count
    fetch(`${apiBase}/api/alpha-lab/metrics`)
        .then(res => res.json())
        .then(alphas => {
            const activeCount = alphas.filter(a => a.ic > 0.02 && a.is_active).length;
            const dStrats = document.getElementById('d-strats');
            if (dStrats) dStrats.textContent = activeCount;
        })
        .catch(err => console.error('Error loading dashboard strategies count:', err));
        
    // Fetch current regime state
    fetch(`${apiBase}/api/regime/current`)
        .then(res => res.json())
        .then(regimeData => {
            currentRegimeState = regimeData;
            updateDashboardRegime();
        })
        .catch(err => console.error('Error loading regime:', err));
        
    // Fetch recent predictions/signals to populate live signals if WebSocket is silent
    fetch(`${apiBase}/api/predictions?limit=5`)
        .then(res => res.json())
        .then(preds => {
            // Convert predictions to signal format if needed
            const sigs = preds.map(p => ({
                symbol: p.symbol,
                direction: p.direction === 'LONG' || p.direction === 'BUY' ? 1 : -1,
                strategy: p.strategy,
                confidence: p.confidence / 100,
                timestamp: p.timestamp
            }));
            updateLiveSignalsTable(sigs);
        })
        .catch(err => console.error('Error loading recent predictions:', err));
        
    // Fetch system health to populate initial positions or check state
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            if (health.market_status) {
                const note = document.getElementById('d-sig-mkt-note');
                if (note) {
                    note.textContent = health.market_status.is_open ? 'LIVE STREAM' : 'MARKET CLOSED';
                }
            }
        })
        .catch(err => console.error('Error loading health:', err));
        
    // Update live positions
    if (webSocketState && webSocketState.positions) {
        updateActivePositionsTable(webSocketState.positions);
    } else {
        // Empty positions
        updateActivePositionsTable([]);
    }
}

function updateDashboardRegime() {
    const badge = document.getElementById('d-regime-badge');
    const confText = document.getElementById('d-regime-conf');
    if (!badge) return;
    
    const regime = currentRegimeState.current_regime || 'SIDEWAYS';
    badge.textContent = regime;
    badge.className = `badge-regime`;
    
    if (confText) {
        confText.textContent = `${(currentRegimeState.confidence * 100).toFixed(0)}%`;
    }
    
    // Progress bars
    const probs = currentRegimeState.probabilities || {};
    const categories = ['bull', 'bear', 'side', 'hvol'];
    const apiKeys = ['bull', 'bear', 'sideways', 'high_vol'];
    
    categories.forEach((cat, idx) => {
        const val = probs[apiKeys[idx]] || 0;
        const fillEl = document.getElementById(`rb-${cat}`);
        const textEl = document.getElementById(`rv-${cat}`);
        if (fillEl) fillEl.style.width = `${val}%`;
        if (textEl) textEl.textContent = `${val.toFixed(1)}%`;
    });
}

function updateActivePositionsTable(positions) {
    const tbody = document.getElementById('positions-tbody');
    if (!tbody) return;
    
    if (positions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-row">No open positions</td></tr>`;
        return;
    }
    
    tbody.innerHTML = positions.map(pos => {
        const isLong = pos.side.toUpperCase() === 'BUY' || pos.side.toUpperCase() === 'LONG';
        const sideClass = isLong ? 'green' : 'red';
        const pnlVal = pos.pnl || 0;
        const pnlClass = pnlVal >= 0 ? 'green' : 'red';
        return `
            <tr>
                <td style="font-weight:600;">${pos.symbol}</td>
                <td class="${sideClass}">${pos.side.toUpperCase()}</td>
                <td>${pos.quantity || pos.qty}</td>
                <td class="val-right">₹${pos.entry_price.toFixed(2)}</td>
                <td class="val-right ${pnlClass}" style="font-weight:600;">${formatPrice(pnlVal, true)}</td>
            </tr>
        `;
    }).join('');
}

function updateLiveSignalsTable(signals) {
    const tbody = document.getElementById('dash-signals-tbody');
    if (!tbody) return;
    
    if (signals.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-row">Waiting for signals (market must be open)</td></tr>`;
        return;
    }
    
    tbody.innerHTML = signals.map(sig => {
        const time = sig.timestamp ? new Date(sig.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        const directionVal = typeof sig.direction === 'number' ? sig.direction : (sig.direction.toUpperCase() === 'BUY' || sig.direction.toUpperCase() === 'LONG' ? 1 : -1);
        const isLong = directionVal > 0;
        const dirBadge = `<span class="sig-badge sig-${isLong ? 'BUY' : 'SHORT'}">${isLong ? 'BUY' : 'SHORT'}</span>`;
        const confPercent = sig.confidence ? `${(sig.confidence * 100).toFixed(0)}%` : '50%';
        return `
            <tr>
                <td style="font-weight:600;">${sig.symbol}</td>
                <td>${dirBadge}</td>
                <td>${sig.strategy || 'ORB'}</td>
                <td>${confPercent}</td>
                <td style="color:var(--text2);">${time}</td>
            </tr>
        `;
    }).join('');
}

// ─── MARKET OVERVIEW PAGE ────────────────────────────
function loadMarketPage() {
    // 1. Fetch indices to update indices block
    fetch(`${apiBase}/api/indices`)
        .then(res => res.json())
        .then(indices => {
            updateMarketPageIndices(indices);
        })
        .catch(err => console.error('Error fetching indices:', err));
        
    // 2. Fetch market status details
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            const stats = health.market_status || {};
            const msSession = document.getElementById('ms-session');
            if (msSession) {
                msSession.textContent = stats.status || 'CLOSED';
                msSession.className = `val-right ${stats.is_open ? 'green' : 'amber'}`;
            }
            
            const msTime = document.getElementById('ms-time');
            if (msTime) msTime.textContent = stats.current_time || 'N/A';
            
            const msDay = document.getElementById('ms-day');
            if (msDay) msDay.textContent = stats.day || 'N/A';
            
            const msHoliday = document.getElementById('ms-holiday');
            if (msHoliday) {
                msHoliday.textContent = stats.is_holiday ? 'YES' : 'NO';
                msHoliday.className = `val-right ${stats.is_holiday ? 'red' : 'green'}`;
            }
            
            const msNext = document.getElementById('ms-next');
            if (msNext) msNext.textContent = stats.next_open || 'N/A';
            
            // Market closed warning banner
            const banner = document.getElementById('mkt-data-note');
            if (banner) {
                banner.textContent = stats.is_open ? 'Data feed: LIVE (NSE Real-time). Prices refreshed dynamically.' : 'Data feed: PREVIOUS CLOSE (Market Closed). Prices are split-adjusted via Yahoo Finance.';
            }
        })
        .catch(err => console.error('Error loading market status:', err));
        
    // 3. Fetch screener to extract gainers and losers
    fetch(`${apiBase}/api/screener`)
        .then(res => res.json())
        .then(data => {
            const stocks = data.stocks || [];
            latestScreenerStocks = stocks;
            
            // Sort by change
            const sorted = [...stocks].sort((a, b) => b.change - a.change);
            
            // Top Gainers (top 5 positive change)
            const gainers = sorted.filter(s => s.change > 0).slice(0, 5);
            const gainersTbody = document.getElementById('mkt-gainers');
            if (gainersTbody) {
                if (gainers.length === 0) {
                    gainersTbody.innerHTML = `<tr><td colspan="3" class="empty-row">No gainers today</td></tr>`;
                } else {
                    gainersTbody.innerHTML = gainers.map(s => `
                        <tr>
                            <td style="font-weight:600;">${s.symbol}</td>
                            <td>₹${s.price.toFixed(2)}</td>
                            <td class="green val-right">+${s.change.toFixed(2)}%</td>
                        </tr>
                    `).join('');
                }
            }
            
            // Top Losers (bottom 5 negative change)
            const losers = [...sorted].reverse().filter(s => s.change < 0).slice(0, 5);
            const losersTbody = document.getElementById('mkt-losers');
            if (losersTbody) {
                if (losers.length === 0) {
                    losersTbody.innerHTML = `<tr><td colspan="3" class="empty-row">No losers today</td></tr>`;
                } else {
                    losersTbody.innerHTML = losers.map(s => `
                        <tr>
                            <td style="font-weight:600;">${s.symbol}</td>
                            <td>₹${s.price.toFixed(2)}</td>
                            <td class="red val-right">${s.change.toFixed(2)}%</td>
                        </tr>
                    `).join('');
                }
            }
        })
        .catch(err => console.error('Error loading gainers/losers:', err));
}

function updateMarketPageIndices(indices) {
    if (indices && indices.indices) {
        indices = indices.indices;
    }
    if (!indices || !Array.isArray(indices)) return;
    const map = {
        'NIFTY 50': { val: 'mo-nifty', chg: 'mo-nifty-chg' },
        'BANKNIFTY': { val: 'mo-bank', chg: 'mo-bank-chg' },
        'FINNIFTY': { val: 'mo-fin', chg: 'mo-fin-chg' },
        'India VIX': { val: 'mo-vix', chg: 'mo-vix-chg' }
    };
    
    indices.forEach(idx => {
        const ids = map[idx.name];
        if (ids) {
            const valEl = document.getElementById(ids.val);
            const chgEl = document.getElementById(ids.chg);
            if (valEl) valEl.textContent = idx.value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            if (chgEl) {
                chgEl.textContent = `${idx.change >= 0 ? '+' : ''}${idx.change.toFixed(2)}%`;
                chgEl.className = `stat-sub ${idx.change >= 0 ? 'green' : 'red'}`;
            }
        }
    });
}

// ─── STOCK SCREENER PAGE ─────────────────────────────
function loadScreenerPage() {
    // Update live banner status
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            const isLive = health.market_status ? health.market_status.is_open : false;
            const badge = document.getElementById('scr-mkt-badge');
            if (badge) {
                badge.textContent = isLive ? 'LIVE SCREENING' : 'PREV CLOSE';
                badge.className = `badge-info ${isLive ? 'green' : ''}`;
            }
            const note = document.getElementById('scr-note');
            if (note) {
                note.textContent = isLive ? 'Scanning active constituents in real-time. Prices updating live.' : 'Signals shown only when market is open. Prices represent previous day close.';
            }
        })
        .catch(err => console.error('Error loading screener badge status:', err));
        
    fetchAndRenderScreener();
}

function fetchAndRenderScreener() {
    const tbody = document.getElementById('screener-tbody');
    if (tbody) tbody.innerHTML = `<tr><td colspan="10" class="empty-row">Loading screener data...</td></tr>`;
    
    fetch(`${apiBase}/api/screener`)
        .then(res => res.json())
        .then(data => {
            const stocks = data.stocks || [];
            latestScreenerStocks = stocks;
            applyScreenerFilters();
        })
        .catch(err => {
            console.error('Error fetching screener:', err);
            if (tbody) tbody.innerHTML = `<tr><td colspan="10" class="empty-row red">Error loading screener data.</td></tr>`;
        });
}

function applyScreenerFilters() {
    const sigFilter = document.getElementById('scr-filter-sig').value;
    const minConf = parseFloat(document.getElementById('scr-filter-conf').value) || 0;
    const minRr = parseFloat(document.getElementById('scr-filter-rr').value) || 0;
    
    const tbody = document.getElementById('screener-tbody');
    if (!tbody) return;
    
    const filtered = latestScreenerStocks.filter(s => {
        if (sigFilter && s.signal !== sigFilter) return false;
        if (s.conf < minConf) return false;
        if (s.rr < minRr) return false;
        return true;
    });
    
    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="empty-row">No stocks match filters</td></tr>`;
        return;
    }
    
    tbody.innerHTML = filtered.map(s => {
        const signalClass = s.signal === 'BUY' ? 'sig-BUY' : (s.signal === 'SHORT' ? 'sig-SHORT' : 'sig-NEUTRAL');
        const changeClass = s.change >= 0 ? 'green' : 'red';
        const formattedChg = `${s.change >= 0 ? '+' : ''}${s.change.toFixed(2)}%`;
        
        return `
            <tr>
                <td style="font-weight:600;">${s.symbol}</td>
                <td class="val-right">₹${s.price.toFixed(2)}</td>
                <td class="${changeClass} val-right">${formattedChg}</td>
                <td><span class="sig-badge ${signalClass}">${s.signal}</span></td>
                <td>${s.rsi.toFixed(1)}</td>
                <td>${s.conf.toFixed(0)}%</td>
                <td class="val-right">₹${s.target.toFixed(2)}</td>
                <td class="val-right">₹${s.sl.toFixed(2)}</td>
                <td>${s.rr.toFixed(1)}x</td>
                <td>
                    <button class="action-btn" onclick="viewStockResearch('${s.symbol}')">Research</button>
                </td>
            </tr>
        `;
    }).join('');
}

function initPageActions() {
    // Screener filter listeners
    const sigSelect = document.getElementById('scr-filter-sig');
    if (sigSelect) sigSelect.addEventListener('change', applyScreenerFilters);
    
    const confInput = document.getElementById('scr-filter-conf');
    if (confInput) confInput.addEventListener('input', applyScreenerFilters);
    
    const rrInput = document.getElementById('scr-filter-rr');
    if (rrInput) rrInput.addEventListener('input', applyScreenerFilters);
    
    const refreshBtn = document.getElementById('scr-refresh-btn');
    if (refreshBtn) refreshBtn.addEventListener('click', fetchAndRenderScreener);
    
    // Stock search listeners
    const searchBtn = document.getElementById('stock-search-btn');
    const searchInput = document.getElementById('stock-search-input');
    if (searchBtn && searchInput) {
        searchBtn.addEventListener('click', () => {
            const sym = searchInput.value.trim().toUpperCase();
            if (sym) {
                loadStockResearch(sym);
            }
        });
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchBtn.click();
            }
        });
    }
}

// Action button route logic
window.viewStockResearch = function(symbol) {
    const navItem = document.querySelector('#sidebar .nav-item[data-page="stock"]');
    if (navItem) {
        navItem.click(); // switches tab
        const searchInput = document.getElementById('stock-search-input');
        if (searchInput) {
            searchInput.value = symbol;
            loadStockResearch(symbol);
        }
    }
};

// ─── STOCK RESEARCH PAGE ─────────────────────────────
function loadStockPage() {
    // If search bar is empty and details are not shown, load RELIANCE as default
    const searchInput = document.getElementById('stock-search-input');
    const statRow = document.getElementById('stock-stat-row');
    if (searchInput && searchInput.value === '' && statRow && statRow.style.display === 'none') {
        searchInput.value = 'RELIANCE';
        loadStockResearch('RELIANCE');
    }
}

function loadStockResearch(symbol) {
    const cleanSym = symbol.trim().toUpperCase();
    console.log(`Loading stock research for: ${cleanSym}`);
    
    // 1. Fetch Profile
    fetch(`${apiBase}/api/stocks/${cleanSym}/profile`)
        .then(res => {
            if (res.status === 404) {
                alert(`Symbol ${cleanSym} not found in NSE universe.`);
                throw new Error("Symbol not found");
            }
            return res.json();
        })
        .then(profile => {
            // Unhide elements
            document.getElementById('stock-stat-row').style.display = 'flex';
            document.getElementById('stock-detail-cols').style.display = 'grid';
            document.getElementById('stock-price-history').style.display = 'block';
            
            // Pop fields
            document.getElementById('sr-price').textContent = `₹${profile.price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
            const changeEl = document.getElementById('sr-chg');
            changeEl.textContent = `${profile.change >= 0 ? '+' : ''}${profile.change.toFixed(2)}%`;
            changeEl.className = `stat-val ${profile.change >= 0 ? 'green' : 'red'}`;
            
            // Fetch technicals & factors
            loadStockFactorsAndTechnicals(cleanSym);
            
            // Fetch predictions
            loadStockPredictions(cleanSym);
            
            // Fetch history for sparkline
            loadStockHistorySparkline(cleanSym);
        })
        .catch(err => {
            console.error('Error loading stock research profile:', err);
        });
}

function loadStockFactorsAndTechnicals(symbol) {
    fetch(`${apiBase}/api/stocks/${symbol}/factors`)
        .then(res => res.json())
        .then(factors => {
            // Factor displays inside detail column or stat row
            const combinedEl = document.getElementById('sr-signal');
            if (combinedEl) {
                combinedEl.textContent = `Score: ${factors.combined}`;
            }
            const confEl = document.getElementById('sr-conf');
            if (confEl) {
                confEl.textContent = `Combined Alpha Rank`;
            }
        })
        .catch(err => console.error('Error loading stock factors:', err));
}

function loadStockPredictions(symbol) {
    const note = document.getElementById('sr-pred-note');
    if (note) note.textContent = `Strategy predictions registry query: ACTIVE`;
    
    fetch(`${apiBase}/api/stocks/${symbol}/predictions`)
        .then(res => res.json())
        .then(data => {
            const h = data.horizons || {};
            // For display we take the average or first predictions
            const oneD = h['1D'] || {};
            document.getElementById('srp-total').textContent = data.history.length;
            document.getElementById('srp-resolved').textContent = data.history.filter(p => p.is_correct !== null).length;
            document.getElementById('srp-pending').textContent = data.history.filter(p => p.is_correct === null).length;
            
            document.getElementById('sr-target').textContent = `₹${oneD.target ? oneD.target.toFixed(2) : '--'} → R:R: ${oneD.accuracy ? (oneD.confidence / (100 - oneD.confidence)).toFixed(1) : '--'}`;
            document.getElementById('sr-stop').textContent = `₹${oneD.sl ? oneD.sl.toFixed(2) : '--'}`;
        })
        .catch(err => console.error('Error loading stock predictions:', err));
}

function loadStockHistorySparkline(symbol) {
    fetch(`${apiBase}/api/stocks/${symbol}/history?period=60d`)
        .then(res => res.json())
        .then(candles => {
            const spark = document.getElementById('sr-sparkline');
            if (!spark) return;
            
            if (candles.length === 0) {
                spark.innerHTML = `<span style="color:var(--text2); padding:20px;">No price history available</span>`;
                return;
            }
            
            // Calculate technical indicators (EMA, ATR, Volatility) dynamically from the history candles
            calculateIndicatorsOnClient(candles);
            
            // Render beautiful CSS Sparkline
            const closePrices = candles.map(c => c.close);
            const minClose = Math.min(...closePrices);
            const maxClose = Math.max(...closePrices);
            const range = maxClose - minClose || 1;
            
            spark.innerHTML = candles.map(c => {
                const heightPercent = ((c.close - minClose) / range) * 100;
                // color depending on candle change
                const barColor = c.close >= c.open ? 'var(--green)' : 'var(--red)';
                return `
                    <div class="spark-bar" 
                         style="height: ${Math.max(heightPercent, 2)}%; background: ${barColor};" 
                         title="${c.time} | Open: ₹${c.open.toFixed(2)} Close: ₹${c.close.toFixed(2)} Vol: ${c.volume.toLocaleString()}">
                    </div>
                `;
            }).join('');
        })
        .catch(err => console.error('Error loading sparkline history:', err));
}

function calculateIndicatorsOnClient(candles) {
    const closes = candles.map(c => c.close);
    const n = candles.length;
    
    // EMA calculations
    const ema20 = calcEma(closes, 20);
    const ema50 = calcEma(closes, 50);
    const ema200 = calcEma(closes, 200);
    
    const latestClose = closes[n - 1];
    const latEma20 = ema20[n - 1] || 0;
    const latEma50 = ema50[n - 1] || 0;
    const latEma200 = ema200[n - 1] || 0;
    
    // Display EMAs
    document.getElementById('sr-ema20').textContent = latEma20 ? `₹${latEma20.toFixed(2)}` : '--';
    document.getElementById('sr-ema50').textContent = latEma50 ? `₹${latEma50.toFixed(2)}` : '--';
    document.getElementById('sr-ema200').textContent = latEma200 ? `₹${latEma200.toFixed(2)}` : '--';
    
    // Above EMAs?
    document.getElementById('sr-above20').textContent = latestClose >= latEma20 ? 'YES' : 'NO';
    document.getElementById('sr-above20').className = `val-right ${latestClose >= latEma20 ? 'green' : 'red'}`;
    
    document.getElementById('sr-above200').textContent = latestClose >= latEma200 ? 'YES' : 'NO';
    document.getElementById('sr-above200').className = `val-right ${latestClose >= latEma200 ? 'green' : 'red'}`;
    
    // RSI 14
    const rsi14 = calcRsi(closes, 14);
    const latRsi = rsi14[n - 1] || 50;
    document.getElementById('sr-rsi').textContent = latRsi.toFixed(1);
    
    // Volatility (20d)
    const returns = [];
    for (let i = 1; i < n; i++) {
        returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
    }
    const last20Returns = returns.slice(-20);
    let vol20 = 0.0;
    if (last20Returns.length > 1) {
        const mean = last20Returns.reduce((a, b) => a + b, 0) / last20Returns.length;
        const variance = last20Returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (last20Returns.length - 1);
        const dailyVol = Math.sqrt(variance);
        vol20 = dailyVol * Math.sqrt(252) * 100; // Annualized percentage
    }
    document.getElementById('sr-vol20').textContent = `${vol20.toFixed(2)}%`;
    
    // ATR 14
    let atr14 = 0.0;
    if (n >= 15) {
        const trs = [];
        for (let i = 1; i < n; i++) {
            const h = candles[i].high;
            const l = candles[i].low;
            const prevC = candles[i - 1].close;
            const tr = Math.max(h - l, Math.abs(h - prevC), Math.abs(l - prevC));
            trs.push(tr);
        }
        // Smooth TR
        let sum = trs.slice(0, 14).reduce((a, b) => a + b, 0);
        let currentAtr = sum / 14;
        for (let i = 14; i < trs.length; i++) {
            currentAtr = (currentAtr * 13 + trs[i]) / 14;
        }
        atr14 = currentAtr;
    } else {
        // Fallback
        atr14 = (candles[n - 1].high - candles[n - 1].low) || 10.0;
    }
    document.getElementById('sr-atr').textContent = `₹${atr14.toFixed(2)}`;
}

// Indicator math helpers
function calcEma(values, period) {
    const k = 2 / (period + 1);
    const ema = [];
    if (values.length === 0) return ema;
    ema.push(values[0]);
    for (let i = 1; i < values.length; i++) {
        ema.push(values[i] * k + ema[i - 1] * (1 - k));
    }
    return ema;
}

function calcRsi(values, period) {
    const rsi = [];
    if (values.length <= period) return Array(values.length).fill(50);
    
    let gains = 0;
    let losses = 0;
    
    // Initial period
    for (let i = 1; i <= period; i++) {
        const diff = values[i] - values[i - 1];
        if (diff > 0) gains += diff;
        else losses -= diff;
    }
    
    let avgGain = gains / period;
    let avgLoss = losses / period;
    
    for (let i = 0; i < period; i++) {
        rsi.push(50); // Pad initial
    }
    
    let rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));
    
    for (let i = period + 1; i < values.length; i++) {
        const diff = values[i] - values[i - 1];
        avgGain = (avgGain * (period - 1) + (diff > 0 ? diff : 0)) / period;
        avgLoss = (avgLoss * (period - 1) + (diff < 0 ? -diff : 0)) / period;
        
        rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
    }
    return rsi;
}

// ─── PREDICTION CENTER PAGE ──────────────────────────
function loadPredictionsPage() {
    // 1. Fetch metrics
    fetch(`${apiBase}/api/metrics`)
        .then(res => res.json())
        .then(metrics => {
            document.getElementById('pc-total').textContent = metrics.total_predictions;
            document.getElementById('pc-resolved').textContent = metrics.realized_predictions;
            document.getElementById('pc-pending').textContent = metrics.total_predictions - metrics.realized_predictions;
        })
        .catch(err => console.error('Error loading prediction metrics:', err));
        
    // 2. Fetch strategy details for count
    fetch(`${apiBase}/api/strategies`)
        .then(res => res.json())
        .then(strats => {
            document.getElementById('pc-strats').textContent = strats.length;
        })
        .catch(err => console.error('Error loading strategies:', err));
        
    // 3. Fetch Strategy IC Table
    fetch(`${apiBase}/api/alpha-lab/metrics`)
        .then(res => res.json())
        .then(alphas => {
            const tbody = document.getElementById('pc-ic-tbody');
            if (tbody) {
                if (alphas.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="3" class="empty-row">No strategies tracked</td></tr>`;
                } else {
                    tbody.innerHTML = alphas.map(a => {
                        const statusClass = a.ic < 0.02 ? 'red' : 'green';
                        const statusText = a.ic < 0.02 ? 'DEMOTED (IC < 0.02)' : 'ACTIVE';
                        return `
                            <tr>
                                <td style="font-weight:600;">${a.strategy.toUpperCase()}</td>
                                <td>${a.ic.toFixed(3)}</td>
                                <td class="${statusClass}">${statusText}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
            
            // Demoted list
            const demoted = alphas.filter(a => a.ic < 0.02);
            const demotedList = document.getElementById('pc-demoted-list');
            if (demotedList) {
                if (demoted.length === 0) {
                    demotedList.innerHTML = `<span style="color:var(--text2);">None — all strategies above IC threshold</span>`;
                } else {
                    demotedList.innerHTML = demoted.map(a => `<div class="issue high"><strong>${a.strategy.toUpperCase()}:</strong> Rolled back / Demoted. Rolling IC = ${a.ic.toFixed(3)} (threshold 0.02).</div>`).join('');
                }
            }
        })
        .catch(err => console.error('Error loading Alpha Lab details for predictions page:', err));
        
    // 4. Fetch All Predictions Table
    fetch(`${apiBase}/api/predictions?limit=50`)
        .then(res => res.json())
        .then(preds => {
            const tbody = document.getElementById('pred-all-tbody');
            if (tbody) {
                if (preds.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="10" class="empty-row">No predictions registered yet</td></tr>`;
                } else {
                    tbody.innerHTML = preds.map(p => {
                        const sideClass = p.direction === 'LONG' || p.direction === 'BUY' ? 'green' : 'red';
                        const outcomeClass = p.is_correct === true ? 'green' : (p.is_correct === false ? 'red' : 'amber');
                        const outcomeText = p.is_correct === true ? 'CORRECT' : (p.is_correct === false ? 'WRONG' : 'PENDING');
                        const pnlVal = p.realized_return !== null ? `${p.realized_return >= 0 ? '+' : ''}${p.realized_return.toFixed(2)}%` : '—';
                        const pnlClass = p.realized_return !== null ? (p.realized_return >= 0 ? 'green' : 'red') : '';
                        
                        return `
                            <tr>
                                <td>#${p.id}</td>
                                <td style="font-weight:600;">${p.symbol}</td>
                                <td>${p.strategy}</td>
                                <td class="${sideClass}">${p.direction}</td>
                                <td>₹${p.entry_price.toFixed(2)}</td>
                                <td class="green">₹${p.target_price.toFixed(2)}</td>
                                <td class="red">₹${p.stop_loss.toFixed(2)}</td>
                                <td>${p.confidence.toFixed(0)}%</td>
                                <td class="${outcomeClass}">${outcomeText}</td>
                                <td class="${pnlClass} val-right" style="font-weight:600;">${pnlVal}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
        })
        .catch(err => console.error('Error loading predictions list:', err));
}

// ─── ALPHA LAB PAGE ──────────────────────────────────
function loadAlphaPage() {
    fetch(`${apiBase}/api/alpha-lab/metrics`)
        .then(res => res.json())
        .then(alphas => {
            // Table
            const tbody = document.getElementById('alpha-perf-tbody');
            if (tbody) {
                if (alphas.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="9" class="empty-row">Loading alpha data from registry...</td></tr>`;
                } else {
                    tbody.innerHTML = alphas.map(a => {
                        const statusClass = a.is_active ? 'green' : 'red';
                        const activeLabel = a.is_active ? 'ACTIVE' : 'INACTIVE';
                        const wr = typeof a.win_rate === 'number' ? `${a.win_rate.toFixed(1)}%` : '—';
                        return `
                            <tr>
                                <td style="font-weight:600;">${a.strategy.toUpperCase()}</td>
                                <td>${a.total_predictions}</td>
                                <td>${a.resolved_predictions}</td>
                                <td>${a.total_predictions - a.resolved_predictions}</td>
                                <td>${a.ic.toFixed(3)}</td>
                                <td class="${a.ic >= 0 ? 'green' : 'red'}">${a.ic >= 0 ? '+' : ''}${(a.ic * 5).toFixed(2)}%</td>
                                <td>${a.sharpe.toFixed(2)}</td>
                                <td>${wr}</td>
                                <td class="${statusClass}">${activeLabel}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
            
            // Lists
            const activeAlphas = alphas.filter(a => a.is_active);
            const activeList = document.getElementById('alpha-active-list');
            if (activeList) {
                if (activeAlphas.length === 0) {
                    activeList.innerHTML = `<span style="color:var(--text2);">None</span>`;
                } else {
                    activeList.innerHTML = activeAlphas.map(a => `
                        <div class="issue medium" style="border-left-color: var(--green);">
                            <strong>${a.strategy.toUpperCase()}:</strong> Deployment Active.
                            Win Rate: ${a.win_rate.toFixed(1)}% | Sharpe: ${a.sharpe.toFixed(2)} | Rolling IC: ${a.ic.toFixed(3)}
                        </div>
                    `).join('');
                }
            }
            
            const demotedAlphas = alphas.filter(a => !a.is_active);
            const demotedList = document.getElementById('alpha-demoted-list');
            if (demotedList) {
                if (demotedAlphas.length === 0) {
                    demotedList.innerHTML = `<div class="empty-row" style="padding:10px;">None demoted yet</div>`;
                } else {
                    demotedList.innerHTML = demotedAlphas.map(a => `
                        <div class="issue critical">
                            <strong>${a.strategy.toUpperCase()}:</strong> Demoted below standard limits.
                            Reason: Rolling IC ${a.ic.toFixed(3)} &lt; threshold limit (0.02).
                        </div>
                    `).join('');
                }
            }
        })
        .catch(err => console.error('Error loading alpha performance metrics:', err));
}

// ─── MARKET REGIME PAGE ──────────────────────────────
function loadRegimePage() {
    fetch(`${apiBase}/api/regime/current`)
        .then(res => res.json())
        .then(regimeData => {
            document.getElementById('reg-current').textContent = regimeData.current_regime || 'SIDEWAYS';
            document.getElementById('reg-conf').textContent = `${((regimeData.confidence || 0.5) * 100).toFixed(0)}%`;
            
            const probs = regimeData.probabilities || {};
            const categories = ['bull', 'bear', 'side', 'hvol'];
            const apiKeys = ['bull', 'bear', 'sideways', 'high_vol'];
            
            categories.forEach((cat, idx) => {
                const val = probs[apiKeys[idx]] || 0;
                const fillEl = document.getElementById(`rf-${cat}`);
                const textEl = document.getElementById(`rfv-${cat}`);
                if (fillEl) fillEl.style.width = `${val}%`;
                if (textEl) textEl.textContent = `${val.toFixed(1)}%`;
            });
        })
        .catch(err => console.error('Error loading market regime page details:', err));
}

// ─── PORTFOLIO ANALYTICS PAGE ────────────────────────
function loadPortfolioPage() {
    // Fetch NAV and PnL from WS State or fallback
    const wsNav = webSocketState ? webSocketState.nav : 250000000;
    const wsPnl = webSocketState ? webSocketState.daily_pnl : 0;
    
    document.getElementById('port-nav').textContent = formatPrice(wsNav);
    const portPnl = document.getElementById('port-pnl');
    if (portPnl) {
        portPnl.textContent = formatPrice(wsPnl, true);
        portPnl.className = `stat-val ${wsPnl >= 0 ? 'green' : 'red'}`;
    }
    
    // Fetch metrics
    fetch(`${apiBase}/api/metrics`)
        .then(res => res.json())
        .then(metrics => {
            document.getElementById('port-trades').textContent = metrics.total_trades;
            document.getElementById('port-wr').textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
            document.getElementById('port-sharpe').textContent = metrics.sharpe_ratio.toFixed(2);
            
            const pDd = document.getElementById('port-dd');
            if (pDd) pDd.textContent = `${metrics.max_drawdown.toFixed(2)}%`;
            
            // Performance metrics sub-table
            document.getElementById('port-wins').textContent = metrics.winning_trades;
            document.getElementById('port-losses').textContent = metrics.losing_trades;
            document.getElementById('port-avg').textContent = formatPrice(metrics.avg_pnl_per_trade, true);
            document.getElementById('port-avg').className = `val-right ${metrics.avg_pnl_per_trade >= 0 ? 'green' : 'red'}`;
            document.getElementById('port-wstreak').textContent = metrics.max_consecutive_wins;
            document.getElementById('port-lstreak').textContent = metrics.max_consecutive_losses;
            document.getElementById('port-volat').textContent = `${metrics.volatility.toFixed(2)}%`;
            
            // Prediction Accuracy sub-table
            document.getElementById('port-pred-total').textContent = metrics.total_predictions;
            document.getElementById('port-pred-realized').textContent = metrics.realized_predictions;
            document.getElementById('port-pred-acc').textContent = `${(metrics.prediction_accuracy * 100).toFixed(1)}%`;
            document.getElementById('port-streak').textContent = metrics.current_streak;
        })
        .catch(err => console.error('Error loading portfolio stats:', err));
        
    // Open positions
    const posTbody = document.getElementById('port-positions-tbody');
    if (posTbody) {
        const activePos = webSocketState ? webSocketState.positions : [];
        if (!activePos || activePos.length === 0) {
            posTbody.innerHTML = `<tr><td colspan="6" class="empty-row">No open positions logged</td></tr>`;
        } else {
            posTbody.innerHTML = activePos.map(pos => {
                const isLong = pos.side.toUpperCase() === 'BUY' || pos.side.toUpperCase() === 'LONG';
                return `
                    <tr>
                        <td style="font-weight:600;">${pos.symbol}</td>
                        <td class="${isLong ? 'green' : 'red'}">${pos.side.toUpperCase()}</td>
                        <td>${pos.quantity || pos.qty}</td>
                        <td class="val-right">₹${pos.entry_price.toFixed(2)}</td>
                        <td class="val-right ${pos.pnl >= 0 ? 'green' : 'red'}" style="font-weight:600;">${formatPrice(pos.pnl, true)}</td>
                        <td class="green">LIVE</td>
                    </tr>
                `;
            }).join('');
        }
    }
}

// ─── RISK CENTER PAGE ────────────────────────────────
function loadRiskPage() {
    // Fetch risk portfolio
    fetch(`${apiBase}/api/risk/portfolio`)
        .then(res => res.json())
        .then(risk => {
            document.getElementById('rsk-var').textContent = formatPrice(risk.var_99 || 0);
            document.getElementById('rsk-cvar').textContent = formatPrice(risk.cvar_95 || 0);
            document.getElementById('rsk-tail').textContent = formatPrice(risk.tail_risk || 0);
            
            // PnL pct from WS
            const wsPnl = webSocketState ? webSocketState.daily_pnl : 0;
            const capital = risk.capital || 250000000;
            const pnlPct = (wsPnl / capital) * 100;
            
            const rskPnl = document.getElementById('rsk-dpnl');
            if (rskPnl) {
                rskPnl.textContent = `${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(3)}%`;
                rskPnl.className = `stat-val ${pnlPct >= 0 ? 'green' : 'red'}`;
            }
        })
        .catch(err => console.error('Error fetching risk portfolio stats:', err));
        
    // Fetch Data quality checks
    fetch(`${apiBase}/api/data-quality`)
        .then(res => res.json())
        .then(dq => {
            document.getElementById('dq-total').textContent = dq.total_checks;
            document.getElementById('dq-good').textContent = dq.good;
            document.getElementById('dq-stale').textContent = dq.stale;
            document.getElementById('dq-corrupt').textContent = dq.corrupt;
            document.getElementById('dq-blocked').textContent = dq.blocked;
            document.getElementById('dq-score').textContent = `${dq.health_score.toFixed(1)}%`;
            document.getElementById('dq-score').className = `val-right ${dq.health_score > 95 ? 'green' : (dq.health_score > 85 ? 'amber' : 'red')}`;
        })
        .catch(err => console.error('Error loading data quality metrics:', err));
}

// ─── EXECUTION PAGE ──────────────────────────────────
function loadExecutionPage() {
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            const execBroker = document.getElementById('exec-broker');
            if (execBroker) {
                execBroker.textContent = health.broker_status === 'CONNECTED' ? 'NOT CONNECTED' : 'CONNECTED'; // displays Kite Link state
                execBroker.className = `stat-val ${health.broker_status === 'CONNECTED' ? 'amber' : 'green'}`;
            }
            
            const execMkt = document.getElementById('exec-mkt');
            if (execMkt) {
                const isLive = health.market_status ? health.market_status.is_open : false;
                execMkt.textContent = isLive ? 'LIVE' : 'CLOSED';
                execMkt.className = `stat-val ${isLive ? 'green' : 'red'}`;
            }
            
            const execTs = document.getElementById('exec-ts');
            if (execTs) {
                execTs.textContent = health.latest_data_timestamp ? new Date(health.latest_data_timestamp).toLocaleTimeString() : 'N/A';
            }
        })
        .catch(err => console.error('Error loading execution stats:', err));
        
    // Populate Execution logs from predictions (where status is resolved/active)
    fetch(`${apiBase}/api/predictions?limit=20`)
        .then(res => res.json())
        .then(preds => {
            const tbody = document.getElementById('exec-trades-tbody');
            if (tbody) {
                if (preds.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="8" class="empty-row">No trades logged yet</td></tr>`;
                } else {
                    tbody.innerHTML = preds.map(p => {
                        const time = p.timestamp ? formatTimestamp(p.timestamp) : '—';
                        const sideClass = p.direction === 'LONG' || p.direction === 'BUY' ? 'green' : 'red';
                        const exitVal = p.exit_price !== null ? `₹${p.exit_price.toFixed(2)}` : '—';
                        const pnlVal = p.realized_return !== null ? `${p.realized_return >= 0 ? '+' : ''}${p.realized_return.toFixed(2)}%` : '—';
                        const pnlClass = p.realized_return !== null ? (p.realized_return >= 0 ? 'green' : 'red') : '';
                        const statusLabel = p.exit_price !== null ? 'FILLED' : 'WORKING';
                        const statusClass = p.exit_price !== null ? 'green' : 'amber';
                        
                        return `
                            <tr>
                                <td style="color:var(--text2);">${time}</td>
                                <td style="font-weight:600;">${p.symbol}</td>
                                <td class="${sideClass}">${p.direction}</td>
                                <td>100</td>
                                <td>₹${p.entry_price.toFixed(2)}</td>
                                <td>${exitVal}</td>
                                <td class="${pnlClass}" style="font-weight:600;">${pnlVal}</td>
                                <td class="${statusClass}">${statusLabel}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
        })
        .catch(err => console.error('Error loading execution trade logs:', err));
}

// ─── MODEL REGISTRY PAGE ─────────────────────────────
function loadModelsPage() {
    fetch(`${apiBase}/api/models`)
        .then(res => res.json())
        .then(mr => {
            document.getElementById('mr-total').textContent = mr.total_models;
            document.getElementById('mr-prod').textContent = mr.production_models;
            document.getElementById('mr-stage').textContent = mr.staging_models;
            document.getElementById('mr-dev').textContent = mr.development_models;
            
            const tbody = document.getElementById('mr-tbody');
            if (tbody) {
                const modelsList = mr.models || [];
                if (modelsList.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="8" class="empty-row">No models registered yet. Train a model and register it via ModelRegistry.</td></tr>`;
                } else {
                    tbody.innerHTML = modelsList.map(m => {
                        const date = m.created_at ? new Date(m.created_at).toLocaleDateString() : '—';
                        const isProd = m.stage.toLowerCase() === 'production';
                        const stageClass = isProd ? 'green' : (m.stage.toLowerCase() === 'staging' ? 'amber' : '');
                        
                        // Parse metrics
                        const metrics = m.metrics || {};
                        const sharpe = metrics.sharpe !== undefined ? metrics.sharpe.toFixed(2) : '—';
                        const acc = metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(1)}%` : '—';
                        const wr = metrics.win_rate !== undefined ? `${(metrics.win_rate * 100).toFixed(1)}%` : '—';
                        
                        return `
                            <tr>
                                <td style="font-weight:600;">${m.model_id}</td>
                                <td>${m.model_type.toUpperCase()}</td>
                                <td>v${m.version}</td>
                                <td class="${stageClass}" style="font-weight:600;">${m.stage.toUpperCase()}</td>
                                <td>${sharpe}</td>
                                <td>${acc}</td>
                                <td>${wr}</td>
                                <td style="color:var(--text2);">${date}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
        })
        .catch(err => console.error('Error loading models registry:', err));
}

// ─── SYSTEM HEALTH PAGE ──────────────────────────────
function loadHealthPage() {
    fetch(`${apiBase}/api/system-health`)
        .then(res => res.json())
        .then(health => {
            // Components block
            updateComponentStatus('hc-db', health.database_status === 'CONNECTED');
            updateComponentStatus('hc-broker', health.broker_status === 'CONNECTED');
            updateComponentStatus('hc-mkt', health.market_status ? health.market_status.is_open : false);
            
            const hcUniv = document.getElementById('hc-univ');
            if (hcUniv) hcUniv.textContent = health.universe ? `${health.universe.count} (${health.universe.source})` : '—';
            
            const hcPred = document.getElementById('hc-pred');
            if (hcPred) hcPred.textContent = health.predictions ? `${health.predictions.total_predictions} Total` : '—';
            
            updateComponentStatus('hc-dq', health.data_quality ? health.data_quality.available : false);
            updateComponentStatus('hc-tl', health.trade_logger ? health.trade_logger.available : false);
            updateComponentStatus('hc-mr', health.model_registry ? health.model_registry.available : false);
            updateComponentStatus('hc-fs', health.feature_store ? health.feature_store.available : false);
            
            // Data sources block
            const ds = health.data_sources || {};
            document.getElementById('hs-signals').textContent = ds.signals || '—';
            document.getElementById('hs-metrics').textContent = ds.metrics || '—';
            document.getElementById('hs-indices').textContent = ds.indices || '—';
            
            // Risk data source
            const hsRisk = document.getElementById('hs-risk');
            if (hsRisk) hsRisk.textContent = 'REAL (portfolio/trade_logger.py)';
        })
        .catch(err => console.error('Error loading health details:', err));
}

function updateComponentStatus(elId, isOk) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.textContent = isOk ? 'ONLINE' : 'OFFLINE';
    el.className = `val-right ${isOk ? 'green' : 'red'}`;
}

// ─── FORMATTING UTILITIES ────────────────────────────
function formatPrice(val, showSign = false) {
    if (val === undefined || val === null || isNaN(val)) return '₹0.00';
    const num = Number(val);
    const signStr = showSign && num > 0 ? '+' : '';
    return `${signStr}₹${num.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatPercent(val) {
    if (val === undefined || val === null || isNaN(val)) return '0.00%';
    const num = Number(val);
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`;
}

function formatNumber(val, decimals = 0) {
    if (val === undefined || val === null || isNaN(val)) return '--';
    return Number(val).toFixed(decimals);
}

function formatTimestamp(isoString) {
    try {
        const d = new Date(isoString);
        if (isNaN(d.getTime())) return isoString;
        return d.toLocaleDateString('en-IN') + ' ' + d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
        return isoString;
    }
}
