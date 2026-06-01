// Quant Trading Dashboard JavaScript

// Simulated data updates
const portfolioData = {
    totalPnL: 1245678,
    dailyPnL: 123456,
    sharpeRatio: 1.35,
    maxDrawdown: -8.5
};

const positions = [
    { symbol: 'RELIANCE', side: 'BUY', quantity: 100, entryPrice: 2450, pnl: 12500 },
    { symbol: 'HDFCBANK', side: 'BUY', quantity: 50, entryPrice: 1550, pnl: -5000 },
    { symbol: 'NIFTY FUT', side: 'SELL', quantity: 2, entryPrice: 20100, pnl: 25000 }
];

const strategies = [
    { name: 'ORB Strategy', sharpe: 1.2, winRate: 65 },
    { name: 'VWAP Trend', sharpe: 0.9, winRate: 58 },
    { name: 'Put-Call Carry', sharpe: 0.7, winRate: 72 },
    { name: 'Volatility Carry', sharpe: 0.6, winRate: 68 }
];

const regimeProbabilities = {
    bull: 65,
    bear: 15,
    sideways: 12,
    highVol: 8
};

// Update dashboard with real-time data
function updateDashboard() {
    // Update portfolio metrics
    updatePortfolioMetrics();
    
    // Update positions table
    updatePositionsTable();
    
    // Update regime probabilities
    updateRegimeProbabilities();
}

function updatePortfolioMetrics() {
    // Simulate random updates
    portfolioData.dailyPnL += Math.random() * 10000 - 5000;
    portfolioData.totalPnL += portfolioData.dailyPnL;
    
    // Update DOM elements
    const metrics = document.querySelectorAll('.metric .value');
    if (metrics.length >= 4) {
        metrics[0].textContent = `₹${portfolioData.totalPnL.toLocaleString()}`;
        metrics[0].className = `value ${portfolioData.dailyPnL >= 0 ? 'positive' : 'negative'}`;
        metrics[1].textContent = `₹${portfolioData.dailyPnL.toLocaleString()}`;
        metrics[1].className = `value ${portfolioData.dailyPnL >= 0 ? 'positive' : 'negative'}`;
    }
}

function updatePositionsTable() {
    // Simulate PnL updates
    positions.forEach(pos => {
        pos.pnl += Math.random() * 1000 - 500;
    });
    
    // Update table
    const tbody = document.querySelector('.positions-table tbody');
    if (tbody) {
        tbody.innerHTML = positions.map(pos => `
            <tr>
                <td>${pos.symbol}</td>
                <td class="${pos.side.toLowerCase()}">${pos.side}</td>
                <td>${pos.quantity}</td>
                <td>₹${pos.entryPrice.toLocaleString()}</td>
                <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">₹${pos.pnl.toLocaleString()}</td>
            </tr>
        `).join('');
    }
}

function updateRegimeProbabilities() {
    // Simulate regime changes
    regimeProbabilities.bull += Math.random() * 2 - 1;
    regimeProbabilities.bear += Math.random() * 2 - 1;
    regimeProbabilities.sideways += Math.random() * 2 - 1;
    regimeProbabilities.highVol += Math.random() * 2 - 1;
    
    // Normalize to 100%
    const total = regimeProbabilities.bull + regimeProbabilities.bear + 
                   regimeProbabilities.sideways + regimeProbabilities.highVol;
    regimeProbabilities.bull = (regimeProbabilities.bull / total * 100).toFixed(0);
    regimeProbabilities.bear = (regimeProbabilities.bear / total * 100).toFixed(0);
    regimeProbabilities.sideways = (regimeProbabilities.sideways / total * 100).toFixed(0);
    regimeProbabilities.highVol = (regimeProbabilities.highVol / total * 100).toFixed(0);
    
    // Update progress bars
    const probItems = document.querySelectorAll('.prob-item');
    if (probItems.length >= 4) {
        probItems[0].querySelector('.progress-fill').style.width = `${regimeProbabilities.bull}%`;
        probItems[0].querySelector('span:last-child').textContent = `${regimeProbabilities.bull}%`;
        
        probItems[1].querySelector('.progress-fill').style.width = `${regimeProbabilities.bear}%`;
        probItems[1].querySelector('span:last-child').textContent = `${regimeProbabilities.bear}%`;
        
        probItems[2].querySelector('.progress-fill').style.width = `${regimeProbabilities.sideways}%`;
        probItems[2].querySelector('span:last-child').textContent = `${regimeProbabilities.sideways}%`;
        
        probItems[3].querySelector('.progress-fill').style.width = `${regimeProbabilities.highVol}%`;
        probItems[3].querySelector('span:last-child').textContent = `${regimeProbabilities.highVol}%`;
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    updateDashboard();
    
    // Update every 5 seconds
    setInterval(updateDashboard, 5000);
});

// WebSocket connection for real-time data (placeholder)
function connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/ws/signals');
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Handle real-time signal updates
        console.log('Signal received:', data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Call WebSocket connection (will fail if server not running)
// connectWebSocket();
