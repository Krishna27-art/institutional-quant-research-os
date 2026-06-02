'use client';

import { useEffect, useState } from 'react';

export default function Home() {
  const [activeScreen, setActiveScreen] = useState('market');
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotMessages, setCopilotMessages] = useState([
    { type: 'ai', content: 'Hello! I\'m monitoring your portfolio. Current regime is <strong>Bull Trend</strong> (87% confidence). No critical alerts at this time. What would you like to analyze?' },
    { type: 'user', content: 'Why did ORB trigger a long on RELIANCE at 9:35?' },
    { type: 'ai', content: 'At 9:30–9:35, RELIANCE opened higher (+0.5%) and held above its opening range. Relative Volume was 2.4× average. Top features: RV (0.32), VWAP distance (0.28), momentum (0.21). Regime: Bull Trend. Similar patterns in Mar 2023, Aug 2023 yielded avg 1.2R gain.' }
  ]);
  const [copilotInput, setCopilotInput] = useState('');
  const [clock, setClock] = useState('');
  const [refreshSec, setRefreshSec] = useState(0);
  const [replyIdx, setReplyIdx] = useState(0);
  
  const copilotReplies = [
    "Analyzing current market conditions... VaR is within safe zone at 2.14%. Regime confidence is strong at 87%.",
    "Based on feature drift analysis, VWAP-distance and relative volume show the strongest predictive power this week.",
    "I recommend monitoring the BANKNIFTY position — PSI drift detected on key features. Consider reducing exposure by 15%.",
    "Rolling Sharpe on ORB Momentum has declined to 1.2 from 2.1 over 30 days. Investigate regime-conditional performance.",
    "Stress test suggests COVID-like scenario would draw down portfolio by 12.4%. Current hedges cover 40% of downside."
  ];

  const screens = ['market', 'regime', 'alpha', 'risk', 'portfolio', 'options', 'signals', 'alerts'];

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, '0');
      const m = String(now.getMinutes()).padStart(2, '0');
      const s = String(now.getSeconds()).padStart(2, '0');
      setClock(`${h}:${m}:${s} IST`);
    };
    updateClock();
    const clockInterval = setInterval(updateClock, 1000);
    
    const refreshInterval = setInterval(() => {
      setRefreshSec(prev => (prev + 1) % 30);
    }, 1000);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.altKey && e.key >= '1' && e.key <= '8') {
        setActiveScreen(screens[parseInt(e.key) - 1]);
      }
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        setCopilotOpen(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      clearInterval(clockInterval);
      clearInterval(refreshInterval);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const sendCopilot = () => {
    const val = copilotInput.trim();
    if (!val) return;
    setCopilotMessages(prev => [...prev, { type: 'user', content: val }]);
    setCopilotInput('');
    setTimeout(() => {
      setCopilotMessages(prev => [...prev, { type: 'ai', content: copilotReplies[replyIdx % copilotReplies.length] }]);
      setReplyIdx(prev => prev + 1);
    }, 800);
  };

  return (
    <div className="os-shell">
      <div className="topbar">
        <div className="logo">QROS</div>
        <div className="nav-tabs">
          {screens.map((screen, i) => (
            <button 
              key={screen}
              className={`nav-tab ${activeScreen === screen ? 'active' : ''}`}
              onClick={() => setActiveScreen(screen)}
            >
              {screen.charAt(0).toUpperCase() + screen.slice(1).replace('-', ' ')}
            </button>
          ))}
        </div>
        <div className="topbar-right">
          <div className="regime-badge">BULL TREND · HMM-0</div>
          <div className="clock">{clock}</div>
        </div>
      </div>

      <div className="main-area">
        <div className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-label">Markets</div>
            <div className={`sidebar-item ${activeScreen === 'market' ? 'active' : ''}`} onClick={() => setActiveScreen('market')}><span className="dot"></span>Global Intelligence</div>
            <div className={`sidebar-item ${activeScreen === 'regime' ? 'active' : ''}`} onClick={() => setActiveScreen('regime')}><span className="dot"></span>Regime Center</div>
            <div className={`sidebar-item ${activeScreen === 'signals' ? 'active' : ''}`} onClick={() => setActiveScreen('signals')}><span className="dot"></span>Signal Monitor</div>
          </div>
          <div className="sidebar-divider"></div>
          <div className="sidebar-section">
            <div className="sidebar-label">Research</div>
            <div className={`sidebar-item ${activeScreen === 'alpha' ? 'active' : ''}`} onClick={() => setActiveScreen('alpha')}><span className="dot"></span>Alpha Lab</div>
            <div className="sidebar-item"><span className="dot"></span>Backtest Center</div>
            <div className="sidebar-item"><span className="dot"></span>Experiment Track</div>
            <div className="sidebar-item"><span className="dot"></span>Feature Explorer</div>
          </div>
          <div className="sidebar-divider"></div>
          <div className="sidebar-section">
            <div className="sidebar-label">Risk & Portfolio</div>
            <div className={`sidebar-item ${activeScreen === 'risk' ? 'active' : ''}`} onClick={() => setActiveScreen('risk')}><span className="dot"></span>Risk War Room</div>
            <div className={`sidebar-item ${activeScreen === 'portfolio' ? 'active' : ''}`} onClick={() => setActiveScreen('portfolio')}><span className="dot"></span>Portfolio Command</div>
            <div className={`sidebar-item ${activeScreen === 'options' ? 'active' : ''}`} onClick={() => setActiveScreen('options')}><span className="dot"></span>Options Analytics</div>
            <div className="sidebar-item"><span className="dot"></span>Liquidity Maps</div>
          </div>
          <div className="sidebar-divider"></div>
          <div className="sidebar-section">
            <div className="sidebar-label">Intelligence</div>
            <div className={`sidebar-item ${activeScreen === 'alerts' ? 'active' : ''}`} onClick={() => setActiveScreen('alerts')}><span className="dot"></span>Alert Center</div>
            <div className="sidebar-item"><span className="dot"></span>Flow Intelligence</div>
            <div className="sidebar-item"><span className="dot"></span>Correlation Maps</div>
            <div className="sidebar-item"><span className="dot"></span>Data Quality</div>
          </div>
        </div>

        <div className="content">
          {activeScreen === 'market' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">NIFTY 50</div><div className="metric-val pos">24,550</div><div className="metric-sub">+73.2 · +0.30%</div></div>
                <div className="metric"><div className="metric-label">BANKNIFTY</div><div className="metric-val pos">52,120</div><div className="metric-sub">+260.5 · +0.50%</div></div>
                <div className="metric"><div className="metric-label">INDIA VIX</div><div className="metric-val neg">14.20</div><div className="metric-sub">-0.18 · -1.25%</div></div>
                <div className="metric"><div className="metric-label">USD/INR</div><div className="metric-val">83.47</div><div className="metric-sub">+0.12 · +0.14%</div></div>
              </div>
              <div className="grid-2" style={{marginBottom:'12px'}}>
                <div className="card">
                  <div className="card-header"><span className="card-title">Sector Heatmap</span><span className="card-action">1D · 5D · 1M</span></div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:'3px'}}>
                    {[
                      {name:'IT',val:1.2},{name:'Pharma',val:0.8},{name:'FMCG',val:0.5},
                      {name:'Auto',val:-0.2},{name:'Metals',val:-0.8},{name:'PSU Bank',val:1.4},
                      {name:'Pvt Bank',val:0.6},{name:'Realty',val:2.1},{name:'Energy',val:-0.4},
                      {name:'Infra',val:0.3},{name:'Media',val:-1.2},{name:'Telecom',val:0.9}
                    ].map(s => {
                      const intensity = Math.abs(s.val) / 2.5;
                      const bg = s.val > 0 ? `rgba(34,197,94,${0.15 + intensity * 0.55})` : `rgba(239,68,68,${0.15 + intensity * 0.55})`;
                      const fc = s.val > 0 ? '#22c55e' : '#ef4444';
                      return (
                        <div key={s.name} style={{background:bg,color:fc,borderRadius:'3px',display:'flex',alignItems:'center',justifyContent:'center',fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:600,cursor:'pointer',padding:'6px 4px',flexDirection:'column',gap:'2px'}}>
                          <span style={{fontSize:'8px',fontWeight:400,opacity:0.8,color:fc}}>{s.name}</span>
                          <span>{s.val > 0 ? '+' : ''}{s.val.toFixed(1)}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Market Breadth</span><span className="card-action">NSE 500</span></div>
                  <div style={{display:'flex',flexDirection:'column',gap:'10px',paddingTop:'4px'}}>
                    <div className="exp-row"><div className="exp-label">ADV/DEC</div><div className="exp-bar-wrap"><div className="exp-bar" style={{width:'65%',background:'var(--green)'}}></div></div><div className="exp-val" style={{color:'var(--green)'}}>323/177</div></div>
                    <div className="exp-row"><div className="exp-label">New Highs</div><div className="exp-bar-wrap"><div className="exp-bar" style={{width:'45%',background:'var(--cyan2)'}}></div></div><div className="exp-val">45</div></div>
                    <div className="exp-row"><div className="exp-label">New Lows</div><div className="exp-bar-wrap"><div className="exp-bar" style={{width:'12%',background:'var(--red)'}}></div></div><div className="exp-val" style={{color:'var(--red)'}}>12</div></div>
                    <div className="exp-row"><div className="exp-label">Above 200D</div><div className="exp-bar-wrap"><div className="exp-bar" style={{width:'72%',background:'var(--blue)'}}></div></div><div className="exp-val">72%</div></div>
                    <div className="exp-row"><div className="exp-label">FII Flow</div><div className="exp-bar-wrap"><div className="exp-bar" style={{width:'58%',background:'var(--purple)'}}></div></div><div className="exp-val" style={{color:'var(--green)'}}>+₹2.3K</div></div>
                  </div>
                </div>
              </div>
              <div className="grid-2">
                <div className="card">
                  <div className="card-header"><span className="card-title">Institutional Flow (FII/DII)</span><span className="card-action">₹ Cr</span></div>
                  <table className="data-table">
                    <thead><tr><th>Type</th><th>Buy</th><th>Sell</th><th>Net</th><th>5D Cum</th></tr></thead>
                    <tbody>
                      <tr><td className="sym">FII</td><td className="pos">8,423</td><td>6,180</td><td className="pos">+2,243</td><td className="pos">+5,812</td></tr>
                      <tr><td className="sym">DII</td><td className="pos">4,102</td><td>5,089</td><td className="neg">-987</td><td className="neg">-1,230</td></tr>
                      <tr><td className="sym">MF</td><td className="pos">2,801</td><td>2,100</td><td className="pos">+701</td><td className="pos">+3,200</td></tr>
                      <tr><td className="sym">Prop</td><td>9,211</td><td>8,990</td><td className="pos">+221</td><td className="neg">-440</td></tr>
                    </tbody>
                  </table>
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Economic Calendar</span><span className="card-action">June 2025</span></div>
                  <div style={{display:'flex',flexDirection:'column',gap:0}}>
                    <div className="alert-item"><div className="alert-icon alert-red">!</div><div><div className="alert-title">RBI MPC Decision</div><div className="alert-meta">Jun 07 · 10:00 IST · HIGH IMPACT</div></div></div>
                    <div className="alert-item"><div className="alert-icon alert-amber">~</div><div><div className="alert-title">CPI Inflation Data</div><div className="alert-meta">Jun 12 · 17:30 IST · MED IMPACT</div></div></div>
                    <div className="alert-item"><div className="alert-icon alert-amber">~</div><div><div className="alert-title">IIP Industrial Output</div><div className="alert-meta">Jun 12 · 17:30 IST · MED IMPACT</div></div></div>
                    <div className="alert-item"><div className="alert-icon alert-blue">i</div><div><div className="alert-title">FOMC Minutes</div><div className="alert-meta">Jun 19 · 01:00 IST · WATCH</div></div></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeScreen === 'regime' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">Current Regime</div><div className="metric-val" style={{color:'var(--green)',fontSize:'14px'}}>BULL TREND</div><div className="metric-sub">HMM State 0 · 87% conf</div></div>
                <div className="metric"><div className="metric-label">Duration</div><div className="metric-val">34d</div><div className="metric-sub">Avg regime: 28.4d</div></div>
                <div className="metric"><div className="metric-label">Transition Prob</div><div className="metric-val" style={{color:'var(--amber)'}}>12.3%</div><div className="metric-sub">→ High Vol in 5d</div></div>
                <div className="metric"><div className="metric-label">Regime Sharpe</div><div className="metric-val pos">2.14</div><div className="metric-sub">vs 0.87 in Bear</div></div>
              </div>
              <div className="card" style={{marginBottom:'12px'}}>
                <div className="card-header"><span className="card-title">HMM Regime Timeline (24 months)</span></div>
                <div style={{padding:'4px 0',display:'flex',gap:'1px',height:'32px'}}>
                  {[0,0,1,0,0,0,2,0,0,0,1,0,0,0,0,3,0,0,0,0,0,0,0,0].map((r, i) => (
                    <div key={i} style={{flex:1,background:['#22c55e','#ef4444','#f59e0b','#3b82f6'][r],borderRadius:'2px'}} title={`Month ${i + 1}`}></div>
                  ))}
                </div>
                <div style={{display:'flex',gap:'16px',marginTop:'10px'}}>
                  <div style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'10px',color:'var(--text2)'}}><div style={{width:'12px',height:'12px',borderRadius:'2px',background:'#22c55e'}}></div>Bull Trend (0)</div>
                  <div style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'10px',color:'var(--text2)'}}><div style={{width:'12px',height:'12px',borderRadius:'2px',background:'#ef4444'}}></div>Bear Trend (1)</div>
                  <div style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'10px',color:'var(--text2)'}}><div style={{width:'12px',height:'12px',borderRadius:'2px',background:'#f59e0b'}}></div>High Vol (2)</div>
                  <div style={{display:'flex',alignItems:'center',gap:'6px',fontSize:'10px',color:'var(--text2)'}}><div style={{width:'12px',height:'12px',borderRadius:'2px',background:'#3b82f6'}}></div>Low Vol (3)</div>
                </div>
              </div>
              <div className="grid-2">
                <div className="card">
                  <div className="card-header"><span className="card-title">Feature Importance</span></div>
                  {[
                    {name:'NIFTY 20d Return',val:0.38},{name:'VIX Level',val:0.31},
                    {name:'Advance/Decline',val:0.22},{name:'FII Flow 5d',val:0.18},
                    {name:'VWAP Slope',val:0.14},{name:'Volume Z-Score',val:0.09}
                  ].map(f => (
                    <div key={f.name} style={{display:'flex',alignItems:'center',gap:'10px',padding:'7px 0',borderBottom:'0.5px solid rgba(0,255,200,0.05)'}}>
                      <div style={{fontSize:'11px',color:'var(--text1)',width:'90px'}}>{f.name}</div>
                      <div style={{flex:1,height:'4px',background:'var(--bg3)',borderRadius:'2px'}}><div style={{height:'100%',borderRadius:'2px',width:`${f.val*100}%`,background:'var(--cyan2)'}}></div></div>
                      <div style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text2)',width:'46px',textAlign:'right'}}>{f.val.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Transition Matrix</span></div>
                  <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:'2px',fontSize:'9px',fontFamily:'var(--font-mono)'}}>
                    {[
                      ['','Bull','Bear','HiVol','LoVol'],
                      ['Bull','0.82','0.08','0.07','0.03'],
                      ['Bear','0.11','0.74','0.12','0.03'],
                      ['HiVol','0.15','0.21','0.58','0.06'],
                      ['LoVol','0.20','0.05','0.08','0.67']
                    ].map((row, ri) => row.map((cell, ci) => (
                      <div key={`${ri}-${ci}`} style={{padding:'5px 2px',textAlign:'center',borderRadius:'3px',fontSize:'9px',color:ri === 0 || ci === 0 ? 'var(--text3)' : ri === ci ? 'var(--cyan)' : 'var(--text2)',fontWeight:ri === 0 || ci === 0 ? '600' : 'normal',background:ri > 0 && ci > 0 ? (ri === ci ? `rgba(0,255,200,${parseFloat(cell) * 0.8})` : `rgba(239,68,68,${parseFloat(cell) * 0.4})`) : 'transparent',fontFamily:ri > 0 && ci > 0 ? 'var(--font-mono)' : 'inherit'}}>{cell}</div>
                    )))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeScreen === 'alpha' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">Live Alphas</div><div className="metric-val">8</div></div>
                <div className="metric"><div className="metric-label">Avg Sharpe</div><div className="metric-val pos">1.87</div></div>
                <div className="metric"><div className="metric-label">Best Alpha</div><div className="metric-val pos">3.21</div><div className="metric-sub">ORB Momentum</div></div>
                <div className="metric"><div className="metric-label">Correlation</div><div className="metric-val">0.18</div><div className="metric-sub">avg alpha-alpha</div></div>
              </div>
              <div className="grid-2">
                <div className="card">
                  <div className="card-header"><span className="card-title">Alpha Catalog</span><span className="card-action">+ New Alpha</span></div>
                  {[
                    {name:'ORB Momentum',sharpe:3.21,status:'live',decay:'18d'},
                    {name:'VWAP Reversion',sharpe:2.67,status:'live',decay:'24d'},
                    {name:'Mean Reversion v7',sharpe:1.94,status:'paper',decay:'31d'},
                    {name:'Momentum + Vol',sharpe:1.82,status:'live',decay:'22d'},
                    {name:'Gap Fill Strategy',sharpe:1.44,status:'live',decay:'15d'},
                    {name:'FII Flow Alpha',sharpe:1.12,status:'research',decay:'42d'},
                  ].map(a => (
                    <div key={a.name} style={{display:'flex',alignItems:'center',gap:'10px',padding:'8px 0',borderBottom:'0.5px solid rgba(0,255,200,0.05)',cursor:'pointer'}}>
                      <div style={{fontSize:'12px',color:'var(--text1)',flex:1}}>{a.name}</div>
                      <span style={{fontSize:'9px',padding:'2px 6px',borderRadius:'3px',fontFamily:'var(--font-mono)',background:a.status === 'live' ? 'rgba(34,197,94,0.15)' : a.status === 'paper' ? 'rgba(59,130,246,0.15)' : 'rgba(167,139,250,0.15)',color:a.status === 'live' ? 'var(--green)' : a.status === 'paper' ? 'var(--blue)' : 'var(--purple)',border:`0.5px solid ${a.status === 'live' ? 'rgba(34,197,94,0.3)' : a.status === 'paper' ? 'rgba(59,130,246,0.3)' : 'rgba(167,139,250,0.3)'}`}}>{a.status.toUpperCase()}</span>
                      <div style={{fontFamily:'var(--font-mono)',fontSize:'11px',color:'var(--green)',width:'36px',textAlign:'right'}}>{a.sharpe.toFixed(2)}</div>
                      <div style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)',width:'30px',textAlign:'right'}}>{a.decay}</div>
                    </div>
                  ))}
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Alpha Performance</span></div>
                  <div style={{position:'relative',width:'100%',height:'200px',display:'flex',alignItems:'flex-end',gap:'8px',padding:'20px'}}>
                    {[
                      {name:'ORB Mom',val:3.21,color:'#00c8a0'},
                      {name:'VWAP Rev',val:2.67,color:'#00c8a0'},
                      {name:'MR v7',val:1.94,color:'#3b82f6'},
                      {name:'Mom+Vol',val:1.82,color:'#00c8a0'},
                      {name:'Gap Fill',val:1.44,color:'#22c55e'},
                      {name:'FII Flow',val:1.12,color:'#a78bfa'}
                    ].map(d => (
                      <div key={d.name} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:'4px'}}>
                        <div style={{width:'100%',height:`${(d.val/3.5)*100}%`,background:d.color,borderRadius:'4px 4px 0 0',transition:'height 0.3s'}}></div>
                        <div style={{fontSize:'8px',color:'var(--text2)',textAlign:'center',fontFamily:'var(--font-mono)'}}>{d.val.toFixed(2)}</div>
                        <div style={{fontSize:'8px',color:'var(--text3)',textAlign:'center'}}>{d.name}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeScreen === 'risk' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">Portfolio VaR (95%)</div><div className="metric-val neg">₹2.14Cr</div><div className="metric-sub">2.14% of AUM</div></div>
                <div className="metric"><div className="metric-label">CVaR (95%)</div><div className="metric-val neg">₹3.82Cr</div><div className="metric-sub">3.82% of AUM</div></div>
                <div className="metric"><div className="metric-label">Gross Exposure</div><div className="metric-val" style={{color:'var(--amber)'}}>142%</div><div className="metric-sub">Limit: 180%</div></div>
                <div className="metric"><div className="metric-label">Max Drawdown</div><div className="metric-val neg">-4.21%</div><div className="metric-sub">MTD</div></div>
              </div>
              <div className="grid-2" style={{marginBottom:'12px'}}>
                <div className="card">
                  <div className="card-header"><span className="card-title">VaR Gauge</span></div>
                  <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:'8px',padding:'8px 0'}}>
                    <svg width="160" height="90" viewBox="0 0 160 90">
                      <defs>
                        <linearGradient id="gGrad" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="#22c55e"/>
                          <stop offset="50%" stopColor="#f59e0b"/>
                          <stop offset="100%" stopColor="#ef4444"/>
                        </linearGradient>
                      </defs>
                      <path d="M20,80 A60,60 0 0,1 140,80" stroke="url(#gGrad)" strokeWidth="10" fill="none" strokeLinecap="round"/>
                      <path d="M20,80 A60,60 0 0,1 140,80" stroke="rgba(255,255,255,0.05)" strokeWidth="10" fill="none"/>
                      <line x1="80" y1="80" x2="47" y2="35" stroke="#00ffc8" strokeWidth="2" strokeLinecap="round"/>
                      <circle cx="80" cy="80" r="4" fill="#00ffc8"/>
                      <text x="15" y="90" fontSize="8" fill="#475569">0%</text>
                      <text x="70" y="26" fontSize="8" fill="#475569">2.5%</text>
                      <text x="132" y="90" fontSize="8" fill="#475569">5%</text>
                    </svg>
                    <div style={{fontFamily:'var(--font-mono)',fontSize:'20px',fontWeight:700,color:'var(--red)'}}>2.14%</div>
                    <div style={{fontSize:'10px',color:'var(--text2)'}}>VaR · SAFE ZONE</div>
                  </div>
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Risk Decomposition</span></div>
                  {[
                    {name:'Market Beta',val:0.48,color:'var(--blue)'},
                    {name:'Sector Risk',val:0.28,color:'var(--purple)'},
                    {name:'Idiosyncratic',val:0.14,color:'var(--amber)'},
                    {name:'Factor Risk',val:0.10,color:'var(--red)'},
                  ].map(item => (
                    <div key={item.name} className="exp-row">
                      <div className="exp-label">{item.name}</div>
                      <div className="exp-bar-wrap"><div className="exp-bar" style={{width:`${item.val*100}%`,background:item.color}}></div></div>
                      <div className="exp-val">{Math.round(item.val*100)}%</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <div className="card-header"><span className="card-title">Stress Test Scenarios</span><span className="card-action">Run Scenario</span></div>
                <table className="data-table">
                  <thead><tr><th>Scenario</th><th>Period</th><th>NIFTY Drop</th><th>VaR</th><th>Portfolio P&L</th><th>Drawdown</th><th>Status</th></tr></thead>
                  <tbody>
                    <tr><td className="sym">COVID Crash</td><td>Feb–Mar 2020</td><td className="neg">-38%</td><td className="neg">₹8.2Cr</td><td className="neg">-₹12.4Cr</td><td className="neg">-12.4%</td><td><span style={{fontSize:'9px',padding:'2px 6px',borderRadius:'3px',fontFamily:'var(--font-mono)',background:'rgba(59,130,246,0.15)',color:'var(--blue)',border:'0.5px solid rgba(59,130,246,0.3)'}}>SIMULATE</span></td></tr>
                    <tr><td className="sym">2008 GFC</td><td>Oct–Nov 2008</td><td className="neg">-52%</td><td className="neg">₹14.1Cr</td><td className="neg">-₹18.7Cr</td><td className="neg">-18.7%</td><td><span style={{fontSize:'9px',padding:'2px 6px',borderRadius:'3px',fontFamily:'var(--font-mono)',background:'rgba(59,130,246,0.15)',color:'var(--blue)',border:'0.5px solid rgba(59,130,246,0.3)'}}>SIMULATE</span></td></tr>
                    <tr><td className="sym">IL&FS Crisis</td><td>Sep 2018</td><td className="neg">-14%</td><td className="neg">₹4.2Cr</td><td className="neg">-₹3.8Cr</td><td className="neg">-3.8%</td><td><span style={{fontSize:'9px',padding:'2px 6px',borderRadius:'3px',fontFamily:'var(--font-mono)',background:'rgba(34,197,94,0.15)',color:'var(--green)',border:'0.5px solid rgba(34,197,94,0.3)'}}>PASS</span></td></tr>
                    <tr><td className="sym">Rate Shock +200bps</td><td>Hypothetical</td><td className="neg">-22%</td><td className="neg">₹6.8Cr</td><td className="neg">-₹7.2Cr</td><td className="neg">-7.2%</td><td><span style={{fontSize:'9px',padding:'2px 6px',borderRadius:'3px',fontFamily:'var(--font-mono)',background:'rgba(167,139,250,0.15)',color:'var(--purple)',border:'0.5px solid rgba(167,139,250,0.3)'}}>WATCH</span></td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeScreen === 'portfolio' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">AUM</div><div className="metric-val">₹500Cr</div></div>
                <div className="metric"><div className="metric-label">Daily P&L</div><div className="metric-val pos">+₹1.42Cr</div><div className="metric-sub">+0.28% today</div></div>
                <div className="metric"><div className="metric-label">MTD P&L</div><div className="metric-val pos">+₹8.74Cr</div><div className="metric-sub">+1.75% MTD</div></div>
                <div className="metric"><div className="metric-label">Net Exposure</div><div className="metric-val">62%</div><div className="metric-sub">Long 82% · Short 20%</div></div>
              </div>
              <div className="grid-2" style={{marginBottom:'12px'}}>
                <div className="card">
                  <div className="card-header"><span className="card-title">Holdings</span><span className="card-action">Export</span></div>
                  <table className="data-table">
                    <thead><tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>LTP</th><th>P&L</th><th>% AUM</th></tr></thead>
                    <tbody>
                      <tr><td className="sym">RELIANCE</td><td>2,400</td><td>2,801</td><td>2,956</td><td className="pos">+₹3.72L</td><td>1.42%</td></tr>
                      <tr><td className="sym">HDFCBANK</td><td>3,100</td><td>1,620</td><td>1,598</td><td className="neg">-₹0.68L</td><td>0.99%</td></tr>
                      <tr><td className="sym">INFY</td><td>5,000</td><td>1,480</td><td>1,562</td><td className="pos">+₹4.10L</td><td>1.56%</td></tr>
                      <tr><td className="sym">TCS</td><td>800</td><td>3,821</td><td>3,949</td><td className="pos">+₹1.02L</td><td>0.63%</td></tr>
                      <tr><td className="sym">WIPRO</td><td>8,000</td><td>487</td><td>471</td><td className="neg">-₹1.28L</td><td>0.75%</td></tr>
                      <tr><td className="sym">AXISBANK</td><td>4,200</td><td>1,098</td><td>1,143</td><td className="pos">+₹1.89L</td><td>0.96%</td></tr>
                    </tbody>
                  </table>
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">Sector Exposure</span></div>
                  {[
                    {name:'Financials',val:0.31,color:'var(--blue)'},
                    {name:'IT',val:0.22,color:'var(--purple)'},
                    {name:'Energy',val:0.14,color:'var(--amber)'},
                    {name:'FMCG',val:0.12,color:'var(--cyan2)'},
                    {name:'Auto',val:0.09,color:'var(--green)'},
                    {name:'Others',val:0.12,color:'var(--text3)'},
                  ].map(s => (
                    <div key={s.name} className="exp-row">
                      <div className="exp-label">{s.name}</div>
                      <div className="exp-bar-wrap"><div className="exp-bar" style={{width:`${s.val*100}%`,background:s.color}}></div></div>
                      <div className="exp-val">{Math.round(s.val*100)}%</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <div className="card-header"><span className="card-title">P&L Attribution</span></div>
                <div style={{position:'relative',width:'100%',height:'160px',display:'flex',alignItems:'flex-end',gap:'8px',padding:'20px'}}>
                  {[
                    {name:'ORB Mom',val:3.72,color:'#22c55e'},
                    {name:'VWAP Rev',val:2.44,color:'#22c55e'},
                    {name:'IT Sector',val:1.89,color:'#22c55e'},
                    {name:'Fin Sector',val:-0.68,color:'#ef4444'},
                    {name:'MR v7',val:1.02,color:'#22c55e'},
                    {name:'Auto Sector',val:-0.42,color:'#ef4444'}
                  ].map(d => (
                    <div key={d.name} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:'4px'}}>
                      <div style={{width:'100%',height:`${(Math.abs(d.val)/4)*100}%`,background:d.color,borderRadius:'4px 4px 0 0',transition:'height 0.3s'}}></div>
                      <div style={{fontSize:'8px',color:'var(--text2)',textAlign:'center',fontFamily:'var(--font-mono)'}}>{d.val.toFixed(2)}</div>
                      <div style={{fontSize:'8px',color:'var(--text3)',textAlign:'center'}}>{d.name}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeScreen === 'options' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">ATM IV (NIFTY)</div><div className="metric-val">14.80%</div><div className="metric-sub">-0.42% vs 30d avg</div></div>
                <div className="metric"><div className="metric-label">IV Rank</div><div className="metric-val" style={{color:'var(--amber)'}}>28</div><div className="metric-sub">0–100 scale</div></div>
                <div className="metric"><div className="metric-label">PCR (OI)</div><div className="metric-val pos">1.24</div><div className="metric-sub">Bullish signal</div></div>
                <div className="metric"><div className="metric-label">Max Pain</div><div className="metric-val">24,500</div><div className="metric-sub">Jun 27 expiry</div></div>
              </div>
              <div className="grid-2" style={{marginBottom:'12px'}}>
                <div className="card">
                  <div className="card-header"><span className="card-title">IV Term Structure</span></div>
                  <div style={{position:'relative',width:'100%',height:'180px',display:'flex',alignItems:'flex-end',gap:'4px',padding:'20px'}}>
                    {[
                      {label:'Jun 27',call:14.8,put:15.2},
                      {label:'Jul 31',call:15.4,put:15.9},
                      {label:'Aug 28',call:16.1,put:16.5},
                      {label:'Sep 25',call:16.8,put:17.2},
                      {label:'Dec 25',call:17.9,put:18.3},
                      {label:'Mar 26',call:18.4,put:18.9}
                    ].map(d => (
                      <div key={d.label} style={{flex:1,display:'flex',flexDirection:'column',gap:'2px'}}>
                        <div style={{height:`${(d.call/20)*100}%`,background:'#00c8a0',borderRadius:'2px 2px 0 0'}}></div>
                        <div style={{height:`${(d.put/20)*100}%`,background:'#ef4444',borderRadius:'0 0 2px 2px'}}></div>
                        <div style={{fontSize:'7px',color:'var(--text3)',textAlign:'center'}}>{d.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="card">
                  <div className="card-header"><span className="card-title">GEX by Strike</span></div>
                  <div style={{position:'relative',width:'100%',height:'180px',display:'flex',alignItems:'center',gap:'4px',padding:'20px'}}>
                    {[
                      {label:'24200',val:-120},{label:'24300',val:80},{label:'24400',val:340},
                      {label:'24500',val:820},{label:'24600',val:-210},{label:'24700',val:-480},{label:'24800',val:-90}
                    ].map(d => (
                      <div key={d.label} style={{flex:1,display:'flex',flexDirection:'column',alignItems:'center',gap:'4px'}}>
                        <div style={{width:'100%',height:`${(Math.abs(d.val)/850)*100}%`,background:d.val >= 0 ? 'rgba(0,200,160,0.7)' : 'rgba(239,68,68,0.7)',borderRadius:'4px',transition:'height 0.3s'}}></div>
                        <div style={{fontSize:'7px',color:'var(--text3)',textAlign:'center'}}>{d.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="card">
                <div className="card-header"><span className="card-title">Options Chain (NIFTY Jun 27)</span></div>
                <table className="data-table">
                  <thead><tr><th>Call OI</th><th>Call IV</th><th>Call LTP</th><th>Strike</th><th>Put LTP</th><th>Put IV</th><th>Put OI</th></tr></thead>
                  <tbody>
                    {[24200,24300,24400,24500,24600,24700,24800].map(k => {
                      const dist = Math.abs(k - 24550) / 100;
                      const callIV = (14.8 + dist * 0.4 + (k > 24550 ? dist*0.2 : 0)).toFixed(2);
                      const putIV = (14.8 + dist * 0.5 + (k < 24550 ? dist*0.3 : 0)).toFixed(2);
                      const callLTP = Math.max(5, Math.round(k > 24550 ? 500 - (k-24550)*1.2 : 500 + (24550-k)*0.9));
                      const putLTP = Math.max(5, Math.round(k < 24550 ? 500 - (24550-k)*1.2 : 500 + (k-24550)*0.9));
                      const isATM = Math.abs(k - 24550) <= 50;
                      return (
                        <tr key={k} style={isATM ? {background:'rgba(0,255,200,0.04)'} : {}}>
                          <td>{(Math.round(Math.random()*80 + 40) * 100).toLocaleString()}</td>
                          <td>{callIV}%</td>
                          <td className="pos">{callLTP}</td>
                          <td className="sym" style={{textAlign:'center'}}>{k}{isATM?' ★':''}</td>
                          <td className="neg">{putLTP}</td>
                          <td>{putIV}%</td>
                          <td>{(Math.round(Math.random()*90 + 50) * 100).toLocaleString()}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeScreen === 'signals' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">Active Signals</div><div className="metric-val">23</div></div>
                <div className="metric"><div className="metric-label">Hit Rate (5d)</div><div className="metric-val pos">62.3%</div></div>
                <div className="metric"><div className="metric-label">Avg R (5d)</div><div className="metric-val pos">+0.82R</div></div>
                <div className="metric"><div className="metric-label">Signal Strength</div><div className="metric-val pos">HIGH</div><div className="metric-sub">↑ vs yesterday</div></div>
              </div>
              <div className="card" style={{marginBottom:'12px'}}>
                <div className="card-header"><span className="card-title">Live Signal Stream</span><span className="card-action">refreshed {refreshSec}s ago</span></div>
                {[
                  {sym:'RELIANCE',dir:'LONG',alpha:'ORB Momentum',str:0.82,time:'15:28:41'},
                  {sym:'HDFCBANK',dir:'SHORT',alpha:'VWAP Reversion',str:0.61,time:'15:27:18'},
                  {sym:'TCS',dir:'LONG',alpha:'Momentum+Vol',str:0.74,time:'15:25:03'},
                  {sym:'INFY',dir:'LONG',alpha:'ORB Momentum',str:0.68,time:'15:22:49'},
                  {sym:'AXISBANK',dir:'SHORT',alpha:'Mean Rev v7',str:0.55,time:'15:19:31'},
                  {sym:'WIPRO',dir:'LONG',alpha:'Gap Fill',str:0.49,time:'15:15:07'},
                ].map(s => (
                  <div key={s.sym} style={{display:'flex',alignItems:'center',gap:'10px',padding:'7px 0',borderBottom:'0.5px solid rgba(0,255,200,0.05)'}}>
                    <div style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)',width:'60px'}}>{s.time}</div>
                    <div style={{fontSize:'11px',color:'var(--text1)',fontWeight:600,width:'80px'}}>{s.sym}</div>
                    <div style={{fontSize:'10px',color:s.dir === 'LONG' ? 'var(--green)' : 'var(--red)',width:'46px',fontWeight:700}}>{s.dir}</div>
                    <div style={{fontSize:'10px',color:'var(--text2)',flex:1}}>{s.alpha}</div>
                    <div style={{width:'80px',height:'4px',background:'var(--bg3)',borderRadius:'2px'}}><div style={{height:'100%',borderRadius:'2px',width:`${s.str*100}%`,background:s.dir === 'LONG' ? 'var(--green)' : 'var(--red)'}}></div></div>
                    <div style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:s.dir === 'LONG' ? 'var(--green)' : 'var(--red)'}}>{s.str.toFixed(2)}</div>
                  </div>
                ))}
              </div>
              <div className="card">
                <div className="card-header"><span className="card-title">Signal Quality by Alpha</span></div>
                {[
                  {name:'ORB Momentum',hit:68.2,avgR:1.21},{name:'VWAP Reversion',hit:63.4,avgR:0.94},
                  {name:'Momentum+Vol',hit:58.7,avgR:0.82},{name:'Mean Rev v7',hit:54.1,avgR:0.71},
                  {name:'Gap Fill',hit:61.2,avgR:0.88},{name:'FII Flow Alpha',hit:52.3,avgR:0.61},
                ].map(a => (
                  <div key={a.name} style={{display:'flex',alignItems:'center',gap:'10px',padding:'7px 0',borderBottom:'0.5px solid rgba(0,255,200,0.05)'}}>
                    <div style={{fontSize:'11px',color:'var(--text1)',width:'90px'}}>{a.name}</div>
                    <div style={{flex:1,height:'4px',background:'var(--bg3)',borderRadius:'2px'}}><div style={{height:'100%',borderRadius:'2px',width:`${a.hit}%`,background:'var(--cyan2)'}}></div></div>
                    <div style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--green)',width:'46px',textAlign:'right'}}>{a.hit.toFixed(1)}%</div>
                    <div style={{fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--amber)',width:'40px',textAlign:'right'}}>{a.avgR.toFixed(2)}R</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeScreen === 'alerts' && (
            <div className="screen active">
              <div className="grid-4" style={{marginBottom:'12px'}}>
                <div className="metric"><div className="metric-label">Alerts Today</div><div className="metric-val neg">7</div></div>
                <div className="metric"><div className="metric-label">Critical</div><div className="metric-val neg">2</div></div>
                <div className="metric"><div className="metric-label">Warning</div><div className="metric-val" style={{color:'var(--amber)'}}>3</div></div>
                <div className="metric"><div className="metric-label">False Positive</div><div className="metric-val">12%</div><div className="metric-sub">30d avg: 18%</div></div>
              </div>
              <div className="card" style={{marginBottom:'12px'}}>
                <div className="card-header"><span className="card-title">Alert Feed</span><span className="card-action">Filter · Rules</span></div>
                <div className="alert-item"><div className="alert-icon alert-red">!</div><div><div className="alert-title">VaR Breach Warning — Portfolio VaR approaching 2.5% threshold</div><div className="alert-meta">15:24:18 IST · RISK ENGINE · UNACKNOWLEDGED</div></div></div>
                <div className="alert-item"><div className="alert-icon alert-red">!</div><div><div className="alert-title">Feature Drift Detected — VWAP distance PSI &gt; 0.25 on BANKNIFTY</div><div className="alert-meta">14:52:03 IST · AI MONITOR · UNACKNOWLEDGED</div></div></div>
                <div className="alert-item"><div className="alert-icon alert-amber">~</div><div><div className="alert-title">Regime Transition Signal — HMM probability shift: High Vol state at 28%</div><div className="alert-meta">14:31:47 IST · REGIME ENGINE · ACKNOWLEDGED</div></div></div>
                <div className="alert-item"><div className="alert-icon alert-amber">~</div><div><div className="alert-title">Alpha Decay Alert — ORB Momentum 30d Sharpe dropped to 1.2 from 2.1</div><div className="alert-meta">13:15:22 IST · ALPHA ENGINE · ACKNOWLEDGED</div></div></div>
                <div className="alert-item"><div className="alert-icon alert-blue">i</div><div><div className="alert-title">Backtest Complete — Mean Reversion v7 finished: Sharpe 1.94, MaxDD -4.1%</div><div className="alert-meta">12:44:01 IST · BACKTEST ENGINE · INFO</div></div></div>
                <div className="alert-item"><div className="alert-icon alert-amber">~</div><div><div className="alert-title">Concentration Risk — Top 5 holdings exceed 28% of AUM</div><div className="alert-meta">11:20:15 IST · RISK ENGINE · ACKNOWLEDGED</div></div></div>
                <div className="alert-item"><div className="alert-icon alert-blue">i</div><div><div className="alert-title">Data Gap Detected — NSEMD feed delayed by 4.2s at 10:32 IST</div><div className="alert-meta">10:33:08 IST · DATA QUALITY · RESOLVED</div></div></div>
              </div>
              <div className="card">
                <div className="card-header"><span className="card-title">System Status</span></div>
                <div style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 0',fontSize:'11px',color:'var(--text2)'}}><div style={{width:'6px',height:'6px',borderRadius:'50%',background:'var(--green)',flexShrink:0}}></div><span>Alpha Engine</span><span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)'}}>OPERATIONAL · 99.98% uptime</span></div>
                <div style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 0',fontSize:'11px',color:'var(--text2)'}}><div style={{width:'6px',height:'6px',borderRadius:'50%',background:'var(--green)',flexShrink:0}}></div><span>Risk Engine</span><span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)'}}>OPERATIONAL · 8ms latency</span></div>
                <div style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 0',fontSize:'11px',color:'var(--text2)'}}><div style={{width:'6px',height:'6px',borderRadius:'50%',background:'var(--amber)',flexShrink:0}}></div><span>NSE Feed</span><span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--amber)'}}>DEGRADED · +4.2s delay</span></div>
                <div style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 0',fontSize:'11px',color:'var(--text2)'}}><div style={{width:'6px',height:'6px',borderRadius:'50%',background:'var(--green)',flexShrink:0}}></div><span>Redis Cache</span><span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)'}}>OPERATIONAL · 0.4ms</span></div>
                <div style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 0',fontSize:'11px',color:'var(--text2)'}}><div style={{width:'6px',height:'6px',borderRadius:'50%',background:'var(--green)',flexShrink:0}}></div><span>ClickHouse</span><span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)'}}>OPERATIONAL · 12ms</span></div>
                <div style={{display:'flex',alignItems:'center',gap:'8px',padding:'6px 0',fontSize:'11px',color:'var(--text2)'}}><div style={{width:'6px',height:'6px',borderRadius:'50%',background:'var(--green)',flexShrink:0}}></div><span>AI Copilot API</span><span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'10px',color:'var(--text3)'}}>OPERATIONAL · 1.8s avg</span></div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="statusbar">
        <div className="sb-item"><div className="sb-dot"></div>LIVE</div>
        <div className="sb-item">NIFTY 24,550.30 +0.30%</div>
        <div className="sb-item">VIX 14.20</div>
        <div className="sb-item">VaR 2.14%</div>
        <div className="sb-item">Regime: BULL TREND</div>
        <div className="sb-item" style={{marginLeft:'auto'}}>WebSocket · 23ms · 0 errors</div>
        <div className="sb-item">Alt+1-8 Switch Screen</div>
      </div>

      <button className="copilot-btn" onClick={() => setCopilotOpen(!copilotOpen)}>✦</button>
      <div className={`copilot-panel ${copilotOpen ? 'open' : ''}`}>
        <div className="copilot-header"><span>✦ AI COPILOT</span><span style={{cursor:'pointer',color:'var(--text3)'}} onClick={() => setCopilotOpen(false)}>✕</span></div>
        <div className="copilot-messages">
          {copilotMessages.map((msg, i) => (
            <div key={i} className={msg.type === 'ai' ? 'msg-ai' : 'msg-user'} dangerouslySetInnerHTML={{__html: msg.content}}></div>
          ))}
        </div>
        <div className="copilot-input-row">
          <input 
            className="copilot-input" 
            placeholder="Ask about signals, risk, P&L..."
            value={copilotInput}
            onChange={(e) => setCopilotInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendCopilot()}
          />
          <button className="copilot-send" onClick={sendCopilot}>→</button>
        </div>
      </div>
    </div>
  );
}
