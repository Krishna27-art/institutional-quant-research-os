'use client'

export default function Header() {
  return (
    <header className="h-16 bg-surface border-b border-surface-light fixed left-64 right-0 top-0 flex items-center justify-between px-6">
      <div className="flex items-center space-x-4">
        <h2 className="text-lg font-semibold text-text-primary">Dashboard</h2>
        <span className="text-sm text-text-secondary">|</span>
        <span className="text-sm text-text-secondary">Last updated: 12:45:32 IST</span>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 text-sm">
          <span className="text-text-secondary">NIFTY:</span>
          <span className="text-success font-mono">22,450.75</span>
          <span className="text-success text-xs">+1.25%</span>
        </div>
        
        <div className="h-6 w-px bg-surface-light"></div>
        
        <div className="flex items-center space-x-3">
          <button className="p-2 rounded hover:bg-surface-light transition-colors">
            <span className="text-lg">🔔</span>
          </button>
          <button className="p-2 rounded hover:bg-surface-light transition-colors">
            <span className="text-lg">⚙️</span>
          </button>
          <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-white text-sm font-medium">
            QS
          </div>
        </div>
      </div>
    </header>
  )
}
