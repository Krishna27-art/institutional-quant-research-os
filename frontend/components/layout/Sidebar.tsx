'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navigation = [
  { name: 'Home', href: '/', icon: '🏠' },
  { name: 'Market Intelligence', href: '/screens/market-intelligence', icon: '📊' },
  { name: 'Regime Command', href: '/screens/regime-command', icon: '🎯' },
  { name: 'Alpha Lab', href: '/screens/alpha-lab', icon: '🧪' },
  { name: 'Portfolio Command', href: '/screens/portfolio-command', icon: '💼' },
  { name: 'Risk War Room', href: '/screens/risk-war-room', icon: '⚠️' },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="w-64 bg-surface border-r border-surface-light h-screen fixed left-0 top-0 flex flex-col">
      <div className="p-6 border-b border-surface-light">
        <h1 className="text-xl font-bold text-text-primary">Quant OS</h1>
        <p className="text-xs text-text-secondary mt-1">Institutional Research</p>
      </div>
      
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navigation.map((item) => {
            const isActive = pathname === item.href
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={`flex items-center px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary text-white'
                      : 'text-text-secondary hover:bg-surface-light hover:text-text-primary'
                  }`}
                >
                  <span className="mr-3">{item.icon}</span>
                  <span className="text-sm font-medium">{item.name}</span>
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-surface-light">
        <div className="text-xs text-text-secondary">
          <div className="flex items-center justify-between mb-2">
            <span>System Status</span>
            <span className="text-success">● Online</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Latency</span>
            <span>12ms</span>
          </div>
        </div>
      </div>
    </div>
  )
}
