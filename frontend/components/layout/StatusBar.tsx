'use client'

import { useState, useEffect } from 'react'

export default function StatusBar() {
  const [currentTime, setCurrentTime] = useState('')

  useEffect(() => {
    setCurrentTime(new Date().toLocaleTimeString('en-US', { hour12: false }))
    const interval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString('en-US', { hour12: false }))
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <footer className="h-8 bg-surface border-t border-surface-light fixed left-64 right-0 bottom-0 flex items-center justify-between px-4 text-xs">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <span className="text-text-secondary">CPU:</span>
          <span className="text-text-primary font-mono">45%</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-text-secondary">Memory:</span>
          <span className="text-text-primary font-mono">2.4GB / 8GB</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-text-secondary">WebSocket:</span>
          <span className="text-success">● Connected</span>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <span className="text-text-secondary">Data Latency:</span>
          <span className="text-text-primary font-mono">12ms</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-text-secondary">Alerts:</span>
          <span className="text-warning font-mono">3</span>
        </div>
        <div className="text-text-secondary" suppressHydrationWarning>
          {currentTime}
        </div>
      </div>
    </footer>
  )
}
