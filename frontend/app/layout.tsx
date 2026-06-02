import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Institutional Quant Research OS',
  description: 'Next-generation quantitative research operating system',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
