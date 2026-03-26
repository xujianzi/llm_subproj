import { useState } from 'react'
import { MapView }     from './components/MapView/MapView'
import { ConfigPanel } from './components/ConfigPanel/ConfigPanel'
import { ChatPanel }   from './components/ChatPanel/ChatPanel'
import { StatsCard }   from './components/StatsCard/StatsCard'
import { DataTable }   from './components/DataTable/DataTable'

type Tab = 'map' | 'table'

export default function App() {
  const [tab, setTab] = useState<Tab>('map')

  const tabCls = (t: Tab) =>
    `px-4 py-2 text-sm font-medium transition border-b-2 ${
      tab === t
        ? 'border-primary text-primary'
        : 'border-transparent text-muted hover:text-text'
    }`

  return (
    <div className="bg-bg h-screen flex flex-col text-text">
      {/* Tab bar */}
      <nav className="flex border-b border-border px-4 pt-2 bg-panel z-30 relative">
        <button className={tabCls('map')}   onClick={() => setTab('map')}>
          Map Visualization
        </button>
        <button className={tabCls('table')} onClick={() => setTab('table')}>
          Data Table
        </button>
      </nav>

      {/* Map tab — full screen relative container */}
      {tab === 'map' && (
        <div className="flex-1 relative overflow-hidden">
          <MapView />
          <ConfigPanel />
          <StatsCard />
          <ChatPanel />
        </div>
      )}

      {/* Data table tab */}
      {tab === 'table' && (
        <div className="flex-1 flex flex-col p-4">
          <DataTable />
        </div>
      )}
    </div>
  )
}
