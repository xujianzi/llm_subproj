import { useEffect } from 'react'
import { useMapStore } from '../../store/useMapStore'
import { fetchVariables, fetchRegions, fetchMapData } from '../../api/mapApi'
import type { Level } from '../../types'

const YEARS = [2019, 2020, 2021, 2022, 2023]
const LEVELS: { value: Level; label: string }[] = [
  { value: 'state',   label: 'State' },
  { value: 'county',  label: 'County' },
  { value: 'zipcode', label: 'Zipcode' },
]

const SELECT_CLS = 'bg-panel border border-border text-text rounded px-2 py-1 text-sm focus:outline-none focus:border-primary'
const BTN_CLS = 'w-full mt-3 py-2 rounded-lg font-semibold text-white text-sm ' +
  'bg-gradient-to-r from-primary to-accent hover:opacity-90 transition'

export function ConfigPanel() {
  const {
    level, setLevel,
    selectedState, setSelectedState,
    selectedCounty, setSelectedCounty,
    selectedVariable, setSelectedVariable,
    selectedYear, setSelectedYear,
    availableVariables, setAvailableVariables,
    availableCounties, setAvailableCounties,
    setMapData,
  } = useMapStore()

  // Load variable list once
  useEffect(() => {
    fetchVariables().then(setAvailableVariables)
  }, [])

  // Load counties when state changes
  useEffect(() => {
    if (selectedState) {
      fetchRegions('county', selectedState).then(setAvailableCounties)
    } else {
      setAvailableCounties([])
      setSelectedCounty(null)
    }
  }, [selectedState])

  const US_STATES = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC',
  ]

  async function handleUpdate() {
    try {
      const result = await fetchMapData({
        level,
        variable: selectedVariable,
        year: selectedYear,
        state: selectedState ?? undefined,
        county: selectedCounty ?? undefined,
      })
      setMapData(result.geojson, result.stats)
    } catch (e) {
      alert(`Error: ${e}`)
    }
  }

  return (
    <div className="absolute top-0 left-0 right-0 z-10 m-3">
      <div className="bg-panel/90 backdrop-blur border border-border rounded-xl p-4 shadow-xl">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {/* Year */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">Year</label>
            <select className={SELECT_CLS} value={selectedYear}
              onChange={e => setSelectedYear(Number(e.target.value))}>
              {YEARS.map(y => <option key={y}>{y}</option>)}
            </select>
          </div>
          {/* Level */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">Level</label>
            <select className={SELECT_CLS} value={level}
              onChange={e => setLevel(e.target.value as Level)}>
              {LEVELS.map(l => <option key={l.value} value={l.value}>{l.label}</option>)}
            </select>
          </div>
          {/* Variable */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">Variable</label>
            <select className={SELECT_CLS} value={selectedVariable}
              onChange={e => setSelectedVariable(e.target.value)}>
              {availableVariables.map(v => <option key={v}>{v}</option>)}
            </select>
          </div>
          {/* State */}
          <div className="flex flex-col gap-1">
            <label className="text-muted text-xs">State</label>
            <select className={SELECT_CLS} value={selectedState ?? ''}
              onChange={e => setSelectedState(e.target.value || null)}>
              <option value="">All States</option>
              {US_STATES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
          {/* County (only when level=county or zipcode) */}
          {level !== 'state' && (
            <div className="flex flex-col gap-1">
              <label className="text-muted text-xs">County</label>
              <select className={SELECT_CLS} value={selectedCounty ?? ''}
                onChange={e => setSelectedCounty(e.target.value || null)}>
                <option value="">All Counties</option>
                {availableCounties.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
          )}
        </div>
        <button className={BTN_CLS} onClick={handleUpdate}>
          Update Map &amp; Stats
        </button>
      </div>
    </div>
  )
}
