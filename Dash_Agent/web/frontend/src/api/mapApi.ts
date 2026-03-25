import type { Level, MapDataResponse } from '../types'

export async function fetchVariables(): Promise<string[]> {
  const r = await fetch('/api/map/variables')
  if (!r.ok) throw new Error(`Failed to fetch variables: ${r.status}`)
  const data = await r.json()
  return data.columns as string[]
}

export async function fetchRegions(level: string, state: string): Promise<string[]> {
  const r = await fetch(`/api/map/regions?level=${level}&state=${encodeURIComponent(state)}`)
  if (!r.ok) throw new Error(`Failed to fetch regions: ${r.status}`)
  const data = await r.json()
  return data.regions as string[]
}

export async function fetchMapData(params: {
  level: Level
  variable: string
  year: number
  state?: string
  county?: string
}): Promise<MapDataResponse> {
  const q = new URLSearchParams({
    level: params.level,
    variables: params.variable,
    year: String(params.year),
    ...(params.state  ? { state:  params.state  } : {}),
    ...(params.county ? { county: params.county } : {}),
  })
  const r = await fetch(`/api/map/data?${q}`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}
