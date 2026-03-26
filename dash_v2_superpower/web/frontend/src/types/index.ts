export type Level = 'state' | 'county' | 'zipcode'

export interface Stats {
  variable: string
  min: number
  max: number
  mean: number
  median: number
}

export interface MapDataResponse {
  geojson: GeoJSON.FeatureCollection
  stats: Stats | null
}

export interface ChatDataPayload {
  rows: Record<string, unknown>[]
  columns: string[]
  geojson: GeoJSON.FeatureCollection
  stats: Stats | null
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}
