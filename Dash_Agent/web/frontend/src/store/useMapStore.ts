import { create } from 'zustand'
import type { Level, Stats, ChatMessage, ChatDataPayload } from '../types'
import type { FeatureCollection } from 'geojson'

interface MapStore {
  // Map config
  level:              Level
  selectedState:      string | null
  selectedCounty:     string | null
  selectedVariable:   string
  selectedYear:       number
  availableVariables: string[]
  availableCounties:  string[]

  // Map data
  geojsonData: FeatureCollection | null
  stats:       Stats | null

  // Chat
  chatHistory: ChatMessage[]
  chatOpen:    boolean

  // Actions
  setLevel:               (l: Level) => void
  setSelectedState:       (s: string | null) => void
  setSelectedCounty:      (c: string | null) => void
  setSelectedVariable:    (v: string) => void
  setSelectedYear:        (y: number) => void
  setAvailableVariables:  (cols: string[]) => void
  setAvailableCounties:   (counties: string[]) => void
  setMapData:             (gj: FeatureCollection, stats: Stats | null) => void
  updateFromChatData:     (payload: ChatDataPayload) => void
  addChatMessage:         (msg: ChatMessage) => void
  setChatOpen:            (open: boolean) => void
}

export const useMapStore = create<MapStore>((set) => ({
  level:              'state',
  selectedState:      null,
  selectedCounty:     null,
  selectedVariable:   'median_income',
  selectedYear:       2020,
  availableVariables: [],
  availableCounties:  [],
  geojsonData:        null,
  stats:              null,
  chatHistory:        [],
  chatOpen:           false,

  setLevel:              (level)              => set({ level }),
  setSelectedState:      (selectedState)      => set({ selectedState }),
  setSelectedCounty:     (selectedCounty)     => set({ selectedCounty }),
  setSelectedVariable:   (selectedVariable)   => set({ selectedVariable }),
  setSelectedYear:       (selectedYear)       => set({ selectedYear }),
  setAvailableVariables: (availableVariables) => set({ availableVariables }),
  setAvailableCounties:  (availableCounties)  => set({ availableCounties }),
  setMapData:            (geojsonData, stats) => set({ geojsonData, stats }),
  updateFromChatData:    ({ geojson, stats }) => set({ geojsonData: geojson, stats }),
  addChatMessage:        (msg)                => set((s) => ({ chatHistory: [...s.chatHistory, msg] })),
  setChatOpen:           (chatOpen)           => set({ chatOpen }),
}))
