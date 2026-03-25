import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useMapStore } from '../../store/useMapStore'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN as string

// Viridis-inspired color stops (low → high)
const COLOR_STOPS = [
  [0,   '#1a237e'],
  [0.2, '#1565c0'],
  [0.4, '#00838f'],
  [0.6, '#2e7d32'],
  [0.8, '#f9a825'],
  [1.0, '#e65100'],
]

export function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const { geojsonData, stats, selectedVariable } = useMapStore()

  // Initialize map once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return
    mapRef.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-96, 38],
      zoom: 3.5,
    })
    mapRef.current.addControl(new mapboxgl.NavigationControl(), 'top-right')
  }, [])

  // Update choropleth when data changes
  useEffect(() => {
    const map = mapRef.current
    if (!map || !geojsonData) return

    const onLoad = () => {
      // Remove existing layers
      if (map.getLayer('choropleth-fill')) map.removeLayer('choropleth-fill')
      if (map.getLayer('choropleth-line')) map.removeLayer('choropleth-line')
      if (map.getSource('acs-data')) map.removeSource('acs-data')

      map.addSource('acs-data', { type: 'geojson', data: geojsonData })

      const minVal = stats?.min ?? 0
      const maxVal = stats?.max ?? 1
      const range = maxVal - minVal || 1

      // Build Mapbox interpolation expression
      const stops: (number | string)[] = []
      COLOR_STOPS.forEach(([ratio, color]) => {
        stops.push(minVal + (ratio as number) * range, color as string)
      })

      map.addLayer({
        id: 'choropleth-fill',
        type: 'fill',
        source: 'acs-data',
        paint: {
          'fill-color': [
            'interpolate', ['linear'],
            ['coalesce', ['get', selectedVariable], minVal],
            ...stops,
          ],
          'fill-opacity': 0.75,
        },
      })

      map.addLayer({
        id: 'choropleth-line',
        type: 'line',
        source: 'acs-data',
        paint: { 'line-color': '#2a2d3e', 'line-width': 0.5 },
      })

      // Tooltip on hover
      const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false })
      map.on('mousemove', 'choropleth-fill', (e) => {
        map.getCanvas().style.cursor = 'pointer'
        const props = e.features?.[0]?.properties ?? {}
        const name = props.NAME ?? props.STUSPS ?? props.ZCTA5CE20 ?? ''
        const val = props[selectedVariable]
        popup.setLngLat(e.lngLat)
          .setHTML(`<div class="text-sm"><b>${name}</b><br/>${selectedVariable}: ${val ?? 'N/A'}</div>`)
          .addTo(map)
      })
      map.on('mouseleave', 'choropleth-fill', () => {
        map.getCanvas().style.cursor = ''
        popup.remove()
      })
    }

    if (map.isStyleLoaded()) onLoad()
    else map.once('load', onLoad)
  }, [geojsonData, stats, selectedVariable])

  return <div ref={mapContainer} className="absolute inset-0" />
}
