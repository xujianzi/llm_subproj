import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useMapStore } from '../../store/useMapStore'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN as string

const COLOR_STOPS: [number, string][] = [
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
  const popupRef = useRef<mapboxgl.Popup | null>(null)
  const { geojsonData, stats, selectedVariable } = useMapStore()

  // Initialize map once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return
    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-96, 38],
      zoom: 3.5,
    })
    map.addControl(new mapboxgl.NavigationControl(), 'top-right')
    popupRef.current = new mapboxgl.Popup({ closeButton: false, closeOnClick: false })
    mapRef.current = map
  }, [])

  // Update choropleth when data changes
  useEffect(() => {
    const map = mapRef.current
    if (!map || !geojsonData) return

    const onLoad = () => {
      // Remove existing layers and source
      if (map.getLayer('choropleth-fill')) map.removeLayer('choropleth-fill')
      if (map.getLayer('choropleth-line')) map.removeLayer('choropleth-line')
      if (map.getSource('acs-data')) map.removeSource('acs-data')

      // Remove old tooltip handlers before re-registering
      map.off('mousemove', 'choropleth-fill', onMouseMove)
      map.off('mouseleave', 'choropleth-fill', onMouseLeave)

      map.addSource('acs-data', { type: 'geojson', data: geojsonData })

      const minVal = stats?.min ?? 0
      const maxVal = stats?.max ?? 1
      const range = maxVal - minVal || 1

      const stops: (number | string)[] = []
      COLOR_STOPS.forEach(([ratio, color]) => {
        stops.push(minVal + ratio * range, color)
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
          ] as mapboxgl.Expression,
          'fill-opacity': 0.75,
        },
      })

      map.addLayer({
        id: 'choropleth-line',
        type: 'line',
        source: 'acs-data',
        paint: { 'line-color': '#2a2d3e', 'line-width': 0.5 },
      })

      map.on('mousemove', 'choropleth-fill', onMouseMove)
      map.on('mouseleave', 'choropleth-fill', onMouseLeave)
    }

    function onMouseMove(e: mapboxgl.MapMouseEvent & { features?: mapboxgl.MapboxGeoJSONFeature[] }) {
      if (!popupRef.current) return
      map.getCanvas().style.cursor = 'pointer'
      const props = e.features?.[0]?.properties ?? {}
      const name = props.NAME ?? props.STUSPS ?? props.ZCTA5CE20 ?? ''
      const val = props[selectedVariable]
      popupRef.current
        .setLngLat(e.lngLat)
        .setHTML(`<div class="text-sm"><b>${name}</b><br/>${selectedVariable}: ${val ?? 'N/A'}</div>`)
        .addTo(map)
    }

    function onMouseLeave() {
      map.getCanvas().style.cursor = ''
      popupRef.current?.remove()
    }

    if (map.isStyleLoaded()) onLoad()
    else map.once('load', onLoad)
  }, [geojsonData, stats, selectedVariable])

  return <div ref={mapContainer} className="absolute inset-0" />
}
