import { useMapStore } from '../../store/useMapStore'
import { AgGridReact } from 'ag-grid-react'
import { AllCommunityModule, ModuleRegistry } from 'ag-grid-community'

ModuleRegistry.registerModules([AllCommunityModule])

export function DataTable() {
  const geojsonData = useMapStore(s => s.geojsonData)

  if (!geojsonData) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted text-sm">
        先在 Map Visualization 标签中查询数据
      </div>
    )
  }

  // Extract row data from GeoJSON features
  const rows = geojsonData.features.map(f => f.properties ?? {})
  const allKeys = rows.length > 0 ? Object.keys(rows[0]) : []
  const colDefs = allKeys.map(k => ({
    field: k,
    sortable: true,
    filter: true,
    resizable: true,
  }))

  return (
    <div className="flex-1 ag-theme-balham-dark" style={{ height: '100%' }}>
      <AgGridReact rowData={rows} columnDefs={colDefs} pagination paginationPageSize={50} />
    </div>
  )
}
