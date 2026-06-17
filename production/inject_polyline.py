import re

path = 'c:/KruthimaOps/production/app/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

poly_patch = """
    // Add GeoJSON boundary lines for context
    const dataSource = await Cesium.GeoJsonDataSource.load(`/static/subdivisions/${districtName}.geojson`);
    
    // Explicitly generate true Polylines from the polygon geometry
    // This perfectly bypasses the Windows WebGL polygon outline bug!
    dataSource.entities.values.forEach(entity => {
      if (entity.polygon) {
        // Hide the buggy polygon completely
        entity.polygon.show = false;
        
        const hierarchy = entity.polygon.hierarchy.getValue(Cesium.JulianDate.now());
        if (hierarchy && hierarchy.positions) {
            // Close the loop
            const pos = hierarchy.positions;
            const closedPos = pos.concat([pos[0]]);
            
            entity.polyline = new Cesium.PolylineGraphics({
                positions: closedPos,
                width: 3,
                material: Cesium.Color.WHITE.withAlpha(0.5),
                clampToGround: true
            });
        }
      }
    });
    
    state.viewer.dataSources.add(dataSource);
    state.subdivisionDataSource = dataSource;
"""

# The previous block was:
#    // Add GeoJSON boundary lines for context
#    const dataSource = await Cesium.GeoJsonDataSource.load(`/static/subdivisions/${districtName}.geojson`, {
# ...
#    state.subdivisionDataSource = dataSource;

content = re.sub(
    r'    // Add GeoJSON boundary lines for context.*?state\.subdivisionDataSource = dataSource;',
    poly_patch,
    content,
    flags=re.DOTALL
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Polyline injection complete")
