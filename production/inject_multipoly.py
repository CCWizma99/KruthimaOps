import re

path = 'c:/KruthimaOps/production/app/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the previous polygon logic
poly_patch = """
    // Add GeoJSON boundary lines for context
    const dataSource = await Cesium.GeoJsonDataSource.load(`/static/subdivisions/${districtName}.geojson`);
    
    // Explicitly generate true Polylines from the polygon geometry
    // This perfectly bypasses the Windows WebGL polygon outline bug!
    state.subdivisionPolylines = [];
    
    dataSource.entities.values.forEach(entity => {
      if (entity.polygon) {
        const hierarchy = entity.polygon.hierarchy?.getValue(Cesium.JulianDate.now());
        if (hierarchy) {
            function extractRings(hier) {
                const rings = [];
                if (hier.positions && hier.positions.length > 0) {
                    rings.push(hier.positions);
                }
                if (hier.holes) {
                    hier.holes.forEach(hole => {
                        rings.push(...extractRings(hole));
                    });
                }
                return rings;
            }
            
            const allRings = extractRings(hierarchy);
            allRings.forEach(ring => {
                if (ring.length > 0) {
                    const closedRing = ring.concat([ring[0]]);
                    const lineEnt = state.viewer.entities.add({
                        polyline: {
                            positions: closedRing,
                            width: 3,
                            material: Cesium.Color.WHITE.withAlpha(0.6),
                            clampToGround: true
                        }
                    });
                    state.subdivisionPolylines.push(lineEnt);
                }
            });
            
            // Hide the buggy polygon completely
            entity.polygon.show = false;
        }
      }
    });
    
    state.viewer.dataSources.add(dataSource);
    state.subdivisionDataSource = dataSource;
"""

content = re.sub(
    r'    // Add GeoJSON boundary lines for context.*?state\.subdivisionDataSource = dataSource;',
    poly_patch,
    content,
    flags=re.DOTALL
)

# Update clearSubdivisions
clear_patch = """function clearSubdivisions() {
  state.subdivisionEntities.forEach(ent => state.viewer.entities.remove(ent));
  state.subdivisionEntities = [];
  if (state.subdivisionDataSource) {
    state.viewer.dataSources.remove(state.subdivisionDataSource);
    state.subdivisionDataSource = null;
  }
  if (state.subdivisionPolylines) {
    state.subdivisionPolylines.forEach(ent => state.viewer.entities.remove(ent));
    state.subdivisionPolylines = [];
  }
}"""

content = re.sub(r'function clearSubdivisions\(\) \{.*?\n\}', clear_patch, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("MultiPolygon polyline extraction complete")
