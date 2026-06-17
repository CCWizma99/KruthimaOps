import re

path = 'c:/KruthimaOps/production/app/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Modify loadSubdivisions to load the GeoJSON Data Source
load_sub_patch = """
    // Add GeoJSON boundary lines for context
    const dataSource = await Cesium.GeoJsonDataSource.load(`/static/subdivisions/${districtName}.geojson`, {
      stroke: Cesium.Color.WHITE.withAlpha(0.7),
      fill: Cesium.Color.TRANSPARENT,
      strokeWidth: 3,
      clampToGround: true
    });
    // Ensure all polygons in the dataset are fully transparent with only the thick stroke visible
    dataSource.entities.values.forEach(entity => {
      if (entity.polygon) {
        entity.polygon.material = Cesium.Color.TRANSPARENT;
        entity.polygon.outline = true;
        entity.polygon.outlineColor = Cesium.Color.WHITE.withAlpha(0.7);
        entity.polygon.outlineWidth = 3;
      }
    });
    
    state.viewer.dataSources.add(dataSource);
    state.subdivisionDataSource = dataSource;

    // Auto-Zoom into the district
"""

content = content.replace("    // Auto-Zoom into the district", load_sub_patch)

# 2. Modify clearSubdivisions to remove the Data Source
clear_sub_patch = """function clearSubdivisions() {
  state.subdivisionEntities.forEach(ent => state.viewer.entities.remove(ent));
  state.subdivisionEntities = [];
  if (state.subdivisionDataSource) {
    state.viewer.dataSources.remove(state.subdivisionDataSource);
    state.subdivisionDataSource = null;
  }
}"""

content = re.sub(r'function clearSubdivisions\(\) \{.*?\n\}', clear_sub_patch, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Boundary injection complete")
