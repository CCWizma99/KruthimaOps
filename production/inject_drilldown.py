import re

path = 'c:/KruthimaOps/production/app/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add a state variable for subdivisions
if 'subdivisionEntities: [],' not in content:
    content = content.replace('riskEntities: {},', 'riskEntities: {},\n  subdivisionEntities: [],\n  nationalViewActive: true,')

drilldown_js = """

// ════════════════════════════════════════════════ DRILL DOWN ══

async function loadSubdivisions(districtName) {
  // 1. Fade out national view
  state.nationalViewActive = false;
  for (let key in state.riskEntities) {
    state.riskEntities[key].forEach(ent => {
      if (ent.polygon && ent.polygon.material) {
        ent.polygon.material.color = ent.polygon.material.color.getValue().withAlpha(0.05);
      }
      if (ent.label) ent.label.show = false;
    });
  }

  // Show "Back" button
  let backBtn = document.getElementById('btn-back-national');
  if (!backBtn) {
    backBtn = document.createElement('button');
    backBtn.id = 'btn-back-national';
    backBtn.innerHTML = '⬅ Back to National View';
    backBtn.style = 'position:absolute; top:20px; left:20px; z-index:999; padding:10px 16px; background:#1e293b; color:#fff; border:1px solid #334155; border-radius:8px; cursor:pointer; font-weight:bold; box-shadow:0 4px 12px rgba(0,0,0,0.5);';
    backBtn.onclick = restoreNationalView;
    document.body.appendChild(backBtn);
  }
  backBtn.style.display = 'block';

  // 2. Clear old subdivisions
  clearSubdivisions();

  // 3. Fetch GeoJSON
  const safeName = districtName.replace(/ /g, '_');
  let geojson;
  try {
    const res = await fetch(`/static/subdivisions/${safeName}.geojson`);
    geojson = await res.json();
  } catch (e) {
    console.error("No subdivisions found for", districtName);
    return;
  }

  // 4. Extract places and prepare batch prediction
  const dData = state.districtRiskData[districtName] || {};
  const rows = [];
  const validFeatures = geojson.features.filter(f => f.geometry && f.geometry.coordinates);
  
  validFeatures.forEach(f => {
    const placeName = f.properties.shapeName;
    rows.push({
      district: districtName,
      place_name: placeName,
      rainfall_7d_mm: dData.rainfall_7d_mm || 0,
      flood_occurrence_current_event: state.flood_occurrence,
      inundation_area_sqm: 0,
      is_good_to_live: state.is_good_to_live
    });
  });

  // 5. Fetch predictions
  let predictions = {};
  try {
    const pRes = await fetch('/api/predict/batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ rows })
    });
    const pData = await pRes.json();
    pData.results.forEach((res, idx) => {
      if (!res.error) {
        predictions[rows[idx].place_name] = res.risk_score;
      }
    });
  } catch (e) {
    console.error("Batch predict failed", e);
  }

  // 6. Draw polygons
  validFeatures.forEach(f => {
    const placeName = f.properties.shapeName;
    const score = predictions[placeName] || 0.1;
    const color = getInterpolatedColor(score, 0.7);
    const height = 5000 + (score * 40000); // Exaggerate 3D height based on risk

    const coords = f.geometry.coordinates;
    let polys = f.geometry.type === 'MultiPolygon' ? coords : [coords];
    
    polys.forEach(polyCoords => {
      const flatCoords = polyCoords[0].flat();
      const ent = state.viewer.entities.add({
        name: placeName,
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(flatCoords),
          extrudedHeight: height,
          material: color,
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#ffffff').withAlpha(0.3)
        }
      });
      state.subdivisionEntities.push(ent);
    });
  });
}

function clearSubdivisions() {
  state.subdivisionEntities.forEach(ent => state.viewer.entities.remove(ent));
  state.subdivisionEntities = [];
}

function restoreNationalView() {
  state.nationalViewActive = true;
  document.getElementById('btn-back-national').style.display = 'none';
  clearSubdivisions();
  
  for (let key in state.riskEntities) {
    state.riskEntities[key].forEach(ent => {
      if (ent.polygon && ent.polygon.material) {
        // Restore original color
        const data = state.districtRiskData[key];
        if (data) {
           ent.polygon.material = getInterpolatedColor(data.risk_score, 0.6);
        }
      }
      if (ent.label) ent.label.show = true;
    });
  }
}

"""

if 'loadSubdivisions(' not in content:
    content += "\n" + drilldown_js

# Hook into selectDistrict
if 'loadSubdivisions(name);' not in content:
    content = content.replace('function selectDistrict(name) {', 'function selectDistrict(name) {\n  if (state.nationalViewActive) loadSubdivisions(name);\n')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Drill-down injected.")
