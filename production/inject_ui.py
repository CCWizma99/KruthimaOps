import re

path = 'c:/KruthimaOps/production/app/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

new_logic = """async function loadSubdivisions(districtName) {
  state.nationalViewActive = false;
  
  for (let key in state.riskEntities) {
    state.riskEntities[key].forEach(ent => {
      if (key === districtName) {
         if (ent.polygon) {
            ent.polygon.outline = true;
            ent.polygon.outlineColor = Cesium.Color.WHITE;
            ent.polygon.outlineWidth = 3;
            // keep material same but maybe slight fade
            ent.polygon.material = ent.polygon.material.color.getValue().withAlpha(0.3);
         }
      } else {
        if (ent.polygon && ent.polygon.material) {
          ent.polygon.material.color = ent.polygon.material.color.getValue().withAlpha(0.05);
        }
      }
      if (ent.label) ent.label.show = false;
    });
  }

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

  clearSubdivisions();

  try {
    const res = await fetch(`/api/predict/subdivisions/${districtName}`);
    const results = await res.json();
    
    results.forEach(sub => {
       const scorePct = Math.round(sub.risk_score * 100);
       const color = getRiskColor(sub.risk_score);
       
       const ent = state.viewer.entities.add({
         position: Cesium.Cartesian3.fromDegrees(sub.lon, sub.lat, 1000), // slight elevation so it's above terrain
         label: {
           text: `${sub.place_name}\\n${scorePct}% Risk | ${sub.rainfall_7d_mm}mm Rain`,
           font: 'bold 13px sans-serif',
           fillColor: Cesium.Color.WHITE,
           style: Cesium.LabelStyle.FILL,
           pixelOffset: new Cesium.Cartesian2(0, -25),
           backgroundColor: color.withAlpha(0.85),
           showBackground: true,
           backgroundPadding: new Cesium.Cartesian2(8, 6),
           disableDepthTestDistance: Number.POSITIVE_INFINITY
         },
         point: {
           pixelSize: 12,
           color: color,
           outlineColor: Cesium.Color.WHITE,
           outlineWidth: 2,
           disableDepthTestDistance: Number.POSITIVE_INFINITY
         }
       });
       state.subdivisionEntities.push(ent);
    });
  } catch(e) {
    console.error("Subdivisions fetch failed", e);
  }
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
        const data = state.districtRiskData[key];
        if (data) {
           ent.polygon.material = getInterpolatedColor(data.risk_score, 0.6);
        }
        ent.polygon.outline = false;
      }
      if (ent.label) ent.label.show = true;
    });
  }
}"""

content = re.sub(r'async function loadSubdivisions\(districtName\).*?function restoreNationalView\(\) \{.*?\}\n\}', new_logic, content, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("UI updated")
