/**
 * FloodGuard SL — Dashboard Application v2
 * 3D CesiumJS terrain + district detail bottom modal + what-if simulator modal
 * Background precompute polling → progressive district risk prism rendering
 */

'use strict';

// ═══════════════════════════════════════════════════════════ CONFIG ══
const API_BASE = '';   // same origin

// ══════════════════════════════════════════════════════════ STATE ══
const state = {
  viewer:              null,
  districts:           {},          // name → reference data
  riskEntities:        {},          // name → [entity, entity, …] for cleanup
  selectedDistrict:    null,
  lastPredictionId:    null,
  logRows:             [],
  flood_occurrence:    'No',
  is_good_to_live:     'Yes',
  currentForecast:     [],
  activeForecastIndex: 0,
  districtRiskData:    {},          // name → {risk_score, risk_level, rainfall_7d_mm}
  districtForecasts:   {},          // name → list of 7 days forecast
  meshPrimitive:       null,
  wireframePrimitive:  null,
  precomputePollTimer: null,
  clickHandler:        null,
  historicalMode:      false,       // true when viewing a past date
  historicalDate:      null,        // ISO date string when in historical mode
  savedLiveForecasts:  null,        // snapshot of live districtForecasts
  savedLiveRiskData:   null,        // snapshot of live districtRiskData
  evacuationEntities:  [],          // List of Cesium Entities for safe zones
  evacuationData:      [],          // JSON data from evacuation_points.json
  showEvacuationPoints: false,      // UI toggle state
  // Geolocation
  myLocationActive:    false,
  myLocationEntity:    null,        // Cesium entity for user's location pin
  myLocationRingEntity: null,       // Cesium entity for pulsing accuracy ring
  geoWatchId:          null,        // navigator.geolocation watchPosition ID
};

// ══════════════════════════════════════════════════════════ INIT ══
async function init() {
  // Cesium token
  if (!window.__CESIUM_TOKEN__) {
    try {
      const r = await fetch('/api/config/cesium-token');
      if (r.ok) { const d = await r.json(); Cesium.Ion.defaultAccessToken = d.token || ''; }
    } catch (_) {}
  } else {
    Cesium.Ion.defaultAccessToken = window.__CESIUM_TOKEN__;
  }

  initCesium();
  await loadDistricts();
  await loadModelCard();
  initSliders();
  initToggles();
  initHistoricalDatePicker();
  await loadActivityLog();
  await loadEvacuationPoints();
  startPrecomputePolling();
}

function initHistoricalDatePicker() {
  const dateInput = document.getElementById('historical-date');
  if (!dateInput) return;
  // Set max to yesterday
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  dateInput.max = yesterday.toISOString().split('T')[0];
  // Set default to 7 days ago
  const weekAgo = new Date();
  weekAgo.setDate(weekAgo.getDate() - 7);
  dateInput.value = weekAgo.toISOString().split('T')[0];
  // Set min to 2020-01-01 (Open-Meteo archive limit)
  dateInput.min = '2020-01-01';
}

// ════════════════════════════════════════════════ CESIUM 3D GLOBE ══
function initCesium() {
  try {
    const opts = {
      baseLayerPicker:      false,
      navigationHelpButton: false,
      sceneModePicker:      false,
      homeButton:           false,
      geocoder:             false,
      fullscreenButton:     false,
      timeline:             false,
      animation:            false,
      infoBox:              false,
      selectionIndicator:   false,
      skyBox:               false,
      skyAtmosphere:        new Cesium.SkyAtmosphere(),
      imageryProvider:      new Cesium.UrlTemplateImageryProvider({
        url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        credit: 'Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
      }),
    };

    if (Cesium.Ion.defaultAccessToken) {
      opts.terrain = Cesium.Terrain.fromWorldTerrain({ requestWaterMask: true });
    }

    state.viewer = new Cesium.Viewer('cesium-container', opts);

    // Show the globe for CartoDB map, but hide sun/moon/atmosphere for clean UI
    state.viewer.scene.globe.show = true;
    if (state.viewer.scene.sun) state.viewer.scene.sun.show = false;
    if (state.viewer.scene.moon) state.viewer.scene.moon.show = false;
    if (state.viewer.scene.skyAtmosphere) state.viewer.scene.skyAtmosphere.show = false;

    // Dark base background color
    state.viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#050d1a');

    // Oblique starting view: south of island, looking north at 65° tilt (North UP)
    state.viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(80.7, 4.5, 880000),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-65),
        roll:    0,
      },
      duration: 3.5,
    });

    // Map click → select district by entity name
    state.clickHandler = new Cesium.ScreenSpaceEventHandler(state.viewer.scene.canvas);
    state.clickHandler.setInputAction(function (click) {
      const picked = state.viewer.scene.pick(click.position);
      if (Cesium.defined(picked) && picked.id) {
        const rawName = picked.id.name || '';
        // Strip suffixes like _glow, _label added internally
        const name = rawName.replace(/_(glow|label)$/, '');
        if (name && state.districts[name]) {
          selectDistrict(name);
          // Hide the click hint after first interaction
          const hint = document.getElementById('map-click-hint');
          if (hint) hint.style.opacity = '0';
        }
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    console.log('[CesiumJS] Viewer ready. Terrain exaggeration: 6×');
  } catch (err) {
    console.error('[CesiumJS] Init failed:', err);
    document.getElementById('cesium-container').innerHTML =
      `<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;
        color:#94a3b8;font-size:14px;background:#050d1a;padding:20px;text-align:center">
        <div style="font-weight:bold;margin-bottom:8px;color:#ef4444">3D Map Initialization Failed</div>
        <div style="font-size:12px;color:#cbd5e1;font-family:monospace">${err.message || err}</div></div>`;
  }
}

// Risk level → Cesium color
function getRiskColor(score) {
  if (score < 0.25) return Cesium.Color.fromCssColorString('#22c55e');
  if (score < 0.50) return Cesium.Color.fromCssColorString('#eab308');
  if (score < 0.75) return Cesium.Color.fromCssColorString('#f97316');
  return Cesium.Color.fromCssColorString('#ef4444');
}

/** Get interpolated color for dynamic 3D surface */
function getInterpolatedColor(score, alpha) {
  let r = 0, g = 0, b = 0;
  if (score < 0.25) {
    const t = score / 0.25;
    r = Math.round(34 + t * (234 - 34));
    g = Math.round(197 + t * (179 - 197));
    b = Math.round(94 + t * (8 - 94));
  } else if (score < 0.50) {
    const t = (score - 0.25) / 0.25;
    r = Math.round(234 + t * (249 - 234));
    g = Math.round(179 + t * (115 - 179));
    b = Math.round(8 + t * (22 - 8));
  } else if (score < 0.75) {
    const t = (score - 0.50) / 0.25;
    r = Math.round(249 + t * (239 - 249));
    g = Math.round(115 + t * (68 - 115));
    b = Math.round(22 + t * (68 - 22));
  } else {
    r = 239;
    g = 68;
    b = 68;
  }
  return {
    red: r,
    green: g,
    blue: b,
    alpha: Math.round(alpha * 255)
  };
}

/** Plot dynamic 3D risk surface using Inverse Distance Weighting (IDW) */
function update3DRiskSurface(dayIdx = 0) {
  if (!state.viewer) return;

  // 1. Clean up old primitives
  if (state.meshPrimitive) {
    state.viewer.scene.primitives.remove(state.meshPrimitive);
    state.meshPrimitive = null;
  }
  if (state.wireframePrimitive) {
    state.viewer.scene.primitives.remove(state.wireframePrimitive);
    state.wireframePrimitive = null;
  }

  const calculatedDistricts = Object.keys(state.districts).filter(name => state.districtForecasts[name]);
  if (calculatedDistricts.length === 0) return;

  // 2. Define grid parameters over Sri Lanka bounding box
  const Nx = 35;
  const Ny = 50;
  const minLon = 79.5;
  const maxLon = 82.2;
  const minLat = 5.8;
  const maxLat = 9.9;

  const positions = [];
  const colors = [];
  const lineColors = [];
  const fades = [];

  for (let j = 0; j < Ny; j++) {
    const lat = minLat + (j / (Ny - 1)) * (maxLat - minLat);
    for (let i = 0; i < Nx; i++) {
      const lon = minLon + (i / (Nx - 1)) * (maxLon - minLon);

      let sumWeight = 0;
      let sumScore = 0;
      let minD = 999;

      for (const name of calculatedDistricts) {
        const ref = state.districts[name];
        const forecastList = state.districtForecasts[name];
        if (!ref || !forecastList) continue;

        const dayData = forecastList[dayIdx] || forecastList[0];
        const score = dayData ? dayData.risk_score : 0;

        const dLon = lon - ref.center_lon;
        const dLat = lat - ref.center_lat;
        const d = Math.sqrt(dLon * dLon + dLat * dLat);

        if (d < minD) minD = d;

        const weight = 1.0 / (d * d + 0.0001);
        sumWeight += weight;
        sumScore += weight * score;
      }

      const score = sumWeight > 0 ? (sumScore / sumWeight) : 0;

      // Coastline falloff filter: tight boundary conforming to Sri Lanka's geography
      const maxDist = 0.44;
      const minDistLimit = 0.12;
      const fade = Math.max(0, Math.min(1, (maxDist - minD) / (maxDist - minDistLimit)));
      const height = score * fade * 85000; // 85 km max height at center peak

      const c = getInterpolatedColor(score, fade * 0.58);
      const lc = getInterpolatedColor(score, fade * 0.82);

      const pos = Cesium.Cartesian3.fromDegrees(lon, lat, height);
      positions.push(pos.x, pos.y, pos.z);
      colors.push(c.red, c.green, c.blue, c.alpha);
      lineColors.push(lc.red, lc.green, lc.blue, lc.alpha);
      fades.push(fade);
    }
  }

  // 3. Grid triangulation indices (ONLY for land triangles where all 3 vertices have fade > 0)
  const indices = [];
  for (let j = 0; j < Ny - 1; j++) {
    for (let i = 0; i < Nx - 1; i++) {
      const bl = j * Nx + i;
      const br = j * Nx + (i + 1);
      const tl = (j + 1) * Nx + i;
      const tr = (j + 1) * Nx + (i + 1);

      if (fades[bl] > 0 && fades[br] > 0 && fades[tr] > 0) {
        indices.push(bl, br, tr);
      }
      if (fades[bl] > 0 && fades[tr] > 0 && fades[tl] > 0) {
        indices.push(bl, tr, tl);
      }
    }
  }

  // 4. Grid wireframe line indices (ONLY for land lines where both endpoints have fade > 0)
  const lineIndices = [];
  for (let j = 0; j < Ny; j++) {
    for (let i = 0; i < Nx; i++) {
      const idx = j * Nx + i;
      if (i < Nx - 1) {
        const br = j * Nx + (i + 1);
        if (fades[idx] > 0 && fades[br] > 0) {
          lineIndices.push(idx, br);
        }
      }
      if (j < Ny - 1) {
        const tl = (j + 1) * Nx + i;
        if (fades[idx] > 0 && fades[tl] > 0) {
          lineIndices.push(idx, tl);
        }
      }
    }
  }

  // 5. Build Cesium primitives
  const meshGeometry = new Cesium.Geometry({
    attributes: {
      position: new Cesium.GeometryAttribute({
        componentDatatype: Cesium.ComponentDatatype.DOUBLE,
        componentsPerAttribute: 3,
        values: new Float64Array(positions)
      }),
      color: new Cesium.GeometryAttribute({
        componentDatatype: Cesium.ComponentDatatype.UNSIGNED_BYTE,
        componentsPerAttribute: 4,
        normalize: true,
        values: new Uint8Array(colors)
      })
    },
    indices: new Uint16Array(indices),
    primitiveType: Cesium.PrimitiveType.TRIANGLES,
    boundingSphere: Cesium.BoundingSphere.fromVertices(positions)
  });

  state.meshPrimitive = new Cesium.Primitive({
    geometryInstances: new Cesium.GeometryInstance({ geometry: meshGeometry }),
    appearance: new Cesium.PerInstanceColorAppearance({ flat: true, translucent: true }),
    asynchronous: false
  });
  state.viewer.scene.primitives.add(state.meshPrimitive);

  const lineGeometry = new Cesium.Geometry({
    attributes: {
      position: new Cesium.GeometryAttribute({
        componentDatatype: Cesium.ComponentDatatype.DOUBLE,
        componentsPerAttribute: 3,
        values: new Float64Array(positions)
      }),
      color: new Cesium.GeometryAttribute({
        componentDatatype: Cesium.ComponentDatatype.UNSIGNED_BYTE,
        componentsPerAttribute: 4,
        normalize: true,
        values: new Uint8Array(lineColors)
      })
    },
    indices: new Uint16Array(lineIndices),
    primitiveType: Cesium.PrimitiveType.LINES,
    boundingSphere: Cesium.BoundingSphere.fromVertices(positions)
  });

  state.wireframePrimitive = new Cesium.Primitive({
    geometryInstances: new Cesium.GeometryInstance({ geometry: lineGeometry }),
    appearance: new Cesium.PerInstanceColorAppearance({ flat: true, translucent: true }),
    asynchronous: false
  });
  state.viewer.scene.primitives.add(state.wireframePrimitive);

  // 6. Update all computed district pin flags to float exactly above the 3D surface
  for (const name of calculatedDistricts) {
    const ref = state.districts[name];
    const forecastList = state.districtForecasts[name];
    if (!ref || !forecastList) continue;

    const dayData = forecastList[dayIdx] || forecastList[0];
    const score = dayData ? dayData.risk_score : 0;

    let sumWeight = 0;
    let sumScore = 0;
    let minD = 999;
    for (const otherName of calculatedDistricts) {
      const otherRef = state.districts[otherName];
      const otherList = state.districtForecasts[otherName];
      if (!otherRef || !otherList) continue;

      const otherDayData = otherList[dayIdx] || otherList[0];
      const otherScore = otherDayData ? otherDayData.risk_score : 0;

      const dLon = ref.center_lon - otherRef.center_lon;
      const dLat = ref.center_lat - otherRef.center_lat;
      const d = Math.sqrt(dLon * dLon + dLat * dLat);

      if (d < minD) minD = d;

      const weight = 1.0 / (d * d + 0.0001);
      sumWeight += weight;
      sumScore += weight * otherScore;
    }

    const localScore = sumWeight > 0 ? (sumScore / sumWeight) : score;
    const fade = Math.max(0, 1.0 - (minD / 0.65));
    const localHeight = localScore * fade * 85000;

    plotDistrictPin(ref.center_lat, ref.center_lon, name, localHeight, score);
  }
}

/** Plot an active glowing pinpoint flag on the 3D surface */
function plotDistrictPin(lat, lon, name, surfaceHeight, score) {
  if (!state.viewer) return;
  clearDistrictEntities(name);

  const entities = [];
  const color = getRiskColor(score);
  const isSelected = (name === state.selectedDistrict);

  // 1. Vertical indicator line (pin stem)
  entities.push(state.viewer.entities.add({
    name: name + '_stem',
    polyline: {
      positions: [
        Cesium.Cartesian3.fromDegrees(lon, lat, 0),
        Cesium.Cartesian3.fromDegrees(lon, lat, surfaceHeight)
      ],
      width: isSelected ? 2.5 : 1.2,
      material: isSelected ? Cesium.Color.fromCssColorString('#22d3ee').withAlpha(0.85) : color.withAlpha(0.55),
    }
  }));

  // 2. Floating flag label at the top
  entities.push(state.viewer.entities.add({
    name: name,
    position: Cesium.Cartesian3.fromDegrees(lon, lat, surfaceHeight + 6000),
    label: {
      text:                    `${name}\n${(score * 100).toFixed(0)}%`,
      font:                    isSelected ? 'bold 12px Inter, sans-serif' : 'bold 10px Inter, sans-serif',
      fillColor:               Cesium.Color.WHITE,
      outlineColor:            isSelected ? Cesium.Color.fromCssColorString('#22d3ee') : Cesium.Color.fromCssColorString('#050d1a'),
      outlineWidth:            isSelected ? 5 : 3,
      style:                   Cesium.LabelStyle.FILL_AND_OUTLINE,
      verticalOrigin:          Cesium.VerticalOrigin.BOTTOM,
      pixelOffset:             new Cesium.Cartesian2(0, -4),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      showBackground:          true,
      backgroundColor:         isSelected ? Cesium.Color.fromCssColorString('#0f2d4a').withAlpha(0.9) : Cesium.Color.fromCssColorString('#0a1628').withAlpha(0.75),
      backgroundPadding:       new Cesium.Cartesian2(8, 6),
    },
    point: {
      pixelSize:                isSelected ? 9 : 6,
      color:                    isSelected ? Cesium.Color.fromCssColorString('#22d3ee') : color,
      outlineColor:             Cesium.Color.WHITE,
      outlineWidth:             isSelected ? 2 : 1.5,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  }));

  state.riskEntities[name] = entities;
}

/** Plot a dim location pin for districts not yet computed */
function plotDistrictDimPin(lat, lon, name) {
  if (!state.viewer) return;
  clearDistrictEntities(name);

  const entities = [];
  entities.push(state.viewer.entities.add({
    name: name,
    position: Cesium.Cartesian3.fromDegrees(lon, lat, 2000),
    point: {
      pixelSize:                5,
      color:                    Cesium.Color.fromCssColorString('#475569'),
      outlineColor:             Cesium.Color.fromCssColorString('#1e293b'),
      outlineWidth:             1,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text:                    name,
      font:                    '9px Inter, sans-serif',
      fillColor:               Cesium.Color.fromCssColorString('#94a3b8'),
      outlineColor:            Cesium.Color.fromCssColorString('#050d1a'),
      outlineWidth:            2,
      style:                   Cesium.LabelStyle.FILL_AND_OUTLINE,
      verticalOrigin:          Cesium.VerticalOrigin.BOTTOM,
      pixelOffset:             new Cesium.Cartesian2(0, -6),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    }
  }));

  state.riskEntities[name] = entities;
}

function clearDistrictEntities(name) {
  if (!state.viewer || !state.riskEntities[name]) return;
  state.riskEntities[name].forEach(e => state.viewer.entities.remove(e));
  delete state.riskEntities[name];
}

function highlightDistrict(name) {
  // Pin will auto-highlight in update3DRiskSurface, this is a clean wrapper
  update3DRiskSurface(state.activeForecastIndex);
}

function flyToDistrict(name, lat, lon) {
  if (!state.viewer) return;
  const overlay = document.getElementById('district-label-overlay');
  const label   = document.getElementById('district-flyto-label');
  label.textContent = `📍 ${name}`;
  overlay.style.display = 'block';
  setTimeout(() => { overlay.style.display = 'none'; }, 3000);

  state.viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat, 200000),
    orientation: {
      heading: Cesium.Math.toRadians(0),
      pitch:   Cesium.Math.toRadians(-48),
      roll:    0,
    },
    duration: 2.2,
  });
}

// ════════════════════════════════════════════ DISTRICT LOADING ══
async function loadDistricts() {
  try {
    const resp = await fetch(`${API_BASE}/api/districts`);
    const data = await resp.json();
    const sel  = document.getElementById('district-select');
    sel.innerHTML = '<option value="">Select district...</option>';

    for (const name of data.districts) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = name;
      sel.appendChild(opt);
    }

    // Fetch all district reference data in parallel
    const refs = await Promise.all(
      data.districts.map(name =>
        fetch(`${API_BASE}/api/district/${encodeURIComponent(name)}`)
          .then(r => r.ok ? r.json() : null)
      )
    );
    data.districts.forEach((name, i) => {
      if (refs[i]) state.districts[name] = refs[i];
    });

    // Plot dim location pins for all districts initially
    for (const [name, info] of Object.entries(state.districts)) {
      plotDistrictDimPin(info.center_lat, info.center_lon, name);
    }

  } catch (err) {
    console.warn('[Districts] Failed to load:', err);
  }
}

// ═════════════════════════════════ PRECOMPUTE POLLING ══
function startPrecomputePolling() {
  const banner   = document.getElementById('header-precompute');
  const fill     = document.getElementById('precompute-fill');
  const label    = document.getElementById('precompute-label');
  const chip     = document.getElementById('district-count-chip');
  const readyCnt = document.getElementById('districts-ready-count');

  banner.style.display = 'flex';
  chip.style.display   = 'flex';

  async function poll() {
    try {
      // 1. Get today's computed scores
      const r    = await fetch(`${API_BASE}/api/forecasts/today`);
      const data = await r.json();
      const pct  = data.total > 0 ? (data.ready / data.total) * 100 : 0;

      fill.style.width = `${pct}%`;
      label.textContent = data.complete
        ? `All ${data.total} risk profiles ready`
        : `Computing district profiles… ${data.ready}/${data.total}`;
      readyCnt.textContent = data.ready;

      // Render newly computed districts as part of the 3D surface
      let newlyAdded = false;
      for (const [name, forecastList] of Object.entries(data.districts)) {
        if (!state.districtForecasts[name]) {
          state.districtForecasts[name] = forecastList;
          state.districtRiskData[name]  = forecastList[0];
          newlyAdded = true;
        }
      }
      if (newlyAdded) {
        update3DRiskSurface(state.activeForecastIndex);
      }

      // Enable simulate button once a district is selected
      if (state.selectedDistrict) {
        document.getElementById('simulate-btn').disabled = false;
      }

      if (data.complete || data.ready >= data.total) {
        // Hide progress after a delay
        setTimeout(() => {
          banner.style.display = 'none';
        }, 4000);
        label.textContent = `✓ All ${data.total} district risk profiles loaded`;
        return; // stop polling
      }

    } catch (e) {
      console.warn('[Precompute] Poll failed:', e);
    }
    state.precomputePollTimer = setTimeout(poll, 5000);
  }

  // Start after a short delay (server needs time to boot)
  state.precomputePollTimer = setTimeout(poll, 4000);
}

// ════════════════════════════════════════════ DISTRICT SELECT ══
function selectDistrict(name) {
  state.selectedDistrict = name;

  // Sync dropdown
  const sel = document.getElementById('district-select');
  if (sel.value !== name) sel.value = name;

  const info = state.districts[name];
  if (!info) return;

  // Enable simulate btn
  document.getElementById('simulate-btn').disabled = false;

  // Fly to district
  flyToDistrict(name, info.center_lat, info.center_lon);

  // Highlight its prism
  highlightDistrict(name);

  // Open detail modal and load forecast
  openDistrictModal(name);
}

// ═════════════════════════════════════ DISTRICT DETAIL MODAL ══
function openDistrictModal(name) {
  const modal = document.getElementById('district-modal');
  modal.classList.add('open');

  // Set header info
  document.getElementById('modal-district-name').textContent = name;
  document.getElementById('modal-district-sub').textContent  =
    `Flood risk outlook for ${name} district, Sri Lanka`;

  // Show loading state
  document.getElementById('modal-forecast-body').innerHTML =
    `<div class="modal-forecast-loading">
       <div class="btn-loader" style="width:14px;height:14px;border-color:rgba(34,211,238,0.2);border-top-color:var(--accent-cyan)"></div>
       <span>Fetching 7-day forecast…</span>
     </div>`;
  document.getElementById('modal-briefing-text').textContent =
    'Loading AI briefing…';

  // Reset gauge
  updateModalGauge(null, null, null);

  // Update Safe Zone
  updateNearestSafeZone(name);

  // Load forecast
  loadDistrictForecast(name);
}

function updateNearestSafeZone(districtName) {
  const el = document.getElementById('modal-safezone-text');
  if (!el) return;
  
  if (!state.evacuationData || state.evacuationData.length === 0) {
    el.textContent = "Safe zone data not loaded.";
    return;
  }
  
  const zones = state.evacuationData.filter(z => z.district === districtName);
  if (zones.length > 0) {
    // Just pick the first one matching the district
    const z = zones[0];
    el.innerHTML = `<strong>${z.name}</strong> (${z.type})<br><span style="font-size:9px; color:var(--text-muted)">Capacity: ~${z.capacity} people</span>`;
  } else {
    // Fallback if no exact district match
    el.textContent = "No designated safe zone mapped for this district yet.";
  }
}

function closeDistrictModal() {
  document.getElementById('district-modal').classList.remove('open');
}

async function loadDistrictForecast(name) {
  try {
    const r = await fetch(`${API_BASE}/api/forecast/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error('Forecast API error');
    const data = await r.json();
    state.currentForecast    = data.forecast;
    state.activeForecastIndex = 0;

    renderModalForecastList();
    selectForecastDay(0);

  } catch (err) {
    console.error('[Forecast] Error:', err);
    document.getElementById('modal-forecast-body').innerHTML =
      `<div style="color:var(--risk-extreme);font-size:11px">⚠ Failed to load forecast.</div>`;
  }
}

function renderModalForecastList() {
  const body = document.getElementById('modal-forecast-body');
  body.innerHTML = '';
  const colors = { LOW: '#22c55e', MEDIUM: '#eab308', HIGH: '#f97316', EXTREME: '#ef4444' };

  state.currentForecast.forEach((day, idx) => {
    const dt      = new Date(day.date);
    const dateStr = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const dayName = idx === 0 ? 'Today' : dt.toLocaleDateString(undefined, { weekday: 'short' });
    const pct     = (day.risk_score * 100).toFixed(0);

    const row = document.createElement('div');
    row.className = `modal-forecast-item ${idx === 0 ? 'active' : ''}`;
    row.id        = `mfi-${idx}`;
    row.onclick   = () => selectForecastDay(idx);
    row.innerHTML = `
      <span class="mfi-date">${dateStr}</span>
      <span class="mfi-day">${dayName}</span>
      <span class="mfi-rain">${day.rainfall_7d_mm.toFixed(0)}mm</span>
      <div class="mfi-bar"><div class="mfi-bar-fill" style="width:${pct}%;background:${colors[day.risk_level]}"></div></div>
      <span class="mfi-badge ${day.risk_level}">${day.risk_level}</span>
    `;
    body.appendChild(row);
  });
}

function selectForecastDay(idx) {
  if (!state.currentForecast || !state.currentForecast[idx]) return;
  state.activeForecastIndex = idx;

  document.querySelectorAll('.modal-forecast-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  const day = state.currentForecast[idx];

  // Update modal gauge
  updateModalGauge(day.risk_score, day.risk_level, state.selectedDistrict);

  // Update map surface for this day
  update3DRiskSurface(idx);

  // Briefing (not available for forecast days — show generic)
  document.getElementById('modal-briefing-text').textContent =
    `${state.selectedDistrict} district — ${day.date}: Forecast 7-day rainfall ${day.rainfall_7d_mm.toFixed(0)}mm. ` +
    `Risk score ${day.risk_score.toFixed(4)} (${day.risk_level}). ` +
    (day.cached ? 'Using cached forecast data.' : 'Live from Open-Meteo API.');
}

function updateModalGauge(score, level, district) {
  const arcLen = 251;
  const arc    = document.getElementById('modal-gauge-arc');
  const needle = document.getElementById('modal-gauge-needle');
  const scoreEl= document.getElementById('modal-gauge-score');
  const labelEl= document.getElementById('modal-gauge-label');
  const badge  = document.getElementById('modal-risk-badge');

  const colors = { LOW:'#22c55e', MEDIUM:'#eab308', HIGH:'#f97316', EXTREME:'#ef4444' };

  if (score === null) {
    arc.style.strokeDashoffset = arcLen;
    arc.style.stroke = '#22d3ee';
    needle.setAttribute('cx', 20);
    needle.setAttribute('cy', 100);
    scoreEl.textContent = '—';
    scoreEl.style.color = 'var(--text-primary)';
    labelEl.textContent = district || 'NO DATA';
    badge.textContent   = 'LOADING';
    badge.className     = 'risk-badge modal-risk-badge';
    return;
  }

  arc.style.strokeDashoffset = arcLen - (score * arcLen);
  arc.style.stroke = colors[level] || '#22d3ee';

  const angle = -180 + score * 180;
  const rad   = angle * Math.PI / 180;
  const cx = 100, cy = 100, r = 80;
  needle.setAttribute('cx', cx + r * Math.cos(rad));
  needle.setAttribute('cy', cy + r * Math.sin(rad));

  scoreEl.textContent = score.toFixed(4);
  scoreEl.style.color = colors[level] || '#22d3ee';
  labelEl.textContent = district || '';

  badge.textContent = level;
  badge.className   = `risk-badge modal-risk-badge ${level}`;
}

// ═══════════════════════════════════════ WHAT-IF MODAL ══
function openWhatIfModal() {
  if (!state.selectedDistrict) {
    alert('Please select a district first.');
    return;
  }
  document.getElementById('whatif-district-chip').textContent = state.selectedDistrict;
  document.getElementById('whatif-overlay').classList.add('visible');
  document.getElementById('whatif-modal').classList.add('visible');

  // Reset result panel
  document.getElementById('whatif-result').style.display = 'none';
}

function closeWhatIfModal() {
  document.getElementById('whatif-overlay').classList.remove('visible');
  document.getElementById('whatif-modal').classList.remove('visible');
}

// ════════════════════════════════════════════ MODEL CARD ══
async function loadModelCard() {
  try {
    const r = await fetch(`${API_BASE}/api/models`);
    if (!r.ok) return;
    const m = await r.json();
    document.getElementById('model-version-label').textContent =
      `v703 pipeline | LB ${m.opt_lb_score?.toFixed(5) ?? '—'}`;
    document.getElementById('stat-pipeline').textContent = m.base_pipeline ?? '—';
    document.getElementById('stat-mae').textContent      = m.oof_mae?.toFixed(5) ?? '—';
    document.getElementById('stat-ev').textContent       = m.oof_ev?.toFixed(5) ?? '—';
    document.getElementById('stat-lb').textContent       = m.opt_lb_score?.toFixed(5) ?? '—';
    document.getElementById('stat-feats').textContent    = `${m.n_total_features} cols`;
    const d = new Date(m.training_date);
    document.getElementById('stat-date').textContent = isNaN(d) ? '—' : d.toLocaleDateString();
  } catch (err) {
    console.warn('[ModelCard] Failed:', err);
  }
}

// ════════════════════════════════════════════════════ SLIDERS ══
function initSliders() {
  const rainfall   = document.getElementById('rainfall-slider');
  const inundation = document.getElementById('inundation-slider');

  const update = (el, displayId, fmt) => {
    const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty('--pct', `${pct}%`);
    document.getElementById(displayId).textContent = fmt(el.value);
  };

  rainfall.addEventListener('input', () =>
    update(rainfall, 'rainfall-value', v => `${v} mm`));
  inundation.addEventListener('input', () =>
    update(inundation, 'inundation-value',
      v => v >= 1000 ? `${(v/1000).toFixed(1)}k sqm` : `${v} sqm`));

  update(rainfall,   'rainfall-value',   v => `${v} mm`);
  update(inundation, 'inundation-value',
    v => v >= 1000 ? `${(v/1000).toFixed(1)}k sqm` : `${v} sqm`);
}

// ════════════════════════════════════════════════ TOGGLES ══
function initToggles() {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const field = btn.dataset.field;
      document.querySelectorAll(`[data-field="${field}"]`)
        .forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (field === 'flood_occurrence') state.flood_occurrence = btn.dataset.value;
      if (field === 'is_good_to_live')  state.is_good_to_live  = btn.dataset.value;
    });
  });

  // Dropdown change → select district
  document.getElementById('district-select').addEventListener('change', function () {
    const name = this.value;
    if (name) selectDistrict(name);
  });
}

// ════════════════════════════════════════════ PREDICTION ══
async function runPrediction() {
  const district = state.selectedDistrict;
  if (!district) { alert('Please select a district first.'); return; }

  const btnText   = document.getElementById('btn-text');
  const btnLoader = document.getElementById('btn-loader');
  const btn       = document.getElementById('predict-btn');
  btnText.style.display   = 'none';
  btnLoader.style.display = 'block';
  btn.disabled = true;

  const payload = {
    district,
    rainfall_7d_mm:                parseFloat(document.getElementById('rainfall-slider').value),
    inundation_area_sqm:           parseFloat(document.getElementById('inundation-slider').value),
    flood_occurrence_current_event: state.flood_occurrence,
    is_good_to_live:                state.is_good_to_live,
    reason_not_good_to_live:        document.getElementById('reason-select').value,
  };

  try {
    const resp = await fetch(`${API_BASE}/api/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json();
      alert(`Prediction error: ${err.detail}`);
      return;
    }

    const result = await resp.json();
    state.lastPredictionId = result.prediction_id;

    // Show result in what-if modal
    showWhatIfResult(result, district);

    // Update the map surface with simulation score for today
    if (state.districtForecasts[district]) {
      const dayData = state.districtForecasts[district][0];
      if (dayData) {
        dayData.risk_score = result.risk_score;
        dayData.risk_level = result.risk_level;
      }
      update3DRiskSurface(0);
    }

    // Update main modal gauge with simulation result
    updateModalGauge(result.risk_score, result.risk_level, district);
    if (result.briefing) {
      document.getElementById('modal-briefing-text').textContent = result.briefing;
    }
    if (result.warnings?.length > 0) {
      const wEl = document.getElementById('modal-warnings');
      wEl.style.display = 'block';
      wEl.innerHTML = result.warnings.map(w => `<div>${w}</div>`).join('');
    }
    document.getElementById('modal-feedback-row').style.display = 'flex';

    appendLogRow(result);

  } catch (err) {
    console.error('[Predict] Error:', err);
    alert('Failed to connect to the prediction API. Is the server running?');
  } finally {
    btnText.style.display   = 'inline';
    btnLoader.style.display = 'none';
    btn.disabled = false;
  }
}

function showWhatIfResult(result, district) {
  const panel = document.getElementById('whatif-result');
  panel.style.display = 'block';
  panel.classList.add('fade-in');

  const score  = result.risk_score;
  const level  = result.risk_level;
  const arcLen = 251;
  const colors = { LOW:'#22c55e', MEDIUM:'#eab308', HIGH:'#f97316', EXTREME:'#ef4444' };

  // Mini gauge
  const arc    = document.getElementById('whatif-gauge-arc');
  const needle = document.getElementById('whatif-gauge-needle');
  arc.style.strokeDashoffset = arcLen - (score * arcLen);
  arc.style.stroke = colors[level] || '#22d3ee';

  const angle = -180 + score * 180;
  const rad   = angle * Math.PI / 180;
  needle.setAttribute('cx', 100 + 80 * Math.cos(rad));
  needle.setAttribute('cy', 100 + 80 * Math.sin(rad));

  document.getElementById('whatif-gauge-score').textContent = score.toFixed(4);
  document.getElementById('whatif-gauge-score').style.color = colors[level] || '#22d3ee';

  const badge = document.getElementById('whatif-risk-badge');
  badge.textContent = level;
  badge.className   = `risk-badge ${level}`;

  document.getElementById('whatif-result-district').textContent = `${district} — Simulated scenario`;

  document.getElementById('whatif-briefing').textContent =
    result.briefing || `Risk score ${score.toFixed(4)} (${level}) for ${district} under specified scenario.`;

  const warnEl = document.getElementById('whatif-warnings');
  if (result.warnings?.length > 0) {
    warnEl.style.display = 'block';
    warnEl.innerHTML = result.warnings.join('<br>');
  } else {
    warnEl.style.display = 'none';
  }
}

// ════════════════════════════════════════ ACTIVITY LOG ══
async function loadActivityLog() {
  try {
    const r = await fetch(`${API_BASE}/api/log?limit=30`);
    if (!r.ok) return;
    const data = await r.json();
    data.predictions.forEach(row => appendLogRow(row, true));
  } catch (_) {}
}

function appendLogRow(result, prepend = false) {
  const body  = document.getElementById('log-body');
  const empty = body.querySelector('.log-empty');
  if (empty) empty.remove();

  const ts      = result.timestamp
    ? new Date(result.timestamp).toLocaleTimeString()
    : new Date().toLocaleTimeString();
  const score   = result.risk_score ?? result.score ?? '—';
  const level   = result.risk_level ?? '—';
  const latency = result.latency_ms ?? '—';
  const hasWarn = result.has_warnings || (result.warnings?.length > 0);

  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${ts}</td>
    <td>${result.district}</td>
    <td>${typeof result.rainfall_7d_mm === 'number' ? result.rainfall_7d_mm.toFixed(0) : result.rainfall_7d ?? '—'}</td>
    <td style="font-weight:600">${typeof score === 'number' ? score.toFixed(4) : score}</td>
    <td><span class="badge badge-${level}">${level}</span></td>
    <td>${latency}</td>
    <td>${hasWarn ? '⚠' : '✓'}</td>
  `;
  tr.classList.add('fade-in');
  body.insertBefore(tr, body.firstChild);
  while (body.rows.length > 100) body.deleteRow(body.rows.length - 1);
  state.logRows.push(result);
}

function exportLog() {
  const rows = state.logRows;
  if (!rows.length) { alert('No data to export.'); return; }
  const headers = ['timestamp','district','rainfall_7d_mm','risk_score','risk_level','latency_ms'];
  const csv = [headers.join(',')].concat(
    rows.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(','))
  ).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), { href: url, download: 'floodguard_log.csv' });
  a.click();
  URL.revokeObjectURL(url);
}

// ════════════════════════════════════════════════ FEEDBACK ══
async function submitFeedback(type) {
  if (!state.lastPredictionId) return;
  try {
    await fetch(`${API_BASE}/api/feedback`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ prediction_id: state.lastPredictionId, feedback_type: type }),
    });
    const id  = type === 'accurate' ? 'btn-thumbup' : 'btn-thumbdown';
    const btn = document.getElementById(id);
    btn.style.transform = 'scale(1.4)';
    setTimeout(() => btn.style.transform = '', 500);
  } catch (err) { console.warn('[Feedback]', err); }
}

// ════════════════════════════════════════ BATCH UPLOAD ══
async function handleBatchUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const text    = await file.text();
  const lines   = text.trim().split('\n');
  if (lines.length < 2) { alert('CSV must have at least a header and one data row.'); return; }

  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const rows    = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
    const obj  = {};
    headers.forEach((h, idx) => { obj[h] = vals[idx] ?? ''; });
    if (!obj.district) continue;
    rows.push({
      district:                       obj.district,
      rainfall_7d_mm:                 parseFloat(obj.rainfall_7d_mm) || 50,
      inundation_area_sqm:            parseFloat(obj.inundation_area_sqm) || 0,
      flood_occurrence_current_event: obj.flood_occurrence_current_event || 'No',
      is_good_to_live:                obj.is_good_to_live || 'Yes',
      reason_not_good_to_live:        obj.reason_not_good_to_live || 'None',
    });
  }

  if (!rows.length) { alert('No valid rows found in CSV.'); return; }

  try {
    const resp = await fetch(`${API_BASE}/api/predict/batch`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ rows }),
    });
    const data = await resp.json();
    data.results.forEach(r => { if (r.risk_score !== undefined) appendLogRow(r); });
    alert(`Batch complete: ${data.total} predictions processed.`);
  } catch (err) {
    alert('Batch upload failed. Check console for details.');
    console.error('[Batch]', err);
  }
}

// ═════════════════════════════════ HISTORICAL SIMULATION ══
async function runHistoricalSimulation() {
  const dateInput = document.getElementById('historical-date');
  const dateVal   = dateInput?.value;
  if (!dateVal) {
    alert('Please select a date first.');
    return;
  }

  // Validate it's a past date
  const selected = new Date(dateVal);
  const today    = new Date();
  today.setHours(0, 0, 0, 0);
  if (selected >= today) {
    alert('Please select a past date for historical simulation.');
    return;
  }

  const btn       = document.getElementById('historical-btn');
  const btnText   = document.getElementById('hist-btn-text');
  const btnLoader = document.getElementById('hist-btn-loader');
  btn.disabled = true;
  btnText.style.display   = 'none';
  btnLoader.style.display = 'block';

  // Show loading overlay on the map
  const mapSection = document.querySelector('.map-section');
  const overlay = document.createElement('div');
  overlay.className = 'historical-loading-overlay';
  overlay.id = 'hist-loading-overlay';
  overlay.innerHTML = `
    <div class="hist-spinner"></div>
    <div class="historical-loading-text">🕰️ Time-Travelling to ${new Date(dateVal).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}...</div>
    <div class="historical-loading-subtext">Fetching actual historical weather data for all 25 districts from Open-Meteo archives</div>
  `;
  mapSection.appendChild(overlay);

  try {
    const resp = await fetch(`${API_BASE}/api/simulate/historical?date=${encodeURIComponent(dateVal)}`);
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Historical simulation failed');
    }

    const data = await resp.json();

    // Save current live forecasts so we can restore later
    if (!state.historicalMode) {
      state.savedLiveForecasts = JSON.parse(JSON.stringify(state.districtForecasts));
      state.savedLiveRiskData  = JSON.parse(JSON.stringify(state.districtRiskData));
    }

    // Inject historical results into districtForecasts (as single-day arrays)
    for (const [name, result] of Object.entries(data.districts)) {
      state.districtForecasts[name] = [{
        date:           result.date,
        rainfall_7d_mm: result.rainfall_7d_mm,
        risk_score:     result.risk_score,
        risk_level:     result.risk_level,
        cached:         false,
        source:         'historical',
      }];
      state.districtRiskData[name] = state.districtForecasts[name][0];
    }

    // Mark historical mode
    state.historicalMode = true;
    state.historicalDate = dateVal;
    state.activeForecastIndex = 0;

    // Re-render the 3D surface with historical data
    update3DRiskSurface(0);

    // Show active banner
    const banner = document.getElementById('historical-active-banner');
    banner.style.display = 'flex';
    document.getElementById('hist-banner-date').textContent =
      new Date(dateVal).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric', weekday: 'long' });

    // Update the LIVE chip to show historical mode
    const liveChip = document.querySelector('.live-chip');
    if (liveChip) {
      liveChip.innerHTML = '<span class="pulse-dot" style="background:var(--accent-purple)"></span><span>HISTORICAL</span>';
      liveChip.style.borderColor = 'rgba(139, 92, 246, 0.3)';
      liveChip.style.color = 'var(--accent-purple)';
    }

    // If a district is currently selected, update its modal with historical data
    if (state.selectedDistrict && data.districts[state.selectedDistrict]) {
      const histResult = data.districts[state.selectedDistrict];
      state.currentForecast = state.districtForecasts[state.selectedDistrict];
      renderModalForecastList();
      updateModalGauge(histResult.risk_score, histResult.risk_level, state.selectedDistrict);
      document.getElementById('modal-briefing-text').textContent =
        `🕰️ Historical Backtest — ${state.selectedDistrict} on ${dateVal}: ` +
        `Actual 7-day rainfall ${histResult.rainfall_7d_mm.toFixed(0)}mm from Open-Meteo archive. ` +
        `Predicted risk score ${histResult.risk_score.toFixed(4)} (${histResult.risk_level}).`;
    }

    console.log(`[Historical] Loaded ${data.ready}/${data.total} districts for ${dateVal}`);
    if (data.errors?.length > 0) {
      console.warn('[Historical] Errors:', data.errors);
    }

  } catch (err) {
    console.error('[Historical] Error:', err);
    alert(`Historical simulation failed: ${err.message}`);
  } finally {
    // Remove loading overlay
    const loadingOverlay = document.getElementById('hist-loading-overlay');
    if (loadingOverlay) loadingOverlay.remove();

    btnText.style.display   = 'inline';
    btnLoader.style.display = 'none';
    btn.disabled = false;
  }
}

function returnToLive() {
  if (!state.historicalMode) return;

  // Restore saved live forecasts
  if (state.savedLiveForecasts) {
    state.districtForecasts = state.savedLiveForecasts;
    state.districtRiskData  = state.savedLiveRiskData;
    state.savedLiveForecasts = null;
    state.savedLiveRiskData  = null;
  }

  state.historicalMode = false;
  state.historicalDate = null;
  state.activeForecastIndex = 0;

  // Re-render the 3D surface with live data
  update3DRiskSurface(0);

  // Hide banner
  document.getElementById('historical-active-banner').style.display = 'none';

  // Restore LIVE chip
  const liveChip = document.querySelector('.live-chip');
  if (liveChip) {
    liveChip.innerHTML = '<span class="pulse-dot"></span><span>LIVE</span>';
    liveChip.style.borderColor = 'rgba(34,197,94,0.3)';
    liveChip.style.color = 'var(--risk-low)';
  }

  // Restore selected district modal if open
  if (state.selectedDistrict && state.districtForecasts[state.selectedDistrict]) {
    state.currentForecast = state.districtForecasts[state.selectedDistrict];
    renderModalForecastList();
    selectForecastDay(0);
  }

  console.log('[Historical] Returned to live mode.');
}

// ════════════════════════════════════════════ EVACUATION POINTS ══
async function loadEvacuationPoints() {
  try {
    const resp = await fetch('/static/evacuation_points.json');
    if (!resp.ok) return;
    state.evacuationData = await resp.json();
  } catch (err) {
    console.warn('[Evacuation] Failed to load evacuation points:', err);
  }
}

function toggleEvacuationPoints(show) {
  state.showEvacuationPoints = show;
  
  if (!state.viewer) return;

  // If turning off, remove all entities
  if (!show) {
    state.evacuationEntities.forEach(e => state.viewer.entities.remove(e));
    state.evacuationEntities = [];
    return;
  }

  // If turning on, render them
  const colors = {
    'School': '#3b82f6',
    'Temple': '#f59e0b',
    'Stadium': '#10b981',
    'default': '#22d3ee'
  };

  state.evacuationData.forEach(pt => {
    const colorStr = colors[pt.type] || colors['default'];
    
    const entity = state.viewer.entities.add({
      name: pt.name,
      position: Cesium.Cartesian3.fromDegrees(pt.lon, pt.lat, pt.elevation_m || 20),
      point: {
        pixelSize: 8,
        color: Cesium.Color.fromCssColorString(colorStr),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `🛡️ ${pt.name}`,
        font: '10px Inter, sans-serif',
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        pixelOffset: new Cesium.Cartesian2(0, -10),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        showBackground: true,
        backgroundColor: Cesium.Color.fromCssColorString('rgba(5,13,26,0.8)'),
        backgroundPadding: new Cesium.Cartesian2(6, 4)
      }
    });
    state.evacuationEntities.push(entity);
  });
}

// ════════════════════════════════════════════════════ MY LOCATION ══

function toggleMyLocation() {
  const btn = document.getElementById('my-location-btn');

  if (state.myLocationActive) {
    // ── Turn OFF ──
    stopMyLocation();
    btn.classList.remove('active');
    return;
  }

  if (!('geolocation' in navigator)) {
    alert('Geolocation is not supported by your browser.');
    return;
  }

  btn.classList.add('active');
  state.myLocationActive = true;

  // Get initial position and fly to it
  navigator.geolocation.getCurrentPosition(
    pos => {
      plotMyLocation(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy);
      flyToMyLocation(pos.coords.latitude, pos.coords.longitude);
    },
    err => {
      console.warn('[GeoLocation] Error:', err.message);
      alert('Could not access your location. Please allow location permissions.');
      stopMyLocation();
      btn.classList.remove('active');
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
  );

  // Continuously watch for location changes
  state.geoWatchId = navigator.geolocation.watchPosition(
    pos => {
      plotMyLocation(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy);
    },
    err => {
      console.warn('[GeoLocation] Watch error:', err.message);
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 }
  );
}

function stopMyLocation() {
  state.myLocationActive = false;

  if (state.geoWatchId !== null) {
    navigator.geolocation.clearWatch(state.geoWatchId);
    state.geoWatchId = null;
  }
  if (state.myLocationEntity && state.viewer) {
    state.viewer.entities.remove(state.myLocationEntity);
    state.myLocationEntity = null;
  }
  if (state.myLocationRingEntity && state.viewer) {
    state.viewer.entities.remove(state.myLocationRingEntity);
    state.myLocationRingEntity = null;
  }
}

function plotMyLocation(lat, lon, accuracy) {
  if (!state.viewer) return;

  // Remove previous entities
  if (state.myLocationEntity) state.viewer.entities.remove(state.myLocationEntity);
  if (state.myLocationRingEntity) state.viewer.entities.remove(state.myLocationRingEntity);

  // Accuracy ring radius (clamp between 50m and 2000m for visibility)
  const ringRadius = Math.max(50, Math.min(accuracy || 100, 2000));

  // Blue accuracy circle on the ground
  state.myLocationRingEntity = state.viewer.entities.add({
    name: '_my_location_ring',
    position: Cesium.Cartesian3.fromDegrees(lon, lat),
    ellipse: {
      semiMinorAxis: ringRadius,
      semiMajorAxis: ringRadius,
      height: 0,
      material: Cesium.Color.fromCssColorString('rgba(59, 130, 246, 0.12)'),
      outline: true,
      outlineColor: Cesium.Color.fromCssColorString('rgba(59, 130, 246, 0.4)'),
      outlineWidth: 1,
    }
  });

  // Bright blue dot
  state.myLocationEntity = state.viewer.entities.add({
    name: '_my_location',
    position: Cesium.Cartesian3.fromDegrees(lon, lat, 200),
    point: {
      pixelSize: 12,
      color: Cesium.Color.fromCssColorString('#3b82f6'),
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: 3,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text: 'You are here',
      font: 'bold 11px Inter, sans-serif',
      fillColor: Cesium.Color.WHITE,
      outlineColor: Cesium.Color.fromCssColorString('#050d1a'),
      outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
      pixelOffset: new Cesium.Cartesian2(0, -12),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      showBackground: true,
      backgroundColor: Cesium.Color.fromCssColorString('rgba(59, 130, 246, 0.85)'),
      backgroundPadding: new Cesium.Cartesian2(8, 5),
    }
  });
}

function flyToMyLocation(lat, lon) {
  if (!state.viewer) return;
  state.viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat, 50000),
    orientation: {
      heading: Cesium.Math.toRadians(0),
      pitch: Cesium.Math.toRadians(-55),
      roll: 0,
    },
    duration: 2.5,
  });
}

// ════════════════════════════════════════════════════ START ══
document.addEventListener('DOMContentLoaded', init);
