/**
 * Flood Timeline — Dashboard Application v2
 * 3D CesiumJS terrain + district detail bottom modal + what-if simulator modal
 * Background precompute polling → progressive district risk prism rendering
 */

'use strict';

// ═══════════════════════════════════════════════════════════ CONFIG ══
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : '';

// ══════════════════════════════════════════════════════════ STATE ══
const state = {
  viewer: null,
  districts: {},          // name → reference data
  riskEntities: {},
  subdivisionEntities: [],
  nationalViewActive: true,          // name → [entity, entity, …] for cleanup
  selectedDistrict: null,
  lastPredictionId: null,
  logRows: [],
  flood_occurrence: 'No',
  is_good_to_live: 'Yes',
  currentForecast: [],
  activeForecastIndex: 0,
  districtRiskData: {},          // name → {risk_score, risk_level, rainfall_7d_mm}
  districtForecasts: {},          // name → list of 7 days forecast
  meshPrimitive: null,
  wireframePrimitive: null,
  precomputePollTimer: null,
  clickHandler: null,
  historicalMode: false,       // true when viewing a past date
  historicalDate: null,        // ISO date string when in historical mode
  savedLiveForecasts: null,        // snapshot of live districtForecasts
  savedLiveRiskData: null,        // snapshot of live districtRiskData
  evacuationEntities: [],          // List of Cesium Entities for safe zones
  evacuationData: [],          // JSON data from evacuation_points.json
  showEvacuationPoints: false,      // UI toggle state
  // Geolocation
  myLocationActive: false,
  myLocationEntity: null,        // Cesium entity for user's location pin
  myLocationRingEntity: null,       // Cesium entity for pulsing accuracy ring
  geoWatchId: null,        // navigator.geolocation watchPosition ID
};

// ══════════════════════════════════════════════════════════ INIT ══
async function init() {
  // Cesium token
  if (!window.__CESIUM_TOKEN__) {
    try {
      const r = await fetch('/api/config/cesium-token');
      if (r.ok) { const d = await r.json(); Cesium.Ion.defaultAccessToken = d.token || ''; }
    } catch (_) { }
  } else {
    Cesium.Ion.defaultAccessToken = window.__CESIUM_TOKEN__;
  }

  initCesium();
  await loadDistricts();
  await loadModelCard();
  await loadActivityLog();
  await loadEvacuationPoints();
  startPrecomputePolling();

  // Show last refresh time (from SW cache or set to now)
  updateLastRefreshTime();
}

/** Update the "Last Refreshed" chip in the header */
function updateLastRefreshTime() {
  const el = document.getElementById('last-refresh-label');
  if (!el) return;

  // Try to read the timestamp stored by the Service Worker
  if ('caches' in window) {
    caches.open('flood-timeline-cache-v4').then(cache => {
      cache.match('/__last_refresh_time__').then(resp => {
        if (resp) {
          resp.text().then(ts => {
            const d = new Date(ts);
            if (!isNaN(d)) {
              el.textContent = formatRefreshTime(d);
              return;
            }
          });
        }
        // No cached timestamp — set to now (first load)
        setRefreshTimeNow();
      });
    }).catch(() => setRefreshTimeNow());
  } else {
    setRefreshTimeNow();
  }
}

function setRefreshTimeNow() {
  const el = document.getElementById('last-refresh-label');
  if (el) el.textContent = formatRefreshTime(new Date());
  // Also store it
  if ('caches' in window) {
    caches.open('flood-timeline-cache-v4').then(cache => {
      cache.put(
        new Request('/__last_refresh_time__'),
        new Response(new Date().toISOString())
      );
    }).catch(() => { });
  }
}

function formatRefreshTime(d) {
  const now = new Date();
  const diffMs = now - d;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

// ════════════════════════════════════════════════ CESIUM 3D GLOBE ══
function initCesium() {
  try {
    const opts = {
      baseLayerPicker: true,
      navigationHelpButton: false,
      sceneModePicker: false,
      homeButton: false,
      geocoder: false,
      fullscreenButton: false,
      timeline: false,
      animation: false,
      infoBox: false,
      selectionIndicator: false,
      skyBox: false,
      skyAtmosphere: new Cesium.SkyAtmosphere(),
    };

    if (Cesium.Ion.defaultAccessToken) {
      opts.terrain = Cesium.Terrain.fromWorldTerrain({ requestWaterMask: true });
    }

    state.viewer = new Cesium.Viewer('cesium-container', opts);

    // Replace default Cesium Ion base layer with CartoDB dark tiles
    // Uses the modern ImageryLayer constructor (addImageryProvider is deprecated)
    try {
      const cartoProvider = new Cesium.UrlTemplateImageryProvider({
        url: 'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        minimumLevel: 0,
        maximumLevel: 18,
        credit: 'Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
      });
      const cartoLayer = new Cesium.ImageryLayer(cartoProvider);
      // Remove default Ion imagery first, then add dark CartoDB
      state.viewer.imageryLayers.removeAll();
      state.viewer.imageryLayers.add(cartoLayer);
      console.log('[CesiumJS] CartoDB dark base layer added successfully.');
    } catch (e) {
      console.warn('[CesiumJS] CartoDB tiles failed, keeping default imagery:', e);
    }

    // Show the globe, hide sun/moon/atmosphere for clean dark UI
    state.viewer.scene.globe.show = true;
    state.viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#0a1628');
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
        pitch: Cesium.Math.toRadians(-65),
        roll: 0,
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
  return Cesium.Color.fromBytes(r, g, b, Math.round(alpha * 255));
}

/** Plot district pins on the map without the 3D surface mesh */
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

  // 2. Update all computed district pin flags to float near the surface
  for (const name of calculatedDistricts) {
    const ref = state.districts[name];
    const forecastList = state.districtForecasts[name];
    if (!ref || !forecastList) continue;

    const dayData = forecastList[dayIdx] || forecastList[0];
    const score = dayData ? dayData.risk_score : 0;

    // Minimal stem height so the pin is visible above terrain
    const localHeight = 2000;

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
      text: `${name}\n${(score * 100).toFixed(0)}%`,
      font: isSelected ? 'bold 12px Inter, sans-serif' : 'bold 10px Inter, sans-serif',
      fillColor: Cesium.Color.WHITE,
      outlineColor: isSelected ? Cesium.Color.fromCssColorString('#22d3ee') : Cesium.Color.fromCssColorString('#050d1a'),
      outlineWidth: isSelected ? 5 : 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
      pixelOffset: new Cesium.Cartesian2(0, -4),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      showBackground: true,
      backgroundColor: isSelected ? Cesium.Color.fromCssColorString('#0f2d4a').withAlpha(0.9) : Cesium.Color.fromCssColorString('#0a1628').withAlpha(0.75),
      backgroundPadding: new Cesium.Cartesian2(8, 6),
    },
    point: {
      pixelSize: isSelected ? 9 : 6,
      color: isSelected ? Cesium.Color.fromCssColorString('#22d3ee') : color,
      outlineColor: Cesium.Color.WHITE,
      outlineWidth: isSelected ? 2 : 1.5,
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
      pixelSize: 5,
      color: Cesium.Color.fromCssColorString('#475569'),
      outlineColor: Cesium.Color.fromCssColorString('#1e293b'),
      outlineWidth: 1,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text: name,
      font: '9px Inter, sans-serif',
      fillColor: Cesium.Color.fromCssColorString('#94a3b8'),
      outlineColor: Cesium.Color.fromCssColorString('#050d1a'),
      outlineWidth: 2,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
      pixelOffset: new Cesium.Cartesian2(0, -6),
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
  const label = document.getElementById('district-flyto-label');
  label.textContent = `📍 ${name}`;
  overlay.style.display = 'block';
  setTimeout(() => { overlay.style.display = 'none'; }, 3000);

  state.viewer.camera.flyTo({
    // Offset the camera 1.5 degrees South to compensate for the -48 degree pitch
    // This perfectly centers the target district in the viewport
    destination: Cesium.Cartesian3.fromDegrees(lon, lat - 1.5, 200000),
    orientation: {
      heading: Cesium.Math.toRadians(0),
      pitch: Cesium.Math.toRadians(-48),
      roll: 0,
    },
    duration: 2.2,
  });
}

// ════════════════════════════════════════════ DISTRICT LOADING ══
async function loadDistricts() {
  try {
    const resp = await fetch(`${API_BASE}/api/districts`);
    const data = await resp.json();
    const sel = document.getElementById('district-select');
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
  const banner = document.getElementById('header-precompute');
  const fill = document.getElementById('precompute-fill');
  const label = document.getElementById('precompute-label');
  const chip = document.getElementById('district-count-chip');
  const readyCnt = document.getElementById('districts-ready-count');

  banner.style.display = 'flex';
  chip.style.display = 'flex';

  const evtSource = new EventSource(`${API_BASE}/api/forecasts/stream`);

  evtSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      const pct = data.total > 0 ? (data.ready / data.total) * 100 : 0;

      fill.style.width = `${pct}%`;
      label.textContent = (data.ready >= data.total)
        ? `All ${data.total} risk profiles ready`
        : `Computing district profiles… ${data.ready}/${data.total}`;
      readyCnt.textContent = data.ready;

      let newlyAdded = false;
      for (const [name, forecastList] of Object.entries(data.districts)) {
        if (!state.districtForecasts[name]) {
          state.districtForecasts[name] = forecastList;
          state.districtRiskData[name] = forecastList[0];
          newlyAdded = true;
        }
      }
      
      if (newlyAdded) {
        update3DRiskSurface(state.activeForecastIndex);
      }

      if (state.selectedDistrict) {
        document.getElementById('simulate-btn').disabled = false;
      }

      if (data.ready >= data.total) {
        setTimeout(() => {
          banner.style.display = 'none';
        }, 4000);
        label.textContent = `✓ All ${data.total} district risk profiles loaded`;
        evtSource.close();
      }
    } catch (e) {
      console.warn('[Precompute] SSE parse failed:', e);
    }
  };

  evtSource.onerror = function() {
    console.warn('[Precompute] SSE connection error');
    evtSource.close();
  };
}

// ════════════════════════════════════════════ DISTRICT SELECT ══
function selectDistrict(name) {

  if (state.nationalViewActive) loadSubdivisions(name);

  // Ensure activity log is closed when district modal opens (mobile)
  const log = document.getElementById('activity-log');
  if (log && log.classList.contains('mobile-visible')) {
    toggleMobileLog();
  }

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
  document.body.classList.add('district-modal-open');

  // Set header info
  document.getElementById('modal-district-name').textContent = name;
  document.getElementById('modal-district-sub').textContent =
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
  document.body.classList.remove('district-modal-open');
  if (!state.nationalViewActive) {
    restoreNationalView();
  }
}

async function loadDistrictForecast(name) {
  try {
    const r = await fetch(`${API_BASE}/api/forecast/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error('Forecast API error');
    const data = await r.json();
    state.currentForecast = data.forecast;
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
    const dt = new Date(day.date);
    const dateStr = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const dayName = idx === 0 ? 'Today' : dt.toLocaleDateString(undefined, { weekday: 'short' });
    const pct = (day.risk_score * 100).toFixed(0);

    const row = document.createElement('div');
    row.className = `modal-forecast-item ${idx === 0 ? 'active' : ''}`;
    row.id = `mfi-${idx}`;
    row.onclick = () => selectForecastDay(idx);
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
  const arc = document.getElementById('modal-gauge-arc');
  const needle = document.getElementById('modal-gauge-needle');
  const scoreEl = document.getElementById('modal-gauge-score');
  const labelEl = document.getElementById('modal-gauge-label');
  const badge = document.getElementById('modal-risk-badge');

  const colors = { LOW: '#22c55e', MEDIUM: '#eab308', HIGH: '#f97316', EXTREME: '#ef4444' };

  if (score === null) {
    arc.style.strokeDashoffset = arcLen;
    arc.style.stroke = '#22d3ee';
    needle.setAttribute('cx', 20);
    needle.setAttribute('cy', 100);
    scoreEl.textContent = '—';
    scoreEl.style.color = 'var(--text-primary)';
    labelEl.textContent = district || 'NO DATA';
    badge.textContent = 'LOADING';
    badge.className = 'risk-badge modal-risk-badge';
    return;
  }

  arc.style.strokeDashoffset = arcLen - (score * arcLen);
  arc.style.stroke = colors[level] || '#22d3ee';

  const angle = -180 + score * 180;
  const rad = angle * Math.PI / 180;
  const cx = 100, cy = 100, r = 80;
  needle.setAttribute('cx', cx + r * Math.cos(rad));
  needle.setAttribute('cy', cy + r * Math.sin(rad));

  scoreEl.textContent = score.toFixed(4);
  scoreEl.style.color = colors[level] || '#22d3ee';
  labelEl.textContent = district || '';

  badge.textContent = level;
  badge.className = `risk-badge modal-risk-badge ${level}`;
}

// ═══════════════════════════════════════ LAB REDIRECT ══
function openDistrictInLab() {
  const district = state.selectedDistrict;
  if (!district) {
    alert('Please select a district first.');
    return;
  }
  window.location.href = `/lab?district=${encodeURIComponent(district)}`;
}

// ════════════════════════════════════════════ MODEL CARD ══
async function loadModelCard() {
  try {
    const r = await fetch(`${API_BASE}/api/models`);
    if (!r.ok) return;
    const m = await r.json();
    const versionText = m.base_pipeline ? `${m.base_pipeline.toUpperCase()} pipeline` : 'V703 pipeline';
    document.getElementById('model-version-label').textContent =
      `${versionText} | LB ${m.opt_lb_score?.toFixed(5) ?? '—'}`;
    document.getElementById('stat-pipeline').textContent = m.base_pipeline ?? '—';
    document.getElementById('stat-mae').textContent = m.oof_mae?.toFixed(5) ?? '—';
    document.getElementById('stat-ev').textContent = m.oof_ev?.toFixed(5) ?? '—';
    document.getElementById('stat-lb').textContent = m.opt_lb_score?.toFixed(5) ?? '—';
    document.getElementById('stat-feats').textContent = `${m.n_total_features} cols`;
    const d = new Date(m.training_date);
    document.getElementById('stat-date').textContent = isNaN(d) ? '—' : d.toLocaleDateString();
  } catch (err) {
    console.warn('[ModelCard] Failed:', err);
  }
}

// ════════════════════════════════════════════ EVENT LISTENERS ══
document.addEventListener('DOMContentLoaded', () => {
  const selectEl = document.getElementById('district-select');
  if (selectEl) {
    selectEl.addEventListener('change', function () {
      const name = this.value;
      if (name) selectDistrict(name);
    });
  }
});



function setReportButtonsEnabled(enabled) {
  const modalBtn = document.getElementById('modal-report-btn');
  if (modalBtn) modalBtn.disabled = !enabled;

  const whatIfBtn = document.getElementById('whatif-report-btn');
  if (whatIfBtn) {
    whatIfBtn.style.display = enabled ? 'inline-flex' : 'none';
    whatIfBtn.disabled = !enabled;
  }
}

function downloadReport() {
  if (!state.lastPredictionId) {
    alert('Run a prediction first. The report is generated from the latest prediction ID.');
    return;
  }
  const url = `${API_BASE}/api/report/${encodeURIComponent(state.lastPredictionId)}`;
  window.open(url, '_blank');
}

// ════════════════════════════════════════════ PREDICTION ══

// ════════════════════════════════════════ ACTIVITY LOG ══
async function loadActivityLog() {
  try {
    const r = await fetch(`${API_BASE}/api/log?limit=30`);
    if (!r.ok) return;
    const data = await r.json();
    data.predictions.forEach(row => appendLogRow(row, true));
  } catch (_) { }
}

function appendLogRow(result, prepend = false) {
  const body = document.getElementById('log-body');
  const empty = body.querySelector('.log-empty');
  if (empty) empty.remove();

  const ts = result.timestamp
    ? new Date(result.timestamp).toLocaleTimeString()
    : new Date().toLocaleTimeString();
  const score = result.risk_score ?? result.score ?? '—';
  const level = result.risk_level ?? '—';
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
  const headers = ['timestamp', 'district', 'rainfall_7d_mm', 'risk_score', 'risk_level', 'latency_ms'];
  const csv = [headers.join(',')].concat(
    rows.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(','))
  ).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href: url, download: 'flood-timeline_log.csv' });
  a.click();
  URL.revokeObjectURL(url);
}

// ════════════════════════════════════════════════ FEEDBACK ══
async function submitFeedback(type) {
  if (!state.lastPredictionId) return;
  try {
    await fetch(`${API_BASE}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction_id: state.lastPredictionId, feedback_type: type }),
    });
    const id = type === 'accurate' ? 'btn-thumbup' : 'btn-thumbdown';
    const btn = document.getElementById(id);
    btn.style.transform = 'scale(1.4)';
    setTimeout(() => btn.style.transform = '', 500);
  } catch (err) { console.warn('[Feedback]', err); }
}

// ════════════════════════════════════════ BATCH UPLOAD ══

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
    // Offset the camera 0.3 degrees South to compensate for the -55 degree pitch
    destination: Cesium.Cartesian3.fromDegrees(lon, lat - 0.3, 50000),
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



// ════════════════════════════════════════════════ DRILL DOWN ══

async function loadSubdivisions(districtName) {
  state.nationalViewActive = false;
  
  for (let key in state.riskEntities) {
    state.riskEntities[key].forEach(ent => {
      if (key === districtName) {
         if (ent.polygon) {
            ent.polygon.outline = true;
            ent.polygon.outlineColor = Cesium.Color.WHITE;
            ent.polygon.outlineWidth = 3;
            ent.polygon.extrudedHeight = 500;
            ent.polygon.material = ent.polygon.material.color.getValue().withAlpha(0.2);
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
    const dateQuery = state.simulationDate ? `?date=${state.simulationDate}` : '';
    const res = await fetch(`/api/predict/subdivisions/${districtName}${dateQuery}`);
    const results = await res.json();
    
    results.forEach(sub => {
       const scorePct = Math.round(sub.risk_score * 100);
       const color = getRiskColor(sub.risk_score);
       
       const ent = state.viewer.entities.add({
         position: Cesium.Cartesian3.fromDegrees(sub.lon, sub.lat, 1000),
         label: {
           text: `${sub.place_name}
${scorePct}% Risk | ${sub.rainfall_7d_mm}mm Rain`,
           font: 'bold 13px sans-serif',
           fillColor: Cesium.Color.WHITE,
           style: Cesium.LabelStyle.FILL,
           pixelOffset: new Cesium.Cartesian2(0, -25),
           backgroundColor: color.withAlpha(0.9),
           showBackground: true,
           backgroundPadding: new Cesium.Cartesian2(8, 6),
           disableDepthTestDistance: Number.POSITIVE_INFINITY,
           show: false // Hidden by default, shown on hover
         },
         point: {
           pixelSize: 14,
           color: color,
           outlineColor: Cesium.Color.WHITE,
           outlineWidth: 2,
           disableDepthTestDistance: Number.POSITIVE_INFINITY
         }
       });
       state.subdivisionEntities.push(ent);
    });




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



    // Auto-Zoom into the district

    if (state.subdivisionEntities.length > 0) {
      state.viewer.flyTo(state.subdivisionEntities, {
        duration: 1.5,
        offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-60), 60000)
      });
    }

    // Attach Hover Handler if not present
    if (!state.hoverHandler) {
      state.hoverHandler = new Cesium.ScreenSpaceEventHandler(state.viewer.scene.canvas);
      let lastHoveredEntity = null;
      state.hoverHandler.setInputAction(function (movement) {
        if (!state.nationalViewActive) {
          const picked = state.viewer.scene.pick(movement.endPosition);
          const isSubEntity = Cesium.defined(picked) && picked.id && state.subdivisionEntities.includes(picked.id);
          
          if (lastHoveredEntity && lastHoveredEntity !== (isSubEntity ? picked.id : null)) {
              if (lastHoveredEntity.label) lastHoveredEntity.label.show = false;
          }
          
          if (isSubEntity) {
              if (picked.id.label) picked.id.label.show = true;
              lastHoveredEntity = picked.id;
          } else {
              lastHoveredEntity = null;
          }
        }
      }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
    }

  } catch(e) {
    console.error("Subdivisions fetch failed", e);
  }
}

function clearSubdivisions() {
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
}

function restoreNationalView() {
  state.nationalViewActive = true;
  document.getElementById('btn-back-national').style.display = 'none';
  clearSubdivisions();
  
  // Restore Colors
  for (let key in state.riskEntities) {
    state.riskEntities[key].forEach(ent => {
      if (ent.polygon && ent.polygon.material) {
        const data = state.districtRiskData[key];
        if (data) {
           ent.polygon.material = getInterpolatedColor(data.risk_score, 0.6);
        }
        ent.polygon.outline = false;
        ent.polygon.extrudedHeight = undefined;
      }
      if (ent.label) ent.label.show = true;
    });
  }

  // Fly back to national view
  state.viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(80.7, 4.5, 880000),
    orientation: {
      heading: Cesium.Math.toRadians(0),
      pitch: Cesium.Math.toRadians(-65),
      roll: 0,
    },
    duration: 1.5,
  });
}


// ════════════════════════════════════════════════════════ ZOOM ══
let lastMousePos = null;
document.addEventListener('mousemove', (e) => {
  const container = document.getElementById('cesium-container');
  if (container && container.contains(e.target)) {
    const rect = container.getBoundingClientRect();
    lastMousePos = new Cesium.Cartesian2(e.clientX - rect.left, e.clientY - rect.top);
  }
});

function zoomMapTowards(isZoomIn, useMouse) {
  if (!state.viewer) return;
  const camera = state.viewer.camera;
  const height = camera.positionCartographic.height;
  
  if (useMouse && lastMousePos) {
    const ray = camera.getPickRay(lastMousePos);
    if (ray) {
      const intersection = state.viewer.scene.globe.pick(ray, state.viewer.scene);
      if (intersection) {
        const direction = Cesium.Cartesian3.subtract(intersection, camera.position, new Cesium.Cartesian3());
        const distance = Cesium.Cartesian3.magnitude(direction);
        Cesium.Cartesian3.normalize(direction, direction);
        const moveAmount = isZoomIn ? distance * 0.2 : -distance * 0.2;
        camera.move(direction, moveAmount);
        return;
      }
    }
  }
  
  // Fallback for button clicks (zooms to center)
  const amount = height * 0.2;
  if (isZoomIn) camera.zoomIn(amount);
  else camera.zoomOut(amount);
}

function zoomMapIn() { zoomMapTowards(true, false); }
function zoomMapOut() { zoomMapTowards(false, false); }

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  
  if (e.key === '+' || e.key === '=' || e.key === 'i' || e.key === 'I') {
    zoomMapTowards(true, true);
  } else if (e.key === '-' || e.key === '_' || e.key === 'o' || e.key === 'O') {
    zoomMapTowards(false, true);
  }
});

// ═══════════════════════════════════════════════════════ MOBILE UI ══

/**
 * Toggle the mobile sidebar drawer (left slide-in)
 */
function toggleMobileSidebar() {
  const panel = document.getElementById('control-panel');
  const backdrop = document.getElementById('sidebar-backdrop');
  const isOpen = panel.classList.contains('mobile-open');
  
  if (isOpen) {
    closeMobileSidebar();
  } else {
    panel.classList.add('mobile-open');
    backdrop.classList.add('visible');
    document.body.style.overflow = 'hidden';
  }
}

function closeMobileSidebar() {
  const panel = document.getElementById('control-panel');
  const backdrop = document.getElementById('sidebar-backdrop');
  panel.classList.remove('mobile-open');
  backdrop.classList.remove('visible');
  document.body.style.overflow = '';
}

/**
 * Toggle the mobile activity log overlay
 */
function toggleMobileLog() {
  const log = document.getElementById('activity-log');
  const btn = document.getElementById('mobile-log-toggle');
  const isVisible = log.classList.contains('mobile-visible');
  
  if (isVisible) {
    log.classList.remove('mobile-visible');
    btn.classList.remove('active');
  } else {
    // Close district modal if it's open to prevent overlap
    const modal = document.getElementById('district-modal');
    if (modal && modal.classList.contains('open')) {
      closeDistrictModal();
    }
    
    log.classList.add('mobile-visible');
    btn.classList.add('active');
  }
}

/**
 * Auto-close mobile drawer when resizing to desktop width
 */
const mobileMediaQuery = window.matchMedia('(max-width: 768px)');

function handleMobileChange(e) {
  if (!e.matches) {
    // Switched to desktop — close mobile drawers
    closeMobileSidebar();
    const log = document.getElementById('activity-log');
    const btn = document.getElementById('mobile-log-toggle');
    if (log) log.classList.remove('mobile-visible');
    if (btn) btn.classList.remove('active');
    document.body.style.overflow = '';
  }
}

// Modern API (addEventListener) with fallback (addListener for older Safari)
if (mobileMediaQuery.addEventListener) {
  mobileMediaQuery.addEventListener('change', handleMobileChange);
} else if (mobileMediaQuery.addListener) {
  mobileMediaQuery.addListener(handleMobileChange);
}

/**
 * Touch swipe gesture: swipe left on sidebar to close
 */
(function initSidebarSwipe() {
  const panel = document.getElementById('control-panel');
  if (!panel) return;

  let touchStartX = 0;
  let touchStartY = 0;
  let isSwiping = false;

  panel.addEventListener('touchstart', (e) => {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
    isSwiping = false;
  }, { passive: true });

  panel.addEventListener('touchmove', (e) => {
    const dx = e.touches[0].clientX - touchStartX;
    const dy = e.touches[0].clientY - touchStartY;
    // Only track horizontal swipes (more horizontal than vertical)
    if (Math.abs(dx) > Math.abs(dy) && dx < -20) {
      isSwiping = true;
    }
  }, { passive: true });

  panel.addEventListener('touchend', () => {
    if (isSwiping) {
      closeMobileSidebar();
    }
    isSwiping = false;
  }, { passive: true });
})();

/**
 * Auto-close sidebar when a district is selected on mobile
 */
const _originalSelectDistrict = selectDistrict;
selectDistrict = function(name) {
  _originalSelectDistrict(name);
  // Close mobile sidebar after selecting a district
  if (mobileMediaQuery.matches) {
    closeMobileSidebar();
  }
};

