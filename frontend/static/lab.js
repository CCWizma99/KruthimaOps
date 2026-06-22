/**
 * Flood Timeline — Experiment Lab Controller
 * Features: Tab switching, Scenario Simulation, Sensitivity Sweeps, Comparative Sandbox, Historical Backtesting, Batch Upload
 */

'use strict';

// ═══════════════════════════════════════════════════════════ CONFIG ══
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : '';

// ══════════════════════════════════════════════════════════ STATE ══
const state = {
  districts: [],
  selectedDistrict: null,
  flood_occurrence: 'No',
  is_good_to_live: 'Yes',
  lastPredictionId: null,
  lastPredictionData: null,
  
  // Sandbox Scenarios
  savedScenarios: [],
  compareSelection: [], // IDs of selected scenarios to compare
  
  // Historical backtest data
  backtestResults: [],
  
  // Batch results
  batchResults: []
};

// ══════════════════════════════════════════════════════════ INITIALIZE ══
document.addEventListener('DOMContentLoaded', async () => {
  await initLab();
});

async function initLab() {
  initSliders();
  initDatePicker();
  await loadDistricts();
  
  // Check query params for prefilled district
  const params = new URLSearchParams(window.location.search);
  const preDistrict = params.get('district');
  if (preDistrict) {
    const sel = document.getElementById('lab-district-select');
    if (sel) {
      sel.value = preDistrict;
      // Trigger select handler
      handleDistrictSelect(preDistrict);
    }
  }
}

// ══════════════════════════════════════════════════════════ NAV TABS ══
function switchTab(tabId) {
  // Hide all sections
  document.querySelectorAll('.lab-content-section').forEach(sec => {
    sec.classList.remove('active');
  });
  
  // Deactivate all tab buttons
  document.querySelectorAll('.lab-tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  
  // Show target section & button
  document.getElementById(`tab-${tabId}`).classList.add('active');
  
  // Find matching button
  const matchingBtn = Array.from(document.querySelectorAll('.lab-tab-btn')).find(btn => 
    btn.getAttribute('onclick').includes(tabId)
  );
  if (matchingBtn) matchingBtn.classList.add('active');
}

// ══════════════════════════════════════════════════════════ DISTRICTS ══
async function loadDistricts() {
  try {
    const resp = await fetch(`${API_BASE}/api/districts`);
    if (!resp.ok) throw new Error('API Error');
    const data = await resp.json();
    state.districts = data.districts;
    
    const sel = document.getElementById('lab-district-select');
    sel.innerHTML = '<option value="">Select district...</option>';
    
    for (const name of data.districts) {
      const opt = document.createElement('option');
      opt.value = opt.textContent = name;
      sel.appendChild(opt);
    }
    
    sel.addEventListener('change', (e) => {
      handleDistrictSelect(e.target.value);
    });
  } catch (err) {
    console.error('[Lab Districts] Load failed:', err);
  }
}

function handleDistrictSelect(districtName) {
  state.selectedDistrict = districtName;
  if (!districtName) {
    document.getElementById('lab-result-title').textContent = 'Select a district & run';
    document.getElementById('lab-save-btn').disabled = true;
    clearSweepChart();
    return;
  }
  
  document.getElementById('lab-result-title').textContent = `${districtName} — Configured scenario`;
  
  // Automatically trigger sensitivity sweep for the district
  runSensitivitySweep(districtName);
}

// ══════════════════════════════════════════════════════════ SLIDERS ══
function initSliders() {
  const rainfall = document.getElementById('lab-rainfall-slider');
  const inundation = document.getElementById('lab-inundation-slider');
  
  const updateReadout = (el, displayId, fmt) => {
    const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty('--pct', `${pct}%`);
    document.getElementById(displayId).textContent = fmt(el.value);
  };
  
  rainfall.addEventListener('input', () => {
    // Physics Guardrail: If rain > 150, inundation cannot be 0.
    if (parseFloat(rainfall.value) > 150) {
      if (parseFloat(inundation.value) < 5000) {
        inundation.value = 5000;
        updateReadout(inundation, 'lab-inundation-value', 
          v => v >= 1000 ? `${(v / 1000).toFixed(1)}k sqm` : `${v} sqm`);
      }
    }
    updateReadout(rainfall, 'lab-rainfall-value', v => `${v} mm`);
  });
  
  inundation.addEventListener('input', () => {
    // Physics Guardrail: If rain > 150, lock inundation minimum.
    if (parseFloat(rainfall.value) > 150 && parseFloat(inundation.value) < 5000) {
      inundation.value = 5000;
    }
    updateReadout(inundation, 'lab-inundation-value', 
      v => v >= 1000 ? `${(v / 1000).toFixed(1)}k sqm` : `${v} sqm`);
  });
  
  updateReadout(rainfall, 'lab-rainfall-value', v => `${v} mm`);
  updateReadout(inundation, 'lab-inundation-value', 
    v => v >= 1000 ? `${(v / 1000).toFixed(1)}k sqm` : `${v} sqm`);
}

function setToggleValue(field, value, btn) {
  // Deactivate siblings
  const parent = btn.parentElement;
  parent.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  
  if (field === 'flood_occurrence') state.flood_occurrence = value;
  if (field === 'is_good_to_live') state.is_good_to_live = value;
}

// ══════════════════════════════════════════════════════════ SIMULATOR RUN ══
async function runLabSimulation() {
  const district = state.selectedDistrict;
  if (!district) {
    alert('Please select a district first.');
    return;
  }
  
  const btnText = document.getElementById('lab-btn-text');
  const btnLoader = document.getElementById('lab-btn-loader');
  const btn = document.getElementById('lab-predict-btn');
  
  btnText.style.display = 'none';
  btnLoader.style.display = 'block';
  btn.disabled = true;
  
  const payload = {
    district,
    rainfall_7d_mm: parseFloat(document.getElementById('lab-rainfall-slider').value),
    inundation_area_sqm: parseFloat(document.getElementById('lab-inundation-slider').value),
    flood_occurrence_current_event: state.flood_occurrence,
    is_good_to_live: state.is_good_to_live,
    reason_not_good_to_live: document.getElementById('lab-reason-select').value
  };
  
  try {
    const resp = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!resp.ok) {
      const err = await resp.json();
      alert(`Simulation failed: ${err.detail}`);
      return;
    }
    
    const result = await resp.json();
    state.lastPredictionId = result.prediction_id;
    state.lastPredictionData = { payload, result };
    
    // Update gauge & metrics
    updateLabGauge(result.risk_score, result.risk_level);
    document.getElementById('lab-result-subtitle').innerHTML = `Simulated Risk: <strong>${(result.risk_score * 100).toFixed(2)}%</strong> (${result.risk_level}) <span style="color:var(--text-muted);font-size:0.75rem;">in ${result.latency_ms}ms</span>`;
    
    if (result.briefing) {
      document.getElementById('lab-briefing-text').textContent = result.briefing;
    } else {
      document.getElementById('lab-briefing-text').textContent = `Risk analysis computed. Score: ${result.risk_score.toFixed(4)} (${result.risk_level}).`;
    }
    
    // Warnings
    const warnEl = document.getElementById('lab-warnings');
    if (result.warnings?.length > 0) {
      warnEl.style.display = 'block';
      warnEl.innerHTML = result.warnings.map(w => `<div>⚠ ${w}</div>`).join('');
    } else {
      warnEl.style.display = 'none';
    }
    
    // Enable actions
    document.getElementById('lab-feedback-row').style.display = 'flex';
    document.getElementById('lab-save-btn').disabled = false;
    document.getElementById('lab-report-btn').disabled = false;
    
  } catch (err) {
    console.error('[Lab Simulation] Error:', err);
    alert('Failed to connect to backend.');
  } finally {
    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
    btn.disabled = false;
  }
}

function updateLabGauge(score, level) {
  const arcLen = 251;
  const arc = document.getElementById('lab-gauge-arc');
  const needle = document.getElementById('lab-gauge-needle');
  const scoreEl = document.getElementById('lab-gauge-score');
  const badge = document.getElementById('lab-risk-badge');
  
  const colors = { LOW: '#22c55e', MEDIUM: '#eab308', HIGH: '#f97316', EXTREME: '#ef4444' };
  
  arc.style.strokeDashoffset = arcLen - (score * arcLen);
  arc.style.stroke = colors[level] || '#22d3ee';
  
  const angle = -180 + score * 180;
  const rad = angle * Math.PI / 180;
  needle.setAttribute('cx', 100 + 80 * Math.cos(rad));
  needle.setAttribute('cy', 100 + 80 * Math.sin(rad));
  
  scoreEl.textContent = score.toFixed(4);
  scoreEl.style.color = colors[level] || '#22d3ee';
  
  badge.textContent = level;
  badge.className = `risk-badge modal-risk-badge ${level}`;
}

async function submitLabFeedback(type) {
  if (!state.lastPredictionId) return;
  try {
    await fetch(`${API_BASE}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction_id: state.lastPredictionId, feedback_type: type })
    });
    const id = type === 'accurate' ? 'lab-btn-thumbup' : 'lab-btn-thumbdown';
    const btn = document.getElementById(id);
    btn.style.transform = 'scale(1.4)';
    setTimeout(() => btn.style.transform = '', 500);
  } catch (err) {
    console.warn('[Lab Feedback] failed:', err);
  }
}

function downloadLabReport() {
  if (!state.lastPredictionId) return;
  window.open(`${API_BASE}/api/report/${encodeURIComponent(state.lastPredictionId)}`, '_blank');
}

// ══════════════════════════════════════════════════════════ SENSITIVITY SWEEP ══
async function runSensitivitySweep(districtName) {
  const rainfallPoints = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
  const sweepData = [];
  
  // Set up mock inputs
  const currentInundation = parseFloat(document.getElementById('lab-inundation-slider').value);
  const currentFlood = state.flood_occurrence;
  const currentLive = state.is_good_to_live;
  const currentReason = document.getElementById('lab-reason-select').value;
  
  // Generate predictions concurrently to optimize performance
  const promises = rainfallPoints.map(async (rain) => {
    // Apply guardrail to simulated runs
    let inundation = currentInundation;
    if (rain > 150 && inundation < 5000) {
      inundation = 5000;
    }
    
    const payload = {
      district: districtName,
      rainfall_7d_mm: rain,
      inundation_area_sqm: inundation,
      flood_occurrence_current_event: currentFlood,
      is_good_to_live: currentLive,
      reason_not_good_to_live: currentReason
    };
    
    try {
      const resp = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (resp.ok) {
        const d = await resp.json();
        return { rain, score: d.risk_score };
      }
    } catch (_) {}
    return { rain, score: 0 };
  });
  
  const results = await Promise.all(promises);
  results.sort((a,b) => a.rain - b.rain);
  plotSweepChart(results);
}

function plotSweepChart(data) {
  const line = document.getElementById('sweep-line');
  const area = document.getElementById('sweep-area');
  const dotsG = document.getElementById('sweep-dots');
  
  if (!line || !area || !dotsG) return;
  dotsG.innerHTML = '';
  
  const width = 440; // 480 - 40
  const height = 150; // 170 - 20
  
  const getX = (rain) => 40 + (rain / 500) * width;
  const getY = (score) => 170 - score * height;
  
  let pathStr = '';
  let areaStr = `M ${getX(0)} 170`;
  
  data.forEach((pt, idx) => {
    const x = getX(pt.rain);
    const y = getY(pt.score);
    
    if (idx === 0) {
      pathStr += `M ${x} ${y}`;
    } else {
      pathStr += ` L ${x} ${y}`;
    }
    areaStr += ` L ${x} ${y}`;
    
    // Add interaction dot
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('cx', x);
    dot.setAttribute('cy', y);
    dot.setAttribute('r', 4);
    dot.setAttribute('fill', 'var(--accent-cyan)');
    dot.setAttribute('stroke', '#050d1a');
    dot.setAttribute('stroke-width', 1.5);
    dot.style.cursor = 'pointer';
    dot.style.transition = 'r 0.15s';
    
    dot.addEventListener('mouseenter', (e) => {
      dot.setAttribute('r', 7);
      dot.setAttribute('fill', '#fff');
      showSweepTooltip(e, pt.rain, pt.score);
    });
    dot.addEventListener('mouseleave', () => {
      dot.setAttribute('r', 4);
      dot.setAttribute('fill', 'var(--accent-cyan)');
      hideSweepTooltip();
    });
    
    dotsG.appendChild(dot);
  });
  
  areaStr += ` L ${getX(500)} 170 Z`;
  
  line.setAttribute('d', pathStr);
  area.setAttribute('d', areaStr);
}

function clearSweepChart() {
  document.getElementById('sweep-line')?.setAttribute('d', 'M 40 170');
  document.getElementById('sweep-area')?.setAttribute('d', 'M 40 170');
  const dotsG = document.getElementById('sweep-dots');
  if (dotsG) dotsG.innerHTML = '';
}

function showSweepTooltip(e, rain, score) {
  const tip = document.getElementById('sweep-tooltip');
  if (!tip) return;
  
  tip.style.display = 'block';
  tip.innerHTML = `<strong>${rain}mm Rain</strong><br>Simulated Risk: ${(score * 100).toFixed(1)}%`;
  
  const chartRect = e.target.ownerSVGElement.getBoundingClientRect();
  const x = e.clientX - chartRect.left + 10;
  const y = e.clientY - chartRect.top - 50;
  
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideSweepTooltip() {
  const tip = document.getElementById('sweep-tooltip');
  if (tip) tip.style.display = 'none';
}

// ══════════════════════════════════════════════════════════ COMPARATIVE SANDBOX ══
function saveToSandbox() {
  if (!state.lastPredictionData) return;
  
  const { payload, result } = state.lastPredictionData;
  const id = 'scenario-' + Date.now();
  const name = prompt("Name this scenario (e.g. Extreme Monsoon Kalutara):", `${payload.district} Simulation`);
  if (!name) return;
  
  state.savedScenarios.push({
    id,
    name,
    payload,
    result
  });
  
  renderSandboxList();
}

function renderSandboxList() {
  const listEl = document.getElementById('sandbox-card-list');
  if (state.savedScenarios.length === 0) {
    listEl.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 20px;">No saved scenarios. Save runs above to perform comparisons.</div>`;
    return;
  }
  
  listEl.innerHTML = '';
  state.savedScenarios.forEach(sc => {
    const item = document.createElement('div');
    item.className = 'sandbox-item';
    
    const isChecked = state.compareSelection.includes(sc.id);
    const scorePct = (sc.result.risk_score * 100).toFixed(0);
    
    item.innerHTML = `
      <div style="display:flex; align-items:center; gap:10px;">
        <input type="checkbox" id="chk-${sc.id}" ${isChecked ? 'checked' : ''} onchange="toggleCompare('${sc.id}')">
        <div>
          <div style="font-weight:700; font-size:0.85rem; color:var(--text-primary);">${sc.name}</div>
          <div style="font-size:0.72rem; color:var(--text-muted);">${sc.payload.district} | ${sc.payload.rainfall_7d_mm}mm Rain</div>
        </div>
      </div>
      <span class="risk-badge ${sc.result.risk_level}" style="font-size:0.7rem; padding: 2px 6px;">${scorePct}% ${sc.result.risk_level}</span>
    `;
    listEl.appendChild(item);
  });
}

function toggleCompare(id) {
  const idx = state.compareSelection.indexOf(id);
  if (idx > -1) {
    state.compareSelection.splice(idx, 1);
  } else {
    if (state.compareSelection.length >= 2) {
      alert("You can select a maximum of 2 scenarios to compare side-by-side.");
      // Uncheck the input
      document.getElementById(`chk-${id}`).checked = false;
      return;
    }
    state.compareSelection.push(id);
  }
  
  if (state.compareSelection.length === 2) {
    renderComparison();
  } else {
    document.getElementById('sandbox-comparison-panel').style.display = 'none';
  }
}

function renderComparison() {
  const sc1 = state.savedScenarios.find(s => s.id === state.compareSelection[0]);
  const sc2 = state.savedScenarios.find(s => s.id === state.compareSelection[1]);
  if (!sc1 || !sc2) return;
  
  const panel = document.getElementById('sandbox-comparison-panel');
  const grid = document.getElementById('comparison-grid-content');
  
  const formatCell = (val1, val2, isBetter = false) => {
    // Basic differential highlighting helper
    if (val1 === val2) return `<span>${val1}</span>`;
    return `<span style="color:var(--accent-cyan)">${val1}</span>`;
  };
  
  grid.innerHTML = `
    <div class="comparison-col highlight-cyan">
      <div style="font-weight:800; font-size:0.9rem; color:var(--accent-cyan); margin-bottom:10px;">A: ${sc1.name}</div>
      <div class="comparison-param-row"><span>District:</span><strong>${sc1.payload.district}</strong></div>
      <div class="comparison-param-row"><span>Rainfall:</span><strong>${sc1.payload.rainfall_7d_mm} mm</strong></div>
      <div class="comparison-param-row"><span>Inundation:</span><strong>${sc1.payload.inundation_area_sqm} sqm</strong></div>
      <div class="comparison-param-row"><span>Active Flood:</span><strong>${sc1.payload.flood_occurrence_current_event}</strong></div>
      <div class="comparison-param-row"><span>Safe to Live:</span><strong>${sc1.payload.is_good_to_live}</strong></div>
      <div class="comparison-param-row"><span>Sim Risk Score:</span><strong style="color:var(--accent-cyan);">${sc1.result.risk_score.toFixed(4)}</strong></div>
      <div class="comparison-param-row"><span>Risk Level:</span><span class="risk-badge ${sc1.result.risk_level}">${sc1.result.risk_level}</span></div>
    </div>
    
    <div class="comparison-col highlight-purple">
      <div style="font-weight:800; font-size:0.9rem; color:var(--accent-purple); margin-bottom:10px;">B: ${sc2.name}</div>
      <div class="comparison-param-row"><span>District:</span><strong>${sc2.payload.district}</strong></div>
      <div class="comparison-param-row"><span>Rainfall:</span><strong>${sc2.payload.rainfall_7d_mm} mm</strong></div>
      <div class="comparison-param-row"><span>Inundation:</span><strong>${sc2.payload.inundation_area_sqm} sqm</strong></div>
      <div class="comparison-param-row"><span>Active Flood:</span><strong>${sc2.payload.flood_occurrence_current_event}</strong></div>
      <div class="comparison-param-row"><span>Safe to Live:</span><strong>${sc2.payload.is_good_to_live}</strong></div>
      <div class="comparison-param-row"><span>Sim Risk Score:</span><strong style="color:var(--accent-purple);">${sc2.result.risk_score.toFixed(4)}</strong></div>
      <div class="comparison-param-row"><span>Risk Level:</span><span class="risk-badge ${sc2.result.risk_level}">${sc2.result.risk_level}</span></div>
    </div>
  `;
  
  panel.style.display = 'block';
}

function clearComparison() {
  state.compareSelection = [];
  renderSandboxList();
  document.getElementById('sandbox-comparison-panel').style.display = 'none';
}

// ══════════════════════════════════════════════════════════ HISTORICAL BACKTEST ══
function initDatePicker() {
  const el = document.getElementById('lab-historical-date');
  if (!el) return;
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  el.value = yesterday.toISOString().split('T')[0];
  el.min = '2020-01-01';
}

async function runLabBacktest() {
  const dateVal = document.getElementById('lab-historical-date').value;
  if (!dateVal) {
    alert("Please select a date first.");
    return;
  }
  
  const text = document.getElementById('lab-backtest-text');
  const loader = document.getElementById('lab-backtest-loader');
  const btn = document.getElementById('lab-backtest-btn');
  
  text.style.display = 'none';
  loader.style.display = 'block';
  btn.disabled = true;
  
  try {
    const resp = await fetch(`${API_BASE}/api/simulate/historical?date=${encodeURIComponent(dateVal)}`);
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Historical backtest failed');
    }
    
    const data = await resp.json();
    
    // Format into rows
    state.backtestResults = Object.entries(data.districts).map(([district, r]) => ({
      district,
      rainfall_7d_mm: r.rainfall_7d_mm,
      risk_score: r.risk_score,
      risk_level: r.risk_level
    }));
    
    // Compute stats
    computeBacktestStats(state.backtestResults);
    
    // Sort & Render
    sortAndRenderBacktest();
    
    // Enable export
    document.getElementById('lab-export-backtest-btn').disabled = false;
    
  } catch (err) {
    console.error('[Historical Backtest] Error:', err);
    alert(`Backtest execution failed: ${err.message}`);
  } finally {
    text.style.display = 'inline';
    loader.style.display = 'none';
    btn.disabled = false;
  }
}

function computeBacktestStats(results) {
  let counts = { LOW: 0, MEDIUM: 0, HIGH: 0, EXTREME: 0 };
  results.forEach(r => {
    if (counts[r.risk_level] !== undefined) {
      counts[r.risk_level]++;
    }
  });
  
  document.getElementById('stat-count-low').textContent = counts.LOW;
  document.getElementById('stat-count-medium').textContent = counts.MEDIUM;
  document.getElementById('stat-count-high').textContent = counts.HIGH;
  document.getElementById('stat-count-extreme').textContent = counts.EXTREME;
}

function sortAndRenderBacktest() {
  const sortVal = document.getElementById('lab-backtest-sort').value;
  const searchVal = document.getElementById('lab-backtest-search').value.toLowerCase();
  
  let filtered = state.backtestResults.filter(r => r.district.toLowerCase().includes(searchVal));
  
  if (sortVal === 'risk-desc') {
    filtered.sort((a,b) => b.risk_score - a.risk_score);
  } else if (sortVal === 'risk-asc') {
    filtered.sort((a,b) => a.risk_score - b.risk_score);
  } else if (sortVal === 'name-asc') {
    filtered.sort((a,b) => a.district.localeCompare(b.district));
  }
  
  const body = document.getElementById('backtest-table-body');
  if (filtered.length === 0) {
    body.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No matching districts found.</td></tr>`;
    return;
  }
  
  body.innerHTML = filtered.map(r => {
    const scorePct = (r.risk_score * 100).toFixed(0);
    return `
      <tr>
        <td style="font-weight:700; color:var(--text-primary);">${r.district}</td>
        <td>${r.rainfall_7d_mm.toFixed(1)} mm</td>
        <td>${r.risk_score.toFixed(4)}</td>
        <td><span class="risk-badge ${r.risk_level}">${scorePct}% ${r.risk_level}</span></td>
      </tr>
    `;
  }).join('');
}

function filterBacktestTable() {
  sortAndRenderBacktest();
}

function sortBacktestTable() {
  sortAndRenderBacktest();
}

function exportBacktestCSV() {
  if (state.backtestResults.length === 0) return;
  const dateVal = document.getElementById('lab-historical-date').value;
  const headers = ['district', 'rainfall_7d_mm', 'risk_score', 'risk_level'];
  const csv = [headers.join(',')].concat(
    state.backtestResults.map(r => [
      r.district,
      r.rainfall_7d_mm.toFixed(2),
      r.risk_score.toFixed(6),
      r.risk_level
    ].join(','))
  ).join('\n');
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { 
    href: url, 
    download: `flood_timeline_backtest_${dateVal}.csv` 
  });
  a.click();
  URL.revokeObjectURL(url);
}

// ══════════════════════════════════════════════════════════ BATCH RUNS ══
async function handleLabBatchUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  const text = await file.text();
  const lines = text.trim().split('\n');
  if (lines.length < 2) {
    alert("CSV file must have a header and at least one data row.");
    return;
  }
  
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const rows = [];
  
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const vals = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
    const obj = {};
    headers.forEach((h, idx) => { obj[h] = vals[idx] ?? ''; });
    if (!obj.district) continue;
    
    rows.push({
      district: obj.district,
      rainfall_7d_mm: parseFloat(obj.rainfall_7d_mm) || 50.0,
      inundation_area_sqm: parseFloat(obj.inundation_area_sqm) || 0.0,
      flood_occurrence_current_event: obj.flood_occurrence_current_event || 'No',
      is_good_to_live: obj.is_good_to_live || 'Yes',
      reason_not_good_to_live: obj.reason_not_good_to_live || 'None'
    });
  }
  
  if (rows.length === 0) {
    alert("No valid rows found in CSV. Make sure the column `district` exists.");
    return;
  }
  
  try {
    const resp = await fetch(`${API_BASE}/api/predict/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rows })
    });
    
    if (!resp.ok) throw new Error("Batch prediction API returned an error");
    const data = await resp.json();
    
    state.batchResults = data.results.map((r, idx) => ({
      row_id: idx + 1,
      district: r.district || rows[idx].district,
      rainfall_7d_mm: r.rainfall_7d_mm ?? rows[idx].rainfall_7d_mm,
      inundation_area_sqm: rows[idx].inundation_area_sqm,
      risk_score: r.risk_score,
      risk_level: r.risk_level,
      error: r.error || null
    }));
    
    renderBatchResults();
    document.getElementById('batch-results-card').style.display = 'block';
    
  } catch (err) {
    console.error('[Batch Prediction] Upload failed:', err);
    alert(`Batch execution failed: ${err.message}`);
  }
}

function renderBatchResults() {
  const searchVal = document.getElementById('lab-batch-search').value.toLowerCase();
  const body = document.getElementById('batch-results-body');
  
  let filtered = state.batchResults.filter(r => 
    r.district.toLowerCase().includes(searchVal) || 
    (r.risk_level && r.risk_level.toLowerCase().includes(searchVal))
  );
  
  if (filtered.length === 0) {
    body.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">No matching records found.</td></tr>`;
    return;
  }
  
  body.innerHTML = filtered.map(r => {
    if (r.error) {
      return `
        <tr>
          <td>#${r.row_id}</td>
          <td style="font-weight:700;color:var(--text-primary);">${r.district}</td>
          <td colspan="4" style="color:var(--risk-extreme)">Error: ${r.error}</td>
          <td><span style="color:var(--risk-extreme)">✗</span></td>
        </tr>
      `;
    }
    const scorePct = (r.risk_score * 100).toFixed(0);
    return `
      <tr>
        <td>#${r.row_id}</td>
        <td style="font-weight:700;color:var(--text-primary);">${r.district}</td>
        <td>${r.rainfall_7d_mm.toFixed(1)} mm</td>
        <td>${r.inundation_area_sqm.toLocaleString()} sqm</td>
        <td>${r.risk_score.toFixed(4)}</td>
        <td><span class="risk-badge ${r.risk_level}">${scorePct}% ${r.risk_level}</span></td>
        <td><span style="color:var(--risk-low);font-weight:bold;">✓</span></td>
      </tr>
    `;
  }).join('');
}

function filterBatchTable() {
  renderBatchResults();
}

function exportBatchResultsCSV() {
  if (state.batchResults.length === 0) return;
  const headers = ['row_id', 'district', 'rainfall_7d_mm', 'inundation_area_sqm', 'risk_score', 'risk_level', 'status'];
  const csv = [headers.join(',')].concat(
    state.batchResults.map(r => [
      r.row_id,
      r.district,
      r.rainfall_7d_mm.toFixed(2),
      r.inundation_area_sqm,
      r.error ? '' : r.risk_score.toFixed(6),
      r.error ? '' : r.risk_level,
      r.error ? `Error: ${r.error}` : 'Success'
    ].join(','))
  ).join('\n');
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { 
    href: url, 
    download: `flood_timeline_batch_results.csv` 
  });
  a.click();
  URL.revokeObjectURL(url);
}

function downloadSampleCSV() {
  const headers = ['district', 'rainfall_7d_mm', 'inundation_area_sqm', 'flood_occurrence_current_event', 'is_good_to_live', 'reason_not_good_to_live'];
  const rows = [
    ['Colombo', '180', '15000', 'Yes', 'No', 'Flood Risk'],
    ['Gampaha', '45', '0', 'No', 'Yes', 'None'],
    ['Kalutara', '220', '35000', 'Yes', 'No', 'Water Contamination']
  ];
  const csv = [headers.join(',')].concat(rows.map(r => r.join(','))).join('\n');
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { 
    href: url, 
    download: 'flood_timeline_batch_sample.csv' 
  });
  a.click();
  URL.revokeObjectURL(url);
}
