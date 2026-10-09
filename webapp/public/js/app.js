// =========================================================
// 🇮🇳 INDIAN SUPPLIER RISK PLATFORM - CLIENT APPLICATION
// Complete Interactive Suite: Single, Directory, Batch, Explainability, RAG, Retraining
// =========================================================

// Global State
let allSuppliers = [];
let lastScoredBatch = [];
let categoryChartInstance = null;
let regionChartInstance = null;
let importanceChartInstance = null;
let sensitivityChartInstance = null;
let livePredictEnabled = true;
let livePredictDebounceTimer = null;
let activeQuickFilter = 'all';

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  fetchInitialData();
  fetchIndexedPdfs();
  
  // Set initial labels
  syncParam('price_creep', 0.05);
  syncParam('rejection_pct', 0.02);
  syncParam('spike_ratio', 1.0);
  syncParam('stock_cover', 45);
  syncParam('days_expiry', 180);

  // Run initial evaluation
  runSinglePrediction();
});

// =========================================================
// 1. DATA INITIALIZATION & BENCHMARK METRICS
// =========================================================
async function fetchInitialData() {
  try {
    // 1. Fetch Indian Suppliers
    const resSuppliers = await fetch('/api/suppliers');
    const dataSuppliers = await resSuppliers.json();
    if (dataSuppliers.status === 'success') {
      allSuppliers = dataSuppliers.data;
      populateSupplierDropdown(allSuppliers);
      renderDirectoryTable(allSuppliers);
      populateCategoryAndRegionFilters(allSuppliers);
      renderCharts(allSuppliers);
      
      const headerTotal = document.getElementById('header-total-suppliers');
      if (headerTotal) headerTotal.innerText = `${allSuppliers.length} Indian Suppliers`;
      
      const kpiSuppliers = document.getElementById('kpi-suppliers');
      if (kpiSuppliers) kpiSuppliers.innerText = allSuppliers.length;
      
      const dirCount = document.getElementById('dir-count');
      if (dirCount) dirCount.innerText = allSuppliers.length;
      
      const singleSourceCount = allSuppliers.filter(s => s.is_single_source === 1).length;
      const kpiSingle = document.getElementById('kpi-single-source');
      if (kpiSingle) kpiSingle.innerText = singleSourceCount;
    }

    // 2. Fetch Metrics
    const resMetrics = await fetch('/api/metrics');
    const dataMetrics = await resMetrics.json();
    if (dataMetrics.status === 'success') {
      updateMetricsUI(dataMetrics.data);
    }
  } catch (err) {
    console.error("Failed to load initial data:", err);
  }
}

function updateMetricsUI(m) {
  const acc = (m.accuracy * 100).toFixed(1) + "%";
  const elHAcc = document.getElementById('header-accuracy');
  if (elHAcc) elHAcc.innerText = `Accuracy: ${acc}`;
  
  const elKAcc = document.getElementById('kpi-accuracy');
  if (elKAcc) elKAcc.innerText = acc;
  
  const elBAcc = document.getElementById('bench-accuracy');
  if (elBAcc) elBAcc.innerText = acc;
  
  const elBPrec = document.getElementById('bench-precision');
  if (elBPrec) elBPrec.innerText = (m.precision * 100).toFixed(1) + "%";
  
  const elBRec = document.getElementById('bench-recall');
  if (elBRec) elBRec.innerText = (m.recall * 100).toFixed(1) + "%";
  
  const elBF1 = document.getElementById('bench-f1');
  if (elBF1) elBF1.innerText = ((m.f1_score || 1.0) * 100).toFixed(1) + "%";
  
  const elBTime = document.getElementById('bench-timestamp');
  if (elBTime) elBTime.innerText = m.timestamp || 'N/A';
  
  const elBSamp = document.getElementById('bench-samples');
  if (elBSamp && m.train_samples) {
    elBSamp.innerText = `${m.train_samples} Train / ${m.val_samples} Val`;
  }
}

// =========================================================
// 2. TAB SWITCHING (All 6 Tabs Supported)
// =========================================================
function switchTab(tabKey) {
  const tabs = ['single', 'directory', 'batch', 'explain', 'rag', 'retrain'];
  tabs.forEach(t => {
    const btn = document.getElementById(`tab-btn-${t}`);
    const content = document.getElementById(`tab-content-${t}`);
    if (btn && content) {
      if (t === tabKey) {
        btn.classList.add('active');
        btn.classList.remove('text-slate-400');
        content.classList.remove('hidden');
      } else {
        btn.classList.remove('active');
        btn.classList.add('text-slate-400');
        content.classList.add('hidden');
      }
    }
  });

  // Tab specific initializers
  if (tabKey === 'explain') {
    fetchFeatureImportance();
    runSensitivitySimulation();
  }
  lucide.createIcons();
}

// =========================================================
// 3. TAB 1: SINGLE EVALUATION & REAL-TIME INFERENCE
// =========================================================
function toggleLivePredict(checked) {
  livePredictEnabled = checked;
  if (livePredictEnabled) {
    runSinglePrediction();
  }
}

function syncParam(param, val, fromInput = false) {
  const slider = document.getElementById(`slider_${param}`);
  const input = document.getElementById(`input_${param}`);
  const label = document.getElementById(`label_${param}`);

  const numVal = parseFloat(val);

  if (fromInput && slider) {
    slider.value = numVal;
  } else if (!fromInput && input) {
    input.value = numVal;
  }

  if (label) {
    if (param === 'price_creep') {
      label.innerText = numVal.toFixed(4);
    } else if (param === 'rejection_pct') {
      label.innerText = `${numVal.toFixed(4)} (${(numVal * 100).toFixed(1)}%)`;
    } else if (param === 'spike_ratio') {
      label.innerText = `${numVal.toFixed(2)}x`;
    } else if (param === 'stock_cover') {
      label.innerText = `${numVal.toFixed(1)} days`;
    } else if (param === 'days_expiry') {
      label.innerText = `${parseInt(numVal)} days`;
    }
  }

  onParameterChanged();
}

function onParameterChanged() {
  if (!livePredictEnabled) return;
  if (livePredictDebounceTimer) clearTimeout(livePredictDebounceTimer);
  livePredictDebounceTimer = setTimeout(() => {
    runSinglePrediction();
  }, 180);
}

function populateSupplierDropdown(suppliers) {
  const select = document.getElementById('supplier-dropdown-select');
  if (!select) return;
  select.innerHTML = '<option value="">-- Choose an Indian Supplier to Auto-Populate --</option>';
  suppliers.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.supplier_id;
    opt.innerText = `🇮🇳 ${s.supplier_id} | ${s.supplier_name} (${s.region})`;
    select.appendChild(opt);
  });
}

function autoFillSupplier(supplierId) {
  if (!supplierId) return;
  const s = allSuppliers.find(item => item.supplier_id === supplierId);
  if (!s) return;

  syncParam('price_creep', s.avg_price_creep);
  syncParam('rejection_pct', s.rec_rejection_pct);
  syncParam('spike_ratio', s.rejection_spike_ratio);
  syncParam('stock_cover', s.stock_cover_days);
  syncParam('days_expiry', s.min_days_to_expiry);

  const crit = document.getElementById('input_criticality');
  if (crit) crit.value = Math.min(Math.max(parseInt(s.criticality_numeric), 1), 5);

  const radios = document.getElementsByName('single_source');
  radios.forEach(r => {
    r.checked = (parseInt(r.value) === parseInt(s.is_single_source));
  });

  runSinglePrediction();
}

function loadPreset(type) {
  if (type === 'high_risk') {
    syncParam('price_creep', 0.85);
    syncParam('rejection_pct', 0.35);
    syncParam('spike_ratio', 4.5);
    syncParam('stock_cover', 8.0);
    syncParam('days_expiry', 25);
    document.getElementById('input_criticality').value = 5;
    document.getElementsByName('single_source')[1].checked = true;
  } else if (type === 'stable') {
    syncParam('price_creep', 0.02);
    syncParam('rejection_pct', 0.005);
    syncParam('spike_ratio', 1.0);
    syncParam('stock_cover', 65.0);
    syncParam('days_expiry', 240);
    document.getElementById('input_criticality').value = 1;
    document.getElementsByName('single_source')[0].checked = true;
  } else if (type === 'strain') {
    syncParam('price_creep', 0.28);
    syncParam('rejection_pct', 0.04);
    syncParam('spike_ratio', 2.2);
    syncParam('stock_cover', 18.0);
    syncParam('days_expiry', 60);
    document.getElementById('input_criticality').value = 3;
    document.getElementsByName('single_source')[1].checked = true;
  } else {
    // Reset
    syncParam('price_creep', 0.05);
    syncParam('rejection_pct', 0.02);
    syncParam('spike_ratio', 1.0);
    syncParam('stock_cover', 45);
    syncParam('days_expiry', 180);
    document.getElementById('input_criticality').value = 2;
    document.getElementsByName('single_source')[0].checked = true;
    const sel = document.getElementById('supplier-dropdown-select');
    if (sel) sel.value = "";
  }

  runSinglePrediction();
}

async function runSinglePrediction() {
  const btn = document.getElementById('btn-predict');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-2">●</span> Evaluating XGBoost...`;
  }

  const singleRadio = Array.from(document.getElementsByName('single_source')).find(r => r.checked);
  const singleSourceVal = singleRadio ? singleRadio.value : "0";

  const payload = {
    avg_price_creep: parseFloat(document.getElementById('input_price_creep').value) || 0.0,
    rec_rejection_pct: parseFloat(document.getElementById('input_rejection_pct').value) || 0.0,
    rejection_spike_ratio: parseFloat(document.getElementById('input_spike_ratio').value) || 1.0,
    min_days_to_expiry: parseInt(document.getElementById('input_days_expiry').value) || 180,
    criticality_numeric: parseFloat(document.getElementById('input_criticality').value) || 2.0,
    stock_cover_days: parseFloat(document.getElementById('input_stock_cover').value) || 45.0,
    is_single_source: parseInt(singleSourceVal)
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderPredictionResult(data, payload);
  } catch (err) {
    console.error("Prediction error:", err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="zap" class="w-4 h-4"></i> Run XGBoost Risk Evaluation`;
      lucide.createIcons();
    }
  }
}

function renderPredictionResult(data, input) {
  const isHighRisk = data.is_high_risk;
  const riskProb = data.risk_probability !== undefined ? data.risk_probability : (isHighRisk ? 95.0 : 5.0);

  const badge = document.getElementById('result-badge');
  const banner = document.getElementById('verdict-banner');
  const icon = document.getElementById('verdict-icon');
  const title = document.getElementById('verdict-title');
  const desc = document.getElementById('verdict-desc');
  const meter = document.getElementById('risk-meter-bar');
  const meterText = document.getElementById('risk-score-text');

  if (meter) meter.style.width = `${Math.min(Math.max(riskProb, 2), 100)}%`;
  if (meterText) meterText.innerText = `${riskProb}%`;

  if (isHighRisk) {
    if (badge) {
      badge.className = "px-3 py-1 rounded-full text-xs font-semibold badge-risk";
      badge.innerText = "CRITICAL RISK";
    }
    if (banner) banner.className = "p-4 rounded-xl bg-rose-950/30 border border-rose-500/40 mb-5";
    if (icon) icon.innerText = "🚨";
    if (title) {
      title.className = "text-base font-bold text-rose-400";
      title.innerText = "CRITICAL INDIAN SUPPLIER FAILURE RISK";
    }
    if (desc) {
      desc.className = "text-xs text-rose-200 mt-1";
      desc.innerText = "The model classifies this Indian vendor as HIGH RISK. Severe failure vectors detected across contract cost escalation, low buffer runway, or sole-source exposure.";
    }
    if (meter) meter.className = "bg-rose-500 h-3 rounded-full transition-all duration-500";
    if (meterText) meterText.className = "text-rose-400 font-bold";
  } else {
    if (badge) {
      badge.className = "px-3 py-1 rounded-full text-xs font-semibold badge-safe";
      badge.innerText = "STABLE / LOW RISK";
    }
    if (banner) banner.className = "p-4 rounded-xl bg-emerald-950/30 border border-emerald-500/40 mb-5";
    if (icon) icon.innerText = "✅";
    if (title) {
      title.className = "text-base font-bold text-emerald-400";
      title.innerText = "SUPPLIER STABLE & HEALTHY";
    }
    if (desc) {
      desc.className = "text-xs text-emerald-200 mt-1";
      desc.innerText = "Operational and quality metrics demonstrate strong resilience and delivery reliability. No immediate intervention required.";
    }
    if (meter) meter.className = "bg-emerald-500 h-3 rounded-full transition-all duration-500";
    if (meterText) meterText.className = "text-emerald-400 font-bold";
  }

  // Mitigation Playbook
  const recList = document.getElementById('mitigation-list');
  if (recList) {
    recList.innerHTML = "";
    const recs = [];

    if (input.is_single_source === 1) {
      recs.push("Sole-Provider Fragility: Qualify a secondary backup provider in domestic industrial clusters (e.g. Pune/Bengaluru).");
    }
    if (input.stock_cover_days < 20) {
      recs.push("Buffer Runway Alert: Stock cover is under 20 days. Issue an emergency expedited shipment order.");
    }
    if (input.rec_rejection_pct > 0.05 || input.rejection_spike_ratio > 2.0) {
      recs.push("Quality Surge: Defect rate exceeds acceptable limits. Issue a Corrective Action Request (CAR) and audit factory lots.");
    }
    if (input.avg_price_creep > 0.25) {
      recs.push("Price Creep: Inflation exceeds 25%. Re-negotiate contract indexation and procurement hedge terms.");
    }
    if (recs.length === 0) {
      recs.push("Maintain routine quarterly monitoring. All metrics are within operational tolerances.");
    }

    recs.forEach(r => {
      const li = document.createElement('li');
      li.innerText = r;
      recList.appendChild(li);
    });
  }
}

// =========================================================
// 4. TAB 2: 210 INDIA DIRECTORY, SEARCH, CHARTS & ADD VENDOR
// =========================================================
function renderDirectoryTable(suppliers) {
  const tbody = document.getElementById('directory-tbody');
  if (!tbody) return;
  tbody.innerHTML = "";

  suppliers.forEach(s => {
    const isSingle = s.is_single_source === 1;
    const isHigh = s.target === 1 || (s.avg_price_creep > 0.25 || s.rec_rejection_pct > 0.10);
    const tr = document.createElement('tr');
    tr.className = "hover:bg-slate-800/40 transition-colors";
    tr.innerHTML = `
      <td class="py-3 px-4 font-bold text-orange-400">${s.supplier_id}</td>
      <td class="py-3 px-4 font-sans text-slate-200 font-medium">${s.supplier_name}</td>
      <td class="py-3 px-4 font-sans text-slate-400">${s.category}</td>
      <td class="py-3 px-4 font-sans text-slate-400 flex items-center gap-1.5"><i data-lucide="map-pin" class="w-3 h-3 text-orange-400"></i> ${s.region}</td>
      <td class="py-3 px-4 ${s.avg_price_creep > 0.25 ? 'text-rose-400 font-bold' : ''}">${parseFloat(s.avg_price_creep).toFixed(4)}</td>
      <td class="py-3 px-4 ${s.rec_rejection_pct > 0.05 ? 'text-amber-400' : ''}">${(parseFloat(s.rec_rejection_pct) * 100).toFixed(1)}%</td>
      <td class="py-3 px-4 ${s.stock_cover_days < 20 ? 'text-rose-400' : ''}">${parseFloat(s.stock_cover_days).toFixed(1)}d</td>
      <td class="py-3 px-4 font-sans">${isSingle ? '<span class="text-rose-400 font-semibold">⚠️ Single</span>' : '<span class="text-emerald-400">🛡️ Multi</span>'}</td>
      <td class="py-3 px-4 font-sans">
        <button onclick="inspectSupplier('${s.supplier_id}')" class="px-2.5 py-1 bg-orange-500/10 border border-orange-500/30 text-orange-400 rounded-md text-[11px] hover:bg-orange-500/20 transition-all font-semibold">Evaluate</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  lucide.createIcons();
}

function inspectSupplier(id) {
  switchTab('single');
  const sel = document.getElementById('supplier-dropdown-select');
  if (sel) sel.value = id;
  autoFillSupplier(id);
}

function populateCategoryAndRegionFilters(suppliers) {
  const catSelect = document.getElementById('dir-cat-filter');
  const regSelect = document.getElementById('dir-reg-filter');
  if (!catSelect || !regSelect) return;

  const categories = Array.from(new Set(suppliers.map(s => s.category))).sort();
  const regions = Array.from(new Set(suppliers.map(s => s.region))).sort();

  catSelect.innerHTML = '<option value="all">All Categories</option>';
  regSelect.innerHTML = '<option value="all">All Indian Hubs</option>';

  categories.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.innerText = c;
    catSelect.appendChild(opt);
  });

  regions.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r;
    opt.innerText = r;
    regSelect.appendChild(opt);
  });
}

function setQuickFilter(type) {
  activeQuickFilter = type;
  const chips = ['all', 'high', 'stable', 'single'];
  chips.forEach(c => {
    const btn = document.getElementById(`qf-${c}`);
    if (btn) {
      if (c === type) {
        btn.className = "px-3 py-1 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 font-medium";
      } else {
        btn.className = "px-3 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600 font-medium";
      }
    }
  });
  filterDirectory();
}

function filterDirectory() {
  const searchInput = document.getElementById('dir-search');
  const search = searchInput ? searchInput.value.toLowerCase() : '';
  const cat = document.getElementById('dir-cat-filter') ? document.getElementById('dir-cat-filter').value : 'all';
  const reg = document.getElementById('dir-reg-filter') ? document.getElementById('dir-reg-filter').value : 'all';

  const filtered = allSuppliers.filter(s => {
    const matchSearch = (s.supplier_name || '').toLowerCase().includes(search) || (s.supplier_id || '').toLowerCase().includes(search);
    const matchCat = (cat === 'all' || s.category === cat);
    const matchReg = (reg === 'all' || s.region === reg);

    let matchQuick = true;
    if (activeQuickFilter === 'high') {
      matchQuick = (s.target === 1 || s.avg_price_creep > 0.25 || s.rec_rejection_pct > 0.10);
    } else if (activeQuickFilter === 'stable') {
      matchQuick = !(s.target === 1 || s.avg_price_creep > 0.25 || s.rec_rejection_pct > 0.10);
    } else if (activeQuickFilter === 'single') {
      matchQuick = (s.is_single_source === 1);
    }

    return matchSearch && matchCat && matchReg && matchQuick;
  });

  const countEl = document.getElementById('dir-count');
  if (countEl) countEl.innerText = filtered.length;
  renderDirectoryTable(filtered);
}

function exportFilteredCSV() {
  const search = document.getElementById('dir-search').value.toLowerCase();
  const cat = document.getElementById('dir-cat-filter').value;
  const reg = document.getElementById('dir-reg-filter').value;

  const filtered = allSuppliers.filter(s => {
    const matchSearch = (s.supplier_name || '').toLowerCase().includes(search) || (s.supplier_id || '').toLowerCase().includes(search);
    const matchCat = (cat === 'all' || s.category === cat);
    const matchReg = (reg === 'all' || s.region === reg);
    return matchSearch && matchCat && matchReg;
  });

  if (filtered.length === 0) return alert("No records to export");

  const headers = Object.keys(filtered[0]).join(',');
  const rows = filtered.map(row => Object.values(row).map(v => `"${v}"`).join(',')).join('\n');
  const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "indian_suppliers_export.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function renderCharts(suppliers) {
  // Category Chart
  const catCounts = {};
  suppliers.forEach(s => {
    catCounts[s.category] = (catCounts[s.category] || 0) + 1;
  });

  const ctxCat = document.getElementById('categoryChart');
  if (ctxCat) {
    if (categoryChartInstance) categoryChartInstance.destroy();
    categoryChartInstance = new Chart(ctxCat.getContext('2d'), {
      type: 'bar',
      data: {
        labels: Object.keys(catCounts),
        datasets: [{
          label: 'Suppliers Count',
          data: Object.values(catCounts),
          backgroundColor: 'rgba(249, 115, 22, 0.4)',
          borderColor: '#f97316',
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }

  // Region Doughnut Chart
  const regCounts = {};
  suppliers.forEach(s => {
    const city = (s.region || '').split(',')[0].trim();
    if (city) regCounts[city] = (regCounts[city] || 0) + 1;
  });

  const ctxReg = document.getElementById('regionChart');
  if (ctxReg) {
    if (regionChartInstance) regionChartInstance.destroy();
    regionChartInstance = new Chart(ctxReg.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels: Object.keys(regCounts),
        datasets: [{
          data: Object.values(regCounts),
          backgroundColor: [
            '#f97316', '#3b82f6', '#10b981', '#a855f7', '#ec4899', '#eab308', '#06b6d4', '#64748b'
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } }
        }
      }
    });
  }
}

// Add Vendor Modal Controls
function openAddVendorModal() {
  const modal = document.getElementById('add-vendor-modal');
  if (modal) modal.classList.remove('hidden');
}

function closeAddVendorModal() {
  const modal = document.getElementById('add-vendor-modal');
  if (modal) modal.classList.add('hidden');
}

async function saveNewVendor() {
  const name = document.getElementById('new-v-name').value.trim();
  const cat = document.getElementById('new-v-cat').value;
  const hub = document.getElementById('new-v-hub').value.trim();
  const creep = parseFloat(document.getElementById('new-v-creep').value) || 0.05;
  const reject = parseFloat(document.getElementById('new-v-reject').value) || 0.02;
  const stock = parseFloat(document.getElementById('new-v-stock').value) || 45;
  const single = parseInt(document.getElementById('new-v-single').value) || 0;

  if (!name || !hub) {
    return alert("Please enter both Company Name and Manufacturing Hub.");
  }

  const payload = {
    supplier_name: name,
    category: cat,
    region: hub,
    avg_price_creep: creep,
    rec_rejection_pct: reject,
    rejection_spike_ratio: 1.0,
    min_days_to_expiry: 180,
    criticality_numeric: 2.0,
    stock_cover_days: stock,
    is_single_source: single
  };

  try {
    const res = await fetch('/api/suppliers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.status === 'success') {
      allSuppliers.unshift(result.data);
      closeAddVendorModal();
      populateSupplierDropdown(allSuppliers);
      renderDirectoryTable(allSuppliers);
      renderCharts(allSuppliers);
      document.getElementById('dir-count').innerText = allSuppliers.length;
      document.getElementById('header-total-suppliers').innerText = `${allSuppliers.length} Indian Suppliers`;
      if (typeof confetti === 'function') confetti();
      alert(`✅ New Indian Supplier Added: ${result.data.supplier_name} (${result.data.supplier_id})`);
    } else {
      alert("Error adding supplier: " + result.message);
    }
  } catch (err) {
    alert("Request error: " + err.message);
  }
}

// =========================================================
// 5. TAB 3: BATCH CSV SCORER & SIMULATOR
// =========================================================
async function scorePreloadedBatch() {
  const btn = document.getElementById('btn-score-preloaded');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-2">●</span> Scoring 210 Vendors...`;
  }

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records: allSuppliers })
    });
    const data = await res.json();
    if (data.status === 'success') {
      renderBatchResults(data);
      if (typeof confetti === 'function') confetti();
    } else {
      alert("Batch evaluation error: " + data.message);
    }
  } catch (e) {
    alert("Batch evaluation request failed: " + e.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="zap" class="w-4 h-4"></i> Score All 210 Indian Vendors`;
      lucide.createIcons();
    }
  }
}

async function generateRandomTestBatch() {
  const categories = ["Semiconductors & Chips", "Precision Machining", "Raw Metals & Alloys", "Chemicals & Resins", "Packaging & Polymers"];
  const cities = ["Bengaluru, Karnataka", "Pune, Maharashtra", "Chennai, Tamil Nadu", "Coimbatore, Tamil Nadu", "Ahmedabad, Gujarat", "Gurugram, Haryana"];
  
  const testRecords = [];
  for (let i = 1; i <= 15; i++) {
    const isSingle = Math.random() > 0.65 ? 1 : 0;
    const isStress = Math.random() > 0.5;
    testRecords.push({
      supplier_id: `SIM-IN-${1000 + i}`,
      supplier_name: `Simulated Enterprise India #${i}`,
      category: categories[Math.floor(Math.random() * categories.length)],
      region: cities[Math.floor(Math.random() * cities.length)],
      avg_price_creep: isStress ? Number((0.25 + Math.random() * 0.40).toFixed(4)) : Number((0.01 + Math.random() * 0.08).toFixed(4)),
      rec_rejection_pct: isStress ? Number((0.08 + Math.random() * 0.15).toFixed(4)) : Number((0.005 + Math.random() * 0.03).toFixed(4)),
      rejection_spike_ratio: isStress ? Number((2.0 + Math.random() * 3.0).toFixed(2)) : 1.0,
      stock_cover_days: isStress ? Math.floor(5 + Math.random() * 15) : Math.floor(35 + Math.random() * 50),
      min_days_to_expiry: Math.floor(40 + Math.random() * 200),
      criticality_numeric: Math.floor(1 + Math.random() * 5),
      is_single_source: isSingle
    });
  }

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records: testRecords })
    });
    const data = await res.json();
    if (data.status === 'success') {
      renderBatchResults(data);
      if (typeof confetti === 'function') confetti();
    }
  } catch (e) {
    alert("Simulation failed: " + e.message);
  }
}

async function handleBatchCsvUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('csv', file);

  try {
    const res = await fetch('/api/upload-csv-batch', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.status === 'success') {
      renderBatchResults(data);
      if (typeof confetti === 'function') confetti();
    } else {
      alert("CSV Scoring Failed: " + data.message);
    }
  } catch (err) {
    alert("Upload failed: " + err.message);
  } finally {
    e.target.value = '';
  }
}

function renderBatchResults(data) {
  lastScoredBatch = data.records || [];

  const metricsBar = document.getElementById('batch-metrics-bar');
  const tableContainer = document.getElementById('batch-table-container');
  const tbody = document.getElementById('batch-results-tbody');

  if (metricsBar) metricsBar.classList.remove('hidden');
  if (tableContainer) tableContainer.classList.remove('hidden');

  document.getElementById('bm-total').innerText = data.total_evaluated;
  document.getElementById('bm-high').innerText = data.high_risk_count;
  document.getElementById('bm-high-rate').innerText = `${data.risk_rate_pct}% Failure Rate`;
  document.getElementById('bm-stable').innerText = data.stable_count;

  if (tbody) {
    tbody.innerHTML = "";
    lastScoredBatch.forEach(r => {
      const isHigh = r.is_high_risk;
      const prob = r.risk_probability !== undefined ? r.risk_probability : (isHigh ? 95.0 : 5.0);
      const tr = document.createElement('tr');
      tr.className = "hover:bg-slate-800/40 transition-colors";
      tr.innerHTML = `
        <td class="py-3 px-4 font-bold text-orange-400">${r.supplier_id || 'N/A'}</td>
        <td class="py-3 px-4 font-sans text-slate-200 font-medium">${r.supplier_name || 'Vendor'}</td>
        <td class="py-3 px-4 font-sans">${isHigh ? '<span class="px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30 font-bold text-[11px]">🚨 High Risk</span>' : '<span class="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-bold text-[11px]">✅ Stable</span>'}</td>
        <td class="py-3 px-4 font-mono font-bold ${isHigh ? 'text-rose-400' : 'text-emerald-400'}">${prob}%</td>
        <td class="py-3 px-4">${parseFloat(r.avg_price_creep || 0).toFixed(4)}</td>
        <td class="py-3 px-4">${(parseFloat(r.rec_rejection_pct || 0) * 100).toFixed(1)}%</td>
        <td class="py-3 px-4">${parseFloat(r.stock_cover_days || 0).toFixed(1)}d</td>
        <td class="py-3 px-4 font-sans">${r.is_single_source === 1 ? '<span class="text-rose-400">⚠️ Single</span>' : '<span class="text-emerald-400">🛡️ Multi</span>'}</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function downloadScoredBatchCSV() {
  if (!lastScoredBatch || lastScoredBatch.length === 0) {
    return alert("No batch results to export. Run a batch evaluation first.");
  }

  const headers = Object.keys(lastScoredBatch[0]).join(',');
  const rows = lastScoredBatch.map(row => Object.values(row).map(v => `"${v}"`).join(',')).join('\n');
  const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `scored_suppliers_report_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// =========================================================
// 6. TAB 4: MODEL EXPLAINABILITY & WHAT-IF SENSITIVITY
// =========================================================
async function fetchFeatureImportance() {
  try {
    const res = await fetch('/api/feature-importance');
    const data = await res.json();
    if (data.status === 'success') {
      renderImportanceChart(data.features);
    }
  } catch (e) {
    console.error("Feature importance error:", e);
  }
}

function renderImportanceChart(features) {
  const ctx = document.getElementById('importanceChart');
  if (!ctx) return;

  const labels = features.map(f => f.feature);
  const values = features.map(f => f.percentage);

  if (importanceChartInstance) importanceChartInstance.destroy();
  importanceChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Weight Contribution (%)',
        data: values,
        backgroundColor: [
          '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e'
        ],
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Contribution: ${ctx.raw}%`
          }
        }
      },
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#cbd5e1', font: { family: 'monospace', size: 11 } }, grid: { display: false } }
      }
    }
  });
}

async function runSensitivitySimulation() {
  const select = document.getElementById('sim-feature-select');
  const featureName = select ? select.value : 'avg_price_creep';

  const singleRadio = Array.from(document.getElementsByName('single_source')).find(r => r.checked);
  const singleSourceVal = singleRadio ? singleRadio.value : "0";

  const baseData = {
    avg_price_creep: parseFloat(document.getElementById('input_price_creep').value) || 0.05,
    rec_rejection_pct: parseFloat(document.getElementById('input_rejection_pct').value) || 0.02,
    rejection_spike_ratio: parseFloat(document.getElementById('input_spike_ratio').value) || 1.0,
    min_days_to_expiry: parseInt(document.getElementById('input_days_expiry').value) || 180,
    criticality_numeric: parseFloat(document.getElementById('input_criticality').value) || 2.0,
    stock_cover_days: parseFloat(document.getElementById('input_stock_cover').value) || 45.0,
    is_single_source: parseInt(singleSourceVal)
  };

  try {
    const res = await fetch('/api/sensitivity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature_name: featureName, base_data: baseData })
    });
    const data = await res.json();
    if (data.status === 'success') {
      renderSensitivityChart(data.curve, featureName);
    }
  } catch (e) {
    console.error("Sensitivity error:", e);
  }
}

function renderSensitivityChart(curve, featureName) {
  const ctx = document.getElementById('sensitivityChart');
  if (!ctx) return;

  const labels = curve.map(c => c.val);
  const data = curve.map(c => c.risk_prob);

  if (sensitivityChartInstance) sensitivityChartInstance.destroy();
  sensitivityChartInstance = new Chart(ctx.getContext('2d'), {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Failure Risk Probability (%)',
        data: data,
        borderColor: '#a855f7',
        backgroundColor: 'rgba(168, 85, 247, 0.15)',
        fill: true,
        tension: 0.3,
        pointBackgroundColor: '#a855f7',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          title: { display: true, text: featureName, color: '#94a3b8', font: { size: 10 } },
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { display: false }
        },
        y: {
          min: 0,
          max: 100,
          title: { display: true, text: 'Risk %', color: '#94a3b8', font: { size: 10 } },
          ticks: { color: '#94a3b8', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      }
    }
  });
}

// =========================================================
// 7. TAB 5: PDF UPLOAD & RAG AI KNOWLEDGE ASSISTANT
// =========================================================
async function handlePdfUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const btn = document.getElementById('btn-upload-pdf');
  const statusBanner = document.getElementById('pdf-upload-status');
  
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-1">●</span> Parsing & Indexing PDF...`;
  }
  
  if (statusBanner) {
    statusBanner.classList.remove('hidden');
    statusBanner.innerHTML = `<div class="text-blue-400">Extracting text & creating semantic vector chunks from <b>${file.name}</b>...</div>`;
  }

  const formData = new FormData();
  formData.append('pdf', file);

  try {
    const res = await fetch('/api/upload-pdf', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.status === 'success') {
      if (statusBanner) {
        statusBanner.innerHTML = `
          <div class="text-emerald-400 font-semibold">
            ✅ Successfully Ingested: <b>${data.filename}</b> (${data.pages} Pages, ${data.chunksCreated} Semantic Chunks Added to RAG Knowledge Base)
          </div>
        `;
      }
      fetchIndexedPdfs();
      if (typeof confetti === 'function') confetti();
    } else {
      if (statusBanner) statusBanner.innerHTML = `<div class="text-rose-400">❌ PDF upload error: ${data.message}</div>`;
    }
  } catch (err) {
    if (statusBanner) statusBanner.innerHTML = `<div class="text-rose-400">❌ Upload failed: ${err.message}</div>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="upload-cloud" class="w-4 h-4"></i> Upload PDF Document`;
      lucide.createIcons();
    }
    e.target.value = '';
  }
}

async function fetchIndexedPdfs() {
  try {
    const res = await fetch('/api/pdf-documents');
    const data = await res.json();
    const listContainer = document.getElementById('indexed-pdf-list');
    const kpiCount = document.getElementById('kpi-pdf-count');

    if (data.status === 'success' && data.data.length > 0) {
      if (kpiCount) kpiCount.innerText = `${data.data.length} PDFs`;
      if (listContainer) {
        listContainer.innerHTML = '';
        data.data.forEach(doc => {
          const item = document.createElement('div');
          item.className = "p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between text-xs";
          item.innerHTML = `
            <div class="flex items-center gap-3">
              <div class="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
                <i data-lucide="file-text" class="w-4 h-4"></i>
              </div>
              <div>
                <div class="font-bold text-slate-200">${doc.filename}</div>
                <div class="text-[11px] text-slate-400">${doc.chunks} Semantic Chunks • ${doc.sample}</div>
              </div>
            </div>
            <span class="px-2 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px]">
              Indexed for RAG
            </span>
          `;
          listContainer.appendChild(item);
        });
        lucide.createIcons();
      }
    } else {
      if (kpiCount) kpiCount.innerText = "0 PDFs";
      if (listContainer) {
        listContainer.innerHTML = '<p class="text-xs text-slate-500 italic">No external PDFs uploaded yet. Upload your first PDF audit/contract report above.</p>';
      }
    }
  } catch (e) {
    console.error("Failed to fetch indexed PDFs:", e);
  }
}

function setRagQuery(q) {
  document.getElementById('rag-input').value = q;
  submitRagQuery();
}

async function submitRagQuery() {
  const query = document.getElementById('rag-input').value.trim();
  if (!query) return;

  const btn = document.getElementById('btn-rag-submit');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-1">●</span> Inquiring...`;
  }

  try {
    const res = await fetch('/api/rag-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 4 })
    });
    const data = await res.json();
    if (data.status === 'success') {
      renderRagResults(data);
    } else {
      alert("RAG query failed: " + data.message);
    }
  } catch (err) {
    alert("RAG request failed: " + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="send" class="w-4 h-4"></i> Inquire`;
      lucide.createIcons();
    }
  }
}

function renderRagResults(data) {
  const container = document.getElementById('rag-output-container');
  const answerBody = document.getElementById('rag-answer-body');
  const citationsList = document.getElementById('rag-citations-list');

  if (container) container.classList.remove('hidden');

  const results = data.results || [];
  const supplierMatches = results.filter(r => r.type === 'supplier');
  const pdfMatches = results.filter(r => r.type === 'pdf_document');

  let htmlAnswer = `<p>Based on semantic analysis across <b>${allSuppliers.length} Indian vendors</b> and <b>uploaded PDF audit reports</b>, here are the findings for: <i>"${data.query}"</i></p>`;
  
  if (supplierMatches.length > 0) {
    htmlAnswer += `<div class="mt-2 text-orange-400 font-semibold text-xs">🇮🇳 Relevant Indian Suppliers:</div>`;
    supplierMatches.forEach(r => {
      const isRisk = r.risk_pred === 1;
      const badge = isRisk ? '<span class="text-rose-400 font-bold">🚨 HIGH RISK</span>' : '<span class="text-emerald-400 font-bold">✅ STABLE</span>';
      htmlAnswer += `
        <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-xs">
          ${badge} <b>${r.name} (${r.id})</b> — ${r.category} (${r.region}) | 
          Stock: <b>${r.stock_cover}d</b> | Price Creep: <b>${parseFloat(r.price_creep).toFixed(2)}</b>
        </div>
      `;
    });
  }

  if (pdfMatches.length > 0) {
    htmlAnswer += `<div class="mt-2 text-indigo-400 font-semibold text-xs">📄 Insights from Uploaded PDF Contracts/Audits:</div>`;
    pdfMatches.forEach(p => {
      htmlAnswer += `
        <div class="p-2.5 rounded-lg bg-indigo-950/30 border border-indigo-500/30 text-xs">
          <b>📄 Document: ${p.filename}</b> (Chunk ${p.chunk_id}): "${p.doc_text.substring(0, 150)}..."
        </div>
      `;
    });
  }

  htmlAnswer += `
    <div class="mt-2 text-indigo-300 font-semibold">📌 Strategic Mitigations:</div>
    <ul class="list-disc list-inside space-y-1 text-slate-300">
      <li>Prioritize domestic backup qualification in hub clusters (Bengaluru, Pune, Chennai) for flagged sole providers.</li>
      <li>Cross-verify defect findings against contractual SLA penalty clauses extracted from uploaded agreements.</li>
    </ul>
  `;

  if (answerBody) answerBody.innerHTML = htmlAnswer;

  // Render Citations
  if (citationsList) {
    citationsList.innerHTML = "";
    results.forEach((r, idx) => {
      const card = document.createElement('div');
      card.className = "p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-400 font-mono";
      
      if (r.type === 'supplier') {
        card.innerHTML = `
          <div class="flex justify-between items-center mb-1 text-slate-300 font-sans font-semibold">
            <span>🇮🇳 Supplier Citation ${idx + 1}: ${r.name} (${r.id})</span>
            <span class="text-orange-400 text-[11px]">Score: ${r.score}</span>
          </div>
          <div>${r.doc_text}</div>
        `;
      } else {
        card.innerHTML = `
          <div class="flex justify-between items-center mb-1 text-indigo-300 font-sans font-semibold">
            <span>📄 PDF Document Citation ${idx + 1}: ${r.filename} (Chunk ${r.chunk_id})</span>
            <span class="text-indigo-400 text-[11px]">Score: ${r.score}</span>
          </div>
          <div>${r.doc_text}</div>
        `;
      }
      citationsList.appendChild(card);
    });
  }
}

// =========================================================
// 8. TAB 6: AUTOMATED RETRAINING LAB & MODEL JSON HUB
// =========================================================
async function triggerRetrain() {
  const btn = document.getElementById('btn-retrain');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-2">●</span> Retraining on India Dataset...`;
  }

  const n_estimators = parseInt(document.getElementById('slider-trees').value) || 120;
  const max_depth = parseInt(document.getElementById('slider-depth').value) || 4;
  const learning_rate = parseFloat(document.getElementById('slider-lr').value) || 0.10;

  try {
    const res = await fetch('/api/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_estimators, max_depth, learning_rate })
    });
    const data = await res.json();
    if (data.status === 'success' && data.metrics) {
      updateMetricsUI(data.metrics);
      if (typeof confetti === 'function') confetti();
      alert(`🎉 XGBoost Retraining Completed Successfully!
Accuracy: ${(data.metrics.accuracy * 100).toFixed(1)}% | Precision: ${(data.metrics.precision * 100).toFixed(1)}% | Recall: ${(data.metrics.recall * 100).toFixed(1)}%
Model serialized & synchronized to xgb_supplier_risk_model.json.`);
    } else {
      alert("Retraining response: " + data.message);
    }
  } catch (err) {
    alert("Retraining request failed: " + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="play" class="w-4 h-4"></i> Execute Pipeline Retraining`;
      lucide.createIcons();
    }
  }
}

async function inspectModelJson() {
  try {
    const res = await fetch('/api/model-json');
    const data = await res.json();
    if (data.status === 'success') {
      const msg = `📄 XGBoost Model JSON Specification:
• File: ${data.filename}
• Size: ${data.sizeKB} (${data.sizeBytes} bytes)
• Objective: ${typeof data.learner === 'object' ? data.learner.objective : 'XGBoost Standard'}
• Features: ${typeof data.learner === 'object' && data.learner.feature_names ? data.learner.feature_names.join(', ') : '7 Features'}
• Path: ${data.path}`;
      alert(msg);
    } else {
      alert("Error inspecting model JSON: " + data.message);
    }
  } catch (e) {
    alert("Failed to fetch model info: " + e.message);
  }
}
