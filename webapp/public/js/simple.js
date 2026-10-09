// =========================================================
// SIMPLE MODE CONTROLLER
// Clean, fast, zero-clutter logic for simple.html
// =========================================================

let simpleSuppliers = [];
let debounceTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  loadSimpleInitialData();
  runSimpleCheck();
});

async function loadSimpleInitialData() {
  try {
    const res = await fetch('/api/suppliers');
    const data = await res.json();
    if (data.status === 'success') {
      simpleSuppliers = data.data;
      const select = document.getElementById('simple-vendor-select');
      select.innerHTML = '<option value="">-- Choose Indian Supplier (210 available) --</option>';
      simpleSuppliers.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.supplier_id;
        opt.innerText = `🇮🇳 ${s.supplier_name} (${s.region})`;
        select.appendChild(opt);
      });
    }

    const resM = await fetch('/api/metrics');
    const dataM = await resM.json();
    if (dataM.status === 'success' && dataM.data.accuracy) {
      document.getElementById('simple-acc-score').innerText = (dataM.data.accuracy * 100).toFixed(1) + "%";
    }
  } catch (e) {
    console.error("Initial load error:", e);
  }
}

function updateSimpleValue(type, val) {
  const num = parseFloat(val);
  if (type === 'creep') {
    document.getElementById('lbl-creep').innerText = (num * 100).toFixed(1) + "%";
  } else if (type === 'reject') {
    document.getElementById('lbl-reject').innerText = (num * 100).toFixed(1) + "%";
  } else if (type === 'stock') {
    document.getElementById('lbl-stock').innerText = Math.round(num) + " days";
  }

  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    runSimpleCheck();
  }, 150);
}

function autoFillSimple(id) {
  if (!id) return;
  const s = simpleSuppliers.find(x => x.supplier_id === id);
  if (!s) return;

  document.getElementById('sim-creep').value = s.avg_price_creep;
  document.getElementById('lbl-creep').innerText = (s.avg_price_creep * 100).toFixed(1) + "%";

  document.getElementById('sim-reject').value = s.rec_rejection_pct;
  document.getElementById('lbl-reject').innerText = (s.rec_rejection_pct * 100).toFixed(1) + "%";

  document.getElementById('sim-stock').value = s.stock_cover_days;
  document.getElementById('lbl-stock').innerText = Math.round(s.stock_cover_days) + " days";

  document.getElementById('sim-single').checked = (s.is_single_source === 1);

  runSimpleCheck();
}

function setSimplePreset(type) {
  if (type === 'safe') {
    document.getElementById('sim-creep').value = 0.02;
    document.getElementById('lbl-creep').innerText = "2.0%";
    document.getElementById('sim-reject').value = 0.005;
    document.getElementById('lbl-reject').innerText = "0.5%";
    document.getElementById('sim-stock').value = 60;
    document.getElementById('lbl-stock').innerText = "60 days";
    document.getElementById('sim-single').checked = false;
  } else {
    document.getElementById('sim-creep').value = 0.75;
    document.getElementById('lbl-creep').innerText = "75.0%";
    document.getElementById('sim-reject').value = 0.30;
    document.getElementById('lbl-reject').innerText = "30.0%";
    document.getElementById('sim-stock').value = 7;
    document.getElementById('lbl-stock').innerText = "7 days";
    document.getElementById('sim-single').checked = true;
  }
  document.getElementById('simple-vendor-select').value = "";
  runSimpleCheck();
}

async function runSimpleCheck() {
  const creep = parseFloat(document.getElementById('sim-creep').value) || 0.05;
  const reject = parseFloat(document.getElementById('sim-reject').value) || 0.02;
  const stock = parseFloat(document.getElementById('sim-stock').value) || 45;
  const single = document.getElementById('sim-single').checked ? 1 : 0;

  const payload = {
    avg_price_creep: creep,
    rec_rejection_pct: reject,
    rejection_spike_ratio: 1.0,
    min_days_to_expiry: 180,
    criticality_numeric: 2.0,
    stock_cover_days: stock,
    is_single_source: single
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderSimpleResult(data);
  } catch (e) {
    console.error("Simple check failed:", e);
  }
}

function renderSimpleResult(data) {
  const box = document.getElementById('simple-verdict-box');
  const icon = document.getElementById('simple-icon');
  const title = document.getElementById('simple-verdict-title');
  const desc = document.getElementById('simple-verdict-desc');
  const score = document.getElementById('simple-score');

  const isRisk = data.is_high_risk;
  const prob = data.risk_probability !== undefined ? data.risk_probability : (isRisk ? 95.0 : 5.0);

  score.innerText = `${prob}%`;

  if (isRisk) {
    box.className = "p-5 rounded-2xl bg-rose-950/40 border border-rose-500/50 transition-all";
    icon.innerText = "🚨";
    title.className = "text-lg font-bold text-rose-400";
    title.innerText = "HIGH RISK SUPPLIER";
    desc.className = "text-xs text-rose-200 mt-0.5";
    desc.innerText = "Failure risk detected due to price escalation, low inventory runway, or sole-source fragility.";
    score.className = "text-2xl font-bold text-rose-400";
  } else {
    box.className = "p-5 rounded-2xl bg-emerald-950/30 border border-emerald-500/40 transition-all";
    icon.innerText = "✅";
    title.className = "text-lg font-bold text-emerald-400";
    title.innerText = "STABLE / SAFE SUPPLIER";
    desc.className = "text-xs text-emerald-200 mt-0.5";
    desc.innerText = "Metrics are healthy. Low probability of disruption.";
    score.className = "text-2xl font-bold text-emerald-400";
  }
}

// RAG Search
function askSimpleQuestion(q) {
  document.getElementById('simple-rag-input').value = q;
  executeSimpleRag();
}

async function executeSimpleRag() {
  const query = document.getElementById('simple-rag-input').value.trim();
  if (!query) return;

  const btn = document.getElementById('btn-simple-rag');
  btn.disabled = true;
  btn.innerText = "Thinking...";

  try {
    const res = await fetch('/api/rag-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 3 })
    });
    const data = await res.json();
    const resultBox = document.getElementById('simple-rag-result');
    const textEl = document.getElementById('simple-rag-text');
    resultBox.classList.remove('hidden');

    if (data.status === 'success' && data.results && data.results.length > 0) {
      let html = `<p class="mb-2">Findings for <i>"${query}"</i>:</p><ul class="list-disc list-inside space-y-1">`;
      data.results.forEach(r => {
        if (r.type === 'supplier') {
          html += `<li><b>${r.name} (${r.region})</b>: Stock ${r.stock_cover}d, Price Creep ${(parseFloat(r.price_creep)*100).toFixed(1)}% — <span class="${r.risk_pred === 1 ? 'text-rose-400 font-bold' : 'text-emerald-400'}">${r.risk_pred === 1 ? '🚨 High Risk' : '✅ Stable'}</span></li>`;
        } else {
          html += `<li><b>PDF: ${r.filename}</b>: "${r.doc_text.substring(0, 100)}..."</li>`;
        }
      });
      html += `</ul>`;
      textEl.innerHTML = html;
    } else {
      textEl.innerText = "No specific supplier or document records matched your inquiry.";
    }
  } catch (e) {
    alert("Query error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="send" class="w-3.5 h-3.5"></i> Ask`;
    lucide.createIcons();
  }
}

// PDF Upload
async function handleSimplePdf(e) {
  const file = e.target.files[0];
  if (!file) return;

  const btn = document.getElementById('btn-simple-upload-pdf');
  const status = document.getElementById('simple-pdf-status');
  btn.disabled = true;
  btn.innerText = "Indexing...";
  status.classList.remove('hidden');
  status.innerText = `Ingesting ${file.name}...`;

  const formData = new FormData();
  formData.append('pdf', file);

  try {
    const res = await fetch('/api/upload-pdf', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.status === 'success') {
      status.innerHTML = `<span class="text-emerald-400 font-semibold">✅ Ingested ${data.filename} (${data.chunksCreated} chunks added to AI knowledge base)</span>`;
      if (typeof confetti === 'function') confetti();
    } else {
      status.innerHTML = `<span class="text-rose-400">Error: ${data.message}</span>`;
    }
  } catch (err) {
    status.innerHTML = `<span class="text-rose-400">Failed: ${err.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="file-up" class="w-3.5 h-3.5"></i> Upload PDF`;
    lucide.createIcons();
  }
}

// 1-Click Retrain
async function retrainSimpleModel() {
  const btn = document.getElementById('btn-simple-retrain');
  btn.disabled = true;
  btn.innerText = "Retraining Model...";

  try {
    const res = await fetch('/api/retrain', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      document.getElementById('simple-acc-score').innerText = (data.metrics.accuracy * 100).toFixed(1) + "%";
      if (typeof confetti === 'function') confetti();
      alert(`🎉 Model Retrained Successfully!\nAccuracy: ${(data.metrics.accuracy * 100).toFixed(1)}%`);
    } else {
      alert("Retrain failed: " + data.message);
    }
  } catch (e) {
    alert("Request error: " + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="refresh-cw" class="w-4 h-4"></i> Retrain AI Model`;
    lucide.createIcons();
  }
}
