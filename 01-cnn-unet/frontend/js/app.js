/**
 * CrackSense – Frontend Application Logic
 * Connects to Flask /api/predict endpoint.
 */

const API_PREDICT = '/api/predict';

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;
let analysisHistory = JSON.parse(localStorage.getItem('cs_history') || '[]');

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadZone  = document.getElementById('upload-zone');
const previewZone = document.getElementById('preview-zone');
const loadingZone = document.getElementById('loading-zone');
const resultsZone = document.getElementById('results-zone');
const errorZone   = document.getElementById('error-zone');

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => { renderHistory(); });

// ═════════════════════════════════════════════════════════════════════════════
// UPLOAD HANDLING
// ═════════════════════════════════════════════════════════════════════════════

function handleDragOver(e) {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
}

function handleDragLeave(e) {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('error', 'Please upload an image file (PNG, JPG, BMP, TIFF, WebP).');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showToast('error', 'File exceeds 16 MB limit.');
    return;
  }

  selectedFile = file;
  hideAll();
  previewZone.classList.remove('hidden');

  // Show metadata
  document.getElementById('file-name-display').textContent = file.name;
  document.getElementById('file-size-display').textContent = formatSize(file.size);
  document.getElementById('preview-meta').textContent = file.name.split('.').pop().toUpperCase();

  // Preview image
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('preview-img').src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ═════════════════════════════════════════════════════════════════════════════
// ANALYSIS
// ═════════════════════════════════════════════════════════════════════════════

async function runAnalysis() {
  if (!selectedFile) return;

  // Transition to loading state
  hideAll();
  loadingZone.classList.remove('hidden');
  document.getElementById('loading-preview-img').src = document.getElementById('preview-img').src;

  // Animate loading steps
  animateLoadingSteps();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(API_PREDICT, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw { message: data.error || 'Unknown server error.', hint: data.hint || '' };
    }

    displayResults(data);

  } catch (err) {
    showError(
      err.message || 'Could not connect to the server.',
      err.hint || 'Make sure the Flask server is running and models are trained.'
    );
  }
}

function animateLoadingSteps() {
  const steps = ['step-unet', 'step-overlay'];
  steps.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.add('opacity-40');
    el.querySelectorAll('.rounded-full').forEach(d => d.classList.remove('pulse-dot','bg-primary'));

    setTimeout(() => {
      el.classList.remove('opacity-40');
      const dot = el.querySelector('.rounded-full');
      if (dot) { dot.classList.add('bg-primary','pulse-dot'); dot.classList.remove('bg-outline'); }
    }, i * 800);
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// RESULTS RENDERING
// ═════════════════════════════════════════════════════════════════════════════

function displayResults(data) {
  hideAll();
  resultsZone.classList.remove('hidden');

  // Images
  document.getElementById('result-original-img').src = `data:image/jpeg;base64,${data.original_b64}`;
  document.getElementById('result-overlay-img').src  = `data:image/jpeg;base64,${data.overlay_b64}`;

  // Crack labels
  const labelsEl = document.getElementById('crack-labels');
  labelsEl.innerHTML = '';
  if (data.label === 'crack') {
    const severityColor = { HIGH: 'error', MEDIUM: 'tertiary', LOW: 'primary' }[data.intensity] || 'primary';
    labelsEl.innerHTML = `
      <div class="px-2 py-1 bg-${severityColor}/20 border border-${severityColor}/50 text-[10px] font-bold text-${severityColor}">
        ${data.intensity}_CRACK
      </div>`;
  }

  // Right panel — confidence
  const confPct = Math.round(data.confidence * 100);
  document.getElementById('confidence-display').textContent = `${confPct}%`;
  setTimeout(() => {
    document.getElementById('confidence-bar').style.width = `${confPct}%`;
  }, 100);

  // Coverage
  document.getElementById('coverage-display').textContent =
    data.label === 'crack' ? `${data.crack_coverage}%` : '0%';

  // Intensity
  const intensityEl = document.getElementById('intensity-display');
  intensityEl.textContent = data.intensity;
  intensityEl.className = 'text-2xl font-bold technical-tight ' + intensityColor(data.intensity);

  // Risk badge
  const badge = document.getElementById('risk-badge');
  badge.textContent = riskLabel(data.intensity);
  badge.className = `px-2 py-1 font-black text-[10px] rounded-sm ${riskBadgeClass(data.intensity)}`;

  // Assessment note
  document.getElementById('assessment-note').innerHTML = `
    <div class="p-4 ${assessmentBg(data.intensity)} border-l-4 ${assessmentBorder(data.intensity)}">
      <p class="text-xs font-medium leading-relaxed ${assessmentText(data.intensity)}">
        ${assessmentMessage(data)}
      </p>
    </div>`;

  // Enable download report
  document.getElementById('download-report-btn').disabled = false;

  // Add to history
  addToHistory(data);

  showToast('info', 'Analysis complete.');
}

// ── Helpers for styling ───────────────────────────────────────────────────────
function intensityColor(intensity) {
  return { HIGH: 'text-error', MEDIUM: 'text-tertiary', LOW: 'text-primary', NONE: 'text-on-surface-variant' }[intensity] || 'text-on-surface';
}

function riskLabel(intensity) {
  return { HIGH: 'HIGH RISK', MEDIUM: 'MODERATE', LOW: 'LOW RISK', NONE: 'CLEAR' }[intensity] || 'PENDING';
}

function riskBadgeClass(intensity) {
  return {
    HIGH:   'bg-error/20 text-error border border-error/40',
    MEDIUM: 'bg-tertiary/20 text-tertiary border border-tertiary/40',
    LOW:    'bg-primary/20 text-primary border border-primary/40',
    NONE:   'bg-surface-container-low text-on-surface-variant',
  }[intensity] || 'bg-surface-container-low text-on-surface-variant';
}

function assessmentBg(intensity) {
  return { HIGH:'bg-error/10', MEDIUM:'bg-tertiary/10', LOW:'bg-primary/10', NONE:'bg-surface-container-low' }[intensity] || 'bg-surface-container-low';
}
function assessmentBorder(i) {
  return { HIGH:'border-error', MEDIUM:'border-tertiary', LOW:'border-primary', NONE:'border-outline-variant/40' }[i] || 'border-outline-variant/40';
}
function assessmentText(i) {
  return { HIGH:'text-on-error-container', MEDIUM:'text-tertiary', LOW:'text-primary', NONE:'text-on-surface-variant' }[i] || 'text-on-surface-variant';
}

function assessmentMessage(data) {
  if (data.label === 'no_crack') return 'No structural cracks detected. Surface integrity appears nominal.';
  const msgs = {
    HIGH:   `Critical crack detected (${data.crack_coverage}% coverage). Immediate structural verification recommended.`,
    MEDIUM: `Moderate crack detected (${data.crack_coverage}% coverage). Schedule inspection within 30 days.`,
    LOW:    `Minor crack detected (${data.crack_coverage}% coverage). Monitor and log for future comparison.`,
  };
  return msgs[data.intensity] || 'Crack detected. Review overlay for spatial distribution.';
}

// ═════════════════════════════════════════════════════════════════════════════
// HISTORY
// ═════════════════════════════════════════════════════════════════════════════

function addToHistory(data) {
  const entry = {
    id:        Date.now(),
    name:      selectedFile?.name || 'Unknown',
    intensity: data.intensity,
    label:     data.label,
    confidence: Math.round(data.confidence * 100),
    coverage:  data.crack_coverage,
    time:      new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  };
  analysisHistory.unshift(entry);
  if (analysisHistory.length > 10) analysisHistory.pop();
  localStorage.setItem('cs_history', JSON.stringify(analysisHistory));
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('history-list');
  if (!analysisHistory.length) {
    list.innerHTML = '<p class="text-xs text-on-surface-variant/60 px-1">No analyses yet.</p>';
    return;
  }
  list.innerHTML = analysisHistory.map(e => `
    <div class="p-3 bg-surface-container-low border border-outline-variant/10 hover:border-primary/30 transition-colors group cursor-pointer">
      <div class="flex justify-between items-start mb-1">
        <p class="text-sm font-bold truncate pr-2 group-hover:text-primary transition-colors">${escHtml(e.name)}</p>
        <span class="text-[9px] font-mono text-on-surface-variant flex-shrink-0">${e.time}</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full ${histDot(e.intensity)}"></span>
        <span class="text-[10px] text-on-surface-variant font-medium">${histLabel(e.label, e.intensity)} · ${e.confidence}%</span>
      </div>
    </div>`).join('');
}

function histDot(i) {
  return { HIGH:'bg-error', MEDIUM:'bg-tertiary', LOW:'bg-primary', NONE:'bg-outline' }[i] || 'bg-outline';
}
function histLabel(label, i) {
  if (label === 'no_crack') return 'No Crack Detected';
  return { HIGH:'Critical Issue Found', MEDIUM:'Warning Observed', LOW:'Minor Crack Found', NONE:'Clear' }[i] || 'Crack Detected';
}

function clearHistory() {
  analysisHistory = [];
  localStorage.removeItem('cs_history');
  renderHistory();
  showToast('info', 'Analysis history cleared.');
}

// ═════════════════════════════════════════════════════════════════════════════
// DOWNLOAD
// ═════════════════════════════════════════════════════════════════════════════

function downloadImage(imgId, filename) {
  const img = document.getElementById(imgId);
  if (!img || !img.src) return;
  const a = document.createElement('a');
  a.href = img.src;
  a.download = filename;
  a.click();
}

function downloadReport() {
  const overlay = document.getElementById('result-overlay-img')?.src;
  const original = document.getElementById('result-original-img')?.src;
  if (!overlay) return;

  const confPct  = document.getElementById('confidence-display').textContent;
  const coverage = document.getElementById('coverage-display').textContent;
  const intensity = document.getElementById('intensity-display').textContent;

  const win = window.open('', '_blank');
  win.document.write(`<!DOCTYPE html><html><head><meta charset="utf-8"/>
    <title>CrackSense Report</title>
    <style>
      body{font-family:Manrope,sans-serif;background:#060e20;color:#dee5ff;padding:40px;max-width:900px;margin:auto;}
      h1{font-size:2rem;font-weight:800;color:#81ecff;}
      .meta{color:#a3aac4;font-size:0.85rem;margin-bottom:32px;}
      .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:24px;}
      img{width:100%;border:1px solid #40485d44;}
      .badge{display:inline-block;padding:4px 12px;font-weight:900;font-size:0.75rem;border-radius:2px;}
      .HIGH{background:#9f0519;color:#ffa8a3;} .MEDIUM{background:#3d2e00;color:#ffc563;} .LOW{background:#00163a;color:#81ecff;}
      table{width:100%;border-collapse:collapse;margin-top:24px;}
      td{padding:12px;border-bottom:1px solid #40485d44;font-size:0.9rem;}
      td:first-child{color:#a3aac4;}
    </style></head><body>
    <h1>CrackSense Analysis Report</h1>
    <p class="meta">Generated: ${new Date().toLocaleString()} · Analytical Engine V4.2</p>
    <span class="badge ${intensity}">${intensity} INTENSITY</span>
    <div class="grid">
      <div><p style="font-size:0.7rem;color:#a3aac4;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Original</p><img src="${original}"/></div>
      <div><p style="font-size:0.7rem;color:#a3aac4;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">Crack Overlay</p><img src="${overlay}"/></div>
    </div>
    <table>
      <tr><td>File</td><td>${escHtml(selectedFile?.name || '—')}</td></tr>
      <tr><td>Classification</td><td>${intensity === 'NONE' ? 'No Crack' : 'Crack Detected'}</td></tr>
      <tr><td>Intensity</td><td>${intensity}</td></tr>
      <tr><td>Confidence Score</td><td>${confPct}</td></tr>
      <tr><td>Crack Coverage</td><td>${coverage}</td></tr>
      <tr><td>Models Used</td><td>UNet Segmentor</td></tr>
    </table>
    </body></html>`);
  win.document.close();
  win.print();
}

// ═════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═════════════════════════════════════════════════════════════════════════════

function resetAnalysis() {
  selectedFile = null;
  hideAll();
  uploadZone.classList.remove('hidden');
  document.getElementById('file-input').value = '';
  // Reset right panel
  document.getElementById('confidence-display').textContent = '—';
  document.getElementById('confidence-bar').style.width = '0%';
  document.getElementById('coverage-display').textContent = '—';
  document.getElementById('intensity-display').textContent = '—';
  document.getElementById('risk-badge').textContent = 'PENDING';
  document.getElementById('risk-badge').className = 'px-2 py-1 bg-surface-container-low text-on-surface-variant font-black text-[10px] rounded-sm';
  document.getElementById('assessment-note').innerHTML = `
    <div class="p-4 bg-surface-container-low border-l-4 border-outline-variant/40">
      <p class="text-xs font-medium text-on-surface-variant leading-relaxed">Upload an image to begin structural assessment.</p>
    </div>`;
  document.getElementById('download-report-btn').disabled = true;
}

function showError(msg, hint = '') {
  hideAll();
  errorZone.classList.remove('hidden');
  document.getElementById('error-message-text').textContent = msg;
  document.getElementById('error-hint-text').textContent = hint;
}

function hideAll() {
  [uploadZone, previewZone, loadingZone, resultsZone, errorZone].forEach(el => {
    el?.classList.add('hidden');
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

let toastTimer;
function showToast(type, message) {
  const toast    = document.getElementById('toast');
  const toastIcon = document.getElementById('toast-icon');
  const toastText = document.getElementById('toast-text');
  const icons = { error:'error', info:'check_circle', warn:'warning' };
  toastIcon.textContent = icons[type] || 'info';
  toastText.textContent  = message;
  toast.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 3500);
}
