/**
 * MarineMine Pro — Main Application Orchestrator & State Management
 */

// Configure Chart.js global defaults for Dark Ocean high-contrast visibility
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#a0c2dc';
  Chart.defaults.borderColor = 'rgba(56, 224, 242, 0.12)';
  if (Chart.defaults.plugins && Chart.defaults.plugins.legend) {
    Chart.defaults.plugins.legend.labels.color = '#e2f1f8';
  }
}

// Determine API base dynamically: fallback to :8000 if frontend is served on another port
const API_BASE = (window.location.port && window.location.port !== '8000')
  ? `${window.location.protocol}//${window.location.hostname}:8000`
  : window.location.origin;

// Application State
const AppState = {
  currentTab: 'dashboard',
  datasetLoaded: false,
  datasetInfo: null,
  activeColumns: [],
  selectedTarget: null,
  charts: {},
};

// UI Notification System
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `ocean-toast ${type}`;

  let icon = 'ℹ️';
  if (type === 'success') icon = '✅';
  if (type === 'error') icon = '⚠️';
  if (type === 'warning') icon = '🔔';

  toast.innerHTML = `<span>${icon}</span> <div>${message}</div>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Tab Switcher
function switchTab(tabId) {
  AppState.currentTab = tabId;

  // Update navigation items
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.dataset.tab === tabId) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Show selected page section
  document.querySelectorAll('.page-section').forEach(sec => {
    if (sec.id === `section-${tabId}`) {
      sec.classList.add('active');
    } else {
      sec.classList.remove('active');
    }
  });

  // Trigger tab-specific refresh
  if (tabId === 'dashboard') {
    if (window.DashboardModule) window.DashboardModule.refresh();
  } else if (tabId === 'dataset') {
    if (window.DatasetModule) window.DatasetModule.loadPreview();
  } else if (tabId === 'quality') {
    if (window.QualityModule) window.QualityModule.refresh();
  } else if (tabId === 'eda') {
    if (window.EDAModule) window.EDAModule.refresh();
  } else if (tabId === 'hotspots') {
    if (window.MapsModule) {
      // Delay slightly to let the browser recalculate layout of the newly visible section
      setTimeout(() => {
        window.MapsModule.refresh();
      }, 80);
    }
  }
}

// Safe API Fetcher
async function apiRequest(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `Request failed with status ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error on [${endpoint}]:`, err);
    showToast(err.message, 'error');
    throw err;
  }
}

// Global Dataset Sync
async function syncDatasetStatus() {
  try {
    const info = await apiRequest('/api/dataset/info');
    if (info && (info.rows > 0 || info.total_records > 0)) {
      AppState.datasetLoaded = true;
      AppState.datasetInfo = info;
      AppState.activeColumns = info.columns || [];

      // Update UI Pills & Badges
      const pill = document.getElementById('navbar-dataset-pill');
      if (pill) {
        const rowCount = info.rows || info.total_records || 0;
        pill.innerHTML = `<span class="status-dot"></span> ${info.name || info.dataset_name || 'Dataset'} (${rowCount.toLocaleString()} records)`;
        pill.style.display = 'inline-flex';
      }

      // Populate column selectors across ML modules with intelligent pre-selection
      populateColumnDropdowns(
        info.columns || [],
        info.numeric_columns || [],
        info.categorical_columns || []
      );

      // Refresh current active view
      switchTab(AppState.currentTab);
    } else {
      AppState.datasetLoaded = false;
      const pill = document.getElementById('navbar-dataset-pill');
      if (pill) pill.style.display = 'none';
    }
  } catch (err) {
    AppState.datasetLoaded = false;
    console.warn('Could not sync dataset status:', err);
  }
}

function populateColumnDropdowns(allCols, numCols, catCols) {
  const populate = (id, list, defaultVal = null, isMulti = false, maxSelect = 4) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '';
    list.forEach((col, idx) => {
      const opt = document.createElement('option');
      opt.value = col;
      opt.textContent = col;
      if (isMulti) {
        if (idx < maxSelect) opt.selected = true;
      } else {
        if (defaultVal) {
          if (col === defaultVal) opt.selected = true;
        } else if (idx === 0) {
          opt.selected = true;
        }
      }
      el.appendChild(opt);
    });
  };

  // EDA Histogram feature selector
  populate('eda-hist-feature', numCols, numCols[0], false);

  // Clustering: multi-select pre-selected with first 3 numeric parameters
  populate('clustering-features', numCols, null, true, 3);

  // Classification: Target column & input features
  const defaultClfTarget = catCols.find(c => c.toLowerCase().includes('type') || c.toLowerCase().includes('category') || c.toLowerCase().includes('level')) || catCols[0] || allCols[0];
  populate('clf-target', catCols.length ? catCols : allCols, defaultClfTarget, false);
  const clfFeats = numCols.filter(c => c !== defaultClfTarget);
  populate('clf-features', clfFeats, null, true, 4);

  // Prediction: Continuous regression target & features
  const defaultPredTarget = numCols.find(c => c.toLowerCase().includes('value') || c.toLowerCase().includes('pollution') || c.toLowerCase().includes('concentration')) || numCols[numCols.length - 1];
  populate('pred-target', numCols, defaultPredTarget, false);
  const predFeatures = numCols.filter(c => c !== defaultPredTarget);
  populate('pred-features', predFeatures, null, true, 4);

  // Association Rules: categorical fields
  populate('assoc-features', catCols, null, true, 3);
}

// Set system date in header
function updateHeaderDate() {
  const dateEl = document.getElementById('navbar-date');
  if (dateEl) {
    const now = new Date();
    dateEl.textContent = now.toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }
}

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  updateHeaderDate();

  // Navigation Links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const tab = link.dataset.tab;
      if (tab) switchTab(tab);
    });
  });

  // Initial Sync
  syncDatasetStatus();
});
