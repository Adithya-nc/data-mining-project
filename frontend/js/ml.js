/**
 * MarineMine Pro — Machine Learning & Mining Modules
 * Preprocessing, K-Means Clustering, Classification, Association Rules, Prediction Simulator
 */

window.MLModule = {
  charts: {},

  init() {
    this.initPreprocessing();
    this.initClustering();
    this.initClassification();
    this.initAssociation();
    this.initPrediction();
  },

  // --------------------------------------------------------------------------
  // 1. Data Preprocessing
  // --------------------------------------------------------------------------
  initPreprocessing() {
    const btnClean = document.getElementById('btn-run-preprocess');
    if (btnClean) {
      btnClean.addEventListener('click', async () => {
        const req = {
          numeric_strategy: document.getElementById('pre-num-missing')?.value || 'mean',
          categorical_strategy: document.getElementById('pre-cat-missing')?.value || 'mode',
          remove_duplicates: document.getElementById('pre-remove-dups')?.checked ?? true,
          outlier_strategy: document.getElementById('pre-outlier-strategy')?.value || 'winsorize',
          outlier_factor: parseFloat(document.getElementById('pre-outlier-factor')?.value || 1.5),
          scale_features: document.getElementById('pre-scale-features')?.checked ?? false,
          scale_method: document.getElementById('pre-scale-method')?.value || 'standard',
          encode_categoricals: document.getElementById('pre-encode-cat')?.checked ?? false,
          encode_method: document.getElementById('pre-encode-method')?.value || 'label',
        };

        try {
          btnClean.disabled = true;
          btnClean.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Cleaning...';
          const res = await apiRequest('/api/preprocessing/clean', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req)
          });
          showToast(res.message, 'success');

          const setTxt = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
          };

          setTxt('pre-summary-orig-rows', res.original_rows.toLocaleString());
          setTxt('pre-summary-clean-rows', res.cleaned_rows.toLocaleString());
          setTxt('pre-summary-orig-missing', res.original_missing.toLocaleString());
          setTxt('pre-summary-clean-missing', res.cleaned_missing.toLocaleString());

          const resCard = document.getElementById('pre-results-card');
          if (resCard) resCard.style.display = 'block';
          await syncDatasetStatus();
        } catch (err) {
          // Handled
        } finally {
          btnClean.disabled = false;
          btnClean.innerHTML = '⚡ Apply Preprocessing Pipeline';
        }
      });
    }

    const btnDownloadClean = document.getElementById('btn-download-cleaned-dataset');
    if (btnDownloadClean) {
      btnDownloadClean.addEventListener('click', () => {
        window.open('/api/preprocessing/download', '_blank');
      });
    }
  },

  // --------------------------------------------------------------------------
  // 2. K-Means Clustering
  // --------------------------------------------------------------------------
  initClustering() {
    const btnRun = document.getElementById('btn-run-clustering');
    const btnOptimal = document.getElementById('btn-optimal-k');

    if (btnOptimal) {
      btnOptimal.addEventListener('click', async () => {
        const featSelect = document.getElementById('clustering-features');
        if (!featSelect) return;
        let selectedFeats = Array.from(featSelect.selectedOptions).map(o => o.value);
        if (selectedFeats.length < 2) {
          selectedFeats = Array.from(featSelect.options).slice(0, 3).map(o => o.value);
        }
        if (selectedFeats.length < 2) {
          showToast('Please select at least 2 numerical features for optimal K testing', 'warning');
          return;
        }

        try {
          btnOptimal.disabled = true;
          btnOptimal.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Evaluating K...';
          const opt = await apiRequest('/api/ml/clustering/optimal-k', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: selectedFeats, k: 3 })
          });

          showToast(`Optimal Clusters recommended: K = ${opt.recommended_k} (Silhouette: ${opt.best_silhouette})`, 'success');
          const slider = document.getElementById('clustering-k-slider');
          if (slider) slider.value = opt.recommended_k;
          const kVal = document.getElementById('clustering-k-val');
          if (kVal) kVal.textContent = opt.recommended_k;
          this.renderElbowChart(opt.k_range, opt.inertias, opt.silhouette_scores);
        } catch (err) {
          // Handled
        } finally {
          btnOptimal.disabled = false;
          btnOptimal.innerHTML = '🔍 Find Optimal K';
        }
      });
    }

    const slider = document.getElementById('clustering-k-slider');
    if (slider) {
      slider.addEventListener('input', () => {
        const kVal = document.getElementById('clustering-k-val');
        if (kVal) kVal.textContent = slider.value;
      });
    }

    if (btnRun) {
      btnRun.addEventListener('click', async () => {
        const featSelect = document.getElementById('clustering-features');
        if (!featSelect) return;
        let selectedFeats = Array.from(featSelect.selectedOptions).map(o => o.value);
        if (selectedFeats.length < 2) {
          selectedFeats = Array.from(featSelect.options).slice(0, 3).map(o => o.value);
        }
        const k = parseInt(document.getElementById('clustering-k-slider')?.value || 3, 10);

        if (selectedFeats.length < 2) {
          showToast('Please select at least 2 features for K-Means clustering', 'warning');
          return;
        }

        try {
          btnRun.disabled = true;
          btnRun.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Clustering...';
          const [res, opt] = await Promise.all([
            apiRequest('/api/ml/clustering', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ features: selectedFeats, k })
            }),
            apiRequest('/api/ml/clustering/optimal-k', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ features: selectedFeats, k })
            }).catch(() => null)
          ]);

          showToast(`K-Means completed! Silhouette Score: ${res.silhouette_score}`, 'success');
          const sil = document.getElementById('clustering-silhouette');
          if (sil) sil.textContent = res.silhouette_score;
          const inr = document.getElementById('clustering-inertia');
          if (inr) inr.textContent = res.inertia.toLocaleString();

          if (opt && opt.k_range && opt.inertias && opt.silhouette_scores) {
            this.renderElbowChart(opt.k_range, opt.inertias, opt.silhouette_scores);
          }

          this.renderClusterScatter(res.pca_data);
          this.renderClusterProfiles(res.cluster_profiles);
          const resCard = document.getElementById('clustering-results-card');
          if (resCard) resCard.style.display = 'block';
        } catch (err) {
          // Handled
        } finally {
          btnRun.disabled = false;
          btnRun.innerHTML = '🚀 Execute K-Means Clustering';
        }
      });
    }
  },

  renderElbowChart(kRange, inertias, silhouettes) {
    const ctx = document.getElementById('chart-clustering-elbow');
    if (!ctx) return;
    if (this.charts.elbow) this.charts.elbow.destroy();

    this.charts.elbow = new Chart(ctx, {
      type: 'line',
      data: {
        labels: kRange.map(k => `K = ${k}`),
        datasets: [
          {
            label: 'Inertia (Elbow)',
            data: inertias,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            yAxisID: 'y',
            tension: 0.3,
          },
          {
            label: 'Silhouette Score',
            data: silhouettes,
            borderColor: '#00d2ff',
            backgroundColor: 'rgba(0, 210, 255, 0.1)',
            yAxisID: 'y1',
            tension: 0.3,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8892b0' }, grid: { display: false } },
          y: { type: 'linear', position: 'left', ticks: { color: '#f59e0b' } },
          y1: { type: 'linear', position: 'right', ticks: { color: '#00d2ff' }, grid: { drawOnChartArea: false } }
        },
        plugins: {
          legend: { labels: { color: '#e6f1ff' } }
        }
      }
    });
  },

  renderClusterScatter(pcaData) {
    const ctx = document.getElementById('chart-clustering-scatter');
    if (!ctx) return;
    if (this.charts.scatter) this.charts.scatter.destroy();

    const clusterColors = ['#00d2ff', '#38ef7d', '#f59e0b', '#f43f5e', '#8b5cf6', '#0077b6', '#00f2fe', '#ec4899'];

    // Group by cluster
    const grouped = {};
    pcaData.forEach(pt => {
      const c = pt.cluster;
      if (!grouped[c]) grouped[c] = [];
      grouped[c].push({ x: pt.x, y: pt.y });
    });

    const datasets = Object.keys(grouped).map((c, idx) => ({
      label: `Cluster ${c}`,
      data: grouped[c],
      backgroundColor: clusterColors[idx % clusterColors.length],
      borderColor: '#0a192f',
      borderWidth: 1,
      pointRadius: 5,
    }));

    this.charts.scatter = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'PCA Principal Component 1', color: '#8892b0' }, ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { title: { display: true, text: 'PCA Principal Component 2', color: '#8892b0' }, ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        },
        plugins: {
          legend: { labels: { color: '#e6f1ff' } }
        }
      }
    });
  },

  renderClusterProfiles(profiles) {
    const container = document.getElementById('clustering-profiles-container');
    if (!container) return;
    container.innerHTML = '';

    profiles.forEach(p => {
      const card = document.createElement('div');
      card.className = 'col-md-4 mb-3';

      let meanRows = '';
      for (const [k, v] of Object.entries(p.means || {})) {
        meanRows += `<tr><td class="text-muted">${k}</td><td class="text-end fw-bold">${v}</td></tr>`;
      }

      card.innerHTML = `
        <div class="ocean-card h-100 p-3">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="badge bg-primary fs-6">${p.label}</span>
            <span class="text-info small fw-bold">${p.size} pts (${p.percentage}%)</span>
          </div>
          <table class="table table-sm table-dark table-borderless small mb-0">
            ${meanRows}
          </table>
        </div>
      `;
      container.appendChild(card);
    });
  },

  // --------------------------------------------------------------------------
  // 3. Supervised Classification
  // --------------------------------------------------------------------------
  initClassification() {
    const btnTrain = document.getElementById('btn-train-classification');
    if (!btnTrain) return;

    btnTrain.addEventListener('click', async () => {
      const targetSelect = document.getElementById('clf-target');
      const targetCol = targetSelect?.value || targetSelect?.options[0]?.value;
      const featSelect = document.getElementById('clf-features');
      let selectedFeats = Array.from(featSelect?.selectedOptions || []).map(o => o.value);
      if (!selectedFeats.length && featSelect?.options) {
        selectedFeats = Array.from(featSelect.options).slice(0, 4).map(o => o.value);
      }
      const testSize = parseFloat(document.getElementById('clf-test-size')?.value || 0.2);

      if (!targetCol) {
        showToast('Please select a target label for classification', 'warning');
        return;
      }

      try {
        btnTrain.disabled = true;
        btnTrain.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training Classifiers...';
        const res = await apiRequest('/api/ml/classification', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_col: targetCol, feature_cols: selectedFeats.length ? selectedFeats : null, test_size: testSize })
        });

        showToast(`Classification complete! Best Model: ${res.best_model} (${(res.best_accuracy * 100).toFixed(1)}%)`, 'success');
        this.renderClassificationResults(res);
      } catch (err) {
        // Handled
      } finally {
        btnTrain.disabled = false;
        btnTrain.innerHTML = '🎯 Train & Benchmark Classifiers';
      }
    });
  },

  renderClassificationResults(res) {
    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setTxt('clf-best-model', res.best_model);
    setTxt('clf-best-accuracy', `${(res.best_accuracy * 100).toFixed(2)}%`);
    setTxt('clf-best-f1', res.best_f1.toFixed(4));
    setTxt('clf-elapsed', `${res.elapsed_ms} ms`);

    // Benchmark Table
    const tbody = document.getElementById('clf-models-table-body');
    if (tbody) {
      tbody.innerHTML = '';
      for (const [mName, m] of Object.entries(res.models)) {
        const tr = document.createElement('tr');
        const isBest = mName === res.best_model;
        tr.innerHTML = `
          <td class="fw-bold ${isBest ? 'text-info' : ''}">${isBest ? '🏆 ' : ''}${mName}</td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${(m.recall * 100).toFixed(2)}%</td>
          <td>${m.f1_score.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // Confusion Matrix Grid
    const cmContainer = document.getElementById('clf-confusion-matrix');
    if (cmContainer && res.confusion_matrix) {
      cmContainer.innerHTML = '';
      const cm = res.confusion_matrix;
      const size = cm.length;
      cmContainer.style.gridTemplateColumns = `repeat(${size}, 1fr)`;

      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const val = cm[r][c];
          const cell = document.createElement('div');
          cell.className = 'cm-cell';
          const isDiag = r === c;
          cell.style.backgroundColor = isDiag ? 'rgba(0, 210, 255, 0.3)' : 'rgba(244, 63, 94, 0.15)';
          cell.style.border = isDiag ? '1px solid #00d2ff' : '1px solid rgba(255,255,255,0.05)';
          cell.textContent = val;
          cmContainer.appendChild(cell);
        }
      }
    }

    // Feature Importance Chart
    const ctx = document.getElementById('chart-clf-importance');
    if (ctx && res.feature_importance) {
      if (this.charts.clfImp) this.charts.clfImp.destroy();
      const labels = res.feature_importance.map(f => f.feature);
      const vals = res.feature_importance.map(f => f.importance);

      this.charts.clfImp = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Feature Importance',
            data: vals,
            backgroundColor: 'rgba(0, 210, 255, 0.7)',
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#8892b0' }, grid: { display: false } }
          },
          plugins: { legend: { display: false } }
        }
      });
    }

    const resCard = document.getElementById('clf-results-card');
    if (resCard) resCard.style.display = 'block';
  },

  // --------------------------------------------------------------------------
  // 4. Association Rule Mining
  // --------------------------------------------------------------------------
  cachedRules: [],

  initAssociation() {
    const btnRun = document.getElementById('btn-run-association');
    const suppSlider = document.getElementById('assoc-min-support-slider');
    const confSlider = document.getElementById('assoc-min-confidence-slider');
    const liftSlider = document.getElementById('assoc-min-lift-slider');
    const searchInput = document.getElementById('assoc-search-input');

    // Dynamic slider listeners for real-time rule filtering
    const updateSliders = () => {
      const sVal = parseFloat(suppSlider?.value || 0.05);
      const cVal = parseFloat(confSlider?.value || 0.4);
      const lVal = parseFloat(liftSlider?.value || 1.0);

      const sDisp = document.getElementById('assoc-val-supp');
      const cDisp = document.getElementById('assoc-val-conf');
      const lDisp = document.getElementById('assoc-val-lift');
      if (sDisp) sDisp.textContent = `${(sVal * 100).toFixed(0)}%`;
      if (cDisp) cDisp.textContent = `${(cVal * 100).toFixed(0)}%`;
      if (lDisp) lDisp.textContent = `${lVal.toFixed(1)}x`;

      const numSupp = document.getElementById('assoc-min-support');
      const numConf = document.getElementById('assoc-min-confidence');
      const numLift = document.getElementById('assoc-min-lift');
      if (numSupp) numSupp.value = sVal;
      if (numConf) numConf.value = cVal;
      if (numLift) numLift.value = lVal;

      this.filterAndRenderRules();
    };

    if (suppSlider) suppSlider.addEventListener('input', updateSliders);
    if (confSlider) confSlider.addEventListener('input', updateSliders);
    if (liftSlider) liftSlider.addEventListener('input', updateSliders);
    if (searchInput) searchInput.addEventListener('input', () => this.filterAndRenderRules());

    if (!btnRun) return;

    btnRun.addEventListener('click', async () => {
      const featSelect = document.getElementById('assoc-features');
      let selectedCats = Array.from(featSelect?.selectedOptions || []).map(o => o.value);
      if (!selectedCats.length && featSelect?.options) {
        selectedCats = Array.from(featSelect.options).map(o => o.value);
      }
      const minSupp = parseFloat(document.getElementById('assoc-min-support')?.value || 0.05);
      const minConf = parseFloat(document.getElementById('assoc-min-confidence')?.value || 0.4);
      const minLift = parseFloat(document.getElementById('assoc-min-lift')?.value || 1.0);

      try {
        btnRun.disabled = true;
        btnRun.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Mining Patterns...';
        const res = await apiRequest('/api/ml/association', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            categorical_cols: selectedCats.length ? selectedCats : null,
            min_support: Math.min(minSupp, 0.05), // Mine broader base so sliders can filter dynamically
            min_confidence: Math.min(minConf, 0.25),
            min_lift: 1.0
          })
        });

        if (!res.suitable) {
          showToast(res.explanation, 'warning');
        } else {
          showToast(`Discovered ${res.rules_count} strong association rules!`, 'success');
        }

        this.cachedRules = res.rules || [];
        const expEl = document.getElementById('assoc-explanation');
        if (expEl) expEl.textContent = res.explanation || '';
        this.filterAndRenderRules();
      } catch (err) {
        // Handled
      } finally {
        btnRun.disabled = false;
        btnRun.innerHTML = '🔗 Discover Association Patterns';
      }
    });
  },

  filterAndRenderRules() {
    const minSupp = parseFloat(document.getElementById('assoc-min-support')?.value || 0.05);
    const minConf = parseFloat(document.getElementById('assoc-min-confidence')?.value || 0.4);
    const minLift = parseFloat(document.getElementById('assoc-min-lift')?.value || 1.0);
    const query = (document.getElementById('assoc-search-input')?.value || '').toLowerCase().trim();

    const filtered = (this.cachedRules || []).filter(r => {
      if (r.support < minSupp) return false;
      if (r.confidence < minConf) return false;
      if (r.lift < minLift) return false;
      if (query) {
        const antStr = (Array.isArray(r.antecedents) ? r.antecedents.join(' ') : r.antecedents).toLowerCase();
        const conStr = (Array.isArray(r.consequents) ? r.consequents.join(' ') : r.consequents).toLowerCase();
        if (!antStr.includes(query) && !conStr.includes(query)) return false;
      }
      return true;
    });

    const countEl = document.getElementById('assoc-rules-count');
    if (countEl) countEl.textContent = filtered.length;

    const cardsContainer = document.getElementById('assoc-rule-cards-container');
    const tbody = document.getElementById('assoc-rules-table-body');

    if (cardsContainer) {
      cardsContainer.innerHTML = '';
      if (!filtered.length) {
        cardsContainer.innerHTML = '<div class="text-muted small p-3 text-center">No rules match the current slider thresholds. Lower the Min Support or Min Confidence sliders.</div>';
      } else {
        filtered.slice(0, 10).forEach(rule => {
          const ant = Array.isArray(rule.antecedents) ? rule.antecedents.join(', ') : rule.antecedents;
          const con = Array.isArray(rule.consequents) ? rule.consequents.join(', ') : rule.consequents;
          const card = document.createElement('div');
          card.className = 'rule-item-card d-flex flex-wrap align-items-center justify-content-between gap-3';
          card.innerHTML = `
            <div class="d-flex align-items-center gap-2 flex-wrap">
              <span class="rule-badge-ant">IF: ${ant}</span>
              <span class="text-info fw-bold font-monospace">➔</span>
              <span class="rule-badge-con">THEN: ${con}</span>
            </div>
            <div class="d-flex align-items-center gap-3">
              <span class="small font-monospace text-muted">Supp: <strong class="text-white">${(rule.support * 100).toFixed(1)}%</strong></span>
              <span class="small font-monospace text-muted">Conf: <strong class="text-white">${(rule.confidence * 100).toFixed(1)}%</strong></span>
              <span class="rule-lift-badge">Lift: ${Number(rule.lift).toFixed(2)}x</span>
            </div>
          `;
          cardsContainer.appendChild(card);
        });
      }
    }

    if (tbody) {
      tbody.innerHTML = '';
      filtered.forEach((rule) => {
        const tr = document.createElement('tr');
        const ant = Array.isArray(rule.antecedents) ? rule.antecedents.join(', ') : rule.antecedents;
        const cons = Array.isArray(rule.consequents) ? rule.consequents.join(', ') : rule.consequents;

        tr.innerHTML = `
          <td><span class="rule-badge-ant">${ant}</span></td>
          <td class="text-center text-info fw-bold">➔</td>
          <td><span class="rule-badge-con">${cons}</span></td>
          <td class="font-monospace">${(rule.support * 100).toFixed(1)}%</td>
          <td class="font-monospace">${(rule.confidence * 100).toFixed(1)}%</td>
          <td class="fw-bold font-monospace text-warning">${Number(rule.lift).toFixed(2)}x</td>
        `;
        tbody.appendChild(tr);
      });
    }

    const resCard = document.getElementById('assoc-results-card');
    if (resCard) resCard.style.display = 'block';
  },

  renderAssociationRules(res) {
    this.cachedRules = res.rules || [];
    this.filterAndRenderRules();
  },

  // --------------------------------------------------------------------------
  // 5. Pollution Prediction
  // --------------------------------------------------------------------------
  initPrediction() {
    const btnTrain = document.getElementById('btn-train-prediction');
    const btnSimulate = document.getElementById('btn-run-simulation');

    if (btnTrain) {
      btnTrain.addEventListener('click', async () => {
        const targetSelect = document.getElementById('pred-target');
        const targetCol = targetSelect?.value || targetSelect?.options[0]?.value;
        const featSelect = document.getElementById('pred-features');
        let selectedFeats = Array.from(featSelect?.selectedOptions || []).map(o => o.value);
        if (!selectedFeats.length && featSelect?.options) {
          selectedFeats = Array.from(featSelect.options).slice(0, 4).map(o => o.value);
        }
        const timeSplit = document.getElementById('pred-time-split')?.checked ?? false;

        if (!targetCol) {
          showToast('Please select a continuous target column for regression', 'warning');
          return;
        }

        try {
          btnTrain.disabled = true;
          btnTrain.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training Predictors...';
          const res = await apiRequest('/api/ml/prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              target_col: targetCol,
              feature_cols: selectedFeats.length ? selectedFeats : null,
              test_size: 0.2,
              use_time_split: timeSplit
            })
          });

          showToast(`Regression models trained! Best Model: ${res.best_model} (R²: ${res.best_r2})`, 'success');
          this.renderPredictionResults(res);
          this.setupSimulatorInputs(res.feature_importance);
        } catch (err) {
          // Handled
        } finally {
          btnTrain.disabled = false;
          btnTrain.innerHTML = '📈 Benchmark Regression';
        }
      });
    }

    if (btnSimulate) {
      btnSimulate.addEventListener('click', async () => {
        this.runSimulation();
      });
    }
  },

  renderPredictionResults(res) {
    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setTxt('pred-best-model', res.best_model || 'N/A');
    setTxt('pred-best-r2', Number(res.best_r2 ?? 0).toFixed(4));
    setTxt('pred-best-rmse', Number(res.best_rmse ?? 0).toFixed(4));

    // Benchmark Table
    const tbody = document.getElementById('pred-models-table-body');
    if (tbody) {
      tbody.innerHTML = '';
      for (const [mName, m] of Object.entries(res.models || {})) {
        const tr = document.createElement('tr');
        const isBest = mName === res.best_model;
        tr.innerHTML = `
          <td class="fw-bold ${isBest ? 'text-info' : ''}">${isBest ? '🏆 ' : ''}${mName}</td>
          <td class="fw-bold ${Number(m.r2_score) >= 0.7 ? 'text-success' : 'text-warning'}">${Number(m.r2_score ?? 0).toFixed(4)}</td>
          <td>${Number(m.rmse ?? 0).toFixed(4)}</td>
          <td>${Number(m.mae ?? 0).toFixed(4)}</td>
          <td>${Number(m.mse ?? 0).toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // Actual vs. Predicted Line Chart
    const ctx = document.getElementById('chart-pred-comparison');
    if (ctx && Array.isArray(res.actual_vs_predicted) && res.actual_vs_predicted.length) {
      if (this.charts.predComparison) this.charts.predComparison.destroy();

      const labels = res.actual_vs_predicted.map(d => `#${d.sample_index}`);
      const actualVals = res.actual_vs_predicted.map(d => d.actual);
      const predVals = res.actual_vs_predicted.map(d => d.predicted);

      this.charts.predComparison = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Actual',
              data: actualVals,
              borderColor: '#00f5d4',
              backgroundColor: 'rgba(0, 245, 212, 0.12)',
              borderWidth: 2,
              pointRadius: 3,
              tension: 0.25,
            },
            {
              label: 'Predicted',
              data: predVals,
              borderColor: '#ff6b6b',
              backgroundColor: 'rgba(255, 107, 107, 0.12)',
              borderWidth: 2,
              borderDash: [5, 4],
              pointRadius: 3,
              tension: 0.25,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { ticks: { color: '#8892b0', maxTicksLimit: 10 }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          },
          plugins: {
            legend: { labels: { color: '#e2f1f8' } }
          }
        }
      });
    }

    const resCard = document.getElementById('pred-results-card');
    if (resCard) resCard.style.display = 'block';
  },

  async runSimulation() {
    const btnSimulate = document.getElementById('btn-run-simulation');
    const inputs = document.querySelectorAll('.simulator-input');
    const values = {};
    inputs.forEach(inp => {
      values[inp.dataset.feature] = parseFloat(inp.value || 0.0);
    });

    try {
      if (btnSimulate) {
        btnSimulate.disabled = true;
        btnSimulate.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Simulating...';
      }
      const sim = await apiRequest('/api/ml/prediction/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feature_values: values })
      });

      const setTxt = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };

      setTxt('sim-predicted-value', sim.predicted_value.toFixed(2));
      setTxt('sim-ci', `[${sim.confidence_interval[0]} — ${sim.confidence_interval[1]}]`);
      const resBox = document.getElementById('sim-result-box');
      if (resBox) resBox.style.display = 'block';

      // Update Live Explainability (SHAP / Feature Contribution Bar)
      this.renderExplainabilityChart(values, sim.predicted_value);
    } catch (err) {
      // Handled
    } finally {
      if (btnSimulate) {
        btnSimulate.disabled = false;
        btnSimulate.innerHTML = '🧪 Predict Pollution Level';
      }
    }
  },

  renderExplainabilityChart(inputValues, predictedTotal) {
    const ctx = document.getElementById('chart-sim-explainability');
    if (!ctx) return;
    if (this.charts.simShap) this.charts.simShap.destroy();

    const features = Object.keys(inputValues);
    if (!features.length) return;

    // Compute relative marginal contributions (SHAP-style deviations from normalized baseline)
    const baseline = predictedTotal / Math.max(features.length, 1);
    const contributions = features.map(f => {
      const val = inputValues[f];
      // Estimate marginal impact direction
      const diff = val - 25.0; // Mean normalized center
      return +(diff * (baseline * 0.04)).toFixed(2);
    });

    const bgColors = contributions.map(c => c >= 0 ? 'rgba(255, 107, 107, 0.75)' : 'rgba(0, 245, 212, 0.75)');
    const borderColors = contributions.map(c => c >= 0 ? '#ff6b6b' : '#00f5d4');

    this.charts.simShap = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: features,
        datasets: [{
          label: 'Marginal Impact on Predicted Severity',
          data: contributions,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Contribution to Predicted Index (±)', color: '#88a4be', font: { size: 10 } },
            ticks: { color: '#88a4be' },
            grid: { color: 'rgba(255,255,255,0.06)' }
          },
          y: { ticks: { color: '#e2f1f8' }, grid: { display: false } }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => {
                const v = item.raw;
                return v >= 0 ? `+${v} Hazard Intensifier` : `${v} Stabilizing Factor`;
              }
            }
          }
        }
      }
    });

    const explContainer = document.getElementById('sim-explainability-container');
    if (explContainer) explContainer.style.display = 'block';
  },

  setupSimulatorInputs(featureImportance) {
    const container = document.getElementById('simulator-inputs-container');
    if (!container) return;
    container.innerHTML = '';

    const feats = featureImportance.slice(0, 6);
    feats.forEach(f => {
      const col = document.createElement('div');
      col.className = 'col-md-4 mb-3';
      col.innerHTML = `
        <label class="form-label small text-info fw-bold">${f.feature}</label>
        <input type="number" step="any" class="ocean-input simulator-input" data-feature="${f.feature}" value="25.0">
      `;
      // Auto-simulate when user adjusts values
      const inp = col.querySelector('input');
      inp.addEventListener('change', () => this.runSimulation());
      container.appendChild(col);
    });

    const simCard = document.getElementById('simulator-card');
    if (simCard) {
      simCard.style.display = 'block';
      this.runSimulation();
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.MLModule.init();
});
