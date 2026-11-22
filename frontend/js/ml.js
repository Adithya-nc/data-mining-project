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
          const res = await apiRequest('/api/ml/clustering', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: selectedFeats, k })
          });

          showToast(`K-Means completed! Silhouette Score: ${res.silhouette_score}`, 'success');
          const sil = document.getElementById('clustering-silhouette');
          if (sil) sil.textContent = res.silhouette_score;
          const inr = document.getElementById('clustering-inertia');
          if (inr) inr.textContent = res.inertia.toLocaleString();

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
  initAssociation() {
    const btnRun = document.getElementById('btn-run-association');
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
            min_support: minSupp,
            min_confidence: minConf,
            min_lift: minLift
          })
        });

        if (!res.suitable) {
          showToast(res.explanation, 'warning');
        } else {
          showToast(`Discovered ${res.rules_count} strong association rules!`, 'success');
        }

        this.renderAssociationRules(res);
      } catch (err) {
        // Handled
      } finally {
        btnRun.disabled = false;
        btnRun.innerHTML = '🔗 Discover Association Patterns';
      }
    });
  },

  renderAssociationRules(res) {
    const countEl = document.getElementById('assoc-rules-count');
    if (countEl) countEl.textContent = res.rules_count;
    const expEl = document.getElementById('assoc-explanation');
    if (expEl) expEl.textContent = res.explanation || '';

    const tbody = document.getElementById('assoc-rules-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!res.rules.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-muted text-center py-3">No association rules exceeded the thresholds. Try lowering minimum support or confidence.</td></tr>';
      return;
    }

    res.rules.forEach((rule) => {
      const tr = document.createElement('tr');
      const ant = Array.isArray(rule.antecedents) ? rule.antecedents.join(', ') : rule.antecedents;
      const cons = Array.isArray(rule.consequents) ? rule.consequents.join(', ') : rule.consequents;

      tr.innerHTML = `
        <td><span class="badge bg-secondary">${ant}</span></td>
        <td class="text-center text-info fw-bold">➔</td>
        <td><span class="badge bg-primary">${cons}</span></td>
        <td>${(rule.support * 100).toFixed(1)}%</td>
        <td>${(rule.confidence * 100).toFixed(1)}%</td>
        <td class="fw-bold text-success">${Number(rule.lift).toFixed(2)}</td>
      `;
      tbody.appendChild(tr);
    });

    const resCard = document.getElementById('assoc-results-card');
    if (resCard) resCard.style.display = 'block';
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
          btnTrain.innerHTML = '📈 Benchmark Regression Models';
        }
      });
    }

    if (btnSimulate) {
      btnSimulate.addEventListener('click', async () => {
        const inputs = document.querySelectorAll('.simulator-input');
        const values = {};
        inputs.forEach(inp => {
          values[inp.dataset.feature] = parseFloat(inp.value || 0.0);
        });

        try {
          btnSimulate.disabled = true;
          btnSimulate.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Simulating...';
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
        } catch (err) {
          // Handled
        } finally {
          btnSimulate.disabled = false;
          btnSimulate.innerHTML = '🧪 Predict Pollution Level';
        }
      });
    }
  },

  renderPredictionResults(res) {
    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setTxt('pred-best-model', res.best_model);
    setTxt('pred-best-r2', res.best_r2.toFixed(4));
    setTxt('pred-best-rmse', res.best_rmse.toFixed(2));

    // Benchmarks
    const tbody = document.getElementById('pred-models-table-body');
    if (tbody) {
      tbody.innerHTML = '';
      for (const [mName, m] of Object.entries(res.models)) {
        const tr = document.createElement('tr');
        const isBest = mName === res.best_model;
        tr.innerHTML = `
          <td class="fw-bold ${isBest ? 'text-info' : ''}">${isBest ? '🏆 ' : ''}${mName}</td>
          <td>${m.r2_score.toFixed(4)}</td>
          <td>${m.rmse.toFixed(2)}</td>
          <td>${m.mae.toFixed(2)}</td>
          <td>${m.mse.toFixed(2)}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // Actual vs Predicted Chart
    const ctx = document.getElementById('chart-pred-comparison');
    if (ctx && res.actual_vs_predicted) {
      if (this.charts.predComp) this.charts.predComp.destroy();
      const labels = res.actual_vs_predicted.map(d => `#${d.sample_index}`);
      const actuals = res.actual_vs_predicted.map(d => d.actual);
      const preds = res.actual_vs_predicted.map(d => d.predicted);

      this.charts.predComp = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Actual Pollution Value',
              data: actuals,
              borderColor: '#00d2ff',
              backgroundColor: 'rgba(0, 210, 255, 0.1)',
              borderWidth: 2,
              tension: 0.2,
            },
            {
              label: 'Predicted Model Value',
              data: preds,
              borderColor: '#f59e0b',
              borderDash: [5, 5],
              borderWidth: 2,
              tension: 0.2,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#8892b0' }, grid: { display: false } },
            y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          },
          plugins: { legend: { labels: { color: '#e6f1ff' } } }
        }
      });
    }

    const resCard = document.getElementById('pred-results-card');
    if (resCard) resCard.style.display = 'block';
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
      container.appendChild(col);
    });

    const simCard = document.getElementById('simulator-card');
    if (simCard) simCard.style.display = 'block';
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.MLModule.init();
});
