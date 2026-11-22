/**
 * MarineMine Pro — Exploratory Data Analysis (EDA) Module
 * Summary statistics, correlation matrices, histograms, and multidimensional parameter analysis
 */

window.EDAModule = {
  charts: {},

  init() {
    const histSelect = document.getElementById('eda-hist-feature');
    if (histSelect) {
      histSelect.addEventListener('change', () => {
        this.renderDistribution(histSelect.value);
      });
    }
  },

  async refresh() {
    if (!AppState.datasetLoaded) return;

    try {
      const [stats, corr] = await Promise.all([
        apiRequest('/api/eda/statistics'),
        apiRequest('/api/eda/correlation'),
      ]);

      this.renderStatsTable(stats.numerical_summary || []);
      this.renderCategoricalTable(stats.categorical_summary || []);
      this.renderCorrelationMatrix(corr.features || [], corr.matrix || []);

      const histSelect = document.getElementById('eda-hist-feature');
      if (histSelect && histSelect.value) {
        this.renderDistribution(histSelect.value);
      } else if (corr.features && corr.features.length) {
        this.renderDistribution(corr.features[0]);
      }
    } catch (err) {
      console.error('Error refreshing EDA:', err);
    }
  },

  renderStatsTable(summary) {
    const tbody = document.getElementById('eda-num-stats-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    summary.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="fw-bold text-info">${s.column}</td>
        <td>${s.count.toLocaleString()}</td>
        <td>${s.mean}</td>
        <td>${s.std}</td>
        <td>${s.min}</td>
        <td>${s.q25}</td>
        <td>${s.median}</td>
        <td>${s.q75}</td>
        <td>${s.max}</td>
        <td>${s.iqr}</td>
      `;
      tbody.appendChild(tr);
    });
  },

  renderCategoricalTable(summary) {
    const tbody = document.getElementById('eda-cat-stats-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    summary.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="fw-bold text-info">${s.column}</td>
        <td>${s.unique_values}</td>
        <td><span class="badge bg-secondary">${s.most_frequent}</span></td>
        <td>${s.frequency.toLocaleString()} (${s.frequency_pct}%)</td>
      `;
      tbody.appendChild(tr);
    });
  },

  renderCorrelationMatrix(features, matrix) {
    const thead = document.getElementById('eda-corr-head');
    const tbody = document.getElementById('eda-corr-body');
    if (!thead || !tbody) return;

    thead.innerHTML = '';
    tbody.innerHTML = '';

    if (!features.length) {
      tbody.innerHTML = '<tr><td class="text-muted">Insufficient numeric features for correlation analysis.</td></tr>';
      return;
    }

    // Headers
    const trHead = document.createElement('tr');
    trHead.innerHTML = '<th>Feature</th>' + features.map(f => `<th title="${f}">${f.slice(0, 10)}</th>`).join('');
    thead.appendChild(trHead);

    // Matrix Rows
    features.forEach((feat, i) => {
      const tr = document.createElement('tr');
      let cells = `<td class="fw-bold text-info">${feat}</td>`;
      matrix[i].forEach(val => {
        const abs = Math.abs(val);
        // Color scale: negative blue, positive teal/cyan, high correlation intense
        let bgColor = 'rgba(255, 255, 255, 0.02)';
        if (val > 0.5) bgColor = `rgba(0, 210, 255, ${abs * 0.7})`;
        else if (val > 0.2) bgColor = `rgba(0, 119, 182, ${abs * 0.5})`;
        else if (val < -0.2) bgColor = `rgba(244, 63, 94, ${abs * 0.5})`;

        cells += `<td style="background-color: ${bgColor}; font-family: monospace; font-size: 0.8rem;">${val.toFixed(2)}</td>`;
      });
      tr.innerHTML = cells;
      tbody.appendChild(tr);
    });
  },

  async renderDistribution(featureName) {
    const ctx = document.getElementById('chart-eda-distribution');
    if (!ctx || !featureName) return;

    try {
      const dist = await apiRequest(`/api/eda/distribution?feature=${encodeURIComponent(featureName)}`);
      if (this.charts.dist) this.charts.dist.destroy();

      const labels = dist.bins.slice(0, -1).map((b, idx) => `${b} - ${dist.bins[idx + 1]}`);

      this.charts.dist = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: `Frequency of ${featureName}`,
            data: dist.counts,
            backgroundColor: 'rgba(0, 210, 255, 0.65)',
            borderColor: '#00d2ff',
            borderWidth: 1,
            borderRadius: 4,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#8892b0', font: { size: 10 } }, grid: { display: false } },
            y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          },
          plugins: {
            legend: { labels: { color: '#e6f1ff' } }
          }
        }
      });

      // Update distribution summary indicators
      const setTxt = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };
      setTxt('eda-dist-mean', dist.mean);
      setTxt('eda-dist-std', dist.std);
      setTxt('eda-dist-min', dist.min);
      setTxt('eda-dist-max', dist.max);

    } catch (err) {
      console.warn('Could not render distribution for', featureName, err);
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.EDAModule.init();
});

