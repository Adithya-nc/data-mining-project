/**
 * MarineMine Pro — Data Quality Audit Module
 * Rigorous validation, completeness scoring, outlier detection, and missingness breakdown
 */

window.QualityModule = {
  charts: {},

  async refresh() {
    if (!AppState.datasetLoaded) return;

    try {
      const summary = await apiRequest('/api/quality/summary');

      const setTxt = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };

      setTxt('q-score-val', `${summary.quality_score}/100`);
      const bar = document.getElementById('q-score-bar');
      if (bar) bar.style.width = `${summary.quality_score}%`;

      setTxt('q-total-records', summary.total_records.toLocaleString());
      setTxt('q-total-features', summary.total_features);
      setTxt('q-completeness', `${summary.completeness_pct}%`);
      setTxt('q-missing-cells', `${summary.total_missing_cells} (${summary.missing_rate_pct}%)`);
      setTxt('q-duplicate-rows', `${summary.duplicate_rows} (${summary.duplicate_rate_pct}%)`);
      setTxt('q-outliers-iqr', summary.outlier_count_iqr.toLocaleString());

      // Render Missing Values Chart
      this.renderMissingChart(summary.column_missing_breakdown || []);

      // Checks, Warnings, Errors
      this.renderCheckList('q-passed-checks', summary.passed_checks || [], '✅', 'text-success');
      this.renderCheckList('q-warnings', summary.warnings || [], '⚠️', 'text-warning');
      this.renderCheckList('q-errors', summary.errors || [], '❌', 'text-danger');

    } catch (err) {
      console.error('Error refreshing Quality Module:', err);
    }
  },

  renderMissingChart(breakdown) {
    const ctx = document.getElementById('chart-quality-missing');
    if (!ctx) return;
    if (this.charts.missing) this.charts.missing.destroy();

    const topMissing = breakdown.filter(b => b.missing_count > 0).slice(0, 10);
    const labels = topMissing.map(b => b.column);
    const data = topMissing.map(b => b.missing_percentage);

    if (!labels.length) {
      labels.push('All Columns Clean');
      data.push(0);
    }

    this.charts.missing = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Missing Rate (%)',
          data: data,
          backgroundColor: 'rgba(244, 63, 94, 0.65)',
          borderColor: '#f43f5e',
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8892b0' }, grid: { display: false } },
          y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' }, max: 100 }
        },
        plugins: {
          legend: { labels: { color: '#e6f1ff' } }
        }
      }
    });
  },

  renderCheckList(elementId, items, icon, colorClass) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.innerHTML = '';

    if (!items.length) {
      el.innerHTML = '<li class="text-muted small">None</li>';
      return;
    }

    items.forEach(item => {
      const li = document.createElement('li');
      li.className = `small mb-1 ${colorClass}`;
      li.innerHTML = `<span class="me-2">${icon}</span> ${item}`;
      el.appendChild(li);
    });
  }
};

