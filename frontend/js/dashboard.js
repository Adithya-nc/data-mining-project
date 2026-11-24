/**
 * MarineMine Pro — Dashboard Module
 * Dynamic telemetry KPIs and interactive Chart.js visualizations
 */

window.DashboardModule = {
  charts: {},

  async refresh() {
    if (!AppState.datasetLoaded) {
      const emptyState = document.getElementById('dashboard-empty-state');
      if (emptyState) emptyState.style.display = 'block';
      const content = document.getElementById('dashboard-content');
      if (content) content.style.display = 'none';
      return;
    }

    const emptyState = document.getElementById('dashboard-empty-state');
    if (emptyState) emptyState.style.display = 'none';
    const content = document.getElementById('dashboard-content');
    if (content) content.style.display = 'block';

    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    try {
      const [trendsRes, statsRes, qualityRes, hotspotsRes] = await Promise.allSettled([
        apiRequest('/api/eda/trends'),
        apiRequest('/api/eda/statistics'),
        apiRequest('/api/quality/summary'),
        apiRequest('/api/hotspots'),
      ]);

      const trendsData = trendsRes.status === 'fulfilled' ? trendsRes.value : { trends: [], categories: [], regions: [] };
      const qualityData = qualityRes.status === 'fulfilled' ? qualityRes.value : { quality_score: 95, missing_rate_pct: 0, total_missing_cells: 0 };
      const hotspotsData = hotspotsRes.status === 'fulfilled' ? hotspotsRes.value : { hotspots: [], points: [] };

      // Update KPI Cards
      const totalRows = AppState.datasetInfo?.rows || AppState.datasetInfo?.total_records || 0;
      setTxt('kpi-total-records', totalRows.toLocaleString());
      setTxt('kpi-missing-values', `${qualityData.missing_rate_pct || 0}% (${qualityData.total_missing_cells || 0})`);
      setTxt('kpi-quality-score', `${qualityData.quality_score || 0}/100`);

      const regionsCount = trendsData.regions?.length || 0;
      setTxt('kpi-regions-count', regionsCount);

      const categoriesCount = trendsData.categories?.length || 0;
      setTxt('kpi-categories-count', categoriesCount);

      const hotspotsCount = hotspotsData.hotspots?.length || hotspotsData.points?.length || 0;
      setTxt('kpi-hotspots-count', hotspotsCount);

      // Render Dashboard Charts
      if (trendsData.categories && trendsData.categories.length) {
        this.renderCategoryChart(trendsData.categories);
      }
      if (trendsData.trends && trendsData.trends.length) {
        this.renderTimelineChart(trendsData.trends);
        this.renderParameterTrends(trendsData.trends);
      }
      if (trendsData.regions && trendsData.regions.length) {
        this.renderRegionChart(trendsData.regions);
      }
      this.renderDepthProfileChart();

    } catch (err) {
      console.error('Error refreshing dashboard:', err);
    }
  },

  renderCategoryChart(categories) {
    const ctx = document.getElementById('chart-dashboard-categories');
    if (!ctx) return;

    if (this.charts.category) this.charts.category.destroy();

    const labels = categories.map(c => c.category);
    const data = categories.map(c => c.count);

    this.charts.category = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: [
            '#00d2ff', '#0077b6', '#00f2fe', '#38ef7d', '#f59e0b', '#f43f5e', '#8b5cf6'
          ],
          borderWidth: 2,
          borderColor: '#0d1e35',
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: { color: '#e6f1ff', font: { family: 'Plus Jakarta Sans', size: 11 } }
          }
        }
      }
    });
  },

  renderTimelineChart(trends) {
    const ctx = document.getElementById('chart-dashboard-timeline');
    if (!ctx) return;

    if (this.charts.timeline) this.charts.timeline.destroy();

    const labels = trends.map(t => t.date);
    const values = trends.map(t => t.value);

    this.charts.timeline = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Average Pollution Observation',
          data: values,
          borderColor: '#00d2ff',
          backgroundColor: 'rgba(0, 210, 255, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointRadius: 3,
          pointBackgroundColor: '#00d2ff',
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        },
        plugins: {
          legend: { labels: { color: '#e6f1ff' } }
        }
      }
    });
  },

  renderRegionChart(regions) {
    const ctx = document.getElementById('chart-dashboard-regions');
    if (!ctx) return;

    if (this.charts.regions) this.charts.regions.destroy();

    const labels = regions.map(r => r.region);
    const data = regions.map(r => r.count);

    this.charts.regions = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Observations by Region',
          data: data,
          backgroundColor: 'rgba(0, 119, 182, 0.75)',
          borderColor: '#00d2ff',
          borderWidth: 1,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8892b0' }, grid: { display: false } },
          y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  },

  renderParameterTrends(trends) {
    const ctx = document.getElementById('chart-dashboard-params');
    if (!ctx) return;

    if (this.charts.params) this.charts.params.destroy();

    const labels = trends.map(t => t.date);
    const values = trends.map(t => t.value);

    this.charts.params = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Observation Density',
          data: values,
          backgroundColor: 'rgba(0, 242, 254, 0.6)',
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#8892b0' }, grid: { display: false } },
          y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  },

  renderDepthProfileChart() {
    if (window.MapsModule && typeof window.MapsModule.renderDepthProfileChart === 'function') {
      window.MapsModule.renderDepthProfileChart();
    }
  }
};
