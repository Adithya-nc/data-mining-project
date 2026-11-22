/**
 * MarineMine Pro — Analytical Report Generator Module
 * Export Markdown, CSV, and Excel reports with live interactive preview
 */

window.ReportsModule = {
  init() {
    const btnMd = document.getElementById('btn-gen-report-md');
    const btnCsv = document.getElementById('btn-gen-report-csv');
    const btnXlsx = document.getElementById('btn-gen-report-excel');

    if (btnMd) {
      btnMd.addEventListener('click', () => this.generate('markdown'));
    }
    if (btnCsv) {
      btnCsv.addEventListener('click', () => this.generate('csv'));
    }
    if (btnXlsx) {
      btnXlsx.addEventListener('click', () => this.generate('excel'));
    }
  },

  async generate(format) {
    if (!AppState.datasetLoaded) {
      showToast('Please load a dataset before generating reports', 'warning');
      return;
    }

    try {
      showToast(`Generating analytical report (${format.toUpperCase()})...`, 'info');
      const res = await apiRequest('/api/reports/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          report_type: 'full_audit',
          format: format
        })
      });

      showToast(res.message, 'success');

      if (format === 'markdown' && res.preview) {
        const viewer = document.getElementById('report-markdown-preview');
        if (viewer) {
          viewer.textContent = res.preview;
          document.getElementById('report-preview-container').style.display = 'block';
        }
      }

      if (res.download_url) {
        window.open(res.download_url, '_blank');
      }
    } catch (err) {
      // Handled
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.ReportsModule.init();
});

