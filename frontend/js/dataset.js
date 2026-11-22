/**
 * MarineMine Pro — Dataset Management Module
 * File uploads, CSV schema validation, previews, and column-mapping
 */

window.DatasetModule = {
  previewRows: [],

  init() {
    const fileInput = document.getElementById('dataset-file-input');
    const dropzone = document.getElementById('dataset-dropzone');
    const btnLoadSample = document.getElementById('btn-load-sample');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', (e) => {
        // Prevent recursive trigger if user clicked browse button
        if (e.target && e.target.id === 'btn-browse-file') return;
        fileInput.click();
      });

      const btnBrowse = document.getElementById('btn-browse-file');
      if (btnBrowse) {
        btnBrowse.addEventListener('click', (e) => {
          e.stopPropagation();
          fileInput.click();
        });
      }

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });

      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
          this.uploadFile(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
          this.uploadFile(fileInput.files[0]);
        }
      });
    }

    if (btnLoadSample) {
      btnLoadSample.addEventListener('click', async () => {
        try {
          btnLoadSample.disabled = true;
          btnLoadSample.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
          await apiRequest('/api/dataset/load-sample', { method: 'POST' });
          showToast('Sample Marine Pollution dataset loaded successfully!', 'success');
          await syncDatasetStatus();
          this.loadPreview();
        } catch (err) {
          // Handled in apiRequest
        } finally {
          btnLoadSample.disabled = false;
          btnLoadSample.innerHTML = '🌊 Load Benchmark Dataset';
        }
      });
    }

    // Column Mapping Form
    const mapForm = document.getElementById('form-column-mapping');
    if (mapForm) {
      mapForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const mapping = {};
        const fields = ['date', 'region', 'pollution_type', 'pollution_value', 'latitude', 'longitude', 'temperature', 'pH', 'dissolved_oxygen', 'salinity', 'turbidity'];
        fields.forEach(f => {
          const sel = document.getElementById(`map-field-${f}`);
          if (sel && sel.value) {
            // Mapping is: dataset_column_name -> standard_role_name
            mapping[sel.value] = f;
          }
        });

        if (!Object.keys(mapping).length) {
          showToast('Please select at least one column to map', 'warning');
          return;
        }

        try {
          await apiRequest('/api/dataset/map-columns', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mapping })
          });
          showToast('Column mapping applied successfully!', 'success');
          await syncDatasetStatus();
          this.loadPreview();
        } catch (err) {
          // Handled
        }
      });
    }

    this.loadPreview();
  },

  async uploadFile(file) {
    if (!file.name.endsWith('.csv')) {
      showToast('Please upload a valid CSV file (.csv)', 'warning');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      showToast(`Uploading and validating '${file.name}'...`, 'info');
      const res = await apiRequest('/api/dataset/upload', {
        method: 'POST',
        body: formData
      });
      showToast(`Dataset '${res.filename}' loaded (${res.rows.toLocaleString()} records)`, 'success');
      await syncDatasetStatus();
      this.loadPreview();
    } catch (err) {
      // Handled
    }
  },

  async loadPreview() {
    try {
      const data = await apiRequest('/api/dataset/preview?limit=15');
      this.renderPreviewTable(data.columns, data.preview);

      // Display metadata summary
      const setTxt = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };

      setTxt('meta-filename', AppState.datasetInfo?.name || AppState.datasetInfo?.dataset_name || 'marine_pollution.csv');
      setTxt('meta-rows', (AppState.datasetInfo?.rows || AppState.datasetInfo?.total_records || 0).toLocaleString());
      setTxt('meta-cols', (AppState.datasetInfo?.columns?.length || AppState.datasetInfo?.total_columns || 0));
      setTxt('meta-size', AppState.datasetInfo?.file_size || AppState.datasetInfo?.file_size_human || 'N/A');

      this.populateMappingSelectors(data.columns);
    } catch (err) {
      console.warn('Could not load preview table:', err);
    }
  },

  renderPreviewTable(columns, rows) {
    const tableHead = document.getElementById('preview-table-head');
    const tableBody = document.getElementById('preview-table-body');
    if (!tableHead || !tableBody) return;

    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    // Headers
    const trHead = document.createElement('tr');
    trHead.innerHTML = '<th>#</th>' + columns.map(c => `<th>${c}</th>`).join('');
    tableHead.appendChild(trHead);

    // Rows
    rows.forEach((row, idx) => {
      const tr = document.createElement('tr');
      const cells = columns.map(c => `<td>${row[c] !== null && row[c] !== undefined ? row[c] : '<span class="text-muted">null</span>'}</td>`).join('');
      tr.innerHTML = `<td>${idx + 1}</td>${cells}`;
      tableBody.appendChild(tr);
    });
  },

  populateMappingSelectors(columns) {
    const fields = ['date', 'region', 'pollution_type', 'pollution_value', 'latitude', 'longitude', 'temperature', 'pH', 'dissolved_oxygen', 'salinity', 'turbidity'];
    fields.forEach(f => {
      const sel = document.getElementById(`map-field-${f}`);
      if (!sel) return;
      sel.innerHTML = '<option value="">-- Auto-Detect / Skip --</option>';
      columns.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.textContent = col;
        if (col.toLowerCase() === f.toLowerCase() || col.toLowerCase().includes(f.toLowerCase())) {
          opt.selected = true;
        }
        sel.appendChild(opt);
      });
    });
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.DatasetModule.init();
});
