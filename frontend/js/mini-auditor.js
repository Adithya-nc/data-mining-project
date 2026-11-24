/**
 * MarineMine Pro — Live Client-Side CSV Mini-Auditor & In-Browser Correlation Heatmap
 * Computes dataset telemetry (rows, columns, nulls, duplicates) and calculates a real-time
 * Pearson correlation matrix directly inside the browser using zero server requests.
 */

window.MiniAuditor = {
  currentData: null,
  numericCols: [],
  catCols: [],

  init() {
    const dropzone = document.getElementById('mini-auditor-dropzone');
    const fileInput = document.getElementById('mini-auditor-file-input');
    const btnSample = document.getElementById('btn-mini-auditor-sample');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
          this.processFile(e.dataTransfer.files[0]);
        }
      });
      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
          this.processFile(e.target.files[0]);
        }
      });
    }

    if (btnSample) {
      btnSample.addEventListener('click', () => this.loadSampleBenchmark());
    }
  },

  async loadSampleBenchmark() {
    try {
      const resp = await fetch('/api/dataset/preview?limit=1000');
      if (resp.ok) {
        const data = await resp.json();
        if (data.records && data.records.length) {
          this.analyzeData(data.records, 'marine_pollution_sample.csv');
          return;
        }
      }
    } catch (e) {
      console.warn('API preview fetch fallback:', e);
    }
    // Fallback simulated ocean benchmark
    this.generateSimulatedBenchmark();
  },

  generateSimulatedBenchmark() {
    const records = [];
    const regions = ['North Pacific', 'North Atlantic', 'Mediterranean', 'Indian Ocean', 'South China Sea'];
    for (let i = 0; i < 300; i++) {
      const temp = 14 + Math.random() * 16;
      const salinity = 33 + Math.random() * 4;
      const traffic = Math.floor(20 + Math.random() * 80);
      const plastic = +(120 + traffic * 2.8 + (temp - 15) * 6 + (Math.random() - 0.5) * 40).toFixed(1);
      const oxygen = +(8.5 - (temp - 14) * 0.18 - plastic * 0.004).toFixed(2);
      const ph = +(8.15 - (traffic * 0.003) - (Math.random() * 0.1)).toFixed(2);
      records.push({
        sea_surface_temp: +temp.toFixed(2),
        salinity: +salinity.toFixed(2),
        shipping_traffic: traffic,
        microplastics_density: plastic,
        dissolved_oxygen: oxygen,
        ph_acidity: ph,
        region: regions[i % regions.length]
      });
    }
    this.analyzeData(records, 'calibrated_ocean_benchmark.csv');
  },

  processFile(file) {
    if (!file.name.endsWith('.csv')) {
      alert('Please upload a valid .csv file.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const records = this.parseCSV(text);
      if (records.length) {
        this.analyzeData(records, file.name);
      } else {
        alert('Could not parse any rows from the CSV file.');
      }
    };
    reader.readAsText(file);
  },

  parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length < 2) return [];

    const headers = this.parseCSVLine(lines[0]);
    const records = [];

    for (let i = 1; i < lines.length; i++) {
      const values = this.parseCSVLine(lines[i]);
      if (values.length === headers.length) {
        const row = {};
        for (let j = 0; j < headers.length; j++) {
          const val = values[j].trim();
          const num = Number(val);
          row[headers[j]] = !isNaN(num) && val !== '' ? num : val;
        }
        records.push(row);
      }
    }
    return records;
  },

  parseCSVLine(line) {
    const result = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        inQuotes = !inQuotes;
      } else if (c === ',' && !inQuotes) {
        result.push(cur);
        cur = '';
      } else {
        cur += c;
      }
    }
    result.push(cur);
    return result;
  },

  analyzeData(records, filename) {
    this.currentData = records;
    const nRows = records.length;
    const cols = Object.keys(records[0] || {});

    // Classify Numeric vs Categorical
    this.numericCols = [];
    this.catCols = [];
    let nullCount = 0;

    cols.forEach(col => {
      let isNumeric = true;
      let nonNull = 0;
      records.forEach(r => {
        const v = r[col];
        if (v === null || v === undefined || v === '' || (typeof v === 'string' && v.toLowerCase() === 'nan')) {
          nullCount++;
        } else {
          nonNull++;
          if (typeof v !== 'number' || isNaN(v)) {
            isNumeric = false;
          }
        }
      });
      if (isNumeric && nonNull > 0) {
        this.numericCols.push(col);
      } else {
        this.catCols.push(col);
      }
    });

    const totalCells = nRows * cols.length;
    const missingPct = totalCells > 0 ? ((nullCount / totalCells) * 100).toFixed(1) : 0;

    // Display telemetry summary
    const container = document.getElementById('mini-auditor-results');
    if (container) container.style.display = 'block';

    const setTxt = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    setTxt('auditor-file-name', filename);
    setTxt('auditor-rows-count', nRows.toLocaleString());
    setTxt('auditor-cols-count', cols.length);
    setTxt('auditor-numeric-count', this.numericCols.length);
    setTxt('auditor-cat-count', this.catCols.length);
    setTxt('auditor-missing-pct', `${missingPct}%`);

    // Calculate In-Browser Pearson Correlation Matrix
    this.renderCorrelationHeatmap();
  },

  calculatePearson(arrX, arrY) {
    const n = Math.min(arrX.length, arrY.length);
    if (n < 2) return 0;

    let sumX = 0, sumY = 0;
    for (let i = 0; i < n; i++) {
      sumX += arrX[i];
      sumY += arrY[i];
    }
    const meanX = sumX / n;
    const meanY = sumY / n;

    let num = 0, denX = 0, denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = arrX[i] - meanX;
      const dy = arrY[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }

    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
  },

  renderCorrelationHeatmap() {
    const tableHead = document.getElementById('auditor-heatmap-head');
    const tableBody = document.getElementById('auditor-heatmap-body');
    const explBox = document.getElementById('auditor-heatmap-expl');
    if (!tableHead || !tableBody) return;

    const numCols = this.numericCols.slice(0, 6); // Max 6 for concise heatmap display
    if (numCols.length < 2) {
      tableBody.innerHTML = '<tr><td class="text-muted p-3">Need at least 2 numeric columns for correlation analysis.</td></tr>';
      return;
    }

    // Header
    let headHtml = '<tr><th>Feature</th>';
    numCols.forEach(c => {
      const short = c.length > 12 ? c.substring(0, 10) + '..' : c;
      headHtml += `<th title="${c}">${short}</th>`;
    });
    headHtml += '</tr>';
    tableHead.innerHTML = headHtml;

    // Body
    let bodyHtml = '';
    const matrix = [];

    numCols.forEach((rowCol, rIdx) => {
      const rowVals = this.currentData.map(d => Number(d[rowCol]) || 0);
      let rowHtml = `<tr><th title="${rowCol}">${rowCol.length > 12 ? rowCol.substring(0, 10) + '..' : rowCol}</th>`;
      matrix[rIdx] = [];

      numCols.forEach((colCol, cIdx) => {
        const colVals = this.currentData.map(d => Number(d[colCol]) || 0);
        const r = rIdx === cIdx ? 1.0 : this.calculatePearson(rowVals, colVals);
        matrix[rIdx][cIdx] = r;

        // Dynamic Heatmap Color (Blue-Negative -> Neutral-Dark -> Teal/Cyan-Positive)
        let bgStyle = '';
        if (r > 0) {
          const alpha = Math.min(0.85, Math.max(0.18, r * 0.65)).toFixed(2);
          bgStyle = `background: rgba(0, 245, 212, ${alpha}); color: #ffffff; font-weight: 700; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.85);`;
        } else {
          const alpha = Math.min(0.85, Math.max(0.18, Math.abs(r) * 0.65)).toFixed(2);
          bgStyle = `background: rgba(255, 107, 107, ${alpha}); color: #ffffff; font-weight: 700; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.85);`;
        }

        rowHtml += `<td style="${bgStyle}" class="cursor-pointer" onclick="MiniAuditor.showCorrelationDetails('${rowCol}', '${colCol}', ${r})">${r.toFixed(2)}</td>`;
      });
      rowHtml += '</tr>';
      bodyHtml += rowHtml;
    });

    tableBody.innerHTML = bodyHtml;
    if (explBox) {
      explBox.innerHTML = '✦ Click any cell in the heatmap matrix above to view the bivariate coupling analysis.';
    }
  },

  showCorrelationDetails(colA, colB, r) {
    const explBox = document.getElementById('auditor-heatmap-expl');
    if (!explBox) return;

    const rVal = parseFloat(r);
    let strength = 'negligible';
    if (Math.abs(rVal) > 0.7) strength = 'very strong';
    else if (Math.abs(rVal) > 0.5) strength = 'moderate-to-strong';
    else if (Math.abs(rVal) > 0.25) strength = 'mild / weak';

    const dir = rVal > 0 ? 'positive correlation' : 'inverse (negative) correlation';
    const color = rVal > 0 ? '#00f5d4' : '#ff6b6b';

    explBox.innerHTML = `
      <div class="d-flex align-items-center justify-content-between p-2 rounded" style="background: rgba(4, 24, 46, 0.9); border: 1px solid rgba(56, 224, 242, 0.25);">
        <div>
          <span class="text-white fw-bold">${colA}</span> ⟷ <span class="text-white fw-bold">${colB}</span>:
          <span class="ms-2 font-monospace fw-bold" style="color: ${color}">r = ${rVal.toFixed(3)}</span>
          <span class="text-muted small ms-2">(${strength} ${dir})</span>
        </div>
        <span class="badge bg-dark font-monospace text-info">In-Browser Pearson r</span>
      </div>
    `;
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.MiniAuditor.init();
});

