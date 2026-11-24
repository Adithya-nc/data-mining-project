/**
 * MarineMine Pro — Geographical Hotspot Analysis Module (Leaflet.js)
 * Interactive GIS mapping of marine pollution hotspots and risk concentrations
 * Advanced GIS Features:
 * - Animated Ocean Current Vectors & Gyre Convergence Flow
 * - Bathymetric Depth Profile Cross-Section Chart (0m - 4000m)
 */

window.MapsModule = {
  map: null,
  markerLayer: null,
  gyreLayer: null,
  showGyres: true,
  tilesInitialized: false,
  depthChart: null,

  init() {
    const mapContainer = document.getElementById('hotspot-map');
    if (!mapContainer) return;

    if (mapContainer.clientWidth === 0 || mapContainer.clientHeight === 0 || mapContainer.offsetParent === null) {
      return;
    }

    if (this.map) {
      try {
        this.map.invalidateSize(true);
      } catch (e) {
        console.warn('Map invalidateSize caught:', e);
      }
      return;
    }

    try {
      this.map = L.map('hotspot-map', {
        center: [18.0, 15.0],
        zoom: 2,
        minZoom: 2,
        maxZoom: 16,
        scrollWheelZoom: true,
      });

      // High-availability dark ocean basemap tile layer
      const tileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO | MarineMine Pro GIS',
        maxZoom: 19,
        crossOrigin: true,
      });
      tileLayer.addTo(this.map);

      this.markerLayer = L.layerGroup().addTo(this.map);
      this.gyreLayer = L.layerGroup().addTo(this.map);
      this.tilesInitialized = true;

      // Filter event listeners
      const regionFilter = document.getElementById('hotspot-filter-region');
      const typeFilter = document.getElementById('hotspot-filter-type');
      const btnGyres = document.getElementById('btn-toggle-gyres');

      if (regionFilter && !regionFilter._bound) {
        regionFilter.addEventListener('change', () => this.refresh());
        regionFilter._bound = true;
      }
      if (typeFilter && !typeFilter._bound) {
        typeFilter.addEventListener('change', () => this.refresh());
        typeFilter._bound = true;
      }
      if (btnGyres && !btnGyres._bound) {
        btnGyres.addEventListener('click', () => {
          this.showGyres = !this.showGyres;
          btnGyres.classList.toggle('active', this.showGyres);
          btnGyres.textContent = this.showGyres ? '🌊 Gyre Flow Vectors: ON' : '🌊 Gyre Flow Vectors: OFF';
          if (this.showGyres) {
            this.map.addLayer(this.gyreLayer);
          } else {
            this.map.removeLayer(this.gyreLayer);
          }
        });
        btnGyres._bound = true;
      }

      this.renderOceanCurrentVectors();
      this.renderDepthProfileChart();

      setTimeout(() => {
        if (this.map) this.map.invalidateSize(true);
      }, 100);

    } catch (err) {
      console.error('Error initializing Leaflet map:', err);
    }
  },

  invalidateSize() {
    if (!this.map) {
      this.init();
    }
    if (this.map) {
      setTimeout(() => {
        try {
          this.map.invalidateSize(true);
        } catch (e) {}
      }, 50);
    }
  },

  renderOceanCurrentVectors() {
    if (!this.gyreLayer) return;
    this.gyreLayer.clearLayers();

    // Major Global Ocean Gyres & Boundary Currents
    const gyres = [
      {
        name: 'North Pacific Subtropical Gyre (Great Pacific Garbage Patch)',
        coords: [
          [20, 130], [28, 145], [35, 170], [38, -160], [34, -140],
          [28, -130], [20, -140], [15, -170], [14, 150], [20, 130]
        ],
        color: '#00f5d4',
        mechanism: 'Anticyclonic circulation driven by Northeast Trade Winds & Westerlies creates an Ekman convergence zone trapping ~80,000 tonnes of floating microplastics.'
      },
      {
        name: 'North Atlantic Subtropical Gyre & Gulf Stream (Sargasso Sea)',
        coords: [
          [18, -75], [28, -78], [38, -65], [44, -40], [42, -20],
          [32, -18], [22, -25], [16, -45], [15, -65], [18, -75]
        ],
        color: '#00bbf9',
        mechanism: 'The western boundary Gulf Stream propels pollutants northeastward into the central calm Sargasso Sea vortex.'
      },
      {
        name: 'South Pacific Subtropical Gyre',
        coords: [
          [-15, -80], [-25, -95], [-35, -120], [-38, -150], [-35, 175],
          [-22, 160], [-15, 180], [-12, -110], [-15, -80]
        ],
        color: '#fee440',
        mechanism: 'Humboldt (Peru) Current drives coastal plastic runoff westward into the remote Southern Pacific gyre.'
      },
      {
        name: 'South Atlantic Gyre & Benguela Current',
        coords: [
          [-12, -35], [-22, -38], [-35, -30], [-38, -10], [-32, 10],
          [-20, 12], [-12, 0], [-10, -20], [-12, -35]
        ],
        color: '#8338ec',
        mechanism: 'Counter-clockwise circulation fed by Brazilian runoff and southern industrial maritime corridors.'
      },
      {
        name: 'Indian Ocean Monsoon Drift & Agulhas Current',
        coords: [
          [10, 55], [15, 70], [10, 85], [-5, 90], [-20, 85],
          [-28, 60], [-25, 40], [-10, 45], [0, 50], [10, 55]
        ],
        color: '#ff6b6b',
        mechanism: 'Semiannual reversing monsoon currents drive high pollutant fluxes between the Arabian Sea, Bay of Bengal, and equatorial convergence zone.'
      },
      {
        name: 'Antarctic Circumpolar Current (West Wind Drift)',
        coords: [
          [-55, -170], [-58, -110], [-56, -60], [-54, 0], [-57, 60],
          [-55, 120], [-56, 170]
        ],
        color: '#38bdf8',
        mechanism: 'The worlds strongest ocean current acts as a circumpolar hydrodynamic barrier isolating Antarctic waters while shearing debris.'
      }
    ];

    gyres.forEach(g => {
      const polyline = L.polyline(g.coords, {
        color: g.color,
        weight: 2.8,
        opacity: 0.85,
        className: 'gyre-current-path'
      });

      polyline.bindPopup(`
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; min-width: 200px; color: #e2f1f8; background: #041426; padding: 6px; border-radius: 6px;">
          <div style="font-size: 10px; font-family: monospace; color: ${g.color}; font-weight: 700;">🌊 OCEAN GYRE VECTOR</div>
          <h6 style="font-weight: 700; margin: 4px 0; color: #ffffff;">${g.name}</h6>
          <p style="font-size: 11px; margin: 0; line-height: 1.4; color: #94c5d8;">${g.mechanism}</p>
        </div>
      `);

      this.gyreLayer.addLayer(polyline);
    });
  },

  renderDepthProfileChart() {
    const ctx = document.getElementById('chart-depth-profile');
    if (!ctx) return;
    if (this.depthChart) this.depthChart.destroy();

    const depths = ['0m (Surface)', '100m', '200m (Epipelagic)', '500m (Thermocline)', '1,000m (Mesopelagic)', '2,000m', '3,000m (Bathypelagic)', '4,000m (Abyssal Floor)'];
    const temps = [24.8, 21.2, 15.6, 7.8, 4.2, 2.8, 2.1, 1.8];
    const dissolvedO2 = [8.4, 7.8, 6.2, 3.1, 2.2, 4.1, 4.8, 5.2]; // Shows Oxygen Minimum Zone (OMZ) at 500-1000m
    const microplastics = [430, 310, 190, 85, 60, 48, 120, 260]; // Surface peak & benthic sedimentation settling

    this.depthChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: depths,
        datasets: [
          {
            label: 'Microplastics (p/m³)',
            data: microplastics,
            borderColor: '#ff6b6b',
            backgroundColor: 'rgba(255, 107, 107, 0.12)',
            yAxisID: 'y1',
            tension: 0.35,
            fill: true,
            borderWidth: 2.2,
            pointRadius: 4
          },
          {
            label: 'Water Temp (°C)',
            data: temps,
            borderColor: '#fee440',
            backgroundColor: 'transparent',
            yAxisID: 'y',
            tension: 0.35,
            borderWidth: 2,
            pointRadius: 3
          },
          {
            label: 'Dissolved O₂ (mg/L)',
            data: dissolvedO2,
            borderColor: '#00f5d4',
            backgroundColor: 'transparent',
            yAxisID: 'y',
            tension: 0.35,
            borderWidth: 2,
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { ticks: { color: '#8892b0', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'Temp (°C) / O₂ (mg/L)', color: '#8892b0' },
            ticks: { color: '#8892b0' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: { display: true, text: 'Microplastics (p/m³)', color: '#ff6b6b' },
            ticks: { color: '#ff6b6b' },
            grid: { drawOnChartArea: false }
          }
        },
        plugins: {
          legend: { labels: { color: '#e2f1f8', boxWidth: 12, font: { size: 11 } } }
        }
      }
    });
  },

  async refresh() {
    const mapContainer = document.getElementById('hotspot-map');
    if (!mapContainer) return;

    if (!this.map) {
      this.init();
    }
    if (this.map) {
      try {
        this.map.invalidateSize(true);
      } catch (e) {}
    }

    try {
      const reg = document.getElementById('hotspot-filter-region')?.value || 'All';
      const typ = document.getElementById('hotspot-filter-type')?.value || 'All';

      const queryParams = new URLSearchParams();
      if (reg !== 'All') queryParams.append('region', reg);
      if (typ !== 'All') queryParams.append('pollutant_type', typ);

      const res = await apiRequest(`/api/hotspots?${queryParams.toString()}`);
      if (res && res.points) {
        this.renderMapPoints(res.points);
        this.renderHotspotsTable(res.hotspots || []);
        this.updateFilterDropdowns(res.points);
      }
    } catch (err) {
      console.warn('Could not refresh hotspots:', err);
    }
  },

  renderMapPoints(points) {
    if (!this.markerLayer || !this.map) return;
    this.markerLayer.clearLayers();

    if (!points || !points.length) return;

    const bounds = [];

    const getColor = (val) => {
      if (val > 450) return '#ff4d4f';
      if (val > 250) return '#faad14';
      if (val > 100) return '#00d2ff';
      return '#52c41a';
    };

    points.forEach(p => {
      if (p.lat === undefined || p.lng === undefined || p.lat === null || p.lng === null) {
        return;
      }

      const color = getColor(p.val);
      const radius = Math.min(Math.max(p.val / 8, 6), 18);

      const marker = L.circleMarker([p.lat, p.lng], {
        radius: radius,
        fillColor: color,
        color: '#ffffff',
        weight: 1.5,
        opacity: 0.95,
        fillOpacity: 0.75,
      });

      marker.bindPopup(`
        <div style="color: #e2f1f8; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; min-width: 175px; padding: 4px;">
          <h6 style="margin: 0 0 6px 0; font-weight: 700; color: #00f5d4;">📍 Pollution Observation</h6>
          <div style="margin-bottom: 3px; color: #bfe0f5;"><strong>Coordinates:</strong> <span style="color: #ffffff;">${p.lat.toFixed(3)}, ${p.lng.toFixed(3)}</span></div>
          <div style="margin-bottom: 3px; color: #bfe0f5;"><strong>Region:</strong> <span style="color: #00bbf9;">${p.region || 'Ocean Basin'}</span></div>
          <div style="margin-bottom: 4px; color: #bfe0f5;"><strong>Category:</strong> <span style="background: rgba(0, 187, 249, 0.2); color: #00bbf9; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid rgba(0, 187, 249, 0.4);">${p.category || 'General'}</span></div>
          <div style="color: #bfe0f5;"><strong>Measurement:</strong> <span style="font-weight: 800; color: #ff6b6b; font-size: 13px;">${p.val}</span></div>
        </div>
      `);

      this.markerLayer.addLayer(marker);
      bounds.push([p.lat, p.lng]);
    });

    if (bounds.length && this.map) {
      setTimeout(() => {
        try {
          this.map.invalidateSize(true);
        } catch (e) {
          this.map.setView([15.0, 0.0], 2);
        }
      }, 100);
    }
  },

  renderHotspotsTable(hotspots) {
    const tbody = document.getElementById('hotspots-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!hotspots.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-muted text-center py-3">No distinct clusters or hotspots identified in the current query.</td></tr>';
      return;
    }

    hotspots.slice(0, 15).forEach((h, idx) => {
      const tr = document.createElement('tr');
      const rank = idx + 1;
      const rankBadge = rank <= 3 ? `🥇 #${rank}` : `#${rank}`;

      const latVal = h.grid_lat !== undefined ? h.grid_lat : h.latitude;
      const lonVal = h.grid_lon !== undefined ? h.grid_lon : h.longitude;
      const coords = (latVal !== undefined && lonVal !== undefined && latVal !== null && lonVal !== null)
        ? `${Number(latVal).toFixed(2)}, ${Number(lonVal).toFixed(2)}`
        : 'N/A';

      const meanVal = h.mean_value !== undefined ? h.mean_value : (h.mean_pollution !== undefined ? h.mean_pollution : h.pollution_value);
      const meanDisplay = meanVal !== undefined && meanVal !== null ? Number(meanVal).toFixed(2) : 'N/A';

      tr.innerHTML = `
        <td class="fw-bold ${rank <= 3 ? 'text-warning' : 'text-muted'}">${rankBadge}</td>
        <td class="fw-bold text-info">${h.region || 'Ocean Basin'}</td>
        <td>${h.category || h.pollution_type || 'General Marine'}</td>
        <td class="font-monospace">${coords}</td>
        <td class="fw-bold text-danger">${meanDisplay}</td>
        <td><span class="badge ${rank <= 3 ? 'badge-risk' : 'badge-warning'}">${rank <= 3 ? 'High Risk' : 'Monitored'}</span></td>
      `;
      tbody.appendChild(tr);
    });
  },

  updateFilterDropdowns(points) {
    const regSelect = document.getElementById('hotspot-filter-region');
    const typSelect = document.getElementById('hotspot-filter-type');

    if (regSelect && regSelect.options.length <= 1) {
      const regions = [...new Set(points.map(p => p.region).filter(Boolean))].sort();
      regions.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r;
        opt.textContent = r;
        regSelect.appendChild(opt);
      });
    }

    if (typSelect && typSelect.options.length <= 1) {
      const types = [...new Set(points.map(p => p.category).filter(Boolean))].sort();
      types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        typSelect.appendChild(opt);
      });
    }
  }
};
