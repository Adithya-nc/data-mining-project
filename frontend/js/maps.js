/**
 * MarineMine Pro — Geographical Hotspot Analysis Module (Leaflet.js)
 * Interactive GIS mapping of marine pollution hotspots and risk concentrations
 */

window.MapsModule = {
  map: null,
  markerLayer: null,
  gyreLayer: null,
  graticuleLayer: null,
  showGyres: true,
  tilesInitialized: false,
  pendingPoints: null,
  _initRetryCount: 0,
  _resizeObserver: null,

  init() {
    const mapContainer = document.getElementById('hotspot-map');
    if (!mapContainer) return;

    if (typeof L === 'undefined') {
      console.error('Leaflet library (L) is not loaded.');
      mapContainer.innerHTML = '<div class="d-flex align-items-center justify-content-center h-100 text-warning p-4 text-center">⚠️ Map library failed to load. Please refresh the page.</div>';
      return;
    }

    // Do not initialize Leaflet if container is completely hidden (width/height 0)
    // Leaflet requires visible DOM container to compute coordinate projections
    if (mapContainer.clientWidth === 0 || mapContainer.clientHeight === 0 || mapContainer.offsetParent === null) {
      if (this._initRetryCount < 15) {
        this._initRetryCount += 1;
        setTimeout(() => this.init(), 80);
      }
      return;
    }
    this._initRetryCount = 0;

    if (this.map) {
      try {
        this.map.invalidateSize(true);
      } catch (e) {
        console.warn('Map invalidateSize caught:', e);
      }
      return;
    }

    try {
      // Initialize Leaflet Map centered over global oceans
      this.map = L.map('hotspot-map', {
        center: [15.0, 0.0],
        zoom: 2,
        minZoom: 2,
        maxZoom: 16,
        scrollWheelZoom: true,
      });

      // High-availability dark ocean basemap tile layer with OSM fallback
      const cartoUrl = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
      const osmUrl = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
      const tileLayer = L.tileLayer(cartoUrl, {
        attribution: '&copy; OpenStreetMap &copy; CARTO | MarineMine Pro GIS',
        maxZoom: 19,
      });

      let switchedToFallback = false;
      tileLayer.on('tileerror', () => {
        if (!switchedToFallback) {
          switchedToFallback = true;
          tileLayer.setUrl(osmUrl);
        }
      });
      tileLayer.addTo(this.map);

      this.graticuleLayer = L.layerGroup().addTo(this.map);
      this.renderOceanGraticule();

      this.markerLayer = L.layerGroup().addTo(this.map);
      this.tilesInitialized = true;

      // Keep map sized properly when container visibility or dimensions change
      if (typeof ResizeObserver !== 'undefined' && !this._resizeObserver) {
        this._resizeObserver = new ResizeObserver(() => {
          if (this.map && mapContainer.clientWidth > 0 && mapContainer.clientHeight > 0) {
            this.map.invalidateSize(true);
          }
        });
        this._resizeObserver.observe(mapContainer);
      }

      // Filter event listeners
      const regionFilter = document.getElementById('hotspot-filter-region');
      const typeFilter = document.getElementById('hotspot-filter-type');

      if (regionFilter && !regionFilter._bound) {
        regionFilter.addEventListener('change', () => this.refresh());
        regionFilter._bound = true;
      }
      if (typeFilter && !typeFilter._bound) {
        typeFilter.addEventListener('change', () => this.refresh());
        typeFilter._bound = true;
      }

      // Force recalculation of map container geometry
      if (this.pendingPoints) {
        this.renderMapPoints(this.pendingPoints);
        this.pendingPoints = null;
      }

      setTimeout(() => {
        if (this.map) this.map.invalidateSize(true);
      }, 100);
      setTimeout(() => {
        if (this.map) this.map.invalidateSize(true);
      }, 360);

    } catch (err) {
      console.error('Error initializing Leaflet map:', err);
    }
  },

  renderOceanGraticule() {
    if (!this.graticuleLayer) return;
    this.graticuleLayer.clearLayers();
    const gridStyle = {
      color: 'rgba(56, 224, 242, 0.14)',
      weight: 1,
      dashArray: '4 4',
      interactive: false,
    };
    for (let lat = -60; lat <= 60; lat += 30) {
      L.polyline([[lat, -180], [lat, 180]], gridStyle).addTo(this.graticuleLayer);
    }
    for (let lng = -150; lng <= 180; lng += 30) {
      L.polyline([[-80, lng], [80, lng]], gridStyle).addTo(this.graticuleLayer);
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
        } catch (e) { }
      }, 50);
    }
  },

  async refresh() {
    const mapContainer = document.getElementById('hotspot-map');
    if (!mapContainer) return;

    // Ensure map is initialized once container is visible
    if (!this.map) {
      this.init();
    }
    if (this.map) {
      try {
        this.map.invalidateSize(true);
      } catch (e) { }
    }

    try {
      const reg = document.getElementById('hotspot-filter-region')?.value || 'All';
      const typ = document.getElementById('hotspot-filter-type')?.value || 'All';

      const queryParams = new URLSearchParams();
      if (reg && reg !== 'All') queryParams.append('region', reg);
      if (typ && typ !== 'All') queryParams.append('type', typ);

    const res = await apiRequest(`/api/hotspots?${queryParams.toString()}`);
    if (res && res.points) {
      if (!this.map || !this.markerLayer) {
        this.pendingPoints = res.points;
        this.init();
      } else {
        this.renderMapPoints(res.points);
      }
      this.renderHotspotsTable(res.hotspots || []);
      this.updateFilterDropdowns(res.points);
    }
  } catch(err) {
    console.error('Error fetching hotspot geospatial data:', err);
  }
},

  renderHotspots(data) {
  if (!this.map) {
    this.init();
  }
  if (!this.map) return;

  try {
    this.map.invalidateSize(true);
  } catch (e) { }

  if (this.markerLayer) {
    this.markerLayer.clearLayers();
  } else {
    this.markerLayer = L.layerGroup().addTo(this.map);
  }

  const points = data.points || [];
  this.updateFilterDropdowns(points);
  this.renderHotspotsTable(data.hotspots || []);

  if (!points.length) return;

  const getColor = (val) => {
    if (val > 100) return '#f43f5e'; // Critical High Risk
    if (val > 50) return '#f59e0b';  // Moderate Risk
    return '#00d2ff';               // Low / Baseline
  };

  const bounds = [];

  points.forEach(p => {
    if (typeof p.lat !== 'number' || typeof p.lng !== 'number' || isNaN(p.lat) || isNaN(p.lng)) {
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
        <div style="color: #071322; font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; min-width: 170px; padding: 4px;">
          <h6 style="margin: 0 0 6px 0; font-weight: 700; color: #0077b6;">📍 Pollution Observation</h6>
          <div style="margin-bottom: 2px;"><strong>Coordinates:</strong> ${p.lat.toFixed(3)}, ${p.lng.toFixed(3)}</div>
          <div style="margin-bottom: 2px;"><strong>Region:</strong> ${p.region || 'Ocean Basin'}</div>
          <div style="margin-bottom: 4px;"><strong>Category:</strong> <span style="background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: 600;">${p.category || 'General'}</span></div>
          <div><strong>Measurement:</strong> <span style="font-weight: 800; color: #f43f5e; font-size: 13px;">${p.val}</span></div>
        </div>
      `);

    this.markerLayer.addLayer(marker);
    bounds.push([p.lat, p.lng]);
  });

  if (bounds.length && this.map) {
    setTimeout(() => {
      try {
        this.map.invalidateSize(true);
        this.map.fitBounds(bounds, { padding: [40, 40], maxZoom: 6 });
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
