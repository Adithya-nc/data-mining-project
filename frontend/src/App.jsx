import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Waves,
  Database,
  ShieldCheck,
  Cpu,
  BarChart3,
  Target,
  Tag,
  GitBranch,
  TrendingUp,
  MapPin,
  FileText,
  Info,
  RefreshCw,
  Upload,
  Download,
  Play,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Filter,
  Sparkles,
  Layers,
  Activity,
  Sliders,
  Search,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  Zap,
  HelpCircle
} from 'lucide-react';

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell
} from 'recharts';

import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE = '/api';
const PALETTE = ['#0077B6', '#00B4D8', '#48CAE4', '#90E0EF', '#023E8A', '#03045E', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [datasetOverview, setDatasetOverview] = useState(null);
  const [qualityData, setQualityData] = useState(null);
  const [edaData, setEdaData] = useState(null);
  const [clusteringResult, setClusteringResult] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [associationRulesResult, setAssociationRulesResult] = useState(null);
  const [hotspotData, setHotspotData] = useState(null);
  const [markdownReport, setMarkdownReport] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Table Search & Pagination State
  const [tableSearch, setTableSearch] = useState('');
  const [tableSortCol, setTableSortCol] = useState(null);
  const [tableSortAsc, setTableSortAsc] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  // Preprocessing Form State
  const [numStrategy, setNumStrategy] = useState('mean');
  const [catStrategy, setCatStrategy] = useState('mode');
  const [removeDups, setRemoveDups] = useState(true);
  const [outlierStrategy, setOutlierStrategy] = useState('winsorize');
  const [outlierFactor, setOutlierFactor] = useState(1.5);
  const [scaleFeatures, setScaleFeatures] = useState(false);
  const [scaleMethod, setScaleMethod] = useState('standard');
  const [encodeCats, setEncodeCats] = useState(false);
  const [encodeMethod, setEncodeMethod] = useState('label');

  // Clustering Form State
  const [clusterK, setClusterK] = useState(3);
  const [selectedClusterFeatures, setSelectedClusterFeatures] = useState([]);
  const [optimalKInfo, setOptimalKInfo] = useState(null);

  // Classification Form State
  const [targetCol, setTargetCol] = useState('');
  const [classFeatures, setClassFeatures] = useState([]);
  const [testSplit, setTestSplit] = useState(0.2);

  // Prediction Form State
  const [predTarget, setPredTarget] = useState('');
  const [predFeatures, setPredFeatures] = useState([]);

  // Association Rules Form State
  const [armSupport, setArmSupport] = useState(0.05);
  const [armConfidence, setArmConfidence] = useState(0.4);

  // What-If Simulation State
  const [simValues, setSimValues] = useState({
    temperature_celsius: 22.5,
    salinity_ppt: 35.0,
    ph_level: 8.1,
    dissolved_oxygen: 6.5,
  });
  const [simPrediction, setSimPrediction] = useState(null);

  // Hotspot Filters
  const [hotspotTypeFilter, setHotspotTypeFilter] = useState('All');
  const [hotspotRegionFilter, setHotspotRegionFilter] = useState('All');

  // Leaflet Map Ref
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersLayerRef = useRef(null);

  // Initial Data Fetching
  const fetchOverview = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/dataset`);
      if (!res.ok) throw new Error('Failed to connect to backend.');
      const data = await res.json();
      setDatasetOverview(data);

      const numCols = data.schema.filter(s => s.dtype.includes('float') || s.dtype.includes('int')).map(s => s.name);
      if (selectedClusterFeatures.length === 0 && numCols.length >= 2) {
        setSelectedClusterFeatures(numCols.slice(0, 4));
      }
      if (classFeatures.length === 0 && numCols.length >= 2) {
        setClassFeatures(numCols.slice(0, 4));
      }
      if (!targetCol) {
        const catCols = data.schema.filter(s => s.dtype === 'object').map(s => s.name);
        if (catCols.length > 0) setTargetCol(catCols[0]);
      }
      if (!predTarget && numCols.length > 0) {
        setPredTarget(numCols[0]);
        setPredFeatures(numCols.slice(1, 5));
      }
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchQuality = async () => {
    try {
      const res = await fetch(`${API_BASE}/quality`);
      if (res.ok) setQualityData(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  const fetchEDA = async () => {
    try {
      const res = await fetch(`${API_BASE}/eda`);
      if (res.ok) setEdaData(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  const fetchHotspots = async (region = hotspotRegionFilter, type = hotspotTypeFilter) => {
    try {
      let url = `${API_BASE}/hotspots`;
      const params = [];
      if (region && region !== 'All') params.push(`region=${encodeURIComponent(region)}`);
      if (type && type !== 'All') params.push(`pollution_type=${encodeURIComponent(type)}`);
      if (params.length > 0) url += `?${params.join('&')}`;

      const res = await fetch(url);
      if (res.ok) setHotspotData(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  const fetchReport = async () => {
    try {
      const res = await fetch(`${API_BASE}/report/markdown`);
      if (res.ok) setMarkdownReport(await res.text());
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchOverview();
    fetchEDA();
    fetchQuality();
    fetchHotspots();
  }, []);

  const handleRetryConnection = () => {
    setErrorMsg(null);
    fetchOverview();
    fetchEDA();
    fetchQuality();
    fetchHotspots();
  };

  const parseJsonResponse = async (res, defaultErrMsg = 'Operation failed.') => {
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      if (!res.ok) throw new Error(`Server error (${res.status}): ${text || res.statusText}`);
      throw new Error('Received unexpected non-JSON response from server.');
    }
    if (!res.ok) throw new Error(data.detail || data.message || defaultErrMsg);
    return data;
  };

  const showNotification = (msg, isError = false) => {
    if (isError) {
      setErrorMsg(msg);
      setTimeout(() => setErrorMsg(null), 5000);
    } else {
      setSuccessMsg(msg);
      setTimeout(() => setSuccessMsg(null), 5000);
    }
  };

  // Switch Telemetry Pipeline View
  const handleToggleViewMode = async (mode) => {
    try {
      await fetch(`${API_BASE}/dataset/view-mode?mode=${mode}`, { method: 'POST' });
      fetchOverview();
      fetchEDA();
      showNotification(`Switched view to ${mode.toUpperCase()} telemetry.`);
    } catch (e) {
      showNotification(e.message, true);
    }
  };

  // Load Benchmark
  const handleLoadBenchmark = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/dataset/load-benchmark`, { method: 'POST' });
      if (res.ok) {
        showNotification('Benchmark oceanographic dataset successfully loaded.');
        fetchOverview();
        fetchEDA();
        fetchQuality();
        fetchHotspots();
      }
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Run Preprocessing
  const handleRunPreprocessing = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          numeric_strategy: numStrategy,
          categorical_strategy: catStrategy,
          remove_duplicates: removeDups,
          outlier_strategy: outlierStrategy,
          outlier_factor: outlierFactor,
          scale_features: scaleFeatures,
          scale_method: scaleMethod,
          encode_categoricals: encodeCats,
          encode_method: encodeMethod,
        }),
      });
      const data = await parseJsonResponse(res, 'Preprocessing failed.');
      showNotification(`Cleaned successfully! Cleaned records: ${data.cleaned_rows.toLocaleString()}`);
      fetchOverview();
      fetchEDA();
      fetchQuality();
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Auto-Detect Optimal k
  const handleAutoDetectOptimalK = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/clustering/optimal-k`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: selectedClusterFeatures, k: 3 }),
      });
      const data = await parseJsonResponse(res, 'Optimal k detection failed.');
      setOptimalKInfo(data);
      setClusterK(data.recommended_k);
      showNotification(`Optimal k detected: k = ${data.recommended_k} (Silhouette: ${data.best_silhouette})`);
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Run Clustering
  const handleRunClustering = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/clustering`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: selectedClusterFeatures,
          k: clusterK,
        }),
      });
      const data = await parseJsonResponse(res, 'Clustering failed.');
      setClusteringResult(data);
      showNotification(`K-Means complete (k=${clusterK}, Silhouette: ${data.silhouette_score})`);
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Run Classification
  const handleRunClassification = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/classification`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_col: targetCol,
          feature_cols: classFeatures,
          test_size: testSplit,
        }),
      });
      const data = await parseJsonResponse(res, 'Classification failed.');
      setClassificationResult(data);
      showNotification(`Supervised models benchmarked in ${data.elapsed_ms}ms!`);
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Run Regression Prediction
  const handleRunPrediction = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/prediction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_col: predTarget,
          feature_cols: predFeatures,
          use_time_split: false,
        }),
      });
      const data = await parseJsonResponse(res, 'Prediction failed.');
      setPredictionResult(data);
      showNotification('Continuous regression benchmark trained!');
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Run Association Rules
  const handleRunAssociationRules = async () => {
    try {
      setLoading(true);
      const catCols = datasetOverview?.schema.filter(s => s.dtype === 'object').map(s => s.name) || [];
      const res = await fetch(`${API_BASE}/association-rules`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cat_cols: catCols.slice(0, 3),
          min_support: armSupport,
          min_confidence: armConfidence,
        }),
      });
      const data = await parseJsonResponse(res, 'Association mining failed.');
      setAssociationRulesResult(data);
      showNotification(`Discovered ${data.rules_count} association rules!`);
    } catch (e) {
      showNotification(e.message, true);
    } finally {
      setLoading(false);
    }
  };

  // Run Live Simulation
  const handleSimulate = async () => {
    try {
      const res = await fetch(`${API_BASE}/prediction/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feature_values: simValues,
        }),
      });
      const data = await parseJsonResponse(res, 'Simulation failed.');
      setSimPrediction(data);
    } catch (e) {
      showNotification(e.message, true);
    }
  };

  // Leaflet Map Initialization & Updates
  useEffect(() => {
    let timer = null;

    if (activeTab === 'hotspots' && mapContainerRef.current) {
      // Remove any existing instance to prevent stale DOM attachments across tab switches
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove();
        } catch (err) {
          console.warn('Map cleanup error:', err);
        }
        mapInstanceRef.current = null;
      }

      // Initialize map on current container
      const map = L.map(mapContainerRef.current, {
        center: [20, 0],
        zoom: 2,
        minZoom: 1,
        maxZoom: 18,
      });

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 18,
      }).addTo(map);

      const markersLayer = L.layerGroup().addTo(map);
      markersLayerRef.current = markersLayer;
      mapInstanceRef.current = map;

      // Update markers if points exist
      if (hotspotData?.points?.length > 0) {
        markersLayer.clearLayers();
        const bounds = [];
        hotspotData.points.forEach((pt) => {
          if (typeof pt.lat === 'number' && typeof pt.lng === 'number') {
            const color = pt.val > 70 ? '#EF4444' : pt.val > 40 ? '#F59E0B' : '#00B4D8';
            const circle = L.circleMarker([pt.lat, pt.lng], {
              radius: pt.val > 70 ? 7 : 5,
              fillColor: color,
              color: '#FFFFFF',
              weight: 1.5,
              opacity: 1,
              fillOpacity: 0.85,
            });
            circle.bindPopup(`
              <div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; line-height: 1.4; padding: 4px;">
                <b style="color: #0A3D62; font-size: 13px;">🌊 Marine Observation Point</b><br/>
                <b>Pollution Level:</b> <span style="color: ${color}; font-weight: 700;">${pt.val}</span><br/>
                <b>Coordinates:</b> ${pt.lat.toFixed(3)}°, ${pt.lng.toFixed(3)}°
              </div>
            `);
            markersLayer.addLayer(circle);
            bounds.push([pt.lat, pt.lng]);
          }
        });

        if (bounds.length > 0) {
          try {
            map.fitBounds(bounds, { padding: [30, 30], maxZoom: 6 });
          } catch (e) {
            map.setView([20, 0], 2);
          }
        }
      }

      // Invalidate size once DOM layout is settled
      timer = setTimeout(() => {
        if (mapInstanceRef.current) {
          mapInstanceRef.current.invalidateSize();
        }
      }, 150);
    }

    return () => {
      if (timer) clearTimeout(timer);
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.remove();
        } catch (err) {
          console.warn('Map cleanup error:', err);
        }
        mapInstanceRef.current = null;
      }
    };
  }, [activeTab, hotspotData]);

  // Filtered and Sorted Table Data
  const filteredTableData = useMemo(() => {
    if (!datasetOverview?.preview) return [];
    let list = [...datasetOverview.preview];
    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase();
      list = list.filter(row => Object.values(row).some(v => v !== null && String(v).toLowerCase().includes(q)));
    }
    if (tableSortCol) {
      list.sort((a, b) => {
        const valA = a[tableSortCol];
        const valB = b[tableSortCol];
        if (valA === valB) return 0;
        if (valA === null) return 1;
        if (valB === null) return -1;
        if (typeof valA === 'number' && typeof valB === 'number') {
          return tableSortAsc ? valA - valB : valB - valA;
        }
        return tableSortAsc ? String(valA).localeCompare(String(valB)) : String(valB).localeCompare(String(valA));
      });
    }
    return list;
  }, [datasetOverview, tableSearch, tableSortCol, tableSortAsc]);

  const totalPages = Math.ceil(filteredTableData.length / rowsPerPage) || 1;
  const paginatedData = filteredTableData.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

  const navItems = [
    { id: 'dashboard', label: 'Home Dashboard', icon: Waves },
    { id: 'dataset', label: 'Dataset Management', icon: Database },
    { id: 'quality', label: 'Data Quality Audit', icon: ShieldCheck },
    { id: 'preprocessing', label: 'Data Preprocessing', icon: Cpu },
    { id: 'eda', label: 'Exploratory EDA', icon: BarChart3 },
    { id: 'clustering', label: 'K-Means Clustering', icon: Target },
    { id: 'classification', label: 'Classification', icon: Tag },
    { id: 'prediction', label: 'Pollution Forecasting', icon: TrendingUp },
    { id: 'arm', label: 'Association Rules', icon: GitBranch },
    { id: 'hotspots', label: 'Hotspot Analysis', icon: MapPin },
    { id: 'reports', label: 'Executive Reports', icon: FileText },
    { id: 'about', label: 'Architecture', icon: Info },
  ];

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div style={{ textAlign: 'center', paddingBottom: '20px', borderBottom: '1px solid rgba(0, 180, 216, 0.25)', marginBottom: '20px' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '4px' }}>🌊</div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#FFFFFF' }}>MarineMine Pro</h1>
          <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#38BDF8', fontWeight: 600 }}>Oceanic Data Mining Suite</p>
        </div>

        {/* Telemetry Status Widget */}
        {datasetOverview && (
          <div style={{ background: 'rgba(6, 26, 47, 0.6)', border: '1px solid rgba(0, 180, 216, 0.3)', borderRadius: '10px', padding: '12px', marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <span className="badge-pill badge-green">● Online</span>
              <span style={{ fontSize: '0.72rem', color: '#90E0EF' }}>Enhanced ML &amp; GIS</span>
            </div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#E0F2FE', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {datasetOverview.dataset_name}
            </div>
            <div style={{ fontSize: '0.72rem', color: '#94A3B8', marginTop: '4px' }}>
              <b>{datasetOverview.total_records.toLocaleString()}</b> records &nbsp;|&nbsp; <b>{datasetOverview.total_columns}</b> cols
            </div>

            {/* View Mode Toggle */}
            <div style={{ display: 'flex', gap: '6px', marginTop: '10px' }}>
              <button
                onClick={() => handleToggleViewMode('cleaned')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  borderRadius: '6px',
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: datasetOverview.view_mode === 'cleaned' ? '#00B4D8' : 'rgba(255,255,255,0.1)',
                  color: datasetOverview.view_mode === 'cleaned' ? '#061A2F' : '#E0F2FE'
                }}
              >
                Cleaned
              </button>
              <button
                onClick={() => handleToggleViewMode('raw')}
                style={{
                  flex: 1,
                  padding: '5px 8px',
                  borderRadius: '6px',
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: datasetOverview.view_mode === 'raw' ? '#00B4D8' : 'rgba(255,255,255,0.1)',
                  color: datasetOverview.view_mode === 'raw' ? '#061A2F' : '#E0F2FE'
                }}
              >
                Raw Data
              </button>
            </div>
          </div>
        )}

        {/* Navigation List */}
        <nav style={{ flex: 1 }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <div
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id);
                  if (item.id === 'reports') fetchReport();
                }}
                className={`nav-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </div>
            );
          })}
        </nav>

        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '14px', fontSize: '0.72rem', color: '#64748B' }}>
          <div>MarineMine Pro Analytics</div>
          <div style={{ color: '#00B4D8' }}>FastAPI + React 18 Suite</div>
        </div>
      </aside>

      {/* Main Workspace Area */}
      <main className="main-content">
        {/* Notification Toasts */}
        {errorMsg && (
          <div style={{ backgroundColor: '#FEF2F2', border: '1px solid #F87171', color: '#991B1B', padding: '12px 16px', borderRadius: '8px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={18} />
              <span>
                {errorMsg.toLowerCase().includes('failed to fetch')
                  ? 'Connecting to MarineMine Pro backend (http://localhost:8000)... If the backend was recently restarted, please click Reconnect.'
                  : errorMsg}
              </span>
            </div>
            <button
              onClick={handleRetryConnection}
              className="btn btn-primary"
              style={{ padding: '6px 14px', fontSize: '0.78rem', whiteSpace: 'nowrap' }}
            >
              <RefreshCw size={12} /> Reconnect
            </button>
          </div>
        )}
        {successMsg && (
          <div style={{ backgroundColor: '#DCFCE7', border: '1px solid #86EFAC', color: '#166534', padding: '12px 16px', borderRadius: '8px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <CheckCircle2 size={18} />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Top Telemetry Header Ribbon */}
        {datasetOverview && (
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: 'linear-gradient(90deg, #061A2F 0%, #0A3D62 50%, #082946 100%)',
            border: '1px solid rgba(0, 180, 216, 0.25)',
            borderRadius: '12px',
            padding: '10px 20px',
            marginBottom: '20px',
            color: '#E0F2FE',
            fontSize: '0.85rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span>🌊 <b>Dataset:</b> {datasetOverview.dataset_name}</span>
              <span className="badge-pill badge-cyan">{datasetOverview.total_records.toLocaleString()} Observations</span>
              <span className="badge-pill badge-cyan">{datasetOverview.total_columns} Features</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="badge-pill badge-green">{datasetOverview.view_mode.toUpperCase()} VIEW</span>
              {qualityData && (
                <span className="badge-pill badge-cyan">Quality: {qualityData.quality_score}/100</span>
              )}
              <button
                onClick={handleLoadBenchmark}
                className="btn btn-outline-cyan"
                style={{ padding: '4px 10px', fontSize: '0.75rem' }}
              >
                <RefreshCw size={12} /> Reload Benchmark
              </button>
            </div>
          </div>
        )}

        {/* TAB 1: HOME DASHBOARD */}
        {activeTab === 'dashboard' && datasetOverview && (
          <div>
            {/* Hero Banner */}
            <div style={{
              background: 'linear-gradient(135deg, #061A2F 0%, #0A3D62 45%, #0077B6 85%, #00B4D8 100%)',
              borderRadius: '16px',
              padding: '36px 40px',
              color: 'white',
              marginBottom: '24px',
              boxShadow: '0 12px 36px rgba(6, 26, 47, 0.2)',
              position: 'relative'
            }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                background: 'rgba(255, 255, 255, 0.15)',
                border: '1px solid rgba(255, 255, 255, 0.25)',
                padding: '4px 12px',
                borderRadius: '30px',
                fontSize: '0.76rem',
                fontWeight: 700,
                color: '#90E0EF',
                marginBottom: '12px'
              }}>
                <Sparkles size={14} /> AUTONOMOUS MARINE GEOINFORMATICS &amp; ML
              </div>
              <h2 style={{ fontSize: '2.4rem', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: '10px' }}>
                Mining Patterns of Marine Pollution 🌊
              </h2>
              <p style={{ fontSize: '1.05rem', color: '#E0F2FE', maxWidth: '850px', lineHeight: 1.6 }}>
                An intelligent data mining ecosystem engineered to discover spatial contamination hotspots,
                temporal anomalies, multi-model classifications, and environmental correlations across global ocean telemetry.
              </p>
            </div>

            {/* Quick Action Buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '24px' }}>
              {[
                { label: 'Quality Audit', tab: 'quality', icon: ShieldCheck },
                { label: 'Clean Data', tab: 'preprocessing', icon: Cpu },
                { label: 'K-Means Clusters', tab: 'clustering', icon: Target },
                { label: 'Classifiers', tab: 'classification', icon: Tag },
                { label: 'Forecasting', tab: 'prediction', icon: TrendingUp },
                { label: 'Hotspot Maps', tab: 'hotspots', icon: MapPin },
              ].map(action => {
                const Icon = action.icon;
                return (
                  <button
                    key={action.tab}
                    onClick={() => {
                      setActiveTab(action.tab);
                      if (action.tab === 'reports') fetchReport();
                    }}
                    className="card"
                    style={{
                      padding: '14px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                      cursor: 'pointer',
                      border: '1px solid #E2E8F0',
                      marginBottom: 0
                    }}
                  >
                    <div style={{ background: '#E0F2FE', padding: '8px', borderRadius: '8px', color: '#0077B6' }}>
                      <Icon size={18} />
                    </div>
                    <span style={{ fontWeight: 700, fontSize: '0.88rem', color: '#0A3D62' }}>{action.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Metric Cards KPI Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <div className="metric-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#90E0EF', fontWeight: 600 }}>Observations</span>
                  <span>📡</span>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800 }}>{datasetOverview.total_records.toLocaleString()}</div>
                <div style={{ fontSize: '0.75rem', color: '#CAF0F8' }}>Telemetry Records</div>
              </div>

              <div className="metric-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#90E0EF', fontWeight: 600 }}>Pollutants</span>
                  <span>🧪</span>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800 }}>{datasetOverview.total_categories}</div>
                <div style={{ fontSize: '0.75rem', color: '#CAF0F8' }}>Top: {datasetOverview.top_category}</div>
              </div>

              <div className="metric-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#90E0EF', fontWeight: 600 }}>Ocean Basins</span>
                  <span>🌐</span>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800 }}>{datasetOverview.total_regions}</div>
                <div style={{ fontSize: '0.75rem', color: '#CAF0F8' }}>Monitored Areas</div>
              </div>

              <div className="metric-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#90E0EF', fontWeight: 600 }}>Mean Level</span>
                  <span>📊</span>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800 }}>{datasetOverview.avg_pollution}</div>
                <div style={{ fontSize: '0.75rem', color: '#CAF0F8' }}>Concentration Index</div>
              </div>

              <div className="metric-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#90E0EF', fontWeight: 600 }}>Missing Data</span>
                  <span>🛡️</span>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 800 }}>{datasetOverview.missing_pct}%</div>
                <div style={{ fontSize: '0.75rem', color: '#CAF0F8' }}>Across all cells</div>
              </div>
            </div>

            {/* AI Domain Intelligence Card */}
            <div style={{
              background: 'linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%)',
              border: '1px solid #BAE6FD',
              borderLeft: '5px solid #0284C7',
              borderRadius: '10px',
              padding: '16px 20px',
              marginBottom: '24px',
              color: '#0369A1'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, marginBottom: '4px' }}>
                <Sparkles size={18} /> Automated Environmental Intelligence
              </div>
              <p style={{ fontSize: '0.88rem', lineHeight: 1.5, margin: 0 }}>
                Dominant recorded oceanic pollutant in this study cohort is <b>{datasetOverview.top_category}</b>.
                Overall data health index is rated <b>{qualityData ? `${qualityData.quality_score}/100 (Grade ${qualityData.quality_grade})` : 'Audit Ready'}</b>.
                Explore spatial hotspots in <b>Hotspot Analysis</b> or run predictive models in <b>Pollution Forecasting</b>.
              </p>
            </div>

            {/* Recharts Visualizations Deck */}
            {edaData && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px', marginBottom: '24px' }}>
                {/* Category Breakdown BarChart */}
                <div className="card">
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '14px' }}>
                    📊 Pollutant Distribution by Class
                  </h3>
                  <div style={{ width: '100%', height: 260 }}>
                    <ResponsiveContainer>
                      <BarChart data={edaData.categories} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                        <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#0077B6" radius={[6, 6, 0, 0]}>
                          {edaData.categories.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={PALETTE[index % PALETTE.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Monthly Time Series AreaChart */}
                <div className="card">
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '14px' }}>
                    📈 Longitudinal Pollution Progression
                  </h3>
                  <div style={{ width: '100%', height: 260 }}>
                    <ResponsiveContainer>
                      <AreaChart data={edaData.time_series} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                        <defs>
                          <linearGradient id="oceanGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#00B4D8" stopOpacity={0.6} />
                            <stop offset="95%" stopColor="#0077B6" stopOpacity={0.05} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                        <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Area type="monotone" dataKey="value" stroke="#0077B6" strokeWidth={2.5} fillOpacity={1} fill="url(#oceanGrad)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: DATASET MANAGEMENT (With Search & Sorting) */}
        {activeTab === 'dataset' && datasetOverview && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>📂 Dataset Management &amp; Ingestion</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Upload external marine pollution CSV datasets or calibrate the benchmark oceanographic series.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '24px' }}>
              <div className="card">
                <h4 style={{ color: '#0A3D62', marginBottom: '8px' }}>📤 Ingest Custom Marine Telemetry CSV</h4>
                <p style={{ fontSize: '0.85rem', color: '#64748B', marginBottom: '14px' }}>
                  Upload arbitrary oceanographic sensor files in standard comma-separated format.
                </p>
                <input
                  type="file"
                  accept=".csv"
                  onChange={async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    const formData = new FormData();
                    formData.append('file', file);
                    try {
                      setLoading(true);
                      const res = await fetch(`${API_BASE}/dataset/upload`, { method: 'POST', body: formData });
                      const d = await parseJsonResponse(res, 'Upload failed.');
                      showNotification(`Uploaded ${file.name} successfully!`);
                      fetchOverview();
                      fetchEDA();
                    } catch (err) {
                      showNotification(err.message, true);
                    } finally {
                      setLoading(false);
                    }
                  }}
                  style={{ fontSize: '0.88rem' }}
                />
              </div>

              <div className="card">
                <h4 style={{ color: '#0A3D62', marginBottom: '8px' }}>🌊 Benchmark Cohort</h4>
                <p style={{ fontSize: '0.85rem', color: '#64748B', marginBottom: '14px' }}>
                  Calibrate with the verified 1,000-record benchmark dataset across 10 global basins.
                </p>
                <button onClick={handleLoadBenchmark} className="btn btn-primary" style={{ width: '100%' }}>
                  <RefreshCw size={16} /> Load Benchmark
                </button>
              </div>
            </div>

            {/* Interactive Data Table with Search and Sorting */}
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62' }}>
                  📄 Telemetry Data Explorer ({filteredTableData.length} records found)
                </h3>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <div style={{ position: 'relative' }}>
                    <Search size={16} style={{ position: 'absolute', left: 10, top: 10, color: '#94A3B8' }} />
                    <input
                      type="text"
                      placeholder="Search telemetry..."
                      value={tableSearch}
                      onChange={e => { setTableSearch(e.target.value); setCurrentPage(1); }}
                      style={{ padding: '8px 12px 8px 32px', borderRadius: '6px', border: '1px solid #CBD5E1', fontSize: '0.85rem' }}
                    />
                  </div>
                  <a href={`${API_BASE}/report/csv`} target="_blank" rel="noreferrer" className="btn btn-primary" style={{ textDecoration: 'none' }}>
                    <Download size={16} /> Export CSV
                  </a>
                </div>
              </div>

              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      {datasetOverview.preview.length > 0 && Object.keys(datasetOverview.preview[0]).map((col) => (
                        <th
                          key={col}
                          onClick={() => {
                            if (tableSortCol === col) {
                              setTableSortAsc(!tableSortAsc);
                            } else {
                              setTableSortCol(col);
                              setTableSortAsc(true);
                            }
                          }}
                          style={{ cursor: 'pointer' }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <span>{col}</span>
                            <ArrowUpDown size={12} style={{ color: tableSortCol === col ? '#0077B6' : '#CBD5E1' }} />
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedData.map((row, idx) => (
                      <tr key={idx}>
                        {Object.keys(datasetOverview.preview[0]).map((col) => (
                          <td key={col}>{row[col] !== null ? String(row[col]) : '-'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination Controls */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '14px', fontSize: '0.85rem' }}>
                <span style={{ color: '#64748B' }}>
                  Page {currentPage} of {totalPages}
                </span>
                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="btn btn-secondary"
                    style={{ padding: '4px 10px' }}
                  >
                    <ChevronLeft size={16} /> Prev
                  </button>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="btn btn-secondary"
                    style={{ padding: '4px 10px' }}
                  >
                    Next <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: DATA QUALITY AUDIT (With Radial Meter) */}
        {activeTab === 'quality' && qualityData && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>🔍 Data Quality &amp; Integrity Audit</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Comprehensive assessment of completeness, uniqueness, IQR outliers, and physical bounds.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', marginBottom: '24px' }}>
              {/* Radial Meter Card */}
              <div className="card" style={{ textAlign: 'center', padding: '24px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ position: 'relative', width: 140, height: 140 }}>
                  <svg width="140" height="140" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="42" fill="none" stroke="#F1F5F9" strokeWidth="10" />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      fill="none"
                      stroke={qualityData.quality_score >= 80 ? '#10B981' : '#F59E0B'}
                      strokeWidth="10"
                      strokeDasharray="264"
                      strokeDashoffset={264 - (264 * qualityData.quality_score) / 100}
                      strokeLinecap="round"
                      style={{ transition: 'stroke-dashoffset 0.8s ease' }}
                    />
                  </svg>
                  <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#0A3D62' }}>{qualityData.quality_score}</div>
                    <div style={{ fontSize: '0.68rem', color: '#64748B', textTransform: 'uppercase' }}>Score / 100</div>
                  </div>
                </div>
                <div className="badge-pill badge-green" style={{ marginTop: '12px', fontSize: '0.85rem' }}>
                  Grade {qualityData.quality_grade} Certified
                </div>
              </div>

              {/* Progress Dimensions */}
              <div className="card">
                <h4 style={{ color: '#0A3D62', marginBottom: '14px' }}>Four-Factor Data Integrity Dimensions</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                  {[
                    { label: 'Completeness', val: qualityData.breakdown.completeness, max: 40, color: '#0077B6' },
                    { label: 'Uniqueness (Zero Duplicates)', val: qualityData.breakdown.uniqueness, max: 20, color: '#00B4D8' },
                    { label: 'Physical Validity (Domain Bounds)', val: qualityData.breakdown.validity, max: 20, color: '#10B981' },
                    { label: 'Type Consistency', val: qualityData.breakdown.consistency, max: 20, color: '#6366F1' },
                  ].map((dim, i) => (
                    <div key={i}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                        <span style={{ fontWeight: 600 }}>{dim.label}</span>
                        <span>{dim.val} / {dim.max}</span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#F1F5F9', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ width: `${(dim.val / dim.max) * 100}%`, height: '100%', background: dim.color, borderRadius: '4px' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: DATA PREPROCESSING PIPELINE (With Outlier Treatment) */}
        {activeTab === 'preprocessing' && datasetOverview && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>⚙️ Data Preprocessing &amp; Cleaning Pipeline</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Non-destructive pipeline execution. Raw observations are preserved intact.</p>
            </div>

            <div className="card">
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '16px' }}>Configure Transformation Steps</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Numeric Imputation</label>
                  <select value={numStrategy} onChange={e => setNumStrategy(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #CBD5E1' }}>
                    <option value="mean">Mean Imputation</option>
                    <option value="median">Median Imputation</option>
                    <option value="drop">Drop Rows</option>
                    <option value="none">None</option>
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Categorical Imputation</label>
                  <select value={catStrategy} onChange={e => setCatStrategy(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #CBD5E1' }}>
                    <option value="mode">Mode Imputation</option>
                    <option value="drop">Drop Rows</option>
                    <option value="none">None</option>
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Outlier Treatment (IQR)</label>
                  <select value={outlierStrategy} onChange={e => setOutlierStrategy(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #CBD5E1' }}>
                    <option value="winsorize">Winsorize / Clip to Bounds</option>
                    <option value="drop">Drop Outlier Rows</option>
                    <option value="none">Retain Outliers</option>
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Feature Scaling</label>
                  <select value={scaleMethod} onChange={e => setScaleMethod(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #CBD5E1' }}>
                    <option value="standard">StandardScaler</option>
                    <option value="minmax">MinMaxScaler</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.88rem', cursor: 'pointer' }}>
                  <input type="checkbox" checked={removeDups} onChange={e => setRemoveDups(e.target.checked)} />
                  Remove Duplicates
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.88rem', cursor: 'pointer' }}>
                  <input type="checkbox" checked={scaleFeatures} onChange={e => setScaleFeatures(e.target.checked)} />
                  Apply Numeric Scaling
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.88rem', cursor: 'pointer' }}>
                  <input type="checkbox" checked={encodeCats} onChange={e => setEncodeCats(e.target.checked)} />
                  Encode Categoricals
                </label>
              </div>

              <button onClick={handleRunPreprocessing} disabled={loading} className="btn btn-primary" style={{ width: '100%' }}>
                <Play size={16} /> {loading ? 'Executing Pipeline...' : 'Execute Preprocessing Pipeline'}
              </button>
            </div>
          </div>
        )}

        {/* TAB 5: K-MEANS CLUSTERING (With Auto-Detect Optimal k) */}
        {activeTab === 'clustering' && datasetOverview && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>🎯 Unsupervised K-Means Pattern Discovery</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Partition multivariate marine observations into coherent pollution clusters.</p>
            </div>

            <div className="card">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '16px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Number of Clusters (k): {clusterK}</label>
                    <button
                      onClick={handleAutoDetectOptimalK}
                      disabled={loading}
                      className="btn btn-outline-cyan"
                      style={{ padding: '3px 8px', fontSize: '0.74rem' }}
                    >
                      <Zap size={12} /> Auto-Detect Optimal k
                    </button>
                  </div>
                  <input type="range" min="2" max="8" value={clusterK} onChange={e => setClusterK(Number(e.target.value))} style={{ width: '100%' }} />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Active Clustering Features</label>
                  <div style={{ fontSize: '0.82rem', color: '#64748B' }}>{selectedClusterFeatures.join(', ')}</div>
                </div>
              </div>

              {optimalKInfo && (
                <div style={{ background: '#F0FDF4', border: '1px solid #86EFAC', padding: '10px 14px', borderRadius: '8px', marginBottom: '14px', fontSize: '0.85rem', color: '#166534' }}>
                  <b>💡 Recommendation:</b> {optimalKInfo.reason}
                </div>
              )}

              <button onClick={handleRunClustering} disabled={loading} className="btn btn-primary" style={{ width: '100%' }}>
                <Play size={16} /> {loading ? 'Clustering...' : 'Execute K-Means Clustering'}
              </button>
            </div>

            {clusteringResult && (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px', marginBottom: '20px' }}>
                  <div className="card" style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '0.85rem', color: '#64748B', fontWeight: 700 }}>Silhouette Coefficient</div>
                    <div style={{ fontSize: '2.8rem', fontWeight: 800, color: '#0077B6', margin: '8px 0' }}>
                      {clusteringResult.silhouette_score}
                    </div>
                    <div className="badge-pill badge-cyan">k = {clusteringResult.n_clusters} Clusters Formed</div>
                  </div>

                  <div className="card">
                    <h4 style={{ color: '#0A3D62', marginBottom: '10px' }}>💡 Plain-Language Cluster Insights</h4>
                    {Object.entries(clusteringResult.descriptions).map(([cid, desc]) => (
                      <div key={cid} style={{ background: '#F8FAFC', padding: '10px 14px', borderRadius: '8px', marginBottom: '8px', fontSize: '0.85rem' }}>
                        <b>Cluster {cid}:</b> {desc}
                      </div>
                    ))}
                  </div>
                </div>

                {/* 2D PCA ScatterChart */}
                <div className="card">
                  <h4 style={{ color: '#0A3D62', marginBottom: '12px' }}>🌌 2D Principal Component Projection (PCA)</h4>
                  <div style={{ width: '100%', height: 320 }}>
                    <ResponsiveContainer>
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis type="number" dataKey="x" name="PC1" unit="" />
                        <YAxis type="number" dataKey="y" name="PC2" unit="" />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Clusters" data={clusteringResult.pca_points}>
                          {clusteringResult.pca_points.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={PALETTE[entry.cluster % PALETTE.length]} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 6: CLASSIFICATION (With Feature Importances) */}
        {activeTab === 'classification' && datasetOverview && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>🏷️ Multi-Model Supervised Classification</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Train Random Forest, Gradient Boosting, Decision Trees, and Logistic Regression.</p>
            </div>

            <div className="card">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Target Column</label>
                  <select value={targetCol} onChange={e => setTargetCol(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #CBD5E1' }}>
                    {datasetOverview.schema.filter(s => s.dtype === 'object').map(s => (
                      <option key={s.name} value={s.name}>{s.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Test Split Ratio: {testSplit * 100}%</label>
                  <input type="range" min="0.1" max="0.4" step="0.05" value={testSplit} onChange={e => setTestSplit(Number(e.target.value))} style={{ width: '100%' }} />
                </div>
              </div>
              <button onClick={handleRunClassification} disabled={loading} className="btn btn-primary" style={{ width: '100%' }}>
                <Play size={16} /> {loading ? 'Training Classifiers...' : 'Train & Compare Classifiers'}
              </button>
            </div>

            {classificationResult && (
              <div>
                <div className="card" style={{ marginBottom: '20px' }}>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '14px' }}>
                    🏆 Model Performance Leaderboard (Completed in {classificationResult.elapsed_ms}ms)
                  </h3>
                  <div className="data-table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Algorithm</th>
                          <th>Accuracy (%)</th>
                          <th>Macro F1-Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {classificationResult.models.map((m, idx) => (
                          <tr key={idx}>
                            <td><b>{m.model_name}</b></td>
                            <td><span className="badge-pill badge-green">{m.accuracy}%</span></td>
                            <td>{m.f1_score}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Feature Importance BarChart */}
                {classificationResult.feature_importances?.length > 0 && (
                  <div className="card">
                    <h4 style={{ color: '#0A3D62', marginBottom: '12px' }}>📊 Feature Importance (Ensemble Model)</h4>
                    <div style={{ width: '100%', height: 250 }}>
                      <ResponsiveContainer>
                        <BarChart layout="vertical" data={classificationResult.feature_importances} margin={{ top: 10, right: 20, left: 40, bottom: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                          <XAxis type="number" />
                          <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} />
                          <Tooltip />
                          <Bar dataKey="importance" fill="#00B4D8" radius={[0, 6, 6, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 7: POLLUTION FORECASTING & WHAT-IF SIMULATOR */}
        {activeTab === 'prediction' && datasetOverview && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>📈 Continuous Pollution Estimation &amp; What-If Simulation</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Forecast numeric concentration levels using Random Forests and test hypothetical scenarios.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
              {/* Train Regressors Card */}
              <div className="card">
                <h4 style={{ color: '#0A3D62', marginBottom: '10px' }}>Train Continuous Regressors</h4>
                <div style={{ marginBottom: '14px' }}>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Target Feature</label>
                  <select value={predTarget} onChange={e => setPredTarget(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '6px', border: '1px solid #CBD5E1' }}>
                    {datasetOverview.schema.filter(s => s.dtype.includes('float') || s.dtype.includes('int')).map(s => (
                      <option key={s.name} value={s.name}>{s.name}</option>
                    ))}
                  </select>
                </div>
                <button onClick={handleRunPrediction} disabled={loading} className="btn btn-primary" style={{ width: '100%' }}>
                  <Play size={16} /> {loading ? 'Training...' : 'Train Regression Models'}
                </button>
              </div>

              {/* What-If Simulation Tool */}
              <div className="card" style={{ border: '1px solid #00B4D8' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#0A3D62', fontWeight: 700, marginBottom: '10px' }}>
                  <Sliders size={18} /> Real-Time "What-If" Pollution Simulator
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px', fontSize: '0.82rem' }}>
                  <div>
                    <label>Temperature (°C): {simValues.temperature_celsius}</label>
                    <input type="range" min="0" max="40" step="0.5" value={simValues.temperature_celsius} onChange={e => setSimValues({ ...simValues, temperature_celsius: Number(e.target.value) })} style={{ width: '100%' }} />
                  </div>
                  <div>
                    <label>Salinity (ppt): {simValues.salinity_ppt}</label>
                    <input type="range" min="10" max="45" step="0.5" value={simValues.salinity_ppt} onChange={e => setSimValues({ ...simValues, salinity_ppt: Number(e.target.value) })} style={{ width: '100%' }} />
                  </div>
                  <div>
                    <label>pH Level: {simValues.ph_level}</label>
                    <input type="range" min="6.5" max="9.0" step="0.1" value={simValues.ph_level} onChange={e => setSimValues({ ...simValues, ph_level: Number(e.target.value) })} style={{ width: '100%' }} />
                  </div>
                  <div>
                    <label>Dissolved Oxygen: {simValues.dissolved_oxygen}</label>
                    <input type="range" min="2" max="12" step="0.5" value={simValues.dissolved_oxygen} onChange={e => setSimValues({ ...simValues, dissolved_oxygen: Number(e.target.value) })} style={{ width: '100%' }} />
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <button onClick={handleSimulate} className="btn btn-outline-cyan">
                    <Sparkles size={14} /> Calculate Impact
                  </button>
                  {simPrediction && (
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '0.78rem', color: '#64748B' }}>Predicted Level: </span>
                      <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0077B6' }}>{simPrediction.predicted_value}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Scatter & Feature Importances */}
            {predictionResult && (
              <div>
                <div className="card" style={{ marginBottom: '20px' }}>
                  <h4 style={{ color: '#0A3D62', marginBottom: '12px' }}>🎯 Actual vs. Model-Predicted Concentration</h4>
                  <div style={{ width: '100%', height: 320 }}>
                    <ResponsiveContainer>
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                        <XAxis type="number" dataKey="actual" name="Actual" />
                        <YAxis type="number" dataKey="predicted" name="Predicted" />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Actual vs Predicted" data={predictionResult.scatter} fill="#0077B6" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {predictionResult.feature_importances?.length > 0 && (
                  <div className="card">
                    <h4 style={{ color: '#0A3D62', marginBottom: '12px' }}>📊 Environmental Predictor Drivers</h4>
                    <div style={{ width: '100%', height: 240 }}>
                      <ResponsiveContainer>
                        <BarChart layout="vertical" data={predictionResult.feature_importances} margin={{ top: 10, right: 20, left: 40, bottom: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                          <XAxis type="number" />
                          <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} />
                          <Tooltip />
                          <Bar dataKey="importance" fill="#0077B6" radius={[0, 6, 6, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 8: ASSOCIATION RULE MINING */}
        {activeTab === 'arm' && datasetOverview && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>🔗 Association Rule Mining (Apriori)</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Discover multi-factor co-occurrence rules across marine pollutants and regional parameters.</p>
            </div>

            <div className="card">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '16px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Min Support Threshold: {armSupport}</label>
                  <input type="range" min="0.01" max="0.3" step="0.01" value={armSupport} onChange={e => setArmSupport(Number(e.target.value))} style={{ width: '100%' }} />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>Min Confidence Threshold: {armConfidence}</label>
                  <input type="range" min="0.1" max="0.9" step="0.05" value={armConfidence} onChange={e => setArmConfidence(Number(e.target.value))} style={{ width: '100%' }} />
                </div>
              </div>
              <button onClick={handleRunAssociationRules} disabled={loading} className="btn btn-primary" style={{ width: '100%' }}>
                <Play size={16} /> {loading ? 'Mining Rules...' : 'Mine Frequent Itemsets & Rules'}
              </button>
            </div>

            {associationRulesResult && (
              <div>
                <div style={{ display: 'flex', gap: '14px', marginBottom: '20px' }}>
                  <div className="card" style={{ flex: 1, textAlign: 'center' }}>
                    <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 700 }}>Frequent Itemsets</div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: '#0077B6' }}>{associationRulesResult.itemsets_count}</div>
                  </div>
                  <div className="card" style={{ flex: 1, textAlign: 'center' }}>
                    <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 700 }}>Discovered Rules</div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: '#10B981' }}>{associationRulesResult.rules_count}</div>
                  </div>
                </div>

                {associationRulesResult.rules.length > 0 && (
                  <div className="card">
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '14px' }}>
                      📜 Top Discovered Association Rules
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {associationRulesResult.rules.map((rule, idx) => (
                        <div key={idx} style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', padding: '12px 16px', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.88rem' }}>
                            <span style={{ fontWeight: 700, color: '#0A3D62' }}>{rule.antecedents.join(', ')}</span>
                            <ArrowRight size={16} style={{ color: '#00B4D8' }} />
                            <span style={{ fontWeight: 700, color: '#0077B6' }}>{rule.consequents.join(', ')}</span>
                          </div>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            <span className="badge-pill badge-cyan">Supp: {rule.support}</span>
                            <span className="badge-pill badge-green">Conf: {rule.confidence}</span>
                            <span className="badge-pill badge-amber">Lift: {rule.lift}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 9: GEOSPATIAL HOTSPOTS (With Filtering and Leaflet) */}
        {activeTab === 'hotspots' && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>🗺️ Interactive Geospatial Hotspot Map</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Real-time spatial visualization powered by Leaflet &amp; OpenStreetMap telemetry.</p>
            </div>

            {!hotspotData ? (
              <div className="card" style={{ textAlign: 'center', padding: '60px 20px' }}>
                <RefreshCw size={36} className="spin" style={{ color: '#00B4D8', margin: '0 auto 16px' }} />
                <h3 style={{ fontSize: '1.2rem', color: '#0A3D62' }}>Loading Geospatial Telemetry...</h3>
                <p style={{ color: '#64748B', fontSize: '0.88rem', marginBottom: '20px' }}>Aggregating spatial GPS coordinates and pollution severity scores.</p>
                <button onClick={() => fetchHotspots()} className="btn btn-primary" style={{ margin: '0 auto' }}>
                  <RefreshCw size={14} /> Retry Map Ingestion
                </button>
              </div>
            ) : (
              <>
                {/* Filter Bar */}
                <div className="card" style={{ display: 'flex', gap: '20px', alignItems: 'center', padding: '14px 20px', marginBottom: '16px', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Filter size={16} style={{ color: '#0077B6' }} />
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0A3D62' }}>Filters:</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <label style={{ fontSize: '0.82rem', fontWeight: 600 }}>Region:</label>
                    <select
                      value={hotspotRegionFilter}
                      onChange={(e) => {
                        setHotspotRegionFilter(e.target.value);
                        fetchHotspots(e.target.value, hotspotTypeFilter);
                      }}
                      style={{ padding: '6px 10px', borderRadius: '6px', border: '1px solid #CBD5E1', fontSize: '0.82rem' }}
                    >
                      <option value="All">All Regions</option>
                      {edaData?.regions.map(r => (
                        <option key={r.region} value={r.region}>{r.region}</option>
                      ))}
                    </select>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <label style={{ fontSize: '0.82rem', fontWeight: 600 }}>Pollutant Type:</label>
                    <select
                      value={hotspotTypeFilter}
                      onChange={(e) => {
                        setHotspotTypeFilter(e.target.value);
                        fetchHotspots(hotspotRegionFilter, e.target.value);
                      }}
                      style={{ padding: '6px 10px', borderRadius: '6px', border: '1px solid #CBD5E1', fontSize: '0.82rem' }}
                    >
                      <option value="All">All Pollutants</option>
                      {edaData?.categories.map(c => (
                        <option key={c.category} value={c.category}>{c.category}</option>
                      ))}
                    </select>
                  </div>

                  <div style={{ marginLeft: 'auto', display: 'flex', gap: '12px', alignItems: 'center', fontSize: '0.78rem' }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#EF4444', display: 'inline-block' }}></span> High Risk (&gt;70)
                    </span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#F59E0B', display: 'inline-block' }}></span> Moderate (40-70)
                    </span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#00B4D8', display: 'inline-block' }}></span> Low Risk (&lt;40)
                    </span>
                  </div>
                </div>

                {/* Interactive Leaflet Map Container */}
                <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: '24px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
                  <div style={{ padding: '14px 20px', background: '#F8FAFC', borderBottom: '1px solid #E2E8F0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 700, color: '#0A3D62', fontSize: '0.95rem' }}>🌍 Global Marine Risk Map</span>
                      <span className="badge-pill badge-cyan">{hotspotData.points.length} Live Coordinates Rendered</span>
                    </div>
                    <button
                      onClick={() => {
                        if (mapInstanceRef.current) {
                          mapInstanceRef.current.invalidateSize();
                          if (hotspotData?.points?.length > 0) {
                            const bounds = hotspotData.points.map(p => [p.lat, p.lng]);
                            mapInstanceRef.current.fitBounds(bounds, { padding: [30, 30], maxZoom: 6 });
                          }
                        }
                      }}
                      className="btn btn-outline-cyan"
                      style={{ padding: '4px 10px', fontSize: '0.72rem' }}
                    >
                      <RefreshCw size={11} /> Reset Map View
                    </button>
                  </div>
                  <div
                    ref={mapContainerRef}
                    className="leaflet-container"
                    style={{
                      height: '520px',
                      width: '100%',
                      minHeight: '520px',
                      position: 'relative',
                      zIndex: 1,
                      backgroundColor: '#E2E8F0'
                    }}
                  />
                </div>
              </>
            )}

            {/* Top 10 Ranked Hotspots Leaderboard */}
            <div className="card">
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '14px' }}>
                🏆 Top 10 High-Intensity Marine Hotspots
              </h3>
              <div className="data-table-container">
                <table className="data-table">
                  <thead>
                    <tr>
                      {hotspotData.hotspots.length > 0 && Object.keys(hotspotData.hotspots[0]).map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {hotspotData.hotspots.map((row, idx) => (
                      <tr key={idx}>
                        {Object.keys(hotspotData.hotspots[0]).map((col) => (
                          <td key={col}>
                            {col.toLowerCase().includes('rank') ? (
                              <span className="badge-pill badge-cyan">#{row[col]}</span>
                            ) : (
                              String(row[col])
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 10: EXECUTIVE REPORTS & EXPORT */}
        {activeTab === 'reports' && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>📋 Comprehensive Report Generation &amp; Export</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>Download structured analytical artifacts in Markdown (.md) and CSV.</p>
            </div>

            <div style={{ display: 'flex', gap: '14px', marginBottom: '20px' }}>
              <a href={`${API_BASE}/report/markdown`} download="MarineMine_Pro_Report.md" className="btn btn-primary" style={{ textDecoration: 'none' }}>
                <Download size={16} /> Download Technical Report (.md)
              </a>
              <a href={`${API_BASE}/report/csv`} download="cleaned_marine_pollution.csv" className="btn btn-secondary" style={{ textDecoration: 'none' }}>
                <Download size={16} /> Export Cleaned Telemetry (.csv)
              </a>
            </div>

            <div className="card">
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#0A3D62', marginBottom: '14px' }}>
                📄 Live Markdown Report Preview
              </h3>
              <pre style={{
                background: '#F8FAFC',
                border: '1px solid #E2E8F0',
                borderRadius: '8px',
                padding: '20px',
                fontSize: '0.85rem',
                whiteSpace: 'pre-wrap',
                maxHeight: '550px',
                overflowY: 'auto',
                color: '#1E293B'
              }}>
                {markdownReport || 'Loading generated report...'}
              </pre>
            </div>
          </div>
        )}

        {/* TAB 11: ABOUT ARCHITECTURE */}
        {activeTab === 'about' && (
          <div>
            <div style={{ marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0A3D62' }}>ℹ️ System Architecture &amp; Methodology</h2>
              <p style={{ color: '#64748B', fontSize: '0.9rem' }}>MarineMine Pro end-to-end environmental data mining and ML framework.</p>
            </div>

            <div className="card">
              <h3 style={{ color: '#0A3D62', marginBottom: '10px' }}>Full-Stack Ecosystem</h3>
              <p style={{ fontSize: '0.9rem', lineHeight: 1.6, color: '#475569', marginBottom: '14px' }}>
                <b>MarineMine Pro</b> combines high-performance Python data science algorithms with a responsive <b>React 18 + Vite SPA</b>, featuring Recharts interactive charts, Leaflet geospatial mapping, and automated statistical diagnostics.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px' }}>
                <div style={{ background: '#F8FAFC', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontWeight: 700, color: '#0A3D62', marginBottom: '4px' }}>Frontend</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748B' }}>React 18, Vite, Recharts, Leaflet, Lucide Icons</div>
                </div>
                <div style={{ background: '#F8FAFC', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontWeight: 700, color: '#0A3D62', marginBottom: '4px' }}>Backend API</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748B' }}>FastAPI, Uvicorn, Pydantic, CORS Middleware</div>
                </div>
                <div style={{ background: '#F8FAFC', padding: '14px', borderRadius: '8px' }}>
                  <div style={{ fontWeight: 700, color: '#0A3D62', marginBottom: '4px' }}>ML &amp; Analytics</div>
                  <div style={{ fontSize: '0.82rem', color: '#64748B' }}>Scikit-learn, MLxtend, NumPy, SciPy, Pandas</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
