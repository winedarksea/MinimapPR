// MapLibre operations exposed to WASM via globalThis.mapInterop.
(function () {
  "use strict";

  let _map = null;
  let _containerId = null;
  let _selectionCallback = null;
  let _contextMenuCallback = null;
  let _resizeObserver = null;
  let _markerStackSequence = 0;
  let _highlightedCopItem = null;
  let _highlightRing = null;
  let _activeTheme = "dark";
  let _zoneDrawSession = null;
  let _detectionLayerHandlersInstalled = false;

  const _markers = {};
  const _vectors = {};
  const _ellipses = {};
  const _zones = {};
  const _zoneClickHandlers = {};
  const _gdop = {};
  const _omniHalos = {};
  const _bearingWedges = {};
  const _heatmapSources = {};
  const _imageOverlays = {};
  const _trackRemoveTimers = {};
  const _detectionRemoveTimers = {};
  const DETECTION_SOURCE_ID = "detections";
  const DETECTION_CLUSTER_LAYER_ID = "detections-clusters";
  const DETECTION_CLUSTER_COUNT_LAYER_ID = "detections-cluster-count";
  const DETECTION_POINT_LAYER_ID = "detections-points";

  const TILE_CACHE_NAME = "mmpr-osm-tiles-v2";
  const OSM_URLS = [
    "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
  ];
  const DROPPED_TRACK_LINGER_MS = 30_000;
  const DETECTION_MARKER_LINGER_MS = 5_000;
  const MAX_DISPLAY_UNCERTAINTY_RADIUS_M = 500;
  const MARKER_STACK_STEP = 8;
  const MARKER_SELECTED_Z_INDEX_OFFSET = 1_000_000;

  function readCssColor(name, fallback) {
    if (typeof document === "undefined") return fallback;
    const root = document.documentElement;
    if (!root || !globalThis.getComputedStyle) return fallback;
    const value = globalThis.getComputedStyle(root).getPropertyValue(name).trim();
    return value || fallback;
  }

  function palette() {
    return {
      node: readCssColor("--mmp-sys-color-map-node", "#58a6ff"),
      effector: readCssColor("--mmp-sys-color-map-effector", "#f0883e"),
      track: readCssColor("--mmp-sys-color-map-track", "#5fd6c4"),
      trackCoasting: readCssColor("--mmp-sys-color-map-track-coasting", "#d29922"),
      trackDropped: readCssColor("--mmp-sys-color-map-track-dropped", "#6e7681"),
      detection: readCssColor("--mmp-sys-color-map-detection", "#f78166"),
      bearing: readCssColor("--mmp-sys-color-map-bearing", "#9ca3af"),
      omni: readCssColor("--mmp-sys-color-map-omni", "#c084fc"),
      warn: readCssColor("--mmp-sys-color-warn", "#d29922"),
      danger: readCssColor("--mmp-sys-color-danger", "#f85149"),
      surface: readCssColor("--md-sys-color-surface-container-low", "#161b22"),
      outline: readCssColor("--md-sys-color-outline-variant", "#30363d"),
    };
  }

  function ensureMap() {
    return _map && !_map._removed ? _map : null;
  }

  function mapContainer() {
    return document.getElementById("mmp-map") || document.getElementById("leaflet-map");
  }

  function fallbackTileDataUrl(tileSize) {
    const size = tileSize || 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    for (let i = 1; i < 4; i += 1) {
      const p = i * (size / 4);
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, size); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(size, p); ctx.stroke();
    }
    ctx.fillStyle = "rgba(226, 232, 240, 0.9)";
    ctx.font = "600 14px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Basemap unavailable", size / 2, size / 2 - 4);
    ctx.fillStyle = "rgba(148, 163, 184, 0.95)";
    ctx.font = "12px ui-monospace, Menlo, monospace";
    ctx.fillText("markers still live", size / 2, size / 2 + 16);
    return canvas.toDataURL("image/png");
  }

  function osmTileUrl(z, x, y) {
    const template = OSM_URLS[Math.abs(x + y) % OSM_URLS.length];
    return template.replace("{z}", z).replace("{x}", x).replace("{y}", y);
  }

  async function tileProtocolLoader(params, abortController) {
    const match = String(params.url || "").match(/mmpr-tiles:\/\/osm\/(\d+)\/(\d+)\/(\d+)\.png/);
    if (!match) {
      throw new Error("invalid mmpr tile url");
    }
    const [, z, x, y] = match;
    const url = osmTileUrl(z, x, y);
    try {
      if (globalThis.caches) {
        const cache = await caches.open(TILE_CACHE_NAME);
        const cached = await cache.match(url);
        if (cached) {
          return { data: await cached.arrayBuffer() };
        }
        const response = await fetch(url, { mode: "cors", signal: abortController?.signal });
        if (response.ok) {
          await cache.put(url, response.clone());
          return { data: await response.arrayBuffer() };
        }
      } else {
        const response = await fetch(url, { mode: "cors", signal: abortController?.signal });
        if (response.ok) {
          return { data: await response.arrayBuffer() };
        }
      }
    } catch (_) {
      // Fall through to generated offline tile.
    }
    const fallback = await fetch(fallbackTileDataUrl(256));
    return { data: await fallback.arrayBuffer() };
  }

  function installProtocol() {
    if (!globalThis.maplibregl || installProtocol.done) return;
    try {
      globalThis.maplibregl.addProtocol("mmpr-tiles", tileProtocolLoader);
      installProtocol.done = true;
    } catch (_) {
      installProtocol.done = true;
    }
  }

  function baseStyle() {
    const paint = basemapPaintForTheme(_activeTheme);
    return {
      version: 8,
      sources: {
        osm: {
          type: "raster",
          tiles: ["mmpr-tiles://osm/{z}/{x}/{y}.png"],
          tileSize: 256,
          attribution: "© OpenStreetMap contributors",
        },
      },
      layers: [{
        id: "osm",
        type: "raster",
        source: "osm",
        paint,
      }],
    };
  }

  function basemapPaintForTheme(theme) {
    if (theme === "light") {
      return {
        "raster-saturation": -0.55,
        "raster-contrast": 0.08,
        "raster-brightness-min": 0.82,
        "raster-brightness-max": 1.0,
      };
    }
    return {
      "raster-saturation": -0.75,
      "raster-contrast": 0.22,
      "raster-brightness-min": 0.08,
      "raster-brightness-max": 0.72,
    };
  }

  function applyBasemapTheme() {
    const map = ensureMap();
    if (!map || !map.getLayer("osm")) return;
    const paint = basemapPaintForTheme(_activeTheme);
    Object.keys(paint).forEach(function (property) {
      try { map.setPaintProperty("osm", property, paint[property]); } catch (_) {}
    });
  }

  function init(lat, lon, zoom) {
    installProtocol();
    const container = mapContainer();
    if (!container || !globalThis.maplibregl) return;
    if (_map && _containerId === container.id) {
      _map.jumpTo({ center: [lon, lat], zoom });
      _map.resize();
      return;
    }
    if (_resizeObserver) _resizeObserver.disconnect();
    if (_map) {
      try { _map.remove(); } catch (_) {}
    }
    _containerId = container.id;
    _detectionLayerHandlersInstalled = false;
    _map = new maplibregl.Map({
      container,
      style: baseStyle(),
      center: [lon, lat],
      zoom,
      attributionControl: false,
    });
    _map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "bottom-right");
    _map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");
    _map.on("load", applyBasemapTheme);
    _map.on("contextmenu", function (event) {
      if (_contextMenuCallback) {
        _contextMenuCallback(event.lngLat.lat, event.lngLat.lng, event.point.x, event.point.y, []);
      }
    });
    _resizeObserver = new ResizeObserver(function () {
      if (_map) _map.resize();
    });
    _resizeObserver.observe(container);
  }

  function resize() {
    const map = ensureMap();
    if (map) map.resize();
  }

  function markerElement(className, html) {
    const el = document.createElement("div");
    el.className = className;
    el.innerHTML = html;
    el.style.cursor = "pointer";
    return el;
  }

  function divSvg(html, width, height) {
    return '<div style="width:' + width + 'px;height:' + height + 'px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.42));">' + html + "</div>";
  }

  function makeNodeElement(color) {
    const c = palette();
    return markerElement("mmpr-map-icon mmpr-map-icon-node", divSvg(
      '<svg viewBox="0 0 34 34" width="34" height="34" aria-hidden="true">' +
        '<rect x="6" y="6" width="22" height="22" rx="7" fill="' + c.surface + '" stroke="' + color + '" stroke-width="2.4"></rect>' +
        '<circle cx="17" cy="17" r="4.2" fill="' + color + '"></circle>' +
      "</svg>", 34, 34));
  }

  function makeEffectorElement(color, bearingDeg, offline) {
    const c = palette();
    const stroke = offline ? c.trackDropped : color;
    return markerElement("mmpr-map-icon mmpr-map-icon-effector", divSvg(
      '<svg viewBox="0 0 40 40" width="40" height="40" aria-hidden="true">' +
        '<g transform="rotate(' + bearingDeg + ' 20 20)"><path d="M20 20 L11 2 A20 20 0 0 1 29 2 Z" fill="' + stroke + '" opacity="0.22"></path></g>' +
        '<rect x="12" y="15" width="16" height="11" rx="2.5" fill="' + c.surface + '" stroke="' + stroke + '" stroke-width="2"></rect>' +
        '<circle cx="20" cy="20.5" r="3.4" fill="none" stroke="' + stroke + '" stroke-width="1.8"></circle>' +
      "</svg>", 40, 40));
  }

  function makeTrackElement(color, tqi, opacity) {
    const c = palette();
    const size = 34 + Math.round((tqi || 0) * 6);
    return markerElement("mmpr-map-icon mmpr-map-icon-track", divSvg(
      '<svg viewBox="0 0 34 34" width="' + size + '" height="' + size + '" aria-hidden="true" style="opacity:' + (opacity ?? 1) + '">' +
        '<polygon points="17,4 30,17 17,30 4,17" fill="' + c.surface + '" stroke="' + color + '" stroke-width="2.2"></polygon>' +
        '<circle cx="17" cy="17" r="4.4" fill="' + color + '"></circle>' +
      "</svg>", size, size));
  }

  function makeDetectionElement(color, bearingOnly) {
    const size = bearingOnly ? 24 : 28;
    const body = bearingOnly
      ? '<circle cx="12" cy="12" r="6.6" fill="' + color + '" opacity="0.12"></circle><circle cx="12" cy="12" r="4.8" fill="none" stroke="' + color + '" stroke-width="1.5" opacity="0.58"></circle><circle cx="12" cy="12" r="2.2" fill="' + color + '" opacity="0.46"></circle>'
      : '<circle cx="14" cy="14" r="8.4" fill="none" stroke="' + color + '" stroke-width="1.5" opacity="0.34"></circle><circle cx="14" cy="14" r="5.9" fill="none" stroke="' + color + '" stroke-width="1.8" opacity="0.94"></circle><path d="M14 4.8v4.3M14 18.9v4.3M4.8 14h4.3M18.9 14h4.3" stroke="' + color + '" stroke-width="1.7" stroke-linecap="round" opacity="0.92"></path>';
    const viewBox = bearingOnly ? "0 0 24 24" : "0 0 28 28";
    return markerElement("mmpr-map-icon mmpr-map-icon-detection", divSvg('<svg viewBox="' + viewBox + '" width="' + size + '" height="' + size + '" aria-hidden="true">' + body + "</svg>", size, size));
  }

  function trackColorForStatus(status, colors) {
    if (status === "dropped" || status === "lost") return colors.trackDropped;
    if (status === "coasting") return colors.trackCoasting;
    return colors.track;
  }

  function trackOpacityForStatus(status) {
    if (status === "dropped") return 0.38;
    if (status === "lost") return 0.62;
    return 1.0;
  }

  function markerKeyForCopItem(kind, id) {
    if (kind === "track") return "track:" + id;
    if (kind === "detection") return "det:" + id;
    if (kind === "node") return "node:" + id;
    if (kind === "effector") return "effector:" + id;
    return null;
  }

  function emitSelection(kind, id) {
    if (_selectionCallback) _selectionCallback(kind, id);
  }

  function setMarker(key, marker, kind, id, timestampNs, status) {
    const old = _markers[key];
    if (old) old.remove();
    marker.getElement().addEventListener("click", function (event) {
      event.stopPropagation();
      emitSelection(kind, id);
    });
    marker._mmprStacking = { kind, id, timestampNs, status, sequence: ++_markerStackSequence };
    _markers[key] = marker;
    const map = ensureMap();
    if (map) marker.addTo(map);
    refreshMarkerStacking();
  }

  function removeMarker(key) {
    if (_markers[key]) {
      _markers[key].remove();
      delete _markers[key];
    }
    refreshMarkerStacking();
  }

  function refreshMarkerStacking() {
    const stackable = Object.values(_markers).filter(Boolean);
    stackable.sort(function (a, b) {
      const at = Number(a._mmprStacking?.timestampNs) || 0;
      const bt = Number(b._mmprStacking?.timestampNs) || 0;
      return at - bt;
    });
    stackable.forEach(function (marker, index) {
      let offset = (index + 1) * MARKER_STACK_STEP;
      if (_highlightedCopItem && marker._mmprStacking?.kind === _highlightedCopItem.kind && marker._mmprStacking?.id === _highlightedCopItem.id) {
        offset += MARKER_SELECTED_Z_INDEX_OFFSET;
      }
      marker.getElement().style.zIndex = String(offset);
    });
  }

  function setNodeMarker(nodeId, lat, lon, healthClass) {
    const color = healthClass === "ok" || healthClass === "active" ? palette().node : palette().warn;
    setMarker("node:" + nodeId, new maplibregl.Marker({ element: makeNodeElement(color) }).setLngLat([lon, lat]), "node", nodeId, 0, healthClass);
  }

  function removeNodeMarker(nodeId) { removeMarker("node:" + nodeId); }

  function setEffectorMarker(effectorId, lat, lon, bearingDeg, state) {
    const c = palette();
    setMarker("effector:" + effectorId, new maplibregl.Marker({ element: makeEffectorElement(c.effector, bearingDeg, state === "offline") }).setLngLat([lon, lat]), "effector", effectorId, 0, state);
  }

  function removeEffectorMarker(effectorId) { removeMarker("effector:" + effectorId); }

  function setTrackMarker(trackId, lat, lon, label, tqi, status, lastUpdateNs) {
    const c = palette();
    const color = trackColorForStatus(status, c);
    setMarker("track:" + trackId, new maplibregl.Marker({ element: makeTrackElement(color, tqi, trackOpacityForStatus(status)) }).setLngLat([lon, lat]), "track", trackId, lastUpdateNs, status);
    if (_trackRemoveTimers[trackId]) clearTimeout(_trackRemoveTimers[trackId]);
    if (status === "dropped") {
      _trackRemoveTimers[trackId] = setTimeout(function () { removeTrack(trackId); }, DROPPED_TRACK_LINGER_MS);
    }
  }

  function setTrackVelocityVector(trackId, lat, lon, velLat, velLon, status) {
    const map = ensureMap();
    if (!map) return;
    const sourceId = "velocity-" + trackId;
    const layerId = sourceId;
    const data = {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "LineString", coordinates: [[lon, lat], [lon + velLon, lat + velLat]] }, properties: {} }],
    };
    const color = trackColorForStatus(status, palette());
    upsertGeojsonLayer(sourceId, layerId, data, { type: "line", paint: { "line-color": color, "line-width": 2, "line-opacity": 0.68 } });
    _vectors[trackId] = layerId;
  }

  function removeTrack(trackId) {
    removeMarker("track:" + trackId);
    removeLayer("velocity-" + trackId);
    delete _vectors[trackId];
    if (_trackRemoveTimers[trackId]) clearTimeout(_trackRemoveTimers[trackId]);
  }

  function pulseTrackMarker(trackId) {
    const marker = _markers["track:" + trackId];
    if (!marker) return;
    marker.getElement().classList.remove("mmpr-map-pulse");
    void marker.getElement().offsetWidth;
    marker.getElement().classList.add("mmpr-map-pulse");
  }

  function addDetectionMarker(eventId, lat, lon, label, receivedNs) {
    setMarker("det:" + eventId, new maplibregl.Marker({ element: makeDetectionElement(palette().detection, false) }).setLngLat([lon, lat]), "detection", eventId, receivedNs, "active");
    if (_detectionRemoveTimers[eventId]) clearTimeout(_detectionRemoveTimers[eventId]);
    _detectionRemoveTimers[eventId] = setTimeout(function () { removeDetectionMarker(eventId); }, DETECTION_MARKER_LINGER_MS);
  }

  function addBearingOnlyDetectionMarker(eventId, lat, lon, label, sourceLat, sourceLon, hasSource, receivedNs) {
    addDetectionMarker(eventId, lat, lon, label, receivedNs);
    if (hasSource) {
      setGdopCircle("bearing:" + eventId, sourceLat, sourceLon, 40);
    }
  }

  function removeDetectionMarker(eventId) {
    removeMarker("det:" + eventId);
    removeGdopCircle("bearing:" + eventId);
    if (_detectionRemoveTimers[eventId]) clearTimeout(_detectionRemoveTimers[eventId]);
  }

  function ensureDetectionLayer() {
    const map = ensureMap();
    if (!map || !map.isStyleLoaded()) {
      if (map) map.once("load", ensureDetectionLayer);
      return null;
    }
    if (!map.getSource(DETECTION_SOURCE_ID)) {
      map.addSource(DETECTION_SOURCE_ID, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        cluster: true,
        clusterRadius: 44,
        clusterMaxZoom: 15,
      });
    }
    if (!map.getLayer(DETECTION_CLUSTER_LAYER_ID)) {
      map.addLayer({
        id: DETECTION_CLUSTER_LAYER_ID,
        type: "circle",
        source: DETECTION_SOURCE_ID,
        filter: ["has", "point_count"],
        paint: {
          "circle-color": palette().detection,
          "circle-opacity": 0.34,
          "circle-radius": ["step", ["get", "point_count"], 16, 10, 22, 30, 30],
          "circle-stroke-color": palette().surface,
          "circle-stroke-width": 2,
        },
      });
    }
    if (!map.getLayer(DETECTION_POINT_LAYER_ID)) {
      map.addLayer({
        id: DETECTION_POINT_LAYER_ID,
        type: "circle",
        source: DETECTION_SOURCE_ID,
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-color": [
            "case",
            ["==", ["get", "display_mode"], "bearing_only"], palette().bearing,
            palette().detection,
          ],
          "circle-radius": [
            "interpolate", ["linear"], ["coalesce", ["get", "confidence"], 0.5],
            0, 4,
            1, 8,
          ],
          "circle-opacity": [
            "interpolate", ["linear"], ["coalesce", ["get", "confidence"], 0.5],
            0, 0.28,
            1, 0.86,
          ],
          "circle-stroke-color": palette().surface,
          "circle-stroke-width": 1.8,
        },
      });
    }
    if (!_detectionLayerHandlersInstalled) {
      map.on("click", DETECTION_POINT_LAYER_ID, function (event) {
        const feature = event.features && event.features[0];
        const id = feature?.properties?.id;
        if (id) emitSelection("detection", id);
      });
      map.on("mouseenter", DETECTION_POINT_LAYER_ID, function () {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", DETECTION_POINT_LAYER_ID, function () {
        if (!_zoneDrawSession) map.getCanvas().style.cursor = "";
      });
      _detectionLayerHandlersInstalled = true;
    }
    return map.getSource(DETECTION_SOURCE_ID);
  }

  function setDetectionLayerData(dataJson) {
    const data = typeof dataJson === "string" ? JSON.parse(dataJson) : dataJson;
    const source = ensureDetectionLayer();
    if (source && typeof source.setData === "function") {
      source.setData(data && data.type === "FeatureCollection" ? data : { type: "FeatureCollection", features: [] });
    }
    Object.keys(_detectionRemoveTimers).forEach(function (eventId) {
      clearTimeout(_detectionRemoveTimers[eventId]);
      delete _detectionRemoveTimers[eventId];
      removeMarker("det:" + eventId);
    });
  }

  function clearDetectionLayer() {
    [DETECTION_CLUSTER_COUNT_LAYER_ID, DETECTION_CLUSTER_LAYER_ID, DETECTION_POINT_LAYER_ID].forEach(removeLayer);
    removeLayer(DETECTION_SOURCE_ID);
  }

  function metersToDegreeOffsets(lat, radiusM, pointCount) {
    const latMeters = 111_320;
    const lonMeters = Math.max(1, latMeters * Math.cos(lat * Math.PI / 180));
    const coords = [];
    for (let i = 0; i <= pointCount; i += 1) {
      const theta = (i / pointCount) * Math.PI * 2;
      coords.push([
        Math.cos(theta) * radiusM / lonMeters,
        Math.sin(theta) * radiusM / latMeters,
      ]);
    }
    return coords;
  }

  function setNodeOmniHalo(nodeId, lat, lon, summary) {
    const active = Number(summary?.active_count || summary?.activeCount || 0);
    const radius = Math.min(240, 45 + active * 14);
    setCirclePolygon("omni:" + nodeId, lat, lon, radius, palette().omni, 0.16, 2);
    _omniHalos[nodeId] = true;
  }

  function removeNodeOmniHalo(nodeId) {
    removeLayer("omni:" + nodeId);
    delete _omniHalos[nodeId];
  }

  function triggerNodeOmniRipple(nodeId, lat, lon, label) {
    setCirclePolygon("omni-ripple:" + nodeId, lat, lon, 80, palette().omni, 0.18, 2);
    setTimeout(function () { removeLayer("omni-ripple:" + nodeId); }, 1600);
  }

  function setCirclePolygon(id, lat, lon, radiusM, color, opacity, strokeWidth) {
    const offsets = metersToDegreeOffsets(lat, radiusM, 72);
    const coordinates = offsets.map(([dx, dy]) => [lon + dx, lat + dy]);
    const data = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "Polygon", coordinates: [coordinates] }, properties: {} }] };
    upsertGeojsonLayer(id, id, data, {
      type: "fill",
      paint: { "fill-color": color, "fill-opacity": opacity || 0.12 },
    });
    const lineId = id + "-line";
    const lineData = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "LineString", coordinates }, properties: {} }] };
    upsertGeojsonLayer(lineId, lineId, lineData, {
      type: "line",
      paint: { "line-color": color, "line-opacity": 0.72, "line-width": strokeWidth || 1 },
    });
  }

  function covarianceEllipseCoordinates(lat, lon, covariance) {
    if (!Array.isArray(covariance) || covariance.length < 2 || !Array.isArray(covariance[0]) || !Array.isArray(covariance[1])) return null;
    const a = Number(covariance[0][0]);
    const b = Number(covariance[0][1] ?? covariance[1][0] ?? 0);
    const d = Number(covariance[1][1]);
    if (![a, b, d].every(Number.isFinite)) return null;
    const trace = a + d;
    const disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
    const major = Math.sqrt(Math.max(0, (trace + disc) / 2));
    const minor = Math.sqrt(Math.max(0, (trace - disc) / 2));
    const angle = 0.5 * Math.atan2(2 * b, a - d);
    const cappedMajor = Math.min(major, MAX_DISPLAY_UNCERTAINTY_RADIUS_M);
    const cappedMinor = Math.min(minor, MAX_DISPLAY_UNCERTAINTY_RADIUS_M);
    const latMeters = 111_320;
    const lonMeters = Math.max(1, latMeters * Math.cos(lat * Math.PI / 180));
    const coordinates = [];
    for (let i = 0; i <= 72; i += 1) {
      const t = (i / 72) * Math.PI * 2;
      const x = cappedMajor * Math.cos(t);
      const y = cappedMinor * Math.sin(t);
      const xr = x * Math.cos(angle) - y * Math.sin(angle);
      const yr = x * Math.sin(angle) + y * Math.cos(angle);
      coordinates.push([lon + xr / lonMeters, lat + yr / latMeters]);
    }
    return coordinates;
  }

  function destinationCoordinate(lat, lon, bearingDeg, distanceM) {
    const radiusM = 6_371_000;
    const angularDistance = distanceM / radiusM;
    const bearing = bearingDeg * Math.PI / 180;
    const lat1 = lat * Math.PI / 180;
    const lon1 = lon * Math.PI / 180;
    const lat2 = Math.asin(
      Math.sin(lat1) * Math.cos(angularDistance) +
      Math.cos(lat1) * Math.sin(angularDistance) * Math.cos(bearing),
    );
    const lon2 = lon1 + Math.atan2(
      Math.sin(bearing) * Math.sin(angularDistance) * Math.cos(lat1),
      Math.cos(angularDistance) - Math.sin(lat1) * Math.sin(lat2),
    );
    return [lon2 * 180 / Math.PI, lat2 * 180 / Math.PI];
  }

  function bearingWedgeCoordinates(lat, lon, bearingDeg, halfAngleDeg, rangeM) {
    const bearing = Number(bearingDeg);
    const halfAngle = Math.max(1, Math.min(80, Number(halfAngleDeg)));
    const range = Math.max(10, Math.min(10_000, Number(rangeM)));
    if (![lat, lon, bearing, halfAngle, range].every(Number.isFinite)) return null;

    const coordinates = [[lon, lat]];
    const steps = Math.max(8, Math.ceil(halfAngle / 4) * 2);
    for (let index = 0; index <= steps; index += 1) {
      const t = index / steps;
      const angle = bearing - halfAngle + (halfAngle * 2 * t);
      coordinates.push(destinationCoordinate(lat, lon, angle, range));
    }
    coordinates.push([lon, lat]);
    return coordinates;
  }

  function setBearingWedge(id, lat, lon, bearingDeg, halfAngleDeg, rangeM) {
    const coordinates = bearingWedgeCoordinates(lat, lon, bearingDeg, halfAngleDeg, rangeM);
    if (!coordinates) return;
    const layerId = "bearing-wedge:" + id;
    const color = palette().bearing;
    const data = {
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [coordinates] },
        properties: { id },
      }],
    };
    upsertGeojsonLayer(layerId, layerId, data, {
      type: "fill",
      paint: { "fill-color": color, "fill-opacity": 0.16 },
    });
    const lineId = layerId + "-line";
    upsertGeojsonLayer(lineId, lineId, {
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        geometry: { type: "LineString", coordinates },
        properties: { id },
      }],
    }, {
      type: "line",
      paint: { "line-color": color, "line-opacity": 0.62, "line-width": 1.4 },
    });
    _bearingWedges[id] = true;
  }

  function removeBearingWedge(id) {
    removeLayer("bearing-wedge:" + id);
    removeLayer("bearing-wedge:" + id + "-line");
    delete _bearingWedges[id];
  }

  function clearBearingWedges() {
    Object.keys(_bearingWedges).forEach(removeBearingWedge);
  }

  function setCopUncertainty(kind, id, lat, lon, covariance) {
    const coordinates = covarianceEllipseCoordinates(lat, lon, covariance);
    if (!coordinates) return;
    const layerId = "uncertainty:" + kind + ":" + id;
    const color = kind === "detection" ? palette().detection : palette().track;
    const data = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "Polygon", coordinates: [coordinates] }, properties: {} }] };
    upsertGeojsonLayer(layerId, layerId, data, { type: "fill", paint: { "fill-color": color, "fill-opacity": 0.14 } });
    _ellipses[layerId] = true;
  }

  function clearAllCopUncertainty() {
    Object.keys(_ellipses).forEach(removeLayer);
    Object.keys(_ellipses).forEach(function (id) { delete _ellipses[id]; });
  }

  function setZone(zoneId, latlngs, label) {
    const coordinates = Array.isArray(latlngs) ? latlngs.map(function (p) { return [p[1], p[0]]; }) : [];
    if (coordinates.length < 3) return;
    if (coordinates[0][0] !== coordinates[coordinates.length - 1][0] || coordinates[0][1] !== coordinates[coordinates.length - 1][1]) {
      coordinates.push(coordinates[0]);
    }
    const id = "zone:" + zoneId;
    const data = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "Polygon", coordinates: [coordinates] }, properties: { label } }] };
    upsertGeojsonLayer(id, id, data, { type: "fill", paint: { "fill-color": palette().warn, "fill-opacity": 0.09 } });
    upsertGeojsonLayer(id + "-line", id + "-line", { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "LineString", coordinates }, properties: {} }] }, { type: "line", paint: { "line-color": palette().warn, "line-opacity": 0.72, "line-width": 2 } });
    const map = ensureMap();
    if (map && !_zoneClickHandlers[zoneId]) {
      const handler = function (event) {
        event.preventDefault();
        emitSelection("zone", zoneId);
      };
      map.on("click", id, handler);
      _zoneClickHandlers[zoneId] = handler;
    }
    _zones[zoneId] = true;
  }

  function removeZone(zoneId) {
    const map = ensureMap();
    if (map && _zoneClickHandlers[zoneId]) {
      map.off("click", "zone:" + zoneId, _zoneClickHandlers[zoneId]);
      delete _zoneClickHandlers[zoneId];
    }
    removeLayer("zone:" + zoneId);
    removeLayer("zone:" + zoneId + "-line");
    delete _zones[zoneId];
  }

  function setGdopCircle(key, lat, lon, radiusM) {
    setCirclePolygon("gdop:" + key, lat, lon, radiusM, palette().bearing, 0.08, 1);
    _gdop[key] = true;
  }

  function removeGdopCircle(key) {
    removeLayer("gdop:" + key);
    removeLayer("gdop:" + key + "-line");
    delete _gdop[key];
  }

  function panTo(lat, lon) {
    const map = ensureMap();
    if (map) map.panTo([lon, lat], { duration: 500 });
  }

  function flyTo(lat, lon, zoom) {
    const map = ensureMap();
    if (map) map.flyTo({ center: [lon, lat], zoom: zoom || map.getZoom(), duration: 700 });
  }

  function highlightCopItem(kind, id) {
    _highlightedCopItem = { kind, id };
    refreshMarkerStacking();
    const marker = _markers[markerKeyForCopItem(kind, id)];
    if (!marker) return;
    const lngLat = marker.getLngLat();
    const color = kind === "detection" ? palette().detection : palette().track;
    if (_highlightRing) removeLayer(_highlightRing);
    _highlightRing = "highlight:" + kind + ":" + id;
    setCirclePolygon(_highlightRing, lngLat.lat, lngLat.lng, 36, color, 0.18, 2);
  }

  function clearCopHighlight() {
    _highlightedCopItem = null;
    if (_highlightRing) {
      removeLayer(_highlightRing);
      removeLayer(_highlightRing + "-line");
      _highlightRing = null;
    }
    refreshMarkerStacking();
  }

  function setCopSelectionCallback(callback) {
    _selectionCallback = typeof callback === "function" ? callback : null;
  }

  function setContextMenuCallback(callback) {
    _contextMenuCallback = typeof callback === "function" ? callback : null;
  }

  function setHeatmapPoints(points, maxIntensity) {
    const features = Array.isArray(points) ? points.map(function (p) {
      const lat = Number(p.lat ?? p[0]);
      const lon = Number(p.lon ?? p[1]);
      const value = Number(p.intensity ?? p.value ?? p[2] ?? 1);
      return { type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: { value } };
    }).filter(function (f) { return Number.isFinite(f.geometry.coordinates[0]) && Number.isFinite(f.geometry.coordinates[1]); }) : [];
    const data = { type: "FeatureCollection", features };
    upsertGeojsonLayer("heatmap", "heatmap", data, {
      type: "heatmap",
      paint: {
        "heatmap-weight": ["interpolate", ["linear"], ["get", "value"], 0, 0, maxIntensity || 1, 1],
        "heatmap-intensity": 1.2,
        "heatmap-radius": 28,
        "heatmap-opacity": 0.72,
      },
    });
    _heatmapSources.heatmap = true;
  }

  function clearHeatmap() { removeLayer("heatmap"); delete _heatmapSources.heatmap; }

  function fitBoundsLatLons(points) {
    const map = ensureMap();
    if (!map || !Array.isArray(points) || points.length === 0) return;
    const bounds = new maplibregl.LngLatBounds();
    points.forEach(function (p) {
      const lat = Number(p.lat ?? p[0]);
      const lon = Number(p.lon ?? p[1]);
      if (Number.isFinite(lat) && Number.isFinite(lon)) bounds.extend([lon, lat]);
    });
    if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 48, duration: 500 });
  }

  function upsertGeojsonLayer(sourceId, layerId, data, spec) {
    const map = ensureMap();
    if (!map || !map.isStyleLoaded()) {
      if (map) map.once("load", function () { upsertGeojsonLayer(sourceId, layerId, data, spec); });
      return;
    }
    if (map.getSource(sourceId)) {
      map.getSource(sourceId).setData(data);
    } else {
      map.addSource(sourceId, { type: "geojson", data });
    }
    if (!map.getLayer(layerId)) {
      map.addLayer(Object.assign({ id: layerId, source: sourceId }, spec));
    }
  }

  function ensureLayer(layerId, specJson) {
    const spec = typeof specJson === "string" ? JSON.parse(specJson) : specJson;
    if (!spec || spec.type !== "geojson") return;
    upsertGeojsonLayer(layerId, layerId, { type: "FeatureCollection", features: [] }, spec.layer || { type: "circle", paint: {} });
  }

  function setLayerData(layerId, dataJson) {
    const map = ensureMap();
    const data = typeof dataJson === "string" ? JSON.parse(dataJson) : dataJson;
    if (map && map.getSource(layerId)) map.getSource(layerId).setData(data);
  }

  function setLayerVisible(layerId, visible) {
    const map = ensureMap();
    if (map && map.getLayer(layerId)) map.setLayoutProperty(layerId, "visibility", visible ? "visible" : "none");
  }

  function removeLayer(layerId) {
    const map = ensureMap();
    if (!map || !map.isStyleLoaded()) return;
    if (map.getLayer(layerId)) map.removeLayer(layerId);
    if (map.getSource(layerId)) map.removeSource(layerId);
  }

  function setImageOverlay(id, url, corners, opacity) {
    const map = ensureMap();
    if (!map || !Array.isArray(corners) || corners.length !== 4) return;
    const coordinates = corners.map(function (p) { return [p[1], p[0]]; });
    removeMapOverlay(id);
    map.addSource(id, { type: "image", url, coordinates });
    map.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": opacity ?? 0.72 } });
    _imageOverlays[id] = true;
  }

  function removeMapOverlay(id) {
    const map = ensureMap();
    if (!map || !map.isStyleLoaded()) return;
    [id + "-point", id + "-line", id + "-fill", id].forEach(function (layerId) {
      if (map.getLayer(layerId)) map.removeLayer(layerId);
    });
    if (map.getSource(id)) map.removeSource(id);
    delete _imageOverlays[id];
  }

  function setGeoJsonOverlay(id, dataJson, opacity) {
    const map = ensureMap();
    if (!map || !map.isStyleLoaded()) return;
    const data = typeof dataJson === "string" ? JSON.parse(dataJson) : dataJson;
    if (!data || data.type !== "FeatureCollection") return;
    const source = map.getSource(id);
    if (source && typeof source.setData === "function") {
      source.setData(data);
    } else {
      if (map.getSource(id)) removeMapOverlay(id);
      map.addSource(id, { type: "geojson", data });
    }
    const alpha = opacity ?? 0.72;
    if (!map.getLayer(id + "-fill")) {
      map.addLayer({
        id: id + "-fill",
        type: "fill",
        source: id,
        filter: ["match", ["geometry-type"], ["Polygon", "MultiPolygon"], true, false],
        paint: {
          "fill-color": readCssColor("--mmp-sys-color-info", "#4cc9f0"),
          "fill-opacity": Math.max(0.08, alpha * 0.28)
        }
      });
    }
    if (!map.getLayer(id + "-line")) {
      map.addLayer({
        id: id + "-line",
        type: "line",
        source: id,
        filter: ["match", ["geometry-type"], ["LineString", "MultiLineString", "Polygon", "MultiPolygon"], true, false],
        paint: {
          "line-color": readCssColor("--mmp-sys-color-info", "#4cc9f0"),
          "line-width": 2,
          "line-opacity": Math.max(0.18, alpha)
        }
      });
    }
    if (!map.getLayer(id + "-point")) {
      map.addLayer({
        id: id + "-point",
        type: "circle",
        source: id,
        filter: ["match", ["geometry-type"], ["Point", "MultiPoint"], true, false],
        paint: {
          "circle-color": readCssColor("--mmp-sys-color-info", "#4cc9f0"),
          "circle-radius": 5,
          "circle-opacity": Math.max(0.18, alpha),
          "circle-stroke-color": readCssColor("--md-sys-color-surface", "#111"),
          "circle-stroke-width": 1.5
        }
      });
    }
  }

  function latLngsToClosedCoordinates(latlngs) {
    const coordinates = Array.isArray(latlngs)
      ? latlngs
        .filter(function (point) { return Array.isArray(point) && point.length >= 2; })
        .map(function (point) { return [Number(point[1]), Number(point[0])]; })
        .filter(function (point) { return Number.isFinite(point[0]) && Number.isFinite(point[1]); })
      : [];
    if (coordinates.length >= 3) {
      const first = coordinates[0];
      const last = coordinates[coordinates.length - 1];
      if (first[0] !== last[0] || first[1] !== last[1]) coordinates.push(first);
    }
    return coordinates;
  }

  function coordinatesToLatLngs(coordinates) {
    if (!Array.isArray(coordinates)) return [];
    const result = coordinates
      .map(function (point) { return [Number(point[1]), Number(point[0])]; })
      .filter(function (point) { return Number.isFinite(point[0]) && Number.isFinite(point[1]); });
    if (result.length > 1) {
      const first = result[0];
      const last = result[result.length - 1];
      if (first[0] === last[0] && first[1] === last[1]) result.pop();
    }
    return result;
  }

  function updateZoneDraftLayer(latlngs) {
    const coordinates = latLngsToClosedCoordinates(latlngs);
    if (coordinates.length < 2) {
      removeLayer("zone-draft");
      removeLayer("zone-draft-line");
      return;
    }
    const lineCoordinates = coordinates.length >= 3 ? coordinates : coordinates.slice();
    if (coordinates.length >= 3) {
      upsertGeojsonLayer("zone-draft", "zone-draft", {
        type: "FeatureCollection",
        features: [{ type: "Feature", geometry: { type: "Polygon", coordinates: [coordinates] }, properties: {} }],
      }, {
        type: "fill",
        paint: {
          "fill-color": palette().warn,
          "fill-opacity": 0.13,
        },
      });
    } else {
      removeLayer("zone-draft");
    }
    upsertGeojsonLayer("zone-draft-line", "zone-draft-line", {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "LineString", coordinates: lineCoordinates }, properties: {} }],
    }, {
      type: "line",
      paint: {
        "line-color": palette().warn,
        "line-opacity": 0.9,
        "line-width": 2,
        "line-dasharray": [2, 1],
      },
    });
  }

  function finishZoneSession() {
    if (!_zoneDrawSession) return;
    const session = _zoneDrawSession;
    if (session.latlngs.length >= 3 && typeof session.callback === "function") {
      session.callback(session.latlngs);
    }
    cancelZoneDraw();
  }

  function startZoneDraw(callback) {
    const map = ensureMap();
    if (!map) return;
    cancelZoneDraw();
    const session = {
      mode: "draw",
      callback,
      latlngs: [],
      markers: [],
      handlers: {},
    };
    session.handlers.click = function (event) {
      session.latlngs.push([event.lngLat.lat, event.lngLat.lng]);
      updateZoneDraftLayer(session.latlngs);
    };
    session.handlers.dblclick = function (event) {
      event.preventDefault();
      finishZoneSession();
    };
    session.handlers.keydown = function (event) {
      if (event.key === "Enter") finishZoneSession();
      if (event.key === "Escape") cancelZoneDraw();
    };
    _zoneDrawSession = session;
    try { map.doubleClickZoom.disable(); } catch (_) {}
    map.getCanvas().style.cursor = "crosshair";
    map.on("click", session.handlers.click);
    map.on("dblclick", session.handlers.dblclick);
    document.addEventListener("keydown", session.handlers.keydown);
  }

  function startZoneEdit(zoneId, latlngs, callback) {
    const map = ensureMap();
    if (!map) return;
    cancelZoneDraw();
    const session = {
      mode: "edit",
      zoneId,
      callback,
      latlngs: Array.isArray(latlngs) ? latlngs.map(function (point) { return [Number(point[0]), Number(point[1])]; }) : [],
      markers: [],
      handlers: {},
    };
    function emitEdit() {
      updateZoneDraftLayer(session.latlngs);
      if (typeof session.callback === "function") session.callback(session.latlngs);
    }
    session.latlngs.forEach(function (point, index) {
      if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) return;
      const marker = new maplibregl.Marker({ draggable: true })
        .setLngLat([point[1], point[0]])
        .addTo(map);
      marker.on("dragend", function () {
        const lngLat = marker.getLngLat();
        session.latlngs[index] = [lngLat.lat, lngLat.lng];
        emitEdit();
      });
      session.markers.push(marker);
    });
    session.handlers.click = function (event) {
      const nextIndex = session.latlngs.length;
      session.latlngs.push([event.lngLat.lat, event.lngLat.lng]);
      const marker = new maplibregl.Marker({ draggable: true })
        .setLngLat([event.lngLat.lng, event.lngLat.lat])
        .addTo(map);
      marker.on("dragend", function () {
        const lngLat = marker.getLngLat();
        session.latlngs[nextIndex] = [lngLat.lat, lngLat.lng];
        emitEdit();
      });
      session.markers.push(marker);
      emitEdit();
    };
    session.handlers.keydown = function (event) {
      if (event.key === "Escape") cancelZoneDraw();
      if (event.key === "Enter") finishZoneSession();
    };
    _zoneDrawSession = session;
    map.getCanvas().style.cursor = "crosshair";
    map.on("click", session.handlers.click);
    document.addEventListener("keydown", session.handlers.keydown);
    emitEdit();
  }

  function cancelZoneDraw() {
    const map = ensureMap();
    const session = _zoneDrawSession;
    if (session && map) {
      if (session.handlers.click) map.off("click", session.handlers.click);
      if (session.handlers.dblclick) map.off("dblclick", session.handlers.dblclick);
      try { map.doubleClickZoom.enable(); } catch (_) {}
      map.getCanvas().style.cursor = "";
    }
    if (session && session.handlers.keydown) {
      document.removeEventListener("keydown", session.handlers.keydown);
    }
    if (session) {
      session.markers.forEach(function (marker) { try { marker.remove(); } catch (_) {} });
    }
    _zoneDrawSession = null;
    removeLayer("zone-draft");
    removeLayer("zone-draft-line");
  }

  function setTheme(theme) {
    _activeTheme = theme === "light" ? "light" : "dark";
    applyBasemapTheme();
  }

  globalThis.mapInterop = {
    init, resize, invalidateMapSize: resize, panTo, flyTo, fitBoundsLatLons, setTheme,
    setSelectionCallback: setCopSelectionCallback,
    setCopSelectionCallback,
    setContextMenuCallback,
    setNodeMarker, removeNodeMarker, setNodeOmniHalo, removeNodeOmniHalo, triggerNodeOmniRipple,
    addDetectionMarker, addBearingOnlyDetectionMarker, removeDetectionMarker,
    setDetectionLayerData, clearDetectionLayer,
    setTrackMarker, setTrackVelocityVector, removeTrack, pulseTrackMarker,
    setEffectorMarker, removeEffectorMarker, setZone, removeZone, setGdopCircle, removeGdopCircle,
    highlightCopItem, clearCopHighlight, setCopUncertainty, clearAllCopUncertainty,
    setBearingWedge, removeBearingWedge, clearBearingWedges,
    setHeatmapPoints, clearHeatmap,
    ensureLayer, setLayerData, setLayerVisible, removeLayer, setImageOverlay, setGeoJsonOverlay, removeMapOverlay,
    startZoneDraw, startZoneEdit, cancelZoneDraw,
    _test: {
      fallbackTileDataUrl,
      covarianceEllipseCoordinates,
      bearingWedgeCoordinates,
      readCssColor,
      latLngsToClosedCoordinates,
      coordinatesToLatLngs,
      basemapPaintForTheme,
    },
  };
}());
