// Minimal shim exposing Leaflet operations to WASM via globalThis.leafletInterop.
// Leaflet is loaded from CDN in index.html before this script runs.

(function () {
  "use strict";

  let _map = null;
  const _markers = {};           // key → L.Marker or L.CircleMarker
  const _vectors = {};           // track_id → L.Polyline (velocity)
  const _ellipses = {};          // track_id → L.Ellipse (covariance)
  const _zones = {};             // zone_id → L.Polygon
  const _gdop = {};              // key → L.Circle
  const _trackRemoveTimers = {}; // track_id → setTimeout handle for dropped cleanup
  let _highlightedCopItem = null;
  let _highlightRing = null;
  let _copSelectionCallback = null;

  const TILE_CACHE_NAME = "mmpr-osm-tiles-v2";
  const OSM_TEMPLATE = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  // Delay before a "dropped" track marker is auto-removed from the map.
  const DROPPED_TRACK_LINGER_MS = 30_000;

  function readCssColor(name, fallback) {
    const root = document.documentElement;
    if (!root || !globalThis.getComputedStyle) return fallback;
    const value = globalThis.getComputedStyle(root).getPropertyValue(name).trim();
    return value || fallback;
  }

  function palette() {
    return {
      node:          readCssColor("--mmp-sys-color-map-node",           "#58a6ff"),
      track:         readCssColor("--mmp-sys-color-map-track",          "#5fd6c4"),
      trackCoasting: readCssColor("--mmp-sys-color-map-track-coasting", "#d29922"),
      trackDropped:  readCssColor("--mmp-sys-color-map-track-dropped",  "#6e7681"),
      detection:     readCssColor("--mmp-sys-color-map-detection",      "#f78166"),
      warn:          readCssColor("--mmp-sys-color-warn",               "#d29922"),
      danger:        readCssColor("--mmp-sys-color-danger",             "#f85149"),
      surface:       readCssColor("--md-sys-color-surface-container-low", "#161b22"),
      outline:       readCssColor("--md-sys-color-outline-variant",    "#30363d"),
    };
  }

  function trackColorForStatus(status, colors) {
    if (status === "dropped" || status === "lost") return colors.trackDropped;
    if (status === "coasting") return colors.trackCoasting;
    return colors.track;
  }

  function trackOpacityForStatus(status) {
    if (status === "dropped") return 0.38;
    if (status === "lost")    return 0.62;
    return 1.0;
  }

  function markerKeyForCopItem(kind, id) {
    if (kind === "track") return "track:" + id;
    if (kind === "detection") return "det:" + id;
    return null;
  }

  function colorForCopItem(kind) {
    const colors = palette();
    if (kind === "detection") return colors.detection;
    return colors.track;
  }

  function setCopSelectionCallback(callback) {
    _copSelectionCallback = typeof callback === "function" ? callback : null;
  }

  function emitCopSelection(kind, id) {
    if (_copSelectionCallback) _copSelectionCallback(kind, id);
  }

  function attachCopSelectionHandler(marker, kind, id) {
    marker.on("click", function (event) {
      if (event && event.originalEvent) L.DomEvent.stop(event.originalEvent);
      emitCopSelection(kind, id);
    });
  }

  function divIcon(html, size, className) {
    return L.divIcon({
      className: className || "mmpr-map-icon",
      html,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
      tooltipAnchor: [0, -(size / 2) + 2],
    });
  }

  function makeNodeIcon(color) {
    const colors = palette();
    return divIcon(
      '<div style="width:34px;height:34px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.42));">' +
        '<svg viewBox="0 0 34 34" width="34" height="34" aria-hidden="true">' +
          '<rect x="6" y="6" width="22" height="22" rx="7" fill="' + colors.surface + '" stroke="' + color + '" stroke-width="2.4"></rect>' +
          '<circle cx="17" cy="17" r="4.2" fill="' + color + '"></circle>' +
        "</svg>" +
      "</div>",
      34,
      "mmpr-map-icon mmpr-map-icon-node"
    );
  }

  function makeTrackIcon(color, tqi, opacity) {
    const colors = palette();
    const size = 34 + Math.round((tqi || 0) * 6);
    const offset = (size - 34) / 2;
    const op = (opacity !== undefined && opacity !== null) ? opacity : 1.0;
    return divIcon(
      '<div style="width:' + size + "px;height:" + size + 'px;filter:drop-shadow(0 2px 7px rgba(0,0,0,.46));opacity:' + op + ';">' +
        '<svg viewBox="0 0 34 34" width="' + size + '" height="' + size + '" aria-hidden="true">' +
          '<g transform="translate(' + offset + "," + offset + ')">' +
            '<polygon points="17,4 30,17 17,30 4,17" fill="' + colors.surface + '" stroke="' + color + '" stroke-width="2.2"></polygon>' +
            '<circle cx="17" cy="17" r="4.4" fill="' + color + '"></circle>' +
          "</g>" +
        "</svg>" +
      "</div>",
      size,
      "mmpr-map-icon mmpr-map-icon-track"
    );
  }

  function makeDetectionIcon(color) {
    return divIcon(
      '<div style="width:28px;height:28px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.42));">' +
        '<svg viewBox="0 0 28 28" width="28" height="28" aria-hidden="true">' +
          '<circle cx="14" cy="14" r="8.4" fill="' + color + '" opacity="0.18"></circle>' +
          '<circle cx="14" cy="14" r="6.2" fill="none" stroke="' + color + '" stroke-width="1.8" opacity="0.92"></circle>' +
          '<circle cx="14" cy="14" r="3.4" fill="' + color + '"></circle>' +
        "</svg>" +
      "</div>",
      28,
      "mmpr-map-icon mmpr-map-icon-detection"
    );
  }

  function makeGenericFallbackTileDataUrl(tileSize) {
    const size = tileSize || 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, size, size);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    const step = size / 4;
    for (let i = 1; i < 4; i += 1) {
      const p = i * step;
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

  function getTileCache() {
    if (!globalThis.caches || !globalThis.caches.open) return Promise.resolve(null);
    return globalThis.caches.open(TILE_CACHE_NAME).catch(() => null);
  }

  function requestFromUrl(url) {
    return new Request(url, { mode: "cors", credentials: "omit" });
  }

  function responseToObjectUrl(response) {
    return response.blob().then(function (blob) {
      return URL.createObjectURL(blob);
    });
  }

  // Resilient OSM tile layer with a fixed createTile implementation.
  //
  // Previous approach: called L.TileLayer.prototype.createTile (which sets
  // up img.onload/img.onerror internally), then added a separate addEventListener
  // for "error". When a tile failed, Leaflet's onerror called done(err), and
  // our handler then changed tile.src, triggering onload again → done(null)
  // called a second time. This double-done confused Leaflet's tile manager,
  // breaking zoom interactions until page refresh.
  //
  // New approach: own createTile entirely. Call done() exactly once per tile.
  // Cache hit → load from blob URL → done(null).
  // Cache miss → load from network → done(null), then opportunistically cache.
  // Network error → render fallback canvas → done(null, tile as "loaded").
  function createResilientOsmTileLayer() {
    const ResilientTileLayer = L.TileLayer.extend({
      createTile: function (coords, done) {
        const tile = document.createElement("img");
        tile.setAttribute("role", "presentation");
        tile.crossOrigin = "";
        const url = this.getTileUrl(coords);
        const tileSize = this.getTileSize();
        const width = (tileSize && tileSize.x) || 256;

        (async () => {
          // 1. Try the persistent cache — avoids the blank-map flash on reload.
          let blobUrl = null;
          try {
            const cache = await getTileCache();
            if (cache) {
              const cached = await cache.match(requestFromUrl(url));
              if (cached) blobUrl = await responseToObjectUrl(cached);
            }
          } catch (_) { /* cache unavailable */ }

          if (blobUrl) {
            tile.onload = function () {
              tile.onload = null; tile.onerror = null;
              URL.revokeObjectURL(blobUrl);
              done(null, tile);
            };
            tile.onerror = function () {
              // Corrupt or expired cache entry — fall through to network.
              tile.onload = null; tile.onerror = null;
              URL.revokeObjectURL(blobUrl);
              loadFromNetwork();
            };
            tile.src = blobUrl;
          } else {
            loadFromNetwork();
          }

          function loadFromNetwork() {
            tile.onload = function () {
              tile.onload = null; tile.onerror = null;
              done(null, tile);
              // Opportunistically cache the tile for offline/rate-limit resilience.
              (async () => {
                try {
                  const cache = await getTileCache();
                  if (!cache) return;
                  if (await cache.match(requestFromUrl(url))) return;
                  const resp = await fetch(requestFromUrl(url), { mode: "cors", credentials: "omit" });
                  if (resp.ok) await cache.put(requestFromUrl(url), resp);
                } catch (_) { /* best-effort */ }
              })();
            };
            tile.onerror = function () {
              tile.onload = null; tile.onerror = null;
              // Show a labelled placeholder so overlay markers remain visible.
              // Pass done(null) so Leaflet treats the tile as loaded (not errored),
              // which keeps zoom and pan working correctly.
              const fallback = makeGenericFallbackTileDataUrl(width);
              tile.onload = function () { tile.onload = null; done(null, tile); };
              tile.src = fallback;
            };
            tile.src = url;
          }
        })();

        return tile;
      },
    });

    return new ResilientTileLayer(OSM_TEMPLATE, {
      attribution: "© OpenStreetMap contributors",
      maxZoom: 18,
      maxNativeZoom: 18,
      updateWhenIdle: true,
      keepBuffer: 2,
      subdomains: "abc",
    });
  }


  // ── Init ──────────────────────────────────────────────────────
  function init(lat, lon, zoom) {
    // During a Leptos route transition both the old and new #leaflet-map divs
    // can briefly coexist in the DOM. getElementById returns only the first
    // match, which is the OLD (already-initialised) element. querySelectorAll
    // gives us all matches so we can prefer the uninitialised (new) one.
    const candidates = document.querySelectorAll("[id='leaflet-map']");
    let target = null;
    for (const el of candidates) {
      if (!el._leaflet_id) { target = el; break; }
    }
    if (!target) target = candidates[0] || null;
    if (!target) return; // container not yet in DOM

    if (_map) {
      const prev = _map.getContainer ? _map.getContainer() : null;
      // Reuse only when the existing map is already bound to this exact element
      // and that element is still live. Any other state means a rebuild is needed.
      if (prev && prev === target && document.body.contains(prev)) {
        _map.invalidateSize();
        return;
      }
      try { _map.remove(); } catch (_) {}
      _map = null;
      for (const k in _markers)  delete _markers[k];
      for (const k in _vectors)  delete _vectors[k];
      for (const k in _ellipses) delete _ellipses[k];
      for (const k in _zones)    delete _zones[k];
      for (const k in _gdop)     delete _gdop[k];
      for (const k in _trackRemoveTimers) { clearTimeout(_trackRemoveTimers[k]); delete _trackRemoveTimers[k]; }
      clearCopHighlight();
    }
    try {
      _map = L.map(target, {
        center: [lat, lon],
        zoom: zoom ?? 17,
        zoomControl: true,
      });
    } catch (err) {
      // "Map container is already initialized." — element still has _leaflet_id
      // from a previous session. Remove the stale state and retry once.
      try { delete target._leaflet_id; _map = L.map(target, { center: [lat, lon], zoom: zoom ?? 17, zoomControl: true }); }
      catch (_) { return; }
    }
    createResilientOsmTileLayer().addTo(_map);
    // Deferred invalidateSize handles cases where the map container's flex
    // dimensions settle after the initial paint (e.g. WASM hydration timing).
    setTimeout(function () { if (_map) _map.invalidateSize(); }, 100);

    // ResizeObserver fires once the container reaches its settled flex
    // dimensions, ensuring invalidateSize sees the real layout. This fixes
    // zoom interactions that break when init() runs before layout is complete.
    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(function (entries) {
        const entry = entries[0];
        if (!entry) return;
        const rect = entry.contentRect;
        if (rect.width > 0 && rect.height > 0) {
          if (_map) _map.invalidateSize();
          ro.disconnect();
        }
      });
      ro.observe(target);
    }
  }

  // ── Node markers ──────────────────────────────────────────────
  function setNodeMarker(nodeId, lat, lon, healthClass) {
    const colors = palette();
    const color = healthClass === "online" ? colors.node
                : healthClass === "degraded" ? colors.warn
                : colors.danger;
    const key = "node:" + nodeId;
    if (_markers[key]) {
      _markers[key].setLatLng([lat, lon]);
      _markers[key].setIcon(makeNodeIcon(color));
    } else {
      _markers[key] = L.marker([lat, lon], {
        icon: makeNodeIcon(color),
      }).bindTooltip(nodeId, { permanent: false }).addTo(_map);
    }
  }

  function removeNodeMarker(nodeId) {
    const key = "node:" + nodeId;
    if (_markers[key]) { _markers[key].remove(); delete _markers[key]; }
  }

  // ── Detection markers ─────────────────────────────────────────
  function addDetectionMarker(eventId, lat, lon, label) {
    const colors = palette();
    const key = "det:" + eventId;
    if (_markers[key]) {
      _markers[key].setLatLng([lat, lon]);
      _markers[key].setIcon(makeDetectionIcon(colors.detection));
      _markers[key].setTooltipContent(label || "detection");
    } else {
      _markers[key] = L.marker([lat, lon], {
        icon: makeDetectionIcon(colors.detection),
      }).bindTooltip(label || "detection", { permanent: false }).addTo(_map);
      attachCopSelectionHandler(_markers[key], "detection", eventId);
    }

    if (_highlightedCopItem && _highlightedCopItem.kind === "detection" && _highlightedCopItem.id === eventId) {
      highlightCopItem("detection", eventId);
    }
  }

  function removeDetectionMarker(eventId) {
    if (_highlightedCopItem && _highlightedCopItem.kind === "detection" && _highlightedCopItem.id === eventId) {
      clearCopHighlight();
    }
    const key = "det:" + eventId;
    if (_markers[key]) { _markers[key].remove(); delete _markers[key]; }
  }

  // ── Track markers + velocity vectors ─────────────────────────
  function setTrackMarker(trackId, lat, lon, label, tqi, status) {
    const colors = palette();
    const color   = trackColorForStatus(status, colors);
    const opacity = trackOpacityForStatus(status);
    const key     = "track:" + trackId;

    // Cancel any pending auto-removal (track came back to life or changed status).
    if (_trackRemoveTimers[trackId]) {
      clearTimeout(_trackRemoveTimers[trackId]);
      delete _trackRemoveTimers[trackId];
    }

    const tooltipText = (label || trackId.slice(0, 8)) +
      (status && status !== "active" && status !== "confirmed" ? " [" + status + "]" : "");

    if (_markers[key]) {
      _markers[key].setLatLng([lat, lon]);
      _markers[key].setIcon(makeTrackIcon(color, tqi, opacity));
      _markers[key].setTooltipContent(tooltipText);
    } else {
      _markers[key] = L.marker([lat, lon], {
        icon: makeTrackIcon(color, tqi, opacity),
      }).bindTooltip(tooltipText, { permanent: false }).addTo(_map);
      attachCopSelectionHandler(_markers[key], "track", trackId);
    }

    if (_highlightedCopItem && _highlightedCopItem.kind === "track" && _highlightedCopItem.id === trackId) {
      highlightCopItem("track", trackId);
    }

    // Schedule auto-removal for dropped tracks so they eventually clear even
    // if the backend keeps them in the list beyond their display window.
    if (status === "dropped") {
      _trackRemoveTimers[trackId] = setTimeout(function () {
        removeTrack(trackId);
      }, DROPPED_TRACK_LINGER_MS);
    }
  }

  function setTrackVelocityVector(trackId, lat, lon, velLat, velLon, status) {
    const colors  = palette();
    const color   = trackColorForStatus(status, colors);
    const opacity = status === "dropped" ? 0.0 : 0.72;
    const endLat  = lat + velLat * 3;
    const endLon  = lon + velLon * 3;
    if (_vectors[trackId]) {
      _vectors[trackId].setLatLngs([[lat, lon], [endLat, endLon]]);
      _vectors[trackId].setStyle({ color, opacity });
    } else {
      _vectors[trackId] = L.polyline([[lat, lon], [endLat, endLon]], {
        color, weight: 1.5, opacity, dashArray: "5,4",
      }).addTo(_map);
    }
  }

  function removeTrack(trackId) {
    if (_trackRemoveTimers[trackId]) {
      clearTimeout(_trackRemoveTimers[trackId]);
      delete _trackRemoveTimers[trackId];
    }
    if (_highlightedCopItem && _highlightedCopItem.kind === "track" && _highlightedCopItem.id === trackId) {
      clearCopHighlight();
    }
    const key = "track:" + trackId;
    if (_markers[key]) { _markers[key].remove(); delete _markers[key]; }
    if (_vectors[trackId]) { _vectors[trackId].remove(); delete _vectors[trackId]; }
    if (_ellipses[trackId]) { _ellipses[trackId].remove(); delete _ellipses[trackId]; }
  }

  function highlightCopItem(kind, id) {
    clearCopHighlight();
    const key = markerKeyForCopItem(kind, id);
    if (!key) return;
    const marker = _markers[key];
    if (!marker || !_map) return;
    _highlightedCopItem = { kind, id };
    if (typeof marker.bringToFront === "function") marker.bringToFront();
    if (typeof marker.setZIndexOffset === "function") marker.setZIndexOffset(1000);
    const element = marker.getElement ? marker.getElement() : null;
    if (element) element.classList.add("mmpr-map-icon-highlighted");
    if (marker.openTooltip) marker.openTooltip();
    const latlng = marker.getLatLng();
    _highlightRing = L.circleMarker(latlng, {
      radius: kind === "detection" ? 20 : 24,
      color: colorForCopItem(kind),
      weight: 2.5,
      fillOpacity: 0,
      opacity: 0.9,
      interactive: false,
      className: "mmpr-cop-highlight-ring",
    }).addTo(_map);
  }

  function clearCopHighlight() {
    if (_highlightedCopItem) {
      const key = markerKeyForCopItem(_highlightedCopItem.kind, _highlightedCopItem.id);
      const marker = key ? _markers[key] : null;
      if (marker) {
        if (typeof marker.setZIndexOffset === "function") marker.setZIndexOffset(0);
        const element = marker.getElement ? marker.getElement() : null;
        if (element) element.classList.remove("mmpr-map-icon-highlighted");
        if (marker.closeTooltip) marker.closeTooltip();
      }
    }
    if (_highlightRing) { _highlightRing.remove(); _highlightRing = null; }
    _highlightedCopItem = null;
  }

  function highlightTrack(trackId) { highlightCopItem("track", trackId); }
  function clearTrackHighlight() { clearCopHighlight(); }

  // ── Zone polygons ─────────────────────────────────────────────
  function setZone(zoneId, latlngs, label) {
    const colors = palette();
    if (_zones[zoneId]) {
      _zones[zoneId].setLatLngs(latlngs);
    } else {
      _zones[zoneId] = L.polygon(latlngs, {
        color: colors.warn, weight: 1.5, fillColor: colors.warn, fillOpacity: 0.16,
      }).bindTooltip(label || zoneId).addTo(_map);
    }
  }

  function removeZone(zoneId) {
    if (_zones[zoneId]) { _zones[zoneId].remove(); delete _zones[zoneId]; }
  }

  // ── GDOP circles ─────────────────────────────────────────────
  function setGdopCircle(key, lat, lon, radiusM) {
    const colors = palette();
    if (_gdop[key]) {
      _gdop[key].setLatLng([lat, lon]).setRadius(radiusM);
    } else {
      _gdop[key] = L.circle([lat, lon], {
        radius: radiusM, color: colors.node, fillColor: colors.node, fillOpacity: 0.12, weight: 1,
      }).addTo(_map);
    }
  }

  function removeGdopCircle(key) {
    if (_gdop[key]) { _gdop[key].remove(); delete _gdop[key]; }
  }

  // ── Pan/zoom ──────────────────────────────────────────────────
  function panTo(lat, lon) {
    if (_map) _map.panTo([lat, lon]);
  }

  // ── Heatmap (Leaflet.heat plugin) ────────────────────────────
  let _heatLayer = null;

  function initHeatmap(lat, lon, zoom) {
    init(lat, lon, zoom);
  }

  function setHeatmapPoints(points, maxIntensity) {
    if (!_map) return;
    if (!L.heatLayer) return; // plugin not loaded
    if (_heatLayer) {
      _heatLayer.setLatLngs(points);
      if (maxIntensity) _heatLayer.setOptions({ max: maxIntensity });
    } else {
      _heatLayer = L.heatLayer(points, {
        radius: 22,
        blur: 18,
        max: maxIntensity || 10,
        minOpacity: 0.35,
      }).addTo(_map);
    }
  }

  function clearHeatmap() {
    if (_heatLayer) { _heatLayer.remove(); _heatLayer = null; }
  }

  function fitBoundsLatLons(points) {
    if (!_map || !points || points.length === 0) return;
    const ll = points.map(function (p) { return [p[0], p[1]]; });
    _map.fitBounds(ll, { padding: [40, 40], maxZoom: 18 });
  }

  function invalidateMapSize() {
    if (_map) _map.invalidateSize();
  }

  // ── Public API ────────────────────────────────────────────────
  globalThis.leafletInterop = {
    init,
    setNodeMarker, removeNodeMarker,
    addDetectionMarker, removeDetectionMarker,
    setTrackMarker, setTrackVelocityVector, removeTrack,
    highlightCopItem, clearCopHighlight, setCopSelectionCallback,
    highlightTrack, clearTrackHighlight,
    setZone, removeZone,
    setGdopCircle, removeGdopCircle,
    panTo,
    initHeatmap, setHeatmapPoints, clearHeatmap, fitBoundsLatLons,
    invalidateMapSize,
  };
})();
