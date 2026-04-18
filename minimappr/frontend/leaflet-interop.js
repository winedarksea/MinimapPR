// Minimal shim exposing Leaflet operations to WASM via globalThis.leafletInterop.
// Leaflet is loaded from CDN in index.html before this script runs.

(function () {
  "use strict";

  let _map = null;
  const _markers = {};   // key → L.Marker or L.CircleMarker
  const _vectors = {};   // track_id → L.Polyline (velocity)
  const _ellipses = {};  // track_id → L.Ellipse (covariance)
  const _zones = {};     // zone_id → L.Polygon
  const _gdop = {};      // key → L.Circle

  const COLORS = {
    node:      "#58a6ff",
    track:     "#3fb950",
    detection: "#f78166",
    zone:      "rgba(255,214,0,0.15)",
    gdop:      "rgba(88,166,255,0.12)",
  };

  const TILE_CACHE_NAME = "mmpr-osm-tiles-v1";
  const OSM_TEMPLATE = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

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

  // Intercepts tile load errors at the L.TileLayer level.
  // Normal (non-error) tiles take the fast path: Leaflet sets img.src directly
  // via the inherited createTile, avoiding any async overhead.  Only on error
  // do we check the persistent cache and then fall back to the generated
  // placeholder canvas tile so the map stays usable even when OSM is down.
  function createResilientOsmTileLayer() {
    const ResilientTileLayer = L.TileLayer.extend({
      createTile: function (coords, done) {
        // Call the standard Leaflet createTile so the img.src is set
        // immediately, giving tiles the fastest possible happy-path load.
        const tile = L.TileLayer.prototype.createTile.call(this, coords, done);
        const url = this.getTileUrl(coords);
        const tileSize = this.getTileSize();
        const width = (tileSize && tileSize.x) || 256;

        // Intercept tile errors: try persistent cache first, then render
        // a labelled placeholder canvas tile so the overlay layer (markers,
        // heatmap, tracks) is still visible on a non-blank background.
        tile.addEventListener("error", function onTileError() {
          tile.removeEventListener("error", onTileError);
          (async () => {
            try {
              const cache = await getTileCache();
              if (cache) {
                const cachedResp = await cache.match(requestFromUrl(url));
                if (cachedResp) {
                  const blobUrl = await responseToObjectUrl(cachedResp);
                  tile.onload = function () { URL.revokeObjectURL(blobUrl); };
                  tile.src = blobUrl;
                  return;
                }
              }
            } catch (_) {
              // Cache unavailable (non-secure context, quota, etc.) — fall through.
            }
            tile.src = makeGenericFallbackTileDataUrl(width);
          })();
        }, { once: true });

        // After a successful network load, opportunistically cache the tile
        // so it survives future rate-limit or offline periods.
        tile.addEventListener("load", function onTileLoad() {
          tile.removeEventListener("load", onTileLoad);
          (async () => {
            try {
              const cache = await getTileCache();
              if (!cache) return;
              // Skip if already cached to avoid redundant writes.
              const alreadyCached = await cache.match(requestFromUrl(url));
              if (alreadyCached) return;
              const resp = await fetch(requestFromUrl(url), {
                mode: "cors",
                credentials: "omit",
              });
              if (resp.ok) {
                await cache.put(requestFromUrl(url), resp);
              }
            } catch (_) {
              // Best-effort write; failure is non-fatal.
            }
          })();
        }, { once: true });

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
    // If a previous map exists but its container was removed (e.g. page nav
    // unmounted and remounted the #leaflet-map div), drop the old map and
    // rebuild against the fresh DOM node. Keeps soft-navigation safe.
    if (_map) {
      const prev = _map.getContainer ? _map.getContainer() : null;
      if (!prev || !document.body.contains(prev)) {
        try { _map.remove(); } catch (_) {}
        _map = null;
        for (const k in _markers)  delete _markers[k];
        for (const k in _vectors)  delete _vectors[k];
        for (const k in _ellipses) delete _ellipses[k];
        for (const k in _zones)    delete _zones[k];
        for (const k in _gdop)     delete _gdop[k];
      } else {
        return;
      }
    }
    _map = L.map("leaflet-map", {
      center: [lat, lon],
      zoom: zoom ?? 17,
      zoomControl: true,
    });
    // updateWhenIdle defers tile fetches until panning stops, reducing OSM load.
    // maxZoom 18 = OSM's nominal limit (19+ just upscales the same tiles).
    createResilientOsmTileLayer().addTo(_map);
    // Deferred invalidateSize handles cases where the map container's flex
    // dimensions settle after the initial paint (e.g. WASM hydration timing).
    setTimeout(function () { if (_map) _map.invalidateSize(); }, 100);
  }

  // ── Node markers ──────────────────────────────────────────────
  function setNodeMarker(nodeId, lat, lon, healthClass) {
    const color = healthClass === "online" ? COLORS.node
                : healthClass === "degraded" ? "#d29922"
                : "#f85149";
    const key = "node:" + nodeId;
    if (_markers[key]) {
      _markers[key].setLatLng([lat, lon]);
      _markers[key].setStyle({ color });
    } else {
      _markers[key] = L.circleMarker([lat, lon], {
        radius: 8, color, fillColor: color, fillOpacity: 0.9, weight: 2,
      }).bindTooltip(nodeId, { permanent: false }).addTo(_map);
    }
  }

  function removeNodeMarker(nodeId) {
    const key = "node:" + nodeId;
    if (_markers[key]) { _markers[key].remove(); delete _markers[key]; }
  }

  // ── Detection markers ─────────────────────────────────────────
  function addDetectionMarker(eventId, lat, lon, label) {
    const key = "det:" + eventId;
    if (_markers[key]) return;
    _markers[key] = L.circleMarker([lat, lon], {
      radius: 5, color: COLORS.detection, fillColor: COLORS.detection, fillOpacity: 0.7, weight: 1,
    }).bindTooltip(label || "detection", { permanent: false }).addTo(_map);
    // Auto-remove after 30s
    setTimeout(() => {
      if (_markers[key]) { _markers[key].remove(); delete _markers[key]; }
    }, 30_000);
  }

  // ── Track markers + velocity vectors ─────────────────────────
  function setTrackMarker(trackId, lat, lon, label, tqi) {
    const key = "track:" + trackId;
    const r = 6 + Math.round((tqi ?? 0) * 4);
    if (_markers[key]) {
      _markers[key].setLatLng([lat, lon]);
    } else {
      _markers[key] = L.circleMarker([lat, lon], {
        radius: r, color: COLORS.track, fillColor: COLORS.track, fillOpacity: 0.75, weight: 2,
      }).bindTooltip(label || trackId.slice(0, 8), { permanent: false }).addTo(_map);
    }
  }

  function setTrackVelocityVector(trackId, lat, lon, velLat, velLon) {
    const endLat = lat + velLat * 3;
    const endLon = lon + velLon * 3;
    if (_vectors[trackId]) {
      _vectors[trackId].setLatLngs([[lat, lon], [endLat, endLon]]);
    } else {
      _vectors[trackId] = L.polyline([[lat, lon], [endLat, endLon]], {
        color: COLORS.track, weight: 1.5, opacity: 0.7, dashArray: "4,4",
      }).addTo(_map);
    }
  }

  function removeTrack(trackId) {
    const key = "track:" + trackId;
    if (_markers[key]) { _markers[key].remove(); delete _markers[key]; }
    if (_vectors[trackId]) { _vectors[trackId].remove(); delete _vectors[trackId]; }
    if (_ellipses[trackId]) { _ellipses[trackId].remove(); delete _ellipses[trackId]; }
  }

  // ── Zone polygons ─────────────────────────────────────────────
  function setZone(zoneId, latlngs, label) {
    if (_zones[zoneId]) {
      _zones[zoneId].setLatLngs(latlngs);
    } else {
      _zones[zoneId] = L.polygon(latlngs, {
        color: "#d29922", weight: 1.5, fillColor: COLORS.zone, fillOpacity: 1,
      }).bindTooltip(label || zoneId).addTo(_map);
    }
  }

  function removeZone(zoneId) {
    if (_zones[zoneId]) { _zones[zoneId].remove(); delete _zones[zoneId]; }
  }

  // ── GDOP circles ─────────────────────────────────────────────
  function setGdopCircle(key, lat, lon, radiusM) {
    if (_gdop[key]) {
      _gdop[key].setLatLng([lat, lon]).setRadius(radiusM);
    } else {
      _gdop[key] = L.circle([lat, lon], {
        radius: radiusM, color: COLORS.gdop, fillColor: COLORS.gdop, fillOpacity: 1, weight: 0,
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

  // ── Public API ────────────────────────────────────────────────
  globalThis.leafletInterop = {
    init,
    setNodeMarker, removeNodeMarker,
    addDetectionMarker,
    setTrackMarker, setTrackVelocityVector, removeTrack,
    setZone, removeZone,
    setGdopCircle, removeGdopCircle,
    panTo,
    initHeatmap, setHeatmapPoints, clearHeatmap, fitBoundsLatLons,
  };
})();
