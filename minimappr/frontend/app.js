const state = {
  config: null,
  nodes: [],
  tracks: [],
  detections: [],
  zones: [],
  copStatus: null,
  selectedTrackId: null,
  trackSort: { key: "tqi", dir: "desc" },
  mapReady: false,
  autoFit: true,
};

const statusEl = document.getElementById("status");
const siteOriginEl = document.getElementById("siteOrigin");
const activeNodesEl = document.getElementById("activeNodes");
const degradedNodesEl = document.getElementById("degradedNodes");
const activeTracksEl = document.getElementById("activeTracks");
const recentAlertsEl = document.getElementById("recentAlerts");
const tracksTableEl = document.getElementById("tracksTable");
const detectionFeedEl = document.getElementById("detectionFeed");
const audioPlayerEl = document.getElementById("audioPlayer");

const filterCategoryEl = document.getElementById("filterCategory");
const filterStatusEl = document.getElementById("filterStatus");
const filterConfidenceEl = document.getElementById("filterConfidence");
const filterConfidenceValueEl = document.getElementById("filterConfidenceValue");
const filterZonesEl = document.getElementById("filterZones");

const map = L.map("map", { zoomControl: true, preferCanvas: true, minZoom: 2, maxZoom: 20 });
const baseLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
});
baseLayer.addTo(map);

const layers = {
  gdop: L.layerGroup().addTo(map),
  detections: L.layerGroup().addTo(map),
  vectors: L.layerGroup().addTo(map),
  ellipses: L.layerGroup().addTo(map),
  zones: L.layerGroup().addTo(map),
  tracks: L.layerGroup().addTo(map),
  nodes: L.layerGroup().addTo(map),
};

const trackMarkerById = new Map();
let ws = null;

function fmtNum(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function nsToTime(ns) {
  if (!ns) return "-";
  return new Date(ns / 1e6).toLocaleTimeString();
}

function nsToAge(ns) {
  if (!ns) return "-";
  const ageSec = Math.max(0, (Date.now() - ns / 1e6) / 1000);
  if (ageSec < 60) return `${Math.round(ageSec)}s`;
  if (ageSec < 3600) return `${Math.round(ageSec / 60)}m`;
  return `${Math.round(ageSec / 3600)}h`;
}

function parseNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function offsetLatLon(lat, lon, eastM, northM) {
  const metersPerDegLat = 111132.92;
  const metersPerDegLon = 111412.84 * Math.cos((lat * Math.PI) / 180.0);
  const outLat = lat + northM / Math.max(metersPerDegLat, 1e-9);
  const outLon = lon + eastM / Math.max(metersPerDegLon, 1e-9);
  return [outLat, outLon];
}

function localToGeo(positionM) {
  if (!state.config?.site_origin || !positionM) return null;
  const origin = state.config.site_origin;
  const lat = parseNumber(origin.lat, 0);
  const lon = parseNumber(origin.lon, 0);
  const out = offsetLatLon(lat, lon, parseNumber(positionM[0]), parseNumber(positionM[1]));
  return { lat: out[0], lon: out[1], alt_m: parseNumber(origin.alt_m) + parseNumber(positionM[2]) };
}

function toLatLng(obj) {
  const geo = obj?.position_geo || localToGeo(obj?.position_m);
  if (!geo) return null;
  return [parseNumber(geo.lat), parseNumber(geo.lon)];
}

function gdopColor(gdop) {
  const v = Math.max(0, Math.min(10, Number(gdop || 0)));
  if (v <= 2) return "#2f9e6a";
  if (v <= 5) return "#cc9c36";
  return "#b8443e";
}

function statusColor(status) {
  if (status === "tentative") return "var(--status-tentative)";
  if (status === "coasting") return "var(--status-coasting)";
  return "var(--status-confirmed)";
}

function categorySymbol(category) {
  if (category === "wildlife") return "W";
  if (category === "human") return "H";
  if (category === "vehicle") return "V";
  if (category === "security") return "S";
  return "?";
}

function normCategory(category) {
  const value = String(category || "unknown").trim().toLowerCase();
  if (["wildlife", "human", "vehicle", "security"].includes(value)) return value;
  return "unknown";
}

function normIff(iff) {
  const value = String(iff || "unknown").trim().toLowerCase();
  if (["friendly", "hostile", "unknown"].includes(value)) return value;
  return "unknown";
}

function computeEllipse2D(covariance) {
  if (!Array.isArray(covariance) || covariance.length < 2) return null;
  const cxx = parseNumber(covariance[0]?.[0], 0);
  const cxy = parseNumber(covariance[0]?.[1], 0);
  const cyy = parseNumber(covariance[1]?.[1], 0);
  if (cxx <= 0 || cyy <= 0) return null;

  const trace = cxx + cyy;
  const det = cxx * cyy - cxy * cxy;
  const term = Math.sqrt(Math.max(0, trace * trace * 0.25 - det));
  const l1 = Math.max(1e-6, trace * 0.5 + term);
  const l2 = Math.max(1e-6, trace * 0.5 - term);
  const major = 2.0 * Math.sqrt(Math.max(l1, l2));
  const minor = 2.0 * Math.sqrt(Math.min(l1, l2));
  const headingRad = 0.5 * Math.atan2(2.0 * cxy, cxx - cyy);
  return { major, minor, headingRad };
}

function ellipsePolygonLatLng(lat, lon, major, minor, headingRad, steps = 24) {
  const points = [];
  const cosH = Math.cos(headingRad);
  const sinH = Math.sin(headingRad);
  for (let i = 0; i < steps; i += 1) {
    const t = (i / steps) * Math.PI * 2;
    const ex = Math.cos(t) * major;
    const ny = Math.sin(t) * minor;
    const east = ex * cosH - ny * sinH;
    const north = ex * sinH + ny * cosH;
    const shifted = offsetLatLon(lat, lon, east, north);
    points.push([shifted[0], shifted[1]]);
  }
  return points;
}

function getSortedTracks() {
  const tracks = state.tracks
    .filter((track) => track.status !== "dropped")
    .slice();

  const { key, dir } = state.trackSort;
  const direction = dir === "asc" ? 1 : -1;
  tracks.sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    const na = Number(av);
    const nb = Number(bv);
    if (Number.isFinite(na) && Number.isFinite(nb)) return (na - nb) * direction;
    return String(av || "").localeCompare(String(bv || "")) * direction;
  });

  return tracks;
}

function renderSystemStatus() {
  if (state.copStatus) {
    activeNodesEl.textContent = String(state.copStatus.active_nodes ?? 0);
    degradedNodesEl.textContent = String(state.copStatus.degraded_nodes ?? 0);
    activeTracksEl.textContent = String(state.copStatus.active_tracks ?? 0);
    recentAlertsEl.textContent = String(state.copStatus.recent_alert_count ?? 0);
    return;
  }

  const activeNodes = state.nodes.filter((node) => node.health_status === "online").length;
  const degradedNodes = state.nodes.filter((node) => node.health_status === "degraded").length;
  const activeTracks = state.tracks.filter((track) => ["confirmed", "tentative", "coasting"].includes(track.status)).length;
  activeNodesEl.textContent = String(activeNodes);
  degradedNodesEl.textContent = String(degradedNodes);
  activeTracksEl.textContent = String(activeTracks);
  recentAlertsEl.textContent = "0";
}

function renderTrackTable() {
  const sortedTracks = getSortedTracks();
  tracksTableEl.innerHTML = sortedTracks
    .map((track) => {
      const selected = track.id === state.selectedTrackId ? "is-selected" : "";
      return `<tr class="${selected}" data-track-id="${track.id}">
        <td>${track.id}</td>
        <td>${track.label}</td>
        <td>${track.label_category || "unknown"}</td>
        <td>${track.status}</td>
        <td>${fmtNum(track.tqi, 3)}</td>
        <td>${nsToAge(track.last_seen_ns)}</td>
      </tr>`;
    })
    .join("");

  tracksTableEl.querySelectorAll("tr[data-track-id]").forEach((row) => {
    row.addEventListener("click", () => {
      const trackId = row.dataset.trackId;
      if (!trackId) return;
      focusTrack(trackId);
    });
  });
}

function playDetectionSnippet(detectionId) {
  audioPlayerEl.src = `/api/v1/detections/${encodeURIComponent(detectionId)}/audio`;
  audioPlayerEl.play().catch(() => {});
}

function renderDetectionFeed() {
  const nowNs = Date.now() * 1e6;
  const recent = state.detections
    .slice()
    .sort((a, b) => Number(b.timestamp_ns || 0) - Number(a.timestamp_ns || 0))
    .slice(0, 50);

  detectionFeedEl.innerHTML = recent
    .map((detection) => {
      const ageSec = Math.max(0, (nowNs - Number(detection.timestamp_ns || 0)) / 1e9);
      const hasTrack = Boolean(detection.track_id);
      const hasSnippet = Boolean(detection.snippet_path);
      return `<li data-detection-id="${detection.id}">
        <div><strong>${detection.label}</strong> <span>(${fmtNum(detection.label_confidence, 2)})</span></div>
        <div class="feed-meta">${nsToTime(detection.timestamp_ns)} | age ${Math.round(ageSec)}s | ${detection.label_category || "unknown"}</div>
        <div class="feed-meta">GDOP ${fmtNum(detection.gdop, 2)} | track ${detection.track_id || "-"}</div>
        <div class="feed-actions">
          ${hasTrack ? `<button type="button" data-focus-track="${detection.track_id}">Focus Track</button>` : ""}
          ${hasSnippet ? `<button type="button" data-play-audio="${detection.id}">Play Audio</button>` : ""}
        </div>
      </li>`;
    })
    .join("");

  detectionFeedEl.querySelectorAll("button[data-focus-track]").forEach((button) => {
    button.addEventListener("click", () => {
      const trackId = button.dataset.focusTrack;
      if (trackId) focusTrack(trackId);
    });
  });

  detectionFeedEl.querySelectorAll("button[data-play-audio]").forEach((button) => {
    button.addEventListener("click", () => {
      const detectionId = button.dataset.playAudio;
      if (detectionId) playDetectionSnippet(detectionId);
    });
  });
}

function renderMap() {
  layers.nodes.clearLayers();
  layers.tracks.clearLayers();
  layers.detections.clearLayers();
  layers.gdop.clearLayers();
  layers.vectors.clearLayers();
  layers.ellipses.clearLayers();
  layers.zones.clearLayers();
  trackMarkerById.clear();

  const bounds = [];
  const nowNs = Date.now() * 1e6;

  for (const zone of state.zones) {
    const polygon = Array.isArray(zone.polygon_geo) ? zone.polygon_geo : [];
    if (polygon.length < 3) continue;
    const latLngs = polygon.map((point) => [parseNumber(point[0]), parseNumber(point[1])]);
    latLngs.forEach((latLng) => bounds.push(latLng));

    const zoneType = String(zone.zone_type || "").toLowerCase();
    const style =
      zoneType === "exclusion_zone"
        ? { color: "#8d5e2b", fillColor: "#d8b17a", fillOpacity: 0.16 }
        : zoneType === "coverage_zone"
          ? { color: "#316f8d", fillColor: "#8ebdd7", fillOpacity: 0.12 }
          : zoneType === "alert_zone"
            ? { color: "#8d2b2b", fillColor: "#d78e8e", fillOpacity: 0.13 }
            : { color: "#4b6d6c", fillColor: "#a2c2c0", fillOpacity: 0.1 };

    L.polygon(latLngs, {
      color: style.color,
      fillColor: style.fillColor,
      fillOpacity: style.fillOpacity,
      weight: 2,
      opacity: 0.8,
    })
      .bindTooltip(`${zone.name} (${zone.zone_type})`)
      .addTo(layers.zones);
  }

  for (const node of state.nodes) {
    const latLng = toLatLng(node);
    if (!latLng) continue;
    bounds.push(latLng);

    const health = String(node.health_status || "offline");
    const color =
      health === "online"
        ? "#2f8f4f"
        : health === "degraded"
          ? "#c78d2b"
          : "#bd3b35";

    const marker = L.circleMarker(latLng, {
      radius: 6,
      color,
      fillColor: color,
      fillOpacity: 0.9,
      weight: 2,
    });
    marker.bindTooltip(`${node.id} (${health})`, { direction: "top", opacity: 0.95 });
    marker.addTo(layers.nodes);
  }

  const detections = state.detections
    .slice()
    .sort((a, b) => Number(b.timestamp_ns || 0) - Number(a.timestamp_ns || 0));

  for (const detection of detections.slice(0, 120)) {
    const latLng = toLatLng(detection);
    if (!latLng) continue;
    bounds.push(latLng);

    const ageSec = Math.max(0, (nowNs - Number(detection.timestamp_ns || 0)) / 1e9);
    if (ageSec > 180) continue;

    const fade = Math.max(0.15, 1.0 - ageSec / 60.0);
    const color = gdopColor(detection.gdop);

    L.circleMarker(latLng, {
      radius: 3,
      color,
      fillColor: color,
      fillOpacity: fade,
      opacity: fade,
      weight: 1,
    }).addTo(layers.detections);

    const gdopRadiusM = Math.max(6, Math.min(50, (Number(detection.gdop || 0) + 1) * 3.5));
    L.circle(latLng, {
      radius: gdopRadiusM,
      color,
      fillColor: color,
      fillOpacity: 0.06,
      opacity: 0.35,
      weight: 1,
    })
      .bindTooltip(`GDOP ${fmtNum(detection.gdop, 2)}`)
      .addTo(layers.gdop);
  }

  for (const track of state.tracks) {
    if (track.status === "dropped") continue;
    const latLng = toLatLng(track);
    if (!latLng) continue;
    bounds.push(latLng);

    const category = normCategory(track.label_category);
    const iff = normIff(track.iff_category);
    const status = String(track.status || "confirmed").toLowerCase();
    const selected = state.selectedTrackId === track.id;

    const icon = L.divIcon({
      className: "track-marker-wrapper",
      html: `<div class="track-icon track-status-${status} track-iff-${iff} track-symbol-${category}"><span>${categorySymbol(category)}</span></div>`,
      iconSize: selected ? [28, 28] : [24, 24],
      iconAnchor: selected ? [14, 14] : [12, 12],
    });

    const marker = L.marker(latLng, { icon });
    marker.bindTooltip(
      `${track.id} | ${track.label} | ${track.status} | TQI ${fmtNum(track.tqi, 3)}`,
      { direction: "top", opacity: 0.96 },
    );
    marker.on("click", () => focusTrack(track.id));
    marker.addTo(layers.tracks);
    trackMarkerById.set(track.id, marker);

    const vx = parseNumber(track.velocity_mps?.[0]);
    const vy = parseNumber(track.velocity_mps?.[1]);
    const speed = Math.hypot(vx, vy);
    if (speed > 0.05) {
      const vectorLenM = Math.min(40, Math.max(6, speed * 3.0));
      const east = (vx / speed) * vectorLenM;
      const north = (vy / speed) * vectorLenM;
      const endpoint = offsetLatLon(latLng[0], latLng[1], east, north);
      L.polyline([latLng, endpoint], {
        color: statusColor(status),
        weight: selected ? 3 : 2,
        opacity: 0.75,
      }).addTo(layers.vectors);
    }

    const ellipse = computeEllipse2D(track.position_covariance_m2);
    if (ellipse) {
      const points = ellipsePolygonLatLng(latLng[0], latLng[1], ellipse.major, ellipse.minor, ellipse.headingRad, 32);
      L.polygon(points, {
        color: statusColor(status),
        weight: 1,
        fillColor: statusColor(status),
        fillOpacity: selected ? 0.18 : 0.1,
        opacity: 0.6,
      }).addTo(layers.ellipses);
    }
  }

  if (!state.mapReady) {
    if (state.config?.site_origin) {
      map.setView([state.config.site_origin.lat, state.config.site_origin.lon], 17);
      state.mapReady = true;
    }
  }

  if (bounds.length > 0 && state.autoFit) {
    const latLngBounds = L.latLngBounds(bounds);
    if (latLngBounds.isValid()) {
      map.fitBounds(latLngBounds.pad(0.15), { maxZoom: 18, animate: false });
    }
  }
}

function renderAll() {
  renderSystemStatus();
  renderTrackTable();
  renderDetectionFeed();
  renderMap();
}

function focusTrack(trackId) {
  state.selectedTrackId = trackId;
  state.autoFit = false;
  const marker = trackMarkerById.get(trackId);
  if (marker) {
    map.panTo(marker.getLatLng(), { animate: true, duration: 0.35 });
    marker.openTooltip();
  }
  renderAll();
}

function onTrackHeaderClick(event) {
  const th = event.target.closest("th[data-sort]");
  if (!th) return;
  const key = th.dataset.sort;
  if (!key) return;

  if (state.trackSort.key === key) {
    state.trackSort.dir = state.trackSort.dir === "asc" ? "desc" : "asc";
  } else {
    state.trackSort = { key, dir: key === "tqi" || key === "last_seen_ns" ? "desc" : "asc" };
  }
  renderTrackTable();
}

async function fetchJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} failed`);
  return await response.json();
}

function wsFilterPayload() {
  const category = String(filterCategoryEl.value || "").trim().toLowerCase();
  const status = String(filterStatusEl.value || "").trim().toLowerCase();
  const zoneIds = String(filterZonesEl.value || "")
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  const minConfidence = parseNumber(filterConfidenceEl.value, 0);

  return {
    action: "set_filter",
    filter: {
      label_categories: category ? [category] : [],
      track_status: status ? [status] : [],
      zone_ids: zoneIds,
      min_confidence: minConfidence,
    },
  };
}

function sendWsFilter() {
  filterConfidenceValueEl.textContent = fmtNum(filterConfidenceEl.value, 2);
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(wsFilterPayload()));
}

function mergeLiveDetection(payload) {
  const detection = payload?.detection;
  const track = payload?.track;
  if (detection && detection.id) {
    state.detections = [detection, ...state.detections.filter((item) => item.id !== detection.id)].slice(0, 300);
  }
  if (track && track.id) {
    const idx = state.tracks.findIndex((item) => item.id === track.id);
    if (idx >= 0) {
      state.tracks[idx] = track;
    } else {
      state.tracks.unshift(track);
    }
    state.tracks = state.tracks.slice(0, 500);
  }
}

function connectLive() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${window.location.host}/ws/live`);

  ws.addEventListener("open", () => {
    statusEl.textContent = "Live feed connected";
    sendWsFilter();
  });

  ws.addEventListener("message", (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }

    if (payload?.type === "subscription_ack") {
      statusEl.textContent = "Live feed connected (filtered)";
      return;
    }

    if (payload?.event_type === "detection" || payload?.type === "detection") {
      mergeLiveDetection(payload);
      renderAll();
    }
  });

  ws.addEventListener("close", () => {
    statusEl.textContent = "Live feed disconnected. Reconnecting...";
    setTimeout(connectLive, 1500);
  });

  ws.addEventListener("error", () => {
    statusEl.textContent = "Live feed error";
    ws.close();
  });
}

async function refreshSnapshot() {
  const [nodes, detections, tracks, zones, copStatus] = await Promise.all([
    fetchJSON("/api/v1/nodes"),
    fetchJSON("/api/v1/detections?limit=200"),
    fetchJSON("/api/v1/tracks?limit=400"),
    fetchJSON("/api/v1/zones"),
    fetchJSON("/api/v1/cop/status"),
  ]);

  state.nodes = nodes;
  state.detections = detections;
  state.tracks = tracks;
  state.zones = zones;
  state.copStatus = copStatus;
}

async function initConfig() {
  state.config = await fetchJSON("/api/v1/config");
  const origin = state.config?.site_origin;
  if (origin) {
    siteOriginEl.textContent = `Site origin: ${fmtNum(origin.lat, 5)}, ${fmtNum(origin.lon, 5)} (${state.config.coordinate_mode})`;
    map.setView([origin.lat, origin.lon], 17);
    state.mapReady = true;
  } else {
    map.setView([0, 0], 2);
  }
}

function bindUiEvents() {
  document.querySelectorAll("#tracksTable, .track-panel thead").forEach((el) => {
    el.addEventListener("click", onTrackHeaderClick);
  });

  [filterCategoryEl, filterStatusEl, filterZonesEl].forEach((el) => {
    el.addEventListener("change", sendWsFilter);
  });
  filterConfidenceEl.addEventListener("input", sendWsFilter);
}

(async function init() {
  try {
    bindUiEvents();
    await initConfig();
    await refreshSnapshot();
    renderAll();
    connectLive();

    setInterval(() => {
      refreshSnapshot()
        .then(renderAll)
        .catch(() => {});
    }, 3000);
  } catch {
    statusEl.textContent = "Backend unavailable";
  }
})();
