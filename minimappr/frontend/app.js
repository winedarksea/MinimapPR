const state = {
  nodes: [],
  tracks: [],
  detections: [],
};

const statusEl = document.getElementById("status");
const nodeCountEl = document.getElementById("nodeCount");
const trackCountEl = document.getElementById("trackCount");
const detCountEl = document.getElementById("detCount");
const detectionsTableEl = document.getElementById("detectionsTable");
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");

function fmtNum(value, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function nsToTime(ns) {
  if (!ns) return "-";
  return new Date(ns / 1e6).toLocaleTimeString();
}

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} failed`);
  return await res.json();
}

function computeBounds() {
  const points = [];
  state.nodes.forEach((n) => points.push(n.position_m));
  state.tracks.forEach((t) => points.push(t.position_m));
  state.detections.slice(0, 50).forEach((d) => points.push(d.position_m));

  if (!points.length) {
    return { minX: -1, maxX: 10, minY: -1, maxY: 10 };
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  const padX = Math.max(1, (maxX - minX) * 0.2);
  const padY = Math.max(1, (maxY - minY) * 0.2);
  return {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY,
  };
}

function worldToCanvas(x, y, bounds) {
  const margin = 40;
  const w = canvas.width - margin * 2;
  const h = canvas.height - margin * 2;
  const nx = (x - bounds.minX) / Math.max(1e-6, bounds.maxX - bounds.minX);
  const ny = (y - bounds.minY) / Math.max(1e-6, bounds.maxY - bounds.minY);
  return {
    x: margin + nx * w,
    y: canvas.height - (margin + ny * h),
  };
}

function drawMap() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const bounds = computeBounds();

  ctx.fillStyle = "#f8fbf9";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#d5ded9";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const x = (canvas.width / 10) * i;
    const y = (canvas.height / 10) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  for (const node of state.nodes) {
    const p = worldToCanvas(node.position_m[0], node.position_m[1], bounds);
    ctx.fillStyle = "#1462a6";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#133245";
    ctx.font = "12px IBM Plex Sans, sans-serif";
    ctx.fillText(node.id, p.x + 9, p.y - 8);
  }

  for (const track of state.tracks) {
    if (track.status === "dropped") continue;
    const p = worldToCanvas(track.position_m[0], track.position_m[1], bounds);
    const isActive = track.status === "confirmed" || track.status === "tentative";
    ctx.globalAlpha = isActive ? 1.0 : 0.45;
    ctx.fillStyle = track.status === "tentative" ? "#f4a261" : "#e76f51";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#411a0f";
    ctx.font = "11px IBM Plex Sans, sans-serif";
    ctx.fillText(`${track.id} ${track.label} [${track.status}]`, p.x + 8, p.y + 4);
    ctx.globalAlpha = 1.0;
  }

  const recent = state.detections.slice(0, 30);
  recent.forEach((detection, idx) => {
    const p = worldToCanvas(detection.position_m[0], detection.position_m[1], bounds);
    const alpha = Math.max(0.2, 1.0 - idx / 30);
    ctx.fillStyle = `rgba(157, 2, 8, ${alpha})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = "#2a4351";
  ctx.font = "11px IBM Plex Sans, sans-serif";
  ctx.fillText(
    `X: ${fmtNum(bounds.minX)}..${fmtNum(bounds.maxX)} m   Y: ${fmtNum(bounds.minY)}..${fmtNum(bounds.maxY)} m`,
    16,
    canvas.height - 12,
  );
}

function renderTable() {
  const rows = state.detections.slice(0, 15).map((d) => {
    return `<tr>
      <td>${nsToTime(d.timestamp_ns)}</td>
      <td>${d.label}</td>
      <td>${fmtNum(d.label_confidence, 3)}</td>
      <td>${d.track_id || "-"}</td>
      <td>${fmtNum(d.position_m[0])}, ${fmtNum(d.position_m[1])}, ${fmtNum(d.position_m[2])}</td>
    </tr>`;
  });
  detectionsTableEl.innerHTML = rows.join("");
}

function render() {
  nodeCountEl.textContent = String(state.nodes.length);
  trackCountEl.textContent = String(state.tracks.filter((t) => t.status === "confirmed" || t.status === "tentative").length);
  detCountEl.textContent = String(state.detections.length);
  drawMap();
  renderTable();
}

async function refreshSnapshot() {
  const [nodes, detections, tracks] = await Promise.all([
    fetchJSON("/api/v1/nodes"),
    fetchJSON("/api/v1/detections?limit=100"),
    fetchJSON("/api/v1/tracks?limit=200"),
  ]);

  state.nodes = nodes;
  state.detections = detections;
  state.tracks = tracks;
  render();
}

function connectLive() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/live`);

  ws.addEventListener("open", () => {
    statusEl.textContent = "Live feed connected";
  });

  ws.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "detection") {
        if (payload.detection) {
          state.detections.unshift(payload.detection);
          state.detections = state.detections.slice(0, 200);
        }
        if (payload.track) {
          const idx = state.tracks.findIndex((item) => item.id === payload.track.id);
          if (idx >= 0) {
            state.tracks[idx] = payload.track;
          } else {
            state.tracks.unshift(payload.track);
          }
          state.tracks = state.tracks.slice(0, 300);
        }
        render();
      }
    } catch {
      // Ignore malformed websocket payloads.
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

(async function init() {
  try {
    await refreshSnapshot();
    setInterval(() => {
      refreshSnapshot().catch(() => {});
    }, 2500);
    connectLive();
    statusEl.textContent = "Ready";
  } catch {
    statusEl.textContent = "Backend unavailable";
  }
})();
