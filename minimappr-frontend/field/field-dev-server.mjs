// Local-only dev/test server for the Field Mode PWA build (`dist-field/`).
//
// Why not `trunk serve`: the PWA must be same-origin with the relay API (the
// production plan serves the PWA and the relay from one Worker deploy, and the
// relay Worker sends no CORS headers), and the BirdNET model lives outside the
// build in the gitignored `data/models/birdnet_v3/`. This server:
//   - serves `dist-field/` with an SPA fallback to index.html,
//   - mounts the model directory at /models/birdnet_v3/,
//   - reverse-proxies /v1/* (HTTP and WebSocket upgrades) to a local
//     `wrangler dev` relay when FIELD_RELAY_ORIGIN is set.
// Never point it at a deployed relay; it is for 127.0.0.1 testing.
//
// Usage: node field/field-dev-server.mjs [--port 18090]
//   env FIELD_DIST_DIRECTORY, FIELD_MODEL_DIRECTORY, FIELD_RELAY_ORIGIN,
//   FIELD_CROSS_ORIGIN_ISOLATE=1 (COOP/COEP headers: lets ORT Web use threads)

import http from "node:http";
import net from "node:net";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_LISTEN_PORT = 18090;
const LISTEN_HOST = "127.0.0.1";
const MODEL_URL_PREFIX = "/models/birdnet_v3/";
const RELAY_URL_PREFIX = "/v1/";

const distDirectory = path.resolve(process.env.FIELD_DIST_DIRECTORY || path.join(frontendDirectory, "dist-field"));
const modelDirectory = path.resolve(
  process.env.FIELD_MODEL_DIRECTORY || path.join(frontendDirectory, "..", "data", "models", "birdnet_v3"),
);
const relayOrigin = process.env.FIELD_RELAY_ORIGIN ? new URL(process.env.FIELD_RELAY_ORIGIN) : null;
const crossOriginIsolationHeaders =
  process.env.FIELD_CROSS_ORIGIN_ISOLATE === "1"
    ? { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" }
    : {};

const CONTENT_TYPE_BY_EXTENSION = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".webmanifest": "application/manifest+json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".onnx": "application/octet-stream",
  ".csv": "text/csv; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".woff2": "font/woff2",
};

function parseListenPort() {
  const flagIndex = process.argv.indexOf("--port");
  if (flagIndex >= 0 && process.argv[flagIndex + 1]) return Number(process.argv[flagIndex + 1]);
  return Number(process.env.FIELD_DEV_SERVER_PORT || DEFAULT_LISTEN_PORT);
}

// Resolves a URL path inside `rootDirectory`, refusing traversal outside it.
function resolveInside(rootDirectory, urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath);
  } catch (_malformedEscape) {
    return null; // e.g. "/%E0": a URIError here would crash the server
  }
  const candidate = path.resolve(rootDirectory, "." + path.posix.normalize("/" + decoded));
  if (candidate !== rootDirectory && !candidate.startsWith(rootDirectory + path.sep)) return null;
  return candidate;
}

function sendFile(response, filePath, request) {
  fs.stat(filePath, (statError, stats) => {
    if (statError || !stats.isFile()) {
      response.writeHead(404, { "content-type": "text/plain" });
      response.end("not found");
      return;
    }
    const contentType = CONTENT_TYPE_BY_EXTENSION[path.extname(filePath)] || "application/octet-stream";
    // The service worker must always revalidate; everything else is the SW's job.
    const cacheControl = path.basename(filePath) === "field-sw.js" ? "no-cache, no-store" : "no-cache";
    response.writeHead(200, {
      ...crossOriginIsolationHeaders,
      "content-type": contentType,
      "content-length": stats.size,
      "cache-control": cacheControl,
    });
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    fs.createReadStream(filePath).pipe(response);
  });
}

function proxyRelayHttp(request, response) {
  const upstream = http.request(
    {
      host: relayOrigin.hostname,
      port: relayOrigin.port,
      method: request.method,
      path: request.url,
      headers: { ...request.headers, host: relayOrigin.host },
    },
    (upstreamResponse) => {
      response.writeHead(upstreamResponse.statusCode || 502, upstreamResponse.headers);
      upstreamResponse.pipe(response);
    },
  );
  upstream.on("error", (error) => {
    response.writeHead(502, { "content-type": "text/plain" });
    response.end(`relay proxy error: ${error.message}`);
  });
  request.pipe(upstream);
}

// WebSocket upgrade: replay the request head upstream, then splice sockets.
function proxyRelayUpgrade(request, clientSocket, head) {
  const upstreamSocket = net.connect(Number(relayOrigin.port), relayOrigin.hostname, () => {
    const headerLines = Object.entries({ ...request.headers, host: relayOrigin.host })
      .map(([name, value]) => `${name}: ${value}`)
      .join("\r\n");
    upstreamSocket.write(`${request.method} ${request.url} HTTP/1.1\r\n${headerLines}\r\n\r\n`);
    if (head && head.length) upstreamSocket.write(head);
    upstreamSocket.pipe(clientSocket);
    clientSocket.pipe(upstreamSocket);
  });
  const closeBoth = () => {
    upstreamSocket.destroy();
    clientSocket.destroy();
  };
  upstreamSocket.on("error", closeBoth);
  clientSocket.on("error", closeBoth);
}

const server = http.createServer((request, response) => {
  const urlPath = new URL(request.url, `http://${LISTEN_HOST}`).pathname;
  if (urlPath.startsWith(RELAY_URL_PREFIX)) {
    if (!relayOrigin) {
      response.writeHead(503, { "content-type": "text/plain" });
      response.end("FIELD_RELAY_ORIGIN not set");
      return;
    }
    proxyRelayHttp(request, response);
    return;
  }
  if (urlPath.startsWith(MODEL_URL_PREFIX)) {
    const modelPath = resolveInside(modelDirectory, urlPath.slice(MODEL_URL_PREFIX.length));
    if (!modelPath) {
      response.writeHead(400);
      response.end();
      return;
    }
    sendFile(response, modelPath, request);
    return;
  }
  const staticPath = resolveInside(distDirectory, urlPath === "/" ? "index.html" : urlPath);
  if (!staticPath) {
    response.writeHead(400);
    response.end();
    return;
  }
  // SPA fallback: extension-less paths are client routes.
  const isClientRoute = path.extname(staticPath) === "" && !fs.existsSync(staticPath);
  sendFile(response, isClientRoute ? path.join(distDirectory, "index.html") : staticPath, request);
});

server.on("upgrade", (request, socket, head) => {
  if (relayOrigin && request.url.startsWith(RELAY_URL_PREFIX)) {
    proxyRelayUpgrade(request, socket, head);
    return;
  }
  socket.destroy();
});

const listenPort = parseListenPort();
server.listen(listenPort, LISTEN_HOST, () => {
  console.log(`field dev server http://${LISTEN_HOST}:${listenPort} dist=${distDirectory} relay=${relayOrigin || "none"}`);
});
