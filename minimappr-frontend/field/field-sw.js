/**
 * Field PWA service worker: offline app shell + offline model.
 *
 * Caching strategy adapted from the user's metal-detecting PWA ("On Device
 * Audio App", web/app-extra/sw.js): network-first for navigations and
 * stable-named code so a new deploy reaches returning users, cache-first for
 * immutable/large assets. One change: that SW only cached hashed assets on
 * first fetch, so assets loaded before the SW took control were missing
 * offline until a second visit. Here `install` parses index.html for the
 * trunk-hashed wasm/js/css it references and precaches them together with the
 * stable-named runtime files, so the app opens offline after ONE online load.
 *
 * The BirdNET model and labels live in their own cache that shell-version
 * bumps never purge (72 MB; plan: models are downloaded before a trip).
 * Relay traffic (/v1/) is never cached.
 *
 * Bump SHELL_CACHE_NAME when this file's caching behaviour changes.
 */
const SHELL_CACHE_NAME = "mmp-field-shell-v1";
const MODEL_CACHE_NAME = "mmp-field-models-v1";
const SHELL_ENTRY_PATH = "/index.html";
const MODEL_PATH_PREFIX = "/models/";
const RELAY_PATH_PREFIX = "/v1/";

// Stable-named files the page loads at runtime (not referenced by index.html
// attributes trunk rewrites, or fetched lazily by onnxruntime-web).
const STABLE_RUNTIME_PATHS = [
  "/",
  SHELL_ENTRY_PATH,
  "/field.webmanifest",
  "/favicon.svg",
  "/field-mic-worklet.js",
  "/field-capture-interop.js",
  "/field-birdnet-interop.js",
  "/ort.wasm.min.js",
  "/ort-wasm-simd-threaded.mjs",
  "/ort-wasm-simd-threaded.wasm",
];

// Large or content-addressed: cache-first.
function isCacheFirstPath(pathname) {
  return (
    pathname.startsWith(MODEL_PATH_PREFIX) ||
    pathname.startsWith("/ort-wasm") ||
    /-[0-9a-f]{16}(_bg)?\.(wasm|js|css)$/.test(pathname)
  );
}

async function hashedAssetPathsFromShell(shellResponse) {
  const html = await shellResponse.clone().text();
  const paths = new Set();
  for (const match of html.matchAll(/(?:href|src)="(\/[^"#?]+)"/g)) paths.add(match[1]);
  return [...paths];
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL_CACHE_NAME);
      const shellResponse = await fetch(SHELL_ENTRY_PATH, { cache: "no-cache" });
      if (shellResponse.ok) {
        await cache.put(SHELL_ENTRY_PATH, shellResponse.clone());
        const referenced = await hashedAssetPathsFromShell(shellResponse);
        const precachePaths = [...new Set([...STABLE_RUNTIME_PATHS, ...referenced])];
        // Tolerate individual failures so install always completes.
        await Promise.allSettled(precachePaths.map((path) => cache.add(path)));
      }
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keep = new Set([SHELL_CACHE_NAME, MODEL_CACHE_NAME]);
      const names = await caches.keys();
      await Promise.all(names.filter((name) => !keep.has(name)).map((name) => caches.delete(name)));
      await self.clients.claim();
    })(),
  );
});

function cacheNameFor(pathname) {
  return pathname.startsWith(MODEL_PATH_PREFIX) ? MODEL_CACHE_NAME : SHELL_CACHE_NAME;
}

async function networkFirst(request, cacheKey, cacheName) {
  try {
    const response = await fetch(request);
    if (response.ok && response.type === "basic") {
      const cache = await caches.open(cacheName);
      await cache.put(cacheKey, response.clone());
    }
    return response;
  } catch (networkError) {
    const cached = await caches.match(cacheKey);
    return cached || Response.error();
  }
}

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok && response.type === "basic") {
    const cache = await caches.open(cacheName);
    await cache.put(request, response.clone());
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith(RELAY_PATH_PREFIX)) return;
  if (url.pathname === "/field-sw.js") return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request, SHELL_ENTRY_PATH, SHELL_CACHE_NAME));
    return;
  }
  const cacheName = cacheNameFor(url.pathname);
  if (isCacheFirstPath(url.pathname)) {
    event.respondWith(cacheFirst(request, cacheName));
    return;
  }
  event.respondWith(networkFirst(request, request, cacheName));
});
