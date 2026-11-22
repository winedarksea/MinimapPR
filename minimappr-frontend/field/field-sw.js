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
 * The BirdNET model and labels live in their own cache (MODEL_CACHE_NAME)
 * that shell-version bumps never purge (72 MB; plan: models are downloaded
 * before a trip). The page owns that cache (field-birdnet-interop.js reads it
 * before fetching), so this worker does not intercept /models/ at all:
 * intercepting would write the 72 MB body twice and double the transient
 * quota demand. Relay traffic (/v1/) is never cached.
 *
 * Invalidation without a SW byte change (deploys do not touch this file):
 *   - trunk-hashed assets referenced by neither the fresh nor the previous
 *     index.html are pruned when a fresh shell lands (bounded growth, and a
 *     page still running the previous shell keeps its files);
 *   - the stable-named ORT companions (/ort-wasm-*, cache-first, ~15 MB) are
 *     dropped whenever the network-first ORT loader changes, so an
 *     onnxruntime-web upgrade never pairs a new loader with a stale wasm.
 *
 * Bump SHELL_CACHE_NAME when this file's caching behaviour changes.
 */
const SHELL_CACHE_NAME = "mmp-field-shell-v2";
const MODEL_CACHE_NAME = "mmp-field-models-v1";
const SHELL_ENTRY_PATH = "/index.html";
const MODEL_PATH_PREFIX = "/models/";
const RELAY_PATH_PREFIX = "/v1/";
const SERVICE_WORKER_PATH = "/field-sw.js";
const ORT_LOADER_PATH = "/ort.wasm.min.js";
const ORT_COMPANION_PATH_PREFIX = "/ort-wasm";
const TRUNK_HASHED_ASSET_PATTERN = /-[0-9a-f]{16}(_bg)?\.(wasm|js|css)$/;
// Lie-fi (one bar in the field): serve the cached shell rather than wait for
// a navigation fetch that can take minutes to fail. The fetch keeps running
// and refreshes the cache when it lands.
const NAVIGATION_NETWORK_TIMEOUT_MILLISECONDS = 4000;

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
  ORT_LOADER_PATH,
  "/ort-wasm-simd-threaded.mjs",
  "/ort-wasm-simd-threaded.wasm",
];

// Large or content-addressed: cache-first.
function isCacheFirstPath(pathname) {
  return pathname.startsWith(ORT_COMPANION_PATH_PREFIX) || TRUNK_HASHED_ASSET_PATTERN.test(pathname);
}

async function referencedPathsFromShell(shellResponse) {
  const html = await shellResponse.clone().text();
  const paths = new Set();
  for (const match of html.matchAll(/(?:href|src)="(\/[^"#?]+)"/g)) paths.add(match[1]);
  return paths;
}

async function deleteCachedEntriesWhere(cache, shouldDelete) {
  const requests = await cache.keys();
  await Promise.all(
    requests.filter((request) => shouldDelete(new URL(request.url).pathname)).map((request) => cache.delete(request)),
  );
}

async function pruneUnreferencedHashedAssets(cache, freshShellResponse) {
  const keep = await referencedPathsFromShell(freshShellResponse);
  const previousShell = await cache.match(SHELL_ENTRY_PATH);
  if (previousShell) for (const path of await referencedPathsFromShell(previousShell)) keep.add(path);
  await deleteCachedEntriesWhere(cache, (pathname) => TRUNK_HASHED_ASSET_PATTERN.test(pathname) && !keep.has(pathname));
}

async function invalidateOrtCompanionsIfLoaderChanged(cache, freshLoaderResponse) {
  const previousLoader = await cache.match(ORT_LOADER_PATH);
  if (!previousLoader) return;
  const [previousText, freshText] = await Promise.all([previousLoader.text(), freshLoaderResponse.clone().text()]);
  if (previousText === freshText) return;
  await deleteCachedEntriesWhere(cache, (pathname) => pathname.startsWith(ORT_COMPANION_PATH_PREFIX));
}

// Runs the invalidation hooks before storing, so the response reaches the
// page only once the cache is consistent with it.
async function storeInShellCache(cacheKey, pathname, response) {
  const cache = await caches.open(SHELL_CACHE_NAME);
  if (cacheKey === SHELL_ENTRY_PATH) await pruneUnreferencedHashedAssets(cache, response);
  if (pathname === ORT_LOADER_PATH) await invalidateOrtCompanionsIfLoaderChanged(cache, response);
  await cache.put(cacheKey, response.clone());
}

async function fetchAndStore(request, cacheKey, pathname) {
  const response = await fetch(request);
  if (response.ok && response.type === "basic") await storeInShellCache(cacheKey, pathname, response);
  return response;
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL_CACHE_NAME);
      const shellResponse = await fetch(SHELL_ENTRY_PATH, { cache: "no-cache" });
      if (shellResponse.ok) {
        await cache.put(SHELL_ENTRY_PATH, shellResponse.clone());
        const referenced = await referencedPathsFromShell(shellResponse);
        const precachePaths = [...new Set([...STABLE_RUNTIME_PATHS, ...referenced])];
        // Tolerate individual failures (incl. quota) so install always completes.
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

async function networkFirst(request, cacheKey, pathname) {
  try {
    return await fetchAndStore(request, cacheKey, pathname);
  } catch (networkError) {
    const cached = await caches.match(cacheKey, { cacheName: SHELL_CACHE_NAME });
    return cached || Response.error();
  }
}

async function navigationNetworkFirst(event, pathname) {
  const networkResponse = fetchAndStore(event.request, SHELL_ENTRY_PATH, pathname);
  // Keep the worker alive until a background refresh settles.
  event.waitUntil(networkResponse.catch(() => undefined));
  const timedOut = new Promise((resolve) => setTimeout(() => resolve(null), NAVIGATION_NETWORK_TIMEOUT_MILLISECONDS));
  const firstSettled = await Promise.race([networkResponse.catch(() => null), timedOut]);
  if (firstSettled) return firstSettled;
  const cached = await caches.match(SHELL_ENTRY_PATH, { cacheName: SHELL_CACHE_NAME });
  if (cached) return cached;
  return networkResponse.catch(() => Response.error());
}

async function cacheFirst(request, pathname) {
  const cached = await caches.match(request, { cacheName: SHELL_CACHE_NAME });
  if (cached) return cached;
  return fetchAndStore(request, request, pathname);
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith(RELAY_PATH_PREFIX)) return;
  if (url.pathname.startsWith(MODEL_PATH_PREFIX)) return;
  if (url.pathname === SERVICE_WORKER_PATH) return;

  if (request.mode === "navigate") {
    event.respondWith(navigationNetworkFirst(event, url.pathname));
    return;
  }
  if (isCacheFirstPath(url.pathname)) {
    event.respondWith(cacheFirst(request, url.pathname));
    return;
  }
  const cacheKey = url.pathname === SHELL_ENTRY_PATH ? SHELL_ENTRY_PATH : request;
  event.respondWith(networkFirst(request, cacheKey, url.pathname));
});
