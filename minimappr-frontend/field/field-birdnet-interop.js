/**
 * BirdNET+ V3.0 (FP16 pruned ONNX) in the browser through ONNX Runtime Web.
 *
 * Inference pattern ported from the user's metal-detecting PWA ("On Device
 * Audio App", webApp/.../web/inference/WebOnnxClassifier.kt): global `ort`
 * from a vendored script, WASM execution provider, single thread unless the
 * page is cross-origin isolated (threads need SharedArrayBuffer, which needs
 * COOP/COEP headers).
 *
 * V3 model I/O (docs/field_mode_progress/birdnet_v3.md): input `input`
 * float32 [batch, samples] at 32 kHz; outputs `predictions` [batch, 11725]
 * (sigmoid applied in-graph) and `embeddings` [batch, 1280]. INT8 was
 * rejected there (quantized conv stack broke the scores), hence FP16 pruned.
 *
 * Exposes window.mmpFieldBirdnet = { load, classify, isLoaded, release }.
 */
(function () {
  "use strict";

  const MODEL_INPUT_NAME = "input";
  const MODEL_PREDICTIONS_OUTPUT_NAME = "predictions";
  const MAXIMUM_INTRA_OP_THREADS_WHEN_ISOLATED = 4;
  // Same name as MODEL_CACHE_NAME in field-sw.js. The page stores the model
  // itself so it is offline-ready after the first load even though the
  // service worker does not control that first load's fetches.
  const MODEL_CACHE_NAME = "mmp-field-models-v1";
  const LABEL_COLUMN_INDEX = { index: 0, birdnetId: 1, scientificName: 2, commonName: 3, taxonGroup: 4 };

  let inferenceSession = null;
  let labelTable = [];

  // Minimal CSV field splitter (quotes + doubled quotes); only the first few
  // columns are kept, the file carries ~40 translated-name columns.
  function splitCsvLine(line, keepColumnCount) {
    const fields = [];
    let current = "";
    let insideQuotes = false;
    for (let position = 0; position < line.length && fields.length < keepColumnCount; position += 1) {
      const symbol = line[position];
      if (insideQuotes) {
        if (symbol === '"' && line[position + 1] === '"') {
          current += '"';
          position += 1;
        } else if (symbol === '"') {
          insideQuotes = false;
        } else {
          current += symbol;
        }
      } else if (symbol === '"') {
        insideQuotes = true;
      } else if (symbol === ",") {
        fields.push(current);
        current = "";
      } else {
        current += symbol;
      }
    }
    if (fields.length < keepColumnCount) fields.push(current);
    return fields;
  }

  function parseLabelCsv(text) {
    const keepColumnCount = LABEL_COLUMN_INDEX.taxonGroup + 1;
    const lines = text.split(/\r?\n/).filter((line) => line.length > 0);
    return lines.slice(1).map((line) => {
      const fields = splitCsvLine(line, keepColumnCount);
      return {
        index: Number(fields[LABEL_COLUMN_INDEX.index]),
        birdnetId: fields[LABEL_COLUMN_INDEX.birdnetId],
        scientificName: fields[LABEL_COLUMN_INDEX.scientificName],
        commonName: fields[LABEL_COLUMN_INDEX.commonName],
        taxonGroup: fields[LABEL_COLUMN_INDEX.taxonGroup],
      };
    });
  }

  function chooseIntraOpThreadCount() {
    if (!self.crossOriginIsolated) return 1;
    return Math.max(1, Math.min(MAXIMUM_INTRA_OP_THREADS_WHEN_ISOLATED, navigator.hardwareConcurrency || 1));
  }

  // Asks the browser not to evict the ~80 MB model under storage pressure
  // (iOS/Chrome otherwise treat Cache Storage as best-effort).
  async function requestPersistentStorage() {
    if (!navigator.storage || !navigator.storage.persist) return false;
    try {
      return (await navigator.storage.persisted()) || (await navigator.storage.persist());
    } catch (_unsupported) {
      return false;
    }
  }

  // Cache-first through the Cache API; `fromCache` feeds the load report.
  // A failed store (QuotaExceededError on a full phone) is not fatal: the
  // model still runs from the network copy, just not offline next time.
  async function fetchThroughModelCache(url) {
    const cache = self.caches ? await caches.open(MODEL_CACHE_NAME) : null;
    const cached = cache ? await cache.match(url) : undefined;
    if (cached) return { response: cached, fromCache: true, storedForOffline: true };
    const response = await fetch(url);
    if (!response.ok) throw new Error(`fetch ${url} -> HTTP ${response.status}`);
    if (!cache) return { response, fromCache: false, storedForOffline: false };
    try {
      await cache.put(url, response.clone());
      return { response, fromCache: false, storedForOffline: true };
    } catch (storeError) {
      console.warn(`model cache store failed for ${url}:`, storeError);
      return { response, fromCache: false, storedForOffline: false };
    }
  }

  async function forgetCachedModelFiles(urls) {
    if (!self.caches) return;
    const cache = await caches.open(MODEL_CACHE_NAME);
    await Promise.all(urls.map((url) => cache.delete(url)));
  }

  async function fetchModelAndLabels(options) {
    const [model, labels] = await Promise.all([
      fetchThroughModelCache(options.modelUrl),
      fetchThroughModelCache(options.labelsUrl),
    ]);
    const [modelBytes, labelText] = await Promise.all([model.response.arrayBuffer(), labels.response.text()]);
    return { model, labels, modelBytes, labelText };
  }

  function createSession(modelBytes) {
    return ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }

  async function load(options) {
    if (typeof ort === "undefined") throw new Error("onnxruntime-web (ort) failed to load");
    // One session per page: a second load would leave the first session's
    // wasm heap (hundreds of MB) unreleased.
    if (inferenceSession) await releaseSession();
    const threadCount = chooseIntraOpThreadCount();
    ort.env.wasm.numThreads = threadCount;
    ort.env.wasm.wasmPaths = options.ortWasmBaseUrl;
    const storagePersisted = await requestPersistentStorage();
    const fetchStarted = performance.now();
    let fetched = await fetchModelAndLabels(options);
    const fetchMilliseconds = performance.now() - fetchStarted;
    const createStarted = performance.now();
    try {
      inferenceSession = await createSession(fetched.modelBytes);
    } catch (createError) {
      // A truncated/corrupt cached copy would otherwise fail every launch
      // forever; refetch once. A network copy that fails is a real error.
      if (!fetched.model.fromCache) throw createError;
      console.warn("cached BirdNET model rejected; refetching", createError);
      await forgetCachedModelFiles([options.modelUrl, options.labelsUrl]);
      fetched = await fetchModelAndLabels(options);
      inferenceSession = await createSession(fetched.modelBytes);
    }
    const sessionCreateMilliseconds = performance.now() - createStarted;
    const { model, labels, modelBytes, labelText } = fetched;
    labelTable = parseLabelCsv(labelText);
    return {
      modelBytes: modelBytes.byteLength,
      modelFromCache: model.fromCache,
      modelStoredForOffline: model.storedForOffline && labels.storedForOffline,
      storagePersisted,
      labelCount: labelTable.length,
      fetchMilliseconds,
      sessionCreateMilliseconds,
      intraOpThreadCount: threadCount,
      crossOriginIsolated: self.crossOriginIsolated === true,
      ortVersion: ort.env.versions ? ort.env.versions.common : "unknown",
    };
  }

  function isLoaded() {
    return inferenceSession !== null;
  }

  async function releaseSession() {
    const session = inferenceSession;
    inferenceSession = null;
    if (session) await session.release();
  }

  // samples: Float32Array at 32 kHz. Returns the top-k classes for one window.
  async function classify(samples, topCount) {
    if (!inferenceSession) throw new Error("BirdNET model not loaded");
    const started = performance.now();
    const feeds = {};
    feeds[MODEL_INPUT_NAME] = new ort.Tensor("float32", samples, [1, samples.length]);
    const outputs = await inferenceSession.run(feeds, [MODEL_PREDICTIONS_OUTPUT_NAME]);
    const inferenceMilliseconds = performance.now() - started;
    const predictions = outputs[MODEL_PREDICTIONS_OUTPUT_NAME].data;
    const ranked = [];
    for (let classIndex = 0; classIndex < predictions.length; classIndex += 1) {
      const score = predictions[classIndex];
      if (ranked.length < topCount || score > ranked[ranked.length - 1].score) {
        ranked.push({ classIndex, score });
        ranked.sort((left, right) => right.score - left.score);
        if (ranked.length > topCount) ranked.pop();
      }
    }
    return {
      inferenceMilliseconds,
      top: ranked.map(({ classIndex, score }) => {
        const label = labelTable[classIndex] || {};
        return {
          classIndex,
          score,
          commonName: label.commonName || `class ${classIndex}`,
          scientificName: label.scientificName || "",
          taxonGroup: label.taxonGroup || "",
        };
      }),
    };
  }

  window.mmpFieldBirdnet = { load, classify, isLoaded, release: releaseSession };
})();
