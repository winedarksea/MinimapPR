// Audio decoding + canvas rendering for the Audio Analysis page.
// Exposes globalThis.audioInterop.{renderWaveform, renderSpectrogram,
//   renderWaveformAndSpectrogram}.
//
// renderWaveformAndSpectrogram is the preferred call site: it decodes the audio
// file exactly once and paints both canvases from the shared buffer.  It also
// stamps a URL token on each canvas element before the async decode so that a
// later render starting on the same canvas ID can be detected and discarded
// (stale-render guard).
(function () {
  "use strict";

  let _ctx = null;
  function ctx() {
    if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)();
    return _ctx;
  }

  async function fetchAndDecode(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    const ab = await r.arrayBuffer();
    return await ctx().decodeAudioData(ab.slice(0));
  }

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  // Cooley-Tukey radix-2 FFT operating in-place on Float64 re/im arrays.
  // Kept at module scope so it is shared between the waveform and spectrogram
  // paint paths without re-allocating closures on every call.
  function fft(re, im) {
    const n = re.length;
    let j = 0;
    for (let i = 1; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < len / 2; k++) {
          const aRe = re[i + k], aIm = im[i + k];
          const bRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
          const bIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
          re[i + k] = aRe + bRe; im[i + k] = aIm + bIm;
          re[i + k + len / 2] = aRe - bRe; im[i + k + len / 2] = aIm - bIm;
          const nxtRe = curRe * wRe - curIm * wIm;
          const nxtIm = curRe * wIm + curIm * wRe;
          curRe = nxtRe; curIm = nxtIm;
        }
      }
    }
  }

  // ── Buffer-level paint functions ─────────────────────────────────────────
  // These accept an already-decoded AudioBuffer so the caller controls when
  // decoding happens (once per renderWaveformAndSpectrogram call).

  function paintWaveformFromBuf(canvas, buf) {
    const ch = buf.getChannelData(0);
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, canvas.clientWidth * dpr);
    const h = Math.max(1, canvas.clientHeight * dpr);
    canvas.width = w; canvas.height = h;
    const g = canvas.getContext("2d");
    g.fillStyle = cssVar("--md-sys-color-surface-container-low", "#111");
    g.fillRect(0, 0, w, h);
    g.strokeStyle = cssVar("--md-sys-color-primary", "#58a6ff");
    g.lineWidth = 1;
    const mid = h / 2;

    // Find peak to auto-scale amplitude to fill canvas height.
    let peak = 0;
    for (let i = 0; i < ch.length; i++) {
      const abs = Math.abs(ch[i]);
      if (abs > peak) peak = abs;
    }
    if (peak < 1e-6) peak = 1.0;
    const scale = (mid * 0.92) / peak;

    const step = Math.max(1, Math.floor(ch.length / w));
    g.beginPath();
    for (let x = 0; x < w; x++) {
      let min = peak, max = -peak;
      const start = x * step;
      const end = Math.min(ch.length, start + step);
      for (let i = start; i < end; i++) {
        const v = ch[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      g.moveTo(x + 0.5, mid + min * scale);
      g.lineTo(x + 0.5, mid + max * scale);
    }
    g.stroke();
    // Center line
    g.strokeStyle = cssVar("--md-sys-color-outline-variant", "#333");
    g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
  }

  function paintSpectrogramFromBuf(canvas, buf, fftSize) {
    const ch = buf.getChannelData(0);
    const N = fftSize || 512;
    const hop = N / 2;
    const frames = Math.max(1, Math.floor((ch.length - N) / hop));
    const bins = N / 2;

    // Hann window
    const win = new Float32Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));

    // Compute magnitude matrix (frames × bins) in dB.
    const mags = new Float32Array(frames * bins);
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    let gmax = -Infinity, gmin = Infinity;
    for (let f = 0; f < frames; f++) {
      const off = f * hop;
      for (let i = 0; i < N; i++) { re[i] = ch[off + i] * win[i]; im[i] = 0; }
      fft(re, im);
      for (let k = 0; k < bins; k++) {
        const m = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
        const db = 20 * Math.log10(m + 1e-9);
        mags[f * bins + k] = db;
        if (db > gmax) gmax = db;
        if (db < gmin) gmin = db;
      }
    }
    // Clip to sensible dB range below gmax.
    const floor = Math.max(gmin, gmax - 80);

    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, canvas.clientWidth * dpr);
    const h = Math.max(1, canvas.clientHeight * dpr);
    canvas.width = w; canvas.height = h;
    const img = new ImageData(w, h);
    const data = img.data;

    // Bilinear interpolation: sample mags at fractional (frame, bin) coords.
    function sampleMag(fPos, kPos) {
      const f0 = Math.max(0, Math.min(frames - 1, Math.floor(fPos)));
      const f1 = Math.min(frames - 1, f0 + 1);
      const k0 = Math.max(0, Math.min(bins - 1, Math.floor(kPos)));
      const k1 = Math.min(bins - 1, k0 + 1);
      const ff = fPos - f0, kf = kPos - k0;
      return (mags[f0 * bins + k0] * (1 - ff) + mags[f1 * bins + k0] * ff) * (1 - kf)
           + (mags[f0 * bins + k1] * (1 - ff) + mags[f1 * bins + k1] * ff) * kf;
    }

    for (let x = 0; x < w; x++) {
      const fPos = (x / (w - 1)) * (frames - 1);
      for (let y = 0; y < h; y++) {
        // y=0 top = highest freq; invert for spectrogram orientation
        const kPos = ((h - 1 - y) / (h - 1)) * (bins - 1);
        const db = sampleMag(fPos, kPos);
        const t = Math.max(0, Math.min(1, (db - floor) / (gmax - floor + 1e-9)));
        // Perceptual ramp: dark blue → cyan → green → yellow → red
        const r = Math.round(255 * Math.min(1, Math.max(0, t * 2 - 0.2)));
        const gg = Math.round(255 * Math.min(1, Math.max(0, t * 2 - 1)));
        const b = Math.round(255 * Math.min(1, Math.max(0, 1 - t * 1.5)));
        const idx = (y * w + x) * 4;
        data[idx] = r; data[idx + 1] = gg; data[idx + 2] = b; data[idx + 3] = 255;
      }
    }
    canvas.getContext("2d").putImageData(img, 0, 0);
    return { frames, bins, db_min: floor, db_max: gmax };
  }

  // ── Public API ───────────────────────────────────────────────────────────

  // Preferred entry point: decodes once, renders both canvases.
  // Uses the audio URL as a render token stamped on each canvas element before
  // the async fetch so that a later call on the same canvas ID can detect and
  // discard the stale result.
  async function renderWaveformAndSpectrogram(waveformId, spectrogramId, url, fftSize) {
    const wCanvas = document.getElementById(waveformId);
    const sCanvas = document.getElementById(spectrogramId);
    if (!wCanvas) return { ok: false, error: "no canvas #" + waveformId };
    if (!sCanvas) return { ok: false, error: "no canvas #" + spectrogramId };

    // Stamp the token before the async boundary so any later caller on the same
    // canvas ID will overwrite it, allowing us to detect staleness on return.
    wCanvas.dataset.renderToken = url;
    sCanvas.dataset.renderToken = url;

    const buf = await fetchAndDecode(url);

    // Discard results if a newer render has claimed these canvases.
    if (wCanvas.dataset.renderToken !== url || sCanvas.dataset.renderToken !== url) {
      return { ok: false, error: "stale" };
    }

    paintWaveformFromBuf(wCanvas, buf);
    const spectInfo = paintSpectrogramFromBuf(sCanvas, buf, fftSize);
    return {
      ok: true,
      duration_s: buf.duration,
      sample_rate: buf.sampleRate,
      channels: buf.numberOfChannels,
      samples: buf.getChannelData(0).length,
      ...spectInfo,
    };
  }

  // Single-canvas wrappers kept for backwards compatibility.
  async function renderWaveform(canvasId, url) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return { ok: false, error: "no canvas #" + canvasId };
    const buf = await fetchAndDecode(url);
    paintWaveformFromBuf(canvas, buf);
    return {
      ok: true,
      duration_s: buf.duration,
      sample_rate: buf.sampleRate,
      channels: buf.numberOfChannels,
      samples: buf.getChannelData(0).length,
    };
  }

  async function renderSpectrogram(canvasId, url, fftSize) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return { ok: false, error: "no canvas #" + canvasId };
    const buf = await fetchAndDecode(url);
    const spectInfo = paintSpectrogramFromBuf(canvas, buf, fftSize);
    return { ok: true, duration_s: buf.duration, sample_rate: buf.sampleRate, ...spectInfo };
  }

  globalThis.audioInterop = { renderWaveform, renderSpectrogram, renderWaveformAndSpectrogram };
})();
