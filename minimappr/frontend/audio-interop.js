// Audio decoding + canvas rendering for the Audio Analysis page.
// Exposes globalThis.audioInterop.{renderWaveform, renderSpectrogram, peakMeter}.
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

  async function renderWaveform(canvasId, url) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return { ok: false, error: "no canvas #" + canvasId };
    const buf = await fetchAndDecode(url);
    const ch = buf.getChannelData(0);
    const w = canvas.clientWidth * (window.devicePixelRatio || 1);
    const h = canvas.clientHeight * (window.devicePixelRatio || 1);
    canvas.width = w; canvas.height = h;
    const g = canvas.getContext("2d");
    g.fillStyle = cssVar("--md-sys-color-surface-container-low", "#111");
    g.fillRect(0, 0, w, h);
    g.strokeStyle = cssVar("--md-sys-color-primary", "#58a6ff");
    g.lineWidth = 1;
    const mid = h / 2;
    const step = Math.max(1, Math.floor(ch.length / w));
    g.beginPath();
    for (let x = 0; x < w; x++) {
      let min = 1.0, max = -1.0;
      const start = x * step;
      const end = Math.min(ch.length, start + step);
      for (let i = start; i < end; i++) {
        const v = ch[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      g.moveTo(x + 0.5, mid + min * mid);
      g.lineTo(x + 0.5, mid + max * mid);
    }
    g.stroke();
    // Center line
    g.strokeStyle = cssVar("--md-sys-color-outline-variant", "#333");
    g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
    return {
      ok: true,
      duration_s: buf.duration,
      sample_rate: buf.sampleRate,
      channels: buf.numberOfChannels,
      samples: ch.length,
    };
  }

  async function renderSpectrogram(canvasId, url, fftSize) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return { ok: false, error: "no canvas #" + canvasId };
    const buf = await fetchAndDecode(url);
    const ch = buf.getChannelData(0);
    const N = fftSize || 512;
    const hop = N / 2;
    const frames = Math.max(1, Math.floor((ch.length - N) / hop));
    const bins = N / 2;

    // Hann window
    const win = new Float32Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));

    // Naive DFT is too slow for large N; use OfflineAudioContext + AnalyserNode is not ideal either.
    // Instead: use Web Audio AnalyserNode.getFloatFrequencyData over an OfflineAudioContext render isn't real-time.
    // We do a simple real DFT via the Cooley-Tukey radix-2 implementation.
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

    // Compute magnitude matrix (frames × bins) in dB, clipped.
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

    const w = canvas.clientWidth * (window.devicePixelRatio || 1);
    const h = canvas.clientHeight * (window.devicePixelRatio || 1);
    canvas.width = w; canvas.height = h;
    const img = new ImageData(w, h);
    const data = img.data;

    for (let x = 0; x < w; x++) {
      const fIdx = Math.min(frames - 1, Math.floor((x / w) * frames));
      for (let y = 0; y < h; y++) {
        // y=0 top = highest freq; invert
        const kIdx = Math.min(bins - 1, Math.floor(((h - 1 - y) / h) * bins));
        const db = mags[fIdx * bins + kIdx];
        const t = Math.max(0, Math.min(1, (db - floor) / (gmax - floor + 1e-9)));
        // Simple perceptual ramp: dark blue → magenta → yellow
        const r = Math.round(255 * Math.min(1, Math.max(0, t * 2 - 0.2)));
        const gg = Math.round(255 * Math.min(1, Math.max(0, t * 2 - 1)));
        const b = Math.round(255 * Math.min(1, Math.max(0, 1 - t * 1.5)));
        const idx = (y * w + x) * 4;
        data[idx] = r; data[idx + 1] = gg; data[idx + 2] = b; data[idx + 3] = 255;
      }
    }
    canvas.getContext("2d").putImageData(img, 0, 0);
    return {
      ok: true,
      duration_s: buf.duration,
      sample_rate: buf.sampleRate,
      frames, bins, db_min: floor, db_max: gmax,
    };
  }

  globalThis.audioInterop = { renderWaveform, renderSpectrogram };
})();
