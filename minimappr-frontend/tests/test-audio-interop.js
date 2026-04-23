'use strict';
// Tests for audio-interop.js using Node.js built-in test runner (Node >= 18).
// Run: node --test tests/test-audio-interop.js
//
// These tests cover the three behaviors most likely to regress:
//   1. Early return when a canvas is missing (null-safety guard).
//   2. Happy-path return shape that Rust reads via js_sys::Reflect::get.
//   3. The URL-token stale-render guard: a superseded render must be discarded.
//
// FFT algorithm correctness is not tested here because the Cooley-Tukey
// implementation is a textbook algorithm; the paint-path integration tests
// (happy path + stale guard) exercise it indirectly through real FFT calls.

const { test } = require('node:test');
const assert = require('node:assert/strict');

// ── Synthetic AudioBuffer returned by the mock AudioContext ───────────────
// 4096 samples at 48 kHz gives 14 STFT frames with fftSize=512/hop=256,
// and 256 frequency bins (fftSize/2).  These exact values are asserted below.
const SAMPLE_COUNT = 4096;
const MOCK_AUDIO_BUFFER = Object.freeze({
    duration: 2.5,
    sampleRate: 48000,
    numberOfChannels: 1,
    getChannelData: () => new Float32Array(SAMPLE_COUNT).fill(0.1),
});

// ── Browser-environment stubs ─────────────────────────────────────────────

class MockCanvas {
    constructor() {
        this.dataset = {};
        this.clientWidth = 320;
        this.clientHeight = 80;
    }
    getContext() {
        // Return a no-op 2D context; property assignments are silently accepted.
        return {
            fillRect() {}, beginPath() {}, stroke() {}, putImageData() {},
            moveTo() {}, lineTo() {}, fillStyle: '', strokeStyle: '', lineWidth: 1,
        };
    }
}

class MockAudioContext {
    decodeAudioData(_arrayBuffer) {
        return Promise.resolve(MOCK_AUDIO_BUFFER);
    }
}

// Per-test canvas registry.
let canvases = new Map();

globalThis.window = {
    AudioContext: MockAudioContext,
    webkitAudioContext: MockAudioContext,
    devicePixelRatio: 1,
};
globalThis.document = {
    getElementById: id => canvases.get(id) ?? null,
    documentElement: {},
};
// cssVar() calls getComputedStyle as a bare global.
globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
// paintSpectrogramFromBuf uses ImageData.
globalThis.ImageData = class {
    constructor(w, h) { this.data = new Uint8ClampedArray(w * h * 4); }
};
// Default fetch: succeeds immediately.
globalThis.fetch = () => Promise.resolve({
    ok: true,
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(128)),
});

// Load the IIFE — must happen after globals are set above.
require('../audio-interop.js');
const { renderWaveformAndSpectrogram } = globalThis.audioInterop;

// ── Helper ────────────────────────────────────────────────────────────────

function addCanvasPair(waveformId, spectrogramId) {
    const wc = new MockCanvas();
    const sc = new MockCanvas();
    canvases.set(waveformId, wc);
    canvases.set(spectrogramId, sc);
    return { wc, sc };
}

// ── Tests ─────────────────────────────────────────────────────────────────

test('missing waveform canvas returns error without throwing', async () => {
    canvases.clear();
    const r = await renderWaveformAndSpectrogram('no-wc', 'no-sc', '/audio/x.wav', 512);
    assert.equal(r.ok, false);
    assert.match(r.error, /no canvas #no-wc/);
});

test('missing spectrogram canvas (waveform present) returns error without throwing', async () => {
    canvases.clear();
    canvases.set('only-wc', new MockCanvas());
    const r = await renderWaveformAndSpectrogram('only-wc', 'no-sc', '/audio/x.wav', 512);
    assert.equal(r.ok, false);
    assert.match(r.error, /no canvas #no-sc/);
});

test('happy path: return value contains fields read by Rust', async () => {
    canvases.clear();
    addCanvasPair('w-ok', 's-ok');
    const r = await renderWaveformAndSpectrogram('w-ok', 's-ok', '/audio/x.wav', 512);

    assert.equal(r.ok, true);
    // These two fields are extracted by js_sys::Reflect::get in detection_analysis.rs.
    assert.equal(r.duration_s, MOCK_AUDIO_BUFFER.duration);
    assert.equal(r.sample_rate, MOCK_AUDIO_BUFFER.sampleRate);

    // frames = floor((SAMPLE_COUNT - fftSize) / hop) = floor((4096-512)/256) = 14
    assert.equal(r.frames, 14, 'expected 14 STFT frames for 4096-sample buffer with fftSize=512');
    // bins = fftSize / 2 = 256
    assert.equal(r.bins, 256, 'expected 256 frequency bins for fftSize=512');
});

test('stale guard: token overwritten before decode completes causes discard', async () => {
    canvases.clear();
    const { wc, sc } = addCanvasPair('w-stale', 's-stale');

    // Hold the fetch open: resolveAb is populated once arrayBuffer() is called
    // (which happens after the `await fetch(url)` microtask resolves).
    let resolveAb;
    globalThis.fetch = () => Promise.resolve({
        ok: true,
        arrayBuffer: () => new Promise(r => { resolveAb = r; }),
    });

    // Start the render for url-A.  Tokens are stamped *synchronously* before
    // the first await inside fetchAndDecode, so we can overwrite them here
    // while still in the same synchronous turn.
    const pending = renderWaveformAndSpectrogram('w-stale', 's-stale', 'url-A', 512);

    // Simulate a newer render claiming these canvases (overwrites the token).
    wc.dataset.renderToken = 'url-B';
    sc.dataset.renderToken = 'url-B';

    // Drain one microtask tick so that `await fetch(url)` resolves and
    // arrayBuffer() is invoked, which populates resolveAb.
    await Promise.resolve();
    assert.ok(resolveAb, 'resolveAb should be set after one microtask tick');

    // Unblock the fetch chain.  AudioContext.decodeAudioData resolves immediately,
    // then renderWaveformAndSpectrogram checks the token ('url-B' !== 'url-A').
    resolveAb(new ArrayBuffer(128));

    const r = await pending;
    assert.equal(r.ok, false);
    assert.equal(r.error, 'stale');

    // Restore the default fetch for subsequent tests.
    globalThis.fetch = () => Promise.resolve({
        ok: true,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(128)),
    });
});
