/**
 * Field PWA microphone capture AudioWorklet.
 *
 * Ported from the user's metal-detecting PWA ("On Device Audio App",
 * web/app-extra/mic-worklet.js, MicProcessor), which already proved raw
 * AudioWorklet capture on iPhone Safari. Changes for Field Mode:
 *   - every chunk carries the AudioWorklet `currentFrame` of its first sample,
 *     so the Rust side keeps a gap-aware device-clock sample index (Stage-1
 *     and clip timing ride sample indices, never wall time);
 *   - a preallocated Float32Array ring replaces Array.push/splice (no garbage
 *     on the audio thread);
 *   - chunk size comes from processorOptions (the main thread owns config).
 */
class FieldMicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = (options && options.processorOptions) || {};
    this.chunkSampleCount = processorOptions.chunkSampleCount;
    this.pendingChunk = new Float32Array(this.chunkSampleCount);
    this.pendingSampleCount = 0;
    this.pendingChunkFirstFrame = 0;
    this.nextExpectedFrame = -1;
  }

  // Posts a partial chunk so every posted chunk is contiguous on the clock.
  flushPartialChunk() {
    if (this.pendingSampleCount === 0) return;
    const partial = this.pendingChunk.slice(0, this.pendingSampleCount);
    this.port.postMessage({ firstFrame: this.pendingChunkFirstFrame, samples: partial }, [partial.buffer]);
    this.pendingSampleCount = 0;
  }

  process(inputs) {
    const input = inputs[0];
    // No active input this quantum (e.g. the track stalled): the frames are
    // lost, so close the open chunk and let the index gap show downstream.
    if (!input || !input[0]) {
      this.flushPartialChunk();
      return true;
    }
    const channel = input[0];
    // `currentFrame` is the context frame of this render quantum's first sample.
    const quantumFrame = currentFrame;
    if (quantumFrame !== this.nextExpectedFrame) this.flushPartialChunk();
    this.nextExpectedFrame = quantumFrame + channel.length;
    for (let sampleIndex = 0; sampleIndex < channel.length; sampleIndex += 1) {
      if (this.pendingSampleCount === 0) this.pendingChunkFirstFrame = quantumFrame + sampleIndex;
      this.pendingChunk[this.pendingSampleCount] = channel[sampleIndex];
      this.pendingSampleCount += 1;
      if (this.pendingSampleCount === this.chunkSampleCount) {
        const fullChunk = this.pendingChunk;
        this.port.postMessage({ firstFrame: this.pendingChunkFirstFrame, samples: fullChunk }, [fullChunk.buffer]);
        this.pendingChunk = new Float32Array(this.chunkSampleCount);
        this.pendingSampleCount = 0;
      }
    }
    return true;
  }
}

registerProcessor("field-mic-capture", FieldMicCaptureProcessor);
