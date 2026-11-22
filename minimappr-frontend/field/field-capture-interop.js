/**
 * Field PWA microphone capture graph (getUserMedia -> AudioWorklet).
 *
 * Ported from the user's metal-detecting PWA ("On Device Audio App",
 * webApp/.../web/audio/MicCapture.kt `getUserMediaStrictlyVerified` and
 * WebMicrophoneInputSource.kt `buildInfCaptureGraph`): processing-off
 * constraints, iOS Safari `resume()` inside the user gesture, and the worklet
 * node routed into a muted gain so the graph keeps pulling audio. Dropped from
 * the port: the speaker pass-through monitor and the ScriptProcessor
 * fallback (every browser the Field PWA targets has AudioWorklet; a silent
 * fallback with different timing would corrupt sample-index bookkeeping).
 *
 * Exposes `window.mmpFieldCapture.start(options, onChunk)` where `onChunk`
 * receives (firstFrame: number, samples: Float32Array). Resolves to the
 * negotiated settings so Rust can check the delivered sample rate.
 */
(function () {
  "use strict";

  let activeCapture = null;

  function captureConstraints(deviceId) {
    const constraints = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    };
    if (deviceId) constraints.deviceId = { exact: deviceId };
    return constraints;
  }

  async function start(options, onChunk) {
    if (activeCapture) throw new Error("capture already running");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser does not support microphone capture.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: captureConstraints(options.deviceId || ""),
      video: false,
    });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass({ sampleRate: options.requestedSampleRateHz });
    // iOS Safari starts the context suspended; resume within the user gesture.
    if (context.state === "suspended" && context.resume) await context.resume();
    await context.audioWorklet.addModule(options.workletUrl);
    const source = context.createMediaStreamSource(stream);
    const captureNode = new AudioWorkletNode(context, "field-mic-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: "explicit",
      processorOptions: { chunkSampleCount: options.chunkSampleCount },
    });
    const silentSink = context.createGain();
    silentSink.gain.value = 0;
    captureNode.port.onmessage = (event) => onChunk(event.data.firstFrame, event.data.samples);
    source.connect(captureNode);
    captureNode.connect(silentSink);
    silentSink.connect(context.destination);

    const track = stream.getAudioTracks()[0];
    const trackSettings = track && track.getSettings ? track.getSettings() : {};
    // Plan §Audio route changes: the PWA cannot pin input, so it watches for
    // route changes and surfaces them as clock-model invalidations.
    const routeChangeListeners = [];
    const notifyRouteChange = (reason) => routeChangeListeners.forEach((listener) => listener(reason));
    if (track) {
      track.addEventListener("mute", () => notifyRouteChange("track-muted"));
      track.addEventListener("ended", () => notifyRouteChange("track-ended"));
    }
    const onDeviceChange = () => notifyRouteChange("devicechange");
    navigator.mediaDevices.addEventListener("devicechange", onDeviceChange);

    activeCapture = { stream, context, captureNode, silentSink, source, onDeviceChange, routeChangeListeners };
    return {
      contextSampleRateHz: context.sampleRate,
      trackSampleRateHz: trackSettings.sampleRate || null,
      deviceLabel: track ? track.label : "",
      echoCancellation: trackSettings.echoCancellation === true,
      noiseSuppression: trackSettings.noiseSuppression === true,
      autoGainControl: trackSettings.autoGainControl === true,
    };
  }

  function onRouteChange(listener) {
    if (activeCapture) activeCapture.routeChangeListeners.push(listener);
  }

  async function stop() {
    if (!activeCapture) return;
    const capture = activeCapture;
    activeCapture = null;
    navigator.mediaDevices.removeEventListener("devicechange", capture.onDeviceChange);
    capture.captureNode.port.onmessage = null;
    capture.source.disconnect();
    capture.captureNode.disconnect();
    capture.silentSink.disconnect();
    capture.stream.getTracks().forEach((track) => track.stop());
    await capture.context.close();
  }

  window.mmpFieldCapture = { start, stop, onRouteChange };
})();
