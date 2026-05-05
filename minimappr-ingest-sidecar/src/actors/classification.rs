use tracing::warn;

use crate::{
    classifier_helper::ManifestClassificationAnnotator, dsp_events::DspEventPublisher,
    dsp_worker::ClassificationRequest, manifests::DspManifest,
};

/// Dedicated actor that owns the BirdNET subprocess bridge and processes
/// `ClassificationRequest`s from a flume bounded channel, decoupling BirdNET
/// latency from the main DSP pipeline.
///
/// PCM bytes arrive in-memory via the channel and are forwarded to the helper
/// over stdin as inline base64 payloads. The annotated manifest is published
/// through the shared SSE/replay publisher — no ManifestStore or temp PCM file write.
pub struct ClassificationWorker {
    pub annotator: ManifestClassificationAnnotator,
    pub rx: flume::Receiver<ClassificationRequest>,
    /// Shared publisher to SSE consumers. When Some, annotated manifests are
    /// replayable and streamed after BirdNET runs. When None, manifests are discarded.
    pub dsp_event_publisher: Option<DspEventPublisher>,
}

impl ClassificationWorker {
    pub async fn run_loop(mut self) {
        while let Ok(req) = self.rx.recv_async().await {
            let mut manifest = req.pending_manifest;
            manifest.raw_render_bytes = req.raw_render_bytes.clone();

            match self
                .annotator
                .classify_render(&req.pcm_bytes, req.sample_rate_hz)
                .await
            {
                Ok(Some(cls)) => {
                    if let Some(bn) = manifest.birdnet.as_mut() {
                        bn.label = Some(cls.label);
                        bn.label_confidence = Some(cls.label_confidence);
                        bn.scores = Some(cls.scores);
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    warn!(error = %err, "ClassificationWorker: BirdNET annotation failed");
                }
            }
            self.publish(manifest).await;
        }
    }

    async fn publish(&self, manifest: DspManifest) {
        if let Some(ref publisher) = self.dsp_event_publisher {
            let _ = publisher.publish(manifest).await;
        }
    }
}
