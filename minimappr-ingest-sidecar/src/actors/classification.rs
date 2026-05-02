use tracing::warn;

use crate::{
    classifier_helper::ManifestClassificationAnnotator,
    dsp_worker::ClassificationRequest,
    manifests::ManifestStore,
};

/// Dedicated actor that owns the BirdNET subprocess bridge and processes
/// `ClassificationRequest`s from a flume bounded channel, decoupling BirdNET
/// latency from the main DSP pipeline.
pub struct ClassificationWorker {
    pub annotator: ManifestClassificationAnnotator,
    pub manifest_store: ManifestStore,
    pub rx: flume::Receiver<ClassificationRequest>,
}

impl ClassificationWorker {
    pub async fn run_loop(mut self) {
        while let Ok(req) = self.rx.recv_async().await {
            let mut manifest = req.pending_manifest;
            match self
                .annotator
                .classify_render(&req.pcm_path, req.sample_rate_hz)
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
            if let Err(err) = self.manifest_store.publish(manifest).await {
                warn!(error = %err, "ClassificationWorker: failed to publish manifest");
            }
        }
    }
}
