use std::io::Write;

use tempfile::NamedTempFile;
use tokio::sync::broadcast;
use tracing::warn;

use crate::{
    classifier_helper::ManifestClassificationAnnotator, dsp_worker::ClassificationRequest,
    manifests::DspManifest,
};

/// Dedicated actor that owns the BirdNET subprocess bridge and processes
/// `ClassificationRequest`s from a flume bounded channel, decoupling BirdNET
/// latency from the main DSP pipeline.
///
/// PCM bytes arrive in-memory via the channel. A transient temp file is written
/// for the BirdNET subprocess (auto-deleted on drop). The annotated manifest is
/// broadcast via `dsp_result_tx` — no ManifestStore disk write.
pub struct ClassificationWorker {
    pub annotator: ManifestClassificationAnnotator,
    pub rx: flume::Receiver<ClassificationRequest>,
    /// Broadcast channel to SSE consumers. When Some, annotated manifests are
    /// sent here after BirdNET runs. When None, manifests are discarded (labels
    /// were already sent unlabeled by run_io before queuing this request).
    pub dsp_result_tx: Option<broadcast::Sender<DspManifest>>,
}

impl ClassificationWorker {
    pub async fn run_loop(mut self) {
        while let Ok(req) = self.rx.recv_async().await {
            let mut manifest = req.pending_manifest;

            // Write PCM bytes to a temp file for the BirdNET subprocess.
            // NamedTempFile auto-deletes on drop — the only accepted disk write.
            let pcm_path_result = tokio::task::spawn_blocking({
                let bytes = req.pcm_bytes.clone();
                move || -> std::io::Result<NamedTempFile> {
                    let mut tmp = NamedTempFile::new()?;
                    tmp.write_all(&bytes)?;
                    tmp.flush()?;
                    Ok(tmp)
                }
            })
            .await;

            let tmp_file: NamedTempFile = match pcm_path_result {
                Ok(Ok(f)) => f,
                Ok(Err(err)) => {
                    warn!(error = %err, "ClassificationWorker: failed to write PCM temp file");
                    self.broadcast(manifest);
                    continue;
                }
                Err(join_err) => {
                    warn!(error = %join_err, "ClassificationWorker: spawn_blocking panicked");
                    self.broadcast(manifest);
                    continue;
                }
            };

            match self
                .annotator
                .classify_render(tmp_file.path(), req.sample_rate_hz)
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
            // tmp_file dropped here — OS temp file auto-deleted.
            drop(tmp_file);
            self.broadcast(manifest);
        }
    }

    fn broadcast(&self, manifest: DspManifest) {
        if let Some(ref tx) = self.dsp_result_tx {
            let _ = tx.send(manifest);
        }
    }
}
