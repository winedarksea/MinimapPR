"""App launcher for the E10 hot-path gate: the checkout's FastAPI app with a slow classifier.

Run as a script (``python hot_path_release_gate_launcher.py <checkout> <port>
<classifier_delay_s> <identity_json>``) with the *target* checkout first on
``sys.path``, so a baseline checkout's ``minimappr`` is what serves. It relies
only on seams that predate the gate (``minimappr.main._build_runtime_classifier``,
``AudioClassifier``), so it can drive commits older than this file.

The classifier sleeps on every call: a classifier consumer slower than the
render rate must back up only its own queue, never ingest or the solver.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main() -> None:
    checkout_root, port, classifier_delay_seconds, identity_path = (
        Path(sys.argv[1]).resolve(),
        int(sys.argv[2]),
        float(sys.argv[3]),
        Path(sys.argv[4]),
    )
    sys.path.insert(0, str(checkout_root))

    import minimappr
    import minimappr.main as app_module
    import uvicorn
    from minimappr.classifiers.base import AudioClassifier, ClassificationResult

    class SlowConstantClassifier(AudioClassifier):
        def classify(self, samples, sample_rate_hz):
            del samples, sample_rate_hz
            time.sleep(classifier_delay_seconds)
            return ClassificationResult(label="sparrow", confidence=0.93, scores={"sparrow": 0.93}, features={})

    app_module._build_runtime_classifier = lambda settings: SlowConstantClassifier()
    # Proves which checkout actually served, since an editable install can shadow sys.path.
    identity_path.write_text(json.dumps({"minimappr_file": str(Path(minimappr.__file__).resolve())}), encoding="utf-8")
    uvicorn.run(app_module.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
