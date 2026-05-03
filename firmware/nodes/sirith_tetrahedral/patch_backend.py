import re

file_path = "../../../minimappr-ingest-sidecar/src/ingest_backend.rs"
with open(file_path, "r") as f:
    content = f.read()

content = content.replace(
"""        if self.runtime_config.skip_binary_writes
            && self.runtime_config.raw_manifest_tx.is_some()
        {""",
"""        if self.runtime_config.raw_manifest_tx.is_some() {"""
)

with open(file_path, "w") as f:
    f.write(content)
