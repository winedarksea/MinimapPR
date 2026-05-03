import re

file_path = "../../../minimappr/api/spool_consumer.py"
with open(file_path, "r") as f:
    content = f.read()

# Remove reverse=True from manifest_paths sort
content = re.sub(
    r'(key=lambda path: path\.stat\(\)\.st_mtime_ns,\n\s*)reverse=True,\n',
    r'\1',
    content
)

# Remove reverse=max_claims is not None from claims sort
content = re.sub(
    r'(key=lambda item: item\[0\],\n\s*)reverse=max_claims is not None,\n',
    r'\1',
    content
)

with open(file_path, "w") as f:
    f.write(content)
