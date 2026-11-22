#!/usr/bin/env python3
"""Build the BirdNET v2.4 -> BirdNET+ V3.0 label-mapping table.

Inputs: the V3 labels CSV and the BirdNET+ Taxonomy v0.5 zip (synonym list,
CC BY-SA 4.0; ``scripts/fetch_birdnet_v3.py --include-taxonomy``), and the v2.4
species list from the installed ``birdnet`` package (or ``--v24-labels``, a
text file with one ``Scientific_Common`` label per line). Output:
``minimappr/assets/birdnet/v24_to_v3_label_map.csv`` plus a count summary.

Usage::

    python scripts/build_birdnet_label_map.py
    python scripts/build_birdnet_label_map.py --v24-labels path/to/BirdNET_GLOBAL_6K_V2.4_Labels.txt
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from minimappr.classifiers.birdnet_v3_labels import (  # noqa: E402
    DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
    MATCH_METHOD_UNMAPPED,
    build_v24_to_v3_mapping_rows,
    load_birdnet_v3_labels,
    write_v24_to_v3_label_map,
)

DEFAULT_V3_LABELS_PATH = REPOSITORY_ROOT / "data/models/birdnet_v3/BirdNET+_V3.0_Global_12K_Labels.csv"
DEFAULT_TAXONOMY_ZIP_PATH = REPOSITORY_ROOT / "data/models/birdnet_v3/BirdNET+_Taxonomy_v0.5-Nov2026.zip"
TAXONOMY_ALIAS_SEPARATOR = "|"


def load_taxonomy_scientific_aliases(taxonomy_zip_path: Path) -> dict[str, str]:
    """Case-folded synonym -> birdnet_id from the taxonomy CSV; ambiguous aliases dropped.

    The zip is untrusted data: only the CSV member is read, as text.
    """
    with zipfile.ZipFile(taxonomy_zip_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".csv"))
        text = archive.read(member).decode("utf-8")
    alias_owner: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in csv.DictReader(io.StringIO(text)):
        for alias in filter(None, (a.strip() for a in row["scientific_name_aliases"].split(TAXONOMY_ALIAS_SEPARATOR))):
            key = alias.casefold()
            if key in alias_owner and alias_owner[key] != row["birdnet_id"]:
                ambiguous.add(key)
            alias_owner[key] = row["birdnet_id"]
    return {alias: owner for alias, owner in alias_owner.items() if alias not in ambiguous}


def load_v24_labels(v24_labels_path: Path | None) -> list[str]:
    if v24_labels_path is not None:
        return [line.strip() for line in v24_labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from birdnet import model_loader  # noqa: PLC0415 - optional dependency

    return list(model_loader.load("acoustic", "2.4", "tf", library="tflite").species_list)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v3-labels", type=Path, default=DEFAULT_V3_LABELS_PATH)
    parser.add_argument("--v24-labels", type=Path, default=None)
    parser.add_argument("--taxonomy-zip", type=Path, default=DEFAULT_TAXONOMY_ZIP_PATH, help="fetch with --include-taxonomy")
    parser.add_argument("--output", type=Path, default=DEFAULT_V24_TO_V3_LABEL_MAP_PATH)
    parser.add_argument("--show-unmapped", type=int, default=40, help="print up to N unmapped labels")
    args = parser.parse_args(argv)

    v3_labels = load_birdnet_v3_labels(args.v3_labels)
    v24_labels = load_v24_labels(args.v24_labels)
    aliases = load_taxonomy_scientific_aliases(args.taxonomy_zip) if args.taxonomy_zip.is_file() else {}
    if not aliases:
        print(f"WARNING: no taxonomy aliases ({args.taxonomy_zip} missing); genus moves fall back to common names")
    rows = build_v24_to_v3_mapping_rows(v24_labels, v3_labels, aliases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_v24_to_v3_label_map(rows, args.output)

    counts = collections.Counter(row.match_method for row in rows)
    targets = collections.Counter(row.v3_index for row in rows if row.v3_index is not None)
    lumped = sum(1 for count in targets.values() if count > 1)
    print(f"v2.4 labels: {len(v24_labels)}  V3 labels: {len(v3_labels)}")
    for method, count in sorted(counts.items()):
        print(f"  {method:16s} {count}")
    print(f"  V3 targets hit by >1 v2.4 label (lumps): {lumped}")
    print(f"  V3 labels with no v2.4 source: {len(v3_labels) - len(targets)}")
    unmapped = [row.v24_label for row in rows if row.match_method == MATCH_METHOD_UNMAPPED]
    for label in unmapped[: args.show_unmapped]:
        print(f"  unmapped: {label}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
