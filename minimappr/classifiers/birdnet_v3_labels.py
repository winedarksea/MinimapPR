"""BirdNET+ V3.0 label table, regional species filter, and v2.4 -> V3 mapping.

The V3 labels CSV (Zenodo, Apache 2.0) has one row per model output, ``ix``
0..11724, with ``birdnet_id``, ``scientific_name``, ``common_name_en`` and
``taxon_group`` (Aves, Insecta, ..., Anthropogenic). Only those columns are
read; the 36 translated common names are ignored.

Pipeline labels stay in the v2.4 convention (lower-cased English common name)
so rules, retention buckets and taxonomy lookups keyed on e.g. "house finch"
keep matching. The v2.4 -> V3 table (``assets/birdnet/v24_to_v3_label_map.csv``,
built by ``scripts/build_birdnet_label_map.py``) records where a v2.4 label
lands in V3 so stored v2.4 label references can be migrated.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_V24_TO_V3_LABEL_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "birdnet" / "v24_to_v3_label_map.csv"
)

# Taxon groups that are living organisms. The rest (Anthropogenic, Geophony,
# Music) are sound sources, which must not inherit the "wildlife" category.
BIRDNET_V3_ANIMAL_TAXON_GROUPS: frozenset[str] = frozenset(
    {"Aves", "Insecta", "Amphibia", "Mammalia", "Actinopterygii", "Reptilia", "Arachnida"}
)

V24_TO_V3_MAP_COLUMNS: tuple[str, ...] = (
    "v24_label",
    "v24_scientific_name",
    "v24_common_name",
    "v3_index",
    "v3_birdnet_id",
    "v3_scientific_name",
    "v3_common_name_en",
    "match_method",
)
MATCH_METHOD_UNMAPPED = "unmapped"


@dataclass(frozen=True, slots=True)
class BirdNETV3Label:
    index: int
    birdnet_id: str
    scientific_name: str
    common_name_en: str
    taxon_group: str

    @property
    def pipeline_label(self) -> str:
        """Lower-cased English common name: the v2.4 pipeline label convention."""
        return self.common_name_en.strip().lower()

    @property
    def raw_species(self) -> str:
        """``Scientific_Common`` - same shape as v2.4's raw species string."""
        return f"{self.scientific_name}_{self.common_name_en}"

    @property
    def is_animal(self) -> bool:
        return self.taxon_group in BIRDNET_V3_ANIMAL_TAXON_GROUPS


@dataclass(frozen=True, slots=True)
class V24ToV3LabelMapping:
    v24_label: str
    v24_scientific_name: str
    v24_common_name: str
    v3_index: int | None
    v3_birdnet_id: str
    v3_scientific_name: str
    v3_common_name_en: str
    match_method: str

    @property
    def is_mapped(self) -> bool:
        return self.v3_index is not None


def load_birdnet_v3_labels(labels_csv_path: str | Path) -> list[BirdNETV3Label]:
    """Read the V3 labels CSV; rows must be dense and ordered by ``ix``."""
    labels: list[BirdNETV3Label] = []
    with Path(labels_csv_path).open(newline="", encoding="utf-8") as handle:
        for expected_index, row in enumerate(csv.DictReader(handle)):
            index = int(row["ix"])
            if index != expected_index:
                raise ValueError(f"BirdNET V3 labels CSV not dense: row {expected_index} has ix={index}")
            labels.append(
                BirdNETV3Label(
                    index=index,
                    birdnet_id=row["birdnet_id"].strip(),
                    scientific_name=row["scientific_name"].strip(),
                    common_name_en=(row.get("common_name_en") or row["common_name"]).strip(),
                    taxon_group=row["taxon_group"].strip(),
                )
            )
    return labels


def load_regional_species_names(species_list_path: str | Path) -> frozenset[str]:
    """One name per line (scientific, English common, or BirdNET id); ``#`` comments.

    This is the regional label-subset hook: produce the file from an eBird
    checklist or the V3.0 geomodel and point ``birdnet_v3_species_list_path``
    at it. Matching is case-insensitive.
    """
    names: set[str] = set()
    for line in Path(species_list_path).read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            names.add(text.casefold())
    return frozenset(names)


def resolve_allowed_label_indices(
    labels: list[BirdNETV3Label],
    allowed_names: frozenset[str] | None,
) -> np.ndarray | None:
    """Output indices kept by the regional filter, or ``None`` for all.

    An allow-list that matches nothing is a configuration error (wrong file,
    wrong column), so it raises instead of silently disabling the classifier.
    """
    if not allowed_names:
        return None
    kept = [
        label.index
        for label in labels
        if label.scientific_name.casefold() in allowed_names
        or label.common_name_en.casefold() in allowed_names
        or label.birdnet_id.casefold() in allowed_names
    ]
    if not kept:
        raise ValueError("BirdNET V3 species list matched no model labels; check the file's names")
    return np.asarray(kept, dtype=np.int64)


def load_v24_to_v3_label_map(
    map_csv_path: str | Path = DEFAULT_V24_TO_V3_LABEL_MAP_PATH,
) -> dict[str, V24ToV3LabelMapping]:
    """Keyed by the v2.4 raw label (``Scientific_Common``)."""
    mapping: dict[str, V24ToV3LabelMapping] = {}
    with Path(map_csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            v3_index_text = row["v3_index"].strip()
            mapping[row["v24_label"]] = V24ToV3LabelMapping(
                v24_label=row["v24_label"],
                v24_scientific_name=row["v24_scientific_name"],
                v24_common_name=row["v24_common_name"],
                v3_index=int(v3_index_text) if v3_index_text else None,
                v3_birdnet_id=row["v3_birdnet_id"],
                v3_scientific_name=row["v3_scientific_name"],
                v3_common_name_en=row["v3_common_name_en"],
                match_method=row["match_method"],
            )
    return mapping


def split_v24_raw_label(v24_raw_label: str) -> tuple[str, str]:
    """``"Haemorhous mexicanus_House Finch"`` -> (scientific, common)."""
    scientific, _, common = v24_raw_label.partition("_")
    return scientific.strip(), (common or scientific).strip()


# v2.4 pseudo-species classes with a V3 counterpart under a different name.
# "Noise" and "Environmental" have no V3 class and stay unmapped on purpose.
V24_NON_SPECIES_LABEL_OVERRIDES: dict[str, str] = {
    "Dog_Dog": "Canis familiaris",
    "Human vocal_Human vocal": "Homo sapiens",
    "Human non-vocal_Human non-vocal": "Homo sapiens",
    "Human whistle_Human whistle": "Homo sapiens",
    "Power tools_Power tools": "Power tool",
}


def build_v24_to_v3_mapping_rows(
    v24_raw_labels: list[str],
    v3_labels: list[BirdNETV3Label],
    scientific_alias_to_birdnet_id: dict[str, str] | None = None,
) -> list[V24ToV3LabelMapping]:
    """Match each v2.4 label to V3, most to least certain.

    1. ``override``: v2.4 pseudo-species classes (Dog, Human vocal, ...).
    2. ``scientific_name``: exact binomial.
    3. ``taxonomy_alias``: the v2.4 binomial is a listed synonym in the
       BirdNET+ Taxonomy (``scientific_name_aliases``; genus moves, lumps such
       as the redpolls into Acanthis flammea). Keys are case-folded aliases.
    4. ``common_name_en``: unique English name, for renames the alias list
       does not carry.
    Anything left is ``unmapped`` (no V3 class, or a split the taxonomy does
    not resolve one-to-one).
    """
    by_scientific = {label.scientific_name.casefold(): label for label in v3_labels}
    by_birdnet_id = {label.birdnet_id: label for label in v3_labels}
    aliases = scientific_alias_to_birdnet_id or {}
    by_common: dict[str, BirdNETV3Label] = {}
    ambiguous_common: set[str] = set()
    for label in v3_labels:
        key = label.common_name_en.casefold()
        if key in by_common:
            ambiguous_common.add(key)
        by_common[key] = label

    rows: list[V24ToV3LabelMapping] = []
    for raw in v24_raw_labels:
        scientific, common = split_v24_raw_label(raw)
        scientific_key, common_key = scientific.casefold(), common.casefold()
        match: BirdNETV3Label | None = None
        method = MATCH_METHOD_UNMAPPED
        if raw in V24_NON_SPECIES_LABEL_OVERRIDES:
            match = by_scientific.get(V24_NON_SPECIES_LABEL_OVERRIDES[raw].casefold())
            method = "override"
        if match is None and scientific_key in by_scientific:
            match, method = by_scientific[scientific_key], "scientific_name"
        if match is None and aliases.get(scientific_key) in by_birdnet_id:
            match, method = by_birdnet_id[aliases[scientific_key]], "taxonomy_alias"
        if match is None and common_key in by_common and common_key not in ambiguous_common:
            match, method = by_common[common_key], "common_name_en"
        if match is None:
            rows.append(V24ToV3LabelMapping(raw, scientific, common, None, "", "", "", MATCH_METHOD_UNMAPPED))
            continue
        rows.append(
            V24ToV3LabelMapping(
                raw,
                scientific,
                common,
                match.index,
                match.birdnet_id,
                match.scientific_name,
                match.common_name_en,
                method,
            )
        )
    return rows


def write_v24_to_v3_label_map(rows: list[V24ToV3LabelMapping], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(V24_TO_V3_MAP_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.v24_label,
                    row.v24_scientific_name,
                    row.v24_common_name,
                    "" if row.v3_index is None else row.v3_index,
                    row.v3_birdnet_id,
                    row.v3_scientific_name,
                    row.v3_common_name_en,
                    row.match_method,
                ]
            )

