# W1-fixtures — notes

Branch `fm/fixtures`. Status: xeno-canto part done; Bluefin snippet part **not pulled** (box
down on 2026-11-21: `nc -z -G 2 127.0.0.1 2222` → exit 1). Manifest, fetch script and loud-skip
helpers are in place so the pull is a mechanical follow-up (commands below).

## What landed

| Path | Purpose |
|---|---|
| `tests/fixtures/field_audio/*.wav` | 9 clips, 48 kHz mono PCM16, 6.63 MB total (budget 10 MB) |
| `tests/fixtures/field_audio/manifest.toml` | machine-readable list: file, species, call type, `band_hz`, licence, XC id, source offset, `same_band_group`, `needs_listen_check` |
| `tests/fixtures/field_audio/ATTRIBUTION` | per-clip XC id, species, recordist, licence + URL, trim offsets, processing |
| `tests/field_audio_fixtures.py` | loader (`field_audio_clips(role=)`, `field_audio_clip("tonal_whistle")`, `load_field_audio_clip`) + `require_field_bundle(name)` (verified cache path, else loud `pytest.skip`; `pytest.fail` when `CI` is set) |
| `tests/test_field_audio_fixtures.py` | 32 tests: format/duration/level per clip, in-band energy ≥ 75 % for birds, attribution row per clip, size budget, bundle fetch/verify/skip |
| `tests/fixtures/field_bundles.toml` | schema for out-of-git bundles (name, sha256, size_bytes, source, cache_path, description, licence, optional extract/speech_checked); empty for now |
| `scripts/fetch_field_fixtures.py` | `fetch [--only]`, `list`, `register FILE ...` (prints a ready `[[bundle]]` table). Sources: https (generic UA), `ssh://user@host:port/path` (rsync), `file://`. Size + sha256 verified before the atomic move into the cache |
| `scripts/build_field_audio_fixtures.py` | rebuilds the committed WAVs from xeno-canto using the manifest offsets (reproducibility) |

Run: `python -m pytest tests/test_field_audio_fixtures.py -q` → 32 passed.

## Clip set

| Role | File | XC | Licence | Notes |
|---|---|---|---|---|
| tonal whistle | `wtsp_whistle_xc569523.wav` | 569523 | BY-SA 4.0 | clean single song, ~3–4.5 kHz pure tones — the GCC-PHAT cycle-skip probe |
| dry trill | `chsp_trill_xc320440.wav` | 320440 | BY-SA 4.0 | broadband-per-note fast trill |
| musical trill | `deju_trill_xc569535.wav` | 569535 | BY-SA 4.0 | same-band pair with Chipping Sparrow (`trill_chsp_deju`) |
| broadband call | `blja_call_xc935610.wav` | 935610 | BY-SA 4.0 | harmonic stacks to ~10 kHz |
| harsh call | `amcr_caw_xc682428.wav` | 682428 | BY-SA 4.0 | 3 caws, noisy-harmonic |
| caroled song | `amro_song_xc132864.wav` | 132864 | BY-SA 3.0 | `robin_tanager_grosbeak` group |
| burry song | `scta_song_xc795439.wav` | 795439 | BY-SA 4.0 | `robin_tanager_grosbeak` group |
| caroled song | `rbgr_song_xc133913.wav` | 133913 | BY-SA 3.0 | `robin_tanager_grosbeak` group (classic robin/tanager/grosbeak confusion triad) |
| ambience | `ambience_forest_breeze_xc980972.wav` | 980972 | **BY-NC-SA 4.0** | 30 s mixed forest, low wind; quietest 30 s of a 7 min take; sparse distant birds remain |

Normalization: birds to −3 dBFS sample peak (their RMS then spans −15.7 to −23.3 dBFS, i.e.
the natural crest-factor differences survive); ambience to −30 dBFS RMS (peak −12 dBFS). One
source channel only (never summed). Details in `ATTRIBUTION` / build script docstring.

## How it was fetched (and what did not work)

- xeno-canto API v3 requires a personal key (`Missing or invalid 'key' parameter`); API v2 is
  retired. Keyless route that works: the HTML explore page
  `https://xeno-canto.org/explore?query=...` (supports `q:A`, `lic:BY-SA`, `lic:BY`,
  `grp:soundscape`, `rmk:`, `cnt:` tags) for search, and `https://xeno-canto.org/<id>/download`
  for the original file. Generic User-Agent `minimappr-fixture-fetch/0.1`, ~1 s between requests.
- Licence filter: BY-SA/BY only for birds (most XC uploads are BY-NC-SA or BY-NC-ND; ND was
  excluded outright since trimming is an adaptation). No BY/BY-SA forest-wind soundscape of
  usable quality turned up (BY-SA soundscapes were dawn choruses, insects, frogs or traffic), so
  the ambience is BY-NC-SA — allowed by the test strategy ("suits a free non-commercial
  project") but flagged in `ATTRIBUTION` for replacement if distribution ever goes commercial.
- freesound was not tried (downloads need an account/OAuth).
- Candidates downloaded and rejected: XC993033 (WTSP, dense multi-species chorus under the
  song — not a clean whistle), XC134880 (RBGR, a pulsed sound overlapping the song band),
  XC1145165/795438 (SCTA, lower SNR), XC570791 (AMRO alt), XC254451/254480/797245/935607/
  175223/1172881 (alternates, lower SNR), soundscapes XC892191/481648 (frog/bird chorus),
  XC861479 (insect bed + tonal line), XC1017626 (constant ~8.5 kHz tone), XC1073352 (bursty
  activity), XC982295 (woodpecker drumming throughout), XC1137410/178043 (busier than 980972).

## Privacy / speech screening

No listening was possible. Screening was spectral: (1) fraction of 20 ms frames where
150–1000 Hz energy is >10 dB above its own floor, (2) share of 3–7 Hz (syllabic-rate)
modulation in that band's envelope, plus visual spectrogram review for formant stacks.
All bird clips except the crow show 0–5 % low-band activity (the crow's 28 % is the caws
themselves); the ambience shows 0 %. None of the recordists' remarks mention people for the
selected uploads (XC801712, whose remarks mention "children calling", was never selected).

**Needs a human listen-check** (marked `needs_listen_check = true`):
`amcr_caw_xc682428.wav` (energy inside the speech band), `ambience_forest_breeze_xc980972.wav`
(30 s is long enough to hide a distant voice).

## Bluefin snippet pull — run when the box is back

Where snippets live (from the code): `AssemblyPipeline` writes `<snippet_dir>/<detection_id>.wav`
(mono PCM16 via `utils.audio.write_wav_mono`, at the classification sample rate) and stores the
path in `detections.snippet_path`; `snippet_dir` defaults to `data/snippets`
(`MINIMAPPR_SNIPPET_DIR` / config `snippet_dir`), DB at `data/minimappr.db`
(`MINIMAPPR_DB_PATH`). Snippets expire by `snippet_expires_ns` / retention buckets, so pull soon
after the box comes back. Labels: `detections.label`, `label_category`, `label_confidence`,
`classifier_scores_json` (BirdNET species labels; check the categories present first).

```bash
# 0. tunnel up?
nc -z -G 2 127.0.0.1 2222 && echo up

# 1. confirm paths on the box (config may override the defaults; relative paths are vs ~/MinimapPR)
ssh -p 2222 colin@127.0.0.1 'cd ~/MinimapPR && grep -nE "snippet_dir|db_path" config.yml 2>/dev/null; \
  env | grep -E "MINIMAPPR_(SNIPPET_DIR|DB_PATH)"; ls data/snippets | wc -l; du -sh data/snippets'

# 2. survey what is there (python's sqlite3, since the ostree box may lack the sqlite3 CLI;
#    use the minimappr env's python — mamba prints to stderr, so redirect it)
ssh -p 2222 colin@127.0.0.1 'cd ~/MinimapPR && python3 - 2>/dev/null' <<'EOF'
import sqlite3
db = sqlite3.connect("file:data/minimappr.db?mode=ro", uri=True)
for row in db.execute("""SELECT label_category, label, COUNT(*), ROUND(MAX(label_confidence),2)
                         FROM detections WHERE snippet_path IS NOT NULL
                         GROUP BY 1,2 ORDER BY 3 DESC LIMIT 60"""):
    print(row)
EOF

# 3. export ~20 bird snippets (best per species, distinct species first) + 2 ambience-ish
#    clips into a tarball with a CSV of metadata. Adjust BIRD_CATEGORIES / AMBIENCE_LIKE after
#    step 2.
ssh -p 2222 colin@127.0.0.1 'cd ~/MinimapPR && python3 - 2>/dev/null' <<'EOF'
import csv, os, shutil, sqlite3, tarfile
BIRD_CATEGORIES = ("bird", "bird_like", "animal")
AMBIENCE_LIKE = ("%wind%", "%rustl%", "%outside%", "%rain%", "%silence%")
N_BIRDS, N_AMBIENCE = 20, 2
name = "bluefin_bird_snippets_v1"
out = os.path.expanduser(f"~/field_fixture_export/{name}")
os.makedirs(out, exist_ok=True)
db = sqlite3.connect("file:data/minimappr.db?mode=ro", uri=True)
db.row_factory = sqlite3.Row
cols = "id, timestamp_ns, source_node_id, label, label_category, label_confidence, snippet_path"
rows = db.execute(f"""SELECT {cols} FROM detections WHERE snippet_path IS NOT NULL
    AND label_category IN ({",".join("?"*len(BIRD_CATEGORIES))})
    ORDER BY label_confidence DESC""", BIRD_CATEGORIES).fetchall()
picked, per_label = [], {}
for r in rows:  # first pass: one per species; second pass fills up to N_BIRDS
    if os.path.exists(r["snippet_path"]) and r["label"] not in per_label:
        per_label[r["label"]] = 1; picked.append(("bird", r))
    if len(picked) >= N_BIRDS: break
for r in rows:
    if len(picked) >= N_BIRDS: break
    if os.path.exists(r["snippet_path"]) and all(r["id"] != p[1]["id"] for p in picked):
        picked.append(("bird", r))
amb = db.execute(f"""SELECT {cols} FROM detections WHERE snippet_path IS NOT NULL AND ("""
    + " OR ".join("LOWER(label) LIKE ?" for _ in AMBIENCE_LIKE) + ") ORDER BY timestamp_ns DESC",
    AMBIENCE_LIKE).fetchall()
picked += [("ambience", r) for r in amb if os.path.exists(r["snippet_path"])][:N_AMBIENCE]
with open(os.path.join(out, "snippets.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["role", "file"] + cols.replace(" ", "").split(","))
    for role, r in picked:
        fname = f"{role}_{r['id']}.wav"
        shutil.copy2(r["snippet_path"], os.path.join(out, fname))
        w.writerow([role, fname] + [r[k] for k in r.keys()])
with tarfile.open(out + ".tar.gz", "w:gz") as tar:
    tar.add(out, arcname=name)
print(len(picked), "clips ->", out + ".tar.gz")
EOF

# 4. pull + register locally (the fetch script itself can do the pull once the entry exists)
mkdir -p /tmp/bluefin_pull && rsync -t -e "ssh -p 2222" \
  colin@127.0.0.1:field_fixture_export/bluefin_bird_snippets_v1.tar.gz /tmp/bluefin_pull/
python scripts/fetch_field_fixtures.py register /tmp/bluefin_pull/bluefin_bird_snippets_v1.tar.gz \
  --name bluefin_bird_snippets_v1 --extract \
  --source ssh://colin@127.0.0.1:2222/home/colin/field_fixture_export/bluefin_bird_snippets_v1.tar.gz \
  --description "~20 BirdNET-labelled detection snippets + 2 ambience clips from the Bluefin box, snippets.csv metadata" \
  --licence "private: own site recordings, do not redistribute"
#    paste the printed [[bundle]] table into tests/fixtures/field_bundles.toml (replace the
#    commented example), then:
python scripts/fetch_field_fixtures.py fetch --only bluefin_bird_snippets_v1
python scripts/fetch_field_fixtures.py list
```

Then: listen to every extracted clip for passers-by (site recordings — this is the important
privacy step), drop any with speech, re-tar/re-register if anything was dropped, and set
`speech_checked = true`. Keep the bundle out of git (it is site audio, not CC). Note the Bluefin
source is a tunnel to a home box, not durable storage; the strategy's long-term home for bundles
is a private R2 bucket — when that exists, change `source` to its https URL (sha256 unchanged).

## Open issues

- Human listen-check of the two flagged clips (and of the whole set once, ideally).
- Ambience is NC-licensed; replace if a BY/BY-SA/CC0 forest-wind bed turns up.
- Bluefin pull pending (box down).
- `band_hz` values are hand estimates checked only by "≥ 75 % of power in band" for birds.
