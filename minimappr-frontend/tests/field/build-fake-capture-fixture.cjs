// Builds the WAV Chrome plays as its fake microphone
// (--use-file-for-fake-audio-capture) for the Field PWA browser checks:
// a quiet forest-ambience lead-in (Stage-1 needs a noise floor before the
// call stands out of it) followed by an American Robin song, with the
// ambience continuing underneath. Chrome loops the file.
// Sources: committed fixtures in tests/fixtures/field_audio (see ATTRIBUTION).

const fs = require("node:fs");
const path = require("node:path");

const FIXTURE_DIRECTORY = path.resolve(__dirname, "../../../tests/fixtures/field_audio");
const AMBIENCE_FILE = "ambience_forest_breeze_xc980972.wav";
const BIRD_FILE = "amro_song_xc132864.wav";
const LEAD_IN_SECONDS = 6;
const TAIL_SECONDS = 3;
const SAMPLE_RATE_HZ = 48000;
const WAV_HEADER_BYTES = 44;
const PCM16_MAXIMUM = 32767;
const PCM16_MINIMUM = -32768;

// Committed fixtures are canonical 44-byte-header PCM16 mono (fixtures.md).
function readPcm16Mono(fileName) {
  const bytes = fs.readFileSync(path.join(FIXTURE_DIRECTORY, fileName));
  const sampleCount = (bytes.length - WAV_HEADER_BYTES) / 2;
  const samples = new Int16Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) samples[index] = bytes.readInt16LE(WAV_HEADER_BYTES + index * 2);
  return samples;
}

function wavBytes(samples) {
  const header = Buffer.alloc(WAV_HEADER_BYTES);
  const dataBytes = samples.length * 2;
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + dataBytes, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(SAMPLE_RATE_HZ, 24);
  header.writeUInt32LE(SAMPLE_RATE_HZ * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(dataBytes, 40);
  const body = Buffer.alloc(dataBytes);
  samples.forEach((sample, index) => body.writeInt16LE(sample, index * 2));
  return Buffer.concat([header, body]);
}

function buildFakeCaptureFixture(outputPath) {
  const ambience = readPcm16Mono(AMBIENCE_FILE);
  const bird = readPcm16Mono(BIRD_FILE);
  const leadIn = LEAD_IN_SECONDS * SAMPLE_RATE_HZ;
  const total = leadIn + bird.length + TAIL_SECONDS * SAMPLE_RATE_HZ;
  const mixed = new Int16Array(total);
  for (let index = 0; index < total; index += 1) {
    const birdIndex = index - leadIn;
    const birdSample = birdIndex >= 0 && birdIndex < bird.length ? bird[birdIndex] : 0;
    const sum = ambience[index % ambience.length] + birdSample;
    mixed[index] = Math.max(PCM16_MINIMUM, Math.min(PCM16_MAXIMUM, sum));
  }
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, wavBytes(mixed));
  return outputPath;
}

module.exports = { buildFakeCaptureFixture };

if (require.main === module) {
  console.log(buildFakeCaptureFixture(process.argv[2] || path.resolve(__dirname, "../../test-results/field-fake-capture.wav")));
}
