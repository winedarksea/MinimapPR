/// Custom IAMF v1.0.0 OBU bitstream writer.
///
/// Implements the Open Bitstream Unit (OBU) framing from the IAMF specification
/// without depending on the iamf-tools C++ build chain, keeping the binary
/// portable to Raspberry Pi.
///
/// Supported layout:
///   • One Codec_Config_OBU (ipcm — integer PCM, 16-bit, caller-specified rate)
///   • One Audio_Element_OBU for the 4-channel FOA ambisonic bed (ACN/SN3D)
///   • N Audio_Element_OBUs for isolated mono object tracks
///   • One Mix_Presentation_OBU with per-element loudness_info (BS.1770-4)
///   • Temporal units: Temporal_Delimiter + Parameter_Block(s) + Audio_Frame(s)
///
/// Reference:
///   https://aomediacodec.github.io/iamf/
use std::collections::HashMap;

/// OBU type codes (5-bit field in the OBU header byte).
mod obu_type {
    pub const IA_SEQUENCE_HEADER: u8 = 0;
    pub const TEMPORAL_DELIMITER: u8 = 1;
    pub const CODEC_CONFIG: u8 = 2;
    pub const AUDIO_ELEMENT: u8 = 3;
    pub const MIX_PRESENTATION: u8 = 4;
    pub const PARAMETER_BLOCK: u8 = 5;
    pub const AUDIO_FRAME_EXPLICIT_ID: u8 = 6;
}

/// Audio element types.
mod audio_element_type {
    pub const CHANNEL_BASED: u8 = 0;
    pub const SCENE_BASED: u8 = 1; // HOA / ambisonics
}

/// First-order ambisonics: W X Y Z (ACN order).
pub const FOA_CHANNEL_COUNT: u8 = 4;

// ── public data types ─────────────────────────────────────────────────────────

/// Loudness metadata measured via BS.1770-4.
#[derive(Clone, Debug, Default)]
pub struct LoudnessInfo {
    /// Integrated LUFS (negative float, e.g. −14.3).
    pub integrated_loudness_lufs: f32,
    /// True peak dBFS.
    pub true_peak_dbfs: f32,
}

/// Position of one object in the mix at one point in time.
#[derive(Clone, Debug)]
pub struct ObjectPosition {
    /// Azimuth in degrees [−180, 180].
    pub azimuth_deg: f32,
    /// Elevation in degrees [−90, 90].
    pub elevation_deg: f32,
    /// Distance in metres.
    pub distance_m: f32,
}

/// Per-temporal-unit position data for all objects.
pub type ObjectPositions = HashMap<u32 /* object_id */, ObjectPosition>;

/// An audio frame for one substream in one temporal unit.
pub struct AudioFrameData<'a> {
    pub substream_id: u32,
    /// Interleaved PCM samples, 16-bit little-endian.
    pub pcm_bytes: &'a [u8],
}

/// Complete scene description passed to the writer.
pub struct IamfScene {
    pub sample_rate_hz: u32,
    /// Number of samples per codec frame (e.g. 512).
    pub samples_per_frame: u32,
    /// BS.1770-4 loudness for the FOA bed (measured on W channel).
    pub bed_loudness: LoudnessInfo,
    /// Per-object loudness, indexed by object_id (0-based).
    pub object_loudness: Vec<LoudnessInfo>,
}

// ── writer ────────────────────────────────────────────────────────────────────

pub struct IamfWriter {
    scene: IamfScene,
    /// codec_config_id assigned to ipcm.
    codec_config_id: u32,
    /// audio_element_id for the FOA bed.
    bed_element_id: u32,
    /// audio_element_id for each object (index = object index).
    object_element_ids: Vec<u32>,
    /// substream_id for each channel of the FOA bed (4 substreams).
    bed_substream_ids: [u32; 4],
    /// substream_id for each mono object.
    object_substream_ids: Vec<u32>,
    /// mix_presentation_id.
    mix_presentation_id: u32,
}

impl IamfWriter {
    pub fn new(scene: IamfScene, n_objects: usize) -> Self {
        let n_objs = n_objects;
        // Assign IDs sequentially.
        let codec_config_id = 0;
        let bed_element_id = 1;
        let first_obj_element = 2u32;
        let object_element_ids: Vec<u32> =
            (first_obj_element..first_obj_element + n_objs as u32).collect();

        // Substream IDs: bed uses 0..3, objects use 4..4+N.
        let bed_substream_ids = [0, 1, 2, 3];
        let object_substream_ids: Vec<u32> = (4u32..4 + n_objs as u32).collect();
        let mix_presentation_id = first_obj_element + n_objs as u32;

        Self {
            scene,
            codec_config_id,
            bed_element_id,
            object_element_ids,
            bed_substream_ids,
            object_substream_ids,
            mix_presentation_id,
        }
    }

    /// Emit the static descriptor OBUs that appear once at the start of the
    /// bitstream (IA_Sequence_Header, Codec_Config, Audio_Elements,
    /// Mix_Presentation).
    pub fn write_descriptor_obus(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(self.ia_sequence_header());
        out.extend(self.codec_config_obu());
        out.extend(self.bed_audio_element_obu());
        for (idx, &elem_id) in self.object_element_ids.iter().enumerate() {
            out.extend(self.object_audio_element_obu(elem_id, self.object_substream_ids[idx]));
        }
        out.extend(self.mix_presentation_obu());
        out
    }

    /// Emit one temporal unit given:
    /// - PCM bytes for each bed substream (W, X, Y, Z)
    /// - PCM bytes for each object mono substream
    /// - spatial positions for each object in this unit
    pub fn write_temporal_unit(
        &self,
        bed_frames: &[Vec<u8>; 4],
        object_frames: &[Vec<u8>],
        positions: &ObjectPositions,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(write_obu(obu_type::TEMPORAL_DELIMITER, &[]));

        // Parameter blocks for object positions.
        for (obj_idx, &elem_id) in self.object_element_ids.iter().enumerate() {
            if let Some(pos) = positions.get(&(obj_idx as u32)) {
                out.extend(self.object_parameter_block(elem_id, pos));
            }
        }

        // Bed audio frames.
        for (i, frame) in bed_frames.iter().enumerate() {
            out.extend(audio_frame_obu(self.bed_substream_ids[i], frame));
        }

        // Object audio frames.
        for (i, frame) in object_frames.iter().enumerate() {
            if let Some(&ss_id) = self.object_substream_ids.get(i) {
                out.extend(audio_frame_obu(ss_id, frame));
            }
        }

        out
    }

    // ── OBU builders ─────────────────────────────────────────────────────────

    fn ia_sequence_header(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(b"iamf"); // ia_code
        payload.push(0); // primary_profile = Base (0)
        payload.push(0); // additional_profile = Base (0)
        write_obu(obu_type::IA_SEQUENCE_HEADER, &payload)
    }

    fn codec_config_obu(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        write_leb128(&mut payload, self.codec_config_id as u64);
        payload.extend_from_slice(b"ipcm"); // codec_id

        // num_samples_per_frame
        write_leb128(&mut payload, self.scene.samples_per_frame as u64);
        // audio_roll_distance (i16 big-endian) — 0 for PCM
        payload.extend_from_slice(&0i16.to_be_bytes());

        // decoder_config_ipcm:
        //   sample_format_flags u8: bit0=0 (signed), others 0
        payload.push(0);
        //   sample_size u8: 16
        payload.push(16);
        //   sample_rate u32 big-endian
        payload.extend_from_slice(&self.scene.sample_rate_hz.to_be_bytes());

        write_obu(obu_type::CODEC_CONFIG, &payload)
    }

    fn bed_audio_element_obu(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        write_leb128(&mut payload, self.bed_element_id as u64);
        // audio_element_type (3 bits) | reserved (5 bits)
        payload.push((audio_element_type::SCENE_BASED << 5) | 0);
        write_leb128(&mut payload, self.codec_config_id as u64);

        // num_substreams = 4 (one per B-format channel)
        write_leb128(&mut payload, FOA_CHANNEL_COUNT as u64);
        for &ss_id in &self.bed_substream_ids {
            write_leb128(&mut payload, ss_id as u64);
        }

        // num_parameters = 0 (no per-element parameter definitions for the bed)
        write_leb128(&mut payload, 0u64);

        // ambisonics_config: MONO_PROJECTION (mode 0)
        payload.push(0); // ambisonics_mode = MONO_PROJECTION
        payload.push(FOA_CHANNEL_COUNT); // output_channel_count
        payload.push(FOA_CHANNEL_COUNT); // substream_count
        // channel_mapping: ACN order (0=W, 1=X, 2=Y, 3=Z) mapped to substreams 0..3
        for i in 0..FOA_CHANNEL_COUNT {
            payload.push(i);
        }

        write_obu(obu_type::AUDIO_ELEMENT, &payload)
    }

    fn object_audio_element_obu(&self, element_id: u32, substream_id: u32) -> Vec<u8> {
        let mut payload = Vec::new();
        write_leb128(&mut payload, element_id as u64);
        // audio_element_type CHANNEL_BASED (0) | reserved
        payload.push((audio_element_type::CHANNEL_BASED << 5) | 0);
        write_leb128(&mut payload, self.codec_config_id as u64);

        // num_substreams = 1 (mono)
        write_leb128(&mut payload, 1u64);
        write_leb128(&mut payload, substream_id as u64);

        // num_parameters = 1 (one parameter definition for object position)
        write_leb128(&mut payload, 1u64);
        // parameter_id, parameter_rate, param_definition_type (demixing=1)
        write_leb128(&mut payload, element_id as u64); // parameter_id = same as element_id
        write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
        payload.push(3); // param_definition_type = OBJECT_BASED (3)
        // default_demix_mode/weight — 0
        payload.push(0);

        // scalable_channel_layout_config: 1 layer, 1 ch mono
        payload.push(0); // num_layers_minus1 = 0
        payload.push(1); // loudspeaker_layout = MONO (1)
        payload.push(0); // output_gain_flags
        payload.push(1); // substream_count = 1
        payload.push(0); // coupled_substream_count = 0

        write_obu(obu_type::AUDIO_ELEMENT, &payload)
    }

    fn mix_presentation_obu(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        write_leb128(&mut payload, self.mix_presentation_id as u64);

        // count_label
        write_leb128(&mut payload, 1u64);
        // language_label[0] as null-terminated string
        payload.extend_from_slice(b"en-US\0");

        // num_sub_mixes = 1
        write_leb128(&mut payload, 1u64);

        // --- sub_mix ---
        // num_audio_elements
        let n_elements = 1 + self.object_element_ids.len();
        write_leb128(&mut payload, n_elements as u64);

        // FOA bed element
        write_leb128(&mut payload, self.bed_element_id as u64);
        // rendering_config.headphones_rendering_mode = STEREO (0)
        payload.push(0);
        // element_mix_config: parameter_id, rate, type=MIX_GAIN(0),
        // default_mix_gain in Q7.8 fixed-point (0 dB = 0x0100 = 256).
        write_leb128(&mut payload, 0u64); // parameter_id
        write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
        payload.push(0); // param_definition_type = MIX_GAIN
        payload.extend_from_slice(&256u16.to_be_bytes()); // default_mix_gain Q7.8

        // Object elements
        for (idx, &elem_id) in self.object_element_ids.iter().enumerate() {
            write_leb128(&mut payload, elem_id as u64);
            payload.push(0); // rendering_config
            write_leb128(&mut payload, (idx + 1) as u64); // parameter_id
            write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
            payload.push(0); // MIX_GAIN
            payload.extend_from_slice(&256u16.to_be_bytes());
        }

        // output_mix_config (parameter_id, rate, MIX_GAIN, default)
        let output_mix_param_id = 100u64;
        write_leb128(&mut payload, output_mix_param_id);
        write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
        payload.push(0);
        payload.extend_from_slice(&256u16.to_be_bytes());

        // num_layouts = 1 (binaural)
        write_leb128(&mut payload, 1u64);
        payload.push(2); // layout_type = BINAURAL (2)
        // loudness_info for the single layout
        self.write_loudness_info_aggregate(&mut payload);

        write_obu(obu_type::MIX_PRESENTATION, &payload)
    }

    /// Aggregate loudness across bed + objects for the Mix_Presentation.
    fn write_loudness_info_aggregate(&self, out: &mut Vec<u8>) {
        // Use the bed W-channel measurement as the integrated loudness reference.
        let integrated = self.scene.bed_loudness.integrated_loudness_lufs;
        let true_peak = self
            .scene
            .object_loudness
            .iter()
            .chain(std::iter::once(&self.scene.bed_loudness))
            .map(|l| l.true_peak_dbfs)
            .fold(f32::NEG_INFINITY, f32::max);

        // Q7.8 fixed-point encoding.
        let integrated_q78 = (integrated * 256.0) as i16;
        let true_peak_q78 = (true_peak * 256.0) as i16;

        // info_type flags: 0 = measured loudness only
        out.push(0);
        out.extend_from_slice(&integrated_q78.to_be_bytes());
        out.extend_from_slice(&true_peak_q78.to_be_bytes());
    }

    fn object_parameter_block(&self, element_id: u32, pos: &ObjectPosition) -> Vec<u8> {
        let mut payload = Vec::new();
        // parameter_id matches the one registered in the Audio_Element_OBU
        write_leb128(&mut payload, element_id as u64);

        // duration and constant_subblock_duration = samples_per_frame
        write_leb128(&mut payload, self.scene.samples_per_frame as u64);
        write_leb128(&mut payload, self.scene.samples_per_frame as u64);

        // One subblock: object metadata.
        // azimuth Q0.8 (degrees, range -180..180), elevation Q0.8, distance Q8.8
        let az_i8 = pos.azimuth_deg.clamp(-128.0, 127.0) as i8;
        let el_i8 = pos.elevation_deg.clamp(-128.0, 127.0) as i8;
        let dist_q88 = (pos.distance_m.max(0.0) * 256.0).min(65535.0) as u16;
        payload.push(az_i8 as u8);
        payload.push(el_i8 as u8);
        payload.extend_from_slice(&dist_q88.to_be_bytes());

        write_obu(obu_type::PARAMETER_BLOCK, &payload)
    }
}

// ── audio frame OBU ───────────────────────────────────────────────────────────

fn audio_frame_obu(substream_id: u32, pcm_bytes: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + pcm_bytes.len());
    write_leb128(&mut payload, substream_id as u64);
    payload.extend_from_slice(pcm_bytes);
    write_obu(obu_type::AUDIO_FRAME_EXPLICIT_ID, &payload)
}

// ── low-level OBU framing helpers ─────────────────────────────────────────────

/// Encode a complete OBU: 1-byte header + LEB128 size + payload.
fn write_obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + payload.len());
    // Header byte: [obu_type: 5 bits | redundant_copy: 1 | trimming_status: 1 | extension: 1]
    out.push((obu_type & 0x1F) << 3);
    write_leb128(&mut out, payload.len() as u64);
    out.extend_from_slice(payload);
    out
}

/// Unsigned LEB128 varint encoding.
pub fn write_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Decode unsigned LEB128 from a byte slice, returning (value, bytes_consumed).
pub fn read_leb128(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in bytes.iter().enumerate() {
        if shift >= 63 {
            return None; // overflow guard
        }
        value |= (u64::from(byte & 0x7F)) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None // truncated
}

// ── high-level convenience: build a full IAMF file from WAV-like inputs ────────

/// Helper to encode one channel of 32-bit float PCM into 16-bit little-endian
/// PCM bytes suitable for ipcm audio frames.
pub fn f32_to_pcm16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let pcm = (clamped * 32767.0) as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    out
}

/// Convert 4-channel interleaved PCM into per-substream byte frames for the
/// bed, split into `samples_per_frame`-sample chunks.
/// Returns `(bed_frame_chunks, n_frames)` where `bed_frame_chunks[ch]` is a
/// `Vec<Vec<u8>>` (one inner Vec per temporal unit).
pub fn split_bed_into_frames(
    channels_first: &[Vec<f32>; 4],
    samples_per_frame: usize,
) -> [Vec<Vec<u8>>; 4] {
    let n_samples = channels_first[0].len();
    let n_frames = (n_samples + samples_per_frame - 1) / samples_per_frame;
    std::array::from_fn(|ch| {
        (0..n_frames)
            .map(|fi| {
                let start = fi * samples_per_frame;
                let end = (start + samples_per_frame).min(n_samples);
                let mut frame: Vec<f32> = channels_first[ch][start..end].to_vec();
                // Zero-pad to full frame size.
                frame.resize(samples_per_frame, 0.0);
                f32_to_pcm16le(&frame)
            })
            .collect()
    })
}

/// Split a mono object track into per-frame byte chunks.
pub fn split_object_into_frames(samples: &[f32], samples_per_frame: usize) -> Vec<Vec<u8>> {
    let n_samples = samples.len();
    let n_frames = (n_samples + samples_per_frame - 1) / samples_per_frame;
    (0..n_frames)
        .map(|fi| {
            let start = fi * samples_per_frame;
            let end = (start + samples_per_frame).min(n_samples);
            let mut frame = samples[start..end].to_vec();
            frame.resize(samples_per_frame, 0.0);
            f32_to_pcm16le(&frame)
        })
        .collect()
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(n_objects: usize) -> (IamfScene, usize) {
        let scene = IamfScene {
            sample_rate_hz: 48_000,
            samples_per_frame: 512,
            bed_loudness: LoudnessInfo {
                integrated_loudness_lufs: -20.0,
                true_peak_dbfs: -3.0,
            },
            object_loudness: (0..n_objects)
                .map(|_| LoudnessInfo {
                    integrated_loudness_lufs: -18.0,
                    true_peak_dbfs: -6.0,
                })
                .collect(),
        };
        (scene, n_objects)
    }

    #[test]
    fn leb128_round_trip() {
        for value in [0u64, 1, 127, 128, 255, 300, 16383, 16384, u32::MAX as u64] {
            let mut buf = Vec::new();
            write_leb128(&mut buf, value);
            let (decoded, _) = read_leb128(&buf).unwrap();
            assert_eq!(decoded, value, "round-trip failed for {value}");
        }
    }

    #[test]
    fn descriptor_obus_not_empty() {
        let (scene, n) = make_scene(2);
        let writer = IamfWriter::new(scene, n);
        let obus = writer.write_descriptor_obus();
        assert!(!obus.is_empty());
        // First byte must be the IA_Sequence_Header OBU header.
        let expected_header = (obu_type::IA_SEQUENCE_HEADER & 0x1F) << 3;
        assert_eq!(obus[0], expected_header);
        // Must contain 'iamf' magic somewhere in the first 32 bytes.
        assert!(
            obus[..32.min(obus.len())].windows(4).any(|w| w == b"iamf"),
            "iamf magic bytes not found in descriptor header region"
        );
    }

    #[test]
    fn temporal_unit_roundtrip_structure() {
        let (scene, n) = make_scene(1);
        let writer = IamfWriter::new(scene, n);
        let spf = 512usize;

        let silence_bed: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; spf * 2]);
        let silence_obj = vec![vec![0u8; spf * 2]];
        let mut positions = ObjectPositions::new();
        positions.insert(
            0,
            ObjectPosition {
                azimuth_deg: 45.0,
                elevation_deg: 0.0,
                distance_m: 5.0,
            },
        );

        let unit = writer.write_temporal_unit(&silence_bed, &silence_obj, &positions);
        assert!(!unit.is_empty());
        // First byte of temporal unit must be Temporal_Delimiter OBU header.
        let expected_td = (obu_type::TEMPORAL_DELIMITER & 0x1F) << 3;
        assert_eq!(unit[0], expected_td);
    }

    #[test]
    fn full_bitstream_has_increasing_bytes() {
        let (scene, n) = make_scene(0);
        let writer = IamfWriter::new(scene, n);

        let mut bitstream = writer.write_descriptor_obus();

        let silence_bed: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; 512 * 2]);
        let unit = writer.write_temporal_unit(&silence_bed, &[], &ObjectPositions::new());
        bitstream.extend(unit);

        // Very basic sanity: total length is at least the sum of a few fixed headers.
        assert!(bitstream.len() > 100, "bitstream suspiciously short");
    }

    #[test]
    fn f32_to_pcm16le_clamps() {
        let samples = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let bytes = f32_to_pcm16le(&samples);
        assert_eq!(bytes.len(), samples.len() * 2);
        let words: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(words[0], -32767); // clamped −1.0 → -32767
        assert_eq!(words[4], 32767);  // clamped +1.0 → 32767
    }
}
