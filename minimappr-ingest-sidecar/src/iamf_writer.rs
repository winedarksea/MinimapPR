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
    pub const SCENE_BASED: u8 = 1; // HOA / ambisonics
    pub const OBJECT_BASED: u8 = 2;
}

mod param_definition_type {
    pub const MIX_GAIN: u8 = 0;
    pub const SINGLE_POSITION: u8 = 3;
}

mod animation_type {
    pub const STEP: u8 = 0;
    pub const LINEAR: u8 = 1;
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
    /// IAMF normalized distance in [0, 1].
    pub distance_norm: f32,
    /// End azimuth for LINEAR animation over this temporal unit.
    pub end_azimuth_deg: Option<f32>,
    /// End elevation for LINEAR animation over this temporal unit.
    pub end_elevation_deg: Option<f32>,
    /// End normalized distance for LINEAR animation over this temporal unit.
    pub end_distance_norm: Option<f32>,
}

/// Per-temporal-unit position data for all objects.
pub type ObjectPositions = HashMap<u32 /* object_id */, ObjectPosition>;

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
        payload.push((audio_element_type::OBJECT_BASED << 5) | 0);
        write_leb128(&mut payload, self.codec_config_id as u64);

        write_leb128(&mut payload, 1u64);
        write_leb128(&mut payload, substream_id as u64);

        write_leb128(&mut payload, 0u64);
        write_leb128(&mut payload, 1u64); // objects_config_size: num_objects only
        payload.push(1); // num_objects

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
        self.write_rendering_config(&mut payload, None);
        // element_mix_config: parameter_id, rate, type=MIX_GAIN(0),
        // default_mix_gain in Q7.8 fixed-point (0 dB = 0x0100 = 256).
        write_leb128(&mut payload, 0u64); // parameter_id
        write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
        payload.push(param_definition_type::MIX_GAIN);
        payload.extend_from_slice(&256u16.to_be_bytes()); // default_mix_gain Q7.8

        // Object elements
        for (idx, &elem_id) in self.object_element_ids.iter().enumerate() {
            write_leb128(&mut payload, elem_id as u64);
            self.write_rendering_config(&mut payload, Some(elem_id));
            write_leb128(&mut payload, (idx + 1) as u64); // parameter_id
            write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
            payload.push(param_definition_type::MIX_GAIN);
            payload.extend_from_slice(&256u16.to_be_bytes());
        }

        // output_mix_config (parameter_id, rate, MIX_GAIN, default)
        let output_mix_param_id = 100u64;
        write_leb128(&mut payload, output_mix_param_id);
        write_leb128(&mut payload, self.scene.sample_rate_hz as u64);
        payload.push(param_definition_type::MIX_GAIN);
        payload.extend_from_slice(&256u16.to_be_bytes());

        // num_layouts = 1 (binaural)
        write_leb128(&mut payload, 1u64);
        payload.push(2); // layout_type = BINAURAL (2)
                         // loudness_info for the single layout
        self.write_loudness_info_aggregate(&mut payload);

        write_obu(obu_type::MIX_PRESENTATION, &payload)
    }

    fn write_rendering_config(&self, out: &mut Vec<u8>, object_element_id: Option<u32>) {
        out.push(0); // headphones_rendering_mode=0, reserved=0
        let mut extension = Vec::new();
        match object_element_id {
            Some(elem_id) => {
                write_leb128(&mut extension, 1u64); // num_parameters
                write_leb128(
                    &mut extension,
                    param_definition_type::SINGLE_POSITION as u64,
                );
                self.write_single_position_param_definition(&mut extension, elem_id);
            }
            None => {
                write_leb128(&mut extension, 0u64); // num_parameters
            }
        }
        write_leb128(out, extension.len() as u64);
        out.extend(extension);
    }

    fn write_single_position_param_definition(&self, out: &mut Vec<u8>, parameter_id: u32) {
        write_leb128(out, parameter_id as u64);
        write_leb128(out, self.scene.sample_rate_hz as u64);
        out.push(0); // param_definition_mode=0, reserved=0
        write_leb128(out, self.scene.samples_per_frame as u64);
        write_leb128(out, self.scene.samples_per_frame as u64);
        pack_single_position_triplet(out, 0, 0, 0);
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

        let end = ObjectPosition {
            azimuth_deg: pos.end_azimuth_deg.unwrap_or(pos.azimuth_deg),
            elevation_deg: pos.end_elevation_deg.unwrap_or(pos.elevation_deg),
            distance_norm: pos.end_distance_norm.unwrap_or(pos.distance_norm),
            end_azimuth_deg: None,
            end_elevation_deg: None,
            end_distance_norm: None,
        };
        payload.extend(single_position_parameter_data(pos, &end));

        write_obu(obu_type::PARAMETER_BLOCK, &payload)
    }
}

fn single_position_parameter_data(start: &ObjectPosition, end: &ObjectPosition) -> Vec<u8> {
    let same = (start.azimuth_deg - end.azimuth_deg).abs() < 0.5
        && (start.elevation_deg - end.elevation_deg).abs() < 0.5
        && (start.distance_norm - end.distance_norm).abs() < 0.01;
    let animation = if same {
        animation_type::STEP
    } else {
        animation_type::LINEAR
    };
    let mut body = Vec::new();
    write_leb128(&mut body, animation as u64);
    pack_single_position_triplet(
        &mut body,
        quantize_azimuth(start.azimuth_deg),
        quantize_elevation(start.elevation_deg),
        quantize_distance(start.distance_norm),
    );
    if animation == animation_type::LINEAR {
        pack_single_position_triplet(
            &mut body,
            quantize_azimuth(end.azimuth_deg),
            quantize_elevation(end.elevation_deg),
            quantize_distance(end.distance_norm),
        );
    }
    let mut out = Vec::new();
    write_leb128(&mut out, body.len() as u64);
    out.extend(body);
    out
}

fn quantize_azimuth(value: f32) -> i16 {
    value.clamp(-180.0, 180.0).round() as i16
}

fn quantize_elevation(value: f32) -> i8 {
    value.clamp(-90.0, 90.0).round() as i8
}

fn quantize_distance(value: f32) -> u8 {
    let quantized = (value.clamp(0.0, 1.0) * 7.0).round();
    quantized.clamp(0.0, 7.0) as u8
}

fn pack_single_position_triplet(out: &mut Vec<u8>, azimuth: i16, elevation: i8, distance: u8) {
    let mut bits = BitWriter::default();
    bits.write_signed(azimuth as i32, 9);
    bits.write_signed(elevation as i32, 8);
    bits.write_unsigned(distance as u32, 3);
    bits.write_unsigned(0, 4);
    out.extend(bits.finish());
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used_bits: u8,
}

impl BitWriter {
    fn write_unsigned(&mut self, value: u32, n_bits: u8) {
        for bit_idx in (0..n_bits).rev() {
            let bit = ((value >> bit_idx) & 1) as u8;
            self.current = (self.current << 1) | bit;
            self.used_bits += 1;
            if self.used_bits == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.used_bits = 0;
            }
        }
    }

    fn write_signed(&mut self, value: i32, n_bits: u8) {
        let mask = (1i32 << n_bits) - 1;
        self.write_unsigned((value & mask) as u32, n_bits);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used_bits > 0 {
            self.current <<= 8 - self.used_bits;
            self.bytes.push(self.current);
        }
        self.bytes
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
                distance_norm: 0.5,
                end_azimuth_deg: None,
                end_elevation_deg: None,
                end_distance_norm: None,
            },
        );

        let unit = writer.write_temporal_unit(&silence_bed, &silence_obj, &positions);
        assert!(!unit.is_empty());
        // First byte of temporal unit must be Temporal_Delimiter OBU header.
        let expected_td = (obu_type::TEMPORAL_DELIMITER & 0x1F) << 3;
        assert_eq!(unit[0], expected_td);
    }

    #[test]
    fn object_audio_element_uses_objects_config() {
        let (scene, n) = make_scene(1);
        let writer = IamfWriter::new(scene, n);
        let obu = writer.object_audio_element_obu(2, 4);
        let payload = obu_payload(&obu);
        let (element_id, consumed) = read_leb128(payload).unwrap();
        assert_eq!(element_id, 2);
        let audio_element_type_byte = payload[consumed];
        assert_eq!(
            audio_element_type_byte >> 5,
            audio_element_type::OBJECT_BASED
        );

        let objects_config = &payload[payload.len() - 2..];
        assert_eq!(objects_config[0], 1, "objects_config_size must be 1 byte");
        assert_eq!(objects_config[1], 1, "YouTube path emits one mono object");
    }

    #[test]
    fn temporal_unit_writes_position_parameter_before_audio_frames() {
        let (scene, n) = make_scene(1);
        let writer = IamfWriter::new(scene, n);
        let spf = 512usize;
        let silence_bed: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; spf * 2]);
        let silence_obj = vec![vec![0u8; spf * 2]];
        let mut positions = ObjectPositions::new();
        positions.insert(
            0,
            ObjectPosition {
                azimuth_deg: 10.0,
                elevation_deg: 5.0,
                distance_norm: 0.25,
                end_azimuth_deg: Some(20.0),
                end_elevation_deg: Some(10.0),
                end_distance_norm: Some(0.35),
            },
        );

        let unit = writer.write_temporal_unit(&silence_bed, &silence_obj, &positions);
        let mut offset = skip_obu(&unit, 0);
        assert_eq!(unit[0] >> 3, obu_type::TEMPORAL_DELIMITER);
        assert_eq!(unit[offset] >> 3, obu_type::PARAMETER_BLOCK);
        let payload = obu_payload(&unit[offset..]);
        let (_parameter_id, consumed) = read_leb128(payload).unwrap();
        let (_param_size, size_consumed) = read_leb128(&payload[consumed..]).unwrap();
        let (animation, _) = read_leb128(&payload[consumed + size_consumed..]).unwrap();
        assert_eq!(animation as u8, animation_type::LINEAR);
        offset = skip_obu(&unit, offset);
        assert_eq!(unit[offset] >> 3, obu_type::AUDIO_FRAME_EXPLICIT_ID);
    }

    #[test]
    fn single_position_parameter_data_uses_linear_for_motion() {
        let start = ObjectPosition {
            azimuth_deg: 0.0,
            elevation_deg: 0.0,
            distance_norm: 0.1,
            end_azimuth_deg: None,
            end_elevation_deg: None,
            end_distance_norm: None,
        };
        let end = ObjectPosition {
            azimuth_deg: 30.0,
            elevation_deg: 5.0,
            distance_norm: 0.6,
            end_azimuth_deg: None,
            end_elevation_deg: None,
            end_distance_norm: None,
        };
        let data = single_position_parameter_data(&start, &end);
        let (_size, consumed) = read_leb128(&data).unwrap();
        let (animation, _) = read_leb128(&data[consumed..]).unwrap();
        assert_eq!(animation as u8, animation_type::LINEAR);
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
        assert_eq!(words[4], 32767); // clamped +1.0 → 32767
    }

    fn obu_payload(obu: &[u8]) -> &[u8] {
        let (_size, consumed) = read_leb128(&obu[1..]).unwrap();
        &obu[1 + consumed..]
    }

    fn skip_obu(obus: &[u8], offset: usize) -> usize {
        let (size, consumed) = read_leb128(&obus[offset + 1..]).unwrap();
        offset + 1 + consumed + size as usize
    }
}
