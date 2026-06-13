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
///
/// NOT ON THE PRODUCTION PATH for object positions. The OBJECT_BASED /
/// SINGLE_POSITION pieces (`audio_element_type::OBJECT_BASED`,
/// `param_definition_type::SINGLE_POSITION`, `ObjectPosition`,
/// `object_parameter_block`) encode a draft IAMF extension added to the spec
/// *after* v1.1.0. ffmpeg has no muxer/demuxer support for it as of 8.0.1,
/// and real `.iamf` files produced by this writer fail to parse with
/// ffmpeg's IAMF demuxer outright. The Python `IamfPipeline` does not call
/// the `/api/v1/capture/encode/iamf` endpoint backed by this writer; the
/// shipped 2-element export (FOA bed + best-active-object mono, no embedded
/// position automation) is produced ffmpeg-side via `_encode_iamf_ffmpeg()`
/// in `minimappr/core/iamf_pipeline.py`. This writer is kept as a reference
/// implementation for if/when ffmpeg and YouTube add OBJECT_BASED /
/// SINGLE_POSITION support; see `_ffmpeg_supports_iamf_object_based()` in
/// `iamf_pipeline.py`.
use std::collections::HashMap;

/// OBU type codes (5-bit field in the OBU header byte).
mod obu_type {
    pub const CODEC_CONFIG: u8 = 0;
    pub const AUDIO_ELEMENT: u8 = 1;
    pub const MIX_PRESENTATION: u8 = 2;
    pub const PARAMETER_BLOCK: u8 = 3;
    pub const TEMPORAL_DELIMITER: u8 = 4;
    pub const AUDIO_FRAME_EXPLICIT_ID: u8 = 5;
    pub const AUDIO_FRAME_ID0: u8 = 6;
    pub const IA_SEQUENCE_HEADER: u8 = 31;
}

/// Audio element types.
mod audio_element_type {
    pub const SCENE_BASED: u8 = 1; // HOA / ambisonics
    pub const OBJECT_BASED: u8 = 2;
}

mod param_definition_type {
    pub const SINGLE_POSITION: u8 = 3;
}

mod animation_type {
    pub const STEP: u8 = 0;
    pub const LINEAR: u8 = 1;
}

/// First-order ambisonics: W X Y Z (ACN order).
pub const FOA_CHANNEL_COUNT: u8 = 4;
const BED_MIX_GAIN_PARAMETER_ID: u32 = 100;
const OBJECT_MIX_GAIN_PARAMETER_ID: u32 = 101;
const OUTPUT_MIX_GAIN_PARAMETER_ID: u32 = 200;
const OBJECT_POSITION_PARAMETER_ID: u32 = 300;

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
///
/// Experimental / future-spec-version — see module doc comment.
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
    /// Number of channels in the ambisonic bed.  Defaults to `FOA_CHANNEL_COUNT` (4).
    /// Set to 4 for standard First-Order Ambisonics; other values for arbitrary arrays.
    pub n_bed_channels: u8,
    /// BS.1770-4 loudness for the FOA bed (measured on W channel).
    pub bed_loudness: LoudnessInfo,
    /// Per-object loudness, indexed by object_id (0-based).
    pub object_loudness: Vec<LoudnessInfo>,
    /// Default coordinate per object, indexed by object_id (0-based).
    pub object_position_defaults: Vec<ObjectPosition>,
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
    /// substream_id for each channel of the bed (one per B-format channel).
    bed_substream_ids: Vec<u32>,
    /// substream_id for each mono object.
    object_substream_ids: Vec<u32>,
    /// mix_presentation_id.
    mix_presentation_id: u32,
}

impl IamfWriter {
    pub fn new(scene: IamfScene, n_objects: usize) -> Self {
        let n_objs = n_objects;
        let n_bed = scene.n_bed_channels.max(1) as u32;
        // Assign IDs sequentially.
        let codec_config_id = 0;
        let bed_element_id = 1;
        let first_obj_element = 2u32;
        let object_element_ids: Vec<u32> =
            (first_obj_element..first_obj_element + n_objs as u32).collect();

        // Substream IDs: bed uses 0..n_bed, objects use n_bed..n_bed+N.
        let bed_substream_ids: Vec<u32> = (0..n_bed).collect();
        let object_substream_ids: Vec<u32> = (n_bed..n_bed + n_objs as u32).collect();
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
    /// - PCM bytes for each bed substream (one per B-format channel)
    /// - PCM bytes for each object mono substream
    /// - spatial positions for each object in this unit
    pub fn write_temporal_unit(
        &self,
        bed_frames: &[Vec<u8>],
        object_frames: &[Vec<u8>],
        positions: &ObjectPositions,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(write_obu(obu_type::TEMPORAL_DELIMITER, &[]));

        // Parameter blocks for object positions.
        for obj_idx in 0..self.object_element_ids.len() {
            let fallback = self.scene.object_position_defaults.get(obj_idx);
            let pos = positions
                .get(&(obj_idx as u32))
                .or(fallback)
                .cloned()
                .unwrap_or_else(default_object_position);
            out.extend(self.object_parameter_block(&pos));
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
        payload.push(1); // primary_profile = Base
        payload.push(1); // additional_profile = Base
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
        //   sample_format_flags u8: bit0=1 (little endian signed PCM)
        payload.push(1);
        //   sample_size u8: 16
        payload.push(16);
        //   sample_rate u32 big-endian
        payload.extend_from_slice(&self.scene.sample_rate_hz.to_be_bytes());

        write_obu(obu_type::CODEC_CONFIG, &payload)
    }

    fn bed_audio_element_obu(&self) -> Vec<u8> {
        let n_bed = self.scene.n_bed_channels;
        let mut payload = Vec::new();
        write_leb128(&mut payload, self.bed_element_id as u64);
        // audio_element_type (3 bits) | reserved (5 bits)
        payload.push(audio_element_type::SCENE_BASED << 5);
        write_leb128(&mut payload, self.codec_config_id as u64);

        // num_substreams = n_bed (one per B-format channel)
        write_leb128(&mut payload, n_bed as u64);
        for &ss_id in &self.bed_substream_ids {
            write_leb128(&mut payload, ss_id as u64);
        }

        // num_parameters = 0 (no per-element parameter definitions for the bed)
        write_leb128(&mut payload, 0u64);

        // ambisonics_config: MONO_PROJECTION (mode 0)
        payload.push(0); // ambisonics_mode = MONO_PROJECTION
        payload.push(n_bed); // output_channel_count
        payload.push(n_bed); // substream_count
                             // channel_mapping: ACN order mapped to substreams 0..n_bed
        for i in 0..n_bed {
            payload.push(i);
        }

        write_obu(obu_type::AUDIO_ELEMENT, &payload)
    }

    fn object_audio_element_obu(&self, element_id: u32, substream_id: u32) -> Vec<u8> {
        let mut payload = Vec::new();
        write_leb128(&mut payload, element_id as u64);
        payload.push(audio_element_type::OBJECT_BASED << 5);
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

        // count_label = 0: no language labels or localized annotations.
        write_leb128(&mut payload, 0u64);

        // num_sub_mixes = 1
        write_leb128(&mut payload, 1u64);

        // --- sub_mix ---
        // num_audio_elements
        let n_elements = 1 + self.object_element_ids.len();
        write_leb128(&mut payload, n_elements as u64);

        // FOA bed element
        write_leb128(&mut payload, self.bed_element_id as u64);
        self.write_rendering_config(&mut payload, None);
        self.write_mix_gain_param_definition(&mut payload, BED_MIX_GAIN_PARAMETER_ID);

        // Object elements
        for (idx, &elem_id) in self.object_element_ids.iter().enumerate() {
            write_leb128(&mut payload, elem_id as u64);
            let default_position = self
                .scene
                .object_position_defaults
                .get(idx)
                .cloned()
                .unwrap_or_else(default_object_position);
            self.write_rendering_config(&mut payload, Some(&default_position));
            self.write_mix_gain_param_definition(&mut payload, OBJECT_MIX_GAIN_PARAMETER_ID);
        }

        // output_mix_config (parameter_id, rate, MIX_GAIN, default)
        self.write_mix_gain_param_definition(&mut payload, OUTPUT_MIX_GAIN_PARAMETER_ID);

        // num_layouts = 1 (binaural)
        write_leb128(&mut payload, 1u64);
        payload.push(0x80); // layout_type = LOUDSPEAKERS_SS_CONVENTION, sound_system A
                            // loudness_info for the single layout
        self.write_loudness_info_aggregate(&mut payload);

        write_obu(obu_type::MIX_PRESENTATION, &payload)
    }

    fn write_mix_gain_param_definition(&self, out: &mut Vec<u8>, parameter_id: u32) {
        write_leb128(out, parameter_id as u64);
        write_leb128(out, self.scene.sample_rate_hz as u64);
        out.push(0); // param_definition_mode=0, reserved=0
        write_leb128(out, self.scene.samples_per_frame as u64);
        write_leb128(out, self.scene.samples_per_frame as u64);
        out.extend_from_slice(&0i16.to_be_bytes()); // default_mix_gain = 0 dB, Q7.8
    }

    fn write_rendering_config(&self, out: &mut Vec<u8>, default_position: Option<&ObjectPosition>) {
        out.push(0); // headphones_rendering_mode=0, reserved=0
        let mut extension = Vec::new();
        match default_position {
            Some(position) => {
                write_leb128(&mut extension, 1u64); // num_parameters
                write_leb128(
                    &mut extension,
                    param_definition_type::SINGLE_POSITION as u64,
                );
                self.write_single_position_param_definition(&mut extension, position);
            }
            None => {
                write_leb128(&mut extension, 0u64); // num_parameters
            }
        }
        write_leb128(out, extension.len() as u64);
        out.extend(extension);
    }

    fn write_single_position_param_definition(&self, out: &mut Vec<u8>, position: &ObjectPosition) {
        write_leb128(out, OBJECT_POSITION_PARAMETER_ID as u64);
        write_leb128(out, self.scene.sample_rate_hz as u64);
        out.push(0); // param_definition_mode=0, reserved=0
        write_leb128(out, self.scene.samples_per_frame as u64);
        write_leb128(out, self.scene.samples_per_frame as u64);
        pack_single_position_triplet(
            out,
            quantize_azimuth(position.azimuth_deg),
            quantize_elevation(position.elevation_deg),
            quantize_distance(position.distance_norm),
        );
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

    /// Experimental / future-spec-version — see module doc comment.
    fn object_parameter_block(&self, pos: &ObjectPosition) -> Vec<u8> {
        let mut payload = Vec::new();
        write_leb128(&mut payload, OBJECT_POSITION_PARAMETER_ID as u64);

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
    if animation == animation_type::LINEAR {
        pack_single_position_parameter_values(
            &mut body,
            (
                quantize_azimuth(start.azimuth_deg),
                quantize_elevation(start.elevation_deg),
                quantize_distance(start.distance_norm),
            ),
            Some((
                quantize_azimuth(end.azimuth_deg),
                quantize_elevation(end.elevation_deg),
                quantize_distance(end.distance_norm),
            )),
        );
    } else {
        pack_single_position_parameter_values(
            &mut body,
            (
                quantize_azimuth(start.azimuth_deg),
                quantize_elevation(start.elevation_deg),
                quantize_distance(start.distance_norm),
            ),
            None,
        );
    }
    let mut out = Vec::new();
    write_leb128(&mut out, body.len() as u64);
    out.extend(body);
    out
}

fn default_object_position() -> ObjectPosition {
    ObjectPosition {
        azimuth_deg: 0.0,
        elevation_deg: 0.0,
        distance_norm: 0.0,
        end_azimuth_deg: None,
        end_elevation_deg: None,
        end_distance_norm: None,
    }
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

fn pack_single_position_parameter_values(
    out: &mut Vec<u8>,
    start: (i16, i8, u8),
    end: Option<(i16, i8, u8)>,
) {
    let mut bits = BitWriter::default();
    bits.write_signed(start.0 as i32, 9);
    if let Some(end) = end {
        bits.write_signed(end.0 as i32, 9);
        bits.write_signed(start.1 as i32, 8);
        bits.write_signed(end.1 as i32, 8);
        bits.write_unsigned(start.2 as u32, 3);
        bits.write_unsigned(end.2 as u32, 3);
    } else {
        bits.write_signed(start.1 as i32, 8);
        bits.write_unsigned(start.2 as u32, 3);
        bits.write_unsigned(0, 4);
    }
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
    if substream_id <= 17 {
        return write_obu(obu_type::AUDIO_FRAME_ID0 + substream_id as u8, pcm_bytes);
    }
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
#[allow(dead_code)]
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
    let n_frames = n_samples.div_ceil(samples_per_frame);
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
    let n_frames = n_samples.div_ceil(samples_per_frame);
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

// ── N-mono object substream writer ────────────────────────────────────────────

/// Metadata for one mono substream in a distributed-node cluster recording.
#[derive(Clone, Debug)]
pub struct MonoSubstreamInfo {
    pub sensor_id: String,
    pub position_m: [f32; 3],
    pub sync_grade: String,
}

/// Write N mono CHANNEL Audio_Elements without a FOA bed.
///
/// Produces a valid (if non-standard for Base Profile) IAMF bitstream where
/// each mic gets its own CHANNEL Audio_Element.  Parsers compliant with the
/// IAMF Simple Profile (≤ 28 elements) will accept up to 28 sensors.
///
/// Returns the raw IAMF byte sequence.
pub fn write_object_substreams(
    streams: &[(MonoSubstreamInfo, Vec<f32>)],
    sample_rate_hz: u32,
    samples_per_frame: usize,
    loudness: &[LoudnessInfo],
) -> Vec<u8> {
    if streams.is_empty() {
        return Vec::new();
    }
    let n = streams.len();
    let n_samples = streams[0].1.len();
    let n_frames = n_samples.div_ceil(samples_per_frame);

    let codec_config_id: u32 = 0;
    // Each sensor gets its own Audio_Element (channel-based, mono).
    let element_ids: Vec<u32> = (1..=n as u32).collect();
    let substream_ids: Vec<u32> = element_ids.clone();
    let mix_id: u32 = n as u32 + 1;

    let mut out = Vec::new();

    // ── Descriptor OBUs ───────────────────────────────────────────────────────
    // IA_Sequence_Header
    let mut sh_payload = b"iamf".to_vec();
    sh_payload.extend_from_slice(&[1u8, 1u8]); // Base profile
    out.extend(write_obu(obu_type::IA_SEQUENCE_HEADER, &sh_payload));

    // Codec_Config (ipcm, 16-bit)
    {
        let mut payload = Vec::new();
        write_leb128(&mut payload, codec_config_id as u64);
        payload.extend_from_slice(b"ipcm");
        write_leb128(&mut payload, samples_per_frame as u64);
        payload.extend_from_slice(&0i16.to_be_bytes());
        payload.push(0);
        payload.push(16);
        payload.extend_from_slice(&sample_rate_hz.to_be_bytes());
        out.extend(write_obu(obu_type::CODEC_CONFIG, &payload));
    }

    // Audio_Elements: one mono CHANNEL element per sensor.
    for (i, elem_id) in element_ids.iter().enumerate() {
        let ss_id = substream_ids[i];
        let mut payload = Vec::new();
        write_leb128(&mut payload, *elem_id as u64);
        payload.push(audio_element_type::OBJECT_BASED << 5);
        write_leb128(&mut payload, codec_config_id as u64);
        write_leb128(&mut payload, 1u64); // num_substreams = 1
        write_leb128(&mut payload, ss_id as u64);
        write_leb128(&mut payload, 0u64); // num_parameters = 0
                                          // Scalable_Channel_Layout_Config: num_layers=1
        write_leb128(&mut payload, 1u64);
        payload.extend_from_slice(&[0x00, 0x00, 1u8, 0u8]); // MONO layout
        out.extend(write_obu(obu_type::AUDIO_ELEMENT, &payload));
    }

    // Mix_Presentation
    {
        let mut payload = Vec::new();
        write_leb128(&mut payload, mix_id as u64);
        write_leb128(&mut payload, 0u64); // count_label = 0
        write_leb128(&mut payload, n as u64); // num_audio_elements
        for (i, elem_id) in element_ids.iter().enumerate() {
            write_leb128(&mut payload, *elem_id as u64);
            write_leb128(&mut payload, 0u64); // rendering mode
            write_leb128(&mut payload, 0u64); // num_parameters
            payload.push(0); // rendering_config
            let lm = loudness.get(i).cloned().unwrap_or_default();
            payload.extend_from_slice(&[0u8]); // info_type
            let lufs = (lm.integrated_loudness_lufs * 256.0)
                .round()
                .clamp(-32768.0, 32767.0) as i16;
            let tp = (lm.true_peak_dbfs * 256.0).round().clamp(-32768.0, 32767.0) as i16;
            payload.extend_from_slice(&lufs.to_be_bytes());
            payload.extend_from_slice(&tp.to_be_bytes());
        }
        write_leb128(&mut payload, 1u64); // num_layouts
        payload.push(0); // layout_type = MONO
                         // Consolidated loudness (use first sensor's loudness as representative)
        let lm = loudness.first().cloned().unwrap_or_default();
        payload.push(0);
        let lufs = (lm.integrated_loudness_lufs * 256.0)
            .round()
            .clamp(-32768.0, 32767.0) as i16;
        let tp = (lm.true_peak_dbfs * 256.0).round().clamp(-32768.0, 32767.0) as i16;
        payload.extend_from_slice(&lufs.to_be_bytes());
        payload.extend_from_slice(&tp.to_be_bytes());
        out.extend(write_obu(obu_type::MIX_PRESENTATION, &payload));
    }

    // ── Temporal units ────────────────────────────────────────────────────────
    for fi in 0..n_frames {
        let frame_start = fi * samples_per_frame;
        let frame_end = (frame_start + samples_per_frame).min(n_samples);

        out.extend(write_obu(obu_type::TEMPORAL_DELIMITER, &[]));

        for (i, ss_id) in substream_ids.iter().enumerate() {
            let samples = &streams[i].1;
            let mut frame: Vec<f32> = samples[frame_start..frame_end.min(samples.len())].to_vec();
            frame.resize(samples_per_frame, 0.0);
            let pcm = f32_to_pcm16le(&frame);
            out.extend(audio_frame_obu(*ss_id, &pcm));
        }
    }

    out
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(n_objects: usize) -> (IamfScene, usize) {
        let scene = IamfScene {
            sample_rate_hz: 48_000,
            samples_per_frame: 512,
            n_bed_channels: FOA_CHANNEL_COUNT,
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
            object_position_defaults: (0..n_objects)
                .map(|_| ObjectPosition {
                    azimuth_deg: 10.0,
                    elevation_deg: 5.0,
                    distance_norm: 0.25,
                    end_azimuth_deg: None,
                    end_elevation_deg: None,
                    end_distance_norm: None,
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
        let (parameter_id, consumed) = read_leb128(payload).unwrap();
        assert_eq!(parameter_id, OBJECT_POSITION_PARAMETER_ID as u64);
        let (animation, values) = decode_single_position_data(&payload[consumed..]);
        assert_eq!(animation as u8, animation_type::LINEAR);
        assert_eq!(values, vec![10, 20, 5, 10, 2, 2]);
        offset = skip_obu(&unit, offset);
        assert_eq!(unit[offset] >> 3, obu_type::AUDIO_FRAME_ID0);
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
        let (animation, values) = decode_single_position_data(&data);
        assert_eq!(animation as u8, animation_type::LINEAR);
        assert_eq!(values, vec![0, 30, 0, 5, 1, 4]);
    }

    #[test]
    fn single_position_parameter_data_uses_step_for_static_position() {
        let start = ObjectPosition {
            azimuth_deg: -30.0,
            elevation_deg: 5.0,
            distance_norm: 0.25,
            end_azimuth_deg: None,
            end_elevation_deg: None,
            end_distance_norm: None,
        };
        let data = single_position_parameter_data(&start, &start);
        let (animation, values) = decode_single_position_data(&data);
        assert_eq!(animation as u8, animation_type::STEP);
        assert_eq!(values, vec![-30, 5, 2]);
    }

    #[test]
    fn mix_presentation_uses_spec_param_definition_shape() {
        let (scene, n) = make_scene(1);
        let writer = IamfWriter::new(scene, n);
        let descriptor_obus = writer.write_descriptor_obus();
        let mut offset = 0;
        let mut mix_payload = None;
        while offset < descriptor_obus.len() {
            let current_obu_type = descriptor_obus[offset] >> 3;
            if current_obu_type == obu_type::MIX_PRESENTATION {
                mix_payload = Some(obu_payload(&descriptor_obus[offset..]));
                break;
            }
            offset = skip_obu(&descriptor_obus, offset);
        }
        let payload = mix_payload.expect("mix presentation OBU");
        let mut offset = 0;
        let (_mix_id, consumed) = read_leb128(payload).unwrap();
        offset += consumed;
        let (count_label, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        assert_eq!(count_label, 0);
        let (num_sub_mixes, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        assert_eq!(num_sub_mixes, 1);
        let (num_audio_elements, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        assert_eq!(num_audio_elements, 2);

        let (_bed_element_id, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        offset += 1; // headphones_rendering_mode/reserved
        let (rendering_config_size, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed + rendering_config_size as usize;
        let (bed_mix_parameter_id, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        assert_eq!(bed_mix_parameter_id, BED_MIX_GAIN_PARAMETER_ID as u64);
        let (_rate, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        assert_eq!(payload[offset], 0); // param_definition_mode
        offset += 1;
        let (duration, consumed) = read_leb128(&payload[offset..]).unwrap();
        offset += consumed;
        let (constant_subblock_duration, _consumed) = read_leb128(&payload[offset..]).unwrap();
        assert_eq!(duration, 512);
        assert_eq!(constant_subblock_duration, 512);
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

    fn decode_single_position_data(data: &[u8]) -> (u64, Vec<i32>) {
        let (_size, size_consumed) = read_leb128(data).unwrap();
        let (animation, animation_consumed) = read_leb128(&data[size_consumed..]).unwrap();
        let mut bit_offset = (size_consumed + animation_consumed) * 8;
        if animation as u8 == animation_type::LINEAR {
            let start_azimuth = read_signed_bits(data, &mut bit_offset, 9);
            let end_azimuth = read_signed_bits(data, &mut bit_offset, 9);
            let start_elevation = read_signed_bits(data, &mut bit_offset, 8);
            let end_elevation = read_signed_bits(data, &mut bit_offset, 8);
            let start_distance = read_unsigned_bits(data, &mut bit_offset, 3) as i32;
            let end_distance = read_unsigned_bits(data, &mut bit_offset, 3) as i32;
            return (
                animation,
                vec![
                    start_azimuth,
                    end_azimuth,
                    start_elevation,
                    end_elevation,
                    start_distance,
                    end_distance,
                ],
            );
        }
        let azimuth = read_signed_bits(data, &mut bit_offset, 9);
        let elevation = read_signed_bits(data, &mut bit_offset, 8);
        let distance = read_unsigned_bits(data, &mut bit_offset, 3) as i32;
        (animation, vec![azimuth, elevation, distance])
    }

    fn read_signed_bits(data: &[u8], bit_offset: &mut usize, n_bits: usize) -> i32 {
        let raw = read_unsigned_bits(data, bit_offset, n_bits) as i32;
        let sign_bit = 1i32 << (n_bits - 1);
        if raw & sign_bit != 0 {
            raw - (1i32 << n_bits)
        } else {
            raw
        }
    }

    fn read_unsigned_bits(data: &[u8], bit_offset: &mut usize, n_bits: usize) -> u32 {
        let mut value = 0u32;
        for _ in 0..n_bits {
            let byte = data[*bit_offset / 8];
            let bit = (byte >> (7 - (*bit_offset % 8))) & 1;
            value = (value << 1) | u32::from(bit);
            *bit_offset += 1;
        }
        value
    }
}
