//! The relay-visible half of a spoken join code.
//!
//! A join code reads like `K7QX-731904`: the lookup part (`K7QX`, generated
//! by the Worker, stored in D1) finds the session; the secret part
//! (`731904`, generated on the host phone) is the SPAKE2 password and is
//! never sent to the Worker. If the Worker saw the password it could run the
//! PAKE itself and win every time (core_codec_proto.md, contract 4).
//!
//! The alphabet drops 0/O and 1/I so a code read aloud is unambiguous; it has
//! exactly 32 symbols so `random_byte % 32` is unbiased.

pub const JOIN_CODE_LOOKUP_ALPHABET: &[u8; 32] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/// Builds a lookup code from caller-supplied random bytes (one per symbol).
pub fn lookup_code_from_random_bytes(random_bytes: &[u8]) -> String {
    random_bytes
        .iter()
        .map(|byte| JOIN_CODE_LOOKUP_ALPHABET[(*byte as usize) % JOIN_CODE_LOOKUP_ALPHABET.len()] as char)
        .collect()
}

/// Uppercases and validates a typed lookup code; `None` if it cannot be one.
pub fn normalize_lookup_code(typed_lookup_code: &str, expected_length: usize) -> Option<String> {
    let normalized: String = typed_lookup_code.trim().to_ascii_uppercase();
    let valid = normalized.len() == expected_length
        && normalized
            .bytes()
            .all(|symbol| JOIN_CODE_LOOKUP_ALPHABET.contains(&symbol));
    valid.then_some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_codes_validate_and_typed_codes_normalize() {
        let code = lookup_code_from_random_bytes(&[0, 31, 200, 77]);
        assert_eq!(code.len(), 4);
        assert_eq!(normalize_lookup_code(&code, 4), Some(code.clone()));
        assert_eq!(
            normalize_lookup_code(&format!(" {} ", code.to_lowercase()), 4),
            Some(code)
        );
        assert_eq!(normalize_lookup_code("K7Q0", 4), None, "0 is not in the alphabet");
        assert_eq!(normalize_lookup_code("K7Q", 4), None);
    }
}
