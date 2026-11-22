//! Spoken join codes: `LOOKUP-SECRET`, e.g. `K7QX-731904`.
//!
//! The lookup half is minted by the relay Worker and is all it ever sees.
//! The secret half is minted here on the host and is the SPAKE2 password
//! (`HostJoinCodeResponder::new(secret)`); it never leaves the phones.

use minimappr_relay_wire::join_code_lookup::normalize_lookup_code;
use rand_core::{CryptoRng, RngCore};

/// Symbols in the Worker-minted lookup half (32^4 ~ 1M live codes).
pub const JOIN_CODE_LOOKUP_LENGTH: usize = 4;
/// Digits in the secret half (~20 bits; SPAKE2 gives the relay one online
/// guess per attempt and the host retires a code after a few wrong ones).
pub const JOIN_CODE_SECRET_DIGIT_COUNT: usize = 6;
const DECIMAL_RADIX: u8 = 10;
/// Largest multiple of 10 that fits in a byte: rejection sampling above it
/// keeps every digit equally likely.
const UNBIASED_DIGIT_BYTE_LIMIT: u8 = 250;
const JOIN_CODE_SEPARATOR: char = '-';

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldJoinCode {
    pub lookup_code: String,
    secret_digits: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinCodeParseError {
    WrongShape,
    InvalidLookupPart,
    InvalidSecretPart,
}

impl FieldJoinCode {
    /// Host side: pairs the Worker's lookup code with a fresh local secret.
    pub fn with_generated_secret(
        lookup_code: String,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> Self {
        let mut secret_digits = String::with_capacity(JOIN_CODE_SECRET_DIGIT_COUNT);
        while secret_digits.len() < JOIN_CODE_SECRET_DIGIT_COUNT {
            let mut byte = [0_u8; 1];
            random_source.fill_bytes(&mut byte);
            if byte[0] < UNBIASED_DIGIT_BYTE_LIMIT {
                secret_digits.push(char::from(b'0' + byte[0] % DECIMAL_RADIX));
            }
        }
        Self {
            lookup_code,
            secret_digits,
        }
    }

    /// Joiner side: accepts `K7QX-731904`, `k7qx 731904` or `K7QX731904`.
    pub fn parse(typed_code: &str) -> Result<Self, JoinCodeParseError> {
        let compact: String = typed_code
            .chars()
            .filter(|symbol| !symbol.is_whitespace() && *symbol != JOIN_CODE_SEPARATOR)
            .collect();
        if compact.len() != JOIN_CODE_LOOKUP_LENGTH + JOIN_CODE_SECRET_DIGIT_COUNT
            || !compact.is_ascii()
        {
            return Err(JoinCodeParseError::WrongShape);
        }
        let (lookup_part, secret_part) = compact.split_at(JOIN_CODE_LOOKUP_LENGTH);
        let lookup_code = normalize_lookup_code(lookup_part, JOIN_CODE_LOOKUP_LENGTH)
            .ok_or(JoinCodeParseError::InvalidLookupPart)?;
        if !secret_part.bytes().all(|symbol| symbol.is_ascii_digit()) {
            return Err(JoinCodeParseError::InvalidSecretPart);
        }
        Ok(Self {
            lookup_code,
            secret_digits: secret_part.to_string(),
        })
    }

    /// The SPAKE2 password. Never put this in a relay request.
    pub fn secret_password_bytes(&self) -> &[u8] {
        self.secret_digits.as_bytes()
    }

    /// What the host shows and birders read aloud.
    pub fn display_text(&self) -> String {
        format!(
            "{}{JOIN_CODE_SEPARATOR}{}",
            self.lookup_code, self.secret_digits
        )
    }
}

impl std::error::Error for JoinCodeParseError {}

impl std::fmt::Display for JoinCodeParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self {
            Self::WrongShape => "join code must be 4 letters/digits then 6 digits",
            Self::InvalidLookupPart => "lookup part has a symbol outside the join-code alphabet",
            Self::InvalidSecretPart => "secret part must be digits",
        };
        formatter.write_str(description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn generated_code_round_trips_through_display_and_parse() {
        let mut random_source = ChaCha20Rng::seed_from_u64(7);
        let code = FieldJoinCode::with_generated_secret("K7QX".to_string(), &mut random_source);
        assert_eq!(code.secret_password_bytes().len(), JOIN_CODE_SECRET_DIGIT_COUNT);
        let parsed = FieldJoinCode::parse(&code.display_text().to_lowercase()).unwrap();
        assert_eq!(parsed, code);
        let spaced = format!("{} {}", code.lookup_code, code.display_text().split('-').nth(1).unwrap());
        assert_eq!(FieldJoinCode::parse(&spaced).unwrap(), code);
    }

    #[test]
    fn malformed_codes_are_rejected() {
        assert_eq!(FieldJoinCode::parse("K7QX-73190"), Err(JoinCodeParseError::WrongShape));
        assert_eq!(FieldJoinCode::parse("K7Q0-731904"), Err(JoinCodeParseError::InvalidLookupPart));
        assert_eq!(FieldJoinCode::parse("K7QX-7319A4"), Err(JoinCodeParseError::InvalidSecretPart));
    }
}
