//! `rand_core` CSPRNG backed by `crypto.getRandomValues`.
//!
//! The core and transport take randomness by injection (no `getrandom` in
//! the wasm tree); this is the browser's injection. It looks `crypto` up on
//! the global object so it works on a page and in a worker.

use rand_core::{CryptoRng, RngCore};
use wasm_bindgen::JsCast;

/// `getRandomValues` throws a QuotaExceededError above 65 536 bytes per call.
const MAXIMUM_BYTES_PER_GET_RANDOM_VALUES_CALL: usize = 65_536;

#[derive(Clone)]
pub struct BrowserCryptoRandomSource {
    crypto: web_sys::Crypto,
}

impl BrowserCryptoRandomSource {
    /// `None` when the global has no WebCrypto (never on a supported browser).
    pub fn from_global() -> Option<Self> {
        let global = js_sys::global();
        let crypto = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("crypto")).ok()?;
        Some(Self {
            crypto: crypto.dyn_into::<web_sys::Crypto>().ok()?,
        })
    }
}

impl RngCore for BrowserCryptoRandomSource {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0_u8; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0_u8; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, destination: &mut [u8]) {
        self.try_fill_bytes(destination)
            .expect("crypto.getRandomValues failed; refusing to continue without a CSPRNG");
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), rand_core::Error> {
        for chunk in destination.chunks_mut(MAXIMUM_BYTES_PER_GET_RANDOM_VALUES_CALL) {
            self.crypto
                .get_random_values_with_u8_array(chunk)
                .map_err(|_| rand_core::Error::from(core::num::NonZeroU32::new(rand_core::Error::CUSTOM_START).expect("non-zero")))?;
        }
        Ok(())
    }
}

impl CryptoRng for BrowserCryptoRandomSource {}
