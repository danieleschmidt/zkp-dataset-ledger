//! Cryptographic primitives and zero-knowledge proof implementations.

pub mod hash;
pub mod merkle;

pub use hash::*;
pub use merkle::*;

use crate::{CryptoConfig, Result};

/// Initialize cryptographic components with default settings.
pub fn init() -> Result<()> {
    // Basic initialization - ensure random number generation is available
    use ark_std::rand::RngCore;
    let mut rng = ark_std::test_rng();
    let _test_bytes = rng.next_u64(); // Test that RNG works

    Ok(())
}

/// Initialize cryptographic components with custom configuration.
pub fn init_with_config(_config: &CryptoConfig) -> Result<()> {
    // For now, just call basic init
    // In a full implementation, this would set up specific curves, parameters, etc.
    init()
}
