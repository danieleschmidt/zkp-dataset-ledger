//! Cryptographic primitives for ZKP Dataset Ledger.
//!
//! Uses standard, well-audited hash functions (SHA-256, SHA3-256) to implement
//! commitment schemes and Fiat-Shamir transforms. No external ZK frameworks required.

use sha2::{Digest, Sha256};
use sha3::Sha3_256;

/// Hash data with SHA-256, returning a hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Hash data with SHA3-256, returning a hex string.
pub fn sha3_256_hex(data: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Hash data with SHA-256, returning raw bytes.
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Hash multiple byte slices together (domain-separated concatenation).
///
/// Uses length-prefixed encoding to avoid collisions:
/// hash(len(a) || a || len(b) || b || ...)
pub fn hash_concat(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        // Length prefix (8 bytes, little-endian) prevents extension attacks
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Pedersen-style binding commitment: `C = Hash("commit" || data || nonce)`
///
/// Binding: computationally infeasible to find two (data, nonce) pairs with same C.
/// Hiding: given C alone, data is computationally hidden (nonce acts as blinding factor).
pub fn commit(data: &[u8], nonce: &[u8]) -> [u8; 32] {
    hash_concat(&[b"zkp-dataset-commit-v1", data, nonce])
}

/// Fiat-Shamir challenge hash for sigma protocols.
///
/// Takes a transcript (public parameters + prover's commitment) and produces
/// a deterministic challenge, replacing the verifier's random coin flip.
pub fn fiat_shamir_challenge(transcript: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"fiat-shamir-v1");
    for part in transcript {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_is_deterministic() {
        let data = b"my dataset rows";
        let nonce = b"random-blinding-factor-42";
        assert_eq!(commit(data, nonce), commit(data, nonce));
    }

    #[test]
    fn commit_hides_data() {
        // Different data → different commitment
        let nonce = b"same-nonce";
        let c1 = commit(b"dataset-A", nonce);
        let c2 = commit(b"dataset-B", nonce);
        assert_ne!(c1, c2);
    }

    #[test]
    fn commit_binds_nonce() {
        // Same data, different nonce → different commitment
        let data = b"same-dataset";
        let c1 = commit(data, b"nonce-1");
        let c2 = commit(data, b"nonce-2");
        assert_ne!(c1, c2);
    }

    #[test]
    fn hash_concat_domain_separated() {
        // Ensure "ab" + "c" ≠ "a" + "bc"
        let h1 = hash_concat(&[b"ab", b"c"]);
        let h2 = hash_concat(&[b"a", b"bc"]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn fiat_shamir_is_deterministic() {
        let t1 = fiat_shamir_challenge(&[b"commitment", b"public-key"]);
        let t2 = fiat_shamir_challenge(&[b"commitment", b"public-key"]);
        assert_eq!(t1, t2);
    }
}
