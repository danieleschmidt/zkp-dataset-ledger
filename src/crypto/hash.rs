//! Hash functions for the ZKP Dataset Ledger.

use crate::error::LedgerError;
use blake3::Hasher as Blake3Hasher;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::io::Read;

/// Supported hash algorithms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HashAlgorithm {
    Sha3_256,
    Blake3,
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        Self::Sha3_256
    }
}

/// Hash a byte slice using the specified algorithm.
pub fn hash_bytes(data: &[u8], algorithm: HashAlgorithm) -> Result<String, LedgerError> {
    match algorithm {
        HashAlgorithm::Sha3_256 => {
            let mut hasher = Sha3_256::new();
            hasher.update(data);
            let result = hasher.finalize();
            Ok(hex::encode(result))
        }
        HashAlgorithm::Blake3 => {
            let hash = blake3::hash(data);
            Ok(hex::encode(hash.as_bytes()))
        }
    }
}

/// Hash a readable stream using the specified algorithm.
pub fn hash_reader<R: Read>(
    mut reader: R,
    algorithm: HashAlgorithm,
) -> Result<String, LedgerError> {
    match algorithm {
        HashAlgorithm::Sha3_256 => {
            let mut hasher = Sha3_256::new();
            let mut buffer = [0; 8192];
            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| LedgerError::IoError(e.to_string()))?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            let result = hasher.finalize();
            Ok(hex::encode(result))
        }
        HashAlgorithm::Blake3 => {
            let mut hasher = Blake3Hasher::new();
            let mut buffer = [0; 8192];
            loop {
                let bytes_read = reader
                    .read(&mut buffer)
                    .map_err(|e| LedgerError::IoError(e.to_string()))?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }
            let result = hasher.finalize();
            Ok(hex::encode(result.as_bytes()))
        }
    }
}

/// Compute a deterministic hash for a dataset file.
pub fn hash_dataset_file<P: AsRef<std::path::Path>>(
    path: P,
    algorithm: HashAlgorithm,
) -> Result<String, LedgerError> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| LedgerError::IoError(format!("Failed to open file: {}", e)))?;
    hash_reader(file, algorithm)
}

/// Compute a hash of multiple values concatenated together.
pub fn hash_combined(
    values: &[&[u8]],
    algorithm: HashAlgorithm,
) -> Result<String, LedgerError> {
    match algorithm {
        HashAlgorithm::Sha3_256 => {
            let mut hasher = Sha3_256::new();
            for value in values {
                hasher.update(value);
            }
            let result = hasher.finalize();
            Ok(hex::encode(result))
        }
        HashAlgorithm::Blake3 => {
            let mut hasher = Blake3Hasher::new();
            for value in values {
                hasher.update(value);
            }
            let result = hasher.finalize();
            Ok(hex::encode(result.as_bytes()))
        }
    }
}

/// Legacy functions for backward compatibility
pub fn sha3_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

pub fn blake3_hash(data: &[u8]) -> Vec<u8> {
    blake3::hash(data).as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_bytes_sha3() {
        let data = b"hello world";
        let hash = hash_bytes(data, HashAlgorithm::Sha3_256).unwrap();
        assert_eq!(hash.len(), 64); // SHA3-256 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_hash_bytes_blake3() {
        let data = b"hello world";
        let hash = hash_bytes(data, HashAlgorithm::Blake3).unwrap();
        assert_eq!(hash.len(), 64); // Blake3 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_hash_combined() {
        let values = [b"hello".as_slice(), b" ".as_slice(), b"world".as_slice()];
        let hash1 = hash_combined(&values, HashAlgorithm::Sha3_256).unwrap();
        let hash2 = hash_bytes(b"hello world", HashAlgorithm::Sha3_256).unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_reader() {
        let data = b"hello world";
        let cursor = std::io::Cursor::new(data);
        let hash1 = hash_reader(cursor, HashAlgorithm::Sha3_256).unwrap();
        let hash2 = hash_bytes(data, HashAlgorithm::Sha3_256).unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_legacy_functions() {
        let data = b"test data";
        let sha3_result = sha3_hash(data);
        let blake3_result = blake3_hash(data);
        
        assert_eq!(sha3_result.len(), 32);
        assert_eq!(blake3_result.len(), 32);
    }
}
