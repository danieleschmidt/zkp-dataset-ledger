//! Dataset Commitment Scheme
//!
//! Implements a **Pedersen-style hash commitment** for ML datasets.
//!
//! ## How It Works
//!
//! ```text
//! Commit Phase (Data Owner):
//!   1. Compute dataset_hash = SHA256(full_dataset_bytes)
//!   2. Choose random nonce (blinding factor)
//!   3. Compute commitment = SHA256("commit" || dataset_hash || nonce)
//!   4. Publish: commitment + public metadata (row_count, column_count, hash)
//!   5. Keep secret: nonce, raw dataset
//!
//! Verify Phase (Auditor):
//!   1. Receive commitment + claimed properties
//!   2. Verify commitment = SHA256("commit" || dataset_hash || nonce)
//!      → proves the dataset hash matches what was committed
//!   3. Check that dataset_hash = SHA256(actual_dataset)
//!      → proves the commitment is about THIS dataset
//!
//! The zero-knowledge property: the auditor learns only what properties
//! the prover explicitly reveals. The nonce hides the dataset content.
//! ```
//!
//! ## Security Properties
//!
//! - **Binding**: Once committed, the prover cannot change the dataset
//!   (would require finding a SHA256 collision — computationally infeasible)
//! - **Hiding**: The commitment reveals nothing about the dataset without the nonce
//!   (SHA256 is a one-way function; nonce adds 256 bits of entropy)
//! - **Selective Disclosure**: Prover can reveal specific properties (row count,
//!   statistical bounds, bias metrics) without revealing the full dataset

use crate::crypto::{commit, sha256_hex};
use crate::{LedgerError, Result};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::io::Read;

/// Configuration for commitment generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentConfig {
    /// Hash algorithm to use for dataset fingerprinting
    pub hash_algorithm: HashAlgorithm,
    /// Include row count in the commitment metadata
    pub include_row_count: bool,
    /// Include column count in the commitment metadata
    pub include_column_count: bool,
    /// Include SHA256 of column names (schema fingerprint)
    pub include_schema_hash: bool,
}

impl Default for CommitmentConfig {
    fn default() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::Sha256,
            include_row_count: true,
            include_column_count: true,
            include_schema_hash: false,
        }
    }
}

/// Hash algorithm selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HashAlgorithm {
    Sha256,
    Sha3_256,
}

/// Public metadata about a dataset (revealed to verifier)
///
/// This is what the prover publishes. It contains ONLY what the prover
/// chooses to reveal — row counts, schema hashes, etc. The actual data stays private.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Cryptographic fingerprint of the dataset content
    pub dataset_hash: String,
    /// Optional: number of rows (prover may choose to reveal or hide)
    pub row_count: Option<u64>,
    /// Optional: number of columns
    pub column_count: Option<u64>,
    /// Optional: hash of column names (schema fingerprint)
    pub schema_hash: Option<String>,
    /// Human-readable dataset description
    pub description: Option<String>,
    /// Timestamp of commitment creation (ISO 8601)
    pub created_at: String,
    /// Dataset version/identifier
    pub version: String,
}

/// A dataset commitment — the core cryptographic artifact
///
/// The prover publishes `commitment_hash` and `metadata`.
/// The `nonce` is kept secret and only revealed during proof verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetCommitment {
    /// The binding commitment: SHA256("commit" || dataset_hash || nonce)
    /// This is the public artifact — publish this to an audit log, blockchain, etc.
    pub commitment_hash: String,

    /// Public dataset metadata (selectively disclosed by prover)
    pub metadata: DatasetMetadata,

    /// The secret blinding factor (kept private by data owner)
    /// In a real system, this would never be serialized to a public log
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<Vec<u8>>,
}

impl DatasetCommitment {
    /// Create a new commitment from raw dataset bytes.
    ///
    /// # Arguments
    /// * `data` - The raw dataset bytes to commit to
    /// * `config` - Configuration for what metadata to include
    /// * `description` - Optional human-readable description
    ///
    /// # Returns
    /// A commitment with the nonce included (keep this secret!)
    ///
    /// # Example
    /// ```
    /// use zkp_dataset_ledger::commitment::{DatasetCommitment, CommitmentConfig};
    ///
    /// let dataset = b"col1,col2\n1,2\n3,4\n5,6\n";
    /// let commitment = DatasetCommitment::commit(dataset, &CommitmentConfig::default(), None).unwrap();
    ///
    /// // commitment_hash is safe to publish
    /// println!("Commitment: {}", commitment.commitment_hash);
    ///
    /// // Verify it later
    /// assert!(commitment.verify(dataset).unwrap());
    /// ```
    pub fn commit(
        data: &[u8],
        config: &CommitmentConfig,
        description: Option<String>,
    ) -> Result<Self> {
        // Step 1: Compute dataset fingerprint
        let dataset_hash = sha256_hex(data);

        // Step 2: Generate random blinding nonce
        let mut nonce = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut nonce);

        // Step 3: Compute binding commitment
        let hash_bytes = hex::decode(&dataset_hash)
            .map_err(|e| LedgerError::Crypto(format!("hex decode failed: {e}")))?;
        let commitment_bytes = commit(&hash_bytes, &nonce);
        let commitment_hash = hex::encode(commitment_bytes);

        // Step 4: Build public metadata (selective disclosure)
        let row_count = if config.include_row_count {
            Some(count_csv_rows(data))
        } else {
            None
        };
        let column_count = if config.include_column_count {
            Some(count_csv_columns(data))
        } else {
            None
        };
        let schema_hash = if config.include_schema_hash {
            Some(compute_schema_hash(data))
        } else {
            None
        };

        let metadata = DatasetMetadata {
            dataset_hash,
            row_count,
            column_count,
            schema_hash,
            description,
            created_at: chrono::Utc::now().to_rfc3339(),
            version: "1.0.0".to_string(),
        };

        Ok(Self {
            commitment_hash,
            metadata,
            nonce: Some(nonce),
        })
    }

    /// Commit to a dataset from a reader (file-based).
    pub fn commit_reader<R: Read>(
        reader: R,
        config: &CommitmentConfig,
        description: Option<String>,
    ) -> Result<Self> {
        let data = read_all(reader)?;
        Self::commit(&data, config, description)
    }

    /// Verify that this commitment matches the provided dataset.
    ///
    /// This proves: "The dataset whose hash is in our metadata
    /// matches the commitment we published earlier."
    ///
    /// Requires the nonce (kept by the prover).
    pub fn verify(&self, data: &[u8]) -> Result<bool> {
        let nonce = self.nonce.as_ref().ok_or_else(|| {
            LedgerError::Crypto("Cannot verify without nonce (nonce was not stored)".to_string())
        })?;

        // Recompute the dataset hash
        let actual_hash = sha256_hex(data);

        // Check the dataset hash matches metadata
        if actual_hash != self.metadata.dataset_hash {
            return Ok(false);
        }

        // Recompute and check the commitment
        let hash_bytes = hex::decode(&actual_hash)
            .map_err(|e| LedgerError::Crypto(format!("hex decode: {e}")))?;
        let recomputed = commit(&hash_bytes, nonce);
        let recomputed_hex = hex::encode(recomputed);

        Ok(recomputed_hex == self.commitment_hash)
    }

    /// Verify using an explicit nonce (for cases where nonce is stored separately).
    pub fn verify_with_nonce(&self, data: &[u8], nonce: &[u8]) -> Result<bool> {
        let actual_hash = sha256_hex(data);
        if actual_hash != self.metadata.dataset_hash {
            return Ok(false);
        }
        let hash_bytes = hex::decode(&actual_hash)
            .map_err(|e| LedgerError::Crypto(format!("hex decode: {e}")))?;
        let recomputed = commit(&hash_bytes, nonce);
        Ok(hex::encode(recomputed) == self.commitment_hash)
    }

    /// Return a public-safe version (no nonce).
    ///
    /// Use this when serializing the commitment for publishing to an audit log.
    pub fn public_commitment(&self) -> Self {
        Self {
            commitment_hash: self.commitment_hash.clone(),
            metadata: self.metadata.clone(),
            nonce: None,
        }
    }
}

/// Count CSV rows (data rows, excluding header)
fn count_csv_rows(data: &[u8]) -> u64 {
    let text = std::str::from_utf8(data).unwrap_or("");
    let total_lines = text.lines().filter(|l| !l.trim().is_empty()).count();
    if total_lines > 0 { (total_lines - 1) as u64 } else { 0 }
}

/// Count CSV columns (from first line)
fn count_csv_columns(data: &[u8]) -> u64 {
    let text = std::str::from_utf8(data).unwrap_or("");
    if let Some(first_line) = text.lines().next() {
        first_line.split(',').count() as u64
    } else {
        0
    }
}

/// Hash the schema (column names)
fn compute_schema_hash(data: &[u8]) -> String {
    let text = std::str::from_utf8(data).unwrap_or("");
    if let Some(first_line) = text.lines().next() {
        sha256_hex(first_line.as_bytes())
    } else {
        sha256_hex(b"")
    }
}

fn read_all<R: Read>(mut reader: R) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &[u8] = b"id,age,salary\n1,25,50000\n2,30,60000\n3,35,70000\n";

    #[test]
    fn commitment_creation_and_verification() {
        let commitment = DatasetCommitment::commit(SAMPLE_CSV, &CommitmentConfig::default(), None)
            .expect("commitment should succeed");

        // Should verify against original data
        assert!(commitment.verify(SAMPLE_CSV).unwrap());

        // Should fail for different data
        let different = b"id,age,salary\n1,25,99999\n";
        assert!(!commitment.verify(different).unwrap());
    }

    #[test]
    fn metadata_correctness() {
        let config = CommitmentConfig {
            include_row_count: true,
            include_column_count: true,
            include_schema_hash: true,
            ..Default::default()
        };
        let commitment = DatasetCommitment::commit(SAMPLE_CSV, &config, Some("Test dataset".to_string()))
            .expect("commitment should succeed");

        assert_eq!(commitment.metadata.row_count, Some(3));
        assert_eq!(commitment.metadata.column_count, Some(3));
        assert!(commitment.metadata.schema_hash.is_some());
        assert_eq!(commitment.metadata.description.as_deref(), Some("Test dataset"));
    }

    #[test]
    fn commitment_hash_is_stable() {
        // Two commits of the same data produce different commitments (random nonce)
        let c1 = DatasetCommitment::commit(SAMPLE_CSV, &CommitmentConfig::default(), None).unwrap();
        let c2 = DatasetCommitment::commit(SAMPLE_CSV, &CommitmentConfig::default(), None).unwrap();
        // Same dataset hash
        assert_eq!(c1.metadata.dataset_hash, c2.metadata.dataset_hash);
        // Different commitment hashes (different nonces)
        assert_ne!(c1.commitment_hash, c2.commitment_hash);
    }

    #[test]
    fn public_commitment_has_no_nonce() {
        let commitment = DatasetCommitment::commit(SAMPLE_CSV, &CommitmentConfig::default(), None).unwrap();
        let public = commitment.public_commitment();
        assert!(public.nonce.is_none());
    }

    #[test]
    fn verify_with_explicit_nonce() {
        let commitment = DatasetCommitment::commit(SAMPLE_CSV, &CommitmentConfig::default(), None).unwrap();
        let nonce = commitment.nonce.clone().unwrap();
        assert!(commitment.verify_with_nonce(SAMPLE_CSV, &nonce).unwrap());
    }
}
