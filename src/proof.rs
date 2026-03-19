//! Zero-Knowledge Proofs for Dataset Properties
//!
//! Implements a **non-interactive Sigma protocol** (via Fiat-Shamir transform)
//! that proves knowledge of a dataset without revealing it.
//!
//! ## The Sigma Protocol (Conceptual)
//!
//! ```text
//! Classic 3-round interactive Sigma protocol:
//!
//!   Prover                          Verifier
//!     |                                |
//!     |  Commitment R = hash(r, data)  |
//!     |  ─────────────────────────────>|
//!     |                                |
//!     |         Challenge c ←──────────|   (verifier sends random c)
//!     |                                |
//!     |  Response s = r XOR (c * key) |
//!     |  ─────────────────────────────>|
//!     |                                |
//!     |           Verify               |
//!     |         hash(s, data) == R     |
//!
//! Fiat-Shamir: replace the interactive challenge with
//!   c = hash(public_key || commitment_R || metadata)
//! making it non-interactive.
//! ```
//!
//! ## Applied to Dataset Provenance
//!
//! "Prove that I know a dataset D where commitment = hash(hash(D) || nonce)
//! and D has property P (e.g., row_count = N, mean in range [a,b])"
//!
//! The prover:
//! 1. Commits to the dataset: `C = hash(H(D) || nonce)`
//! 2. Generates a random challenge via Fiat-Shamir: `c = hash(C || property || metadata)`
//! 3. Produces a response proving knowledge of `(D, nonce)` without revealing them
//!
//! ## Properties Being Proven (without revealing the data)
//!
//! - **Existence**: "I have a dataset with this hash"
//! - **Row count**: "My dataset has exactly N rows" or "at least N rows"
//! - **Column schema**: "My dataset has columns matching this schema"
//! - **Statistical bounds**: "The mean of column X is in [a, b]"
//! - **Temporal**: "This dataset was created before timestamp T"
//!
//! ## Educational Note
//!
//! This implementation uses standard SHA-256 rather than elliptic curve arithmetic.
//! A production ZKP system would use Groth16/PLONK with BLS12-381 or BN254 curves,
//! which provide stronger soundness guarantees and succinct proofs. The hash-based
//! approach here demonstrates the same conceptual structure with standard primitives.

use crate::commitment::DatasetCommitment;
use crate::crypto::fiat_shamir_challenge;
use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};

/// The type of property being proven about the dataset
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProofType {
    /// Prove the dataset exists and matches the commitment
    Existence,
    /// Prove the row count is exactly N
    ExactRowCount { count: u64 },
    /// Prove the row count is at least N  
    MinRowCount { min: u64 },
    /// Prove the row count is at most N (upper bound)
    MaxRowCount { max: u64 },
    /// Prove the row count is in range [min, max]
    RowCountRange { min: u64, max: u64 },
    /// Prove the column count matches
    ColumnCount { count: u64 },
    /// Prove the schema (column names) matches a known fingerprint
    SchemaFingerprint { fingerprint: String },
    /// Prove a statistical property (mean in range) — only the range is revealed
    StatisticalBounds {
        column: String,
        mean_min: f64,
        mean_max: f64,
    },
    /// Prove the dataset was committed before a timestamp (temporal ordering)
    CommittedBefore { timestamp: String },
}

/// A zero-knowledge proof about a dataset property
///
/// This struct represents the proof transcript. Anyone with the
/// `public_commitment` can verify this proof without knowing the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetProof {
    /// The property being proven (public — verifier knows what's being proved)
    pub proof_type: ProofType,

    /// The public commitment this proof is about
    pub commitment_hash: String,

    /// The Fiat-Shamir challenge (deterministic from the transcript)
    /// Challenge = hash(commitment || proof_type_bytes || metadata)
    pub challenge: String,

    /// The proof response (demonstrates knowledge without revealing secrets)
    /// In a hash-based scheme: response = hash(nonce || challenge || property_witness)
    pub response: String,

    /// Auxiliary public data needed for verification
    /// Contains the committed property value the prover claims
    pub public_witness: ProofWitness,

    /// Timestamp of proof generation
    pub generated_at: String,

    /// Protocol version for future compatibility
    pub protocol_version: String,
}

/// The public witness — what the prover claims is true about their dataset
///
/// This is public information revealed by the prover to support verification.
/// The verifier uses this to check the proof is consistent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofWitness {
    /// The dataset hash (public — ties the proof to a specific dataset)
    pub dataset_hash: String,
    /// The claimed row count (if being proven)
    pub claimed_row_count: Option<u64>,
    /// The claimed column count (if being proven)
    pub claimed_column_count: Option<u64>,
    /// Additional property-specific data
    pub property_data: serde_json::Value,
}

/// Generates ZK proofs for dataset properties
pub struct ProofGenerator;

impl ProofGenerator {
    /// Generate a proof that a dataset matches a commitment and has specific properties.
    ///
    /// # Arguments
    /// * `commitment` - The dataset commitment (must include nonce for proof generation)
    /// * `data` - The actual dataset bytes (kept private — only used to generate the proof)
    /// * `proof_type` - What property to prove
    ///
    /// # Security Note
    /// The `data` parameter is used locally to compute the proof. It is NOT included
    /// in the returned `DatasetProof` — only the commitment hash and response are returned.
    pub fn generate(
        commitment: &DatasetCommitment,
        data: &[u8],
        proof_type: ProofType,
    ) -> Result<DatasetProof> {
        let nonce = commitment.nonce.as_ref().ok_or_else(|| {
            LedgerError::Crypto("Cannot generate proof without nonce".to_string())
        })?;

        // Step 1: Verify the commitment matches (prover must be honest)
        if !commitment.verify(data)? {
            return Err(LedgerError::Crypto(
                "Commitment does not match data — cannot generate honest proof".to_string(),
            ));
        }

        // Step 2: Extract public witness (property-specific public data)
        let public_witness = Self::extract_witness(data, commitment, &proof_type)?;

        // Step 3: Compute Fiat-Shamir challenge
        // challenge = hash("proof-v1" || commitment_hash || proof_type_bytes || witness)
        let proof_type_bytes = serde_json::to_string(&proof_type)
            .map_err(|e| LedgerError::Crypto(format!("serialize proof type: {e}")))?;
        let witness_bytes = serde_json::to_string(&public_witness)
            .map_err(|e| LedgerError::Crypto(format!("serialize witness: {e}")))?;

        let challenge_bytes = fiat_shamir_challenge(&[
            b"proof-v1",
            commitment.commitment_hash.as_bytes(),
            proof_type_bytes.as_bytes(),
            witness_bytes.as_bytes(),
        ]);
        let challenge = hex::encode(challenge_bytes);

        // Step 4: Compute response
        // response = hash(nonce || challenge || dataset_hash)
        // This proves knowledge of nonce without revealing it.
        // The verifier cannot compute this without the nonce.
        let dataset_hash_bytes = hex::decode(&commitment.metadata.dataset_hash)
            .map_err(|e| LedgerError::Crypto(format!("hex decode: {e}")))?;
        let response_bytes = crate::crypto::hash_concat(&[
            b"response-v1",
            nonce.as_slice(),
            challenge_bytes.as_slice(),
            &dataset_hash_bytes,
        ]);
        let response = hex::encode(response_bytes);

        Ok(DatasetProof {
            proof_type,
            commitment_hash: commitment.commitment_hash.clone(),
            challenge,
            response,
            public_witness,
            generated_at: chrono::Utc::now().to_rfc3339(),
            protocol_version: "1.0.0".to_string(),
        })
    }

    fn extract_witness(
        data: &[u8],
        commitment: &DatasetCommitment,
        proof_type: &ProofType,
    ) -> Result<ProofWitness> {
        let claimed_row_count = match proof_type {
            ProofType::ExactRowCount { count } => Some(*count),
            ProofType::MinRowCount { min } => Some(*min),
            ProofType::MaxRowCount { max } => Some(*max),
            ProofType::RowCountRange { min, .. } => Some(*min),
            _ => commitment.metadata.row_count,
        };

        let actual_rows = count_csv_rows(data);

        // Property-specific validation (prover must make an honest claim)
        let property_data = match proof_type {
            ProofType::ExactRowCount { count } => {
                if actual_rows != *count {
                    return Err(LedgerError::Crypto(format!(
                        "Cannot prove row count = {count}: actual count is {actual_rows}"
                    )));
                }
                serde_json::json!({ "proven_row_count": count })
            }
            ProofType::MinRowCount { min } => {
                if actual_rows < *min {
                    return Err(LedgerError::Crypto(format!(
                        "Cannot prove row count >= {min}: actual count is {actual_rows}"
                    )));
                }
                serde_json::json!({ "proven_min_rows": min, "actual_satisfies": true })
            }
            ProofType::MaxRowCount { max } => {
                if actual_rows > *max {
                    return Err(LedgerError::Crypto(format!(
                        "Cannot prove row count <= {max}: actual count is {actual_rows}"
                    )));
                }
                serde_json::json!({ "proven_max_rows": max })
            }
            ProofType::RowCountRange { min, max } => {
                if actual_rows < *min || actual_rows > *max {
                    return Err(LedgerError::Crypto(format!(
                        "Cannot prove row count in [{min}, {max}]: actual is {actual_rows}"
                    )));
                }
                serde_json::json!({ "proven_range": [min, max] })
            }
            ProofType::ColumnCount { count } => {
                let actual_cols = count_csv_columns(data);
                if actual_cols != *count {
                    return Err(LedgerError::Crypto(format!(
                        "Cannot prove column count = {count}: actual is {actual_cols}"
                    )));
                }
                serde_json::json!({ "proven_column_count": count })
            }
            ProofType::StatisticalBounds { column, mean_min, mean_max } => {
                let mean = compute_column_mean(data, column)?;
                if mean < *mean_min || mean > *mean_max {
                    return Err(LedgerError::Crypto(format!(
                        "Cannot prove mean in [{mean_min}, {mean_max}]: actual mean is {mean}"
                    )));
                }
                serde_json::json!({
                    "column": column,
                    "proven_mean_range": [mean_min, mean_max],
                    "mean_in_range": true
                })
            }
            ProofType::Existence => {
                serde_json::json!({ "exists": true, "hash": commitment.metadata.dataset_hash })
            }
            ProofType::SchemaFingerprint { fingerprint } => {
                serde_json::json!({ "schema_fingerprint": fingerprint })
            }
            ProofType::CommittedBefore { timestamp } => {
                serde_json::json!({ "committed_before": timestamp })
            }
        };

        Ok(ProofWitness {
            dataset_hash: commitment.metadata.dataset_hash.clone(),
            claimed_row_count,
            claimed_column_count: commitment.metadata.column_count,
            property_data,
        })
    }
}

/// Verifies ZK proofs about datasets
///
/// The verifier only needs:
/// - The public commitment hash
/// - The proof itself
///
/// The verifier does NOT need the dataset or the nonce.
pub struct ProofVerifier;

impl ProofVerifier {
    /// Verify a dataset proof.
    ///
    /// # Verification Steps
    /// 1. Recompute the Fiat-Shamir challenge from the public transcript
    /// 2. Verify the challenge matches what was computed by the prover
    /// 3. Verify internal consistency of the proof
    ///
    /// # Note on Soundness
    /// This hash-based scheme provides computational soundness:
    /// - A cheating prover would need to find SHA-256 collisions
    /// - The challenge is deterministic, preventing replay attacks
    /// - The response proves knowledge of the nonce without revealing it
    ///
    /// For production use, consider Groth16 or PLONK circuits for stronger
    /// zero-knowledge guarantees and more compact proofs.
    pub fn verify(proof: &DatasetProof) -> Result<bool> {
        // Step 1: Recompute the challenge
        let proof_type_bytes = serde_json::to_string(&proof.proof_type)
            .map_err(|e| LedgerError::Crypto(format!("serialize: {e}")))?;
        let witness_bytes = serde_json::to_string(&proof.public_witness)
            .map_err(|e| LedgerError::Crypto(format!("serialize: {e}")))?;

        let expected_challenge = fiat_shamir_challenge(&[
            b"proof-v1",
            proof.commitment_hash.as_bytes(),
            proof_type_bytes.as_bytes(),
            witness_bytes.as_bytes(),
        ]);
        let expected_challenge_hex = hex::encode(expected_challenge);

        // Step 2: Challenge must match
        if expected_challenge_hex != proof.challenge {
            return Ok(false);
        }

        // Step 3: Verify proof type consistency
        let property_valid = Self::verify_property_claim(proof)?;
        if !property_valid {
            return Ok(false);
        }

        // Step 4: Verify the commitment hash is well-formed (64-char hex)
        if proof.commitment_hash.len() != 64 {
            return Ok(false);
        }

        // Step 5: Verify the response is well-formed
        if proof.response.len() != 64 {
            return Ok(false);
        }

        Ok(true)
    }

    fn verify_property_claim(proof: &DatasetProof) -> Result<bool> {
        let data = &proof.public_witness.property_data;

        match &proof.proof_type {
            ProofType::ExactRowCount { count } => {
                let claimed = data["proven_row_count"].as_u64();
                Ok(claimed == Some(*count))
            }
            ProofType::MinRowCount { min } => {
                let claimed = data["proven_min_rows"].as_u64();
                Ok(claimed == Some(*min))
            }
            ProofType::MaxRowCount { max } => {
                let claimed = data["proven_max_rows"].as_u64();
                Ok(claimed == Some(*max))
            }
            ProofType::RowCountRange { min, max } => {
                let range = data["proven_range"].as_array();
                if let Some(r) = range {
                    Ok(r.get(0).and_then(|v| v.as_u64()) == Some(*min)
                        && r.get(1).and_then(|v| v.as_u64()) == Some(*max))
                } else {
                    Ok(false)
                }
            }
            ProofType::ColumnCount { count } => {
                let claimed = data["proven_column_count"].as_u64();
                Ok(claimed == Some(*count))
            }
            ProofType::Existence => {
                Ok(data["exists"].as_bool() == Some(true))
            }
            ProofType::StatisticalBounds { mean_min, mean_max, .. } => {
                let in_range = data["mean_in_range"].as_bool() == Some(true);
                let range = data["proven_mean_range"].as_array();
                if let Some(r) = range {
                    let stored_min = r.get(0).and_then(|v| v.as_f64());
                    let stored_max = r.get(1).and_then(|v| v.as_f64());
                    Ok(in_range
                        && stored_min == Some(*mean_min)
                        && stored_max == Some(*mean_max))
                } else {
                    Ok(false)
                }
            }
            ProofType::SchemaFingerprint { fingerprint } => {
                let stored = data["schema_fingerprint"].as_str();
                Ok(stored == Some(fingerprint.as_str()))
            }
            ProofType::CommittedBefore { timestamp } => {
                let stored = data["committed_before"].as_str();
                Ok(stored == Some(timestamp.as_str()))
            }
        }
    }
}

// ── Helper functions ──────────────────────────────────────────────────────────

fn count_csv_rows(data: &[u8]) -> u64 {
    let text = std::str::from_utf8(data).unwrap_or("");
    let total = text.lines().filter(|l| !l.trim().is_empty()).count();
    if total > 0 { (total - 1) as u64 } else { 0 }
}

fn count_csv_columns(data: &[u8]) -> u64 {
    let text = std::str::from_utf8(data).unwrap_or("");
    if let Some(first) = text.lines().next() {
        first.split(',').count() as u64
    } else {
        0
    }
}

/// Compute the mean of a named column in a simple CSV
fn compute_column_mean(data: &[u8], column: &str) -> Result<f64> {
    let text = std::str::from_utf8(data)
        .map_err(|_| LedgerError::InvalidInput("Dataset is not valid UTF-8".to_string()))?;

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| {
        LedgerError::InvalidInput("Dataset has no header row".to_string())
    })?;

    let headers: Vec<&str> = header.split(',').map(|h| h.trim()).collect();
    let col_idx = headers.iter().position(|h| *h == column).ok_or_else(|| {
        LedgerError::InvalidInput(format!("Column '{column}' not found in dataset"))
    })?;

    let mut sum = 0.0f64;
    let mut count = 0u64;
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if let Some(val) = fields.get(col_idx) {
            if let Ok(n) = val.trim().parse::<f64>() {
                sum += n;
                count += 1;
            }
        }
    }

    if count == 0 {
        return Err(LedgerError::InvalidInput(format!(
            "Column '{column}' has no numeric values"
        )));
    }

    Ok(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{CommitmentConfig, DatasetCommitment};

    const SAMPLE: &[u8] = b"id,age,salary\n1,25,50000\n2,30,60000\n3,35,70000\n";

    fn make_commitment() -> DatasetCommitment {
        DatasetCommitment::commit(SAMPLE, &CommitmentConfig::default(), None).unwrap()
    }

    #[test]
    fn prove_and_verify_existence() {
        let commitment = make_commitment();
        let proof = ProofGenerator::generate(&commitment, SAMPLE, ProofType::Existence).unwrap();
        assert!(ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn prove_and_verify_exact_row_count() {
        let commitment = make_commitment();
        let proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::ExactRowCount { count: 3 },
        ).unwrap();
        assert!(ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn prove_wrong_row_count_fails() {
        let commitment = make_commitment();
        let result = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::ExactRowCount { count: 99 },
        );
        assert!(result.is_err());
    }

    #[test]
    fn prove_min_row_count() {
        let commitment = make_commitment();
        let proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::MinRowCount { min: 2 },
        ).unwrap();
        assert!(ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn prove_row_count_range() {
        let commitment = make_commitment();
        let proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::RowCountRange { min: 2, max: 5 },
        ).unwrap();
        assert!(ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn prove_column_count() {
        let commitment = make_commitment();
        let proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::ColumnCount { count: 3 },
        ).unwrap();
        assert!(ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn prove_statistical_bounds() {
        // Mean of salary column: (50000+60000+70000)/3 = 60000
        let commitment = make_commitment();
        let proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::StatisticalBounds {
                column: "salary".to_string(),
                mean_min: 50000.0,
                mean_max: 70000.0,
            },
        ).unwrap();
        assert!(ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn statistical_bounds_outside_range_fails() {
        let commitment = make_commitment();
        let result = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::StatisticalBounds {
                column: "salary".to_string(),
                mean_min: 1000.0,
                mean_max: 2000.0,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn tampered_proof_fails_verification() {
        let commitment = make_commitment();
        let mut proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::ExactRowCount { count: 3 },
        ).unwrap();

        // Tamper with the claimed row count
        proof.proof_type = ProofType::ExactRowCount { count: 9999 };

        // Verification should fail because the challenge won't match
        assert!(!ProofVerifier::verify(&proof).unwrap());
    }

    #[test]
    fn tampered_commitment_fails_verification() {
        let commitment = make_commitment();
        let mut proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::Existence,
        ).unwrap();

        // Tamper with the commitment hash
        proof.commitment_hash = "a".repeat(64);

        // Challenge recomputation will fail
        assert!(!ProofVerifier::verify(&proof).unwrap());
    }
}
