//! Dataset Ledger — append-only log of dataset commitments and proofs
//!
//! The ledger provides an audit trail for ML dataset provenance.
//! Each entry records a commitment and associated proofs.
//!
//! ## Ledger Design
//!
//! The ledger is an append-only log where each entry contains:
//! - A dataset commitment (the public hash)
//! - Any proofs generated about that dataset
//! - A chain hash linking to the previous entry (Merkle-like chaining)
//!
//! The chain hash ensures tamper-evidence: modifying any entry would
//! invalidate all subsequent entries.
//!
//! ```text
//! Entry 0          Entry 1          Entry 2
//! ┌─────────┐      ┌─────────┐      ┌─────────┐
//! │ commit0 │      │ commit1 │      │ commit2 │
//! │ proofs  │      │ proofs  │      │ proofs  │
//! │ chain=0 │ ───> │ chain=  │ ───> │ chain=  │
//! └─────────┘      │ H(e0)   │      │ H(e1)   │
//!                  └─────────┘      └─────────┘
//! ```

use crate::commitment::DatasetCommitment;
use crate::crypto::sha256_hex;
use crate::proof::DatasetProof;
use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A single entry in the dataset ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Sequential index (0-based)
    pub index: usize,
    /// The dataset commitment
    pub commitment: DatasetCommitment,
    /// Zero-knowledge proofs associated with this commitment
    pub proofs: Vec<DatasetProof>,
    /// Hash of the previous entry (empty string for first entry)
    /// Provides tamper-evidence — changing any entry breaks the chain
    pub previous_hash: String,
    /// Hash of this entry (used as `previous_hash` for the next entry)
    pub entry_hash: String,
    /// When this entry was appended
    pub appended_at: String,
}

/// An append-only ledger of dataset commitments and proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetLedger {
    /// The ledger's name/identifier
    pub name: String,
    /// All entries in order
    pub entries: Vec<LedgerEntry>,
    /// Creation timestamp
    pub created_at: String,
}

impl DatasetLedger {
    /// Create a new empty ledger
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Load a ledger from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let ledger: Self = serde_json::from_str(&content)?;
        Ok(ledger)
    }

    /// Save the ledger to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Append a new commitment (without proofs)
    pub fn append(&mut self, commitment: DatasetCommitment) -> Result<usize> {
        self.append_with_proofs(commitment, vec![])
    }

    /// Append a commitment with associated proofs
    pub fn append_with_proofs(
        &mut self,
        commitment: DatasetCommitment,
        proofs: Vec<DatasetProof>,
    ) -> Result<usize> {
        let index = self.entries.len();

        // Chain hash: links this entry to the previous one
        let previous_hash = if let Some(prev) = self.entries.last() {
            prev.entry_hash.clone()
        } else {
            "genesis".to_string()
        };

        // Public commitment (no nonce)
        let public_commitment = commitment.public_commitment();

        // Compute entry hash
        let entry_content = serde_json::to_string(&(&index, &public_commitment.commitment_hash, &previous_hash))
            .map_err(|e| LedgerError::Serialization(e))?;
        let entry_hash = sha256_hex(entry_content.as_bytes());

        let entry = LedgerEntry {
            index,
            commitment: public_commitment,
            proofs,
            previous_hash,
            entry_hash,
            appended_at: chrono::Utc::now().to_rfc3339(),
        };

        self.entries.push(entry);
        Ok(index)
    }

    /// Verify the integrity of the entire ledger (chain hash verification)
    ///
    /// Returns true if no entries have been tampered with.
    pub fn verify_integrity(&self) -> bool {
        let mut prev_hash = "genesis".to_string();

        for entry in &self.entries {
            if entry.previous_hash != prev_hash {
                return false;
            }

            // Recompute entry hash
            let entry_content = serde_json::to_string(&(
                &entry.index,
                &entry.commitment.commitment_hash,
                &entry.previous_hash,
            ));
            match entry_content {
                Ok(content) => {
                    let expected = sha256_hex(content.as_bytes());
                    if expected != entry.entry_hash {
                        return false;
                    }
                }
                Err(_) => return false,
            }

            prev_hash = entry.entry_hash.clone();
        }

        true
    }

    /// Look up a ledger entry by commitment hash
    pub fn find_by_commitment(&self, commitment_hash: &str) -> Option<&LedgerEntry> {
        self.entries.iter().find(|e| e.commitment.commitment_hash == commitment_hash)
    }

    /// Total number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the ledger is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{CommitmentConfig, DatasetCommitment};
    use crate::proof::{ProofGenerator, ProofType};

    const SAMPLE: &[u8] = b"id,value\n1,100\n2,200\n3,300\n";

    #[test]
    fn append_and_verify_chain() {
        let mut ledger = DatasetLedger::new("test-ledger");

        let c1 = DatasetCommitment::commit(SAMPLE, &CommitmentConfig::default(), None).unwrap();
        let c2 = DatasetCommitment::commit(b"id,value\n4,400\n5,500\n", &CommitmentConfig::default(), None).unwrap();

        ledger.append(c1).unwrap();
        ledger.append(c2).unwrap();

        assert_eq!(ledger.len(), 2);
        assert!(ledger.verify_integrity());
    }

    #[test]
    fn tampered_entry_fails_integrity() {
        let mut ledger = DatasetLedger::new("test-ledger");
        let c1 = DatasetCommitment::commit(SAMPLE, &CommitmentConfig::default(), None).unwrap();
        ledger.append(c1).unwrap();

        // Tamper with the entry hash
        ledger.entries[0].entry_hash = "a".repeat(64);

        assert!(!ledger.verify_integrity());
    }

    #[test]
    fn ledger_with_proofs() {
        let mut ledger = DatasetLedger::new("test-ledger");

        let commitment = DatasetCommitment::commit(SAMPLE, &CommitmentConfig::default(), None).unwrap();
        let proof = ProofGenerator::generate(
            &commitment,
            SAMPLE,
            ProofType::ExactRowCount { count: 3 },
        ).unwrap();

        let idx = ledger.append_with_proofs(commitment, vec![proof]).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(ledger.entries[0].proofs.len(), 1);
        assert!(ledger.verify_integrity());
    }

    #[test]
    fn find_by_commitment() {
        let mut ledger = DatasetLedger::new("test-ledger");
        let c = DatasetCommitment::commit(SAMPLE, &CommitmentConfig::default(), None).unwrap();
        let hash = c.commitment_hash.clone();
        ledger.append(c).unwrap();

        assert!(ledger.find_by_commitment(&hash).is_some());
        assert!(ledger.find_by_commitment(&"a".repeat(64)).is_none());
    }

    #[test]
    fn ledger_json_roundtrip() {
        let mut ledger = DatasetLedger::new("roundtrip-ledger");
        let c = DatasetCommitment::commit(SAMPLE, &CommitmentConfig::default(), None).unwrap();
        ledger.append(c).unwrap();

        let json = serde_json::to_string(&ledger).unwrap();
        let restored: DatasetLedger = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert!(restored.verify_integrity());
    }
}
