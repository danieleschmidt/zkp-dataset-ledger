// Simplified ZKP Dataset Ledger for Generation 1: MAKE IT WORK

pub mod error;

// Simple modules for basic functionality
pub use error::{LedgerError, Result};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simplified dataset representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub hash: String,
    pub size: u64,
    pub row_count: Option<u64>,
    pub column_count: Option<u64>,
    pub path: Option<String>,
}

impl Dataset {
    pub fn new(name: String, path: String) -> Result<Self> {
        let metadata = std::fs::metadata(&path)
            .map_err(|e| LedgerError::Io(e))?;
        
        let hash = crate::simple_hash(&std::fs::read(&path)?);
        
        Ok(Dataset {
            name,
            hash,
            size: metadata.len(),
            row_count: None,
            column_count: None,
            path: Some(path),
        })
    }
}

/// Simplified proof structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    pub dataset_hash: String,
    pub proof_data: String,
    pub timestamp: String,
    pub proof_type: String,
}

impl Proof {
    pub fn new(dataset_hash: String, proof_type: String) -> Self {
        Proof {
            dataset_hash,
            proof_data: format!("simple_proof_{}", proof_type),
            timestamp: chrono::Utc::now().to_rfc3339(),
            proof_type,
        }
    }

    pub fn verify(&self) -> bool {
        !self.proof_data.is_empty() && !self.dataset_hash.is_empty()
    }
}

/// Simplified ledger for tracking datasets
#[derive(Debug)]
pub struct Ledger {
    pub name: String,
    entries: HashMap<String, LedgerEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub dataset_name: String,
    pub dataset_hash: String,
    pub proof: Proof,
    pub timestamp: String,
    pub operation: String,
}

impl Ledger {
    pub fn new(name: String) -> Self {
        Ledger {
            name,
            entries: HashMap::new(),
        }
    }

    pub fn notarize_dataset(&mut self, dataset: Dataset, proof_type: String) -> Result<Proof> {
        let proof = Proof::new(dataset.hash.clone(), proof_type);
        
        let entry = LedgerEntry {
            dataset_name: dataset.name.clone(),
            dataset_hash: dataset.hash.clone(),
            proof: proof.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            operation: "notarize".to_string(),
        };

        self.entries.insert(dataset.name, entry);
        Ok(proof)
    }

    pub fn get_dataset_history(&self, dataset_name: &str) -> Option<&LedgerEntry> {
        self.entries.get(dataset_name)
    }

    pub fn verify_proof(&self, proof: &Proof) -> bool {
        proof.verify()
    }
}

/// Simple hash function using SHA-256
pub fn simple_hash(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Configuration for simplified version
#[derive(Debug, Clone)]
pub struct Config {
    pub ledger_name: String,
    pub storage_path: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            ledger_name: "default".to_string(),
            storage_path: "./ledger_data".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_hash() {
        let data = b"hello world";
        let hash = simple_hash(data);
        assert_eq!(hash.len(), 64); // SHA-256 produces 64 hex chars
    }

    #[test]
    fn test_proof_creation_and_verification() {
        let proof = Proof::new("test_hash".to_string(), "integrity".to_string());
        assert!(proof.verify());
        assert_eq!(proof.proof_type, "integrity");
    }

    #[test]
    fn test_ledger_operations() {
        let mut ledger = Ledger::new("test_ledger".to_string());
        
        // Create a mock dataset
        let dataset = Dataset {
            name: "test_dataset".to_string(),
            hash: "test_hash".to_string(),
            size: 1000,
            row_count: Some(100),
            column_count: Some(5),
            path: None,
        };

        // Notarize dataset
        let proof = ledger.notarize_dataset(dataset.clone(), "integrity".to_string()).unwrap();
        assert!(proof.verify());

        // Check history
        let entry = ledger.get_dataset_history("test_dataset").unwrap();
        assert_eq!(entry.dataset_name, "test_dataset");
        assert_eq!(entry.operation, "notarize");
    }
}