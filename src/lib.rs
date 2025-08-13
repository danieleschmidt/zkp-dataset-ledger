// Simplified ZKP Dataset Ledger for Generation 1: MAKE IT WORK

pub mod error;

// Simple modules for basic functionality
pub use error::{LedgerError, Result};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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

/// Simplified ledger for tracking datasets with persistence
#[derive(Debug)]
pub struct Ledger {
    pub name: String,
    entries: HashMap<String, LedgerEntry>,
    storage_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub dataset_name: String,
    pub dataset_hash: String,
    pub proof: Proof,
    pub timestamp: String,
    pub operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerStats {
    pub total_datasets: usize,
    pub total_operations: usize,
    pub storage_path: Option<String>,
}

impl Ledger {
    pub fn new(name: String) -> Self {
        Ledger {
            name,
            entries: HashMap::new(),
            storage_path: None,
        }
    }

    pub fn with_storage<P: AsRef<Path>>(name: String, storage_path: P) -> Result<Self> {
        let path = storage_path.as_ref().to_path_buf();
        
        // Create storage directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut ledger = Ledger {
            name,
            entries: HashMap::new(),
            storage_path: Some(path),
        };

        // Load existing entries if file exists
        ledger.load_entries()?;
        Ok(ledger)
    }

    fn load_entries(&mut self) -> Result<()> {
        if let Some(ref path) = self.storage_path {
            if path.exists() {
                let content = fs::read_to_string(path)?;
                if !content.trim().is_empty() {
                    self.entries = serde_json::from_str(&content)?;
                }
            }
        }
        Ok(())
    }

    fn save_entries(&self) -> Result<()> {
        if let Some(ref path) = self.storage_path {
            let content = serde_json::to_string_pretty(&self.entries)?;
            fs::write(path, content)?;
        }
        Ok(())
    }

    pub fn notarize_dataset(&mut self, dataset: Dataset, proof_type: String) -> Result<Proof> {
        // Check if dataset already exists
        if self.entries.contains_key(&dataset.name) {
            return Err(LedgerError::already_exists(
                format!("Dataset '{}' already exists in ledger", dataset.name)
            ));
        }

        let proof = Proof::new(dataset.hash.clone(), proof_type);
        
        let entry = LedgerEntry {
            dataset_name: dataset.name.clone(),
            dataset_hash: dataset.hash.clone(),
            proof: proof.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            operation: "notarize".to_string(),
        };

        self.entries.insert(dataset.name, entry);
        
        // Save to persistent storage
        self.save_entries()?;
        
        Ok(proof)
    }

    pub fn get_dataset_history(&self, dataset_name: &str) -> Option<&LedgerEntry> {
        self.entries.get(dataset_name)
    }

    pub fn verify_proof(&self, proof: &Proof) -> bool {
        proof.verify()
    }

    pub fn list_datasets(&self) -> Vec<&LedgerEntry> {
        self.entries.values().collect()
    }

    pub fn get_statistics(&self) -> LedgerStats {
        LedgerStats {
            total_datasets: self.entries.len(),
            total_operations: self.entries.len(), // Simplified for now
            storage_path: self.storage_path.as_ref().map(|p| p.to_string_lossy().to_string()),
        }
    }

    pub fn verify_integrity(&self) -> Result<bool> {
        // Verify all stored proofs
        for entry in self.entries.values() {
            if !self.verify_proof(&entry.proof) {
                return Ok(false);
            }
        }
        Ok(true)
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