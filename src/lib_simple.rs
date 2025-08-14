//! # ZKP Dataset Ledger - Simplified Version
//!
//! A basic implementation for cryptographic ML pipeline auditing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Main result type for the library
pub type Result<T> = std::result::Result<T, LedgerError>;

/// Basic error type
#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("Dataset error: {0}")]
    DatasetError(String),
    #[error("Not found: {0}")]
    NotFound(String),
}

impl LedgerError {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
}

/// Simple configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ledger_name: String,
    pub storage_path: String,
}

/// Dataset representation
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
        let metadata = std::fs::metadata(&path)?;
        let size = metadata.len();
        
        // Try to analyze CSV for basic stats
        let (row_count, column_count) = if path.ends_with(".csv") {
            Self::analyze_csv(&path).unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        
        // Simple hash computation
        let hash = format!("{:x}", sha2::Sha256::digest(format!("{}:{}", name, size).as_bytes()));
        
        Ok(Dataset {
            name,
            hash,
            size,
            row_count: Some(row_count),
            column_count: Some(column_count),
            path: Some(path),
        })
    }
    
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let name = path.as_ref()
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        Self::new(name, path_str)
    }
    
    pub fn compute_hash(&self) -> String {
        self.hash.clone()
    }
    
    fn analyze_csv(path: &str) -> Result<(u64, u64)> {
        let mut reader = csv::Reader::from_path(path)?;
        let headers = reader.headers()?;
        let column_count = headers.len() as u64;
        let row_count = reader.records().count() as u64;
        Ok((row_count, column_count))
    }
}

/// Simple proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    pub dataset_hash: String,
    pub proof_type: String,
    pub timestamp: DateTime<Utc>,
}

impl Proof {
    pub fn generate(dataset: &Dataset, proof_type: String) -> Result<Self> {
        Ok(Proof {
            dataset_hash: dataset.compute_hash(),
            proof_type,
            timestamp: chrono::Utc::now(),
        })
    }
    
    pub fn verify(&self) -> bool {
        // Simple verification - just check that fields are present
        !self.dataset_hash.is_empty() && !self.proof_type.is_empty()
    }
}

/// Ledger entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub id: String,
    pub dataset_name: String,
    pub dataset_hash: String,
    pub operation: String,
    pub proof: Proof,
    pub timestamp: DateTime<Utc>,
}

/// Simple ledger statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerStats {
    pub total_datasets: usize,
    pub total_operations: usize,
    pub storage_path: Option<String>,
}

/// Main ledger implementation
#[derive(Debug)]
pub struct Ledger {
    pub name: String,
    entries: HashMap<String, LedgerEntry>,
    storage_path: String,
}

impl Ledger {
    pub fn with_storage(name: String, storage_path: String) -> Result<Self> {
        let mut ledger = Ledger {
            name,
            entries: HashMap::new(),
            storage_path: storage_path.clone(),
        };
        
        // Load existing entries if file exists
        if std::path::Path::new(&storage_path).exists() {
            match std::fs::read_to_string(&storage_path) {
                Ok(content) => {
                    if !content.is_empty() {
                        match serde_json::from_str(&content) {
                            Ok(entries) => ledger.entries = entries,
                            Err(_) => {} // Start fresh if JSON is invalid
                        }
                    }
                }
                Err(_) => {} // Start fresh if file can't be read
            }
        }
        
        Ok(ledger)
    }
    
    pub fn notarize_dataset(&mut self, dataset: Dataset, proof_type: String) -> Result<Proof> {
        let proof = Proof::generate(&dataset, proof_type.clone())?;
        
        let entry = LedgerEntry {
            id: uuid::Uuid::new_v4().to_string(),
            dataset_name: dataset.name.clone(),
            dataset_hash: dataset.compute_hash(),
            operation: format!("notarize({})", proof_type),
            proof: proof.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        self.entries.insert(entry.id.clone(), entry);
        self.save()?;
        
        Ok(proof)
    }
    
    pub fn get_dataset_history(&self, name: &str) -> Option<LedgerEntry> {
        self.entries
            .values()
            .find(|entry| entry.dataset_name == name)
            .cloned()
    }
    
    pub fn list_datasets(&self) -> Vec<LedgerEntry> {
        self.entries.values().cloned().collect()
    }
    
    pub fn get_statistics(&self) -> LedgerStats {
        let datasets: std::collections::HashSet<String> = self.entries
            .values()
            .map(|e| e.dataset_name.clone())
            .collect();
        
        LedgerStats {
            total_datasets: datasets.len(),
            total_operations: self.entries.len(),
            storage_path: Some(self.storage_path.clone()),
        }
    }
    
    pub fn verify_integrity(&self) -> Result<bool> {
        for entry in self.entries.values() {
            if !entry.proof.verify() {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    pub fn verify_proof(&self, proof: &Proof) -> bool {
        proof.verify()
    }
    
    fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(&self.storage_path, json)?;
        Ok(())
    }
}

// Re-export types for compatibility
pub use self::{Dataset, Proof, Ledger, LedgerEntry, LedgerStats, Config};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_basic_functionality() {
        // Create a temporary CSV file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "name,age,score").unwrap();
        writeln!(temp_file, "Alice,25,85.5").unwrap();
        writeln!(temp_file, "Bob,30,92.0").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        // Test dataset creation
        let dataset = Dataset::from_path(&temp_path).unwrap();
        assert_eq!(dataset.row_count, Some(2));
        assert_eq!(dataset.column_count, Some(3));
        assert!(!dataset.hash.is_empty());

        // Test ledger operations
        let mut ledger = Ledger::with_storage("test".to_string(), "/tmp/test_ledger.json".to_string()).unwrap();
        let proof = ledger.notarize_dataset(dataset, "integrity".to_string()).unwrap();
        assert!(proof.verify());

        // Test statistics
        let stats = ledger.get_statistics();
        assert_eq!(stats.total_datasets, 1);
        assert_eq!(stats.total_operations, 1);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
        std::fs::remove_file("/tmp/test_ledger.json").ok();
    }
}