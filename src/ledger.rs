//! Core ledger implementation for ZKP Dataset tracking.

use crate::crypto::hash::{hash_bytes, HashAlgorithm};
use crate::crypto::merkle::MerkleTree;
use crate::storage::{Storage, MemoryStorage};
use crate::{Dataset, LedgerError, Proof, ProofConfig, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// The main ledger for tracking dataset operations and proofs.
pub struct Ledger {
    pub name: String,
    pub storage: Box<dyn Storage>,
    pub config: LedgerConfig,
    entries_cache: HashMap<String, LedgerEntry>,
}

/// Configuration for the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerConfig {
    pub hash_algorithm: HashAlgorithm,
    pub enable_merkle_trees: bool,
    pub auto_verify: bool,
    pub max_cache_entries: usize,
    pub compression: bool,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::default(),
            enable_merkle_trees: true,
            auto_verify: true,
            max_cache_entries: 1000,
            compression: false,
        }
    }
}

/// A single entry in the ledger representing a dataset operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub id: String,
    pub dataset_name: String,
    pub dataset_hash: String,
    pub proof: Proof,
    pub parent_hash: Option<String>,
    pub operation: Operation,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub merkle_root: Option<String>,
    pub block_height: u64,
    pub metadata: HashMap<String, String>,
}

/// Types of operations that can be recorded in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Initial dataset notarization
    Notarize { 
        dataset_size: u64,
        row_count: Option<u64>,
        column_count: Option<u64>,
    },
    /// Data transformation operation
    Transform { 
        from: String, 
        operation: String,
        parameters: HashMap<String, String>,
    },
    /// Train/test split operation
    Split { 
        ratio: f64, 
        seed: Option<u64>,
        stratify: Option<String>,
    },
    /// Data validation operation
    Validate {
        validation_type: String,
        passed: bool,
        details: Option<String>,
    },
    /// Custom operation
    Custom {
        operation_type: String,
        parameters: HashMap<String, String>,
    },
}

/// Query parameters for ledger history searches.
#[derive(Debug, Clone)]
pub struct LedgerQuery {
    pub dataset_name: Option<String>,
    pub operation_type: Option<String>,
    pub time_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    pub include_proofs: bool,
    pub limit: Option<usize>,
}

impl Default for LedgerQuery {
    fn default() -> Self {
        Self {
            dataset_name: None,
            operation_type: None,
            time_range: None,
            include_proofs: true,
            limit: Some(100),
        }
    }
}

/// Summary of ledger state and statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerSummary {
    pub name: String,
    pub total_entries: usize,
    pub datasets_tracked: usize,
    pub operations_by_type: HashMap<String, usize>,
    pub latest_entry_time: Option<chrono::DateTime<chrono::Utc>>,
    pub storage_size_bytes: u64,
    pub integrity_status: bool,
}

impl Ledger {
    /// Create a new ledger with the given name and default configuration.
    pub fn new(name: &str) -> Result<Self> {
        let storage = Box::new(MemoryStorage::new());
        Self::with_storage(name, storage, LedgerConfig::default())
    }

    /// Create a new ledger with custom storage backend.
    pub fn with_storage(
        name: &str,
        storage: Box<dyn Storage>,
        config: LedgerConfig,
    ) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            storage,
            config,
            entries_cache: HashMap::new(),
        })
    }

    /// Notarize a dataset by creating a cryptographic proof and ledger entry.
    pub fn notarize_dataset(
        &mut self,
        dataset: Dataset,
        name: &str,
        config: ProofConfig,
    ) -> Result<Proof> {
        // Generate the proof
        let proof = Proof::generate(&dataset, &config)?;

        // Create the ledger entry
        let entry_id = Uuid::new_v4().to_string();
        let block_height = self.get_latest_block_height()? + 1;
        
        let operation = Operation::Notarize {
            dataset_size: dataset.size,
            row_count: dataset.row_count,
            column_count: dataset.column_count,
        };

        let merkle_root = if self.config.enable_merkle_trees {
            self.compute_merkle_root(&[&proof])?
        } else {
            None
        };

        let entry = LedgerEntry {
            id: entry_id.clone(),
            dataset_name: name.to_string(),
            dataset_hash: dataset.compute_hash(),
            proof: proof.clone(),
            parent_hash: self.get_latest_entry_hash()?,
            operation,
            timestamp: chrono::Utc::now(),
            merkle_root,
            block_height,
            metadata: HashMap::new(),
        };

        // Store the entry
        self.store_entry(&entry)?;
        
        // Update cache
        if self.entries_cache.len() < self.config.max_cache_entries {
            self.entries_cache.insert(entry_id, entry);
        }

        // Auto-verify if enabled
        if self.config.auto_verify {
            if !proof.verify()? {
                return Err(LedgerError::ProofVerificationFailed);
            }
        }

        Ok(proof)
    }

    /// Record a transformation operation on a dataset.
    pub fn record_transformation(
        &mut self,
        from_dataset: &str,
        to_dataset: &str,
        operation: &str,
        parameters: HashMap<String, String>,
        proof: Proof,
    ) -> Result<String> {
        let entry_id = Uuid::new_v4().to_string();
        let block_height = self.get_latest_block_height()? + 1;

        let operation = Operation::Transform {
            from: from_dataset.to_string(),
            operation: operation.to_string(),
            parameters,
        };

        let merkle_root = if self.config.enable_merkle_trees {
            self.compute_merkle_root(&[&proof])?
        } else {
            None
        };

        let entry = LedgerEntry {
            id: entry_id.clone(),
            dataset_name: to_dataset.to_string(),
            dataset_hash: proof.dataset_hash.clone(),
            proof,
            parent_hash: self.get_latest_entry_hash()?,
            operation,
            timestamp: chrono::Utc::now(),
            merkle_root,
            block_height,
            metadata: HashMap::new(),
        };

        self.store_entry(&entry)?;
        
        if self.entries_cache.len() < self.config.max_cache_entries {
            self.entries_cache.insert(entry_id.clone(), entry);
        }

        Ok(entry_id)
    }

    /// Record a data split operation.
    pub fn record_split(
        &mut self,
        dataset_name: &str,
        ratio: f64,
        seed: Option<u64>,
        stratify: Option<String>,
        proof: Proof,
    ) -> Result<String> {
        let entry_id = Uuid::new_v4().to_string();
        let block_height = self.get_latest_block_height()? + 1;

        let operation = Operation::Split {
            ratio,
            seed,
            stratify,
        };

        let merkle_root = if self.config.enable_merkle_trees {
            self.compute_merkle_root(&[&proof])?
        } else {
            None
        };

        let entry = LedgerEntry {
            id: entry_id.clone(),
            dataset_name: dataset_name.to_string(),
            dataset_hash: proof.dataset_hash.clone(),
            proof,
            parent_hash: self.get_latest_entry_hash()?,
            operation,
            timestamp: chrono::Utc::now(),
            merkle_root,
            block_height,
            metadata: HashMap::new(),
        };

        self.store_entry(&entry)?;
        
        if self.entries_cache.len() < self.config.max_cache_entries {
            self.entries_cache.insert(entry_id.clone(), entry);
        }

        Ok(entry_id)
    }

    /// Verify a proof independently of the ledger.
    pub fn verify_proof(&self, proof: &Proof) -> Result<bool> {
        proof.verify()
    }

    /// Get the complete history for a dataset.
    pub fn get_dataset_history(&self, name: &str) -> Result<Vec<LedgerEntry>> {
        let query = LedgerQuery {
            dataset_name: Some(name.to_string()),
            ..LedgerQuery::default()
        };
        self.query_entries(&query)
    }

    /// Query ledger entries based on criteria.
    pub fn query_entries(&self, query: &LedgerQuery) -> Result<Vec<LedgerEntry>> {
        let all_keys = self.storage.list_keys("entry:")?;
        let mut matching_entries = Vec::new();

        for key in all_keys {
            if let Some(entry_data) = self.storage.get(&key)? {
                let entry: LedgerEntry = bincode::deserialize(&entry_data)
                    .map_err(|e| LedgerError::Serialization(e))?;

                // Apply filters
                if let Some(dataset_name) = &query.dataset_name {
                    if entry.dataset_name != *dataset_name {
                        continue;
                    }
                }

                if let Some(operation_type) = &query.operation_type {
                    if entry.operation.operation_type() != *operation_type {
                        continue;
                    }
                }

                if let Some((start, end)) = query.time_range {
                    if entry.timestamp < start || entry.timestamp > end {
                        continue;
                    }
                }

                matching_entries.push(entry);
            }
        }

        // Sort by timestamp
        matching_entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Apply limit
        if let Some(limit) = query.limit {
            matching_entries.truncate(limit);
        }

        Ok(matching_entries)
    }

    /// Get a summary of the ledger state.
    pub fn get_summary(&self) -> Result<LedgerSummary> {
        let all_entries = self.query_entries(&LedgerQuery::default())?;
        
        let total_entries = all_entries.len();
        let mut datasets_tracked = std::collections::HashSet::new();
        let mut operations_by_type = HashMap::new();
        let mut latest_entry_time = None;

        for entry in &all_entries {
            datasets_tracked.insert(entry.dataset_name.clone());
            
            let op_type = entry.operation.operation_type();
            *operations_by_type.entry(op_type).or_insert(0) += 1;

            if latest_entry_time.is_none() || entry.timestamp > latest_entry_time.unwrap() {
                latest_entry_time = Some(entry.timestamp);
            }
        }

        let storage_stats = self.storage.stats()?;
        let integrity_status = self.verify_chain_integrity()?;

        Ok(LedgerSummary {
            name: self.name.clone(),
            total_entries,
            datasets_tracked: datasets_tracked.len(),
            operations_by_type,
            latest_entry_time,
            storage_size_bytes: storage_stats.total_size_bytes,
            integrity_status,
        })
    }

    /// Verify the integrity of the entire ledger chain.
    pub fn verify_chain_integrity(&self) -> Result<bool> {
        let entries = self.query_entries(&LedgerQuery {
            limit: None,
            ..LedgerQuery::default()
        })?;

        // Check each entry's proof
        for entry in &entries {
            if !entry.proof.verify()? {
                return Ok(false);
            }
        }

        // Check parent hash chain
        for i in 1..entries.len() {
            let current = &entries[i];
            let previous = &entries[i - 1];
            
            if let Some(parent_hash) = &current.parent_hash {
                let expected_parent_hash = self.compute_entry_hash(previous)?;
                if *parent_hash != expected_parent_hash {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Export ledger entries to JSON format.
    pub fn export_to_json(&self, query: &LedgerQuery) -> Result<String> {
        let entries = self.query_entries(query)?;
        serde_json::to_string_pretty(&entries)
            .map_err(|e| LedgerError::Json(e))
    }

    /// Helper methods
    fn store_entry(&mut self, entry: &LedgerEntry) -> Result<()> {
        let key = format!("entry:{}", entry.id);
        let data = if self.config.compression {
            // Would implement compression here
            bincode::serialize(entry).map_err(|e| LedgerError::Serialization(e))?
        } else {
            bincode::serialize(entry).map_err(|e| LedgerError::Serialization(e))?
        };
        
        self.storage.put(&key, &data)?;
        Ok(())
    }

    fn get_latest_block_height(&self) -> Result<u64> {
        let entries = self.query_entries(&LedgerQuery {
            limit: Some(1),
            ..LedgerQuery::default()
        })?;
        Ok(entries.last().map(|e| e.block_height).unwrap_or(0))
    }

    fn get_latest_entry_hash(&self) -> Result<Option<String>> {
        let entries = self.query_entries(&LedgerQuery {
            limit: Some(1),
            ..LedgerQuery::default()
        })?;
        
        if let Some(entry) = entries.last() {
            Ok(Some(self.compute_entry_hash(entry)?))
        } else {
            Ok(None)
        }
    }

    fn compute_entry_hash(&self, entry: &LedgerEntry) -> Result<String> {
        let entry_data = format!(
            "{}:{}:{}:{}",
            entry.id,
            entry.dataset_hash,
            entry.timestamp.timestamp(),
            entry.block_height
        );
        hash_bytes(entry_data.as_bytes(), self.config.hash_algorithm.clone())
    }

    fn compute_merkle_root(&self, proofs: &[&Proof]) -> Result<Option<String>> {
        if proofs.is_empty() {
            return Ok(None);
        }

        let leaves: Vec<Vec<u8>> = proofs
            .iter()
            .map(|proof| proof.dataset_hash.as_bytes().to_vec())
            .collect();

        let tree = MerkleTree::new(leaves, self.config.hash_algorithm.clone())?;
        Ok(Some(tree.root_hash().to_string()))
    }
}

impl Operation {
    /// Get a string representation of the operation type.
    pub fn operation_type(&self) -> String {
        match self {
            Operation::Notarize { .. } => "notarize".to_string(),
            Operation::Transform { .. } => "transform".to_string(),
            Operation::Split { .. } => "split".to_string(),
            Operation::Validate { .. } => "validate".to_string(),
            Operation::Custom { operation_type, .. } => operation_type.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ledger_creation() {
        let ledger = Ledger::new("test-ledger").unwrap();
        assert_eq!(ledger.name, "test-ledger");
    }

    #[test]
    fn test_dataset_notarization() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "id,value").unwrap();
        writeln!(temp_file, "1,100").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let mut ledger = Ledger::new("test-ledger").unwrap();
        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();

        let proof = ledger.notarize_dataset(dataset, "test-dataset", config).unwrap();
        assert!(!proof.dataset_hash.is_empty());

        let history = ledger.get_dataset_history("test-dataset").unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].dataset_name, "test-dataset");

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_transformation_recording() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "data").unwrap();
        writeln!(temp_file, "test").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let mut ledger = Ledger::new("test-ledger").unwrap();
        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();

        // First notarize the original dataset
        ledger.notarize_dataset(dataset.clone(), "original", config.clone()).unwrap();

        // Then record a transformation
        let transform_proof = Proof::generate(&dataset, &config).unwrap();
        let mut params = HashMap::new();
        params.insert("operation".to_string(), "normalize".to_string());

        let transform_id = ledger.record_transformation(
            "original",
            "transformed",
            "normalize",
            params,
            transform_proof,
        ).unwrap();

        assert!(!transform_id.is_empty());

        let history = ledger.get_dataset_history("transformed").unwrap();
        assert_eq!(history.len(), 1);

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_ledger_summary() {
        let mut ledger = Ledger::new("test-ledger").unwrap();
        
        // Create a test dataset and notarize it
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test_data").unwrap();
        writeln!(temp_file, "value").unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();

        ledger.notarize_dataset(dataset, "test-dataset", config).unwrap();

        let summary = ledger.get_summary().unwrap();
        assert_eq!(summary.name, "test-ledger");
        assert_eq!(summary.total_entries, 1);
        assert_eq!(summary.datasets_tracked, 1);
        assert!(summary.operations_by_type.contains_key("notarize"));

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_chain_integrity() {
        let mut ledger = Ledger::new("test-ledger").unwrap();
        
        // Add multiple entries
        for i in 0..3 {
            let mut temp_file = NamedTempFile::new().unwrap();
            writeln!(temp_file, "id,value").unwrap();
            writeln!(temp_file, "{},data_{}", i, i).unwrap();

            let temp_path = temp_file.path().with_extension("csv");
            std::fs::copy(temp_file.path(), &temp_path).unwrap();

            let dataset = Dataset::from_path(&temp_path).unwrap();
            let config = ProofConfig::default();

            ledger.notarize_dataset(dataset, &format!("dataset-{}", i), config).unwrap();

            std::fs::remove_file(temp_path).ok();
        }

        // Verify chain integrity
        assert!(ledger.verify_chain_integrity().unwrap());
    }
}
