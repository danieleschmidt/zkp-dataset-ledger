//! # ZKP Dataset Ledger - Simplified Version
//!
//! A basic implementation for cryptographic ML pipeline auditing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use std::sync::{Arc, RwLock};
use dashmap::DashMap;
use rayon::prelude::*;

/// Main result type for the library
pub type Result<T> = std::result::Result<T, LedgerError>;

/// Enhanced error type for comprehensive error handling
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
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Security violation: {0}")]
    SecurityViolation(String),
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    #[error("Data integrity error: {0}")]
    DataIntegrityError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Storage error: {0}")]
    StorageError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
}

impl LedgerError {
    pub fn not_found(entity: &str, msg: impl Into<String>) -> Self {
        Self::NotFound(format!("{}: {}", entity, msg.into()))
    }
    
    pub fn validation_error(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }
    
    pub fn security_violation(msg: impl Into<String>) -> Self {
        Self::SecurityViolation(msg.into())
    }
    
    pub fn data_integrity_error(msg: impl Into<String>) -> Self {
        Self::DataIntegrityError(msg.into())
    }
}

/// Simple configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ledger_name: String,
    pub storage_path: String,
}

/// Dataset format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatasetFormat {
    Csv,
    Json,
    Parquet,
    Binary,
}

/// Dataset schema placeholder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSchema {
    pub columns: Vec<String>,
}

/// Dataset statistics placeholder  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStatistics {
    pub row_count: u64,
    pub column_count: u64,
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
    pub schema: Option<DatasetSchema>,
    pub statistics: Option<DatasetStatistics>,
    pub format: DatasetFormat,
}

impl Dataset {
    pub fn new(name: String, path: String) -> Result<Self> {
        // Input validation
        if name.is_empty() {
            return Err(LedgerError::validation_error("Dataset name cannot be empty"));
        }
        
        if name.len() > 255 {
            return Err(LedgerError::validation_error("Dataset name cannot exceed 255 characters"));
        }
        
        // Validate name contains only safe characters
        if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.') {
            return Err(LedgerError::validation_error("Dataset name can only contain alphanumeric characters, hyphens, underscores, and dots"));
        }
        
        if path.is_empty() {
            return Err(LedgerError::validation_error("Dataset path cannot be empty"));
        }
        
        // Check file exists and is readable
        if !std::path::Path::new(&path).exists() {
            return Err(LedgerError::not_found("file", format!("Dataset file not found: {}", path)));
        }
        
        let metadata = std::fs::metadata(&path)?;
        let size = metadata.len();
        
        // Validate file size (max 1GB for simple implementation)
        if size > 1_073_741_824 {
            return Err(LedgerError::validation_error("Dataset file exceeds 1GB limit"));
        }
        
        // Try to analyze CSV for basic stats
        let (row_count, column_count) = if path.ends_with(".csv") {
            Self::analyze_csv(&path).unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        
        // Simple hash computation
        let hash = format!("{:x}", Sha256::digest(format!("{}:{}", name, size).as_bytes()));
        
        let format = if path.ends_with(".csv") {
            DatasetFormat::Csv
        } else if path.ends_with(".json") {
            DatasetFormat::Json
        } else {
            DatasetFormat::Binary
        };

        Ok(Dataset {
            name,
            hash,
            size,
            row_count: Some(row_count),
            column_count: Some(column_count),
            path: Some(path),
            schema: None,
            statistics: None,
            format,
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
        let start_time = std::time::Instant::now();
        
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| LedgerError::ValidationError(format!("Failed to read CSV file: {}", e)))?;
        
        let headers = reader.headers()
            .map_err(|e| LedgerError::ValidationError(format!("Invalid CSV headers: {}", e)))?;
        
        let column_count = headers.len() as u64;
        
        // Validate CSV has at least one column
        if column_count == 0 {
            return Err(LedgerError::validation_error("CSV file has no columns"));
        }
        
        // For large files, use memory-mapped reading for better performance
        let file_size = std::fs::metadata(path)?.len();
        
        let row_count = if file_size > 10_000_000 { // > 10MB, use optimized approach
            log::info!("Using optimized processing for large CSV file ({}MB)", file_size / 1_000_000);
            Self::analyze_large_csv(path, column_count)?
        } else {
            Self::analyze_small_csv(reader, column_count)?
        };
        
        let duration = start_time.elapsed();
        log::info!("CSV analysis completed in {:?}: {} rows, {} columns", 
            duration, row_count, column_count);
        
        Ok((row_count, column_count))
    }
    
    /// Optimized analysis for small CSV files
    fn analyze_small_csv(mut reader: csv::Reader<std::fs::File>, column_count: u64) -> Result<u64> {
        let mut row_count = 0u64;
        
        for result in reader.records() {
            match result {
                Ok(record) => {
                    // Validate record has correct number of fields
                    if record.len() != column_count as usize {
                        return Err(LedgerError::validation_error(
                            format!("Row {} has {} fields but expected {}", row_count + 1, record.len(), column_count)
                        ));
                    }
                    row_count += 1;
                }
                Err(e) => return Err(LedgerError::ValidationError(format!("Invalid CSV row {}: {}", row_count + 1, e))),
            }
        }
        
        Ok(row_count)
    }
    
    /// Optimized analysis for large CSV files using parallel processing
    fn analyze_large_csv(path: &str, column_count: u64) -> Result<u64> {
        // Read file in chunks for parallel processing
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.is_empty() {
            return Ok(0);
        }
        
        // Skip header line
        let data_lines = &lines[1..];
        
        if data_lines.len() > 10_000_000 {
            return Err(LedgerError::validation_error("CSV file has too many rows (>10M)"));
        }
        
        // Process lines in parallel chunks
        let chunk_size = std::cmp::max(1000, data_lines.len() / num_cpus::get());
        
        let validation_results: Result<Vec<()>> = data_lines
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                for (line_idx, line) in chunk.iter().enumerate() {
                    let field_count = line.split(',').count();
                    if field_count != column_count as usize {
                        let row_number = chunk_idx * chunk_size + line_idx + 2; // +2 for 1-based and header
                        return Err(LedgerError::validation_error(
                            format!("Row {} has {} fields but expected {}", row_number, field_count, column_count)
                        ));
                    }
                }
                Ok(())
            })
            .collect();
        
        validation_results?;
        
        Ok(data_lines.len() as u64)
    }
}

/// Simple proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof {
    pub dataset_hash: String,
    pub proof_type: String,
    pub timestamp: DateTime<Utc>,
}

/// Health status for ledger monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub last_check: DateTime<Utc>,
    pub storage_accessible: bool,
    pub integrity_verified: bool,
    pub entry_count: usize,
    pub storage_size_bytes: u64,
    pub issues: Vec<String>,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_operations: u64,
    pub average_proof_time_ms: f64,
    pub average_verify_time_ms: f64,
    pub storage_utilization: f64,
    pub cache_hit_rate: f64,
    pub parallel_operations: u64,
    pub memory_usage_mb: f64,
    pub throughput_ops_per_sec: f64,
}

/// Cache entry for performance optimization
#[derive(Debug, Clone)]
struct CacheEntry {
    pub data: Vec<u8>,
    #[allow(dead_code)]
    pub hash: String,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
}

/// High-performance cache with LRU eviction
#[derive(Debug)]
pub struct PerformanceCache {
    entries: Arc<DashMap<String, CacheEntry>>,
    max_entries: usize,
    #[allow(dead_code)]
    max_size_bytes: usize,
    hits: Arc<parking_lot::Mutex<u64>>,
    misses: Arc<parking_lot::Mutex<u64>>,
}

impl PerformanceCache {
    pub fn new(max_entries: usize, max_size_mb: usize) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            max_entries,
            max_size_bytes: max_size_mb * 1024 * 1024,
            hits: Arc::new(parking_lot::Mutex::new(0)),
            misses: Arc::new(parking_lot::Mutex::new(0)),
        }
    }
    
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(mut entry) = self.entries.get_mut(key) {
            entry.last_accessed = Utc::now();
            entry.access_count += 1;
            *self.hits.lock() += 1;
            Some(entry.data.clone())
        } else {
            *self.misses.lock() += 1;
            None
        }
    }
    
    pub fn put(&self, key: String, data: Vec<u8>, hash: String) {
        // Check cache size limits
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        
        let entry = CacheEntry {
            data,
            hash,
            last_accessed: Utc::now(),
            access_count: 1,
        };
        
        self.entries.insert(key, entry);
    }
    
    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.lock();
        let misses = *self.misses.lock();
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    fn evict_lru(&self) {
        // Simple LRU eviction - remove oldest accessed entry
        let mut oldest_key = String::new();
        let mut oldest_time = Utc::now();
        
        for entry in self.entries.iter() {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_key = entry.key().clone();
            }
        }
        
        if !oldest_key.is_empty() {
            self.entries.remove(&oldest_key);
        }
    }
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
        // Enhanced verification with security checks
        if self.dataset_hash.is_empty() || self.proof_type.is_empty() {
            return false;
        }
        
        // Validate hash format (SHA-256 produces 64 hex characters)
        if self.dataset_hash.len() != 64 {
            return false;
        }
        
        if !self.dataset_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }
        
        // Validate proof type contains only safe characters
        if !self.proof_type.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            return false;
        }
        
        // Validate timestamp is not in the future (with 5 minute tolerance)
        let now = chrono::Utc::now();
        let tolerance = chrono::Duration::minutes(5);
        if self.timestamp > now + tolerance {
            return false;
        }
        
        // Validate timestamp is not too old (1 year limit)
        let max_age = chrono::Duration::days(365);
        if self.timestamp < now - max_age {
            return false;
        }
        
        true
    }
    
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "proof_type": self.proof_type,
            "timestamp": self.timestamp,
            "dataset_hash": self.dataset_hash,
            "verified": self.verify(),
        })
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

/// Main ledger implementation with performance optimization
#[derive(Debug)]
pub struct Ledger {
    pub name: String,
    entries: HashMap<String, LedgerEntry>,
    storage_path: String,
    cache: PerformanceCache,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    operation_start_times: Arc<DashMap<String, std::time::Instant>>,
}

impl Ledger {
    pub fn with_storage(name: String, storage_path: String) -> Result<Self> {
        let cache = PerformanceCache::new(1000, 100); // 1000 entries, 100MB max
        let metrics = Arc::new(RwLock::new(PerformanceMetrics {
            total_operations: 0,
            average_proof_time_ms: 0.0,
            average_verify_time_ms: 0.0,
            storage_utilization: 0.0,
            cache_hit_rate: 0.0,
            parallel_operations: 0,
            memory_usage_mb: 0.0,
            throughput_ops_per_sec: 0.0,
        }));
        
        let mut ledger = Ledger {
            name,
            entries: HashMap::new(),
            storage_path: storage_path.clone(),
            cache,
            performance_metrics: metrics,
            operation_start_times: Arc::new(DashMap::new()),
        };
        
        // Load existing entries if file exists
        if std::path::Path::new(&storage_path).exists() {
            if let Ok(content) = std::fs::read_to_string(&storage_path) {
                if !content.is_empty() {
                    if let Ok(entries) = serde_json::from_str(&content) {
                        ledger.entries = entries;
                    }
                }
            }
        }
        
        Ok(ledger)
    }
    
    pub fn notarize_dataset(&mut self, dataset: Dataset, proof_type: String) -> Result<Proof> {
        let operation_id = uuid::Uuid::new_v4().to_string();
        let start_time = std::time::Instant::now();
        self.operation_start_times.insert(operation_id.clone(), start_time);
        
        // Validate inputs
        if proof_type.is_empty() {
            return Err(LedgerError::validation_error("Proof type cannot be empty"));
        }
        
        if proof_type.len() > 100 {
            return Err(LedgerError::validation_error("Proof type cannot exceed 100 characters"));
        }
        
        // Check if dataset name already exists
        if self.entries.values().any(|entry| entry.dataset_name == dataset.name) {
            return Err(LedgerError::validation_error(
                format!("Dataset '{}' already exists in ledger", dataset.name)
            ));
        }
        
        // Check cache for existing dataset hash computation
        let cache_key = format!("dataset_hash_{}", dataset.name);
        let proof = if let Some(_cached_data) = self.cache.get(&cache_key) {
            log::debug!("Cache hit for dataset: {}", dataset.name);
            Proof::generate(&dataset, proof_type.clone())?
        } else {
            log::debug!("Cache miss for dataset: {}", dataset.name);
            let proof = Proof::generate(&dataset, proof_type.clone())?;
            
            // Cache the dataset hash
            self.cache.put(cache_key, dataset.hash.as_bytes().to_vec(), dataset.hash.clone());
            proof
        };
        
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
        
        // Update performance metrics
        if let Some(start) = self.operation_start_times.remove(&operation_id) {
            let duration = start.1.elapsed();
            let duration_ms = duration.as_millis() as f64;
            
            if let Ok(mut metrics) = self.performance_metrics.write() {
                metrics.total_operations += 1;
                metrics.cache_hit_rate = self.cache.hit_rate();
                
                // Update average proof time
                let total_ops = metrics.total_operations as f64;
                metrics.average_proof_time_ms = 
                    (metrics.average_proof_time_ms * (total_ops - 1.0) + duration_ms) / total_ops;
                    
                log::info!("Operation completed in {}ms, cache hit rate: {:.2}%", 
                    duration_ms, metrics.cache_hit_rate * 100.0);
            }
        }
        
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
        log::info!("Starting ledger integrity verification");
        
        if self.entries.is_empty() {
            log::info!("Ledger is empty - integrity check passed");
            return Ok(true);
        }
        
        let mut verification_errors = Vec::new();
        
        for (entry_id, entry) in &self.entries {
            // Verify entry ID format
            if entry.id != *entry_id {
                verification_errors.push(format!("Entry ID mismatch: key '{}' vs entry.id '{}'", entry_id, entry.id));
                continue;
            }
            
            // Verify UUID format
            if uuid::Uuid::parse_str(&entry.id).is_err() {
                verification_errors.push(format!("Invalid UUID format in entry: {}", entry.id));
            }
            
            // Verify proof
            if !entry.proof.verify() {
                verification_errors.push(format!("Proof verification failed for entry: {}", entry.id));
            }
            
            // Verify dataset name consistency
            if entry.dataset_name.is_empty() {
                verification_errors.push(format!("Empty dataset name in entry: {}", entry.id));
            }
            
            // Verify hash consistency
            if entry.dataset_hash != entry.proof.dataset_hash {
                verification_errors.push(format!("Hash mismatch in entry {}: ledger='{}' vs proof='{}'", 
                    entry.id, entry.dataset_hash, entry.proof.dataset_hash));
            }
            
            // Check for duplicate dataset names
            let duplicate_count = self.entries.values()
                .filter(|e| e.dataset_name == entry.dataset_name)
                .count();
            
            if duplicate_count > 1 {
                verification_errors.push(format!("Duplicate dataset name '{}' found in entry: {}", 
                    entry.dataset_name, entry.id));
            }
        }
        
        if !verification_errors.is_empty() {
            log::warn!("Integrity verification failed with {} errors:", verification_errors.len());
            for error in &verification_errors {
                log::warn!("  - {}", error);
            }
            Ok(false)
        } else {
            log::info!("Ledger integrity verification passed for {} entries", self.entries.len());
            Ok(true)
        }
    }
    
    pub fn verify_proof(&self, proof: &Proof) -> bool {
        proof.verify()
    }
    
    pub fn health_check(&self) -> Result<HealthStatus> {
        let storage_accessible = std::path::Path::new(&self.storage_path).exists();
        let integrity_verified = self.verify_integrity().unwrap_or(false);
        let entry_count = self.entries.len();
        let storage_size_bytes = if storage_accessible {
            std::fs::metadata(&self.storage_path)
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };
        
        let mut issues = Vec::new();
        if !storage_accessible {
            issues.push("Storage file not accessible".to_string());
        }
        if !integrity_verified {
            issues.push("Ledger integrity verification failed".to_string());
        }
        
        Ok(HealthStatus {
            is_healthy: storage_accessible && integrity_verified,
            last_check: chrono::Utc::now(),
            storage_accessible,
            integrity_verified,
            entry_count,
            storage_size_bytes,
            issues,
        })
    }
    
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        if let Ok(metrics) = self.performance_metrics.read() {
            let mut result = metrics.clone();
            result.cache_hit_rate = self.cache.hit_rate();
            
            // Calculate memory usage
            result.memory_usage_mb = self.estimate_memory_usage_mb();
            
            // Calculate throughput (operations per second)
            if result.average_proof_time_ms > 0.0 {
                result.throughput_ops_per_sec = 1000.0 / result.average_proof_time_ms;
            }
            
            result
        } else {
            PerformanceMetrics {
                total_operations: self.entries.len() as u64,
                average_proof_time_ms: 0.0,
                average_verify_time_ms: 0.0,
                storage_utilization: 0.0,
                cache_hit_rate: self.cache.hit_rate(),
                parallel_operations: 0,
                memory_usage_mb: self.estimate_memory_usage_mb(),
                throughput_ops_per_sec: 0.0,
            }
        }
    }
    
    /// Estimate current memory usage in MB
    fn estimate_memory_usage_mb(&self) -> f64 {
        let entry_count = self.entries.len();
        let cache_entries = self.cache.entries.len();
        
        // Rough estimation: 1KB per ledger entry, 10KB per cache entry
        let ledger_size = entry_count * 1024;
        let cache_size = cache_entries * 10240;
        
        (ledger_size + cache_size) as f64 / (1024.0 * 1024.0)
    }
    
    fn save(&self) -> Result<()> {
        // Create backup before saving
        if std::path::Path::new(&self.storage_path).exists() {
            let backup_path = format!("{}.backup.{}", 
                self.storage_path, 
                chrono::Utc::now().format("%Y%m%d_%H%M%S")
            );
            
            if let Err(e) = std::fs::copy(&self.storage_path, &backup_path) {
                log::warn!("Failed to create backup: {}", e);
            } else {
                log::info!("Created backup at: {}", backup_path);
            }
        }
        
        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&self.storage_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LedgerError::StorageError(format!("Failed to create directory: {}", e))
            })?;
        }
        
        // Serialize to JSON
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| LedgerError::StorageError(format!("Failed to serialize ledger data: {}", e)))?;
        
        // Atomic write using temporary file
        let temp_path = format!("{}.tmp", self.storage_path);
        std::fs::write(&temp_path, &json)
            .map_err(|e| LedgerError::StorageError(format!("Failed to write temporary file: {}", e)))?;
        
        // Atomic rename
        std::fs::rename(&temp_path, &self.storage_path)
            .map_err(|e| LedgerError::StorageError(format!("Failed to save ledger file: {}", e)))?;
        
        // Verify write integrity
        let verification = std::fs::read_to_string(&self.storage_path)
            .map_err(|e| LedgerError::StorageError(format!("Failed to verify saved file: {}", e)))?;
        
        if verification != json {
            return Err(LedgerError::data_integrity_error("Saved file verification failed"));
        }
        
        log::info!("Ledger saved successfully to: {}", self.storage_path);
        Ok(())
    }
}

// Types are already public and available

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