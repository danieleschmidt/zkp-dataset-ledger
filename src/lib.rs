//! # ZKP Dataset Ledger - Simplified Version
//!
//! A basic implementation for cryptographic ML pipeline auditing.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Main result type for the library
pub type Result<T> = std::result::Result<T, LedgerError>;

/// Retry strategies for error recovery
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// Linear retry with fixed delay
    Linear { max_attempts: u32, delay_ms: u64 },
    /// Exponential backoff retry
    ExponentialBackoff { max_attempts: u32, base_delay_ms: u64 },
}

impl RetryStrategy {
    /// Execute a function with retry logic
    pub async fn execute<T, F, Fut>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let (max_attempts, get_delay) = match self {
            RetryStrategy::Linear { max_attempts, delay_ms } => {
                (*max_attempts, Box::new(|_: u32| *delay_ms) as Box<dyn Fn(u32) -> u64>)
            }
            RetryStrategy::ExponentialBackoff { max_attempts, base_delay_ms } => {
                (*max_attempts, Box::new(move |attempt: u32| base_delay_ms * 2_u64.pow(attempt)) as Box<dyn Fn(u32) -> u64>)
            }
        };

        let mut last_error = None;
        
        for attempt in 0..max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error.clone());
                    
                    if attempt < max_attempts - 1 && error.is_recoverable() {
                        let delay_ms = get_delay(attempt);
                        log::warn!("Operation failed (attempt {}), retrying in {}ms: {}", attempt + 1, delay_ms, error);
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    } else {
                        log::error!("Operation failed after {} attempts: {}", attempt + 1, error);
                        break;
                    }
                }
            }
        }
        
        Err(last_error.unwrap())
    }
}

/// Comprehensive error types with recovery information
#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("I/O error: {0}")]
    Io(std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("CSV processing error: {0}")]
    Csv(#[from] csv::Error),
    #[error("Dataset validation error: {message}")]
    DatasetError { message: String, recoverable: bool },
    #[error("Resource not found: {resource} - {details}")]
    NotFound { resource: String, details: String },
    #[error("Invalid input: {field} - {reason}")]
    InvalidInput { field: String, reason: String },
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Security validation failed: {0}")]
    Security(String),
    #[error("Storage operation failed: {operation} - {reason}")]
    Storage { operation: String, reason: String },
    #[error("Proof verification failed: {reason}")]
    ProofError { reason: String, code: u32 },
}

impl LedgerError {
    pub fn not_found(resource: impl Into<String>, details: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
            details: details.into(),
        }
    }

    pub fn dataset_error(message: impl Into<String>, recoverable: bool) -> Self {
        Self::DatasetError {
            message: message.into(),
            recoverable,
        }
    }

    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            field: field.into(),
            reason: reason.into(),
        }
    }

    pub fn storage_error(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Storage {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    pub fn proof_error(reason: impl Into<String>, code: u32) -> Self {
        Self::ProofError {
            reason: reason.into(),
            code,
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::DatasetError { recoverable, .. } => *recoverable,
            Self::Io(_) | Self::Storage { .. } => true, // May be transient
            Self::Json(_) | Self::Csv(_) => false,      // Data format issues
            Self::InvalidInput { .. } | Self::Config(_) | Self::Security(_) => false,
            Self::NotFound { .. } => false,
            Self::ProofError { code, .. } => *code < 1000, // Recoverable codes < 1000
        }
    }

    /// Get retry strategy for this error
    pub fn retry_strategy(&self) -> Option<RetryStrategy> {
        if !self.is_recoverable() {
            return None;
        }
        
        match self {
            Self::Io(_) => Some(RetryStrategy::ExponentialBackoff { max_attempts: 3, base_delay_ms: 100 }),
            Self::Storage { .. } => Some(RetryStrategy::ExponentialBackoff { max_attempts: 5, base_delay_ms: 50 }),
            Self::ProofError { code, .. } if *code < 500 => Some(RetryStrategy::Linear { max_attempts: 2, delay_ms: 200 }),
            _ => None,
        }
    }

    /// Get error category for monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::Io(_) => "io",
            Self::Json(_) => "serialization",
            Self::Csv(_) => "data_format",
            Self::DatasetError { .. } => "dataset_validation",
            Self::NotFound { .. } => "not_found",
            Self::InvalidInput { .. } => "invalid_input",
            Self::Config(_) => "configuration",
            Self::Security(_) => "security",
            Self::Storage { .. } => "storage",
            Self::ProofError { .. } => "proof",
        }
    }
}

impl Clone for LedgerError {
    fn clone(&self) -> Self {
        match self {
            Self::Io(e) => Self::Io(std::io::Error::new(e.kind(), e.to_string())),
            Self::Json(_) => Self::Json(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, "JSON error"))),
            Self::Csv(_) => Self::Csv(csv::Error::from(std::io::Error::new(std::io::ErrorKind::Other, "CSV error"))),
            Self::DatasetError { message, recoverable } => Self::DatasetError { message: message.clone(), recoverable: *recoverable },
            Self::NotFound { resource, details } => Self::NotFound { resource: resource.clone(), details: details.clone() },
            Self::InvalidInput { field, reason } => Self::InvalidInput { field: field.clone(), reason: reason.clone() },
            Self::Config(msg) => Self::Config(msg.clone()),
            Self::Security(msg) => Self::Security(msg.clone()),
            Self::Storage { operation, reason } => Self::Storage { operation: operation.clone(), reason: reason.clone() },
            Self::ProofError { reason, code } => Self::ProofError { reason: reason.clone(), code: *code },
        }
    }
}

impl From<std::io::Error> for LedgerError {
    fn from(error: std::io::Error) -> Self {
        LedgerError::Io(error)
    }
}

/// Simple configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ledger_name: String,
    pub storage_path: String,
}

/// Supported file formats
#[derive(Debug, Clone, PartialEq)]
enum FileFormat {
    Csv,
    Json,
    Unknown,
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
        // Input validation
        Self::validate_name(&name)?;
        Self::validate_path(&path)?;

        log::info!("Creating dataset '{}' from path '{}'", name, path);

        let metadata = std::fs::metadata(&path)
            .map_err(|e| LedgerError::not_found("file", format!("Path '{}': {}", path, e)))?;

        let size = metadata.len();

        // Validate file size constraints
        Self::validate_file_size(size)?;

        // Try to analyze supported file formats
        let (row_count, column_count) = match Self::get_file_format(&path)? {
            FileFormat::Csv => {
                log::debug!("Analyzing CSV file: {}", path);
                Self::analyze_csv(&path).map_err(|e| {
                    LedgerError::dataset_error(format!("CSV analysis failed: {}", e), true)
                })?
            }
            FileFormat::Json => {
                log::debug!("Analyzing JSON file: {}", path);
                Self::analyze_json(&path).unwrap_or((0, 0))
            }
            FileFormat::Unknown => {
                log::warn!("Unknown file format for: {}", path);
                (0, 0)
            }
        };

        // Compute secure hash
        let hash = Self::compute_secure_hash(&name, size, &path)?;

        log::info!(
            "Dataset created successfully: {} rows, {} columns",
            row_count,
            column_count
        );

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
        let name = path
            .as_ref()
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        Self::new(name, path_str)
    }

    pub fn compute_hash(&self) -> String {
        self.hash.clone()
    }

    /// Validate dataset name
    fn validate_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(LedgerError::invalid_input("name", "cannot be empty"));
        }

        if name.len() > 255 {
            return Err(LedgerError::invalid_input("name", "exceeds 255 characters"));
        }

        // Check for invalid characters
        if name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) {
            return Err(LedgerError::invalid_input(
                "name",
                "contains invalid characters",
            ));
        }

        Ok(())
    }

    /// Validate file path
    fn validate_path(path: &str) -> Result<()> {
        if path.is_empty() {
            return Err(LedgerError::invalid_input("path", "cannot be empty"));
        }

        let path_obj = std::path::Path::new(path);
        if !path_obj.exists() {
            return Err(LedgerError::not_found("file", path.to_string()));
        }

        if !path_obj.is_file() {
            return Err(LedgerError::invalid_input("path", "must be a file"));
        }

        Ok(())
    }

    /// Validate file size constraints
    fn validate_file_size(size: u64) -> Result<()> {
        const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024 * 1024; // 10 GB

        if size == 0 {
            return Err(LedgerError::dataset_error("File is empty", false));
        }

        if size > MAX_FILE_SIZE {
            return Err(LedgerError::dataset_error(
                format!(
                    "File size {} bytes exceeds maximum of {} bytes",
                    size, MAX_FILE_SIZE
                ),
                false,
            ));
        }

        Ok(())
    }

    /// Get file format from extension
    fn get_file_format(path: &str) -> Result<FileFormat> {
        let path_obj = std::path::Path::new(path);
        let extension = path_obj
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        match extension.as_str() {
            "csv" => Ok(FileFormat::Csv),
            "json" | "jsonl" => Ok(FileFormat::Json),
            _ => Ok(FileFormat::Unknown),
        }
    }

    /// Compute secure hash with additional entropy
    fn compute_secure_hash(name: &str, size: u64, path: &str) -> Result<String> {
        // Read first 1KB of file for hash computation
        let mut file = std::fs::File::open(path)?;
        use std::io::Read;
        let mut buffer = vec![0u8; 1024.min(size as usize)];
        file.read_exact(&mut buffer).or_else(|_| {
            // If file is smaller than buffer, read what we can
            let mut file = std::fs::File::open(path)?;
            let mut actual_buffer = Vec::new();
            file.read_to_end(&mut actual_buffer)?;
            buffer = actual_buffer;
            Ok::<(), std::io::Error>(())
        })?;

        // Include metadata in hash for uniqueness
        let hash_input = format!(
            "{}:{}:{}:{:?}",
            name,
            size,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            &buffer[..buffer.len().min(256)]
        );

        Ok(format!("{:x}", Sha256::digest(hash_input.as_bytes())))
    }

    /// Analyze CSV file structure
    fn analyze_csv(path: &str) -> Result<(u64, u64)> {
        let mut reader = csv::Reader::from_path(path)?;

        // Validate CSV headers
        let headers = reader.headers()?;
        let column_count = headers.len() as u64;

        if column_count == 0 {
            return Err(LedgerError::dataset_error("CSV has no columns", false));
        }

        if column_count > 10000 {
            return Err(LedgerError::dataset_error(
                format!("CSV has too many columns: {}", column_count),
                false,
            ));
        }

        // Count rows with progress logging
        let mut row_count = 0u64;
        for (index, record) in reader.records().enumerate() {
            match record {
                Ok(_) => row_count += 1,
                Err(e) => {
                    log::warn!("Error reading CSV row {}: {}", index, e);
                    return Err(LedgerError::dataset_error(
                        format!("CSV parsing error at row {}: {}", index, e),
                        true,
                    ));
                }
            }

            // Log progress for large files
            if row_count > 0 && row_count % 100000 == 0 {
                log::debug!("Processed {} rows", row_count);
            }
        }

        log::info!(
            "CSV analysis complete: {} rows, {} columns",
            row_count,
            column_count
        );
        Ok((row_count, column_count))
    }

    /// Analyze JSON file structure (basic implementation)
    fn analyze_json(path: &str) -> Result<(u64, u64)> {
        let content = std::fs::read_to_string(path)?;

        // Try to parse as JSON array
        match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(serde_json::Value::Array(array)) => {
                let row_count = array.len() as u64;
                let column_count = if let Some(first_obj) = array.first() {
                    if let serde_json::Value::Object(map) = first_obj {
                        map.len() as u64
                    } else {
                        1 // Single value per row
                    }
                } else {
                    0
                };
                Ok((row_count, column_count))
            }
            Ok(_) => Ok((1, 1)), // Single JSON object
            Err(e) => Err(LedgerError::dataset_error(
                format!("Invalid JSON format: {}", e),
                false,
            )),
        }
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
    /// Supported proof types
    const VALID_PROOF_TYPES: &'static [&'static str] =
        &["integrity", "schema", "statistics", "privacy", "custom"];

    pub fn generate(dataset: &Dataset, proof_type: String) -> Result<Self> {
        // Validate proof type
        Self::validate_proof_type(&proof_type)?;

        log::info!(
            "Generating {} proof for dataset: {}",
            proof_type,
            dataset.name
        );

        // Enhanced proof generation with validation
        let dataset_hash = dataset.compute_hash();
        if dataset_hash.is_empty() {
            return Err(LedgerError::proof_error("Invalid dataset hash", 1001));
        }

        let proof = Proof {
            dataset_hash,
            proof_type: proof_type.clone(),
            timestamp: chrono::Utc::now(),
        };

        // Validate generated proof
        if !proof.verify_structure()? {
            return Err(LedgerError::proof_error(
                "Generated proof failed validation",
                1002,
            ));
        }

        log::info!("Proof generated successfully for dataset: {}", dataset.name);
        Ok(proof)
    }

    pub fn verify(&self) -> bool {
        self.verify_structure().unwrap_or(false)
    }

    /// Comprehensive proof structure validation
    fn verify_structure(&self) -> Result<bool> {
        // Check hash format
        if self.dataset_hash.is_empty() {
            log::error!("Proof verification failed: empty dataset hash");
            return Ok(false);
        }

        if self.dataset_hash.len() != 64 {
            log::error!("Proof verification failed: invalid hash length");
            return Ok(false);
        }

        // Validate hash is hex
        if !self.dataset_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            log::error!("Proof verification failed: invalid hash format");
            return Ok(false);
        }

        // Validate proof type
        Self::validate_proof_type(&self.proof_type)?;

        // Check timestamp validity
        let now = chrono::Utc::now();
        if self.timestamp > now {
            log::error!("Proof verification failed: future timestamp");
            return Ok(false);
        }

        // Check timestamp is not too old (1 year)
        let one_year_ago = now - chrono::Duration::days(365);
        if self.timestamp < one_year_ago {
            log::warn!("Proof timestamp is very old: {}", self.timestamp);
        }

        log::debug!("Proof structure validation passed");
        Ok(true)
    }

    /// Validate proof type is supported
    fn validate_proof_type(proof_type: &str) -> Result<()> {
        if proof_type.is_empty() {
            return Err(LedgerError::invalid_input("proof_type", "cannot be empty"));
        }

        if !Self::VALID_PROOF_TYPES.contains(&proof_type) {
            return Err(LedgerError::invalid_input(
                "proof_type",
                format!("must be one of: {}", Self::VALID_PROOF_TYPES.join(", ")),
            ));
        }

        Ok(())
    }

    /// Get proof metadata for auditing
    pub fn metadata(&self) -> ProofMetadata {
        ProofMetadata {
            hash: self.dataset_hash.clone(),
            proof_type: self.proof_type.clone(),
            timestamp: self.timestamp,
            age_hours: chrono::Utc::now()
                .signed_duration_since(self.timestamp)
                .num_hours(),
            is_valid: self.verify(),
        }
    }
}

/// Metadata about a proof for auditing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub hash: String,
    pub proof_type: String,
    pub timestamp: DateTime<Utc>,
    pub age_hours: i64,
    pub is_valid: bool,
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

/// High-performance cache for frequently accessed data
#[derive(Debug, Clone)]
pub struct LedgerCache {
    pub dataset_hashes: Arc<DashMap<String, String>>, // dataset_name -> hash
    pub proof_cache: Arc<DashMap<String, Proof>>,     // dataset_hash -> proof
    pub metadata_cache: Arc<DashMap<String, ProofMetadata>>, // entry_id -> metadata
    pub stats_cache: Arc<RwLock<Option<(LedgerStats, Instant)>>>, // cached stats with timestamp
}

impl LedgerCache {
    pub fn new() -> Self {
        Self {
            dataset_hashes: Arc::new(DashMap::new()),
            proof_cache: Arc::new(DashMap::new()),
            metadata_cache: Arc::new(DashMap::new()),
            stats_cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.dataset_hashes.clear();
        self.proof_cache.clear();
        self.metadata_cache.clear();
        *self.stats_cache.write() = None;
        log::debug!("All caches cleared");
    }

    /// Get cached stats if not expired (cache for 5 minutes)
    pub fn get_cached_stats(&self) -> Option<LedgerStats> {
        let cache = self.stats_cache.read();
        if let Some((stats, timestamp)) = cache.as_ref() {
            if timestamp.elapsed().as_secs() < 300 {
                // 5 minutes
                return Some(stats.clone());
            }
        }
        None
    }

    /// Update cached stats
    pub fn update_stats_cache(&self, stats: LedgerStats) {
        *self.stats_cache.write() = Some((stats, Instant::now()));
    }
}

/// Optimized main ledger implementation with caching
#[derive(Debug)]
pub struct Ledger {
    pub name: String,
    entries: HashMap<String, LedgerEntry>,
    storage_path: String,
    cache: LedgerCache,
    config: LedgerConfig,
    monitoring: MonitoringState,
}

/// Internal monitoring state
#[derive(Debug)]
struct MonitoringState {
    start_time: Instant,
    requests_processed: Arc<parking_lot::Mutex<u64>>,
    errors_encountered: Arc<parking_lot::Mutex<u64>>,
    response_times: Arc<parking_lot::Mutex<Vec<u64>>>, // in milliseconds
    alerts: Arc<parking_lot::Mutex<Vec<Alert>>>,
}

/// Ledger configuration for advanced features
#[derive(Debug, Clone)]
pub struct LedgerConfig {
    pub enable_backup: bool,
    pub backup_interval_hours: u64,
    pub max_entries: Option<usize>,
    pub enable_compression: bool,
    pub enable_integrity_checks: bool,
    pub health_check_interval_seconds: u64,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            enable_backup: true,
            backup_interval_hours: 24,
            max_entries: Some(100000),
            enable_compression: false,
            enable_integrity_checks: true,
            health_check_interval_seconds: 3600, // 1 hour
        }
    }
}

/// Health status information
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

impl Ledger {
    pub fn with_storage(name: String, storage_path: String) -> Result<Self> {
        Self::with_storage_and_config(name, storage_path, LedgerConfig::default())
    }

    pub fn with_storage_and_config(
        name: String,
        storage_path: String,
        config: LedgerConfig,
    ) -> Result<Self> {
        // Validate inputs
        if name.is_empty() {
            return Err(LedgerError::invalid_input("name", "cannot be empty"));
        }

        log::info!(
            "Initializing high-performance ledger '{}' with storage: {}",
            name,
            storage_path
        );

        let mut ledger = Ledger {
            name,
            entries: HashMap::new(),
            storage_path: storage_path.clone(),
            cache: LedgerCache::new(),
            config: config.clone(),
            monitoring: MonitoringState {
                start_time: Instant::now(),
                requests_processed: Arc::new(parking_lot::Mutex::new(0)),
                errors_encountered: Arc::new(parking_lot::Mutex::new(0)),
                response_times: Arc::new(parking_lot::Mutex::new(Vec::new())),
                alerts: Arc::new(parking_lot::Mutex::new(Vec::new())),
            },
        };

        // Create parent directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&storage_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LedgerError::storage_error(
                    "create_directory",
                    format!("Failed to create {}: {}", parent.display(), e),
                )
            })?;
        }

        // Load existing entries with error recovery
        ledger.load_with_recovery(&storage_path, &config)?;

        // Perform initial health check
        let health = ledger.health_check()?;
        if !health.is_healthy {
            log::warn!("Ledger health issues detected: {:?}", health.issues);
        }

        log::info!(
            "Ledger initialized successfully with {} entries",
            ledger.entries.len()
        );
        Ok(ledger)
    }

    /// Load entries with automatic recovery from corruption
    fn load_with_recovery(&mut self, storage_path: &str, config: &LedgerConfig) -> Result<()> {
        if !std::path::Path::new(storage_path).exists() {
            log::info!("Storage file doesn't exist, starting with empty ledger");
            return Ok(());
        }

        // Try to load normally first
        match self.load_entries(storage_path) {
            Ok(entries) => {
                self.entries = entries;
                log::info!("Loaded {} entries from storage", self.entries.len());

                // Check for size limits
                if let Some(max_entries) = config.max_entries {
                    if self.entries.len() > max_entries {
                        log::warn!(
                            "Entry count {} exceeds limit {}, archiving old entries",
                            self.entries.len(),
                            max_entries
                        );
                        self.archive_old_entries(max_entries)?;
                    }
                }

                Ok(())
            }
            Err(e) => {
                log::error!("Failed to load entries: {}", e);

                // Try to recover from backup
                let backup_path = format!("{}.backup", storage_path);
                if std::path::Path::new(&backup_path).exists() {
                    log::info!("Attempting recovery from backup: {}", backup_path);
                    match self.load_entries(&backup_path) {
                        Ok(entries) => {
                            self.entries = entries;
                            log::info!(
                                "Successfully recovered {} entries from backup",
                                self.entries.len()
                            );

                            // Save recovered data to main file
                            self.save().map_err(|e| {
                                LedgerError::storage_error(
                                    "recovery_save",
                                    format!("Failed to save recovered data: {}", e),
                                )
                            })?;

                            return Ok(());
                        }
                        Err(backup_error) => {
                            log::error!("Backup recovery also failed: {}", backup_error);
                        }
                    }
                }

                // If recovery fails, start fresh but preserve old file
                let corrupt_path = format!(
                    "{}.corrupt.{}",
                    storage_path,
                    chrono::Utc::now().timestamp()
                );
                if let Err(rename_err) = std::fs::rename(storage_path, &corrupt_path) {
                    log::error!("Failed to preserve corrupt file: {}", rename_err);
                } else {
                    log::info!("Corrupt file preserved as: {}", corrupt_path);
                }

                log::warn!("Starting with fresh ledger due to unrecoverable corruption");
                Ok(())
            }
        }
    }

    /// Load entries from file
    fn load_entries(&self, path: &str) -> Result<HashMap<String, LedgerEntry>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| LedgerError::storage_error("read", e.to_string()))?;

        if content.is_empty() {
            return Ok(HashMap::new());
        }

        serde_json::from_str(&content)
            .map_err(|e| LedgerError::storage_error("parse", format!("JSON parsing failed: {}", e)))
    }

    /// Archive old entries to keep ledger size manageable
    fn archive_old_entries(&mut self, max_entries: usize) -> Result<()> {
        if self.entries.len() <= max_entries {
            return Ok(());
        }

        // Sort entries by timestamp (oldest first)
        let mut entries_vec: Vec<_> = self.entries.iter().collect();
        entries_vec.sort_by_key(|(_, entry)| entry.timestamp);

        let num_to_archive = self.entries.len() - max_entries;
        let to_archive: Vec<_> = entries_vec.into_iter().take(num_to_archive).collect();

        // Create archive
        let archive_path = format!(
            "{}.archive.{}.json",
            self.storage_path,
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        );

        let archive_data: HashMap<_, _> = to_archive
            .iter()
            .map(|(k, v)| ((*k).clone(), (*v).clone()))
            .collect();
        let archive_json = serde_json::to_string_pretty(&archive_data)?;

        std::fs::write(&archive_path, archive_json)
            .map_err(|e| LedgerError::storage_error("archive", e.to_string()))?;

        log::info!("Archived {} entries to: {}", num_to_archive, archive_path);

        // Remove archived entries from current ledger
        let keys_to_remove: Vec<_> = to_archive.iter().map(|(k, _)| (*k).clone()).collect();
        for key in keys_to_remove {
            self.entries.remove(&key);
        }

        // Save updated ledger
        self.save()?;

        Ok(())
    }

    pub fn notarize_dataset(&mut self, dataset: Dataset, proof_type: String) -> Result<Proof> {
        let _monitor = self.start_request_monitoring();
        
        // Enhanced validation with specific error handling
        if dataset.name.is_empty() {
            self.record_alert(AlertLevel::Warning, "Attempted to notarize dataset with empty name".to_string(), "notarize_dataset".to_string());
            return Err(LedgerError::invalid_input("dataset_name", "cannot be empty"));
        }
        
        if dataset.size == 0 {
            self.record_alert(AlertLevel::Warning, format!("Attempted to notarize empty dataset: {}", dataset.name), "notarize_dataset".to_string());
            return Err(LedgerError::dataset_error("Dataset is empty", false));
        }

        // Check cache first for duplicate datasets
        if let Some(cached_hash) = self.cache.dataset_hashes.get(&dataset.name) {
            if *cached_hash == dataset.hash {
                if let Some(cached_proof) = self.cache.proof_cache.get(&dataset.hash) {
                    log::info!("Using cached proof for dataset: {}", dataset.name);
                    self.record_alert(AlertLevel::Info, format!("Cache hit for dataset: {}", dataset.name), "notarize_dataset".to_string());
                    return Ok(cached_proof.clone());
                }
            }
        }

        let proof = match Proof::generate(&dataset, proof_type.clone()) {
            Ok(p) => p,
            Err(e) => {
                self.record_alert(AlertLevel::Critical, format!("Proof generation failed for {}: {}", dataset.name, e), "notarize_dataset".to_string());
                return Err(e);
            }
        };

        let entry = LedgerEntry {
            id: uuid::Uuid::new_v4().to_string(),
            dataset_name: dataset.name.clone(),
            dataset_hash: dataset.compute_hash(),
            operation: format!("notarize({})", proof_type),
            proof: proof.clone(),
            timestamp: chrono::Utc::now(),
        };

        let entry_id = entry.id.clone();
        self.entries.insert(entry_id.clone(), entry);

        // Update caches for future performance
        self.cache
            .dataset_hashes
            .insert(dataset.name.clone(), dataset.hash.clone());
        self.cache
            .proof_cache
            .insert(dataset.hash.clone(), proof.clone());
        self.cache.metadata_cache.insert(entry_id, proof.metadata());

        // Clear stats cache since we added an entry
        self.cache
            .update_stats_cache(self.compute_stats_optimized());

        self.save()?;

        self.record_alert(AlertLevel::Info, format!("Successfully notarized dataset: {}", dataset.name), "notarize_dataset".to_string());
        log::info!("Dataset notarized successfully");

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
        // Use cached stats if available and recent
        if let Some(cached_stats) = self.cache.get_cached_stats() {
            log::debug!("Using cached statistics");
            return cached_stats;
        }

        let stats = self.compute_stats_optimized();
        self.cache.update_stats_cache(stats.clone());
        stats
    }

    /// Optimized statistics computation using parallel processing
    fn compute_stats_optimized(&self) -> LedgerStats {
        let start_time = Instant::now();

        if self.entries.is_empty() {
            return LedgerStats {
                total_datasets: 0,
                total_operations: 0,
                storage_path: Some(self.storage_path.clone()),
            };
        }

        // Use rayon for parallel processing of large datasets
        let datasets: std::collections::HashSet<String> = if self.entries.len() > 1000 {
            log::debug!("Using parallel processing for statistics computation");
            self.entries
                .par_iter()
                .map(|(_, e)| e.dataset_name.clone())
                .collect()
        } else {
            self.entries
                .values()
                .map(|e| e.dataset_name.clone())
                .collect()
        };

        let stats = LedgerStats {
            total_datasets: datasets.len(),
            total_operations: self.entries.len(),
            storage_path: Some(self.storage_path.clone()),
        };

        let duration = start_time.elapsed();
        log::debug!("Statistics computed in {:?}ms", duration.as_millis());

        stats
    }

    pub fn verify_integrity(&self) -> Result<bool> {
        self.verify_integrity_parallel()
    }

    /// High-performance parallel integrity verification
    pub fn verify_integrity_parallel(&self) -> Result<bool> {
        let start_time = Instant::now();

        if self.entries.is_empty() {
            return Ok(true);
        }

        // Use parallel processing for large ledgers
        let is_valid = if self.entries.len() > 100 {
            log::debug!(
                "Using parallel integrity verification for {} entries",
                self.entries.len()
            );
            self.entries
                .par_iter()
                .all(|(_, entry)| entry.proof.verify())
        } else {
            self.entries.values().all(|entry| entry.proof.verify())
        };

        let duration = start_time.elapsed();
        log::info!(
            "Integrity verification completed in {:?}ms: {}",
            duration.as_millis(),
            if is_valid { "VALID" } else { "INVALID" }
        );

        Ok(is_valid)
    }

    /// Batch notarize multiple datasets in parallel
    pub fn notarize_datasets_batch(
        &mut self,
        datasets: Vec<(Dataset, String)>,
    ) -> Result<Vec<Proof>> {
        if datasets.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        log::info!("Starting batch notarization of {} datasets", datasets.len());

        // Process datasets in parallel
        let results: Result<Vec<(LedgerEntry, Proof)>> = datasets
            .into_par_iter()
            .map(|(dataset, proof_type)| {
                // Generate proof
                let proof = Proof::generate(&dataset, proof_type.clone())?;

                // Create entry
                let entry = LedgerEntry {
                    id: uuid::Uuid::new_v4().to_string(),
                    dataset_name: dataset.name.clone(),
                    dataset_hash: dataset.compute_hash(),
                    operation: format!("notarize({})", proof_type),
                    proof: proof.clone(),
                    timestamp: chrono::Utc::now(),
                };

                // Update caches
                self.cache
                    .dataset_hashes
                    .insert(dataset.name.clone(), dataset.hash.clone());
                self.cache
                    .proof_cache
                    .insert(dataset.hash.clone(), proof.clone());
                self.cache
                    .metadata_cache
                    .insert(entry.id.clone(), proof.metadata());

                Ok((entry, proof))
            })
            .collect();

        let entries_and_proofs = results?;
        let proofs: Vec<_> = entries_and_proofs
            .iter()
            .map(|(_, proof)| proof.clone())
            .collect();

        // Add all entries to ledger
        for (entry, _) in entries_and_proofs {
            self.entries.insert(entry.id.clone(), entry);
        }

        // Clear stats cache and save
        self.cache
            .update_stats_cache(self.compute_stats_optimized());
        self.save()?;

        let duration = start_time.elapsed();
        log::info!(
            "Batch notarization completed in {:?}ms ({} datasets)",
            duration.as_millis(),
            proofs.len()
        );

        Ok(proofs)
    }

    pub fn verify_proof(&self, proof: &Proof) -> bool {
        proof.verify()
    }

    /// Comprehensive health check
    pub fn health_check(&self) -> Result<HealthStatus> {
        let start_time = std::time::Instant::now();
        let mut issues = Vec::new();

        log::debug!("Running health check for ledger: {}", self.name);

        // Check storage accessibility
        let storage_accessible = self.check_storage_health(&mut issues);

        // Check data integrity
        let integrity_verified = self.check_data_integrity(&mut issues);

        // Get storage size
        let storage_size_bytes = std::fs::metadata(&self.storage_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let is_healthy = issues.is_empty();
        let check_duration = start_time.elapsed();

        let status = HealthStatus {
            is_healthy,
            last_check: chrono::Utc::now(),
            storage_accessible,
            integrity_verified,
            entry_count: self.entries.len(),
            storage_size_bytes,
            issues,
        };

        log::info!(
            "Health check completed in {:?}ms: {} (Issues: {})",
            check_duration.as_millis(),
            if is_healthy { "HEALTHY" } else { "UNHEALTHY" },
            status.issues.len()
        );

        Ok(status)
    }

    /// Check storage system health
    fn check_storage_health(&self, issues: &mut Vec<String>) -> bool {
        // Test write permissions by creating a temporary file
        let test_path = format!("{}.healthcheck", self.storage_path);
        match std::fs::write(&test_path, "health_check") {
            Ok(_) => {
                // Clean up test file
                if let Err(e) = std::fs::remove_file(&test_path) {
                    log::warn!("Failed to clean up health check file: {}", e);
                }
                true
            }
            Err(e) => {
                issues.push(format!("Storage write test failed: {}", e));
                false
            }
        }
    }

    /// Check data integrity
    fn check_data_integrity(&self, issues: &mut Vec<String>) -> bool {
        let mut valid_count = 0;
        let mut invalid_count = 0;

        for (id, entry) in &self.entries {
            // Check entry structure
            if entry.id != *id {
                issues.push(format!(
                    "Entry ID mismatch: stored as '{}' but contains '{}'",
                    id, entry.id
                ));
                invalid_count += 1;
                continue;
            }

            // Validate proof
            if !entry.proof.verify() {
                issues.push(format!("Invalid proof in entry: {}", id));
                invalid_count += 1;
                continue;
            }

            // Check timestamp consistency
            if entry.proof.timestamp != entry.timestamp {
                log::warn!(
                    "Timestamp mismatch in entry {}: entry={}, proof={}",
                    id,
                    entry.timestamp,
                    entry.proof.timestamp
                );
            }

            valid_count += 1;
        }

        let integrity_rate = if self.entries.is_empty() {
            100.0
        } else {
            (valid_count as f64 / self.entries.len() as f64) * 100.0
        };

        log::debug!(
            "Data integrity: {:.1}% ({}/{} valid entries)",
            integrity_rate,
            valid_count,
            self.entries.len()
        );

        if integrity_rate < 95.0 {
            issues.push(format!(
                "Low data integrity: {:.1}% valid entries",
                integrity_rate
            ));
        }

        invalid_count == 0
    }

    /// Create backup of current ledger
    pub fn create_backup(&self) -> Result<String> {
        let backup_path = format!("{}.backup", self.storage_path);

        log::info!("Creating backup: {}", backup_path);

        // Create backup with timestamp in filename for uniqueness
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let timestamped_backup = format!("{}.backup.{}", self.storage_path, timestamp);

        // Copy current ledger to backup
        std::fs::copy(&self.storage_path, &timestamped_backup)
            .map_err(|e| LedgerError::storage_error("backup", e.to_string()))?;

        // Also maintain a "latest" backup
        std::fs::copy(&self.storage_path, &backup_path)
            .map_err(|e| LedgerError::storage_error("backup", e.to_string()))?;

        log::info!("Backup created successfully: {}", timestamped_backup);
        Ok(timestamped_backup)
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> LedgerMetrics {
        let storage_size = std::fs::metadata(&self.storage_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let datasets: std::collections::HashSet<String> = self
            .entries
            .values()
            .map(|e| e.dataset_name.clone())
            .collect();

        // Calculate average entry size
        let avg_entry_size = if !self.entries.is_empty() {
            storage_size / self.entries.len() as u64
        } else {
            0
        };

        // Find latest activity
        let latest_timestamp = self
            .entries
            .values()
            .map(|e| e.timestamp)
            .max()
            .unwrap_or_else(chrono::Utc::now);

        LedgerMetrics {
            total_entries: self.entries.len(),
            unique_datasets: datasets.len(),
            storage_size_bytes: storage_size,
            average_entry_size_bytes: avg_entry_size,
            latest_activity: latest_timestamp,
            health_score: self.calculate_health_score(),
        }
    }

    /// Calculate overall health score (0-100)
    fn calculate_health_score(&self) -> u8 {
        let mut score = 100u8;

        // Deduct for storage issues
        if !std::path::Path::new(&self.storage_path).exists() {
            score = score.saturating_sub(50);
        }

        // Check data integrity
        let invalid_proofs = self.entries.values().filter(|e| !e.proof.verify()).count();

        if !self.entries.is_empty() {
            let integrity_rate =
                ((self.entries.len() - invalid_proofs) as f64 / self.entries.len() as f64) * 100.0;
            score = score.saturating_sub((100.0 - integrity_rate) as u8);
        }

        score
    }

    /// Record a monitoring alert
    pub fn record_alert(&self, level: AlertLevel, message: String, source: String) {
        let alert = Alert {
            level: level.clone(),
            timestamp: chrono::Utc::now(),
            message: message.clone(),
            source: source.clone(),
            metadata: HashMap::new(),
        };
        
        self.monitoring.alerts.lock().push(alert);
        
        // Log alert based on level
        match level {
            AlertLevel::Info => log::info!("[{}] {}", source, message),
            AlertLevel::Warning => log::warn!("[{}] {}", source, message),
            AlertLevel::Critical => log::error!("[{}] CRITICAL: {}", source, message),
            AlertLevel::Emergency => log::error!("[{}] EMERGENCY: {}", source, message),
        }
    }

    /// Get current monitoring metrics
    pub fn get_monitoring_metrics(&self) -> MonitoringMetrics {
        let uptime = self.monitoring.start_time.elapsed();
        let requests = *self.monitoring.requests_processed.lock();
        let errors = *self.monitoring.errors_encountered.lock();
        let response_times = self.monitoring.response_times.lock();
        
        let average_response_time = if response_times.is_empty() {
            0.0
        } else {
            response_times.iter().sum::<u64>() as f64 / response_times.len() as f64
        };
        
        // Calculate cache hit rate (approximate)
        let cache_entries = self.cache.dataset_hashes.len() + self.cache.proof_cache.len();
        let cache_hit_rate = if requests > 0 {
            (cache_entries as f64 / requests as f64).min(1.0)
        } else {
            0.0
        };
        
        // Storage utilization (basic calculation)
        let storage_size = std::fs::metadata(&self.storage_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let max_storage = self.config.max_entries.unwrap_or(100000) as u64 * 1024; // rough estimate
        let storage_utilization = if max_storage > 0 {
            (storage_size as f64 / max_storage as f64).min(1.0)
        } else {
            0.0
        };
        
        MonitoringMetrics {
            uptime_seconds: uptime.as_secs(),
            requests_processed: requests,
            errors_encountered: errors,
            cache_hit_rate,
            average_response_time_ms: average_response_time,
            memory_usage_mb: 0.0, // Would need external memory profiling
            storage_utilization,
            proof_generation_rate: if uptime.as_secs() > 0 { requests as f64 / uptime.as_secs() as f64 } else { 0.0 },
            verification_success_rate: if requests > 0 { (requests - errors) as f64 / requests as f64 } else { 1.0 },
            last_backup_timestamp: None, // Would be tracked separately
        }
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, limit: usize) -> Vec<Alert> {
        let alerts = self.monitoring.alerts.lock();
        alerts.iter().rev().take(limit).cloned().collect()
    }

    /// Clear old alerts (keep only recent ones)  
    pub fn cleanup_alerts(&self, keep_hours: u64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(keep_hours as i64);
        let mut alerts = self.monitoring.alerts.lock();
        alerts.retain(|alert| alert.timestamp > cutoff);
    }
    
    /// Record request start and return a monitoring guard
    fn start_request_monitoring(&self) -> RequestMonitor {
        RequestMonitor {
            start_time: Instant::now(),
            requests_counter: Arc::clone(&self.monitoring.requests_processed),
            response_times: Arc::clone(&self.monitoring.response_times),
            errors_counter: Arc::clone(&self.monitoring.errors_encountered),
        }
    }

    fn save(&self) -> Result<()> {
        // Create backup before saving if file exists and is substantial
        if std::path::Path::new(&self.storage_path).exists() {
            let metadata = std::fs::metadata(&self.storage_path)?;
            if metadata.len() > 1024 {
                // Only backup if > 1KB
                if let Err(e) = self.create_backup() {
                    log::warn!("Failed to create backup before save: {}", e);
                }
            }
        }

        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| LedgerError::storage_error("serialize", e.to_string()))?;

        // Atomic write using temporary file
        let temp_path = format!("{}.tmp", self.storage_path);
        std::fs::write(&temp_path, &json)
            .map_err(|e| LedgerError::storage_error("write_temp", e.to_string()))?;

        std::fs::rename(&temp_path, &self.storage_path)
            .map_err(|e| LedgerError::storage_error("atomic_move", e.to_string()))?;

        log::debug!("Ledger saved successfully: {} bytes", json.len());
        Ok(())
    }
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerMetrics {
    pub total_entries: usize,
    pub unique_datasets: usize,
    pub storage_size_bytes: u64,
    pub average_entry_size_bytes: u64,
    pub latest_activity: DateTime<Utc>,
    pub health_score: u8,
}

/// Advanced monitoring metrics for production environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringMetrics {
    pub uptime_seconds: u64,
    pub requests_processed: u64,
    pub errors_encountered: u64,
    pub cache_hit_rate: f64,
    pub average_response_time_ms: f64,
    pub memory_usage_mb: f64,
    pub storage_utilization: f64,
    pub proof_generation_rate: f64, // proofs per second
    pub verification_success_rate: f64,
    pub last_backup_timestamp: Option<DateTime<Utc>>,
}

/// Alert levels for monitoring
#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Monitoring alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub level: AlertLevel,
    pub timestamp: DateTime<Utc>,
    pub message: String,
    pub source: String,
    pub metadata: HashMap<String, String>,
}

/// Request monitoring helper
struct RequestMonitor {
    start_time: Instant,
    requests_counter: Arc<parking_lot::Mutex<u64>>,
    response_times: Arc<parking_lot::Mutex<Vec<u64>>>,
    errors_counter: Arc<parking_lot::Mutex<u64>>,
}

impl Drop for RequestMonitor {
    fn drop(&mut self) {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        
        // Record response time
        self.response_times.lock().push(duration_ms);
        
        // Increment request counter
        *self.requests_counter.lock() += 1;
        
        // Keep only recent response times (last 1000)
        let mut times = self.response_times.lock();
        let len = times.len();
        if len > 1000 {
            times.drain(..len - 1000);
        }
    }
}

impl RequestMonitor {
    fn record_error(&self) {
        *self.errors_counter.lock() += 1;
    }
}

// Types are already public, no need to re-export

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
        let mut ledger =
            Ledger::with_storage("test".to_string(), "/tmp/test_ledger.json".to_string()).unwrap();
        let proof = ledger
            .notarize_dataset(dataset, "integrity".to_string())
            .unwrap();
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
