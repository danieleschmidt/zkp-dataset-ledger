//! Backup, recovery, and fault tolerance mechanisms for ZKP Dataset Ledger

use crate::{LedgerError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enable_auto_backup: bool,
    /// Backup interval in hours
    pub backup_interval_hours: u32,
    /// Maximum number of backups to retain
    pub max_backup_count: u32,
    /// Backup storage location
    pub backup_directory: PathBuf,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Enable incremental backups
    pub enable_incremental: bool,
    /// Enable encryption for backups
    pub enable_encryption: bool,
    /// Backup verification enabled
    pub verify_backups: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enable_auto_backup: true,
            backup_interval_hours: 24, // Daily backups
            max_backup_count: 7,       // Keep 7 days
            backup_directory: PathBuf::from("./backups"),
            compression_level: 6,
            enable_incremental: true,
            enable_encryption: false,
            verify_backups: true,
        }
    }
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    pub backup_id: String,
    pub backup_type: BackupType,
    pub created_at: DateTime<Utc>,
    pub size_bytes: u64,
    pub checksum: String,
    pub compressed: bool,
    pub encrypted: bool,
    pub ledger_entries_count: u64,
    pub version: String,
    pub base_backup_id: Option<String>, // For incremental backups
}

/// Types of backups
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BackupType {
    Full,
    Incremental,
    Snapshot,
    Emergency,
}

/// Recovery point with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPoint {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub ledger_state_hash: String,
    pub entries_count: u64,
    pub backup_metadata: Option<BackupMetadata>,
    pub verified: bool,
}

/// Backup and recovery system
pub struct BackupRecoverySystem {
    config: BackupConfig,
    recovery_points: Vec<RecoveryPoint>,
    last_backup: Option<DateTime<Utc>>,
}

impl BackupRecoverySystem {
    pub fn new(config: BackupConfig) -> Result<Self> {
        // Create backup directory if it doesn't exist
        std::fs::create_dir_all(&config.backup_directory)?;

        Ok(Self {
            config,
            recovery_points: Vec::new(),
            last_backup: None,
        })
    }

    /// Create a full backup of the ledger
    pub fn create_full_backup(
        &mut self,
        ledger_data: &[u8],
        entries_count: u64,
    ) -> Result<BackupMetadata> {
        let backup_id = uuid::Uuid::new_v4().to_string();
        let backup_path = self
            .config
            .backup_directory
            .join(format!("full_backup_{}.zkpbackup", backup_id));

        tracing::info!(backup_id = %backup_id, "Creating full backup");

        // Calculate checksum
        let checksum = crate::crypto::hash::hash_bytes(
            ledger_data,
            crate::crypto::hash::HashAlgorithm::default(),
        )?;

        // Create backup file
        let mut backup_data = ledger_data.to_vec();
        let compressed = if self.config.compression_level > 0 {
            backup_data = self.compress_data(&backup_data)?;
            true
        } else {
            false
        };

        let encrypted = if self.config.enable_encryption {
            backup_data = self.encrypt_data(&backup_data)?;
            true
        } else {
            false
        };

        std::fs::write(&backup_path, &backup_data)?;

        let metadata = BackupMetadata {
            backup_id: backup_id.clone(),
            backup_type: BackupType::Full,
            created_at: Utc::now(),
            size_bytes: backup_data.len() as u64,
            checksum,
            compressed,
            encrypted,
            ledger_entries_count: entries_count,
            version: env!("CARGO_PKG_VERSION").to_string(),
            base_backup_id: None,
        };

        // Save metadata
        self.save_backup_metadata(&metadata)?;

        // Verify backup if enabled
        if self.config.verify_backups {
            self.verify_backup(&metadata)?;
        }

        self.last_backup = Some(metadata.created_at);

        // Clean up old backups
        self.cleanup_old_backups()?;

        tracing::info!(backup_id = %backup_id, size = metadata.size_bytes, "Full backup created");

        Ok(metadata)
    }

    /// Create an incremental backup
    pub fn create_incremental_backup(
        &mut self,
        changed_data: &[u8],
        base_backup_id: &str,
        entries_count: u64,
    ) -> Result<BackupMetadata> {
        if !self.config.enable_incremental {
            return Err(LedgerError::ConfigError(
                "Incremental backups are disabled".to_string(),
            ));
        }

        let backup_id = uuid::Uuid::new_v4().to_string();
        let backup_path = self
            .config
            .backup_directory
            .join(format!("incr_backup_{}.zkpbackup", backup_id));

        tracing::info!(backup_id = %backup_id, base = %base_backup_id, "Creating incremental backup");

        let checksum = crate::crypto::hash::hash_bytes(
            changed_data,
            crate::crypto::hash::HashAlgorithm::default(),
        )?;

        let mut backup_data = changed_data.to_vec();
        let compressed = if self.config.compression_level > 0 {
            backup_data = self.compress_data(&backup_data)?;
            true
        } else {
            false
        };

        let encrypted = if self.config.enable_encryption {
            backup_data = self.encrypt_data(&backup_data)?;
            true
        } else {
            false
        };

        std::fs::write(&backup_path, &backup_data)?;

        let metadata = BackupMetadata {
            backup_id: backup_id.clone(),
            backup_type: BackupType::Incremental,
            created_at: Utc::now(),
            size_bytes: backup_data.len() as u64,
            checksum,
            compressed,
            encrypted,
            ledger_entries_count: entries_count,
            version: env!("CARGO_PKG_VERSION").to_string(),
            base_backup_id: Some(base_backup_id.to_string()),
        };

        self.save_backup_metadata(&metadata)?;

        if self.config.verify_backups {
            self.verify_backup(&metadata)?;
        }

        self.last_backup = Some(metadata.created_at);

        tracing::info!(backup_id = %backup_id, size = metadata.size_bytes, "Incremental backup created");

        Ok(metadata)
    }

    /// Restore from a backup
    pub fn restore_from_backup(&self, backup_id: &str) -> Result<Vec<u8>> {
        tracing::info!(backup_id = %backup_id, "Starting restore from backup");

        let metadata = self.get_backup_metadata(backup_id)?;
        let backup_path = self.get_backup_path(&metadata);

        if !backup_path.exists() {
            return Err(LedgerError::InvalidInput(format!(
                "Backup file not found: {}",
                backup_path.display()
            )));
        }

        let mut backup_data = std::fs::read(&backup_path)?;

        // Decrypt if necessary
        if metadata.encrypted {
            backup_data = self.decrypt_data(&backup_data)?;
        }

        // Decompress if necessary
        if metadata.compressed {
            backup_data = self.decompress_data(&backup_data)?;
        }

        // Verify checksum
        let checksum = crate::crypto::hash::hash_bytes(
            &backup_data,
            crate::crypto::hash::HashAlgorithm::default(),
        )?;
        if checksum != metadata.checksum {
            return Err(LedgerError::InvalidInput(
                "Backup checksum verification failed".to_string(),
            ));
        }

        // For incremental backups, we need to reconstruct from base backup
        if metadata.backup_type == BackupType::Incremental {
            if let Some(base_id) = &metadata.base_backup_id {
                let base_data = self.restore_from_backup(base_id)?;
                backup_data = self.apply_incremental_backup(&base_data, &backup_data)?;
            }
        }

        tracing::info!(backup_id = %backup_id, "Restore completed successfully");

        Ok(backup_data)
    }

    /// Create a recovery point
    pub fn create_recovery_point(
        &mut self,
        ledger_state_hash: &str,
        entries_count: u64,
    ) -> Result<RecoveryPoint> {
        let recovery_point = RecoveryPoint {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            ledger_state_hash: ledger_state_hash.to_string(),
            entries_count,
            backup_metadata: None,
            verified: true,
        };

        self.recovery_points.push(recovery_point.clone());

        // Limit the number of recovery points
        if self.recovery_points.len() > 100 {
            self.recovery_points.remove(0);
        }

        tracing::info!(recovery_point_id = %recovery_point.id, "Recovery point created");

        Ok(recovery_point)
    }

    /// Get all available recovery points
    pub fn get_recovery_points(&self) -> &[RecoveryPoint] {
        &self.recovery_points
    }

    /// Verify the integrity of a backup
    pub fn verify_backup(&self, metadata: &BackupMetadata) -> Result<()> {
        let backup_path = self.get_backup_path(metadata);

        if !backup_path.exists() {
            return Err(LedgerError::InvalidInput(
                "Backup file does not exist".to_string(),
            ));
        }

        let file_size = std::fs::metadata(&backup_path)?.len();
        if file_size != metadata.size_bytes {
            return Err(LedgerError::InvalidInput(
                "Backup file size mismatch".to_string(),
            ));
        }

        // Additional verification could include:
        // - Attempting to restore a small portion
        // - Checking file format integrity
        // - Verifying encryption/compression headers

        tracing::debug!(backup_id = %metadata.backup_id, "Backup verification passed");
        Ok(())
    }

    /// Check if automatic backup is due
    pub fn should_create_backup(&self) -> bool {
        if !self.config.enable_auto_backup {
            return false;
        }

        if let Some(last_backup) = self.last_backup {
            let hours_since_backup =
                Utc::now().signed_duration_since(last_backup).num_hours() as u32;
            hours_since_backup >= self.config.backup_interval_hours
        } else {
            true // No previous backup
        }
    }

    /// Get backup statistics
    pub fn get_backup_stats(&self) -> Result<BackupStats> {
        let backup_dir = &self.config.backup_directory;
        let mut total_size = 0u64;
        let mut backup_count = 0u32;

        if backup_dir.exists() {
            for entry in std::fs::read_dir(backup_dir)? {
                let entry = entry?;
                if let Some(ext) = entry.path().extension() {
                    if ext == "zkpbackup" {
                        backup_count += 1;
                        total_size += entry.metadata()?.len();
                    }
                }
            }
        }

        Ok(BackupStats {
            total_backups: backup_count,
            total_size_bytes: total_size,
            last_backup_time: self.last_backup,
            backup_directory: backup_dir.clone(),
        })
    }

    // Private helper methods

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified compression - in production, use proper compression library
        Ok(data.to_vec()) // Placeholder
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified decompression - in production, use proper compression library
        Ok(data.to_vec()) // Placeholder
    }

    fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified encryption - in production, use proper encryption
        Ok(data.to_vec()) // Placeholder
    }

    fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified decryption - in production, use proper encryption
        Ok(data.to_vec()) // Placeholder
    }

    fn save_backup_metadata(&self, metadata: &BackupMetadata) -> Result<()> {
        let metadata_path = self
            .config
            .backup_directory
            .join(format!("{}.metadata", metadata.backup_id));
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        Ok(())
    }

    fn get_backup_metadata(&self, backup_id: &str) -> Result<BackupMetadata> {
        let metadata_path = self
            .config
            .backup_directory
            .join(format!("{}.metadata", backup_id));
        let metadata_json = std::fs::read_to_string(metadata_path)?;
        let metadata: BackupMetadata = serde_json::from_str(&metadata_json)?;
        Ok(metadata)
    }

    fn get_backup_path(&self, metadata: &BackupMetadata) -> PathBuf {
        let filename = match metadata.backup_type {
            BackupType::Full => format!("full_backup_{}.zkpbackup", metadata.backup_id),
            BackupType::Incremental => format!("incr_backup_{}.zkpbackup", metadata.backup_id),
            BackupType::Snapshot => format!("snapshot_{}.zkpbackup", metadata.backup_id),
            BackupType::Emergency => format!("emergency_{}.zkpbackup", metadata.backup_id),
        };
        self.config.backup_directory.join(filename)
    }

    fn apply_incremental_backup(
        &self,
        base_data: &[u8],
        incremental_data: &[u8],
    ) -> Result<Vec<u8>> {
        // Simplified incremental application - in production, implement proper diff/patch
        let mut result = base_data.to_vec();
        result.extend_from_slice(incremental_data);
        Ok(result)
    }

    fn cleanup_old_backups(&self) -> Result<()> {
        // Get all backup files and sort by creation time
        let mut backups = Vec::new();

        for entry in std::fs::read_dir(&self.config.backup_directory)? {
            let entry = entry?;
            if let Some(ext) = entry.path().extension() {
                if ext == "metadata" {
                    let metadata_content = std::fs::read_to_string(entry.path())?;
                    if let Ok(metadata) = serde_json::from_str::<BackupMetadata>(&metadata_content)
                    {
                        backups.push(metadata);
                    }
                }
            }
        }

        // Sort by creation time (newest first)
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        // Remove old backups beyond the retention limit
        if backups.len() > self.config.max_backup_count as usize {
            for backup in backups.iter().skip(self.config.max_backup_count as usize) {
                let backup_path = self.get_backup_path(backup);
                let metadata_path = self
                    .config
                    .backup_directory
                    .join(format!("{}.metadata", backup.backup_id));

                if backup_path.exists() {
                    std::fs::remove_file(&backup_path)?;
                }
                if metadata_path.exists() {
                    std::fs::remove_file(&metadata_path)?;
                }

                tracing::info!(backup_id = %backup.backup_id, "Old backup cleaned up");
            }
        }

        Ok(())
    }
}

/// Backup statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStats {
    pub total_backups: u32,
    pub total_size_bytes: u64,
    pub last_backup_time: Option<DateTime<Utc>>,
    pub backup_directory: PathBuf,
}

/// Fault tolerance manager
pub struct FaultToleranceManager {
    retry_config: RetryConfig,
    circuit_breaker: CircuitBreaker,
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Circuit breaker for fault tolerance
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    reset_timeout_ms: u64,
    current_failures: u32,
    last_failure_time: Option<DateTime<Utc>>,
    state: CircuitBreakerState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing if service recovered
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, reset_timeout_ms: u64) -> Self {
        Self {
            failure_threshold,
            reset_timeout_ms,
            current_failures: 0,
            last_failure_time: None,
            state: CircuitBreakerState::Closed,
        }
    }

    pub fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    let time_since_failure = Utc::now()
                        .signed_duration_since(last_failure)
                        .num_milliseconds() as u64;

                    if time_since_failure >= self.reset_timeout_ms {
                        self.state = CircuitBreakerState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    self.state = CircuitBreakerState::Closed;
                    true
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }

    pub fn record_success(&mut self) {
        self.current_failures = 0;
        self.state = CircuitBreakerState::Closed;
    }

    pub fn record_failure(&mut self) {
        self.current_failures += 1;
        self.last_failure_time = Some(Utc::now());

        if self.current_failures >= self.failure_threshold {
            self.state = CircuitBreakerState::Open;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_backup_config_default() {
        let config = BackupConfig::default();
        assert!(config.enable_auto_backup);
        assert_eq!(config.backup_interval_hours, 24);
        assert_eq!(config.max_backup_count, 7);
    }

    #[test]
    fn test_circuit_breaker() {
        let mut breaker = CircuitBreaker::new(3, 1000);

        // Should allow execution initially
        assert!(breaker.can_execute());
        assert_eq!(breaker.state, CircuitBreakerState::Closed);

        // Record failures
        breaker.record_failure();
        breaker.record_failure();
        assert!(breaker.can_execute());
        assert_eq!(breaker.state, CircuitBreakerState::Closed);

        // Third failure should open circuit
        breaker.record_failure();
        assert_eq!(breaker.state, CircuitBreakerState::Open);
        assert!(!breaker.can_execute());

        // Success should close circuit
        breaker.record_success();
        assert_eq!(breaker.state, CircuitBreakerState::Closed);
        assert!(breaker.can_execute());
    }

    #[test]
    fn test_backup_system() {
        let temp_dir = tempdir().unwrap();
        let config = BackupConfig {
            backup_directory: temp_dir.path().to_path_buf(),
            ..BackupConfig::default()
        };

        let mut backup_system = BackupRecoverySystem::new(config).unwrap();

        let test_data = b"test ledger data";
        let metadata = backup_system.create_full_backup(test_data, 10).unwrap();

        assert_eq!(metadata.backup_type, BackupType::Full);
        assert_eq!(metadata.ledger_entries_count, 10);

        // Test restore
        let restored_data = backup_system
            .restore_from_backup(&metadata.backup_id)
            .unwrap();
        assert_eq!(restored_data, test_data);
    }

    #[test]
    fn test_recovery_points() {
        let temp_dir = tempdir().unwrap();
        let config = BackupConfig {
            backup_directory: temp_dir.path().to_path_buf(),
            ..BackupConfig::default()
        };

        let mut backup_system = BackupRecoverySystem::new(config).unwrap();

        let recovery_point = backup_system.create_recovery_point("test_hash", 5).unwrap();
        assert_eq!(recovery_point.entries_count, 5);
        assert_eq!(recovery_point.ledger_state_hash, "test_hash");

        let points = backup_system.get_recovery_points();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].id, recovery_point.id);
    }
}
