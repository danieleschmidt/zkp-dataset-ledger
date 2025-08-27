//! Configuration Management for ZKP Dataset Ledger
//!
//! Provides centralized configuration handling with environment variable support,
//! configuration file parsing, and runtime validation.

use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Global configuration for the ZKP Dataset Ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Ledger settings
    pub ledger: LedgerConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
    /// Storage settings
    pub storage: StorageConfig,
    /// Logging settings
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerConfig {
    /// Default hash algorithm
    pub hash_algorithm: String,
    /// Default proof type
    pub default_proof_type: String,
    /// Enable automatic backups
    pub auto_backup: bool,
    /// Backup retention days
    pub backup_retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable encryption for stored data
    pub encrypt_storage: bool,
    /// Key rotation interval in days
    pub key_rotation_days: u32,
    /// Audit log retention in days
    pub audit_retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size in MB
    pub cache_size_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage backend type
    pub backend: String,
    /// Default storage path
    pub default_path: String,
    /// Compression enabled
    pub compression: bool,
    /// Maximum file size in MB
    pub max_file_size_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log to file
    pub log_to_file: bool,
    /// Log file path
    pub log_file: Option<String>,
    /// Enable structured logging
    pub structured: bool,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            ledger: LedgerConfig {
                hash_algorithm: "sha3-256".to_string(),
                default_proof_type: "integrity".to_string(),
                auto_backup: true,
                backup_retention_days: 30,
            },
            security: SecurityConfig {
                encrypt_storage: false,
                key_rotation_days: 90,
                audit_retention_days: 365,
            },
            performance: PerformanceConfig {
                parallel_processing: true,
                worker_threads: 0,
                enable_cache: true,
                cache_size_mb: 100,
            },
            storage: StorageConfig {
                backend: "json".to_string(),
                default_path: "./zkp_ledger".to_string(),
                compression: false,
                max_file_size_mb: 1024,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                log_to_file: false,
                log_file: None,
                structured: false,
            },
        }
    }
}

/// Configuration manager with environment variable and file support
#[derive(Debug)]
pub struct ConfigManager {
    config: GlobalConfig,
    config_sources: Vec<String>,
}

impl ConfigManager {
    /// Create new configuration manager with default settings
    pub fn new() -> Self {
        Self {
            config: GlobalConfig::default(),
            config_sources: vec!["default".to_string()],
        }
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path).map_err(|e| {
            LedgerError::not_found("config", format!("Failed to read config file: {}", e))
        })?;

        let config: GlobalConfig = toml::from_str(&content)
            .map_err(|e| LedgerError::validation_error(format!("Invalid config format: {}", e)))?;

        Ok(Self {
            config,
            config_sources: vec![path.as_ref().to_string_lossy().to_string()],
        })
    }

    /// Load configuration with environment variable override
    pub fn load_with_env() -> Result<Self> {
        let mut config = GlobalConfig::default();
        let mut sources = vec!["default".to_string()];

        // Override with environment variables
        if let Ok(hash_algo) = std::env::var("ZKP_HASH_ALGORITHM") {
            config.ledger.hash_algorithm = hash_algo;
            sources.push("env:ZKP_HASH_ALGORITHM".to_string());
        }

        if let Ok(proof_type) = std::env::var("ZKP_DEFAULT_PROOF_TYPE") {
            config.ledger.default_proof_type = proof_type;
            sources.push("env:ZKP_DEFAULT_PROOF_TYPE".to_string());
        }

        if let Ok(log_level) = std::env::var("ZKP_LOG_LEVEL") {
            config.logging.level = log_level;
            sources.push("env:ZKP_LOG_LEVEL".to_string());
        }

        if let Ok(storage_path) = std::env::var("ZKP_STORAGE_PATH") {
            config.storage.default_path = storage_path;
            sources.push("env:ZKP_STORAGE_PATH".to_string());
        }

        if let Ok(parallel) = std::env::var("ZKP_PARALLEL_PROCESSING") {
            config.performance.parallel_processing = parallel.parse().unwrap_or(true);
            sources.push("env:ZKP_PARALLEL_PROCESSING".to_string());
        }

        // Try to load additional config file
        let home_config = std::env::var("HOME")
            .map(|h| format!("{}/.zkp-ledger.toml", h))
            .unwrap_or_default();
        let config_paths = [
            "./zkp-ledger.toml",
            "./config/zkp-ledger.toml",
            home_config.as_str(),
        ];

        for config_path in &config_paths {
            if Path::new(config_path).exists() {
                if let Ok(file_config) = Self::load_from_file(config_path) {
                    config = file_config.config;
                    sources.push(config_path.to_string());
                    break;
                }
            }
        }

        // Apply environment variables again after loading file (env vars take precedence)
        if let Ok(hash_algo) = std::env::var("ZKP_HASH_ALGORITHM") {
            config.ledger.hash_algorithm = hash_algo;
        }

        if let Ok(proof_type) = std::env::var("ZKP_DEFAULT_PROOF_TYPE") {
            config.ledger.default_proof_type = proof_type;
        }

        if let Ok(log_level) = std::env::var("ZKP_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        if let Ok(storage_path) = std::env::var("ZKP_STORAGE_PATH") {
            config.storage.default_path = storage_path;
        }

        if let Ok(parallel) = std::env::var("ZKP_PARALLEL_PROCESSING") {
            config.performance.parallel_processing = parallel.parse().unwrap_or(true);
        }

        let manager = Self {
            config,
            config_sources: sources,
        };

        manager.validate()?;
        Ok(manager)
    }

    /// Get the current configuration
    pub fn config(&self) -> &GlobalConfig {
        &self.config
    }

    /// Get configuration sources
    pub fn sources(&self) -> &[String] {
        &self.config_sources
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate hash algorithm
        let valid_hash_algos = ["sha2-256", "sha3-256", "blake3"];
        if !valid_hash_algos.contains(&self.config.ledger.hash_algorithm.as_str()) {
            return Err(LedgerError::validation_error(format!(
                "Invalid hash algorithm: {}. Must be one of: {:?}",
                self.config.ledger.hash_algorithm, valid_hash_algos
            )));
        }

        // Validate proof type
        let valid_proof_types = ["integrity", "statistical", "zk-integrity"];
        if !valid_proof_types.contains(&self.config.ledger.default_proof_type.as_str()) {
            return Err(LedgerError::validation_error(format!(
                "Invalid proof type: {}. Must be one of: {:?}",
                self.config.ledger.default_proof_type, valid_proof_types
            )));
        }

        // Validate log level
        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.config.logging.level.as_str()) {
            return Err(LedgerError::validation_error(format!(
                "Invalid log level: {}. Must be one of: {:?}",
                self.config.logging.level, valid_log_levels
            )));
        }

        // Validate storage backend
        let valid_backends = ["json", "rocksdb", "postgres"];
        if !valid_backends.contains(&self.config.storage.backend.as_str()) {
            return Err(LedgerError::validation_error(format!(
                "Invalid storage backend: {}. Must be one of: {:?}",
                self.config.storage.backend, valid_backends
            )));
        }

        Ok(())
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(&self.config).map_err(|e| {
            LedgerError::validation_error(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(&path, content).map_err(|e| {
            LedgerError::data_integrity_error(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    /// Get configuration value by key path (e.g., "ledger.hash_algorithm")
    pub fn get_value(&self, key: &str) -> Option<String> {
        let parts: Vec<&str> = key.split('.').collect();
        match parts.as_slice() {
            ["ledger", "hash_algorithm"] => Some(self.config.ledger.hash_algorithm.clone()),
            ["ledger", "default_proof_type"] => Some(self.config.ledger.default_proof_type.clone()),
            ["security", "encrypt_storage"] => {
                Some(self.config.security.encrypt_storage.to_string())
            }
            ["performance", "parallel_processing"] => {
                Some(self.config.performance.parallel_processing.to_string())
            }
            ["storage", "backend"] => Some(self.config.storage.backend.clone()),
            ["storage", "default_path"] => Some(self.config.storage.default_path.clone()),
            ["logging", "level"] => Some(self.config.logging.level.clone()),
            _ => None,
        }
    }

    /// Update configuration value by key path
    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        let parts: Vec<&str> = key.split('.').collect();
        match parts.as_slice() {
            ["ledger", "hash_algorithm"] => {
                self.config.ledger.hash_algorithm = value.to_string();
            }
            ["ledger", "default_proof_type"] => {
                self.config.ledger.default_proof_type = value.to_string();
            }
            ["logging", "level"] => {
                self.config.logging.level = value.to_string();
            }
            ["storage", "backend"] => {
                self.config.storage.backend = value.to_string();
            }
            ["storage", "default_path"] => {
                self.config.storage.default_path = value.to_string();
            }
            _ => {
                return Err(LedgerError::validation_error(format!(
                    "Unknown configuration key: {}",
                    key
                )));
            }
        }

        self.validate()
    }

    /// Generate configuration file template
    pub fn generate_template() -> String {
        let template_config = GlobalConfig::default();
        let content = toml::to_string_pretty(&template_config)
            .unwrap_or_else(|_| "# Error generating template".to_string());

        format!(
            "# ZKP Dataset Ledger Configuration\n# Generated template - customize as needed\n\n{}\n\n# Environment Variables:\n# ZKP_HASH_ALGORITHM - Override hash algorithm\n# ZKP_DEFAULT_PROOF_TYPE - Override default proof type\n# ZKP_LOG_LEVEL - Override log level\n# ZKP_STORAGE_PATH - Override storage path\n# ZKP_PARALLEL_PROCESSING - Enable/disable parallel processing\n",
            content
        )
    }

    /// Get performance settings as a map
    pub fn performance_settings(&self) -> HashMap<String, String> {
        let mut settings = HashMap::new();
        settings.insert(
            "parallel_processing".to_string(),
            self.config.performance.parallel_processing.to_string(),
        );
        settings.insert(
            "worker_threads".to_string(),
            self.config.performance.worker_threads.to_string(),
        );
        settings.insert(
            "enable_cache".to_string(),
            self.config.performance.enable_cache.to_string(),
        );
        settings.insert(
            "cache_size_mb".to_string(),
            self.config.performance.cache_size_mb.to_string(),
        );
        settings
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let manager = ConfigManager::new();
        assert_eq!(manager.config().ledger.hash_algorithm, "sha3-256");
        assert_eq!(manager.config().ledger.default_proof_type, "integrity");
        assert!(manager.config().performance.parallel_processing);
    }

    #[test]
    fn test_config_validation() {
        let mut manager = ConfigManager::new();

        // Valid configuration should pass
        assert!(manager.validate().is_ok());

        // Invalid hash algorithm should fail
        manager.config.ledger.hash_algorithm = "invalid-hash".to_string();
        assert!(manager.validate().is_err());
    }

    #[test]
    fn test_config_get_set() {
        let mut manager = ConfigManager::new();

        // Test get
        assert_eq!(
            manager.get_value("ledger.hash_algorithm"),
            Some("sha3-256".to_string())
        );

        // Test set
        assert!(manager
            .set_value("ledger.hash_algorithm", "sha2-256")
            .is_ok());
        assert_eq!(
            manager.get_value("ledger.hash_algorithm"),
            Some("sha2-256".to_string())
        );

        // Test invalid key
        assert!(manager.set_value("invalid.key", "value").is_err());
    }

    #[test]
    fn test_config_file_operations() {
        let manager = ConfigManager::new();
        let temp_file = NamedTempFile::new().unwrap();

        // Test save
        assert!(manager.save_to_file(temp_file.path()).is_ok());

        // Test load
        let loaded = ConfigManager::load_from_file(temp_file.path()).unwrap();
        assert_eq!(
            loaded.config().ledger.hash_algorithm,
            manager.config().ledger.hash_algorithm
        );
    }

    #[test]
    fn test_template_generation() {
        let template = ConfigManager::generate_template();
        assert!(template.contains("ZKP Dataset Ledger Configuration"));
        assert!(template.contains("[ledger]"));
        assert!(template.contains("[security]"));
        assert!(template.contains("Environment Variables"));
    }
}
