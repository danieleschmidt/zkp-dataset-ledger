//! Configuration management for ZKP Dataset Ledger.

use crate::crypto::hash::HashAlgorithm;
use crate::error::LedgerError;
use crate::proof::ProofType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Main application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub storage: StorageConfig,
    pub crypto: CryptoConfig,
    pub performance: PerformanceConfig,
    pub security: SecurityConfig,
    pub monitoring: MonitoringConfig,
}

/// Storage backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub backend: String,
    pub connection_string: String,
    pub options: HashMap<String, String>,
}

/// Cryptographic configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    pub default_hash_algorithm: HashAlgorithm,
    pub default_proof_type: ProofType,
    pub use_groth16: bool,
    pub merkle_tree_depth: u32,
    pub chunk_size: usize,
}

/// Performance configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub parallel_proof_generation: bool,
    pub max_concurrent_operations: usize,
    pub batch_size: usize,
    pub cache_size_mb: usize,
    pub connection_pool_size: usize,
}

/// Security configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_access_control: bool,
    pub require_authentication: bool,
    pub audit_logging: bool,
    pub rate_limiting: RateLimitConfig,
    pub encryption: EncryptionConfig,
}

/// Rate limiting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

/// Encryption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub encrypt_at_rest: bool,
    pub encryption_algorithm: String,
    pub key_rotation_days: u32,
}

/// Monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub metrics_endpoint: String,
    pub log_level: String,
    pub health_check_interval_seconds: u64,
    pub alert_thresholds: AlertThresholds,
}

/// Alert threshold configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub max_response_time_ms: u64,
    pub max_error_rate_percent: f64,
    pub max_memory_usage_percent: f64,
    pub max_disk_usage_percent: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            crypto: CryptoConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: "memory".to_string(),
            connection_string: "".to_string(),
            options: HashMap::new(),
        }
    }
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            default_hash_algorithm: HashAlgorithm::default(),
            default_proof_type: ProofType::DatasetIntegrity,
            use_groth16: true,
            merkle_tree_depth: 20,
            chunk_size: 1000,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel_proof_generation: true,
            max_concurrent_operations: num_cpus::get(),
            batch_size: 100,
            cache_size_mb: 256,
            connection_pool_size: 10,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_access_control: false,
            require_authentication: false,
            audit_logging: true,
            rate_limiting: RateLimitConfig::default(),
            encryption: EncryptionConfig::default(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 1000,
            burst_size: 100,
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            encrypt_at_rest: false,
            encryption_algorithm: "AES-256-GCM".to_string(),
            key_rotation_days: 90,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_endpoint: "http://localhost:9090/metrics".to_string(),
            log_level: "info".to_string(),
            health_check_interval_seconds: 30,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_response_time_ms: 5000,
            max_error_rate_percent: 5.0,
            max_memory_usage_percent: 80.0,
            max_disk_usage_percent: 85.0,
        }
    }
}

impl Config {
    /// Load configuration from a TOML file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, LedgerError> {
        let contents = fs::read_to_string(path)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        Self::from_toml(&contents)
    }

    /// Load configuration from TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, LedgerError> {
        toml::from_str(toml_str)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to parse config: {}", e)))
    }

    /// Load configuration from JSON file.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, LedgerError> {
        let contents = fs::read_to_string(path)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        Self::from_json(&contents)
    }

    /// Load configuration from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, LedgerError> {
        serde_json::from_str(json_str)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to parse JSON config: {}", e)))
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self, LedgerError> {
        let mut config = Config::default();

        // Storage configuration
        if let Ok(backend) = std::env::var("ZKP_STORAGE_BACKEND") {
            config.storage.backend = backend;
        }
        if let Ok(connection) = std::env::var("ZKP_STORAGE_CONNECTION") {
            config.storage.connection_string = connection;
        }

        // Crypto configuration
        if let Ok(use_groth16) = std::env::var("ZKP_USE_GROTH16") {
            config.crypto.use_groth16 = use_groth16.parse()
                .map_err(|e| LedgerError::ConfigError(format!("Invalid ZKP_USE_GROTH16: {}", e)))?;
        }

        // Performance configuration
        if let Ok(parallel) = std::env::var("ZKP_PARALLEL_PROOFS") {
            config.performance.parallel_proof_generation = parallel.parse()
                .map_err(|e| LedgerError::ConfigError(format!("Invalid ZKP_PARALLEL_PROOFS: {}", e)))?;
        }
        if let Ok(batch_size) = std::env::var("ZKP_BATCH_SIZE") {
            config.performance.batch_size = batch_size.parse()
                .map_err(|e| LedgerError::ConfigError(format!("Invalid ZKP_BATCH_SIZE: {}", e)))?;
        }

        // Security configuration
        if let Ok(access_control) = std::env::var("ZKP_ACCESS_CONTROL") {
            config.security.enable_access_control = access_control.parse()
                .map_err(|e| LedgerError::ConfigError(format!("Invalid ZKP_ACCESS_CONTROL: {}", e)))?;
        }

        // Monitoring configuration
        if let Ok(log_level) = std::env::var("ZKP_LOG_LEVEL") {
            config.monitoring.log_level = log_level;
        }
        if let Ok(metrics_endpoint) = std::env::var("ZKP_METRICS_ENDPOINT") {
            config.monitoring.metrics_endpoint = metrics_endpoint;
        }

        Ok(config)
    }

    /// Save configuration to a TOML file.
    pub fn save_toml<P: AsRef<Path>>(&self, path: P) -> Result<(), LedgerError> {
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to serialize config: {}", e)))?;

        fs::write(path, toml_str)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to write config file: {}", e)))
    }

    /// Save configuration to a JSON file.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), LedgerError> {
        let json_str = serde_json::to_string_pretty(self)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to serialize config: {}", e)))?;

        fs::write(path, json_str)
            .map_err(|e| LedgerError::ConfigError(format!("Failed to write config file: {}", e)))
    }

    /// Validate configuration settings.
    pub fn validate(&self) -> Result<(), LedgerError> {
        // Validate storage backend
        match self.storage.backend.as_str() {
            "memory" | "rocksdb" | "postgres" => {},
            _ => return Err(LedgerError::ConfigError(
                format!("Unsupported storage backend: {}", self.storage.backend)
            )),
        }

        // Validate performance settings
        if self.performance.max_concurrent_operations == 0 {
            return Err(LedgerError::ConfigError(
                "max_concurrent_operations must be greater than 0".to_string()
            ));
        }

        if self.performance.batch_size == 0 {
            return Err(LedgerError::ConfigError(
                "batch_size must be greater than 0".to_string()
            ));
        }

        // Validate crypto settings
        if self.crypto.merkle_tree_depth == 0 || self.crypto.merkle_tree_depth > 64 {
            return Err(LedgerError::ConfigError(
                "merkle_tree_depth must be between 1 and 64".to_string()
            ));
        }

        // Validate monitoring settings
        if !["trace", "debug", "info", "warn", "error"].contains(&self.monitoring.log_level.as_str()) {
            return Err(LedgerError::ConfigError(
                format!("Invalid log level: {}", self.monitoring.log_level)
            ));
        }

        Ok(())
    }

    /// Merge with another configuration, with the other config taking precedence.
    pub fn merge(&mut self, other: Config) {
        // Only merge non-default values to avoid overriding with defaults
        if other.storage.backend != "memory" {
            self.storage.backend = other.storage.backend;
        }
        if !other.storage.connection_string.is_empty() {
            self.storage.connection_string = other.storage.connection_string;
        }
        if !other.storage.options.is_empty() {
            self.storage.options.extend(other.storage.options);
        }

        // Merge other sections similarly...
        self.crypto = other.crypto;
        self.performance = other.performance;
        self.security = other.security;
        self.monitoring = other.monitoring;
    }

    /// Get storage configuration for creating storage backend.
    pub fn get_storage_config(&self) -> (&str, &str) {
        (&self.storage.backend, &self.storage.connection_string)
    }

    /// Generate example configuration file content.
    pub fn example_toml() -> String {
        let config = Config::default();
        toml::to_string_pretty(&config).unwrap_or_else(|_| String::from("# Failed to generate example"))
    }

    /// Generate example configuration as JSON.
    pub fn example_json() -> String {
        let config = Config::default();
        serde_json::to_string_pretty(&config).unwrap_or_else(|_| String::from("{}"))
    }
}

/// Configuration builder for fluent configuration creation.
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    pub fn storage_backend(mut self, backend: &str) -> Self {
        self.config.storage.backend = backend.to_string();
        self
    }

    pub fn connection_string(mut self, connection: &str) -> Self {
        self.config.storage.connection_string = connection.to_string();
        self
    }

    pub fn use_groth16(mut self, use_groth16: bool) -> Self {
        self.config.crypto.use_groth16 = use_groth16;
        self
    }

    pub fn parallel_proofs(mut self, parallel: bool) -> Self {
        self.config.performance.parallel_proof_generation = parallel;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.performance.batch_size = size;
        self
    }

    pub fn enable_monitoring(mut self, enabled: bool) -> Self {
        self.config.monitoring.enabled = enabled;
        self
    }

    pub fn log_level(mut self, level: &str) -> Self {
        self.config.monitoring.log_level = level.to_string();
        self
    }

    pub fn build(self) -> Result<Config, LedgerError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.storage.backend, "memory");
        assert!(config.crypto.use_groth16);
        assert!(config.performance.parallel_proof_generation);
        assert!(config.monitoring.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        // Test invalid storage backend
        config.storage.backend = "invalid".to_string();
        assert!(config.validate().is_err());

        // Test invalid performance settings
        config.storage.backend = "memory".to_string();
        config.performance.max_concurrent_operations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_toml_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.storage.backend, deserialized.storage.backend);
        assert_eq!(config.crypto.use_groth16, deserialized.crypto.use_groth16);
    }

    #[test]
    fn test_config_json_serialization() {
        let config = Config::default();
        let json_str = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json_str).unwrap();

        assert_eq!(config.storage.backend, deserialized.storage.backend);
        assert_eq!(config.crypto.use_groth16, deserialized.crypto.use_groth16);
    }

    #[test]
    fn test_config_file_operations() -> Result<(), Box<dyn std::error::Error>> {
        let config = Config::default();

        // Test TOML file operations
        let mut toml_file = NamedTempFile::new()?;
        let toml_path = toml_file.path().with_extension("toml");
        config.save_toml(&toml_path)?;
        let loaded_config = Config::from_file(&toml_path)?;
        assert_eq!(config.storage.backend, loaded_config.storage.backend);

        // Test JSON file operations
        let mut json_file = NamedTempFile::new()?;
        let json_path = json_file.path().with_extension("json");
        config.save_json(&json_path)?;
        let loaded_config = Config::from_json_file(&json_path)?;
        assert_eq!(config.storage.backend, loaded_config.storage.backend);

        std::fs::remove_file(toml_path).ok();
        std::fs::remove_file(json_path).ok();
        Ok(())
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .storage_backend("rocksdb")
            .connection_string("./test.db")
            .use_groth16(false)
            .parallel_proofs(true)
            .batch_size(50)
            .log_level("debug")
            .build()
            .unwrap();

        assert_eq!(config.storage.backend, "rocksdb");
        assert_eq!(config.storage.connection_string, "./test.db");
        assert!(!config.crypto.use_groth16);
        assert!(config.performance.parallel_proof_generation);
        assert_eq!(config.performance.batch_size, 50);
        assert_eq!(config.monitoring.log_level, "debug");
    }

    #[test]
    fn test_config_merge() {
        let mut base_config = Config::default();
        let override_config = ConfigBuilder::new()
            .storage_backend("postgres")
            .use_groth16(false)
            .build()
            .unwrap();

        base_config.merge(override_config);

        assert_eq!(base_config.storage.backend, "postgres");
        assert!(!base_config.crypto.use_groth16);
        assert!(base_config.performance.parallel_proof_generation); // Should remain default
    }
}