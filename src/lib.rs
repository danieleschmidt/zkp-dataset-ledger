//! ZKP Dataset Ledger - Cryptographic provenance for ML pipelines
//!
//! This library provides zero-knowledge proof based dataset tracking, integrity verification,
//! and audit trail generation for machine learning pipelines.

pub mod circuits;
pub mod config;
pub mod crypto;
pub mod dataset;
pub mod error;
pub mod ledger;
pub mod monitoring;
pub mod performance;
pub mod proof;
pub mod recovery;
pub mod research;
pub mod security;
pub mod storage;

// Distributed module - advanced feature
#[cfg(feature = "distributed")]
pub mod distributed;

// Re-export the most commonly used types
pub use config::{Config, ConfigBuilder, CryptoConfig, PerformanceConfig, StorageConfig};
pub use dataset::{Dataset, DatasetFormat, DatasetStatistics};
pub use error::{LedgerError, Result};
pub use ledger::{Ledger, LedgerEntry, LedgerQuery, LedgerSummary};
pub use monitoring::{HealthStatus, Monitor, SystemMetrics};
pub use performance::{ParallelProcessor, PerformanceOptimizer};
pub use proof::{Proof, ProofConfig, ProofType};
pub use recovery::{BackupConfig, BackupRecoverySystem, FaultToleranceManager};
pub use security::{Permission, Role, SecurityConfig, SecurityValidator, User};
pub use storage::create_storage;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PROTOCOL_VERSION: &str = "1.0";

/// Initialize the ZKP Dataset Ledger library with default settings.
///
/// This function sets up logging and performs basic initialization.
/// Call this once at the start of your application.
pub fn init() -> Result<()> {
    // Initialize default tracing subscriber if none is set
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize cryptographic components
    crypto::init()?;

    Ok(())
}

/// Initialize the library with custom configuration.
pub fn init_with_config(config: &Config) -> Result<()> {
    // Set up logging based on config
    std::env::set_var("RUST_LOG", &config.monitoring.log_level);

    // Initialize crypto with specific configuration
    crypto::init_with_config(&config.crypto)?;

    Ok(())
}

/// Get library version information
pub fn version_info() -> VersionInfo {
    VersionInfo {
        library_version: VERSION.to_string(),
        protocol_version: PROTOCOL_VERSION.to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("unknown").to_string(),
        features: get_enabled_features(),
    }
}

/// Version information for the library
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VersionInfo {
    pub library_version: String,
    pub protocol_version: String,
    pub build_date: String,
    pub features: Vec<String>,
}

fn get_enabled_features() -> Vec<String> {
    let mut features = vec!["core".to_string()];

    #[cfg(feature = "rocksdb")]
    features.push("rocksdb".to_string());

    #[cfg(feature = "postgres")]
    features.push("postgres".to_string());

    #[cfg(feature = "benchmarks")]
    features.push("benchmarks".to_string());

    #[cfg(feature = "property-testing")]
    features.push("property-testing".to_string());

    #[cfg(feature = "distributed")]
    features.push("distributed".to_string());

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let version = version_info();
        assert!(!version.library_version.is_empty());
        assert!(!version.protocol_version.is_empty());
        assert!(version.features.contains(&"core".to_string()));
    }

    #[test]
    fn test_init() {
        // This should not panic
        let result = init();
        assert!(result.is_ok());
    }
}
