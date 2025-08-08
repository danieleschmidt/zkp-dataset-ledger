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
pub mod security;
pub mod storage;

pub use config::{Config, ConfigBuilder, StorageConfig, CryptoConfig, PerformanceConfig as ConfigPerformanceConfig};
pub use dataset::Dataset;
pub use error::LedgerError;
pub use ledger::Ledger;
pub use monitoring::{HealthStatus, Monitor, SystemMetrics};
pub use performance::{ParallelProcessor, PerformanceConfig, PerformanceOptimizer};
pub use proof::{Proof, ProofConfig};
pub use recovery::{BackupConfig, BackupRecoverySystem, FaultToleranceManager};
pub use security::{Permission, Role, SecurityConfig, SecurityValidator, User};

pub type Result<T> = std::result::Result<T, LedgerError>;
