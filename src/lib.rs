pub mod circuits;
pub mod crypto;
pub mod dataset;
pub mod error;
pub mod ledger;
pub mod proof;
pub mod storage;
pub mod security;
pub mod monitoring;
pub mod recovery;
pub mod performance;

pub use dataset::Dataset;
pub use error::LedgerError;
pub use ledger::Ledger;
pub use proof::{Proof, ProofConfig};
pub use security::{SecurityConfig, SecurityValidator, User, Role, Permission};
pub use monitoring::{Monitor, SystemMetrics, HealthStatus};
pub use recovery::{BackupRecoverySystem, BackupConfig, FaultToleranceManager};
pub use performance::{PerformanceOptimizer, PerformanceConfig, ParallelProcessor};

pub type Result<T> = std::result::Result<T, LedgerError>;
