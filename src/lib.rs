// ZKP Dataset Ledger - Main Library Entry Point
//
// This library provides cryptographic provenance for ML pipelines using zero-knowledge proofs.
// It implements a secure, auditable ledger for dataset operations and transformations.

// Generation 1-3 SDLC Implementation - Progressive Enhancement Complete

// Core simple implementation (Generation 1 - MAKE IT WORK)
pub use cache_system::{CacheManager, CacheStats, PerformanceCache};
pub use concurrent_engine::{
    BatchProcessor, ConcurrentEngine, ExecutionContext, ParallelProofGenerator, TaskPriority,
};
pub use config_manager::{ConfigManager, GlobalConfig};
pub use error_handling::{
    ContextualError, ErrorCategory, ErrorContext, ErrorHandler, ErrorSeverity,
};
pub use lib_simple::*;
pub use monitoring_system::{HealthStatus, MonitoringSystem, PerformanceMetrics, SystemMetrics};

// Advanced cryptographic modules (Generation 1-3 enhancements)
pub mod advanced_ledger;
pub mod distributed_consensus;
// pub mod research;  // Disabled temporarily for compilation
pub mod security_enhanced;
pub mod zkp_circuits;

// Re-export advanced features for production use
pub use zkp_circuits::{
    DatasetIntegrityCircuit, DifferentialPrivacyCircuit, StatisticalBounds,
    StatisticalPropertiesCircuit, ZkCircuitType, ZkIntegrityProof, ZkProofConfig, ZkProofSystem,
    ZkStatisticalProof,
};

pub use advanced_ledger::{
    AdvancedLedger, AdvancedLedgerEntry, AdvancedMetrics, AdvancedProof, CompositeProof,
    ConcurrencyMetrics, LedgerMetadata, MerkleInclusionProof, MerkleTree, MultiSigProof,
    ProofTimingMetrics, RetentionPolicy, SecurityLevel, SecurityMetrics, StorageMetrics,
    VerificationTimingMetrics,
};

// Core modules
pub mod cache_system;
pub mod concurrent_engine;
pub mod config_manager;
pub mod error_handling;
mod lib_simple;
pub mod monitoring_system;
