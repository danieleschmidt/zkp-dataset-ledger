// ZKP Dataset Ledger - Main Library Entry Point
//
// This library provides cryptographic provenance for ML pipelines using zero-knowledge proofs.
// It implements a secure, auditable ledger for dataset operations and transformations.

// Generation 1-3 SDLC Implementation - Progressive Enhancement Complete

// Core simple implementation (Generation 1 - MAKE IT WORK)
pub use lib_simple::*;

// Advanced cryptographic modules (Generation 1-3 enhancements)
pub mod zkp_circuits;
pub mod advanced_ledger;
pub mod distributed_consensus;
pub mod security_enhanced;

// Re-export advanced features for production use
pub use zkp_circuits::{
    ZkProofSystem, ZkIntegrityProof, ZkStatisticalProof, StatisticalBounds,
    DatasetIntegrityCircuit, StatisticalPropertiesCircuit, DifferentialPrivacyCircuit,
    ZkProofConfig, ZkCircuitType
};

pub use advanced_ledger::{
    AdvancedLedger, AdvancedProof, AdvancedLedgerEntry, LedgerMetadata,
    MerkleInclusionProof, MultiSigProof, CompositeProof, SecurityLevel, RetentionPolicy,
    AdvancedMetrics, ProofTimingMetrics, VerificationTimingMetrics, StorageMetrics,
    ConcurrencyMetrics, SecurityMetrics, MerkleTree
};

// Core modules
mod lib_simple;
