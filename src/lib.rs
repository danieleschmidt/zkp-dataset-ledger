// ZKP Dataset Ledger - Main Library Entry Point
//
// This library provides cryptographic provenance for ML pipelines using zero-knowledge proofs.
// It implements a secure, auditable ledger for dataset operations and transformations.

// For Generation 1 (MAKE IT WORK), we export the simple implementation
pub use lib_simple::*;

// Simple implementation is the default for now
mod lib_simple;

// Additional modules for advanced features can be added with feature flags in production