//! # ZKP Dataset Ledger
//!
//! A zero-knowledge proof system for ML dataset provenance.
//!
//! ## Core Concept
//!
//! When training ML models, data owners need to prove properties about their training
//! data to third parties (auditors, regulators, model consumers) WITHOUT revealing the
//! actual dataset. This is the dataset provenance problem.
//!
//! This library implements two cryptographic primitives to address it:
//!
//! ### 1. Hash Commitment Scheme
//!
//! A Pedersen-style commitment: `C = Hash(dataset_content || blinding_factor)`
//!
//! - The prover commits to dataset properties by publishing the commitment hash
//! - The prover can later reveal selected properties with a proof
//! - The verifier learns nothing about the dataset beyond what's explicitly proven
//!
//! ### 2. Schnorr-Style Sigma Protocol (Non-Interactive via Fiat-Shamir)
//!
//! Proves "I know a secret x such that H = hash(x)" without revealing x.
//! Applied here: "I know dataset D such that commitment = hash(D || nonce)".
//!
//! The Fiat-Shamir heuristic converts the interactive proof to non-interactive
//! by replacing the verifier's random challenge with a hash of the transcript.
//!
//! ## Why This Matters for ML
//!
//! - **Data sovereignty**: Dataset owners can prove training compliance without sharing data
//! - **Regulatory audit**: Prove GDPR compliance, bias metrics, data provenance
//! - **Supply chain security**: Cryptographic chain of custody for model ingredients
//! - **Federated learning**: Each party can prove their contribution without revealing it

pub mod commitment;
pub mod crypto;
pub mod ledger;
pub mod proof;

pub use commitment::{DatasetCommitment, CommitmentConfig};
pub use ledger::{DatasetLedger, LedgerEntry};
pub use proof::{DatasetProof, ProofType, ProofVerifier};

/// Core error type
#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, LedgerError>;
