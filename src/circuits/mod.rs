//! Cryptographic circuit implementations for zero-knowledge proofs

pub mod zkp_core;

pub use zkp_core::*;

// Re-export common ZK types for consistency across the codebase
pub type Curve = ark_bls12_381::Bls12_381;
pub type Fr = ark_bls12_381::Fr;
pub type G1Projective = ark_bls12_381::G1Projective;
pub type G2Projective = ark_bls12_381::G2Projective;
