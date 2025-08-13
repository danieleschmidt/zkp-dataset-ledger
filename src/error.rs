use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LedgerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Polars error: {0}")]
    Polars(String),

    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Proof generation failed: {0}")]
    ProofGeneration(String),

    #[error("Proof verification failed: {0}")]
    ProofVerification(String),

    #[error("Dataset error: {0}")]
    Dataset(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Timeout error: operation timed out after {seconds} seconds")]
    Timeout { seconds: u64 },

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<polars::error::PolarsError> for LedgerError {
    fn from(err: polars::error::PolarsError) -> Self {
        LedgerError::Polars(err.to_string())
    }
}

impl From<toml::de::Error> for LedgerError {
    fn from(err: toml::de::Error) -> Self {
        LedgerError::Config(err.to_string())
    }
}

impl From<anyhow::Error> for LedgerError {
    fn from(err: anyhow::Error) -> Self {
        LedgerError::Internal(err.to_string())
    }
}

// Custom result type for the crate
pub type Result<T> = std::result::Result<T, LedgerError>;

// Helper functions for common error patterns
impl LedgerError {
    pub fn crypto(msg: impl Into<String>) -> Self {
        LedgerError::Crypto(msg.into())
    }

    pub fn dataset(msg: impl Into<String>) -> Self {
        LedgerError::Dataset(msg.into())
    }

    pub fn storage(msg: impl Into<String>) -> Self {
        LedgerError::Storage(msg.into())
    }

    pub fn config(msg: impl Into<String>) -> Self {
        LedgerError::Config(msg.into())
    }

    pub fn validation(msg: impl Into<String>) -> Self {
        LedgerError::Validation(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        LedgerError::NotFound(msg.into())
    }

    pub fn already_exists(msg: impl Into<String>) -> Self {
        LedgerError::AlreadyExists(msg.into())
    }

    pub fn invalid_operation(msg: impl Into<String>) -> Self {
        LedgerError::InvalidOperation(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        LedgerError::Internal(msg.into())
    }
}