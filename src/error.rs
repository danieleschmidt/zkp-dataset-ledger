use thiserror::Error;

#[derive(Error, Debug)]
pub enum LedgerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Proof verification failed")]
    ProofVerificationFailed,

    #[error("Invalid dataset: {0}")]
    InvalidDataset(String),

    #[error("Storage error: {0}")]
    Storage(String),
}
