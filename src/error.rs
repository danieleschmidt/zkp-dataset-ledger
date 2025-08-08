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
    StorageError(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Dataset processing error: {0}")]
    DatasetError(String),

    #[error("Proof generation error: {0}")]
    ProofError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Polars error: {0}")]
    PolarsError(String),

    #[error("Arithmetic operation failed: {0}")]
    ArithmeticError(String),

    #[error("Circuit error: {0}")]
    CircuitError(String),
}
