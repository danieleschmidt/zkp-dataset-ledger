use thiserror::Error;

#[derive(Error, Debug)]
pub enum LedgerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

// Custom result type for the crate
pub type Result<T> = std::result::Result<T, LedgerError>;

// Helper functions for common error patterns
impl LedgerError {
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