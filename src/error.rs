use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Comprehensive error types with detailed context and recovery information
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum LedgerError {
    // === I/O and System Errors ===
    #[error("I/O operation failed: {operation} - {source}")]
    Io {
        operation: String,
        source: String,
        path: Option<String>,
        recoverable: bool,
    },

    #[error("Permission denied: {resource} - {reason}")]
    PermissionDenied {
        resource: String,
        reason: String,
        required_permission: Option<String>,
    },

    #[error("Resource not found: {resource_type} '{resource_name}'")]
    NotFound {
        resource_type: String,
        resource_name: String,
        suggestions: Vec<String>,
    },

    #[error("Resource already exists: {resource_type} '{resource_name}'")]
    AlreadyExists {
        resource_type: String,
        resource_name: String,
        existing_location: Option<String>,
    },

    #[error("Timeout occurred: {operation} exceeded {timeout_ms}ms")]
    Timeout {
        operation: String,
        timeout_ms: u64,
        elapsed_ms: u64,
        retryable: bool,
    },

    // === Configuration and Validation Errors ===
    #[error("Configuration error in {component}: {message}")]
    Configuration {
        component: String,
        message: String,
        invalid_values: HashMap<String, String>,
        valid_examples: Vec<String>,
    },

    #[error("Validation failed for {field}: {reason}")]
    Validation {
        field: String,
        reason: String,
        provided_value: String,
        constraints: Vec<String>,
        suggestions: Vec<String>,
    },

    #[error("Schema validation failed: {details}")]
    SchemaValidation {
        details: String,
        expected_schema: Option<String>,
        actual_schema: Option<String>,
        violations: Vec<SchemaViolation>,
    },

    // === Security and Authentication Errors ===
    #[error("Authentication failed: {reason}")]
    Authentication {
        reason: String,
        user_id: Option<String>,
        attempted_operation: String,
        lockout_time: Option<DateTime<Utc>>,
    },

    #[error("Authorization failed: insufficient permissions for {operation}")]
    Authorization {
        operation: String,
        required_permissions: Vec<String>,
        user_permissions: Vec<String>,
        user_id: Option<String>,
    },

    #[error("Security policy violation: {policy} - {details}")]
    SecurityViolation {
        policy: String,
        details: String,
        severity: SecuritySeverity,
        remediation_steps: Vec<String>,
    },

    #[error("Rate limit exceeded: {limit} requests per {window}")]
    RateLimit {
        limit: u32,
        window: String,
        current_count: u32,
        reset_time: DateTime<Utc>,
    },

    #[error("Input sanitization failed: {reason}")]
    InputSanitization {
        reason: String,
        original_input: String,
        detected_threats: Vec<String>,
    },

    // === Cryptographic Errors ===
    #[error("Cryptographic operation failed: {operation} - {details}")]
    Cryptographic {
        operation: String,
        details: String,
        algorithm: Option<String>,
        key_id: Option<String>,
        recoverable: bool,
    },

    #[error("Key management error: {operation} failed for key '{key_id}'")]
    KeyManagement {
        operation: String,
        key_id: String,
        reason: String,
        security_level: String,
    },

    #[error("Proof verification failed: {reason}")]
    ProofVerification {
        reason: String,
        proof_type: String,
        expected_hash: Option<String>,
        actual_hash: Option<String>,
        verification_steps: Vec<String>,
    },

    #[error("Circuit constraint violation in {circuit}: {constraint}")]
    CircuitConstraint {
        circuit: String,
        constraint: String,
        witness_values: HashMap<String, String>,
        debugging_info: Vec<String>,
    },

    // === Data Processing Errors ===
    #[error("Dataset processing failed: {stage} - {reason}")]
    DatasetProcessing {
        stage: String,
        reason: String,
        dataset_name: String,
        row_count: Option<u64>,
        column_count: Option<u64>,
        partial_results: bool,
    },

    #[error("Data corruption detected in {location}: {details}")]
    DataCorruption {
        location: String,
        details: String,
        corruption_type: CorruptionType,
        checksum_expected: Option<String>,
        checksum_actual: Option<String>,
        recovery_possible: bool,
    },

    #[error("Format error: expected {expected_format}, found {actual_format}")]
    FormatError {
        expected_format: String,
        actual_format: String,
        file_path: Option<String>,
        conversion_available: bool,
        supported_formats: Vec<String>,
    },

    #[error("Arithmetic operation failed: {operation} - {reason}")]
    Arithmetic {
        operation: String,
        reason: String,
        operands: Vec<String>,
        overflow_detected: bool,
    },

    // === Storage and Persistence Errors ===
    #[error("Storage backend error: {backend} - {operation} failed")]
    Storage {
        backend: String,
        operation: String,
        details: String,
        connection_info: Option<String>,
        retry_after_ms: Option<u64>,
    },

    #[error("Database transaction failed: {reason}")]
    Transaction {
        reason: String,
        transaction_id: Option<String>,
        rollback_successful: bool,
        affected_tables: Vec<String>,
    },

    #[error("Backup operation failed: {operation} - {reason}")]
    Backup {
        operation: String,
        reason: String,
        backup_id: Option<String>,
        partial_backup_available: bool,
    },

    #[error("Recovery operation failed: {stage} - {reason}")]
    Recovery {
        stage: String,
        reason: String,
        recovery_point_id: Option<String>,
        data_loss_risk: bool,
    },

    // === Network and Communication Errors ===
    #[error("Network error: {operation} to {endpoint} failed - {reason}")]
    Network {
        operation: String,
        endpoint: String,
        reason: String,
        status_code: Option<u16>,
        retry_count: u32,
    },

    #[error("Service unavailable: {service} is {status}")]
    ServiceUnavailable {
        service: String,
        status: String,
        estimated_recovery: Option<DateTime<Utc>>,
        alternative_endpoints: Vec<String>,
    },

    // === Performance and Resource Errors ===
    #[error("Resource exhaustion: {resource} limit exceeded")]
    ResourceExhaustion {
        resource: String,
        limit: u64,
        current_usage: u64,
        suggested_actions: Vec<String>,
    },

    #[error("Performance degradation detected: {metric} exceeded threshold")]
    PerformanceDegradation {
        metric: String,
        threshold: f64,
        current_value: f64,
        trend_analysis: Vec<String>,
    },

    // === Serialization and Communication Errors ===
    #[error("Serialization failed: {format} - {reason}")]
    Serialization {
        format: String,
        reason: String,
        data_type: String,
        fallback_available: bool,
    },

    #[error("Deserialization failed: {format} - {reason}")]
    Deserialization {
        format: String,
        reason: String,
        expected_type: String,
        partial_data_recovered: bool,
    },

    // === Generic and Legacy Errors ===
    #[error("Internal error: {details}")]
    Internal {
        details: String,
        error_code: String,
        debug_info: HashMap<String, String>,
        contact_support: bool,
    },

    #[error("Operation not supported: {operation} is not available in {context}")]
    NotSupported {
        operation: String,
        context: String,
        alternatives: Vec<String>,
        feature_flag: Option<String>,
    },

    // Legacy compatibility variants
    #[error("Legacy I/O error: {0}")]
    LegacyIo(String),

    #[error("Legacy serialization error: {0}")]
    LegacySerialization(String),

    #[error("Legacy JSON error: {0}")]
    LegacyJson(String),
}

/// Security violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Types of data corruption
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CorruptionType {
    ChecksumMismatch,
    StructuralDamage,
    PartialCorruption,
    MissingData,
    InvalidFormat,
}

/// Schema validation violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaViolation {
    pub field: String,
    pub violation_type: String,
    pub expected: String,
    pub actual: String,
    pub path: Option<String>,
}

/// Error context for enhanced debugging and recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub timestamp: DateTime<Utc>,
    pub trace_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub request_id: Option<String>,
    pub environment: String,
    pub version: String,
    pub metadata: HashMap<String, String>,
}

/// Recovery suggestion for error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverySuggestion {
    pub action: String,
    pub description: String,
    pub automatic: bool,
    pub risk_level: String,
    pub estimated_time_ms: Option<u64>,
    pub prerequisites: Vec<String>,
}

/// Enhanced error with context and recovery information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedError {
    pub error: LedgerError,
    pub context: ErrorContext,
    pub recovery_suggestions: Vec<RecoverySuggestion>,
    pub related_errors: Vec<String>,
    pub error_chain: Vec<String>,
    pub user_message: Option<String>,
    pub technical_details: HashMap<String, String>,
}

// Conversion implementations for backward compatibility
impl From<std::io::Error> for LedgerError {
    fn from(err: std::io::Error) -> Self {
        LedgerError::Io {
            operation: "unknown".to_string(),
            source: err.to_string(),
            path: None,
            recoverable: true,
        }
    }
}

impl From<bincode::Error> for LedgerError {
    fn from(err: bincode::Error) -> Self {
        LedgerError::Serialization {
            format: "bincode".to_string(),
            reason: err.to_string(),
            data_type: "unknown".to_string(),
            fallback_available: false,
        }
    }
}

impl From<serde_json::Error> for LedgerError {
    fn from(err: serde_json::Error) -> Self {
        LedgerError::Serialization {
            format: "json".to_string(),
            reason: err.to_string(),
            data_type: "unknown".to_string(),
            fallback_available: false,
        }
    }
}

impl From<csv::Error> for LedgerError {
    fn from(err: csv::Error) -> Self {
        LedgerError::DatasetProcessing {
            stage: "csv_parsing".to_string(),
            reason: err.to_string(),
            dataset_name: "unknown".to_string(),
            row_count: None,
            column_count: None,
            partial_results: false,
        }
    }
}

impl LedgerError {
    /// Check if the error is recoverable through retry
    pub fn is_recoverable(&self) -> bool {
        match self {
            LedgerError::Io { recoverable, .. } => *recoverable,
            LedgerError::Timeout { retryable, .. } => *retryable,
            LedgerError::Network { .. } => true,
            LedgerError::ServiceUnavailable { .. } => true,
            LedgerError::Storage { retry_after_ms, .. } => retry_after_ms.is_some(),
            LedgerError::Cryptographic { recoverable, .. } => *recoverable,
            LedgerError::DataCorruption {
                recovery_possible, ..
            } => *recovery_possible,
            _ => false,
        }
    }

    /// Get retry delay in milliseconds if applicable
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            LedgerError::RateLimit { reset_time, .. } => {
                let now = Utc::now();
                if *reset_time > now {
                    Some(reset_time.signed_duration_since(now).num_milliseconds() as u64)
                } else {
                    Some(0)
                }
            }
            LedgerError::Storage { retry_after_ms, .. } => *retry_after_ms,
            LedgerError::Timeout { .. } => Some(1000), // 1 second base delay
            LedgerError::Network { retry_count, .. } => {
                // Exponential backoff: 2^retry_count * 100ms, max 30 seconds
                let delay = (2_u64.pow(*retry_count)).saturating_mul(100);
                Some(delay.min(30_000))
            }
            _ => None,
        }
    }

    /// Get user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            LedgerError::NotFound {
                resource_type,
                resource_name,
                suggestions,
            } => {
                let mut msg = format!("Could not find {} '{}'", resource_type, resource_name);
                if !suggestions.is_empty() {
                    msg.push_str(&format!(". Did you mean: {}?", suggestions.join(", ")));
                }
                msg
            }
            LedgerError::Validation {
                field,
                reason,
                suggestions,
                ..
            } => {
                let mut msg = format!("Invalid {}: {}", field, reason);
                if !suggestions.is_empty() {
                    msg.push_str(&format!(". Try: {}", suggestions.join(", ")));
                }
                msg
            }
            LedgerError::Authentication {
                reason,
                lockout_time,
                ..
            } => {
                let mut msg = format!("Authentication failed: {}", reason);
                if let Some(lockout) = lockout_time {
                    msg.push_str(&format!(
                        ". Account locked until {}",
                        lockout.format("%Y-%m-%d %H:%M:%S UTC")
                    ));
                }
                msg
            }
            LedgerError::RateLimit {
                limit,
                window,
                reset_time,
                ..
            } => {
                format!(
                    "Too many requests. Limit: {} per {}. Try again after {}",
                    limit,
                    window,
                    reset_time.format("%H:%M:%S UTC")
                )
            }
            _ => self.to_string(),
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            LedgerError::Internal { .. } => ErrorSeverity::Critical,
            LedgerError::DataCorruption { .. } => ErrorSeverity::Critical,
            LedgerError::SecurityViolation { severity, .. } => match severity {
                SecuritySeverity::Critical => ErrorSeverity::Critical,
                SecuritySeverity::High => ErrorSeverity::High,
                SecuritySeverity::Medium => ErrorSeverity::Medium,
                SecuritySeverity::Low => ErrorSeverity::Low,
            },
            LedgerError::Authentication { .. } => ErrorSeverity::High,
            LedgerError::Authorization { .. } => ErrorSeverity::High,
            LedgerError::ProofVerification { .. } => ErrorSeverity::High,
            LedgerError::CircuitConstraint { .. } => ErrorSeverity::High,
            LedgerError::KeyManagement { .. } => ErrorSeverity::High,
            LedgerError::Configuration { .. } => ErrorSeverity::Medium,
            LedgerError::Validation { .. } => ErrorSeverity::Medium,
            LedgerError::DatasetProcessing { .. } => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }

    /// Get technical debugging information
    pub fn debug_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("error_type".to_string(), self.error_type());
        info.insert("severity".to_string(), format!("{:?}", self.severity()));
        info.insert("recoverable".to_string(), self.is_recoverable().to_string());

        match self {
            LedgerError::Cryptographic {
                operation,
                algorithm,
                key_id,
                ..
            } => {
                info.insert("crypto_operation".to_string(), operation.clone());
                if let Some(alg) = algorithm {
                    info.insert("algorithm".to_string(), alg.clone());
                }
                if let Some(kid) = key_id {
                    info.insert("key_id".to_string(), kid.clone());
                }
            }
            LedgerError::CircuitConstraint {
                circuit,
                witness_values,
                ..
            } => {
                info.insert("circuit".to_string(), circuit.clone());
                for (var, val) in witness_values {
                    info.insert(format!("witness_{}", var), val.clone());
                }
            }
            LedgerError::DatasetProcessing {
                dataset_name,
                row_count,
                column_count,
                ..
            } => {
                info.insert("dataset".to_string(), dataset_name.clone());
                if let Some(rows) = row_count {
                    info.insert("row_count".to_string(), rows.to_string());
                }
                if let Some(cols) = column_count {
                    info.insert("column_count".to_string(), cols.to_string());
                }
            }
            _ => {}
        }

        info
    }

    /// Get error type as string for categorization
    pub fn error_type(&self) -> String {
        match self {
            LedgerError::Io { .. } | LedgerError::LegacyIo(_) => "io",
            LedgerError::PermissionDenied { .. } => "permission",
            LedgerError::NotFound { .. } => "not_found",
            LedgerError::AlreadyExists { .. } => "already_exists",
            LedgerError::Timeout { .. } => "timeout",
            LedgerError::Configuration { .. } => "configuration",
            LedgerError::Validation { .. } => "validation",
            LedgerError::SchemaValidation { .. } => "schema_validation",
            LedgerError::Authentication { .. } => "authentication",
            LedgerError::Authorization { .. } => "authorization",
            LedgerError::SecurityViolation { .. } => "security_violation",
            LedgerError::RateLimit { .. } => "rate_limit",
            LedgerError::InputSanitization { .. } => "input_sanitization",
            LedgerError::Cryptographic { .. } => "cryptographic",
            LedgerError::KeyManagement { .. } => "key_management",
            LedgerError::ProofVerification { .. } => "proof_verification",
            LedgerError::CircuitConstraint { .. } => "circuit_constraint",
            LedgerError::DatasetProcessing { .. } => "dataset_processing",
            LedgerError::DataCorruption { .. } => "data_corruption",
            LedgerError::FormatError { .. } => "format_error",
            LedgerError::Arithmetic { .. } => "arithmetic",
            LedgerError::Storage { .. } => "storage",
            LedgerError::Transaction { .. } => "transaction",
            LedgerError::Backup { .. } => "backup",
            LedgerError::Recovery { .. } => "recovery",
            LedgerError::Network { .. } => "network",
            LedgerError::ServiceUnavailable { .. } => "service_unavailable",
            LedgerError::ResourceExhaustion { .. } => "resource_exhaustion",
            LedgerError::PerformanceDegradation { .. } => "performance_degradation",
            LedgerError::Serialization { .. } | LedgerError::LegacySerialization(_) => {
                "serialization"
            }
            LedgerError::Deserialization { .. } => "deserialization",
            LedgerError::Internal { .. } => "internal",
            LedgerError::NotSupported { .. } => "not_supported",
            LedgerError::LegacyJson(_) => "json",
        }
        .to_string()
    }
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error handling utilities
pub struct ErrorHandler {
    context: ErrorContext,
}

impl ErrorHandler {
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            context: ErrorContext {
                operation: operation.to_string(),
                component: component.to_string(),
                timestamp: Utc::now(),
                trace_id: Some(uuid::Uuid::new_v4().to_string()),
                user_id: None,
                session_id: None,
                request_id: None,
                environment: std::env::var("ENVIRONMENT")
                    .unwrap_or_else(|_| "development".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                metadata: HashMap::new(),
            },
        }
    }

    pub fn with_user(mut self, user_id: &str) -> Self {
        self.context.user_id = Some(user_id.to_string());
        self
    }

    pub fn with_session(mut self, session_id: &str) -> Self {
        self.context.session_id = Some(session_id.to_string());
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.context
            .metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Enhance an error with context and recovery suggestions
    pub fn enhance_error(&self, error: LedgerError) -> EnhancedError {
        let recovery_suggestions = self.generate_recovery_suggestions(&error);
        let user_message = error.user_message();

        EnhancedError {
            error: error.clone(),
            context: self.context.clone(),
            recovery_suggestions,
            related_errors: vec![],
            error_chain: vec![error.to_string()],
            user_message: Some(user_message),
            technical_details: error.debug_info(),
        }
    }

    /// Generate recovery suggestions based on error type
    fn generate_recovery_suggestions(&self, error: &LedgerError) -> Vec<RecoverySuggestion> {
        let mut suggestions = Vec::new();

        match error {
            LedgerError::NotFound {
                suggestions: error_suggestions,
                ..
            } => {
                for suggestion in error_suggestions {
                    suggestions.push(RecoverySuggestion {
                        action: "retry_with_suggestion".to_string(),
                        description: format!("Try using '{}'", suggestion),
                        automatic: false,
                        risk_level: "low".to_string(),
                        estimated_time_ms: Some(100),
                        prerequisites: vec![],
                    });
                }
            }
            LedgerError::Network { retry_count, .. } if *retry_count < 3 => {
                suggestions.push(RecoverySuggestion {
                    action: "retry_with_backoff".to_string(),
                    description: "Retry the network operation with exponential backoff".to_string(),
                    automatic: true,
                    risk_level: "low".to_string(),
                    estimated_time_ms: error.retry_delay_ms(),
                    prerequisites: vec!["network_connectivity".to_string()],
                });
            }
            LedgerError::Storage { .. } => {
                suggestions.push(RecoverySuggestion {
                    action: "check_storage_connection".to_string(),
                    description: "Verify storage backend connectivity and retry".to_string(),
                    automatic: false,
                    risk_level: "medium".to_string(),
                    estimated_time_ms: Some(5000),
                    prerequisites: vec!["storage_permissions".to_string()],
                });
            }
            LedgerError::DataCorruption {
                recovery_possible: true,
                ..
            } => {
                suggestions.push(RecoverySuggestion {
                    action: "restore_from_backup".to_string(),
                    description: "Attempt to restore corrupted data from the latest backup"
                        .to_string(),
                    automatic: false,
                    risk_level: "high".to_string(),
                    estimated_time_ms: Some(30000),
                    prerequisites: vec![
                        "backup_available".to_string(),
                        "admin_permission".to_string(),
                    ],
                });
            }
            _ => {}
        }

        // Always offer to contact support for critical errors
        if error.severity() == ErrorSeverity::Critical {
            suggestions.push(RecoverySuggestion {
                action: "contact_support".to_string(),
                description: "Contact technical support with error details".to_string(),
                automatic: false,
                risk_level: "none".to_string(),
                estimated_time_ms: None,
                prerequisites: vec!["support_access".to_string()],
            });
        }

        suggestions
    }

    /// Log error with structured information
    pub fn log_error(&self, error: &EnhancedError) {
        match error.error.severity() {
            ErrorSeverity::Critical => {
                tracing::error!(
                    error_type = %error.error.error_type(),
                    component = %error.context.component,
                    operation = %error.context.operation,
                    trace_id = ?error.context.trace_id,
                    user_id = ?error.context.user_id,
                    technical_details = ?error.technical_details,
                    "Critical error occurred: {}",
                    error.error
                );
            }
            ErrorSeverity::High => {
                tracing::error!(
                    error_type = %error.error.error_type(),
                    component = %error.context.component,
                    operation = %error.context.operation,
                    trace_id = ?error.context.trace_id,
                    "High severity error: {}",
                    error.error
                );
            }
            ErrorSeverity::Medium => {
                tracing::warn!(
                    error_type = %error.error.error_type(),
                    component = %error.context.component,
                    operation = %error.context.operation,
                    "Medium severity error: {}",
                    error.error
                );
            }
            ErrorSeverity::Low => {
                tracing::info!(
                    error_type = %error.error.error_type(),
                    component = %error.context.component,
                    "Low severity error: {}",
                    error.error
                );
            }
        }
    }
}

/// Result type with enhanced error information
pub type EnhancedResult<T> = std::result::Result<T, EnhancedError>;

/// Macro for creating enhanced errors with context
#[macro_export]
macro_rules! enhanced_error {
    ($error:expr, $component:expr, $operation:expr) => {
        $crate::error::ErrorHandler::new($component, $operation).enhance_error($error)
    };
    ($error:expr, $component:expr, $operation:expr, user = $user:expr) => {
        $crate::error::ErrorHandler::new($component, $operation)
            .with_user($user)
            .enhance_error($error)
    };
    ($error:expr, $component:expr, $operation:expr, session = $session:expr) => {
        $crate::error::ErrorHandler::new($component, $operation)
            .with_session($session)
            .enhance_error($error)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recoverability() {
        let recoverable_error = LedgerError::Network {
            operation: "api_call".to_string(),
            endpoint: "https://api.example.com".to_string(),
            reason: "connection timeout".to_string(),
            status_code: None,
            retry_count: 1,
        };
        assert!(recoverable_error.is_recoverable());

        let non_recoverable_error = LedgerError::Authentication {
            reason: "invalid credentials".to_string(),
            user_id: Some("user123".to_string()),
            attempted_operation: "login".to_string(),
            lockout_time: None,
        };
        assert!(!non_recoverable_error.is_recoverable());
    }

    #[test]
    fn test_error_severity() {
        let critical_error = LedgerError::DataCorruption {
            location: "ledger_data".to_string(),
            details: "checksum mismatch".to_string(),
            corruption_type: CorruptionType::ChecksumMismatch,
            checksum_expected: Some("abc123".to_string()),
            checksum_actual: Some("def456".to_string()),
            recovery_possible: false,
        };
        assert_eq!(critical_error.severity(), ErrorSeverity::Critical);

        let medium_error = LedgerError::Validation {
            field: "dataset_name".to_string(),
            reason: "contains invalid characters".to_string(),
            provided_value: "test<script>".to_string(),
            constraints: vec!["alphanumeric only".to_string()],
            suggestions: vec!["test_script".to_string()],
        };
        assert_eq!(medium_error.severity(), ErrorSeverity::Medium);
    }

    #[test]
    fn test_user_message() {
        let error = LedgerError::NotFound {
            resource_type: "dataset".to_string(),
            resource_name: "my_data.csv".to_string(),
            suggestions: vec![
                "my_data_v1.csv".to_string(),
                "my_data_backup.csv".to_string(),
            ],
        };
        let message = error.user_message();
        assert!(message.contains("Could not find dataset 'my_data.csv'"));
        assert!(message.contains("Did you mean: my_data_v1.csv, my_data_backup.csv?"));
    }

    #[test]
    fn test_error_handler() {
        let handler = ErrorHandler::new("ledger", "create_entry")
            .with_user("user123")
            .with_metadata("dataset", "test.csv");

        let error = LedgerError::Validation {
            field: "name".to_string(),
            reason: "too long".to_string(),
            provided_value: "very_long_name".to_string(),
            constraints: vec!["max 50 characters".to_string()],
            suggestions: vec!["shorten the name".to_string()],
        };

        let enhanced = handler.enhance_error(error);
        assert_eq!(enhanced.context.component, "ledger");
        assert_eq!(enhanced.context.operation, "create_entry");
        assert_eq!(enhanced.context.user_id, Some("user123".to_string()));
        assert!(enhanced.context.metadata.contains_key("dataset"));
        assert!(enhanced.user_message.is_some());
    }

    #[test]
    fn test_retry_delay() {
        let rate_limit_error = LedgerError::RateLimit {
            limit: 100,
            window: "1 minute".to_string(),
            current_count: 101,
            reset_time: Utc::now() + chrono::Duration::seconds(30),
        };
        let delay = rate_limit_error.retry_delay_ms();
        assert!(delay.is_some());
        assert!(delay.unwrap() > 0);

        let network_error = LedgerError::Network {
            operation: "get".to_string(),
            endpoint: "api.example.com".to_string(),
            reason: "timeout".to_string(),
            status_code: None,
            retry_count: 2,
        };
        assert_eq!(network_error.retry_delay_ms(), Some(400)); // 2^2 * 100ms
    }
}
