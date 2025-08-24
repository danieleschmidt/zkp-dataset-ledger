//! Advanced Error Handling System for ZKP Dataset Ledger
//!
//! Provides comprehensive error categorization, context tracking,
//! recovery strategies, and diagnostic information.

use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Error severity levels for prioritization and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Critical errors that require immediate attention
    Critical,
    /// High priority errors affecting functionality
    High,
    /// Medium priority errors with workarounds
    Medium,
    /// Low priority errors and warnings
    Low,
    /// Information-only events
    Info,
}

/// Error categories for organized handling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Input validation failures
    Validation,
    /// Cryptographic operation failures
    Cryptographic,
    /// Storage and I/O errors
    Storage,
    /// Network and communication errors
    Network,
    /// Authentication and authorization errors
    Security,
    /// Configuration and setup errors
    Configuration,
    /// System resource errors
    Resource,
    /// Business logic errors
    Business,
    /// External dependency errors
    External,
    /// Unknown or unclassified errors
    Unknown,
}

/// Detailed error context with diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error ID for tracking and correlation
    pub error_id: String,
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Timestamp when error occurred
    pub timestamp: u64,
    /// Operation that caused the error
    pub operation: String,
    /// Component or module where error occurred
    pub component: String,
    /// User-friendly error message
    pub user_message: String,
    /// Technical error details
    pub technical_details: String,
    /// Possible recovery actions
    pub recovery_suggestions: Vec<String>,
    /// Additional context data
    pub context_data: HashMap<String, String>,
    /// Stack trace or call chain
    pub stack_trace: Option<String>,
    /// Related error IDs for correlation
    pub related_errors: Vec<String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(
        severity: ErrorSeverity,
        category: ErrorCategory,
        operation: &str,
        component: &str,
        user_message: &str,
        technical_details: &str,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        Self {
            error_id: uuid::Uuid::new_v4().to_string(),
            severity,
            category,
            timestamp,
            operation: operation.to_string(),
            component: component.to_string(),
            user_message: user_message.to_string(),
            technical_details: technical_details.to_string(),
            recovery_suggestions: Vec::new(),
            context_data: HashMap::new(),
            stack_trace: None,
            related_errors: Vec::new(),
        }
    }

    /// Add recovery suggestion
    pub fn with_recovery_suggestion(mut self, suggestion: &str) -> Self {
        self.recovery_suggestions.push(suggestion.to_string());
        self
    }

    /// Add context data
    pub fn with_context_data(mut self, key: &str, value: &str) -> Self {
        self.context_data.insert(key.to_string(), value.to_string());
        self
    }

    /// Add related error ID
    pub fn with_related_error(mut self, error_id: &str) -> Self {
        self.related_errors.push(error_id.to_string());
        self
    }

    /// Set stack trace
    pub fn with_stack_trace(mut self, stack_trace: &str) -> Self {
        self.stack_trace = Some(stack_trace.to_string());
        self
    }

    /// Check if error is critical
    pub fn is_critical(&self) -> bool {
        matches!(self.severity, ErrorSeverity::Critical)
    }

    /// Check if error requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(self.severity, ErrorSeverity::Critical | ErrorSeverity::High)
    }

    /// Generate user-friendly error report
    pub fn user_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("❌ {}\n", self.user_message));
        report.push_str(&format!("🔍 Error ID: {}\n", self.error_id));

        if !self.recovery_suggestions.is_empty() {
            report.push_str("\n💡 Suggested Actions:\n");
            for (i, suggestion) in self.recovery_suggestions.iter().enumerate() {
                report.push_str(&format!("   {}. {}\n", i + 1, suggestion));
            }
        }

        report
    }

    /// Generate technical error report
    pub fn technical_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("Error ID: {}\n", self.error_id));
        report.push_str(&format!("Severity: {:?}\n", self.severity));
        report.push_str(&format!("Category: {:?}\n", self.category));
        report.push_str(&format!("Timestamp: {}\n", self.timestamp));
        report.push_str(&format!("Operation: {}\n", self.operation));
        report.push_str(&format!("Component: {}\n", self.component));
        report.push_str(&format!("Technical Details: {}\n", self.technical_details));

        if !self.context_data.is_empty() {
            report.push_str("\nContext Data:\n");
            for (key, value) in &self.context_data {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        if let Some(stack_trace) = &self.stack_trace {
            report.push_str(&format!("\nStack Trace:\n{}\n", stack_trace));
        }

        if !self.related_errors.is_empty() {
            report.push_str(&format!(
                "\nRelated Errors: {}\n",
                self.related_errors.join(", ")
            ));
        }

        report
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} - {}",
            self.error_id[..8].to_uppercase(),
            self.component,
            self.user_message
        )
    }
}

/// Enhanced error wrapper with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualError {
    /// Base error
    pub base_error: String,
    /// Enhanced error context
    pub context: ErrorContext,
}

impl ContextualError {
    /// Create new contextual error from base error
    pub fn from_base_error(base_error: &LedgerError, operation: &str, component: &str) -> Self {
        let (severity, category, user_message, technical_details) = match base_error {
            LedgerError::NotFound(message) => (
                ErrorSeverity::Medium,
                ErrorCategory::Validation,
                "Resource not found".to_string(),
                message.clone(),
            ),
            LedgerError::ValidationError(message) => (
                ErrorSeverity::High,
                ErrorCategory::Validation,
                "Input validation failed".to_string(),
                message.clone(),
            ),
            LedgerError::SecurityViolation(message) => (
                ErrorSeverity::Critical,
                ErrorCategory::Security,
                "Security violation detected".to_string(),
                message.clone(),
            ),
            LedgerError::DataIntegrityError(message) => (
                ErrorSeverity::Critical,
                ErrorCategory::Cryptographic,
                "Data integrity compromised".to_string(),
                message.clone(),
            ),
            LedgerError::Io(err) => (
                ErrorSeverity::High,
                ErrorCategory::Storage,
                "I/O operation failed".to_string(),
                err.to_string(),
            ),
            _ => (
                ErrorSeverity::Medium,
                ErrorCategory::Unknown,
                "Unknown error occurred".to_string(),
                format!("{:?}", base_error),
            ),
        };

        let context = ErrorContext::new(
            severity,
            category,
            operation,
            component,
            &user_message,
            &technical_details,
        );

        Self {
            base_error: format!("{:?}", base_error),
            context,
        }
    }

    /// Add recovery suggestions based on error type
    pub fn with_auto_recovery_suggestions(mut self) -> Self {
        match self.context.category {
            ErrorCategory::Validation => {
                self.context = self
                    .context
                    .with_recovery_suggestion("Check input format and try again")
                    .with_recovery_suggestion("Refer to documentation for expected format");
            }
            ErrorCategory::Storage => {
                self.context = self
                    .context
                    .with_recovery_suggestion("Check file permissions and disk space")
                    .with_recovery_suggestion("Verify storage path exists")
                    .with_recovery_suggestion("Try alternative storage location");
            }
            ErrorCategory::Security => {
                self.context = self
                    .context
                    .with_recovery_suggestion("Review security configuration")
                    .with_recovery_suggestion("Check authentication credentials")
                    .with_recovery_suggestion("Contact system administrator");
            }
            ErrorCategory::Cryptographic => {
                self.context = self
                    .context
                    .with_recovery_suggestion("Verify cryptographic parameters")
                    .with_recovery_suggestion("Check data hasn't been tampered with")
                    .with_recovery_suggestion("Regenerate proof if possible");
            }
            _ => {
                self.context = self
                    .context
                    .with_recovery_suggestion("Try the operation again")
                    .with_recovery_suggestion("Contact support if problem persists");
            }
        }
        self
    }
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.context)
    }
}

/// Error handler with logging and recovery capabilities
pub struct ErrorHandler {
    /// Error log for tracking and analysis
    error_log: Vec<ErrorContext>,
    /// Maximum number of errors to keep in memory
    max_log_size: usize,
    /// Error statistics
    error_stats: HashMap<ErrorCategory, u64>,
}

impl ErrorHandler {
    /// Create new error handler
    pub fn new(max_log_size: usize) -> Self {
        Self {
            error_log: Vec::new(),
            max_log_size,
            error_stats: HashMap::new(),
        }
    }

    /// Handle error with enhanced context
    pub fn handle_error(&mut self, contextual_error: ContextualError) -> Result<()> {
        let context = &contextual_error.context;

        // Update statistics
        *self
            .error_stats
            .entry(context.category.clone())
            .or_insert(0) += 1;

        // Log error based on severity
        match context.severity {
            ErrorSeverity::Critical => {
                log::error!("CRITICAL: {}", context.technical_report());
                // In production, this could trigger alerts
            }
            ErrorSeverity::High => {
                log::error!("HIGH: {}", context);
            }
            ErrorSeverity::Medium => {
                log::warn!("MEDIUM: {}", context);
            }
            ErrorSeverity::Low => {
                log::info!("LOW: {}", context);
            }
            ErrorSeverity::Info => {
                log::debug!("INFO: {}", context);
            }
        }

        // Add to error log
        self.error_log.push(context.clone());

        // Trim log if needed
        if self.error_log.len() > self.max_log_size {
            self.error_log.remove(0);
        }

        Ok(())
    }

    /// Get error statistics
    pub fn error_statistics(&self) -> &HashMap<ErrorCategory, u64> {
        &self.error_stats
    }

    /// Get recent errors
    pub fn recent_errors(&self, count: usize) -> Vec<&ErrorContext> {
        let start = if self.error_log.len() > count {
            self.error_log.len() - count
        } else {
            0
        };

        self.error_log[start..].iter().collect()
    }

    /// Get errors by category
    pub fn errors_by_category(&self, category: ErrorCategory) -> Vec<&ErrorContext> {
        self.error_log
            .iter()
            .filter(|ctx| ctx.category == category)
            .collect()
    }

    /// Get critical errors
    pub fn critical_errors(&self) -> Vec<&ErrorContext> {
        self.error_log
            .iter()
            .filter(|ctx| ctx.is_critical())
            .collect()
    }

    /// Generate error report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Error Analysis Report\n\n");

        report.push_str("## Statistics\n");
        for (category, count) in &self.error_stats {
            report.push_str(&format!("- {:?}: {}\n", category, count));
        }

        report.push_str(&format!(
            "\n## Recent Errors ({} total)\n",
            self.error_log.len()
        ));
        for error in self.recent_errors(10) {
            report.push_str(&format!("- {}\n", error));
        }

        let critical_count = self.critical_errors().len();
        if critical_count > 0 {
            report.push_str(&format!(
                "\n⚠️  {} Critical Errors Detected\n",
                critical_count
            ));
        }

        report
    }

    /// Clear error log
    pub fn clear_log(&mut self) {
        self.error_log.clear();
        self.error_stats.clear();
    }
}

impl Default for ErrorHandler {
    fn default() -> Self {
        Self::new(1000) // Keep last 1000 errors
    }
}

/// Helper functions for common error patterns
/// Create validation error with context
pub fn validation_error(
    operation: &str,
    component: &str,
    field: &str,
    message: &str,
) -> ContextualError {
    let base_error = LedgerError::validation_error(message);
    ContextualError::from_base_error(&base_error, operation, component)
        .with_auto_recovery_suggestions()
        .context
        .with_context_data("field", field)
        .into()
}

impl From<ErrorContext> for ContextualError {
    fn from(context: ErrorContext) -> Self {
        Self {
            base_error: "Custom".to_string(),
            context,
        }
    }
}

/// Create storage error with context
pub fn storage_error(
    operation: &str,
    component: &str,
    path: &str,
    message: &str,
) -> ContextualError {
    let base_error = LedgerError::data_integrity_error(message);
    ContextualError::from_base_error(&base_error, operation, component)
        .with_auto_recovery_suggestions()
        .context
        .with_context_data("path", path)
        .into()
}

/// Create security error with context
pub fn security_error(
    operation: &str,
    component: &str,
    violation_type: &str,
    message: &str,
) -> ContextualError {
    let base_error = LedgerError::security_violation(message);
    ContextualError::from_base_error(&base_error, operation, component)
        .with_auto_recovery_suggestions()
        .context
        .with_context_data("violation_type", violation_type)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new(
            ErrorSeverity::High,
            ErrorCategory::Validation,
            "test_operation",
            "test_component",
            "Test error",
            "Technical details",
        );

        assert_eq!(context.severity, ErrorSeverity::High);
        assert_eq!(context.category, ErrorCategory::Validation);
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.user_message, "Test error");
    }

    #[test]
    fn test_error_handler() {
        let mut handler = ErrorHandler::new(10);

        let base_error = LedgerError::validation_error("Test error");
        let contextual_error =
            ContextualError::from_base_error(&base_error, "test_op", "test_component");

        assert!(handler.handle_error(contextual_error).is_ok());
        assert_eq!(handler.error_log.len(), 1);
        assert_eq!(
            handler.error_statistics().get(&ErrorCategory::Validation),
            Some(&1)
        );
    }

    #[test]
    fn test_contextual_error_display() {
        let base_error = LedgerError::validation_error("Test validation error");
        let contextual_error =
            ContextualError::from_base_error(&base_error, "validation", "input_processor");

        let display = format!("{}", contextual_error);
        assert!(display.contains("input_processor"));
        assert!(display.contains("Input validation failed"));
    }

    #[test]
    fn test_recovery_suggestions() {
        let contextual_error =
            validation_error("test_op", "test_component", "test_field", "Invalid input");

        assert!(!contextual_error.context.recovery_suggestions.is_empty());
        assert!(contextual_error
            .context
            .recovery_suggestions
            .iter()
            .any(|s| s.contains("input format")));
    }
}
