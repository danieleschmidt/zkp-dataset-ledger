//! Security utilities and access control for ZKP Dataset Ledger

use crate::{LedgerError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Security configuration for the ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable access control
    pub enable_access_control: bool,
    /// Enable audit logging
    pub enable_audit_logging: bool,
    /// Maximum file size allowed for datasets (bytes)
    pub max_file_size_bytes: u64,
    /// Allowed file extensions
    pub allowed_extensions: Vec<String>,
    /// Rate limiting: max operations per minute
    pub rate_limit_per_minute: u32,
    /// Enable content scanning
    pub enable_content_scanning: bool,
    /// Trusted certificate authorities for proof verification
    pub trusted_cas: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_access_control: true,
            enable_audit_logging: true,
            max_file_size_bytes: 1_000_000_000, // 1GB
            allowed_extensions: vec![
                "csv".to_string(),
                "json".to_string(),
                "parquet".to_string(),
                "arrow".to_string(),
            ],
            rate_limit_per_minute: 100,
            enable_content_scanning: true,
            trusted_cas: vec![],
        }
    }
}

/// User permissions and roles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Permission {
    /// Read ledger entries and proofs
    Read,
    /// Create new ledger entries
    Write,
    /// Modify existing entries (dangerous)
    Modify,
    /// Delete entries (very dangerous)
    Delete,
    /// Administrative functions
    Admin,
    /// Export ledger data
    Export,
    /// Import ledger data
    Import,
    /// Verify proofs
    Verify,
}

/// User role with associated permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub permissions: HashSet<Permission>,
    pub description: String,
}

impl Role {
    /// Create a read-only role
    pub fn read_only() -> Self {
        let mut permissions = HashSet::new();
        permissions.insert(Permission::Read);
        permissions.insert(Permission::Verify);
        
        Self {
            name: "ReadOnly".to_string(),
            permissions,
            description: "Can read ledger data and verify proofs".to_string(),
        }
    }

    /// Create a data scientist role
    pub fn data_scientist() -> Self {
        let mut permissions = HashSet::new();
        permissions.insert(Permission::Read);
        permissions.insert(Permission::Write);
        permissions.insert(Permission::Verify);
        permissions.insert(Permission::Export);
        
        Self {
            name: "DataScientist".to_string(),
            permissions,
            description: "Can create proofs and export data".to_string(),
        }
    }

    /// Create an admin role
    pub fn admin() -> Self {
        let mut permissions = HashSet::new();
        permissions.insert(Permission::Read);
        permissions.insert(Permission::Write);
        permissions.insert(Permission::Modify);
        permissions.insert(Permission::Delete);
        permissions.insert(Permission::Admin);
        permissions.insert(Permission::Export);
        permissions.insert(Permission::Import);
        permissions.insert(Permission::Verify);
        
        Self {
            name: "Admin".to_string(),
            permissions,
            description: "Full access to all functions".to_string(),
        }
    }
}

/// User identity and authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub role: Role,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub api_key_hash: Option<String>,
}

impl User {
    pub fn new(username: &str, email: &str, role: Role) -> Self {
        Self {
            id: Uuid::new_v4(),
            username: username.to_string(),
            email: email.to_string(),
            role,
            created_at: Utc::now(),
            last_login: None,
            is_active: true,
            api_key_hash: None,
        }
    }

    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.is_active && self.role.permissions.contains(permission)
    }
}

/// Audit log entry for tracking operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub operation: String,
    pub resource: String,
    pub timestamp: DateTime<Utc>,
    pub ip_address: Option<String>,
    pub success: bool,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl AuditLogEntry {
    pub fn new(
        user_id: Option<Uuid>,
        operation: &str,
        resource: &str,
        success: bool,
        error_message: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            operation: operation.to_string(),
            resource: resource.to_string(),
            timestamp: Utc::now(),
            ip_address: None,
            success,
            error_message,
            metadata: HashMap::new(),
        }
    }
}

/// Security validator for input sanitization and validation
pub struct SecurityValidator {
    config: SecurityConfig,
    audit_log: Vec<AuditLogEntry>,
    rate_limiter: HashMap<String, Vec<DateTime<Utc>>>,
}

impl SecurityValidator {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            config,
            audit_log: Vec::new(),
            rate_limiter: HashMap::new(),
        }
    }

    /// Validate file path for security issues
    pub fn validate_file_path(&self, path: &str) -> Result<()> {
        // Check for path traversal attacks
        if path.contains("..") || path.contains("~") {
            return Err(LedgerError::InvalidInput(
                "Path traversal detected in file path".to_string()
            ));
        }

        // Check file extension
        if let Some(extension) = std::path::Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
        {
            let ext_lower = extension.to_lowercase();
            if !self.config.allowed_extensions.contains(&ext_lower) {
                return Err(LedgerError::InvalidInput(
                    format!("File extension '{}' not allowed", extension)
                ));
            }
        } else {
            return Err(LedgerError::InvalidInput(
                "File must have a valid extension".to_string()
            ));
        }

        // Check if file exists and is readable
        if !std::path::Path::new(path).exists() {
            return Err(LedgerError::InvalidInput(
                "File does not exist".to_string()
            ));
        }

        // Check file size
        match std::fs::metadata(path) {
            Ok(metadata) => {
                if metadata.len() > self.config.max_file_size_bytes {
                    return Err(LedgerError::InvalidInput(
                        format!("File size {} exceeds maximum allowed size of {} bytes",
                                metadata.len(), self.config.max_file_size_bytes)
                    ));
                }
            }
            Err(e) => {
                return Err(LedgerError::IoError(
                    format!("Failed to read file metadata: {}", e)
                ));
            }
        }

        Ok(())
    }

    /// Sanitize dataset name to prevent injection attacks
    pub fn sanitize_dataset_name(&self, name: &str) -> Result<String> {
        // Check length
        if name.len() > 255 {
            return Err(LedgerError::InvalidInput(
                "Dataset name too long (max 255 characters)".to_string()
            ));
        }

        // Allow only alphanumeric, dashes, underscores, and dots
        let sanitized: String = name
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
            .collect();

        if sanitized.is_empty() {
            return Err(LedgerError::InvalidInput(
                "Dataset name cannot be empty after sanitization".to_string()
            ));
        }

        Ok(sanitized)
    }

    /// Check rate limiting for operations
    pub fn check_rate_limit(&mut self, client_id: &str) -> Result<()> {
        let now = Utc::now();
        let one_minute_ago = now - chrono::Duration::minutes(1);

        // Clean old entries and count recent ones
        let entries = self.rate_limiter.entry(client_id.to_string()).or_insert_with(Vec::new);
        entries.retain(|&timestamp| timestamp > one_minute_ago);

        if entries.len() >= self.config.rate_limit_per_minute as usize {
            return Err(LedgerError::InvalidInput(
                "Rate limit exceeded. Too many requests.".to_string()
            ));
        }

        entries.push(now);
        Ok(())
    }

    /// Scan content for sensitive data patterns
    pub fn scan_sensitive_content(&self, content: &[u8]) -> Result<Vec<String>> {
        let mut findings = Vec::new();

        if !self.config.enable_content_scanning {
            return Ok(findings);
        }

        let content_str = String::from_utf8_lossy(content);
        
        // Common patterns for sensitive data
        let patterns = vec![
            (r"\b\d{3}-\d{2}-\d{4}\b", "Potential SSN"),
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "Potential Credit Card"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email Address"),
            (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "IP Address"),
            (r"password|pwd|passwd|secret|token", "Potential Secret"),
        ];

        for (pattern, description) in patterns {
            if regex::Regex::new(pattern)
                .map_err(|e| LedgerError::InvalidInput(format!("Regex error: {}", e)))?
                .is_match(&content_str.to_lowercase())
            {
                findings.push(format!("{}: pattern matched", description));
            }
        }

        Ok(findings)
    }

    /// Log audit event
    pub fn log_audit_event(&mut self, entry: AuditLogEntry) {
        if self.config.enable_audit_logging {
            tracing::info!(
                user_id = ?entry.user_id,
                operation = %entry.operation,
                resource = %entry.resource,
                success = entry.success,
                error = ?entry.error_message,
                "Audit log entry"
            );
            self.audit_log.push(entry);
        }
    }

    /// Get audit log entries
    pub fn get_audit_log(&self) -> &[AuditLogEntry] {
        &self.audit_log
    }

    /// Clear old audit log entries
    pub fn cleanup_audit_log(&mut self, older_than: DateTime<Utc>) {
        self.audit_log.retain(|entry| entry.timestamp > older_than);
    }
}

/// Input validation utilities
pub struct InputValidator;

impl InputValidator {
    /// Validate proof type string
    pub fn validate_proof_type(proof_type: &str) -> Result<()> {
        let valid_types = vec![
            "integrity", "row-count", "schema", "statistics",
            "transformation", "data-split", "custom"
        ];
        
        if !valid_types.contains(&proof_type) {
            return Err(LedgerError::InvalidInput(
                format!("Invalid proof type '{}'. Valid types: {}", 
                        proof_type, valid_types.join(", "))
            ));
        }
        
        Ok(())
    }

    /// Validate hash algorithm string
    pub fn validate_hash_algorithm(algorithm: &str) -> Result<()> {
        let valid_algorithms = vec!["sha3-256", "blake3"];
        
        if !valid_algorithms.contains(&algorithm) {
            return Err(LedgerError::InvalidInput(
                format!("Invalid hash algorithm '{}'. Valid algorithms: {}", 
                        algorithm, valid_algorithms.join(", "))
            ));
        }
        
        Ok(())
    }

    /// Validate ratio for data splits
    pub fn validate_split_ratio(ratio: f64) -> Result<()> {
        if ratio <= 0.0 || ratio >= 1.0 {
            return Err(LedgerError::InvalidInput(
                "Split ratio must be between 0.0 and 1.0 (exclusive)".to_string()
            ));
        }
        Ok(())
    }

    /// Validate operation parameters
    pub fn validate_operation_params(params: &HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            if key.is_empty() || value.is_empty() {
                return Err(LedgerError::InvalidInput(
                    "Parameter keys and values cannot be empty".to_string()
                ));
            }
            
            if key.len() > 100 || value.len() > 1000 {
                return Err(LedgerError::InvalidInput(
                    "Parameter key/value too long".to_string()
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert!(config.enable_access_control);
        assert!(config.enable_audit_logging);
        assert_eq!(config.max_file_size_bytes, 1_000_000_000);
        assert!(config.allowed_extensions.contains(&"csv".to_string()));
    }

    #[test]
    fn test_role_permissions() {
        let read_only = Role::read_only();
        assert!(read_only.permissions.contains(&Permission::Read));
        assert!(!read_only.permissions.contains(&Permission::Write));

        let admin = Role::admin();
        assert!(admin.permissions.contains(&Permission::Admin));
        assert!(admin.permissions.contains(&Permission::Delete));
    }

    #[test]
    fn test_user_permissions() {
        let user = User::new("test_user", "test@example.com", Role::read_only());
        assert!(user.has_permission(&Permission::Read));
        assert!(!user.has_permission(&Permission::Write));
    }

    #[test]
    fn test_path_validation() {
        let validator = SecurityValidator::new(SecurityConfig::default());
        
        // Should reject path traversal
        assert!(validator.validate_file_path("../../../etc/passwd").is_err());
        assert!(validator.validate_file_path("~/secret.txt").is_err());
        
        // Should reject invalid extensions
        assert!(validator.validate_file_path("test.exe").is_err());
    }

    #[test]
    fn test_dataset_name_sanitization() {
        let validator = SecurityValidator::new(SecurityConfig::default());
        
        let sanitized = validator.sanitize_dataset_name("test-dataset_v1.0").unwrap();
        assert_eq!(sanitized, "test-dataset_v1.0");
        
        let sanitized = validator.sanitize_dataset_name("test<script>alert()</script>").unwrap();
        assert_eq!(sanitized, "testscriptalertscript");
    }

    #[test]
    fn test_rate_limiting() {
        let mut validator = SecurityValidator::new(SecurityConfig::default());
        
        // Should allow within limits
        for _ in 0..50 {
            assert!(validator.check_rate_limit("client1").is_ok());
        }
        
        // Should reject when over limit
        for _ in 50..120 {
            validator.check_rate_limit("client1").ok();
        }
        assert!(validator.check_rate_limit("client1").is_err());
    }

    #[test]
    fn test_input_validation() {
        assert!(InputValidator::validate_proof_type("integrity").is_ok());
        assert!(InputValidator::validate_proof_type("invalid").is_err());
        
        assert!(InputValidator::validate_hash_algorithm("sha3-256").is_ok());
        assert!(InputValidator::validate_hash_algorithm("md5").is_err());
        
        assert!(InputValidator::validate_split_ratio(0.8).is_ok());
        assert!(InputValidator::validate_split_ratio(1.5).is_err());
    }
}