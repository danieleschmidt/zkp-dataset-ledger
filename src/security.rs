//! Security validation and access control for ZKP Dataset Ledger.

use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Security configuration for the ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_rbac: bool,
    pub default_permissions: Vec<Permission>,
    pub session_timeout_minutes: u32,
    pub max_failed_attempts: u32,
    pub require_mfa: bool,
    pub allowed_operations: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_rbac: true,
            default_permissions: vec![Permission::Read],
            session_timeout_minutes: 60,
            max_failed_attempts: 3,
            require_mfa: false,
            allowed_operations: vec![
                "notarize".to_string(),
                "verify".to_string(),
                "query".to_string(),
            ],
        }
    }
}

/// User permissions in the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Permission {
    Read,
    Write,
    Admin,
    Audit,
    Config,
}

/// User roles with associated permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    Viewer,
    DataScientist,
    Auditor,
    Admin,
    SystemAdmin,
}

impl Role {
    pub fn permissions(&self) -> Vec<Permission> {
        match self {
            Role::Viewer => vec![Permission::Read],
            Role::DataScientist => vec![Permission::Read, Permission::Write],
            Role::Auditor => vec![Permission::Read, Permission::Audit],
            Role::Admin => vec![Permission::Read, Permission::Write, Permission::Admin],
            Role::SystemAdmin => vec![
                Permission::Read,
                Permission::Write,
                Permission::Admin,
                Permission::Audit,
                Permission::Config,
            ],
        }
    }
}

/// User in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub roles: Vec<Role>,
    pub permissions: Vec<Permission>,
    pub active: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_login: Option<chrono::DateTime<chrono::Utc>>,
}

impl User {
    pub fn new(id: String, username: String) -> Self {
        Self {
            id,
            username,
            roles: vec![Role::Viewer],
            permissions: vec![Permission::Read],
            active: true,
            created_at: chrono::Utc::now(),
            last_login: None,
        }
    }

    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
            || self
                .roles
                .iter()
                .any(|role| role.permissions().contains(permission))
    }

    pub fn can_perform_operation(&self, operation: &str) -> bool {
        match operation {
            "notarize" | "transform" | "split" => self.has_permission(&Permission::Write),
            "verify" | "history" | "status" => self.has_permission(&Permission::Read),
            "audit" => self.has_permission(&Permission::Audit),
            "config" => self.has_permission(&Permission::Config),
            _ => false,
        }
    }
}

/// Security validator for operations
pub struct SecurityValidator {
    config: SecurityConfig,
    users: HashMap<String, User>,
    failed_attempts: HashMap<String, u32>,
}

impl SecurityValidator {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            config,
            users: HashMap::new(),
            failed_attempts: HashMap::new(),
        }
    }

    pub fn add_user(&mut self, user: User) {
        self.users.insert(user.id.clone(), user);
    }

    pub fn authenticate(&mut self, user_id: &str, _password: &str) -> Result<&User> {
        // Simple authentication - in production this would hash passwords, etc.
        if let Some(user) = self.users.get(user_id) {
            if user.active {
                self.failed_attempts.remove(user_id);
                return Ok(user);
            }
        }

        let attempts = self.failed_attempts.get(user_id).unwrap_or(&0) + 1;
        self.failed_attempts.insert(user_id.to_string(), attempts);

        if attempts >= self.config.max_failed_attempts {
            return Err(LedgerError::security(
                "Account locked due to too many failed attempts",
            ));
        }

        Err(LedgerError::security("Authentication failed"))
    }

    pub fn authorize(&self, user: &User, operation: &str) -> Result<()> {
        if !self.config.enable_rbac {
            return Ok(());
        }

        if !user.active {
            return Err(LedgerError::security("User account is disabled"));
        }

        if !self
            .config
            .allowed_operations
            .contains(&operation.to_string())
        {
            return Err(LedgerError::security(&format!(
                "Operation '{}' is not allowed",
                operation
            )));
        }

        if !user.can_perform_operation(operation) {
            return Err(LedgerError::security(&format!(
                "User '{}' does not have permission for operation '{}'",
                user.username, operation
            )));
        }

        Ok(())
    }

    pub fn validate_input(&self, input: &str) -> Result<()> {
        // Basic input validation to prevent injection attacks
        if input.contains("';") || input.contains("--") || input.contains("/*") {
            return Err(LedgerError::security(
                "Potentially malicious input detected",
            ));
        }

        if input.len() > 10000 {
            return Err(LedgerError::security("Input too large"));
        }

        Ok(())
    }

    pub fn log_access(&self, user_id: &str, operation: &str, success: bool) {
        tracing::info!(
            user_id = user_id,
            operation = operation,
            success = success,
            "Access log"
        );
    }
}

impl Default for SecurityValidator {
    fn default() -> Self {
        Self::new(SecurityConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_permissions() {
        let user = User::new("1".to_string(), "alice".to_string());
        assert!(user.has_permission(&Permission::Read));
        assert!(!user.has_permission(&Permission::Write));
    }

    #[test]
    fn test_role_permissions() {
        let admin_role = Role::Admin;
        let permissions = admin_role.permissions();
        assert!(permissions.contains(&Permission::Read));
        assert!(permissions.contains(&Permission::Write));
        assert!(permissions.contains(&Permission::Admin));
    }

    #[test]
    fn test_security_validator() {
        let mut validator = SecurityValidator::default();
        let user = User::new("test".to_string(), "testuser".to_string());
        validator.add_user(user);

        assert!(validator.authenticate("test", "password").is_ok());
        assert!(validator.authenticate("invalid", "password").is_err());
    }

    #[test]
    fn test_input_validation() {
        let validator = SecurityValidator::default();
        assert!(validator.validate_input("normal input").is_ok());
        assert!(validator.validate_input("malicious'; DROP TABLE").is_err());
    }
}
