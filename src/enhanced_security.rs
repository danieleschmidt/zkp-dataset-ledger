//! Enhanced Security Module for ZKP Dataset Ledger
//! 
//! This module provides advanced security features including input validation,
//! cryptographic security analysis, and security monitoring.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use crate::{LedgerError, Result};

/// Security configuration for the ledger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_rate_limiting: bool,
    pub max_requests_per_minute: u32,
    pub enable_input_validation: bool,
    pub max_input_size_bytes: usize,
    pub enable_audit_logging: bool,
    pub enable_intrusion_detection: bool,
    pub security_level: SecurityLevel,
    pub allowed_ip_ranges: Vec<String>,
    pub blocked_ips: Vec<IpAddr>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_rate_limiting: true,
            max_requests_per_minute: 60,
            enable_input_validation: true,
            max_input_size_bytes: 100 * 1024 * 1024, // 100 MB
            enable_audit_logging: true,
            enable_intrusion_detection: true,
            security_level: SecurityLevel::Standard,
            allowed_ip_ranges: Vec::new(),
            blocked_ips: Vec::new(),
        }
    }
}

/// Security levels for different environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Development,
    Testing,
    Standard,
    High,
    Maximum,
}

/// Security validator for inputs and operations
#[derive(Debug)]
pub struct SecurityValidator {
    config: SecurityConfig,
    rate_limiter: RateLimiter,
    audit_logger: AuditLogger,
    intrusion_detector: IntrusionDetector,
}

/// Rate limiting implementation
#[derive(Debug)]
pub struct RateLimiter {
    requests: HashMap<String, Vec<Instant>>,
    max_requests: u32,
    window_duration: Duration,
}

/// Audit logging for security events
#[derive(Debug)]
pub struct AuditLogger {
    events: Vec<SecurityEvent>,
    max_events: usize,
}

/// Intrusion detection system
#[derive(Debug)]
pub struct IntrusionDetector {
    suspicious_patterns: Vec<SuspiciousPattern>,
    threat_scores: HashMap<String, f64>,
    detection_rules: Vec<DetectionRule>,
}

/// Security event for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub event_id: String,
    pub event_type: SecurityEventType,
    pub timestamp: DateTime<Utc>,
    pub source_ip: Option<IpAddr>,
    pub user_id: Option<String>,
    pub operation: String,
    pub result: SecurityResult,
    pub details: HashMap<String, String>,
    pub risk_score: f64,
}

/// Types of security events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    Authentication,
    Authorization,
    InputValidation,
    RateLimitExceeded,
    SuspiciousActivity,
    IntrusionAttempt,
    DataAccess,
    ConfigurationChange,
    ErrorCondition,
}

/// Result of security validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityResult {
    Allowed,
    Denied { reason: String },
    Blocked { reason: String },
    Suspicious { reason: String, confidence: f64 },
}

/// Suspicious activity patterns
#[derive(Debug, Clone)]
pub struct SuspiciousPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub threshold: f64,
    pub time_window: Duration,
    pub severity: ThreatSeverity,
}

/// Types of suspicious patterns
#[derive(Debug, Clone)]
pub enum PatternType {
    HighFrequencyRequests,
    LargeDataAccess,
    UnusualTiming,
    FailedAuthentication,
    InvalidInputs,
    UnknownUserAgent,
    GeolocationAnomaly,
    DataExfiltration,
}

/// Threat severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Detection rules for intrusion detection
#[derive(Debug, Clone)]
pub struct DetectionRule {
    pub rule_id: String,
    pub name: String,
    pub condition: String,
    pub action: DetectionAction,
    pub enabled: bool,
}

/// Actions to take when detection rules are triggered
#[derive(Debug, Clone)]
pub enum DetectionAction {
    Log,
    Alert,
    Block,
    Quarantine,
}

/// Input validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub sanitized_input: Option<String>,
}

/// Input validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub error_type: ValidationErrorType,
    pub message: String,
    pub severity: ThreatSeverity,
}

/// Types of validation errors
#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    TooLarge,
    InvalidFormat,
    MaliciousContent,
    Injection,
    PathTraversal,
    InvalidCharacters,
    ExcessiveLength,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub recommendation: String,
}

impl SecurityValidator {
    /// Create a new security validator
    pub fn new(config: SecurityConfig) -> Self {
        let rate_limiter = RateLimiter::new(
            config.max_requests_per_minute,
            Duration::from_secs(60),
        );
        
        let audit_logger = AuditLogger::new(10000); // Keep last 10k events
        
        let intrusion_detector = IntrusionDetector::new();
        
        Self {
            config,
            rate_limiter,
            audit_logger,
            intrusion_detector,
        }
    }
    
    /// Validate a request for security compliance
    pub fn validate_request(
        &mut self,
        source_ip: Option<IpAddr>,
        user_id: Option<String>,
        operation: &str,
        input_data: &[u8],
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        
        // Generate request ID for tracking
        let request_id = self.generate_request_id();
        
        log::debug!(
            "Validating security request {} for operation: {}",
            request_id,
            operation
        );
        
        let mut validation_result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            sanitized_input: None,
        };
        
        // IP-based validation
        if let Some(ip) = source_ip {
            if !self.validate_ip_address(ip)? {
                validation_result.is_valid = false;
                validation_result.errors.push(ValidationError {
                    field: "source_ip".to_string(),
                    error_type: ValidationErrorType::InvalidFormat,
                    message: format!("IP address {} is blocked or not allowed", ip),
                    severity: ThreatSeverity::High,
                });
            }
        }
        
        // Rate limiting check
        if self.config.enable_rate_limiting {
            let client_id = source_ip
                .map(|ip| ip.to_string())
                .or(user_id.clone())
                .unwrap_or_else(|| "anonymous".to_string());
            
            if !self.rate_limiter.check_rate_limit(&client_id)? {
                validation_result.is_valid = false;
                validation_result.errors.push(ValidationError {
                    field: "rate_limit".to_string(),
                    error_type: ValidationErrorType::ExcessiveLength,
                    message: "Rate limit exceeded".to_string(),
                    severity: ThreatSeverity::Medium,
                });
                
                // Log security event
                self.log_security_event(
                    SecurityEventType::RateLimitExceeded,
                    source_ip,
                    user_id.clone(),
                    operation.to_string(),
                    SecurityResult::Denied { reason: "Rate limit exceeded".to_string() },
                    HashMap::new(),
                )?;
            }
        }
        
        // Input validation
        if self.config.enable_input_validation {
            let mut input_validation = self.validate_input_data(input_data)?;
            validation_result.errors.append(&mut input_validation.errors);
            validation_result.warnings.extend(input_validation.warnings);
            
            if !validation_result.errors.is_empty() {
                validation_result.is_valid = false;
            }
            
            validation_result.sanitized_input = input_validation.sanitized_input;
        }
        
        // Intrusion detection
        if self.config.enable_intrusion_detection {
            let threat_score = self.intrusion_detector.analyze_request(
                source_ip,
                user_id.as_deref(),
                operation,
                input_data,
            )?;
            
            if threat_score > 0.7 {
                validation_result.is_valid = false;
                validation_result.errors.push(ValidationError {
                    field: "intrusion_detection".to_string(),
                    error_type: ValidationErrorType::MaliciousContent,
                    message: format!("High threat score detected: {:.2}", threat_score),
                    severity: ThreatSeverity::Critical,
                });
                
                // Log intrusion attempt
                let mut details = HashMap::new();
                details.insert("threat_score".to_string(), threat_score.to_string());
                
                self.log_security_event(
                    SecurityEventType::IntrusionAttempt,
                    source_ip,
                    user_id.clone(),
                    operation.to_string(),
                    SecurityResult::Blocked { reason: "Intrusion attempt detected".to_string() },
                    details,
                )?;
            } else if threat_score > 0.4 {
                validation_result.warnings.push(ValidationWarning {
                    field: "threat_analysis".to_string(),
                    message: format!("Moderate threat score: {:.2}", threat_score),
                    recommendation: "Monitor this request closely".to_string(),
                });
            }
        }
        
        // Log the validation result
        let result_type = if validation_result.is_valid {
            SecurityResult::Allowed
        } else {
            SecurityResult::Denied { 
                reason: validation_result.errors.iter()
                    .map(|e| e.message.clone())
                    .collect::<Vec<_>>()
                    .join("; ")
            }
        };
        
        if self.config.enable_audit_logging {
            let mut details = HashMap::new();
            details.insert("validation_time_ms".to_string(), start_time.elapsed().as_millis().to_string());
            details.insert("errors_count".to_string(), validation_result.errors.len().to_string());
            details.insert("warnings_count".to_string(), validation_result.warnings.len().to_string());
            
            self.log_security_event(
                SecurityEventType::InputValidation,
                source_ip,
                user_id,
                operation.to_string(),
                result_type,
                details,
            )?;
        }
        
        log::info!(
            "Security validation completed for request {} in {:?}ms: valid={}",
            request_id,
            start_time.elapsed().as_millis(),
            validation_result.is_valid
        );
        
        Ok(validation_result)
    }
    
    /// Validate IP address against allow/block lists
    fn validate_ip_address(&self, ip: IpAddr) -> Result<bool> {
        // Check if IP is explicitly blocked
        if self.config.blocked_ips.contains(&ip) {
            return Ok(false);
        }
        
        // If allow list is configured, check if IP is allowed
        if !self.config.allowed_ip_ranges.is_empty() {
            // Simplified IP range checking (in practice, use CIDR library)
            let ip_str = ip.to_string();
            let is_allowed = self.config.allowed_ip_ranges.iter()
                .any(|range| ip_str.starts_with(range));
            
            return Ok(is_allowed);
        }
        
        // Default allow if no restrictions configured
        Ok(true)
    }
    
    /// Validate input data for security issues
    fn validate_input_data(&self, input_data: &[u8]) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            sanitized_input: None,
        };
        
        // Size validation
        if input_data.len() > self.config.max_input_size_bytes {
            result.errors.push(ValidationError {
                field: "input_size".to_string(),
                error_type: ValidationErrorType::TooLarge,
                message: format!(
                    "Input size {} bytes exceeds maximum of {} bytes",
                    input_data.len(),
                    self.config.max_input_size_bytes
                ),
                severity: ThreatSeverity::Medium,
            });
        }
        
        // Content validation (if it's text)
        if let Ok(input_str) = std::str::from_utf8(input_data) {
            // Check for potential injection attacks
            let malicious_patterns = [
                "SELECT", "DROP", "DELETE", "UPDATE", "INSERT",
                "<script", "javascript:", "eval(", "exec(",
                "../", "..\\", "/etc/passwd", "cmd.exe",
                "'; DROP TABLE", "UNION SELECT", "OR 1=1",
                "${", "<%", "#{", "{{",
            ];
            
            for pattern in &malicious_patterns {
                if input_str.to_uppercase().contains(&pattern.to_uppercase()) {
                    result.errors.push(ValidationError {
                        field: "input_content".to_string(),
                        error_type: ValidationErrorType::Injection,
                        message: format!("Potential injection pattern detected: {}", pattern),
                        severity: ThreatSeverity::High,
                    });
                }
            }
            
            // Check for path traversal attempts
            if input_str.contains("../") || input_str.contains("..\\") {
                result.errors.push(ValidationError {
                    field: "input_content".to_string(),
                    error_type: ValidationErrorType::PathTraversal,
                    message: "Path traversal attempt detected".to_string(),
                    severity: ThreatSeverity::High,
                });
            }
            
            // Sanitize input by removing/escaping dangerous characters
            let sanitized = input_str
                .replace("<script", "&lt;script")
                .replace("javascript:", "")
                .replace("eval(", "")
                .replace("../", "");
            
            if sanitized != input_str {
                result.sanitized_input = Some(sanitized);
                result.warnings.push(ValidationWarning {
                    field: "input_content".to_string(),
                    message: "Input was sanitized to remove potentially dangerous content".to_string(),
                    recommendation: "Review input validation rules".to_string(),
                });
            }
        }
        
        // Binary data validation
        if input_data.len() > 1024 {
            // Check for executable headers
            let executable_signatures: &[&[u8]] = &[
                b"MZ",      // Windows PE
                &[0x7f, 0x45, 0x4c, 0x46], // Linux ELF
                &[0xcf, 0xfa], // Mach-O
                b"PK",      // ZIP/JAR
            ];
            
            for signature in executable_signatures {
                if input_data.starts_with(signature) {
                    result.warnings.push(ValidationWarning {
                        field: "input_content".to_string(),
                        message: "Executable file signature detected".to_string(),
                        recommendation: "Verify this is expected binary data".to_string(),
                    });
                    break;
                }
            }
        }
        
        result.is_valid = result.errors.is_empty();
        Ok(result)
    }
    
    /// Log a security event
    fn log_security_event(
        &mut self,
        event_type: SecurityEventType,
        source_ip: Option<IpAddr>,
        user_id: Option<String>,
        operation: String,
        result: SecurityResult,
        details: HashMap<String, String>,
    ) -> Result<()> {
        let risk_score = self.calculate_risk_score(&event_type, &result);
        
        let event = SecurityEvent {
            event_id: self.generate_event_id(),
            event_type,
            timestamp: Utc::now(),
            source_ip,
            user_id,
            operation,
            result,
            details,
            risk_score,
        };
        
        self.audit_logger.log_event(event.clone());
        
        // Log to system logger based on severity
        match event.risk_score {
            score if score >= 0.8 => log::error!("High-risk security event: {:?}", event),
            score if score >= 0.5 => log::warn!("Medium-risk security event: {:?}", event),
            _ => log::info!("Security event logged: {:?}", event),
        }
        
        Ok(())
    }
    
    /// Calculate risk score for an event
    fn calculate_risk_score(&self, event_type: &SecurityEventType, result: &SecurityResult) -> f64 {
        let base_score = match event_type {
            SecurityEventType::IntrusionAttempt => 0.9,
            SecurityEventType::RateLimitExceeded => 0.6,
            SecurityEventType::SuspiciousActivity => 0.7,
            SecurityEventType::InputValidation => 0.4,
            SecurityEventType::Authentication => 0.3,
            SecurityEventType::Authorization => 0.5,
            SecurityEventType::DataAccess => 0.3,
            SecurityEventType::ConfigurationChange => 0.6,
            SecurityEventType::ErrorCondition => 0.2,
        };
        
        let result_modifier = match result {
            SecurityResult::Blocked { .. } => 0.3,
            SecurityResult::Denied { .. } => 0.2,
            SecurityResult::Suspicious { confidence, .. } => confidence / 2.0,
            SecurityResult::Allowed => 0.0,
        };
        
        (base_score + result_modifier).min(1.0)
    }
    
    /// Generate unique request ID
    fn generate_request_id(&self) -> String {
        format!("req-{}", uuid::Uuid::new_v4().to_string()[..8].to_string())
    }
    
    /// Generate unique event ID
    fn generate_event_id(&self) -> String {
        format!("evt-{}", uuid::Uuid::new_v4().to_string()[..8].to_string())
    }
    
    /// Get security statistics
    pub fn get_security_stats(&self) -> SecurityStats {
        self.audit_logger.get_stats()
    }
    
    /// Get recent security events
    pub fn get_recent_events(&self, limit: usize) -> Vec<SecurityEvent> {
        self.audit_logger.get_recent_events(limit)
    }
}

/// Security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStats {
    pub total_events: usize,
    pub events_by_type: HashMap<String, usize>,
    pub high_risk_events: usize,
    pub blocked_requests: usize,
    pub average_risk_score: f64,
    pub last_24h_events: usize,
}

impl RateLimiter {
    fn new(max_requests: u32, window_duration: Duration) -> Self {
        Self {
            requests: HashMap::new(),
            max_requests,
            window_duration,
        }
    }
    
    fn check_rate_limit(&mut self, client_id: &str) -> Result<bool> {
        let now = Instant::now();
        
        // Clean up old requests
        let cutoff = now - self.window_duration;
        self.requests.entry(client_id.to_string())
            .or_insert_with(Vec::new)
            .retain(|&request_time| request_time > cutoff);
        
        // Check current count
        let request_count = self.requests.get(client_id).map(|v| v.len()).unwrap_or(0);
        
        if request_count >= self.max_requests as usize {
            log::warn!(
                "Rate limit exceeded for client {}: {} requests in {:?}",
                client_id,
                request_count,
                self.window_duration
            );
            return Ok(false);
        }
        
        // Add current request
        self.requests.get_mut(client_id).unwrap().push(now);
        
        Ok(true)
    }
}

impl AuditLogger {
    fn new(max_events: usize) -> Self {
        Self {
            events: Vec::new(),
            max_events,
        }
    }
    
    fn log_event(&mut self, event: SecurityEvent) {
        self.events.push(event);
        
        // Keep only recent events
        if self.events.len() > self.max_events {
            self.events.drain(0..self.events.len() - self.max_events);
        }
    }
    
    fn get_stats(&self) -> SecurityStats {
        let mut events_by_type = HashMap::new();
        let mut high_risk_count = 0;
        let mut blocked_count = 0;
        let mut total_risk_score = 0.0;
        let mut last_24h_count = 0;
        
        let now = Utc::now();
        let day_ago = now - chrono::Duration::hours(24);
        
        for event in &self.events {
            // Count by type
            let type_key = format!("{:?}", event.event_type);
            *events_by_type.entry(type_key).or_insert(0) += 1;
            
            // High risk events
            if event.risk_score >= 0.7 {
                high_risk_count += 1;
            }
            
            // Blocked requests
            if matches!(event.result, SecurityResult::Blocked { .. }) {
                blocked_count += 1;
            }
            
            // Risk score accumulation
            total_risk_score += event.risk_score;
            
            // Last 24h events
            if event.timestamp >= day_ago {
                last_24h_count += 1;
            }
        }
        
        SecurityStats {
            total_events: self.events.len(),
            events_by_type,
            high_risk_events: high_risk_count,
            blocked_requests: blocked_count,
            average_risk_score: if self.events.is_empty() {
                0.0
            } else {
                total_risk_score / self.events.len() as f64
            },
            last_24h_events: last_24h_count,
        }
    }
    
    fn get_recent_events(&self, limit: usize) -> Vec<SecurityEvent> {
        self.events.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

impl IntrusionDetector {
    fn new() -> Self {
        let detection_rules = vec![
            DetectionRule {
                rule_id: "high_frequency".to_string(),
                name: "High Frequency Requests".to_string(),
                condition: "requests_per_minute > 100".to_string(),
                action: DetectionAction::Alert,
                enabled: true,
            },
            DetectionRule {
                rule_id: "large_data".to_string(),
                name: "Large Data Access".to_string(),
                condition: "data_size > 100MB".to_string(),
                action: DetectionAction::Log,
                enabled: true,
            },
            DetectionRule {
                rule_id: "injection_attempt".to_string(),
                name: "SQL Injection Attempt".to_string(),
                condition: "contains_sql_keywords".to_string(),
                action: DetectionAction::Block,
                enabled: true,
            },
        ];
        
        Self {
            suspicious_patterns: Vec::new(),
            threat_scores: HashMap::new(),
            detection_rules,
        }
    }
    
    fn analyze_request(
        &mut self,
        source_ip: Option<IpAddr>,
        user_id: Option<&str>,
        operation: &str,
        input_data: &[u8],
    ) -> Result<f64> {
        let mut threat_score = 0.0;
        
        // Analyze request frequency
        let client_id = source_ip
            .map(|ip| ip.to_string())
            .or_else(|| user_id.map(|u| u.to_string()))
            .unwrap_or_else(|| "anonymous".to_string());
        
        // Simple frequency analysis
        let current_score = self.threat_scores.get(&client_id).unwrap_or(&0.0);
        let frequency_score = if *current_score > 0.5 { 0.3 } else { 0.0 };
        threat_score += frequency_score;
        
        // Analyze operation type
        let operation_score = match operation {
            op if op.contains("delete") || op.contains("drop") => 0.4,
            op if op.contains("admin") || op.contains("config") => 0.3,
            op if op.contains("backup") || op.contains("export") => 0.2,
            _ => 0.0,
        };
        threat_score += operation_score;
        
        // Analyze input data
        let data_score = if input_data.len() > 10 * 1024 * 1024 { // > 10MB
            0.3
        } else if input_data.len() > 1024 * 1024 { // > 1MB
            0.1
        } else {
            0.0
        };
        threat_score += data_score;
        
        // Check for suspicious patterns in data
        if let Ok(input_str) = std::str::from_utf8(input_data) {
            let suspicious_keywords = [
                "password", "secret", "key", "token",
                "admin", "root", "system", "config",
                "backup", "dump", "export", "sync",
            ];
            
            let keyword_count = suspicious_keywords.iter()
                .filter(|&&keyword| input_str.to_lowercase().contains(keyword))
                .count();
            
            threat_score += (keyword_count as f64) * 0.1;
        }
        
        // Update threat score for client
        self.threat_scores.insert(client_id.clone(), threat_score);
        
        // Log high threat scores
        if threat_score > 0.7 {
            log::warn!(
                "High threat score {:.2} detected for client {} in operation {}",
                threat_score,
                client_id,
                operation
            );
        }
        
        Ok(threat_score.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    
    #[test]
    fn test_security_validator_creation() {
        let config = SecurityConfig::default();
        let validator = SecurityValidator::new(config);
        
        assert!(validator.config.enable_input_validation);
        assert!(validator.config.enable_rate_limiting);
    }
    
    #[test]
    fn test_input_validation() {
        let config = SecurityConfig::default();
        let mut validator = SecurityValidator::new(config);
        
        // Test valid input
        let valid_input = b"normal data content";
        let result = validator.validate_request(
            None,
            None,
            "test_operation",
            valid_input,
        ).unwrap();
        
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        
        // Test malicious input
        let malicious_input = b"SELECT * FROM users; DROP TABLE users;";
        let result = validator.validate_request(
            None,
            None,
            "test_operation",
            malicious_input,
        ).unwrap();
        
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }
    
    #[test]
    fn test_rate_limiting() {
        let mut config = SecurityConfig::default();
        config.max_requests_per_minute = 2;
        
        let mut validator = SecurityValidator::new(config);
        let test_ip = Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        
        // First two requests should pass
        let result1 = validator.validate_request(test_ip, None, "test", b"data").unwrap();
        assert!(result1.is_valid);
        
        let result2 = validator.validate_request(test_ip, None, "test", b"data").unwrap();
        assert!(result2.is_valid);
        
        // Third request should be rate limited
        let result3 = validator.validate_request(test_ip, None, "test", b"data").unwrap();
        assert!(!result3.is_valid);
        assert!(result3.errors.iter().any(|e| e.error_type.matches(&ValidationErrorType::ExcessiveLength)));
    }
}

impl ValidationErrorType {
    fn matches(&self, other: &ValidationErrorType) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}