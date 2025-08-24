//! Enhanced Security Module for ZKP Dataset Ledger

use crate::{LedgerError, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use log::{info, warn};
use rand::{thread_rng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub max_failed_attempts: u32,
    pub lockout_duration_minutes: u32,
    pub require_mfa: bool,
    pub session_timeout_minutes: u32,
    pub audit_retention_days: u32,
    pub key_rotation_days: u32,
    pub threat_detection_enabled: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            max_failed_attempts: 5,
            lockout_duration_minutes: 30,
            require_mfa: true,
            session_timeout_minutes: 60,
            audit_retention_days: 2555,
            key_rotation_days: 90,
            threat_detection_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    Authentication {
        user_id: String,
        success: bool,
        ip_address: String,
    },
    SecurityViolation {
        description: String,
        source_ip: String,
    },
    KeyRotation {
        key_id: String,
        operation: String,
        success: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: SecurityEventType,
    pub user_id: Option<String>,
    pub ip_address: String,
    pub risk_score: f64,
}

#[derive(Debug)]
pub struct SecurityManager {
    config: SecurityConfig,
    audit_log: Arc<Mutex<VecDeque<SecurityAuditEntry>>>,
    blocked_ips: Arc<RwLock<HashSet<String>>>,
    suspicious_activities: Arc<DashMap<String, SuspiciousActivity>>,
    encryption_keys: Arc<RwLock<HashMap<String, EncryptionKey>>>,
    security_metrics: Arc<RwLock<SecurityMetrics>>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SuspiciousActivity {
    ip_address: String,
    activity_count: u32,
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
    activity_types: HashSet<String>,
    risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncryptionKey {
    key_id: String,
    key_data: Vec<u8>,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    version: u32,
    active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub total_auth_attempts: u64,
    pub successful_auths: u64,
    pub failed_auths: u64,
    pub blocked_ips: usize,
    pub security_violations: u64,
    pub audit_entries: usize,
    pub key_rotations: u64,
    pub compliance_score: f64,
    pub last_updated: DateTime<Utc>,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Self {
        let initial_metrics = SecurityMetrics {
            total_auth_attempts: 0,
            successful_auths: 0,
            failed_auths: 0,
            blocked_ips: 0,
            security_violations: 0,
            audit_entries: 0,
            key_rotations: 0,
            compliance_score: 1.0,
            last_updated: Utc::now(),
        };

        let mut manager = Self {
            config,
            audit_log: Arc::new(Mutex::new(VecDeque::new())),
            blocked_ips: Arc::new(RwLock::new(HashSet::new())),
            suspicious_activities: Arc::new(DashMap::new()),
            encryption_keys: Arc::new(RwLock::new(HashMap::new())),
            security_metrics: Arc::new(RwLock::new(initial_metrics)),
        };

        manager.generate_master_keys().unwrap_or_else(|e| {
            log::error!("Failed to generate master keys: {}", e);
        });

        manager
    }

    pub fn track_suspicious_activity(&self, ip_address: &str, activity_type: &str) {
        let now = Utc::now();

        self.suspicious_activities
            .entry(ip_address.to_string())
            .and_modify(|activity| {
                activity.activity_count += 1;
                activity.last_seen = now;
                activity.activity_types.insert(activity_type.to_string());
                activity.risk_score = (activity.risk_score + 0.1).min(1.0);
            })
            .or_insert(SuspiciousActivity {
                ip_address: ip_address.to_string(),
                activity_count: 1,
                first_seen: now,
                last_seen: now,
                activity_types: [activity_type.to_string()].iter().cloned().collect(),
                risk_score: 0.1,
            });

        if let Some(activity) = self.suspicious_activities.get(ip_address) {
            if activity.risk_score > 0.8 || activity.activity_count > 20 {
                let mut blocked_ips = self.blocked_ips.write().unwrap();
                blocked_ips.insert(ip_address.to_string());

                log::warn!(
                    "Blocked suspicious IP: {} (risk: {:.2}, activities: {})",
                    ip_address,
                    activity.risk_score,
                    activity.activity_count
                );
            }
        }
    }

    pub fn is_ip_blocked(&self, ip_address: &str) -> bool {
        let blocked_ips = self.blocked_ips.read().unwrap();
        blocked_ips.contains(ip_address)
    }

    pub fn log_security_event(&self, event_type: SecurityEventType) {
        let entry = SecurityAuditEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type: event_type.clone(),
            user_id: self.extract_user_id_from_event(&event_type),
            ip_address: self.extract_ip_from_event(&event_type),
            risk_score: self.calculate_event_risk_score(&event_type),
        };

        let mut audit_log = self.audit_log.lock().unwrap();
        audit_log.push_back(entry);

        let max_entries = (self.config.audit_retention_days * 1000) as usize;
        while audit_log.len() > max_entries {
            audit_log.pop_front();
        }
    }

    fn generate_master_keys(&mut self) -> Result<()> {
        let mut keys = self.encryption_keys.write().unwrap();

        let mut key_data = vec![0u8; 32];
        thread_rng().fill_bytes(&mut key_data);

        let primary_key = EncryptionKey {
            key_id: "primary-v1".to_string(),
            key_data,
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(self.config.key_rotation_days as i64),
            version: 1,
            active: true,
        };

        keys.insert("primary".to_string(), primary_key);

        log::info!("Generated master encryption keys");
        Ok(())
    }

    pub fn rotate_keys(&mut self) -> Result<()> {
        let mut keys = self.encryption_keys.write().unwrap();

        let current_version = keys.get("primary").map(|k| k.version).unwrap_or(0);

        let mut new_key_data = vec![0u8; 32];
        thread_rng().fill_bytes(&mut new_key_data);

        let new_key = EncryptionKey {
            key_id: format!("primary-v{}", current_version + 1),
            key_data: new_key_data,
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::days(self.config.key_rotation_days as i64),
            version: current_version + 1,
            active: true,
        };

        if let Some(old_key) = keys.get_mut("primary") {
            old_key.active = false;
        }

        keys.insert("primary".to_string(), new_key);

        self.log_security_event(SecurityEventType::KeyRotation {
            key_id: format!("primary-v{}", current_version + 1),
            operation: "rotate".to_string(),
            success: true,
        });

        let mut metrics = self.security_metrics.write().unwrap();
        metrics.key_rotations += 1;

        log::info!(
            "Successfully rotated encryption keys to version {}",
            current_version + 1
        );
        Ok(())
    }

    pub fn get_security_metrics(&self) -> SecurityMetrics {
        let mut metrics = self.security_metrics.read().unwrap().clone();

        metrics.blocked_ips = self.blocked_ips.read().unwrap().len();
        metrics.audit_entries = self.audit_log.lock().unwrap().len();
        metrics.last_updated = Utc::now();

        metrics
    }

    pub fn get_audit_log(&self, limit: Option<usize>) -> Vec<SecurityAuditEntry> {
        let audit_log = self.audit_log.lock().unwrap();

        match limit {
            Some(n) => audit_log.iter().rev().take(n).cloned().collect(),
            None => audit_log.iter().cloned().collect(),
        }
    }

    fn extract_user_id_from_event(&self, event: &SecurityEventType) -> Option<String> {
        match event {
            SecurityEventType::Authentication { user_id, .. } => Some(user_id.clone()),
            _ => None,
        }
    }

    fn extract_ip_from_event(&self, event: &SecurityEventType) -> String {
        match event {
            SecurityEventType::Authentication { ip_address, .. } => ip_address.clone(),
            SecurityEventType::SecurityViolation { source_ip, .. } => source_ip.clone(),
            _ => "unknown".to_string(),
        }
    }

    fn calculate_event_risk_score(&self, event: &SecurityEventType) -> f64 {
        match event {
            SecurityEventType::Authentication { success: false, .. } => 0.3,
            SecurityEventType::SecurityViolation { .. } => 0.8,
            _ => 0.1,
        }
    }
}

/// Secure memory management with mlock protection
pub struct SecureMemory {
    data: Vec<u8>,
}

impl SecureMemory {
    /// Create new secure memory region
    pub fn new(size: usize) -> Result<Self> {
        let data = vec![0u8; size];

        // In production, use mlock() to prevent swapping to disk
        #[cfg(unix)]
        {
            unsafe {
                if libc::mlock(data.as_ptr() as *const libc::c_void, size) != 0 {
                    log::warn!("Failed to lock memory pages - sensitive data may be swapped");
                }
            }
        }

        Ok(Self { data })
    }

    /// Write data securely
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.data.len() {
            return Err(LedgerError::SecurityViolation(
                "Write would exceed secure memory bounds".to_string(),
            ));
        }

        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data securely
    pub fn read(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > self.data.len() {
            return Err(LedgerError::SecurityViolation(
                "Read would exceed secure memory bounds".to_string(),
            ));
        }

        Ok(&self.data[offset..offset + len])
    }
}

impl Drop for SecureMemory {
    fn drop(&mut self) {
        // Securely wipe memory before deallocation
        for byte in &mut self.data {
            *byte = 0;
        }

        // Unlock memory pages
        #[cfg(unix)]
        {
            unsafe {
                libc::munlock(self.data.as_ptr() as *const libc::c_void, self.data.len());
            }
        }
    }
}

/// Cryptographic key manager with secure storage and rotation
pub struct KeyManager {
    keys: Arc<RwLock<HashMap<String, SecureKey>>>,
    rotation_schedule: Arc<Mutex<HashMap<String, DateTime<Utc>>>>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct SecureKey {
    key_id: String,
    key_data: Vec<u8>,
    created_at: DateTime<Utc>,
    algorithm: String,
    usage: KeyUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyUsage {
    Encryption,
    Signing,
    Authentication,
    KeyDerivation,
}

impl Default for KeyManager {
    fn default() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            rotation_schedule: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl KeyManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate new cryptographic key
    pub fn generate_key(&self, key_id: &str, algorithm: &str, usage: KeyUsage) -> Result<String> {
        use rand::RngCore;

        let mut key_data = vec![0u8; 32]; // 256-bit key
        rand::thread_rng().fill_bytes(&mut key_data);

        let secure_key = SecureKey {
            key_id: key_id.to_string(),
            key_data,
            created_at: Utc::now(),
            algorithm: algorithm.to_string(),
            usage,
        };

        let mut keys = self.keys.write().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire keys lock".to_string())
        })?;

        keys.insert(key_id.to_string(), secure_key);

        // Schedule rotation
        let mut schedule = self.rotation_schedule.lock().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire rotation schedule lock".to_string())
        })?;

        schedule.insert(key_id.to_string(), Utc::now() + chrono::Duration::days(90));

        info!("Generated new key: {} ({})", key_id, algorithm);
        Ok(key_id.to_string())
    }

    /// Rotate key if needed
    pub fn check_and_rotate_keys(&self) -> Result<Vec<String>> {
        let mut rotated_keys = Vec::new();
        let now = Utc::now();

        let schedule = self.rotation_schedule.lock().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire rotation schedule lock".to_string())
        })?;

        for (key_id, &rotation_time) in schedule.iter() {
            if now >= rotation_time {
                // Get current key info for rotation
                let keys = self.keys.read().map_err(|_| {
                    LedgerError::ConcurrencyError("Failed to acquire keys lock".to_string())
                })?;

                if let Some(current_key) = keys.get(key_id) {
                    let new_key_id = format!("{}_rotated_{}", key_id, now.timestamp());
                    let algorithm = current_key.algorithm.clone();
                    let usage = current_key.usage.clone();
                    drop(keys); // Release read lock before write operations

                    self.generate_key(&new_key_id, &algorithm, usage)?;
                    rotated_keys.push(new_key_id);

                    warn!("Key {} rotated due to age", key_id);
                }
            }
        }

        Ok(rotated_keys)
    }

    /// Get key for cryptographic operations
    pub fn get_key(&self, key_id: &str) -> Result<Vec<u8>> {
        let keys = self.keys.read().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire keys lock".to_string())
        })?;

        match keys.get(key_id) {
            Some(key) => Ok(key.key_data.clone()),
            None => Err(LedgerError::SecurityViolation(format!(
                "Key not found: {}",
                key_id
            ))),
        }
    }
}

/// Timing attack protection utilities
pub struct TimingProtection;

impl TimingProtection {
    /// Constant-time string comparison
    pub fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (byte_a, byte_b) in a.iter().zip(b.iter()) {
            result |= byte_a ^ byte_b;
        }

        result == 0
    }

    /// Add random delay to prevent timing analysis
    pub fn add_random_delay() {
        use rand::Rng;
        use std::thread;
        use std::time::Duration;

        let mut rng = rand::thread_rng();
        let delay_ms = rng.gen_range(1..10);
        thread::sleep(Duration::from_millis(delay_ms));
    }

    /// Constant-time conditional assignment
    pub fn constant_time_select(condition: bool, true_val: &[u8], false_val: &[u8]) -> Vec<u8> {
        let mask = if condition { 0xFF } else { 0x00 };
        true_val
            .iter()
            .zip(false_val.iter())
            .map(|(t, f)| (t & mask) | (f & !mask))
            .collect()
    }
}

/// Circuit breaker for external dependencies
pub struct CircuitBreaker {
    failure_count: Arc<Mutex<u32>>,
    last_failure: Arc<Mutex<Option<DateTime<Utc>>>>,
    failure_threshold: u32,
    recovery_timeout: chrono::Duration,
    state: Arc<Mutex<CircuitState>>,
}

#[derive(Debug, Clone, Copy)]
enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, reject all requests
    HalfOpen, // Testing if service has recovered
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout_secs: i64) -> Self {
        Self {
            failure_count: Arc::new(Mutex::new(0)),
            last_failure: Arc::new(Mutex::new(None)),
            failure_threshold,
            recovery_timeout: chrono::Duration::seconds(recovery_timeout_secs),
            state: Arc::new(Mutex::new(CircuitState::Closed)),
        }
    }

    /// Execute operation with circuit breaker protection
    pub fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let current_state = {
            let state = self.state.lock().map_err(|_| {
                LedgerError::ConcurrencyError(
                    "Failed to acquire circuit breaker state lock".to_string(),
                )
            })?;
            *state
        };

        match current_state {
            CircuitState::Open => {
                // Check if we should transition to half-open
                let last_failure = self.last_failure.lock().map_err(|_| {
                    LedgerError::ConcurrencyError("Failed to acquire last failure lock".to_string())
                })?;

                if let Some(last_fail_time) = *last_failure {
                    let now = Utc::now();
                    if now.signed_duration_since(last_fail_time) > self.recovery_timeout {
                        drop(last_failure);
                        let mut state = self.state.lock().map_err(|_| {
                            LedgerError::ConcurrencyError(
                                "Failed to acquire circuit breaker state lock".to_string(),
                            )
                        })?;
                        *state = CircuitState::HalfOpen;
                        drop(state);
                        return self.try_operation(operation);
                    }
                }

                Err(LedgerError::ServiceUnavailable(
                    "Circuit breaker is open - service unavailable".to_string(),
                ))
            }
            CircuitState::HalfOpen => self.try_operation(operation),
            CircuitState::Closed => self.try_operation(operation),
        }
    }

    fn try_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        match operation() {
            Ok(result) => {
                self.on_success()?;
                Ok(result)
            }
            Err(e) => {
                self.on_failure()?;
                Err(e)
            }
        }
    }

    fn on_success(&self) -> Result<()> {
        let mut failure_count = self.failure_count.lock().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire failure count lock".to_string())
        })?;
        *failure_count = 0;

        let mut state = self.state.lock().map_err(|_| {
            LedgerError::ConcurrencyError(
                "Failed to acquire circuit breaker state lock".to_string(),
            )
        })?;
        *state = CircuitState::Closed;

        Ok(())
    }

    fn on_failure(&self) -> Result<()> {
        let mut failure_count = self.failure_count.lock().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire failure count lock".to_string())
        })?;
        *failure_count += 1;

        let mut last_failure = self.last_failure.lock().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire last failure lock".to_string())
        })?;
        *last_failure = Some(Utc::now());

        if *failure_count >= self.failure_threshold {
            let mut state = self.state.lock().map_err(|_| {
                LedgerError::ConcurrencyError(
                    "Failed to acquire circuit breaker state lock".to_string(),
                )
            })?;
            *state = CircuitState::Open;
            warn!(
                "Circuit breaker opened due to {} consecutive failures",
                *failure_count
            );
        }

        Ok(())
    }
}

/// Secure random number generator with entropy validation
pub struct SecureRandom {
    entropy_threshold: f64,
}

impl SecureRandom {
    pub fn new(entropy_threshold: f64) -> Self {
        Self { entropy_threshold }
    }

    /// Generate cryptographically secure random bytes
    pub fn generate_bytes(&self, len: usize) -> Result<Vec<u8>> {
        use rand::RngCore;

        let mut bytes = vec![0u8; len];
        rand::thread_rng().fill_bytes(&mut bytes);

        // Validate entropy
        let entropy = self.calculate_entropy(&bytes);
        if entropy < self.entropy_threshold {
            warn!("Generated random bytes have low entropy: {}", entropy);
            // In production, might want to reseed or use hardware RNG
        }

        Ok(bytes)
    }

    /// Calculate Shannon entropy of data
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        let mut counts = [0u64; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }

        let total = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &counts {
            if count > 0 {
                let probability = count as f64 / total;
                entropy -= probability * probability.log2();
            }
        }

        entropy
    }

    /// Generate secure random string
    pub fn generate_string(&self, len: usize, charset: &[u8]) -> Result<String> {
        let random_bytes = self.generate_bytes(len)?;
        let result: String = random_bytes
            .iter()
            .map(|&b| charset[(b as usize) % charset.len()] as char)
            .collect();

        Ok(result)
    }

    /// Generate secure session token
    pub fn generate_session_token(&self) -> Result<String> {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                 abcdefghijklmnopqrstuvwxyz\
                                 0123456789";
        self.generate_string(32, CHARSET)
    }
}

/// Content sanitization and validation
pub struct ContentSanitizer;

impl ContentSanitizer {
    /// Sanitize CSV content
    pub fn sanitize_csv(&self, content: &str) -> Result<String> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(LedgerError::InvalidInput("Empty CSV content".to_string()));
        }

        // Validate header
        let header = lines[0];
        if header.is_empty() || header.contains(';') || header.contains('|') {
            return Err(LedgerError::InvalidInput(
                "Invalid CSV header format".to_string(),
            ));
        }

        // Check for injection attempts in CSV
        let mut sanitized_lines = Vec::new();
        for line in lines {
            if line.starts_with('=')
                || line.starts_with('@')
                || line.starts_with('+')
                || line.starts_with('-')
            {
                // Potential CSV injection
                let sanitized_line = format!("'{}", line);
                sanitized_lines.push(sanitized_line);
            } else {
                sanitized_lines.push(line.to_string());
            }
        }

        Ok(sanitized_lines.join("\n"))
    }

    /// Validate JSON structure
    pub fn validate_json(&self, content: &str) -> Result<()> {
        serde_json::from_str::<serde_json::Value>(content)
            .map_err(|e| LedgerError::InvalidInput(format!("Invalid JSON: {}", e)))?;

        // Check for potential JSON injection patterns
        if content.contains("__proto__") || content.contains("constructor") {
            return Err(LedgerError::SecurityViolation(
                "Potential prototype pollution detected in JSON".to_string(),
            ));
        }

        Ok(())
    }

    /// Remove or escape dangerous characters
    pub fn escape_dangerous_chars(&self, input: &str) -> String {
        input
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
            .replace('&', "&amp;")
            .replace('\0', "")
    }
}
