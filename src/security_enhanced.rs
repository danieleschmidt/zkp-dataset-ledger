//! Enhanced security features for production deployment

use crate::{LedgerError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;
use tracing::{debug, info, warn, error};

/// Memory protection utilities for sensitive data
pub struct SecureMemory {
    data: Vec<u8>,
}

impl SecureMemory {
    /// Create new secure memory region
    pub fn new(size: usize) -> Result<Self> {
        let mut data = vec![0u8; size];
        
        // In production, use mlock() to prevent swapping to disk
        #[cfg(unix)]
        {
            unsafe {
                if libc::mlock(data.as_ptr() as *const libc::c_void, size) != 0 {
                    warn!("Failed to lock memory pages - sensitive data may be swapped");
                }
            }
        }
        
        Ok(Self { data })
    }
    
    /// Write data securely
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.data.len() {
            return Err(LedgerError::SecurityViolation(
                "Write would exceed secure memory bounds".to_string()
            ));
        }
        
        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
    
    /// Read data securely
    pub fn read(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > self.data.len() {
            return Err(LedgerError::SecurityViolation(
                "Read would exceed secure memory bounds".to_string()
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

impl KeyManager {
    pub fn new() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            rotation_schedule: Arc::new(Mutex::new(HashMap::new())),
        }
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
        
        schedule.insert(
            key_id.to_string(), 
            Utc::now() + chrono::Duration::days(90)
        );
        
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
                    drop(keys); // Release read lock before write operations
                    
                    self.generate_key(&new_key_id, &current_key.algorithm, current_key.usage.clone())?;
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
            None => Err(LedgerError::SecurityViolation(
                format!("Key not found: {}", key_id)
            )),
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
        use std::thread;
        use std::time::Duration;
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let delay_ms = rng.gen_range(1..10);
        thread::sleep(Duration::from_millis(delay_ms));
    }
    
    /// Constant-time conditional assignment
    pub fn constant_time_select(condition: bool, true_val: &[u8], false_val: &[u8]) -> Vec<u8> {
        let mask = if condition { 0xFF } else { 0x00 };
        true_val.iter()
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
                LedgerError::ConcurrencyError("Failed to acquire circuit breaker state lock".to_string())
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
                            LedgerError::ConcurrencyError("Failed to acquire circuit breaker state lock".to_string())
                        })?;
                        *state = CircuitState::HalfOpen;
                        drop(state);
                        return self.try_operation(operation);
                    }
                }
                
                Err(LedgerError::ServiceUnavailable(
                    "Circuit breaker is open - service unavailable".to_string()
                ))
            },
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
            },
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
            LedgerError::ConcurrencyError("Failed to acquire circuit breaker state lock".to_string())
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
                LedgerError::ConcurrencyError("Failed to acquire circuit breaker state lock".to_string())
            })?;
            *state = CircuitState::Open;
            warn!("Circuit breaker opened due to {} consecutive failures", *failure_count);
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
            return Err(LedgerError::InvalidInput("Invalid CSV header format".to_string()));
        }
        
        // Check for injection attempts in CSV
        let mut sanitized_lines = Vec::new();
        for line in lines {
            if line.starts_with('=') || line.starts_with('@') || line.starts_with('+') || line.starts_with('-') {
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
                "Potential prototype pollution detected in JSON".to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_secure_memory() {
        let mut secure_mem = SecureMemory::new(1024).unwrap();
        let test_data = b"sensitive data";
        
        secure_mem.write(0, test_data).unwrap();
        let read_data = secure_mem.read(0, test_data.len()).unwrap();
        
        assert_eq!(read_data, test_data);
    }
    
    #[test]
    fn test_timing_protection() {
        let a = b"password123";
        let b = b"password123";
        let c = b"wrongpass11";
        
        assert!(TimingProtection::constant_time_compare(a, b));
        assert!(!TimingProtection::constant_time_compare(a, c));
    }
    
    #[test]
    fn test_key_manager() {
        let key_manager = KeyManager::new();
        
        let key_id = key_manager.generate_key(
            "test_key", 
            "AES-256-GCM", 
            KeyUsage::Encryption
        ).unwrap();
        
        let key_data = key_manager.get_key(&key_id).unwrap();
        assert_eq!(key_data.len(), 32);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let circuit_breaker = CircuitBreaker::new(2, 1);
        
        // First failure
        let result = circuit_breaker.execute(|| {
            Err::<(), _>(LedgerError::ServiceUnavailable("Test failure".to_string()))
        });
        assert!(result.is_err());
        
        // Second failure should open circuit
        let result = circuit_breaker.execute(|| {
            Err::<(), _>(LedgerError::ServiceUnavailable("Test failure".to_string()))
        });
        assert!(result.is_err());
    }
    
    #[test]
    fn test_secure_random() {
        let secure_random = SecureRandom::new(6.0);
        
        let bytes = secure_random.generate_bytes(32).unwrap();
        assert_eq!(bytes.len(), 32);
        
        let token = secure_random.generate_session_token().unwrap();
        assert_eq!(token.len(), 32);
    }
    
    #[test]
    fn test_content_sanitizer() {
        let sanitizer = ContentSanitizer();
        
        // Test CSV sanitization
        let csv_with_injection = "name,age\n=cmd|'/C calc'!A0,25";
        let sanitized = sanitizer.sanitize_csv(csv_with_injection).unwrap();
        assert!(sanitized.contains("'=cmd"));
        
        // Test dangerous character escaping
        let dangerous_input = "<script>alert('xss')</script>";
        let escaped = sanitizer.escape_dangerous_chars(dangerous_input);
        assert!(!escaped.contains("<script>"));
    }
}