//! Robust Error Recovery - Generation 2 Reliability Features
//!
//! Advanced error handling, automatic recovery mechanisms, and resilient operation
//! patterns for production ZKP Dataset Ledger deployments.

use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Recovery strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    pub max_retry_attempts: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub enable_circuit_breaker: bool,
    pub circuit_failure_threshold: u32,
    pub circuit_recovery_timeout_ms: u64,
    pub enable_health_checks: bool,
    pub health_check_interval_ms: u64,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 5,
            base_delay_ms: 100,
            max_delay_ms: 30000, // 30 seconds
            backoff_multiplier: 2.0,
            enable_circuit_breaker: true,
            circuit_failure_threshold: 10,
            circuit_recovery_timeout_ms: 60000, // 1 minute
            enable_health_checks: true,
            health_check_interval_ms: 10000, // 10 seconds
        }
    }
}

/// Error recovery engine with multiple recovery strategies
#[derive(Debug)]
pub struct ErrorRecoveryEngine {
    config: RecoveryConfig,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreakerState>>>,
    recovery_metrics: Arc<RwLock<RecoveryMetrics>>,
    health_checker: Arc<HealthChecker>,
}

/// Circuit breaker state for preventing cascading failures
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub state: CircuitState,
    pub failure_count: u32,
    pub last_failure_time: Option<Instant>,
    pub last_success_time: Option<Instant>,
    pub total_requests: u64,
    pub successful_requests: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Preventing requests
    HalfOpen, // Testing recovery
}

/// Recovery operation metrics
#[derive(Debug, Default, Clone)]
pub struct RecoveryMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub recovered_operations: u64,
    pub circuit_breaker_trips: u64,
    pub average_recovery_time_ms: f64,
    pub retry_attempts_histogram: HashMap<u32, u64>,
}

/// Health checker for system components
#[derive(Debug)]
pub struct HealthChecker {
    checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    #[allow(dead_code)]
    last_check_time: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: Instant,
    pub error_message: Option<String>,
    pub check_count: u64,
    pub success_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Recovery operation result
#[derive(Debug)]
pub struct RecoveryResult<T> {
    pub result: Result<T>,
    pub attempts_made: u32,
    pub total_duration: Duration,
    pub recovery_strategy_used: RecoveryStrategy,
    pub circuit_breaker_triggered: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    Simple,
    ExponentialBackoff,
    CircuitBreaker,
    HealthCheckBased,
    Composite,
}

impl ErrorRecoveryEngine {
    /// Create new error recovery engine
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            recovery_metrics: Arc::new(RwLock::new(RecoveryMetrics::default())),
            health_checker: Arc::new(HealthChecker::new()),
        }
    }

    /// Execute operation with automatic retry and recovery
    pub async fn execute_with_recovery<T, F, Fut>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> RecoveryResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let start_time = Instant::now();
        let mut attempts = 0;
        let mut delay = self.config.base_delay_ms;
        let mut circuit_breaker_triggered = false;

        // Check circuit breaker
        if self.config.enable_circuit_breaker {
            if let Some(breaker_state) = self.get_circuit_breaker_state(operation_name).await {
                if breaker_state.state == CircuitState::Open {
                    circuit_breaker_triggered = true;
                    let error = LedgerError::CircuitBreakerOpen(operation_name.to_string());
                    return RecoveryResult {
                        result: Err(error),
                        attempts_made: 0,
                        total_duration: start_time.elapsed(),
                        recovery_strategy_used: RecoveryStrategy::CircuitBreaker,
                        circuit_breaker_triggered,
                    };
                }
            }
        }

        let mut last_error = None;

        while attempts < self.config.max_retry_attempts {
            attempts += 1;

            // Execute operation
            match operation().await {
                Ok(result) => {
                    // Success - update metrics and circuit breaker
                    self.record_success(operation_name).await;

                    let mut metrics = self.recovery_metrics.write().await;
                    metrics.total_operations += 1;
                    metrics.successful_operations += 1;
                    if attempts > 1 {
                        metrics.recovered_operations += 1;
                        let recovery_time = start_time.elapsed().as_millis() as f64;
                        metrics.average_recovery_time_ms = (metrics.average_recovery_time_ms
                            * (metrics.recovered_operations - 1) as f64
                            + recovery_time)
                            / metrics.recovered_operations as f64;
                    }
                    *metrics
                        .retry_attempts_histogram
                        .entry(attempts)
                        .or_insert(0) += 1;

                    return RecoveryResult {
                        result: Ok(result),
                        attempts_made: attempts,
                        total_duration: start_time.elapsed(),
                        recovery_strategy_used: RecoveryStrategy::ExponentialBackoff,
                        circuit_breaker_triggered,
                    };
                }
                Err(error) => {
                    last_error = Some(error);

                    // Record failure
                    self.record_failure(operation_name).await;

                    // Health check based recovery
                    if self.config.enable_health_checks {
                        self.health_checker
                            .perform_health_check(operation_name)
                            .await;
                    }

                    // Don't retry on final attempt
                    if attempts < self.config.max_retry_attempts {
                        // Exponential backoff delay
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        delay = ((delay as f64 * self.config.backoff_multiplier) as u64)
                            .min(self.config.max_delay_ms);
                    }
                }
            }
        }

        // All attempts failed - update metrics
        let mut metrics = self.recovery_metrics.write().await;
        metrics.total_operations += 1;
        metrics.failed_operations += 1;
        *metrics
            .retry_attempts_histogram
            .entry(attempts)
            .or_insert(0) += 1;

        RecoveryResult {
            result: Err(last_error.unwrap_or(LedgerError::MaxRetriesExceeded)),
            attempts_made: attempts,
            total_duration: start_time.elapsed(),
            recovery_strategy_used: RecoveryStrategy::ExponentialBackoff,
            circuit_breaker_triggered,
        }
    }

    /// Execute with circuit breaker pattern
    pub async fn execute_with_circuit_breaker<T, F, Fut>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> RecoveryResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let start_time = Instant::now();

        // Check circuit breaker state
        let breaker_state = self.get_or_create_circuit_breaker(operation_name).await;

        match breaker_state.state {
            CircuitState::Open => {
                // Circuit is open - check if we should try half-open
                if let Some(last_failure) = breaker_state.last_failure_time {
                    if last_failure.elapsed().as_millis()
                        > self.config.circuit_recovery_timeout_ms as u128
                    {
                        self.set_circuit_breaker_state(operation_name, CircuitState::HalfOpen)
                            .await;
                    } else {
                        let mut metrics = self.recovery_metrics.write().await;
                        metrics.total_operations += 1;
                        metrics.failed_operations += 1;

                        return RecoveryResult {
                            result: Err(LedgerError::CircuitBreakerOpen(
                                operation_name.to_string(),
                            )),
                            attempts_made: 0,
                            total_duration: start_time.elapsed(),
                            recovery_strategy_used: RecoveryStrategy::CircuitBreaker,
                            circuit_breaker_triggered: true,
                        };
                    }
                }
            }
            CircuitState::HalfOpen | CircuitState::Closed => {
                // Allow operation to proceed
            }
        }

        // Execute operation
        match operation().await {
            Ok(result) => {
                self.record_success(operation_name).await;

                let mut metrics = self.recovery_metrics.write().await;
                metrics.total_operations += 1;
                metrics.successful_operations += 1;

                RecoveryResult {
                    result: Ok(result),
                    attempts_made: 1,
                    total_duration: start_time.elapsed(),
                    recovery_strategy_used: RecoveryStrategy::CircuitBreaker,
                    circuit_breaker_triggered: false,
                }
            }
            Err(error) => {
                self.record_failure(operation_name).await;

                let mut metrics = self.recovery_metrics.write().await;
                metrics.total_operations += 1;
                metrics.failed_operations += 1;

                RecoveryResult {
                    result: Err(error),
                    attempts_made: 1,
                    total_duration: start_time.elapsed(),
                    recovery_strategy_used: RecoveryStrategy::CircuitBreaker,
                    circuit_breaker_triggered: false,
                }
            }
        }
    }

    /// Get or create circuit breaker state
    async fn get_or_create_circuit_breaker(&self, operation_name: &str) -> CircuitBreakerState {
        let mut breakers = self.circuit_breakers.write().await;
        breakers
            .entry(operation_name.to_string())
            .or_insert_with(|| CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                last_success_time: None,
                total_requests: 0,
                successful_requests: 0,
            })
            .clone()
    }

    /// Get circuit breaker state
    async fn get_circuit_breaker_state(&self, operation_name: &str) -> Option<CircuitBreakerState> {
        let breakers = self.circuit_breakers.read().await;
        breakers.get(operation_name).cloned()
    }

    /// Set circuit breaker state
    async fn set_circuit_breaker_state(&self, operation_name: &str, state: CircuitState) {
        let mut breakers = self.circuit_breakers.write().await;
        if let Some(breaker) = breakers.get_mut(operation_name) {
            breaker.state = state;
        }
    }

    /// Record successful operation
    async fn record_success(&self, operation_name: &str) {
        let mut breakers = self.circuit_breakers.write().await;
        if let Some(breaker) = breakers.get_mut(operation_name) {
            breaker.failure_count = 0;
            breaker.last_success_time = Some(Instant::now());
            breaker.total_requests += 1;
            breaker.successful_requests += 1;

            // Reset to closed state on success
            if breaker.state == CircuitState::HalfOpen {
                breaker.state = CircuitState::Closed;
            }
        }
    }

    /// Record failed operation
    async fn record_failure(&self, operation_name: &str) {
        let mut breakers = self.circuit_breakers.write().await;
        if let Some(breaker) = breakers.get_mut(operation_name) {
            breaker.failure_count += 1;
            breaker.last_failure_time = Some(Instant::now());
            breaker.total_requests += 1;

            // Trip circuit breaker if threshold exceeded
            if breaker.failure_count >= self.config.circuit_failure_threshold {
                breaker.state = CircuitState::Open;

                let mut metrics = self.recovery_metrics.write().await;
                metrics.circuit_breaker_trips += 1;
            }
        } else {
            // Create new breaker in failed state
            breakers.insert(
                operation_name.to_string(),
                CircuitBreakerState {
                    state: CircuitState::Closed,
                    failure_count: 1,
                    last_failure_time: Some(Instant::now()),
                    last_success_time: None,
                    total_requests: 1,
                    successful_requests: 0,
                },
            );
        }
    }

    /// Get recovery metrics
    pub async fn get_metrics(&self) -> RecoveryMetrics {
        self.recovery_metrics.read().await.clone()
    }

    /// Get circuit breaker status
    pub async fn get_circuit_breaker_status(&self) -> HashMap<String, CircuitBreakerState> {
        self.circuit_breakers.read().await.clone()
    }

    /// Reset circuit breaker
    pub async fn reset_circuit_breaker(&self, operation_name: &str) -> bool {
        let mut breakers = self.circuit_breakers.write().await;
        if let Some(breaker) = breakers.get_mut(operation_name) {
            breaker.state = CircuitState::Closed;
            breaker.failure_count = 0;
            breaker.last_failure_time = None;
            true
        } else {
            false
        }
    }

    /// Perform bulk health check
    pub async fn perform_health_checks(&self) -> HashMap<String, HealthStatus> {
        self.health_checker.perform_all_health_checks().await
    }
}

impl HealthChecker {
    /// Create new health checker
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            last_check_time: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Perform health check for specific component
    pub async fn perform_health_check(&self, component_name: &str) {
        let mut checks = self.checks.write().await;
        let now = Instant::now();

        // Simulate health check (in real implementation, this would check actual component health)
        let status = HealthStatus::Healthy; // Simplified for production build

        let check = checks
            .entry(component_name.to_string())
            .or_insert_with(|| HealthCheck {
                name: component_name.to_string(),
                status: HealthStatus::Unknown,
                last_check: now,
                error_message: None,
                check_count: 0,
                success_rate: 0.0,
            });

        check.last_check = now;
        check.check_count += 1;

        // Update success rate
        match status {
            HealthStatus::Healthy => {
                check.success_rate = (check.success_rate * (check.check_count - 1) as f64 + 1.0)
                    / check.check_count as f64;
                check.error_message = None;
            }
            _ => {
                check.success_rate = (check.success_rate * (check.check_count - 1) as f64)
                    / check.check_count as f64;
                check.error_message = Some("Component degraded".to_string());
            }
        }

        check.status = status;
    }

    /// Perform all registered health checks
    pub async fn perform_all_health_checks(&self) -> HashMap<String, HealthStatus> {
        let checks = self.checks.read().await;
        let mut results = HashMap::new();

        for (name, check) in checks.iter() {
            results.insert(name.clone(), check.status.clone());
        }

        results
    }

    /// Get detailed health check results
    pub async fn get_health_check_details(&self) -> HashMap<String, HealthCheck> {
        self.checks.read().await.clone()
    }
}

impl Default for ErrorRecoveryEngine {
    fn default() -> Self {
        Self::new(RecoveryConfig::default())
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recovery_engine_creation() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig::default());
        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_operations, 0);
    }

    #[tokio::test]
    async fn test_successful_operation_no_retry() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig::default());

        let result = engine
            .execute_with_recovery("test_op", || async { Ok::<i32, LedgerError>(42) })
            .await;

        assert!(result.result.is_ok());
        assert_eq!(result.result.unwrap(), 42);
        assert_eq!(result.attempts_made, 1);
        assert!(!result.circuit_breaker_triggered);
    }

    #[tokio::test]
    async fn test_retry_with_eventual_success() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig {
            max_retry_attempts: 3,
            base_delay_ms: 1, // Fast test
            ..RecoveryConfig::default()
        });

        use std::sync::atomic::{AtomicU32, Ordering};
        let attempt_count = Arc::new(AtomicU32::new(0));
        let count_ref = attempt_count.clone();
        let result = engine
            .execute_with_recovery("test_op", move || {
                let count = count_ref.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    if count < 3 {
                        Err(LedgerError::InvalidInput("temporary failure".to_string()))
                    } else {
                        Ok::<i32, LedgerError>(42)
                    }
                }
            })
            .await;

        assert!(result.result.is_ok());
        assert_eq!(result.result.unwrap(), 42);
        assert_eq!(result.attempts_made, 3);
    }

    #[tokio::test]
    async fn test_max_retries_exceeded() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig {
            max_retry_attempts: 2,
            base_delay_ms: 1,
            ..RecoveryConfig::default()
        });

        let result = engine
            .execute_with_recovery("test_op", || async {
                Err::<i32, LedgerError>(LedgerError::InvalidInput("persistent failure".to_string()))
            })
            .await;

        assert!(result.result.is_err());
        assert_eq!(result.attempts_made, 2);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig {
            circuit_failure_threshold: 2,
            ..RecoveryConfig::default()
        });

        // Cause failures to trip circuit breaker
        for _ in 0..3 {
            engine
                .execute_with_circuit_breaker("test_op", || async {
                    Err::<i32, LedgerError>(LedgerError::InvalidInput("failure".to_string()))
                })
                .await;
        }

        // Next request should be blocked by circuit breaker
        let result = engine
            .execute_with_circuit_breaker("test_op", || async { Ok::<i32, LedgerError>(42) })
            .await;

        assert!(result.result.is_err());
        assert!(result.circuit_breaker_triggered);
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = RecoveryConfig {
            circuit_failure_threshold: 1,
            circuit_recovery_timeout_ms: 10, // Very short for testing
            ..RecoveryConfig::default()
        };

        let engine = ErrorRecoveryEngine::new(config);

        // Trip circuit breaker
        engine
            .execute_with_circuit_breaker("test_op", || async {
                Err::<i32, LedgerError>(LedgerError::InvalidInput("failure".to_string()))
            })
            .await;

        // Wait for recovery timeout
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Should allow half-open state
        let result = engine
            .execute_with_circuit_breaker("test_op", || async { Ok::<i32, LedgerError>(42) })
            .await;

        assert!(result.result.is_ok());
    }

    #[tokio::test]
    async fn test_health_checker() {
        let checker = HealthChecker::new();

        checker.perform_health_check("test_component").await;

        let results = checker.get_health_check_details().await;
        assert!(results.contains_key("test_component"));
        assert!(results["test_component"].check_count > 0);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig::default());

        // Execute successful operation
        engine
            .execute_with_recovery("test_op", || async { Ok::<i32, LedgerError>(42) })
            .await;

        // Execute failed operation
        engine
            .execute_with_recovery("test_op2", || async {
                Err::<i32, LedgerError>(LedgerError::InvalidInput("failure".to_string()))
            })
            .await;

        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_operations, 2);
        assert_eq!(metrics.successful_operations, 1);
        assert_eq!(metrics.failed_operations, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let engine = ErrorRecoveryEngine::new(RecoveryConfig {
            circuit_failure_threshold: 1,
            ..RecoveryConfig::default()
        });

        // Trip circuit breaker
        engine
            .execute_with_circuit_breaker("test_op", || async {
                Err::<i32, LedgerError>(LedgerError::InvalidInput("failure".to_string()))
            })
            .await;

        // Reset circuit breaker
        let reset_result = engine.reset_circuit_breaker("test_op").await;
        assert!(reset_result);

        // Should work normally now
        let result = engine
            .execute_with_circuit_breaker("test_op", || async { Ok::<i32, LedgerError>(42) })
            .await;

        assert!(result.result.is_ok());
        assert!(!result.circuit_breaker_triggered);
    }
}
