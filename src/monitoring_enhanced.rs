//! Advanced monitoring, metrics, and observability for production deployment

use crate::{LedgerError, Result};
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn, error, instrument, span, Level};
use uuid::Uuid;

/// Comprehensive system health monitoring
pub struct HealthMonitor {
    checks: Arc<RwLock<HashMap<String, Box<dyn HealthCheck + Send + Sync>>>>,
    status_cache: Arc<Mutex<HealthStatus>>,
    alert_manager: AlertManager,
    metrics_collector: MetricsCollector,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            status_cache: Arc::new(Mutex::new(HealthStatus::healthy())),
            alert_manager: AlertManager::new(),
            metrics_collector: MetricsCollector::new(),
        }
    }
    
    /// Register a health check
    pub fn register_check(&self, name: String, check: Box<dyn HealthCheck + Send + Sync>) -> Result<()> {
        let mut checks = self.checks.write().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire health checks lock".to_string())
        })?;
        
        checks.insert(name.clone(), check);
        info!("Registered health check: {}", name);
        Ok(())
    }
    
    /// Perform all health checks
    #[instrument(skip(self))]
    pub async fn perform_health_checks(&self) -> Result<HealthStatus> {
        let checks = self.checks.read().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire health checks lock".to_string())
        })?;
        
        let mut overall_status = HealthStatus::healthy();
        let mut individual_results = HashMap::new();
        
        for (name, check) in checks.iter() {
            let start_time = Instant::now();
            
            let result = match check.check().await {
                Ok(status) => {
                    self.metrics_collector.record_health_check_duration(
                        name, 
                        start_time.elapsed(),
                        true
                    );
                    status
                },
                Err(e) => {
                    self.metrics_collector.record_health_check_duration(
                        name, 
                        start_time.elapsed(),
                        false
                    );
                    
                    HealthCheckResult::unhealthy(format!("Health check failed: {}", e))
                }
            };
            
            if !result.is_healthy {
                overall_status.is_healthy = false;
                overall_status.issues.push(format!("{}: {}", name, result.message));
                
                // Trigger alert for failed health check
                self.alert_manager.trigger_alert(Alert::new(
                    AlertLevel::Warning,
                    format!("Health check failed: {}", name),
                    result.message.clone(),
                )).await?;
            }
            
            individual_results.insert(name.clone(), result);
        }
        
        overall_status.last_updated = Utc::now();
        overall_status.individual_checks = individual_results;
        
        // Cache the status
        if let Ok(mut cache) = self.status_cache.lock() {
            *cache = overall_status.clone();
        }
        
        Ok(overall_status)
    }
    
    /// Get cached health status
    pub fn get_cached_status(&self) -> Result<HealthStatus> {
        let status = self.status_cache.lock().map_err(|_| {
            LedgerError::ConcurrencyError("Failed to acquire status cache lock".to_string())
        })?;
        Ok(status.clone())
    }
    
    /// Get metrics collector
    pub fn metrics(&self) -> &MetricsCollector {
        &self.metrics_collector
    }
}

/// Health check trait for implementing custom checks
#[async_trait::async_trait]
pub trait HealthCheck {
    async fn check(&self) -> Result<HealthCheckResult>;
}

/// Result of a health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub is_healthy: bool,
    pub message: String,
    pub details: HashMap<String, String>,
    pub checked_at: DateTime<Utc>,
}

impl HealthCheckResult {
    pub fn healthy(message: String) -> Self {
        Self {
            is_healthy: true,
            message,
            details: HashMap::new(),
            checked_at: Utc::now(),
        }
    }
    
    pub fn unhealthy(message: String) -> Self {
        Self {
            is_healthy: false,
            message,
            details: HashMap::new(),
            checked_at: Utc::now(),
        }
    }
    
    pub fn with_detail(mut self, key: String, value: String) -> Self {
        self.details.insert(key, value);
        self
    }
}

/// Overall system health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub issues: Vec<String>,
    pub last_updated: DateTime<Utc>,
    pub individual_checks: HashMap<String, HealthCheckResult>,
    pub system_info: SystemInfo,
}

impl HealthStatus {
    pub fn healthy() -> Self {
        Self {
            is_healthy: true,
            issues: Vec::new(),
            last_updated: Utc::now(),
            individual_checks: HashMap::new(),
            system_info: SystemInfo::collect(),
        }
    }
}

/// System information for health reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub uptime_seconds: u64,
    pub rust_version: String,
}

impl SystemInfo {
    pub fn collect() -> Self {
        let uptime = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            hostname: hostname::get()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            uptime_seconds: uptime,
            rust_version: env!("RUSTC_VERSION", "unknown").to_string(),
        }
    }
}

/// Advanced metrics collection with time-series data
pub struct MetricsCollector {
    counters: Arc<RwLock<HashMap<String, u64>>>,
    histograms: Arc<RwLock<HashMap<String, Histogram>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    time_series: Arc<RwLock<HashMap<String, TimeSeries>>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            time_series: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str, value: u64) {
        if let Ok(mut counters) = self.counters.write() {
            *counters.entry(name.to_string()).or_insert(0) += value;
            
            // Also record in time series
            self.record_time_series_point(name, value as f64);
        }
    }
    
    /// Record a histogram value
    pub fn record_histogram(&self, name: &str, value: f64) {
        if let Ok(mut histograms) = self.histograms.write() {
            histograms.entry(name.to_string())
                .or_insert_with(|| Histogram::new())
                .record(value);
                
            // Also record in time series
            self.record_time_series_point(name, value);
        }
    }
    
    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        if let Ok(mut gauges) = self.gauges.write() {
            gauges.insert(name.to_string(), value);
            
            // Also record in time series
            self.record_time_series_point(name, value);
        }
    }
    
    /// Record time series data point
    fn record_time_series_point(&self, name: &str, value: f64) {
        if let Ok(mut time_series) = self.time_series.write() {
            time_series.entry(name.to_string())
                .or_insert_with(|| TimeSeries::new(1000)) // Keep last 1000 points
                .add_point(Utc::now(), value);
        }
    }
    
    /// Record health check duration
    pub fn record_health_check_duration(&self, check_name: &str, duration: Duration, success: bool) {
        let duration_ms = duration.as_millis() as f64;
        self.record_histogram(&format!("health_check_duration_{}", check_name), duration_ms);
        
        if success {
            self.increment_counter(&format!("health_check_success_{}", check_name), 1);
        } else {
            self.increment_counter(&format!("health_check_failure_{}", check_name), 1);
        }
    }
    
    /// Get counter value
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters.read()
            .map(|counters| counters.get(name).copied().unwrap_or(0))
            .unwrap_or(0)
    }
    
    /// Get histogram statistics
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.histograms.read()
            .ok()
            .and_then(|histograms| histograms.get(name).map(|h| h.stats()))
    }
    
    /// Get gauge value
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.read()
            .ok()
            .and_then(|gauges| gauges.get(name).copied())
    }
    
    /// Get time series data
    pub fn get_time_series(&self, name: &str, since: Option<DateTime<Utc>>) -> Option<Vec<(DateTime<Utc>, f64)>> {
        self.time_series.read()
            .ok()
            .and_then(|series| series.get(name).map(|ts| ts.get_points_since(since)))
    }
    
    /// Export all metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();
        
        // Export counters
        if let Ok(counters) = self.counters.read() {
            for (name, value) in counters.iter() {
                output.push_str(&format!("# TYPE {} counter\n{} {}\n", name, name, value));
            }
        }
        
        // Export gauges
        if let Ok(gauges) = self.gauges.read() {
            for (name, value) in gauges.iter() {
                output.push_str(&format!("# TYPE {} gauge\n{} {}\n", name, name, value));
            }
        }
        
        // Export histogram summaries
        if let Ok(histograms) = self.histograms.read() {
            for (name, histogram) in histograms.iter() {
                let stats = histogram.stats();
                output.push_str(&format!("# TYPE {}_count counter\n{}_count {}\n", name, name, stats.count));
                output.push_str(&format!("# TYPE {}_sum counter\n{}_sum {}\n", name, name, stats.sum));
                output.push_str(&format!("# TYPE {} histogram\n", name));
                
                for (percentile, value) in &[
                    (50, stats.p50),
                    (90, stats.p90),
                    (95, stats.p95),
                    (99, stats.p99),
                ] {
                    output.push_str(&format!("{}{{quantile=\"0.{}\"}} {}\n", name, percentile, value));
                }
            }
        }
        
        output
    }
}

/// Histogram for tracking value distributions
pub struct Histogram {
    values: Vec<f64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            sum: 0.0,
            count: 0,
        }
    }
    
    pub fn record(&mut self, value: f64) {
        self.values.push(value);
        self.sum += value;
        self.count += 1;
        
        // Keep only recent values to prevent memory growth
        if self.values.len() > 10000 {
            self.values.drain(..5000);
        }
    }
    
    pub fn stats(&self) -> HistogramStats {
        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let len = sorted_values.len();
        let p50 = if len > 0 { sorted_values[len / 2] } else { 0.0 };
        let p90 = if len > 0 { sorted_values[(len * 90) / 100] } else { 0.0 };
        let p95 = if len > 0 { sorted_values[(len * 95) / 100] } else { 0.0 };
        let p99 = if len > 0 { sorted_values[(len * 99) / 100] } else { 0.0 };
        
        HistogramStats {
            count: self.count,
            sum: self.sum,
            average: if self.count > 0 { self.sum / self.count as f64 } else { 0.0 },
            p50,
            p90,
            p95,
            p99,
        }
    }
}

/// Histogram statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    pub count: u64,
    pub sum: f64,
    pub average: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Time series data storage
pub struct TimeSeries {
    points: VecDeque<(DateTime<Utc>, f64)>,
    max_size: usize,
}

impl TimeSeries {
    pub fn new(max_size: usize) -> Self {
        Self {
            points: VecDeque::new(),
            max_size,
        }
    }
    
    pub fn add_point(&mut self, timestamp: DateTime<Utc>, value: f64) {
        self.points.push_back((timestamp, value));
        
        // Maintain maximum size
        while self.points.len() > self.max_size {
            self.points.pop_front();
        }
    }
    
    pub fn get_points_since(&self, since: Option<DateTime<Utc>>) -> Vec<(DateTime<Utc>, f64)> {
        match since {
            Some(since_time) => {
                self.points.iter()
                    .filter(|(timestamp, _)| *timestamp >= since_time)
                    .copied()
                    .collect()
            },
            None => self.points.iter().copied().collect(),
        }
    }
}

/// Alert management system
pub struct AlertManager {
    alerts: Arc<Mutex<Vec<Alert>>>,
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    notification_channels: Vec<Box<dyn NotificationChannel + Send + Sync>>,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            alerts: Arc::new(Mutex::new(Vec::new())),
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            notification_channels: Vec::new(),
        }
    }
    
    pub async fn trigger_alert(&self, alert: Alert) -> Result<()> {
        // Store alert
        if let Ok(mut alerts) = self.alerts.lock() {
            alerts.push(alert.clone());
            
            // Keep only recent alerts
            if alerts.len() > 10000 {
                alerts.drain(..5000);
            }
        }
        
        // Send notifications
        for channel in &self.notification_channels {
            if let Err(e) = channel.send_notification(&alert).await {
                error!("Failed to send alert notification: {}", e);
            }
        }
        
        info!(
            alert_id = %alert.id,
            level = ?alert.level,
            title = %alert.title,
            "Alert triggered"
        );
        
        Ok(())
    }
    
    pub fn add_notification_channel(&mut self, channel: Box<dyn NotificationChannel + Send + Sync>) {
        self.notification_channels.push(channel);
    }
    
    pub fn get_recent_alerts(&self, limit: usize) -> Vec<Alert> {
        self.alerts.lock()
            .map(|alerts| {
                alerts.iter()
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Alert levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: Uuid,
    pub level: AlertLevel,
    pub title: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Alert {
    pub fn new(level: AlertLevel, title: String, message: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            level,
            title,
            message,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Alert rule for automated alerting
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration: ChronoDuration,
}

/// Alert conditions
#[derive(Debug, Clone)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
}

/// Notification channel trait
#[async_trait::async_trait]
pub trait NotificationChannel {
    async fn send_notification(&self, alert: &Alert) -> Result<()>;
}

/// Console notification channel
pub struct ConsoleNotificationChannel;

#[async_trait::async_trait]
impl NotificationChannel for ConsoleNotificationChannel {
    async fn send_notification(&self, alert: &Alert) -> Result<()> {
        println!("🚨 ALERT: {} - {}", alert.title, alert.message);
        Ok(())
    }
}

/// Health checks for common components
pub struct DatabaseHealthCheck {
    connection_test: Arc<dyn Fn() -> Result<bool> + Send + Sync>,
}

impl DatabaseHealthCheck {
    pub fn new<F>(connection_test: F) -> Self
    where
        F: Fn() -> Result<bool> + Send + Sync + 'static,
    {
        Self {
            connection_test: Arc::new(connection_test),
        }
    }
}

#[async_trait::async_trait]
impl HealthCheck for DatabaseHealthCheck {
    async fn check(&self) -> Result<HealthCheckResult> {
        match (self.connection_test)() {
            Ok(true) => Ok(HealthCheckResult::healthy("Database connection OK".to_string())),
            Ok(false) => Ok(HealthCheckResult::unhealthy("Database connection failed".to_string())),
            Err(e) => Ok(HealthCheckResult::unhealthy(format!("Database check error: {}", e))),
        }
    }
}

/// Storage health check
pub struct StorageHealthCheck {
    storage_path: String,
}

impl StorageHealthCheck {
    pub fn new(storage_path: String) -> Self {
        Self { storage_path }
    }
}

#[async_trait::async_trait]
impl HealthCheck for StorageHealthCheck {
    async fn check(&self) -> Result<HealthCheckResult> {
        use std::fs;
        
        // Check if path exists and is writable
        if !std::path::Path::new(&self.storage_path).exists() {
            return Ok(HealthCheckResult::unhealthy("Storage path does not exist".to_string()));
        }
        
        // Try to write a test file
        let test_file = format!("{}/health_check_{}", self.storage_path, Uuid::new_v4());
        
        match fs::write(&test_file, "health_check") {
            Ok(_) => {
                // Clean up test file
                let _ = fs::remove_file(&test_file);
                Ok(HealthCheckResult::healthy("Storage is writable".to_string()))
            },
            Err(e) => Ok(HealthCheckResult::unhealthy(format!("Storage write failed: {}", e))),
        }
    }
}

/// Memory health check
pub struct MemoryHealthCheck {
    max_memory_mb: u64,
}

impl MemoryHealthCheck {
    pub fn new(max_memory_mb: u64) -> Self {
        Self { max_memory_mb }
    }
}

#[async_trait::async_trait]
impl HealthCheck for MemoryHealthCheck {
    async fn check(&self) -> Result<HealthCheckResult> {
        // Get current memory usage (simplified - in production use proper memory monitoring)
        let memory_usage_mb = get_memory_usage_mb();
        
        if memory_usage_mb > self.max_memory_mb {
            Ok(HealthCheckResult::unhealthy(
                format!("Memory usage {}MB exceeds limit {}MB", memory_usage_mb, self.max_memory_mb)
            ))
        } else {
            Ok(HealthCheckResult::healthy(
                format!("Memory usage {}MB within limit", memory_usage_mb)
            ).with_detail("memory_usage_mb".to_string(), memory_usage_mb.to_string()))
        }
    }
}

// Simplified memory usage calculation
fn get_memory_usage_mb() -> u64 {
    // In production, use proper system monitoring libraries like `sysinfo`
    // For now, return a placeholder value
    100
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[test]
    fn test_histogram() {
        let mut histogram = Histogram::new();
        
        histogram.record(1.0);
        histogram.record(2.0);
        histogram.record(3.0);
        histogram.record(4.0);
        histogram.record(5.0);
        
        let stats = histogram.stats();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.sum, 15.0);
        assert_eq!(stats.average, 3.0);
    }
    
    #[test]
    fn test_time_series() {
        let mut time_series = TimeSeries::new(3);
        let now = Utc::now();
        
        time_series.add_point(now, 1.0);
        time_series.add_point(now + ChronoDuration::seconds(1), 2.0);
        time_series.add_point(now + ChronoDuration::seconds(2), 3.0);
        time_series.add_point(now + ChronoDuration::seconds(3), 4.0);
        
        let points = time_series.get_points_since(None);
        assert_eq!(points.len(), 3); // Should only keep last 3 points
        assert_eq!(points[2].1, 4.0); // Latest point should be 4.0
    }
    
    #[tokio::test]
    async fn test_health_monitor() {
        let monitor = HealthMonitor::new();
        
        // Register a test health check
        let check = Box::new(TestHealthCheck { should_pass: true });
        monitor.register_check("test_check".to_string(), check).unwrap();
        
        let status = monitor.perform_health_checks().await.unwrap();
        assert!(status.is_healthy);
    }
    
    struct TestHealthCheck {
        should_pass: bool,
    }
    
    #[async_trait::async_trait]
    impl HealthCheck for TestHealthCheck {
        async fn check(&self) -> Result<HealthCheckResult> {
            if self.should_pass {
                Ok(HealthCheckResult::healthy("Test passed".to_string()))
            } else {
                Ok(HealthCheckResult::unhealthy("Test failed".to_string()))
            }
        }
    }
    
    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        collector.increment_counter("test_counter", 5);
        collector.record_histogram("test_histogram", 10.5);
        collector.set_gauge("test_gauge", 42.0);
        
        assert_eq!(collector.get_counter("test_counter"), 5);
        assert_eq!(collector.get_gauge("test_gauge"), Some(42.0));
        
        let histogram_stats = collector.get_histogram_stats("test_histogram").unwrap();
        assert_eq!(histogram_stats.count, 1);
        assert_eq!(histogram_stats.sum, 10.5);
    }
    
    #[tokio::test]
    async fn test_alert_manager() {
        let mut alert_manager = AlertManager::new();
        alert_manager.add_notification_channel(Box::new(ConsoleNotificationChannel));
        
        let alert = Alert::new(
            AlertLevel::Warning,
            "Test Alert".to_string(),
            "This is a test alert".to_string(),
        );
        
        alert_manager.trigger_alert(alert).await.unwrap();
        
        let recent_alerts = alert_manager.get_recent_alerts(10);
        assert_eq!(recent_alerts.len(), 1);
    }
}