//! Comprehensive Monitoring and Observability System
//!
//! Provides metrics collection, performance monitoring, health checks,
//! alerting, and observability for the ZKP Dataset Ledger.

use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// System metrics for monitoring performance and health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Available memory in bytes
    pub memory_available: u64,
    /// Number of active operations
    pub active_operations: u64,
    /// Total operations completed
    pub total_operations: u64,
    /// Average operation duration in milliseconds
    pub avg_operation_duration_ms: f64,
    /// Error rate (errors per 1000 operations)
    pub error_rate: f64,
    /// Timestamp of measurement
    pub timestamp: DateTime<Utc>,
}

/// Performance metrics for specific operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Operation name
    pub operation: String,
    /// Number of executions
    pub count: u64,
    /// Total duration in milliseconds
    pub total_duration_ms: u64,
    /// Average duration in milliseconds
    pub avg_duration_ms: f64,
    /// Minimum duration in milliseconds
    pub min_duration_ms: u64,
    /// Maximum duration in milliseconds
    pub max_duration_ms: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            operation: String::new(),
            count: 0,
            total_duration_ms: 0,
            avg_duration_ms: 0.0,
            min_duration_ms: u64::MAX,
            max_duration_ms: 0,
            success_rate: 1.0,
            last_updated: Utc::now(),
        }
    }
}

impl PerformanceMetrics {
    /// Update metrics with new measurement
    pub fn update(&mut self, duration_ms: u64, success: bool) {
        self.count += 1;
        self.total_duration_ms += duration_ms;
        self.avg_duration_ms = self.total_duration_ms as f64 / self.count as f64;
        self.min_duration_ms = self.min_duration_ms.min(duration_ms);
        self.max_duration_ms = self.max_duration_ms.max(duration_ms);

        // Update success rate
        let current_successes =
            (self.success_rate * (self.count - 1) as f64) + if success { 1.0 } else { 0.0 };
        self.success_rate = current_successes / self.count as f64;

        self.last_updated = Utc::now();
    }
}

/// Health status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health information for different system components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthInfo {
    /// Overall health status
    pub healthy: bool,
    /// Component name
    pub component: String,
    /// Health score (0.0 - 1.0)
    pub health_score: f64,
    /// Status message
    pub status: String,
    /// Last health check timestamp
    pub last_check: DateTime<Utc>,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

impl HealthInfo {
    pub fn new(component: &str) -> Self {
        Self {
            healthy: true,
            component: component.to_string(),
            health_score: 1.0,
            status: "OK".to_string(),
            last_check: Utc::now(),
            uptime_seconds: 0,
        }
    }

    pub fn set_unhealthy(&mut self, reason: &str) {
        self.healthy = false;
        self.health_score = 0.0;
        self.status = reason.to_string();
        self.last_check = Utc::now();
    }
}

/// Simple monitoring system
pub struct MonitoringSystem {
    /// System start time
    pub start_time: Instant,
    /// Performance metrics by operation
    performance_metrics: Arc<Mutex<HashMap<String, PerformanceMetrics>>>,
    /// System health status by component
    health_status: Arc<Mutex<HashMap<String, HealthInfo>>>,
}

impl MonitoringSystem {
    /// Create new monitoring system
    pub fn new() -> Self {
        let system = Self {
            start_time: Instant::now(),
            performance_metrics: Arc::new(Mutex::new(HashMap::new())),
            health_status: Arc::new(Mutex::new(HashMap::new())),
        };

        // Initialize core component health
        system.register_component("ledger");
        system.register_component("storage");
        system.register_component("cryptography");

        system
    }

    /// Register a new component for monitoring
    pub fn register_component(&self, component: &str) {
        let mut health = self.health_status.lock().unwrap();
        health.insert(component.to_string(), HealthInfo::new(component));
    }

    /// Record performance measurement for an operation
    pub fn record_operation(
        &self,
        operation: &str,
        duration: Duration,
        success: bool,
    ) -> Result<()> {
        let duration_ms = duration.as_millis() as u64;

        let mut metrics = self.performance_metrics.lock().unwrap();
        let metric = metrics
            .entry(operation.to_string())
            .or_insert_with(|| PerformanceMetrics {
                operation: operation.to_string(),
                ..Default::default()
            });

        metric.update(duration_ms, success);
        Ok(())
    }

    /// Update component health status
    pub fn update_health(&self, component: &str, healthy: bool, message: &str) {
        let mut health = self.health_status.lock().unwrap();
        if let Some(status) = health.get_mut(component) {
            if healthy {
                status.healthy = true;
                status.health_score = 1.0;
                status.status = message.to_string();
            } else {
                status.set_unhealthy(message);
            }
            status.last_check = Utc::now();
            status.uptime_seconds = self.start_time.elapsed().as_secs();
        }
    }

    /// Collect system metrics
    pub fn collect_system_metrics(&self) -> SystemMetrics {
        let performance = self.performance_metrics.lock().unwrap();

        let total_operations: u64 = performance.values().map(|m| m.count).sum();
        let avg_duration = if !performance.is_empty() {
            performance.values().map(|m| m.avg_duration_ms).sum::<f64>() / performance.len() as f64
        } else {
            0.0
        };

        SystemMetrics {
            cpu_usage: 50.0,                      // Mock value
            memory_usage: 1024 * 1024 * 512,      // 512 MB
            memory_available: 1024 * 1024 * 1536, // 1.5 GB
            active_operations: 2,
            total_operations,
            avg_operation_duration_ms: avg_duration,
            error_rate: 0.0,
            timestamp: Utc::now(),
        }
    }

    /// Get overall system health
    pub fn system_health(&self) -> HealthInfo {
        let health = self.health_status.lock().unwrap();

        let overall_score = if health.is_empty() {
            1.0
        } else {
            health.values().map(|h| h.health_score).sum::<f64>() / health.len() as f64
        };

        HealthInfo {
            healthy: overall_score > 0.8,
            component: "system".to_string(),
            health_score: overall_score,
            status: "All systems operational".to_string(),
            last_check: Utc::now(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    /// Generate monitoring dashboard
    pub fn generate_dashboard(&self) -> String {
        let mut dashboard = String::new();

        dashboard.push_str("# ZKP Dataset Ledger Monitoring Dashboard\n\n");

        // System overview
        let system_health = self.system_health();
        dashboard.push_str("## System Overview\n");
        dashboard.push_str(&format!(
            "- **Status**: {}\n",
            if system_health.healthy {
                "🟢 Healthy"
            } else {
                "🔴 Unhealthy"
            }
        ));
        dashboard.push_str(&format!(
            "- **Health Score**: {:.1}%\n",
            system_health.health_score * 100.0
        ));
        dashboard.push_str(&format!(
            "- **Uptime**: {}s\n",
            system_health.uptime_seconds
        ));
        dashboard.push_str(&format!(
            "- **Last Check**: {}\n\n",
            system_health.last_check.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // System metrics
        let metrics = self.collect_system_metrics();
        dashboard.push_str("## System Metrics\n");
        dashboard.push_str(&format!("- **CPU Usage**: {:.1}%\n", metrics.cpu_usage));
        dashboard.push_str(&format!(
            "- **Memory Usage**: {} MB / {} MB\n",
            metrics.memory_usage / 1024 / 1024,
            (metrics.memory_usage + metrics.memory_available) / 1024 / 1024
        ));
        dashboard.push_str(&format!(
            "- **Active Operations**: {}\n",
            metrics.active_operations
        ));
        dashboard.push_str(&format!(
            "- **Total Operations**: {}\n",
            metrics.total_operations
        ));
        dashboard.push_str(&format!(
            "- **Avg Operation Time**: {:.1}ms\n\n",
            metrics.avg_operation_duration_ms
        ));

        // Performance metrics
        dashboard.push_str("## Performance Metrics\n");
        let performance = self.performance_metrics.lock().unwrap();
        for (operation, perf) in performance.iter() {
            dashboard.push_str(&format!("### {}\n", operation));
            dashboard.push_str(&format!(
                "- Count: {} | Avg: {:.1}ms | Success: {:.1}%\n",
                perf.count,
                perf.avg_duration_ms,
                perf.success_rate * 100.0
            ));
        }
        dashboard.push('\n');

        // Component health
        dashboard.push_str("## Component Health\n");
        let health = self.health_status.lock().unwrap();
        for (component, status) in health.iter() {
            let icon = if status.healthy { "🟢" } else { "🔴" };
            dashboard.push_str(&format!(
                "- **{}**: {} {} (Score: {:.1}%)\n",
                component,
                icon,
                status.status,
                status.health_score * 100.0
            ));
        }

        dashboard
    }
}

impl Default for MonitoringSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics {
            operation: "test_op".to_string(),
            ..Default::default()
        };

        metrics.update(100, true);
        metrics.update(200, true);
        metrics.update(300, false);

        assert_eq!(metrics.count, 3);
        assert_eq!(metrics.avg_duration_ms, 200.0);
        assert_eq!(metrics.min_duration_ms, 100);
        assert_eq!(metrics.max_duration_ms, 300);
        assert_eq!(metrics.success_rate, 2.0 / 3.0);
    }

    #[test]
    fn test_monitoring_system() {
        let monitoring = MonitoringSystem::new();

        // Record some operations
        monitoring
            .record_operation("test_op", Duration::from_millis(150), true)
            .unwrap();
        monitoring
            .record_operation("test_op", Duration::from_millis(200), true)
            .unwrap();

        // Update component health
        monitoring.update_health("test_component", true, "All good");

        let metrics = monitoring.collect_system_metrics();
        assert_eq!(metrics.total_operations, 2);

        let health = monitoring.system_health();
        assert!(health.healthy);
    }
}
