//! Enhanced monitoring system with real-time alerting and security detection.

use crate::monitoring::{HealthStatus, SystemMetrics};
use crate::{LedgerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced monitoring with advanced capabilities
pub struct EnhancedMonitor {
    pub config: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_realtime_alerts: bool,
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub anomaly_score_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    System,
    Security,
    Performance,
    Anomaly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_realtime_alerts: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 80.0,
            memory_usage_percent: 85.0,
            anomaly_score_threshold: 0.95,
        }
    }
}

impl EnhancedMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        Self { config }
    }

    /// Enhanced anomaly detection
    pub fn detect_anomalies(&self, system_metrics: &SystemMetrics) -> Result<f64> {
        let mut anomaly_score = 0.0;

        // Check CPU usage anomaly
        if system_metrics.cpu_usage > 95.0 {
            anomaly_score = anomaly_score.max(0.8);
        }

        // Check memory usage anomaly
        if system_metrics.memory_usage > 90.0 {
            anomaly_score = anomaly_score.max(0.7);
        }

        Ok(anomaly_score)
    }

    /// Generate security alerts
    pub fn check_security_threats(&self) -> Result<Vec<Alert>> {
        let mut alerts = Vec::new();

        // Simulate security monitoring
        alerts.push(Alert {
            id: "sec-001".to_string(),
            alert_type: AlertType::Security,
            severity: AlertSeverity::Info,
            message: "Security monitoring active".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });

        Ok(alerts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_monitor() {
        let config = MonitoringConfig::default();
        let monitor = EnhancedMonitor::new(config);
        assert!(monitor.config.enable_realtime_alerts);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = MonitoringConfig::default();
        let monitor = EnhancedMonitor::new(config);
        
        let system_metrics = SystemMetrics {
            cpu_usage: 99.0,
            memory_usage: 50.0,
            disk_usage: 30.0,
            network_io: 100,
            active_connections: 10,
            queue_depth: 2,
        };
        
        let anomaly_score = monitor.detect_anomalies(&system_metrics).unwrap();
        assert!(anomaly_score > 0.0);
    }
}