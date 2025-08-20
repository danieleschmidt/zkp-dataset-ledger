//! Auto-scaling system for dynamically adjusting compute resources based on load and performance.

use crate::{LedgerError, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::time::Interval;

/// Auto-scaling manager for dynamic resource allocation
#[derive(Debug)]
pub struct AutoScaler {
    config: ScalingConfig,
    metrics_collector: Arc<Mutex<MetricsCollector>>,
    scaling_decisions: Arc<Mutex<VecDeque<ScalingDecision>>>,
    current_resources: Arc<Mutex<ResourceAllocation>>,
    scaling_policies: Vec<ScalingPolicy>,
}

/// Configuration for auto-scaling behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub cooldown_period_minutes: i64,
    pub evaluation_interval_seconds: u64,
    pub enable_predictive_scaling: bool,
    pub enable_cost_optimization: bool,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 20,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 40.0,
            cooldown_period_minutes: 5,
            evaluation_interval_seconds: 30,
            enable_predictive_scaling: true,
            enable_cost_optimization: true,
        }
    }
}

/// Metrics collector for monitoring system performance
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    cpu_usage_history: VecDeque<MetricPoint>,
    memory_usage_history: VecDeque<MetricPoint>,
    request_rate_history: VecDeque<MetricPoint>,
    response_time_history: VecDeque<MetricPoint>,
    error_rate_history: VecDeque<MetricPoint>,
    queue_depth_history: VecDeque<MetricPoint>,
    max_history_size: usize,
}

/// A single metric measurement point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub source: String,
}

/// Resource allocation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub worker_count: usize,
    pub cpu_cores_per_worker: usize,
    pub memory_gb_per_worker: usize,
    pub total_cpu_cores: usize,
    pub total_memory_gb: usize,
    pub cost_per_hour: f64,
    pub last_updated: DateTime<Utc>,
}

/// Scaling decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub timestamp: DateTime<Utc>,
    pub decision_type: ScalingDecisionType,
    pub from_workers: usize,
    pub to_workers: usize,
    pub trigger_metrics: HashMap<String, f64>,
    pub reasoning: String,
    pub confidence_score: f64,
    pub executed: bool,
    pub execution_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingDecisionType {
    ScaleUp,
    ScaleDown,
    NoAction,
    Emergency,
}

/// Scaling policy defines rules for when and how to scale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    pub name: String,
    pub condition: ScalingCondition,
    pub action: ScalingAction,
    pub priority: u32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingCondition {
    CpuThreshold {
        threshold: f64,
        duration_seconds: u64,
    },
    MemoryThreshold {
        threshold: f64,
        duration_seconds: u64,
    },
    QueueDepthThreshold {
        threshold: usize,
        duration_seconds: u64,
    },
    ResponseTimeThreshold {
        threshold_ms: u64,
        duration_seconds: u64,
    },
    ErrorRateThreshold {
        threshold: f64,
        duration_seconds: u64,
    },
    PredictiveLoad {
        predicted_increase_percent: f64,
    },
    CompositeCondition {
        conditions: Vec<ScalingCondition>,
        operator: LogicalOperator,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalOperator {
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    ScaleToCount(usize),
    ScaleByPercent(i32),
    ScaleByCount(i32),
    Emergency,
}

impl AutoScaler {
    /// Create a new auto-scaler with configuration
    pub fn new(config: ScalingConfig) -> Self {
        let metrics_collector = Arc::new(Mutex::new(MetricsCollector::new()));
        let scaling_decisions = Arc::new(Mutex::new(VecDeque::new()));

        let initial_resources = ResourceAllocation {
            worker_count: config.min_workers,
            cpu_cores_per_worker: 2,
            memory_gb_per_worker: 4,
            total_cpu_cores: config.min_workers * 2,
            total_memory_gb: config.min_workers * 4,
            cost_per_hour: config.min_workers as f64 * 0.50, // $0.50 per worker per hour
            last_updated: Utc::now(),
        };
        let current_resources = Arc::new(Mutex::new(initial_resources));

        let scaling_policies = Self::create_default_policies(&config);

        Self {
            config,
            metrics_collector,
            scaling_decisions,
            current_resources,
            scaling_policies,
        }
    }

    /// Start the auto-scaling monitoring loop
    pub async fn start_monitoring(&self) -> Result<()> {
        log::info!("Starting auto-scaling monitoring");

        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
            self.config.evaluation_interval_seconds,
        ));

        loop {
            interval.tick().await;

            if let Err(e) = self.evaluate_and_scale().await {
                log::error!("Auto-scaling evaluation error: {}", e);
            }
        }
    }

    /// Evaluate current metrics and make scaling decisions
    async fn evaluate_and_scale(&self) -> Result<()> {
        // Collect current metrics
        self.collect_current_metrics().await?;

        // Evaluate scaling policies
        let mut scaling_decision = self.evaluate_scaling_policies()?;

        // Apply predictive scaling if enabled
        if self.config.enable_predictive_scaling {
            scaling_decision = self.apply_predictive_scaling(scaling_decision)?;
        }

        // Apply cost optimization if enabled
        if self.config.enable_cost_optimization {
            scaling_decision = self.apply_cost_optimization(scaling_decision)?;
        }

        // Check cooldown period
        if self.is_in_cooldown_period()? {
            log::debug!("Scaling action skipped due to cooldown period");
            scaling_decision.decision_type = ScalingDecisionType::NoAction;
            scaling_decision.reasoning = "In cooldown period".to_string();
        }

        // Execute scaling decision
        if matches!(
            scaling_decision.decision_type,
            ScalingDecisionType::ScaleUp
                | ScalingDecisionType::ScaleDown
                | ScalingDecisionType::Emergency
        ) {
            self.execute_scaling_decision(&scaling_decision).await?;
        }

        // Record decision
        self.record_scaling_decision(scaling_decision);

        Ok(())
    }

    /// Collect current system metrics
    async fn collect_current_metrics(&self) -> Result<()> {
        let now = Utc::now();

        // In a real implementation, these would be collected from system monitoring
        let cpu_usage = self.get_current_cpu_usage().await;
        let memory_usage = self.get_current_memory_usage().await;
        let request_rate = self.get_current_request_rate().await;
        let response_time = self.get_current_response_time().await;
        let error_rate = self.get_current_error_rate().await;
        let queue_depth = self.get_current_queue_depth().await;

        let mut collector = self.metrics_collector.lock().unwrap();

        collector.add_metric("cpu", cpu_usage, now, "system".to_string());
        collector.add_metric("memory", memory_usage, now, "system".to_string());
        collector.add_metric("requests", request_rate, now, "application".to_string());
        collector.add_metric(
            "response_time",
            response_time,
            now,
            "application".to_string(),
        );
        collector.add_metric("error_rate", error_rate, now, "application".to_string());
        collector.add_metric("queue_depth", queue_depth, now, "application".to_string());

        Ok(())
    }

    /// Evaluate all scaling policies and determine action
    fn evaluate_scaling_policies(&self) -> Result<ScalingDecision> {
        let collector = self.metrics_collector.lock().unwrap();
        let current_resources = self.current_resources.lock().unwrap();

        let mut triggered_policies = Vec::new();

        for policy in &self.scaling_policies {
            if !policy.enabled {
                continue;
            }

            if self.evaluate_condition(&policy.condition, &collector)? {
                triggered_policies.push(policy);
            }
        }

        // Sort by priority and select the highest priority action
        triggered_policies.sort_by_key(|p| p.priority);

        if let Some(policy) = triggered_policies.first() {
            let (decision_type, target_workers) = match &policy.action {
                ScalingAction::ScaleToCount(count) => {
                    if *count > current_resources.worker_count {
                        (ScalingDecisionType::ScaleUp, *count)
                    } else if *count < current_resources.worker_count {
                        (ScalingDecisionType::ScaleDown, *count)
                    } else {
                        (
                            ScalingDecisionType::NoAction,
                            current_resources.worker_count,
                        )
                    }
                }
                ScalingAction::ScaleByCount(delta) => {
                    let new_count = (current_resources.worker_count as i32 + delta)
                        .max(self.config.min_workers as i32)
                        .min(self.config.max_workers as i32)
                        as usize;
                    if new_count > current_resources.worker_count {
                        (ScalingDecisionType::ScaleUp, new_count)
                    } else if new_count < current_resources.worker_count {
                        (ScalingDecisionType::ScaleDown, new_count)
                    } else {
                        (
                            ScalingDecisionType::NoAction,
                            current_resources.worker_count,
                        )
                    }
                }
                ScalingAction::ScaleByPercent(percent) => {
                    let new_count = ((current_resources.worker_count as f64
                        * (1.0 + *percent as f64 / 100.0))
                        .round() as usize)
                        .max(self.config.min_workers)
                        .min(self.config.max_workers);
                    if new_count > current_resources.worker_count {
                        (ScalingDecisionType::ScaleUp, new_count)
                    } else if new_count < current_resources.worker_count {
                        (ScalingDecisionType::ScaleDown, new_count)
                    } else {
                        (
                            ScalingDecisionType::NoAction,
                            current_resources.worker_count,
                        )
                    }
                }
                ScalingAction::Emergency => {
                    (ScalingDecisionType::Emergency, self.config.max_workers)
                }
            };

            let mut trigger_metrics = HashMap::new();
            self.populate_trigger_metrics(&mut trigger_metrics, &collector);

            Ok(ScalingDecision {
                timestamp: Utc::now(),
                decision_type,
                from_workers: current_resources.worker_count,
                to_workers: target_workers,
                trigger_metrics,
                reasoning: format!("Policy '{}' triggered", policy.name),
                confidence_score: 0.8,
                executed: false,
                execution_time_ms: None,
            })
        } else {
            Ok(ScalingDecision {
                timestamp: Utc::now(),
                decision_type: ScalingDecisionType::NoAction,
                from_workers: current_resources.worker_count,
                to_workers: current_resources.worker_count,
                trigger_metrics: HashMap::new(),
                reasoning: "No policies triggered".to_string(),
                confidence_score: 1.0,
                executed: false,
                execution_time_ms: None,
            })
        }
    }

    /// Apply predictive scaling based on historical trends
    fn apply_predictive_scaling(&self, mut decision: ScalingDecision) -> Result<ScalingDecision> {
        let collector = self.metrics_collector.lock().unwrap();

        // Analyze request rate trend over the last hour
        let request_trend =
            collector.calculate_trend(&collector.request_rate_history, Duration::minutes(60));

        if request_trend > 0.2 {
            // 20% increase trend
            // Predict we'll need more capacity soon
            if matches!(decision.decision_type, ScalingDecisionType::NoAction) {
                decision.decision_type = ScalingDecisionType::ScaleUp;
                decision.to_workers = (decision.from_workers + 1).min(self.config.max_workers);
                decision.reasoning = format!(
                    "{} + Predictive scaling (trend: +{:.1}%)",
                    decision.reasoning,
                    request_trend * 100.0
                );
                decision.confidence_score *= 0.7; // Lower confidence for predictive actions
            }
        } else if request_trend < -0.2 {
            // 20% decrease trend
            // We might be able to scale down soon
            if matches!(decision.decision_type, ScalingDecisionType::NoAction) {
                decision.decision_type = ScalingDecisionType::ScaleDown;
                decision.to_workers = (decision.from_workers - 1).max(self.config.min_workers);
                decision.reasoning = format!(
                    "{} + Predictive scaling (trend: {:.1}%)",
                    decision.reasoning,
                    request_trend * 100.0
                );
                decision.confidence_score *= 0.6; // Even lower confidence for predictive scale-down
            }
        }

        Ok(decision)
    }

    /// Apply cost optimization to scaling decisions
    fn apply_cost_optimization(&self, mut decision: ScalingDecision) -> Result<ScalingDecision> {
        let current_resources = self.current_resources.lock().unwrap();
        let current_cost_per_hour = current_resources.cost_per_hour;
        let new_cost_per_hour = decision.to_workers as f64 * 0.50;

        // If cost would increase significantly, apply more conservative scaling
        let cost_increase_percent =
            (new_cost_per_hour - current_cost_per_hour) / current_cost_per_hour;

        if cost_increase_percent > 0.5 {
            // More than 50% cost increase
            match decision.decision_type {
                ScalingDecisionType::ScaleUp => {
                    // Scale up more conservatively
                    decision.to_workers = (decision.from_workers + 1).min(decision.to_workers);
                    decision.reasoning = format!(
                        "{} + Cost optimization (reduced scale-up)",
                        decision.reasoning
                    );
                }
                _ => {}
            }
        } else if cost_increase_percent < -0.3 {
            // More than 30% cost decrease opportunity
            match decision.decision_type {
                ScalingDecisionType::ScaleDown | ScalingDecisionType::NoAction => {
                    // Be more aggressive about scaling down to save cost
                    if decision.to_workers > self.config.min_workers {
                        decision.decision_type = ScalingDecisionType::ScaleDown;
                        decision.reasoning = format!(
                            "{} + Cost optimization (aggressive scale-down)",
                            decision.reasoning
                        );
                    }
                }
                _ => {}
            }
        }

        Ok(decision)
    }

    /// Check if we're in cooldown period after last scaling action
    fn is_in_cooldown_period(&self) -> Result<bool> {
        let decisions = self.scaling_decisions.lock().unwrap();

        if let Some(last_decision) = decisions.iter().find(|d| d.executed) {
            let cooldown_duration = Duration::minutes(self.config.cooldown_period_minutes);
            let time_since_last_action = Utc::now() - last_decision.timestamp;
            Ok(time_since_last_action < cooldown_duration)
        } else {
            Ok(false)
        }
    }

    /// Execute a scaling decision
    async fn execute_scaling_decision(&self, decision: &ScalingDecision) -> Result<()> {
        let start_time = std::time::Instant::now();

        log::info!(
            "Executing scaling decision: {:?} from {} to {} workers",
            decision.decision_type,
            decision.from_workers,
            decision.to_workers
        );

        // In a real implementation, this would interact with container orchestration (K8s, Docker Swarm, etc.)
        self.simulate_scaling_execution(decision).await?;

        // Update current resource allocation
        let execution_time = start_time.elapsed().as_millis() as u64;
        self.update_resource_allocation(decision.to_workers, execution_time)?;

        log::info!(
            "Scaling completed in {}ms. New worker count: {}",
            execution_time,
            decision.to_workers
        );

        Ok(())
    }

    /// Simulate scaling execution (would be real infrastructure calls)
    async fn simulate_scaling_execution(&self, decision: &ScalingDecision) -> Result<()> {
        match decision.decision_type {
            ScalingDecisionType::ScaleUp => {
                let workers_to_add = decision.to_workers - decision.from_workers;
                log::info!("Simulating addition of {} workers", workers_to_add);
                // Simulate time for workers to start up
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    500 * workers_to_add as u64,
                ))
                .await;
            }
            ScalingDecisionType::ScaleDown => {
                let workers_to_remove = decision.from_workers - decision.to_workers;
                log::info!("Simulating removal of {} workers", workers_to_remove);
                // Simulate time for graceful shutdown
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    200 * workers_to_remove as u64,
                ))
                .await;
            }
            ScalingDecisionType::Emergency => {
                log::warn!("Executing emergency scaling to maximum capacity");
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            _ => {}
        }
        Ok(())
    }

    /// Update resource allocation after scaling
    fn update_resource_allocation(
        &self,
        new_worker_count: usize,
        execution_time_ms: u64,
    ) -> Result<()> {
        let mut resources = self.current_resources.lock().unwrap();

        resources.worker_count = new_worker_count;
        resources.total_cpu_cores = new_worker_count * resources.cpu_cores_per_worker;
        resources.total_memory_gb = new_worker_count * resources.memory_gb_per_worker;
        resources.cost_per_hour = new_worker_count as f64 * 0.50;
        resources.last_updated = Utc::now();

        // Update the last decision as executed
        let mut decisions = self.scaling_decisions.lock().unwrap();
        if let Some(last_decision) = decisions.back_mut() {
            last_decision.executed = true;
            last_decision.execution_time_ms = Some(execution_time_ms);
        }

        Ok(())
    }

    /// Record a scaling decision
    fn record_scaling_decision(&self, decision: ScalingDecision) {
        let mut decisions = self.scaling_decisions.lock().unwrap();
        decisions.push_back(decision);

        // Keep only the last 100 decisions
        if decisions.len() > 100 {
            decisions.pop_front();
        }
    }

    /// Create default scaling policies
    fn create_default_policies(config: &ScalingConfig) -> Vec<ScalingPolicy> {
        vec![
            ScalingPolicy {
                name: "High CPU Scale Up".to_string(),
                condition: ScalingCondition::CpuThreshold {
                    threshold: config.scale_up_threshold,
                    duration_seconds: 120,
                },
                action: ScalingAction::ScaleByCount(1),
                priority: 10,
                enabled: true,
            },
            ScalingPolicy {
                name: "Low CPU Scale Down".to_string(),
                condition: ScalingCondition::CpuThreshold {
                    threshold: config.scale_down_threshold,
                    duration_seconds: 300,
                },
                action: ScalingAction::ScaleByCount(-1),
                priority: 5,
                enabled: true,
            },
            ScalingPolicy {
                name: "Emergency High Queue".to_string(),
                condition: ScalingCondition::QueueDepthThreshold {
                    threshold: 1000,
                    duration_seconds: 60,
                },
                action: ScalingAction::Emergency,
                priority: 100,
                enabled: true,
            },
            ScalingPolicy {
                name: "High Error Rate".to_string(),
                condition: ScalingCondition::ErrorRateThreshold {
                    threshold: 0.1, // 10% error rate
                    duration_seconds: 120,
                },
                action: ScalingAction::ScaleByPercent(50),
                priority: 50,
                enabled: true,
            },
        ]
    }

    // Simulation methods for metrics (would be replaced with real monitoring)
    async fn get_current_cpu_usage(&self) -> f64 {
        // Simulate CPU usage based on current worker load
        let resources = self.current_resources.lock().unwrap();
        let base_usage = 30.0;
        let load_factor = (resources.worker_count as f64 / self.config.max_workers as f64) * 40.0;
        (base_usage + load_factor).min(95.0)
    }

    async fn get_current_memory_usage(&self) -> f64 {
        50.0 + (rand::random::<f64>() * 30.0)
    }

    async fn get_current_request_rate(&self) -> f64 {
        100.0 + (rand::random::<f64>() * 200.0)
    }

    async fn get_current_response_time(&self) -> f64 {
        150.0 + (rand::random::<f64>() * 100.0)
    }

    async fn get_current_error_rate(&self) -> f64 {
        0.01 + (rand::random::<f64>() * 0.05)
    }

    async fn get_current_queue_depth(&self) -> f64 {
        10.0 + (rand::random::<f64>() * 50.0)
    }

    fn evaluate_condition(
        &self,
        condition: &ScalingCondition,
        collector: &MetricsCollector,
    ) -> Result<bool> {
        match condition {
            ScalingCondition::CpuThreshold {
                threshold,
                duration_seconds,
            } => Ok(collector.check_threshold(
                &collector.cpu_usage_history,
                *threshold,
                *duration_seconds,
            )),
            ScalingCondition::MemoryThreshold {
                threshold,
                duration_seconds,
            } => Ok(collector.check_threshold(
                &collector.memory_usage_history,
                *threshold,
                *duration_seconds,
            )),
            ScalingCondition::QueueDepthThreshold {
                threshold,
                duration_seconds,
            } => Ok(collector.check_threshold(
                &collector.queue_depth_history,
                *threshold as f64,
                *duration_seconds,
            )),
            ScalingCondition::ResponseTimeThreshold {
                threshold_ms,
                duration_seconds,
            } => Ok(collector.check_threshold(
                &collector.response_time_history,
                *threshold_ms as f64,
                *duration_seconds,
            )),
            ScalingCondition::ErrorRateThreshold {
                threshold,
                duration_seconds,
            } => Ok(collector.check_threshold(
                &collector.error_rate_history,
                *threshold,
                *duration_seconds,
            )),
            _ => Ok(false), // TODO: Implement other conditions
        }
    }

    fn populate_trigger_metrics(
        &self,
        metrics: &mut HashMap<String, f64>,
        collector: &MetricsCollector,
    ) {
        if let Some(cpu) = collector.cpu_usage_history.back() {
            metrics.insert("cpu_usage".to_string(), cpu.value);
        }
        if let Some(memory) = collector.memory_usage_history.back() {
            metrics.insert("memory_usage".to_string(), memory.value);
        }
        if let Some(requests) = collector.request_rate_history.back() {
            metrics.insert("request_rate".to_string(), requests.value);
        }
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            cpu_usage_history: VecDeque::new(),
            memory_usage_history: VecDeque::new(),
            request_rate_history: VecDeque::new(),
            response_time_history: VecDeque::new(),
            error_rate_history: VecDeque::new(),
            queue_depth_history: VecDeque::new(),
            max_history_size: 1000,
        }
    }

    fn add_metric(
        &mut self,
        metric_type: &str,
        value: f64,
        timestamp: DateTime<Utc>,
        source: String,
    ) {
        let point = MetricPoint {
            timestamp,
            value,
            source,
        };

        let history = match metric_type {
            "cpu" => &mut self.cpu_usage_history,
            "memory" => &mut self.memory_usage_history,
            "requests" => &mut self.request_rate_history,
            "response_time" => &mut self.response_time_history,
            "error_rate" => &mut self.error_rate_history,
            "queue_depth" => &mut self.queue_depth_history,
            _ => return,
        };

        history.push_back(point);

        if history.len() > self.max_history_size {
            history.pop_front();
        }
    }

    fn check_threshold(
        &self,
        history: &VecDeque<MetricPoint>,
        threshold: f64,
        duration_seconds: u64,
    ) -> bool {
        let cutoff_time = Utc::now() - Duration::seconds(duration_seconds as i64);

        let recent_values: Vec<f64> = history
            .iter()
            .filter(|point| point.timestamp > cutoff_time)
            .map(|point| point.value)
            .collect();

        if recent_values.is_empty() {
            return false;
        }

        // Check if average over duration exceeds threshold
        let average = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        average > threshold
    }

    fn calculate_trend(&self, history: &VecDeque<MetricPoint>, duration: Duration) -> f64 {
        let cutoff_time = Utc::now() - duration;

        let recent_values: Vec<(i64, f64)> = history
            .iter()
            .filter(|point| point.timestamp > cutoff_time)
            .map(|point| (point.timestamp.timestamp(), point.value))
            .collect();

        if recent_values.len() < 2 {
            return 0.0;
        }

        // Simple linear regression to calculate trend
        let n = recent_values.len() as f64;
        let sum_x: i64 = recent_values.iter().map(|(x, _)| *x).sum();
        let sum_y: f64 = recent_values.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = recent_values.iter().map(|(x, y)| (*x as f64) * y).sum();
        let sum_x2: i64 = recent_values.iter().map(|(x, _)| x * x).sum();

        let slope =
            (n * sum_xy - (sum_x as f64) * sum_y) / (n * (sum_x2 as f64) - (sum_x as f64).powi(2));

        // Normalize to percentage change per hour
        slope / (sum_y / n) * 3600.0
    }
}

/// Get auto-scaling status and recommendations
pub async fn get_scaling_status(scaler: &AutoScaler) -> Result<ScalingStatus> {
    let resources = scaler.current_resources.lock().unwrap();
    let decisions = scaler.scaling_decisions.lock().unwrap();
    let metrics = scaler.metrics_collector.lock().unwrap();

    let recent_decisions: Vec<_> = decisions.iter().rev().take(5).cloned().collect();

    let current_cpu = metrics
        .cpu_usage_history
        .back()
        .map(|m| m.value)
        .unwrap_or(0.0);
    let current_memory = metrics
        .memory_usage_history
        .back()
        .map(|m| m.value)
        .unwrap_or(0.0);

    let mut recommendations = Vec::new();

    if current_cpu > 85.0 {
        recommendations.push("Consider increasing CPU threshold or max workers".to_string());
    }

    if resources.worker_count >= scaler.config.max_workers {
        recommendations.push("At maximum capacity - consider increasing max_workers".to_string());
    }

    Ok(ScalingStatus {
        current_resources: resources.clone(),
        current_cpu_usage: current_cpu,
        current_memory_usage: current_memory,
        recent_decisions,
        recommendations,
        is_healthy: current_cpu < 90.0 && current_memory < 90.0,
        cost_per_day: resources.cost_per_hour * 24.0,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingStatus {
    pub current_resources: ResourceAllocation,
    pub current_cpu_usage: f64,
    pub current_memory_usage: f64,
    pub recent_decisions: Vec<ScalingDecision>,
    pub recommendations: Vec<String>,
    pub is_healthy: bool,
    pub cost_per_day: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_config_default() {
        let config = ScalingConfig::default();
        assert_eq!(config.min_workers, 2);
        assert_eq!(config.max_workers, 20);
        assert_eq!(config.target_cpu_utilization, 70.0);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        let now = Utc::now();

        collector.add_metric("cpu", 75.0, now, "system".to_string());
        collector.add_metric("memory", 60.0, now, "system".to_string());

        assert_eq!(collector.cpu_usage_history.len(), 1);
        assert_eq!(collector.memory_usage_history.len(), 1);

        let threshold_exceeded = collector.check_threshold(&collector.cpu_usage_history, 70.0, 60);
        assert!(threshold_exceeded);
    }

    #[tokio::test]
    async fn test_auto_scaler_creation() {
        let config = ScalingConfig::default();
        let scaler = AutoScaler::new(config);

        let resources = scaler.current_resources.lock().unwrap();
        assert_eq!(resources.worker_count, 2); // min_workers
    }
}
