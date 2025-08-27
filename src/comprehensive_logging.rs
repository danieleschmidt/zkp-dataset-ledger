//! Comprehensive Logging - Generation 2 Observability Features
//!
//! Advanced logging, tracing, and observability infrastructure for production
//! ZKP Dataset Ledger operations with structured logging and metrics collection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

// Add rand import for testing functionality
#[cfg(test)]
use rand;

/// Comprehensive logging system with structured events
#[derive(Debug)]
pub struct LoggingSystem {
    config: LoggingConfig,
    event_store: Arc<RwLock<Vec<LogEvent>>>,
    metrics: Arc<RwLock<LoggingMetrics>>,
    contexts: Arc<RwLock<HashMap<String, LogContext>>>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub log_level: LogLevel,
    pub enable_structured_logging: bool,
    pub enable_metrics_collection: bool,
    pub enable_distributed_tracing: bool,
    pub max_events_in_memory: usize,
    pub event_retention_hours: u64,
    pub enable_sensitive_data_masking: bool,
    pub output_formats: Vec<LogOutputFormat>,
    pub sampling_rate: f64, // 0.0 to 1.0
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            enable_structured_logging: true,
            enable_metrics_collection: true,
            enable_distributed_tracing: true,
            max_events_in_memory: 10000,
            event_retention_hours: 24,
            enable_sensitive_data_masking: true,
            output_formats: vec![LogOutputFormat::Json, LogOutputFormat::Console],
            sampling_rate: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Critical => "CRITICAL",
        }
    }

    pub fn severity(&self) -> u8 {
        match self {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Warning => 3,
            LogLevel::Error => 4,
            LogLevel::Critical => 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutputFormat {
    Json,
    Console,
    Structured,
    Metrics,
}

/// Structured log event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    pub id: String,
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub component: String,
    pub operation: String,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub fields: HashMap<String, LogValue>,
    pub tags: HashMap<String, String>,
    pub metrics: HashMap<String, f64>,
    pub duration_ms: Option<u64>,
    pub error_details: Option<ErrorDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<LogValue>),
    Object(HashMap<String, LogValue>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub error_type: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub error_code: Option<String>,
    pub context: HashMap<String, String>,
}

/// Logging context for request/operation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation: String,
    #[serde(skip, default = "default_instant")]
    pub start_time: Instant,
    pub tags: HashMap<String, String>,
    pub baggage: HashMap<String, String>,
}

fn default_instant() -> Instant {
    Instant::now()
}

/// Logging metrics for observability
#[derive(Debug, Default, Clone)]
pub struct LoggingMetrics {
    pub total_events: u64,
    pub events_by_level: HashMap<String, u64>,
    pub events_by_component: HashMap<String, u64>,
    pub errors_by_type: HashMap<String, u64>,
    pub average_event_processing_time_ms: f64,
    pub dropped_events: u64,
    pub sensitive_data_masked: u64,
}

/// Log event builder for fluent API
pub struct LogEventBuilder {
    event: LogEvent,
}

impl LoggingSystem {
    /// Create new logging system
    pub fn new(config: LoggingConfig) -> Self {
        Self {
            config,
            event_store: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(LoggingMetrics::default())),
            contexts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Log structured event
    pub async fn log(&self, event: LogEvent) {
        // Check sampling rate (simplified for non-test builds)
        #[cfg(test)]
        {
            if self.config.sampling_rate < 1.0 && rand::random::<f64>() > self.config.sampling_rate
            {
                return;
            }
        }

        // Check log level
        if event.level.severity() < self.config.log_level.severity() {
            return;
        }

        let start_time = Instant::now();

        // Apply sensitive data masking if enabled
        let mut processed_event = event;
        if self.config.enable_sensitive_data_masking {
            processed_event = self.mask_sensitive_data(processed_event).await;
        }

        // Store event
        if self.config.max_events_in_memory > 0 {
            let mut store = self.event_store.write().await;
            store.push(processed_event.clone());

            // Maintain size limit
            while store.len() > self.config.max_events_in_memory {
                store.remove(0);
            }
        }

        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        let mut metrics = self.metrics.write().await;
        metrics.total_events += 1;

        *metrics
            .events_by_level
            .entry(processed_event.level.as_str().to_string())
            .or_insert(0) += 1;

        *metrics
            .events_by_component
            .entry(processed_event.component.clone())
            .or_insert(0) += 1;

        if let Some(error_details) = &processed_event.error_details {
            *metrics
                .errors_by_type
                .entry(error_details.error_type.clone())
                .or_insert(0) += 1;
        }

        metrics.average_event_processing_time_ms = (metrics.average_event_processing_time_ms
            * (metrics.total_events - 1) as f64
            + processing_time)
            / metrics.total_events as f64;

        // Output to configured formats
        for format in &self.config.output_formats {
            self.output_event(&processed_event, format).await;
        }
    }

    /// Create log event builder
    pub fn event(&self) -> LogEventBuilder {
        LogEventBuilder::new()
    }

    /// Start new trace context
    pub async fn start_trace(&self, operation: &str) -> String {
        let trace_id = uuid::Uuid::new_v4().to_string();
        let span_id = uuid::Uuid::new_v4().to_string();

        let context = LogContext {
            trace_id: trace_id.clone(),
            span_id,
            parent_span_id: None,
            operation: operation.to_string(),
            start_time: Instant::now(),
            tags: HashMap::new(),
            baggage: HashMap::new(),
        };

        let mut contexts = self.contexts.write().await;
        contexts.insert(trace_id.clone(), context);

        trace_id
    }

    /// Start child span
    pub async fn start_span(&self, parent_trace_id: &str, operation: &str) -> Option<String> {
        let contexts = self.contexts.read().await;
        if let Some(parent_context) = contexts.get(parent_trace_id) {
            let span_id = uuid::Uuid::new_v4().to_string();

            let child_context = LogContext {
                trace_id: parent_context.trace_id.clone(),
                span_id: span_id.clone(),
                parent_span_id: Some(parent_context.span_id.clone()),
                operation: operation.to_string(),
                start_time: Instant::now(),
                tags: parent_context.tags.clone(),
                baggage: parent_context.baggage.clone(),
            };

            drop(contexts);
            let mut contexts = self.contexts.write().await;
            contexts.insert(span_id.clone(), child_context);

            Some(span_id)
        } else {
            None
        }
    }

    /// Finish trace/span
    pub async fn finish_span(&self, span_id: &str) -> Option<Duration> {
        let mut contexts = self.contexts.write().await;
        if let Some(context) = contexts.remove(span_id) {
            Some(context.start_time.elapsed())
        } else {
            None
        }
    }

    /// Get current trace context
    pub async fn get_context(&self, trace_id: &str) -> Option<LogContext> {
        let contexts = self.contexts.read().await;
        contexts.get(trace_id).cloned()
    }

    /// Add trace baggage
    pub async fn add_baggage(&self, trace_id: &str, key: &str, value: &str) {
        let mut contexts = self.contexts.write().await;
        if let Some(context) = contexts.get_mut(trace_id) {
            context.baggage.insert(key.to_string(), value.to_string());
        }
    }

    /// Mask sensitive data in log event
    async fn mask_sensitive_data(&self, mut event: LogEvent) -> LogEvent {
        let sensitive_patterns = [
            "password",
            "pwd",
            "secret",
            "token",
            "key",
            "credential",
            "ssn",
            "credit",
            "card",
            "account",
            "email",
        ];

        // Mask message
        for pattern in &sensitive_patterns {
            if event.message.to_lowercase().contains(pattern) {
                event.message = self.apply_masking(&event.message);
            }
        }

        // Mask fields
        for (key, value) in event.fields.iter_mut() {
            if sensitive_patterns
                .iter()
                .any(|p| key.to_lowercase().contains(p))
            {
                *value = LogValue::String("***MASKED***".to_string());

                let mut metrics = self.metrics.write().await;
                metrics.sensitive_data_masked += 1;
            }
        }

        // Mask tags
        for (key, value) in event.tags.iter_mut() {
            if sensitive_patterns
                .iter()
                .any(|p| key.to_lowercase().contains(p))
            {
                *value = "***MASKED***".to_string();
            }
        }

        event
    }

    /// Apply masking pattern
    fn apply_masking(&self, text: &str) -> String {
        // Simple masking - replace middle characters
        if text.len() <= 4 {
            "*".repeat(text.len())
        } else {
            let start = &text[..2];
            let end = &text[text.len() - 2..];
            let middle = "*".repeat(text.len() - 4);
            format!("{}{}{}", start, middle, end)
        }
    }

    /// Output event to specified format
    async fn output_event(&self, event: &LogEvent, format: &LogOutputFormat) {
        match format {
            LogOutputFormat::Json => {
                if let Ok(json) = serde_json::to_string(event) {
                    println!("{}", json);
                }
            }
            LogOutputFormat::Console => {
                let timestamp = event
                    .timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                println!(
                    "[{}] [{}] [{}] {}",
                    timestamp,
                    event.level.as_str(),
                    event.component,
                    event.message
                );
            }
            LogOutputFormat::Structured => {
                println!("{:?}", event);
            }
            LogOutputFormat::Metrics => {
                if !event.metrics.is_empty() {
                    println!("METRICS: {:?}", event.metrics);
                }
            }
        }
    }

    /// Query events by criteria
    pub async fn query_events(&self, query: LogQuery) -> Vec<LogEvent> {
        let events = self.event_store.read().await;
        let mut results = Vec::new();

        for event in events.iter() {
            if self.matches_query(event, &query) {
                results.push(event.clone());
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        results
    }

    /// Check if event matches query
    fn matches_query(&self, event: &LogEvent, query: &LogQuery) -> bool {
        // Level filter
        if let Some(min_level) = &query.min_level {
            if event.level.severity() < min_level.severity() {
                return false;
            }
        }

        // Component filter
        if let Some(component) = &query.component {
            if event.component != *component {
                return false;
            }
        }

        // Operation filter
        if let Some(operation) = &query.operation {
            if event.operation != *operation {
                return false;
            }
        }

        // Time range filter
        if let Some((start, end)) = query.time_range {
            if event.timestamp < start || event.timestamp > end {
                return false;
            }
        }

        // Text search
        if let Some(text) = &query.text_search {
            if !event.message.contains(text) {
                return false;
            }
        }

        // Tag filters
        for (key, value) in &query.tags {
            if event.tags.get(key) != Some(value) {
                return false;
            }
        }

        true
    }

    /// Get logging metrics
    pub async fn get_metrics(&self) -> LoggingMetrics {
        self.metrics.read().await.clone()
    }

    /// Clear old events based on retention policy
    pub async fn cleanup_old_events(&self) {
        let retention_duration = Duration::from_secs(self.config.event_retention_hours * 3600);
        let cutoff_time = SystemTime::now() - retention_duration;

        let mut events = self.event_store.write().await;
        events.retain(|event| event.timestamp > cutoff_time);
    }
}

/// Log query for searching events
#[derive(Debug, Clone)]
pub struct LogQuery {
    pub min_level: Option<LogLevel>,
    pub component: Option<String>,
    pub operation: Option<String>,
    pub time_range: Option<(SystemTime, SystemTime)>,
    pub text_search: Option<String>,
    pub tags: HashMap<String, String>,
    pub limit: Option<usize>,
}

impl Default for LogQuery {
    fn default() -> Self {
        Self {
            min_level: None,
            component: None,
            operation: None,
            time_range: None,
            text_search: None,
            tags: HashMap::new(),
            limit: Some(100),
        }
    }
}

impl LogEventBuilder {
    /// Create new log event builder
    pub fn new() -> Self {
        Self {
            event: LogEvent {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                level: LogLevel::Info,
                message: String::new(),
                component: "unknown".to_string(),
                operation: "unknown".to_string(),
                trace_id: None,
                span_id: None,
                user_id: None,
                session_id: None,
                fields: HashMap::new(),
                tags: HashMap::new(),
                metrics: HashMap::new(),
                duration_ms: None,
                error_details: None,
            },
        }
    }

    /// Set log level
    pub fn level(mut self, level: LogLevel) -> Self {
        self.event.level = level;
        self
    }

    /// Set message
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.event.message = message.into();
        self
    }

    /// Set component
    pub fn component(mut self, component: impl Into<String>) -> Self {
        self.event.component = component.into();
        self
    }

    /// Set operation
    pub fn operation(mut self, operation: impl Into<String>) -> Self {
        self.event.operation = operation.into();
        self
    }

    /// Set trace ID
    pub fn trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.event.trace_id = Some(trace_id.into());
        self
    }

    /// Set span ID
    pub fn span_id(mut self, span_id: impl Into<String>) -> Self {
        self.event.span_id = Some(span_id.into());
        self
    }

    /// Add field
    pub fn field(mut self, key: impl Into<String>, value: LogValue) -> Self {
        self.event.fields.insert(key.into(), value);
        self
    }

    /// Add tag
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.event.tags.insert(key.into(), value.into());
        self
    }

    /// Add metric
    pub fn metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.event.metrics.insert(key.into(), value);
        self
    }

    /// Set duration
    pub fn duration(mut self, duration: Duration) -> Self {
        self.event.duration_ms = Some(duration.as_millis() as u64);
        self
    }

    /// Set error details
    pub fn error(
        mut self,
        error_type: impl Into<String>,
        error_message: impl Into<String>,
    ) -> Self {
        self.event.error_details = Some(ErrorDetails {
            error_type: error_type.into(),
            error_message: error_message.into(),
            stack_trace: None,
            error_code: None,
            context: HashMap::new(),
        });
        self
    }

    /// Build the log event
    pub fn build(self) -> LogEvent {
        self.event
    }
}

impl Default for LogEventBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_logging_system_creation() {
        let config = LoggingConfig::default();
        let logging = LoggingSystem::new(config);

        let metrics = logging.get_metrics().await;
        assert_eq!(metrics.total_events, 0);
    }

    #[tokio::test]
    async fn test_log_event_creation() {
        let event = LogEventBuilder::new()
            .level(LogLevel::Info)
            .message("Test message")
            .component("test")
            .operation("test_op")
            .field("key1", LogValue::String("value1".to_string()))
            .tag("env", "test")
            .metric("duration", 123.45)
            .build();

        assert_eq!(event.level.as_str(), "INFO");
        assert_eq!(event.message, "Test message");
        assert_eq!(event.component, "test");
        assert!(event.fields.contains_key("key1"));
        assert!(event.tags.contains_key("env"));
        assert!(event.metrics.contains_key("duration"));
    }

    #[tokio::test]
    async fn test_log_event_storage() {
        let config = LoggingConfig {
            max_events_in_memory: 5,
            ..LoggingConfig::default()
        };
        let logging = LoggingSystem::new(config);

        // Log several events
        for i in 0..10 {
            let event = LogEventBuilder::new()
                .message(format!("Event {}", i))
                .build();
            logging.log(event).await;
        }

        let events = logging.event_store.read().await;
        assert_eq!(events.len(), 5); // Should be limited by max_events_in_memory
    }

    #[tokio::test]
    async fn test_trace_context_creation() {
        let config = LoggingConfig::default();
        let logging = LoggingSystem::new(config);

        let trace_id = logging.start_trace("test_operation").await;
        assert!(!trace_id.is_empty());

        let context = logging.get_context(&trace_id).await;
        assert!(context.is_some());
        assert_eq!(context.unwrap().operation, "test_operation");
    }

    #[tokio::test]
    async fn test_span_creation() {
        let config = LoggingConfig::default();
        let logging = LoggingSystem::new(config);

        let trace_id = logging.start_trace("parent_operation").await;
        let span_id = logging.start_span(&trace_id, "child_operation").await;

        assert!(span_id.is_some());

        let span_context = logging.get_context(&span_id.unwrap()).await;
        assert!(span_context.is_some());
        assert_eq!(span_context.unwrap().operation, "child_operation");
    }

    #[tokio::test]
    async fn test_sensitive_data_masking() {
        let config = LoggingConfig {
            enable_sensitive_data_masking: true,
            output_formats: vec![], // Disable output for test
            ..LoggingConfig::default()
        };
        let logging = LoggingSystem::new(config);

        let event = LogEventBuilder::new()
            .message("User password is secret123")
            .field("password", LogValue::String("secret123".to_string()))
            .tag("api_key", "abc123xyz")
            .build();

        logging.log(event).await;

        let events = logging.event_store.read().await;
        assert!(!events.is_empty());

        let logged_event = &events[0];
        assert!(logged_event.message.contains("**"));

        if let Some(LogValue::String(masked_password)) = logged_event.fields.get("password") {
            assert_eq!(masked_password, "***MASKED***");
        }

        assert_eq!(logged_event.tags.get("api_key").unwrap(), "***MASKED***");
    }

    #[tokio::test]
    async fn test_log_level_filtering() {
        let config = LoggingConfig {
            log_level: LogLevel::Warning,
            output_formats: vec![],
            ..LoggingConfig::default()
        };
        let logging = LoggingSystem::new(config);

        // Log events at different levels
        logging
            .log(
                LogEventBuilder::new()
                    .level(LogLevel::Debug)
                    .message("Debug")
                    .build(),
            )
            .await;
        logging
            .log(
                LogEventBuilder::new()
                    .level(LogLevel::Info)
                    .message("Info")
                    .build(),
            )
            .await;
        logging
            .log(
                LogEventBuilder::new()
                    .level(LogLevel::Warning)
                    .message("Warning")
                    .build(),
            )
            .await;
        logging
            .log(
                LogEventBuilder::new()
                    .level(LogLevel::Error)
                    .message("Error")
                    .build(),
            )
            .await;

        let events = logging.event_store.read().await;
        assert_eq!(events.len(), 2); // Only Warning and Error should be logged
    }

    #[tokio::test]
    async fn test_event_querying() {
        let config = LoggingConfig {
            output_formats: vec![],
            ..LoggingConfig::default()
        };
        let logging = LoggingSystem::new(config);

        // Log test events
        logging
            .log(
                LogEventBuilder::new()
                    .component("ledger")
                    .operation("notarize")
                    .level(LogLevel::Info)
                    .message("Notarizing dataset")
                    .build(),
            )
            .await;

        logging
            .log(
                LogEventBuilder::new()
                    .component("crypto")
                    .operation("verify")
                    .level(LogLevel::Error)
                    .message("Verification failed")
                    .build(),
            )
            .await;

        // Query by component
        let query = LogQuery {
            component: Some("ledger".to_string()),
            ..LogQuery::default()
        };
        let results = logging.query_events(query).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].component, "ledger");

        // Query by level
        let query = LogQuery {
            min_level: Some(LogLevel::Error),
            ..LogQuery::default()
        };
        let results = logging.query_events(query).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].level.as_str(), "ERROR");
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = LoggingConfig {
            output_formats: vec![],
            ..LoggingConfig::default()
        };
        let logging = LoggingSystem::new(config);

        // Log various events
        logging
            .log(
                LogEventBuilder::new()
                    .level(LogLevel::Info)
                    .component("test1")
                    .build(),
            )
            .await;
        logging
            .log(
                LogEventBuilder::new()
                    .level(LogLevel::Error)
                    .component("test2")
                    .error("TestError", "Test error")
                    .build(),
            )
            .await;

        let metrics = logging.get_metrics().await;
        assert_eq!(metrics.total_events, 2);
        assert_eq!(metrics.events_by_level.get("INFO"), Some(&1));
        assert_eq!(metrics.events_by_level.get("ERROR"), Some(&1));
        assert_eq!(metrics.events_by_component.get("test1"), Some(&1));
        assert_eq!(metrics.events_by_component.get("test2"), Some(&1));
        assert_eq!(metrics.errors_by_type.get("TestError"), Some(&1));
    }
}
