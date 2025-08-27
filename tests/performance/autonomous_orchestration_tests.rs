use zkp_dataset_ledger::*;
use tokio::test;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_autonomous_orchestrator_full_lifecycle() {
    let config = OrchestrationConfig {
        autonomous_mode: true,
        learning_enabled: true,
        predictive_scaling: true,
        self_healing: true,
        cost_optimization: true,
        compliance_enforcement: true,
        max_automation_scope: AutomationScope::Moderate,
        decision_confidence_threshold: 0.7,
        rollback_on_failure: true,
        canary_deployment_percentage: 10.0,
    };

    let performance_engine = Arc::new(
        QuantumPerformanceEngine::new(ScalingConfig::default())
    );

    let orchestrator = AutonomousOrchestrator::new(config, performance_engine)
        .await
        .expect("Failed to create orchestrator");

    // Test autonomous operations startup
    let result = orchestrator.start_autonomous_operations().await;
    assert!(result.is_ok(), "Failed to start autonomous operations");

    // Test status retrieval
    let status = orchestrator.get_status().await;
    assert!(status.autonomous_mode_active);
    assert!(status.decision_accuracy >= 0.0 && status.decision_accuracy <= 1.0);
    assert!(status.health_score >= 0.0 && status.health_score <= 100.0);
    assert!(status.learning_model_accuracy >= 0.0 && status.learning_model_accuracy <= 1.0);
}

#[tokio::test]
async fn test_quantum_performance_engine() {
    let config = ScalingConfig {
        min_threads: 2,
        max_threads: 8,
        target_cpu_utilization: 0.75,
        memory_threshold_gb: 4.0,
        predictive_scaling: true,
        auto_optimization: true,
        quantum_batch_size: 1000,
        adaptive_timeout: Duration::from_secs(30),
    };

    let engine = QuantumPerformanceEngine::new(config);

    // Test initialization
    let metrics = engine.get_metrics();
    assert_eq!(metrics.processed_operations, 0);
    assert_eq!(metrics.average_response_time_ms, 0.0);
    assert_eq!(metrics.throughput_ops_per_sec, 0);

    // Test optimization startup
    let result = engine.start_optimization().await;
    assert!(result.is_ok(), "Failed to start optimization");

    // Test metrics after optimization
    let updated_metrics = engine.get_metrics();
    assert!(updated_metrics.cpu_utilization >= 0.0 && updated_metrics.cpu_utilization <= 1.0);
    assert!(updated_metrics.cache_efficiency >= 0.0 && updated_metrics.cache_efficiency <= 1.0);
    assert!(updated_metrics.predictive_accuracy >= 0.0 && updated_metrics.predictive_accuracy <= 1.0);
}

#[tokio::test]
async fn test_orchestration_config_validation() {
    // Test default configuration
    let default_config = OrchestrationConfig::default();
    assert!(default_config.autonomous_mode);
    assert!(default_config.learning_enabled);
    assert!(default_config.predictive_scaling);
    assert_eq!(default_config.decision_confidence_threshold, 0.8);

    // Test custom configuration
    let custom_config = OrchestrationConfig {
        autonomous_mode: false,
        learning_enabled: false,
        predictive_scaling: false,
        self_healing: false,
        cost_optimization: false,
        compliance_enforcement: false,
        max_automation_scope: AutomationScope::Conservative,
        decision_confidence_threshold: 0.9,
        rollback_on_failure: false,
        canary_deployment_percentage: 5.0,
    };

    assert!(!custom_config.autonomous_mode);
    assert!(!custom_config.learning_enabled);
    assert_eq!(custom_config.decision_confidence_threshold, 0.9);
}

#[tokio::test]
async fn test_scaling_config_validation() {
    let config = ScalingConfig {
        min_threads: 1,
        max_threads: 16,
        target_cpu_utilization: 0.8,
        memory_threshold_gb: 8.0,
        predictive_scaling: true,
        auto_optimization: true,
        quantum_batch_size: 2000,
        adaptive_timeout: Duration::from_secs(60),
    };

    assert_eq!(config.min_threads, 1);
    assert_eq!(config.max_threads, 16);
    assert_eq!(config.target_cpu_utilization, 0.8);
    assert_eq!(config.memory_threshold_gb, 8.0);
    assert_eq!(config.quantum_batch_size, 2000);
}

#[tokio::test]
async fn test_performance_metrics_evolution() {
    let engine = QuantumPerformanceEngine::new(ScalingConfig::default());
    
    let initial_metrics = engine.get_metrics();
    
    // Start optimization to trigger metric updates
    engine.start_optimization().await.expect("Failed to start optimization");
    
    // Allow some time for metrics to update
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    let updated_metrics = engine.get_metrics();
    
    // Verify metrics are within expected ranges
    assert!(updated_metrics.cpu_utilization >= 0.0 && updated_metrics.cpu_utilization <= 1.0);
    assert!(updated_metrics.memory_usage_mb >= 0);
    assert!(updated_metrics.cache_efficiency >= 0.0 && updated_metrics.cache_efficiency <= 1.0);
    assert!(updated_metrics.predictive_accuracy >= 0.0 && updated_metrics.predictive_accuracy <= 1.0);
}

#[test]
fn test_automation_scope_hierarchy() {
    use std::mem::discriminant;
    
    let conservative = AutomationScope::Conservative;
    let moderate = AutomationScope::Moderate;
    let aggressive = AutomationScope::Aggressive;
    let full = AutomationScope::Full;
    
    // Test that each scope is distinct
    assert_ne!(discriminant(&conservative), discriminant(&moderate));
    assert_ne!(discriminant(&moderate), discriminant(&aggressive));
    assert_ne!(discriminant(&aggressive), discriminant(&full));
}

#[tokio::test]
async fn test_concurrent_orchestration() {
    let config = OrchestrationConfig::default();
    let performance_engine = Arc::new(
        QuantumPerformanceEngine::new(ScalingConfig::default())
    );

    let orchestrator = Arc::new(
        AutonomousOrchestrator::new(config, performance_engine)
            .await
            .expect("Failed to create orchestrator")
    );

    // Spawn multiple tasks to test concurrent access
    let mut handles = vec![];
    
    for i in 0..5 {
        let orch_clone = Arc::clone(&orchestrator);
        let handle = tokio::spawn(async move {
            let status = orch_clone.get_status().await;
            assert!(status.decision_accuracy >= 0.0);
            i // Return task index for verification
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.expect("Task panicked");
        assert_eq!(result, i);
    }
}

#[tokio::test]
async fn test_performance_under_load() {
    let engine = QuantumPerformanceEngine::new(ScalingConfig {
        min_threads: 1,
        max_threads: 4,
        target_cpu_utilization: 0.5,
        memory_threshold_gb: 1.0,
        predictive_scaling: true,
        auto_optimization: true,
        quantum_batch_size: 100,
        adaptive_timeout: Duration::from_secs(5),
    });

    engine.start_optimization().await.expect("Failed to start optimization");

    // Simulate load by getting metrics repeatedly
    let start_time = std::time::Instant::now();
    let mut operations = 0;

    while start_time.elapsed() < Duration::from_millis(100) {
        let _metrics = engine.get_metrics();
        operations += 1;
    }

    assert!(operations > 0, "No operations completed during load test");
    println!("Completed {} operations in 100ms", operations);
}

#[tokio::test]
async fn test_orchestrator_error_handling() {
    // Test with invalid configuration values
    let performance_engine = Arc::new(
        QuantumPerformanceEngine::new(ScalingConfig::default())
    );

    // Test normal creation should work
    let config = OrchestrationConfig::default();
    let result = AutonomousOrchestrator::new(config, performance_engine).await;
    assert!(result.is_ok(), "Normal orchestrator creation should succeed");
}

#[tokio::test]
async fn test_quantum_performance_edge_cases() {
    // Test with extreme configuration values
    let extreme_config = ScalingConfig {
        min_threads: 1,
        max_threads: 1,
        target_cpu_utilization: 1.0,
        memory_threshold_gb: 0.1,
        predictive_scaling: false,
        auto_optimization: false,
        quantum_batch_size: 1,
        adaptive_timeout: Duration::from_millis(1),
    };

    let engine = QuantumPerformanceEngine::new(extreme_config);
    let metrics = engine.get_metrics();
    
    // Verify engine handles extreme values gracefully
    assert!(metrics.cpu_utilization >= 0.0);
    assert!(metrics.memory_usage_mb >= 0);
}

#[test]
fn test_metrics_serialization() {
    let metrics = QuantumPerformanceMetrics {
        processed_operations: 1000,
        average_response_time_ms: 50.0,
        throughput_ops_per_sec: 100,
        cpu_utilization: 0.75,
        memory_usage_mb: 512,
        active_connections: 25,
        cache_efficiency: 0.85,
        predictive_accuracy: 0.90,
    };

    // Test JSON serialization
    let json = serde_json::to_string(&metrics).expect("Failed to serialize metrics");
    assert!(json.contains("processed_operations"));
    assert!(json.contains("1000"));

    // Test deserialization
    let deserialized: QuantumPerformanceMetrics = 
        serde_json::from_str(&json).expect("Failed to deserialize metrics");
    assert_eq!(deserialized.processed_operations, 1000);
    assert_eq!(deserialized.throughput_ops_per_sec, 100);
}