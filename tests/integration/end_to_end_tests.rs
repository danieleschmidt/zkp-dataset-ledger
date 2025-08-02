// End-to-end integration tests
use zkp_dataset_ledger::{Ledger, Dataset, LedgerConfig, ProofConfig};
use crate::fixtures::{TestLedger, TestDataGenerator, PerformanceTester, assertions, constants};
use std::path::PathBuf;
use tempfile::TempDir;

/// Test complete workflow from dataset creation to audit report
#[tokio::test]
async fn test_complete_ml_pipeline_workflow() {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    
    // Step 1: Notarize raw dataset
    let raw_data_path = generator.create_medium_csv("raw_data", 1000);
    let raw_dataset = Dataset::from_path(&raw_data_path)
        .expect("Failed to load raw dataset");
    
    let raw_proof = ledger.notarize_dataset(
        raw_dataset,
        "raw-customer-data-v1",
        ProofConfig::default(),
    ).expect("Failed to notarize raw dataset");
    
    assertions::assert_proof_valid(&raw_proof);
    
    // Step 2: Record data cleaning transformation
    let cleaned_data_path = generator.create_medium_csv("cleaned_data", 950); // Some rows removed
    let cleaned_dataset = Dataset::from_path(&cleaned_data_path)
        .expect("Failed to load cleaned dataset");
    
    let transform_proof = ledger.record_transformation(
        "raw-customer-data-v1",
        "cleaned-customer-data-v1",
        cleaned_dataset,
        vec!["remove_nulls", "deduplicate", "validate_schema"],
        ProofConfig::default(),
    ).expect("Failed to record transformation");
    
    assertions::assert_proof_valid(&transform_proof);
    
    // Step 3: Create train/test split
    let train_data_path = generator.create_medium_csv("train_data", 760); // 80% of 950
    let test_data_path = generator.create_medium_csv("test_data", 190);  // 20% of 950
    
    let train_dataset = Dataset::from_path(&train_data_path)
        .expect("Failed to load train dataset");
    let test_dataset = Dataset::from_path(&test_data_path)
        .expect("Failed to load test dataset");
    
    let split_proof = ledger.record_split(
        "cleaned-customer-data-v1",
        vec![
            ("train-data-v1".to_string(), train_dataset),
            ("test-data-v1".to_string(), test_dataset),
        ],
        0.8, // 80/20 split
        Some(42), // Random seed for reproducibility
        ProofConfig::default(),
    ).expect("Failed to record split");
    
    assertions::assert_proof_valid(&split_proof);
    
    // Step 4: Generate comprehensive audit report
    let audit_report = ledger.generate_audit_report(
        Some("raw-customer-data-v1".to_string()), // Start from raw data
        None, // Include all operations
        true, // Include proofs
        "json-ld".to_string(),
    ).expect("Failed to generate audit report");
    
    // Verify audit report contents
    assert!(!audit_report.is_empty(), "Audit report should not be empty");
    assert!(audit_report.contains("raw-customer-data-v1"));
    assert!(audit_report.contains("cleaned-customer-data-v1"));
    assert!(audit_report.contains("train-data-v1"));
    assert!(audit_report.contains("test-data-v1"));
    
    // Step 5: Verify chain integrity
    assertions::assert_ledger_integrity(&ledger);
    
    // Step 6: Test proof verification
    assert!(ledger.verify_proof(&raw_proof).expect("Raw proof verification failed"));
    assert!(ledger.verify_proof(&transform_proof).expect("Transform proof verification failed"));
    assert!(ledger.verify_proof(&split_proof).expect("Split proof verification failed"));
}

/// Test concurrent operations on the same ledger
#[tokio::test]
async fn test_concurrent_ledger_operations() {
    let test_ledger = TestLedger::new();
    let ledger_path = test_ledger.path().to_path_buf();
    let generator = TestDataGenerator::new();
    
    // Create initial ledger
    {
        let _ledger = Ledger::new(&ledger_path, LedgerConfig::default())
            .expect("Failed to initialize ledger");
    }
    
    // Spawn concurrent tasks
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let ledger_path = ledger_path.clone();
        let dataset_path = generator.create_small_csv(&format!("concurrent_{}", i));
        
        let handle = tokio::spawn(async move {
            let mut ledger = Ledger::open(&ledger_path, LedgerConfig::default())
                .expect("Failed to open ledger");
            
            let dataset = Dataset::from_path(&dataset_path)
                .expect("Failed to load dataset");
            
            ledger.notarize_dataset(
                dataset,
                &format!("concurrent-dataset-{}", i),
                ProofConfig::default(),
            ).expect("Failed to notarize dataset")
        });
        
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    let mut proofs = Vec::new();
    for handle in handles {
        let proof = handle.await.expect("Task failed");
        proofs.push(proof);
    }
    
    // Verify all proofs
    let ledger = Ledger::open(&ledger_path, LedgerConfig::default())
        .expect("Failed to open final ledger");
    
    for proof in proofs {
        assert!(ledger.verify_proof(&proof).expect("Concurrent proof verification failed"));
    }
    
    assertions::assert_ledger_integrity(&ledger);
}

/// Test large dataset processing and streaming
#[tokio::test]
async fn test_large_dataset_processing() {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    let large_dataset_path = generator.create_medium_csv("large_dataset", constants::LARGE_DATASET_ROWS);
    
    let timer = PerformanceTester::new();
    
    let dataset = Dataset::from_path(&large_dataset_path)
        .expect("Failed to load large dataset");
    
    let proof = ledger.notarize_dataset(
        dataset,
        "large-dataset-v1",
        ProofConfig {
            streaming: true,
            chunk_size: Some(10_000),
            ..Default::default()
        },
    ).expect("Failed to notarize large dataset");
    
    // Should complete within reasonable time
    timer.assert_under_secs(30.0, "Large dataset notarization");
    
    assertions::assert_proof_valid(&proof);
    assert!(ledger.verify_proof(&proof).expect("Large dataset proof verification failed"));
}

/// Test error handling and recovery scenarios
#[tokio::test]
async fn test_error_handling_and_recovery() {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    
    // Test 1: Invalid dataset format
    let malformed_path = generator.create_malformed_csv("malformed");
    let result = Dataset::from_path(&malformed_path);
    assert!(result.is_err(), "Should reject malformed dataset");
    
    // Test 2: Binary file as dataset
    let binary_path = generator.create_binary_file("binary");
    let result = Dataset::from_path(&binary_path);
    assert!(result.is_err(), "Should reject binary file as dataset");
    
    // Test 3: Nonexistent file
    let nonexistent_path = PathBuf::from("/nonexistent/file.csv");
    let result = Dataset::from_path(&nonexistent_path);
    assert!(result.is_err(), "Should handle nonexistent file gracefully");
    
    // Test 4: Duplicate dataset name
    let valid_path = generator.create_small_csv("valid");
    let dataset = Dataset::from_path(&valid_path)
        .expect("Failed to load valid dataset");
    
    // First notarization should succeed
    let _proof1 = ledger.notarize_dataset(
        dataset.clone(),
        "duplicate-name",
        ProofConfig::default(),
    ).expect("First notarization should succeed");
    
    // Second notarization with same name should fail
    let result = ledger.notarize_dataset(
        dataset,
        "duplicate-name",
        ProofConfig::default(),
    );
    assert!(result.is_err(), "Should reject duplicate dataset name");
    
    // Ledger should still be in valid state
    assertions::assert_ledger_integrity(&ledger);
}

/// Test different export formats and compatibility
#[tokio::test]
async fn test_export_formats_and_compatibility() {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_small_csv("export_test");
    let dataset = Dataset::from_path(&dataset_path)
        .expect("Failed to load dataset");
    
    let _proof = ledger.notarize_dataset(
        dataset,
        "export-test-v1",
        ProofConfig::default(),
    ).expect("Failed to notarize dataset");
    
    // Test JSON-LD export
    let json_ld_report = ledger.generate_audit_report(
        None, None, true, "json-ld".to_string(),
    ).expect("Failed to generate JSON-LD report");
    
    // Validate JSON-LD structure
    let json_value: serde_json::Value = serde_json::from_str(&json_ld_report)
        .expect("JSON-LD report should be valid JSON");
    assert!(json_value.get("@context").is_some(), "Should have @context field");
    
    // Test JSON export
    let json_report = ledger.generate_audit_report(
        None, None, true, "json".to_string(),
    ).expect("Failed to generate JSON report");
    
    let json_value: serde_json::Value = serde_json::from_str(&json_report)
        .expect("JSON report should be valid JSON");
    assert!(json_value.get("datasets").is_some(), "Should have datasets field");
    
    // Test HTML export (if implemented)
    let html_result = ledger.generate_audit_report(
        None, None, true, "html".to_string(),
    );
    
    match html_result {
        Ok(html_report) => {
            assert!(html_report.contains("<html>"), "Should contain HTML tags");
            assert!(html_report.contains("export-test-v1"), "Should contain dataset name");
        }
        Err(_) => {
            // HTML export might not be implemented yet
            println!("HTML export not yet implemented");
        }
    }
}

/// Test integration with external storage backends
#[tokio::test]
async fn test_storage_backend_integration() {
    // Test RocksDB backend (default)
    let test_ledger_rocks = TestLedger::with_name("rocks_test");
    let rocks_config = LedgerConfig {
        storage_backend: "rocksdb".to_string(),
        ..Default::default()
    };
    
    let mut rocks_ledger = Ledger::new(test_ledger_rocks.path(), rocks_config)
        .expect("Failed to create RocksDB ledger");
    
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_small_csv("rocks_test");
    let dataset = Dataset::from_path(&dataset_path)
        .expect("Failed to load dataset");
    
    let rocks_proof = rocks_ledger.notarize_dataset(
        dataset,
        "rocks-test-v1",
        ProofConfig::default(),
    ).expect("Failed to notarize with RocksDB");
    
    assertions::assert_proof_valid(&rocks_proof);
    
    // Test PostgreSQL backend (if available)
    // This test requires a running PostgreSQL instance
    if let Ok(db_url) = std::env::var("TEST_DATABASE_URL") {
        let test_ledger_pg = TestLedger::with_name("pg_test");
        let pg_config = LedgerConfig {
            storage_backend: "postgres".to_string(),
            postgres_connection_string: Some(db_url),
            ..Default::default()
        };
        
        let pg_ledger_result = Ledger::new(test_ledger_pg.path(), pg_config);
        
        match pg_ledger_result {
            Ok(mut pg_ledger) => {
                let dataset_path = generator.create_small_csv("pg_test");
                let dataset = Dataset::from_path(&dataset_path)
                    .expect("Failed to load dataset");
                
                let pg_proof = pg_ledger.notarize_dataset(
                    dataset,
                    "pg-test-v1",
                    ProofConfig::default(),
                ).expect("Failed to notarize with PostgreSQL");
                
                assertions::assert_proof_valid(&pg_proof);
            }
            Err(_) => {
                println!("PostgreSQL test skipped - database not available");
            }
        }
    } else {
        println!("PostgreSQL test skipped - TEST_DATABASE_URL not set");
    }
}

/// Test performance under load
#[tokio::test]
async fn test_performance_under_load() {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    let timer = PerformanceTester::new();
    
    // Process multiple datasets quickly
    let mut proofs = Vec::new();
    for i in 0..10 {
        let dataset_path = generator.create_small_csv(&format!("load_test_{}", i));
        let dataset = Dataset::from_path(&dataset_path)
            .expect("Failed to load dataset");
        
        let proof = ledger.notarize_dataset(
            dataset,
            &format!("load-test-{}", i),
            ProofConfig::default(),
        ).expect("Failed to notarize dataset under load");
        
        proofs.push(proof);
    }
    
    // Should complete all operations within reasonable time
    timer.assert_under_secs(60.0, "Batch processing of 10 datasets");
    
    // Verify all proofs
    for proof in proofs {
        assert!(ledger.verify_proof(&proof).expect("Load test proof verification failed"));
    }
    
    assertions::assert_ledger_integrity(&ledger);
}