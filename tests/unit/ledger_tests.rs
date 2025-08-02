// Unit tests for ledger core functionality
use zkp_dataset_ledger::{Ledger, Dataset, LedgerConfig, ProofConfig};
use crate::fixtures::{TestLedger, TestDataGenerator, PerformanceTester, assertions, constants};
use tempfile::TempDir;

#[tokio::test]
async fn test_ledger_initialization() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig::default();
    
    let ledger = Ledger::new(test_ledger.path(), config)
        .expect("Failed to initialize ledger");
    
    assert!(test_ledger.path().exists());
    assertions::assert_ledger_integrity(&ledger);
}

#[tokio::test]
async fn test_ledger_initialization_with_custom_config() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig {
        hash_algorithm: "blake3".to_string(),
        compression: false,
        ..Default::default()
    };
    
    let ledger = Ledger::new(test_ledger.path(), config)
        .expect("Failed to initialize ledger with custom config");
    
    assert!(test_ledger.path().exists());
    assertions::assert_ledger_integrity(&ledger);
}

#[tokio::test]
async fn test_ledger_initialization_performance() {
    let test_ledger = TestLedger::new();
    let timer = PerformanceTester::new();
    
    let _ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    timer.assert_under_ms(constants::MAX_LEDGER_INIT_TIME_MS, "Ledger initialization");
}

#[tokio::test]
async fn test_ledger_reopen() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig::default();
    
    // Create and close ledger
    {
        let _ledger = Ledger::new(test_ledger.path(), config.clone())
            .expect("Failed to initialize ledger");
    }
    
    // Reopen existing ledger
    let ledger = Ledger::open(test_ledger.path(), config)
        .expect("Failed to reopen ledger");
    
    assertions::assert_ledger_integrity(&ledger);
}

#[tokio::test]
async fn test_ledger_with_invalid_path() {
    let invalid_path = "/invalid/nonexistent/path";
    let config = LedgerConfig::default();
    
    let result = Ledger::new(invalid_path, config);
    assert!(result.is_err(), "Should fail with invalid path");
}

#[tokio::test]
async fn test_ledger_concurrent_access() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig::default();
    
    let ledger1 = Ledger::new(test_ledger.path(), config.clone())
        .expect("Failed to create first ledger");
    
    // Second access should handle concurrent access gracefully
    let result = Ledger::open(test_ledger.path(), config);
    // Depending on implementation, this might succeed or fail with specific error
    // Adjust assertion based on actual concurrent access behavior
}

#[tokio::test]
async fn test_ledger_storage_backend_switching() {
    let test_ledger = TestLedger::new();
    
    // Create ledger with RocksDB
    let config_rocks = LedgerConfig {
        storage_backend: "rocksdb".to_string(),
        ..Default::default()
    };
    
    let _ledger = Ledger::new(test_ledger.path(), config_rocks)
        .expect("Failed to create RocksDB ledger");
    
    // Note: PostgreSQL backend test would require actual PostgreSQL instance
    // This is more appropriate for integration tests
}

#[tokio::test]
async fn test_ledger_configuration_validation() {
    let test_ledger = TestLedger::new();
    
    // Test invalid hash algorithm
    let invalid_config = LedgerConfig {
        hash_algorithm: "invalid_algorithm".to_string(),
        ..Default::default()
    };
    
    let result = Ledger::new(test_ledger.path(), invalid_config);
    assert!(result.is_err(), "Should reject invalid hash algorithm");
}

#[tokio::test]
async fn test_ledger_metadata_persistence() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig {
        name: "test-ledger".to_string(),
        ..Default::default()
    };
    
    // Create ledger with metadata
    {
        let _ledger = Ledger::new(test_ledger.path(), config.clone())
            .expect("Failed to create ledger");
    }
    
    // Reopen and verify metadata
    let ledger = Ledger::open(test_ledger.path(), config)
        .expect("Failed to reopen ledger");
    
    assert_eq!(ledger.name(), "test-ledger");
}

#[tokio::test]
async fn test_ledger_size_limits() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig {
        max_size_bytes: 1024, // Very small limit for testing
        ..Default::default()
    };
    
    let mut ledger = Ledger::new(test_ledger.path(), config)
        .expect("Failed to create ledger");
    
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_medium_csv("large", 1000);
    let dataset = Dataset::from_path(&dataset_path)
        .expect("Failed to load dataset");
    
    // This should eventually hit size limits
    let mut operations = 0;
    loop {
        let result = ledger.notarize_dataset(
            dataset.clone(),
            &format!("dataset-{}", operations),
            ProofConfig::default(),
        );
        
        match result {
            Ok(_) => operations += 1,
            Err(_) => break, // Hit size limit
        }
        
        if operations > 100 {
            panic!("Size limit not enforced");
        }
    }
    
    assert!(operations > 0, "Should allow at least one operation");
}

#[tokio::test]
async fn test_ledger_recovery_from_corruption() {
    let test_ledger = TestLedger::new();
    let config = LedgerConfig::default();
    
    // Create ledger
    {
        let _ledger = Ledger::new(test_ledger.path(), config.clone())
            .expect("Failed to create ledger");
    }
    
    // Simulate corruption by writing invalid data
    let corrupt_file = test_ledger.path().join("corrupt_data");
    std::fs::write(&corrupt_file, b"invalid_data")
        .expect("Failed to write corrupt data");
    
    // Attempt to open corrupted ledger
    let result = Ledger::open(test_ledger.path(), config);
    
    // Behavior depends on implementation - might recover, fail, or repair
    match result {
        Ok(ledger) => {
            // If recovery succeeds, integrity should be maintained
            assertions::assert_ledger_integrity(&ledger);
        }
        Err(e) => {
            // If recovery fails, error should be informative
            assert!(!e.to_string().is_empty());
        }
    }
}