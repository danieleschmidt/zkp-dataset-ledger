use zkp_dataset_ledger::{Ledger, Dataset};
use tempfile::TempDir;
use std::path::PathBuf;

#[tokio::test]
async fn test_ledger_initialization() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("test_ledger");
    
    let ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
    assert!(ledger_path.exists());
}

#[tokio::test]
async fn test_dataset_notarization() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("test_ledger");
    
    // Create test CSV data
    let test_csv = temp_dir.path().join("test_data.csv");
    std::fs::write(&test_csv, "id,value\n1,100\n2,200\n3,300\n").unwrap();
    
    let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
    let dataset = Dataset::from_path(&test_csv).expect("Failed to load dataset");
    
    let proof = ledger.notarize_dataset(
        dataset,
        "test-dataset-v1",
        Default::default()
    ).expect("Failed to notarize dataset");
    
    assert!(ledger.verify_proof(&proof).expect("Verification failed"));
}

#[tokio::test]
async fn test_audit_trail() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("test_ledger");
    
    let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
    
    // Create and notarize multiple datasets
    for i in 1..=3 {
        let test_csv = temp_dir.path().join(format!("test_data_{}.csv", i));
        std::fs::write(&test_csv, format!("id,value\n1,{}\n", i * 100)).unwrap();
        
        let dataset = Dataset::from_path(&test_csv).expect("Failed to load dataset");
        ledger.notarize_dataset(
            dataset,
            &format!("test-dataset-v{}", i),
            Default::default()
        ).expect("Failed to notarize dataset");
    }
    
    let history = ledger.get_audit_trail().expect("Failed to get audit trail");
    assert_eq!(history.len(), 3);
}