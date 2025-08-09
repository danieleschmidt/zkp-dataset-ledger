//! End-to-end workflow tests for typical ML pipeline scenarios

use std::io::Write;
use tempfile::NamedTempFile;
use zkp_dataset_ledger::{Dataset, Ledger, ProofConfig};

#[test]
fn test_complete_ml_pipeline_workflow() {
    // Test a complete ML pipeline workflow:
    // 1. Load original dataset
    // 2. Record data preprocessing transformations
    // 3. Record train/test split
    // 4. Verify chain integrity throughout

    let mut ledger = Ledger::new("ml-pipeline-test").unwrap();

    // Step 1: Create and notarize original dataset
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "feature1,feature2,target").unwrap();
    for i in 1..=100 {
        writeln!(temp_file, "{},{},{}", i, i * 2, i % 2).unwrap();
    }

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let original_dataset = Dataset::from_path(&temp_path).unwrap();
    let proof1 = ledger
        .notarize_dataset(
            original_dataset.clone(),
            "raw-data-v1",
            ProofConfig::default(),
        )
        .unwrap();

    assert!(proof1.verify().unwrap());
    assert_eq!(proof1.dataset_hash, original_dataset.compute_hash());

    // Step 2: Record preprocessing transformation
    let preprocessing_proof =
        zkp_dataset_ledger::proof::Proof::generate(&original_dataset, &ProofConfig::default())
            .unwrap();

    let mut params = std::collections::HashMap::new();
    params.insert("method".to_string(), "standardization".to_string());
    params.insert("features".to_string(), "feature1,feature2".to_string());

    let transform_id = ledger
        .record_transformation(
            "raw-data-v1",
            "preprocessed-data-v1",
            "standardize_features",
            params,
            preprocessing_proof,
        )
        .unwrap();

    assert!(!transform_id.is_empty());

    // Step 3: Record train/test split
    let split_proof =
        zkp_dataset_ledger::proof::Proof::generate(&original_dataset, &ProofConfig::default())
            .unwrap();

    let split_id = ledger
        .record_split("preprocessed-data-v1", 0.8, Some(42), None, split_proof)
        .unwrap();

    assert!(!split_id.is_empty());

    // Step 4: Verify complete chain integrity
    let is_chain_valid = ledger.verify_chain_integrity().unwrap();
    assert!(is_chain_valid, "Chain integrity should be valid");

    // Step 5: Check audit trail
    let history = ledger.get_dataset_history("preprocessed-data-v1").unwrap();
    assert_eq!(history.len(), 1, "Should have one transformation entry");

    let summary = ledger.get_summary().unwrap();
    assert_eq!(summary.total_entries, 3, "Should have 3 total entries");
    assert_eq!(
        summary.datasets_tracked, 2,
        "Should track 2 unique datasets"
    );
    assert!(summary.operations_by_type.contains_key("notarize"));
    assert!(summary.operations_by_type.contains_key("transform"));
    assert!(summary.operations_by_type.contains_key("split"));

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_proof_verification_workflow() {
    // Test proof generation and verification workflow
    let mut ledger = Ledger::new("proof-test").unwrap();

    // Create test dataset
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "id,value").unwrap();
    writeln!(temp_file, "1,10").unwrap();
    writeln!(temp_file, "2,20").unwrap();

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let dataset = Dataset::from_path(&temp_path).unwrap();

    // Test different proof types
    let proof_types = vec![
        zkp_dataset_ledger::proof::ProofType::DatasetIntegrity,
        zkp_dataset_ledger::proof::ProofType::RowCount,
        zkp_dataset_ledger::proof::ProofType::Schema,
    ];

    for proof_type in proof_types {
        let config = ProofConfig {
            proof_type: proof_type.clone(),
            include_merkle_proof: true,
            ..ProofConfig::default()
        };

        let proof = ledger
            .notarize_dataset(
                dataset.clone(),
                &format!("dataset-{:?}", proof_type),
                config,
            )
            .unwrap();

        // Verify proof
        assert!(proof.verify().unwrap(), "Proof should be valid");
        assert_eq!(proof.proof_type, proof_type);
        assert!(proof.merkle_proof.is_some(), "Should have Merkle proof");

        // Test JSON serialization/deserialization
        let proof_json = proof.to_json().unwrap();
        let deserialized_proof = zkp_dataset_ledger::proof::Proof::from_json(&proof_json).unwrap();
        assert_eq!(proof.dataset_hash, deserialized_proof.dataset_hash);
        assert_eq!(proof.proof_type, deserialized_proof.proof_type);
    }

    // Verify chain integrity after all operations
    assert!(ledger.verify_chain_integrity().unwrap());

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_multiple_datasets_tracking() {
    let mut ledger = Ledger::new("multi-dataset-test").unwrap();
    let mut dataset_paths = Vec::new();

    // Create multiple test datasets
    for i in 1..=5 {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "id,data").unwrap();
        for j in 1..=10 {
            writeln!(temp_file, "{},{}", j, j * i).unwrap();
        }

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();
        dataset_paths.push(temp_path);

        let dataset = Dataset::from_path(&dataset_paths[i - 1]).unwrap();
        let _proof = ledger
            .notarize_dataset(dataset, &format!("dataset-{}", i), ProofConfig::default())
            .unwrap();
    }

    // Verify we're tracking all datasets
    let summary = ledger.get_summary().unwrap();
    assert_eq!(summary.total_entries, 5);
    assert_eq!(summary.datasets_tracked, 5);

    // Test querying specific datasets
    for i in 1..=5 {
        let history = ledger
            .get_dataset_history(&format!("dataset-{}", i))
            .unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].dataset_name, format!("dataset-{}", i));
    }

    // Test export functionality
    let query = zkp_dataset_ledger::ledger::LedgerQuery::default();
    let export_json = ledger.export_to_json(&query).unwrap();
    assert!(!export_json.is_empty());

    // Verify JSON contains all datasets
    for i in 1..=5 {
        assert!(export_json.contains(&format!("dataset-{}", i)));
    }

    // Cleanup
    for path in dataset_paths {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn test_ledger_persistence_and_recovery() {
    // This test would verify that ledger state can be persisted and recovered
    // For now, testing in-memory behavior

    let ledger_name = "persistence-test";
    let mut ledger = Ledger::new(ledger_name).unwrap();

    // Add some entries
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "test,data").unwrap();
    writeln!(temp_file, "1,100").unwrap();

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let dataset = Dataset::from_path(&temp_path).unwrap();
    let _proof = ledger
        .notarize_dataset(dataset, "test-dataset", ProofConfig::default())
        .unwrap();

    // Verify ledger state
    let summary = ledger.get_summary().unwrap();
    assert_eq!(summary.name, ledger_name);
    assert_eq!(summary.total_entries, 1);

    // For in-memory storage, create a new ledger and verify independence
    let ledger2 = Ledger::new("different-ledger").unwrap();
    let summary2 = ledger2.get_summary().unwrap();
    assert_eq!(summary2.total_entries, 0); // Should be empty

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}
