//! Integration test for complete ZKP dataset ledger workflow.

use std::io::Write;
use tempfile::NamedTempFile;
use zkp_dataset_ledger::{Dataset, Ledger, Result};
use zkp_dataset_ledger::proof::{Proof, ProofConfig, ProofType};
use zkp_dataset_ledger::storage::MemoryStorage;

/// Test complete workflow from dataset to ledger entry.
#[test]
fn test_complete_workflow() -> Result<()> {
    // Step 1: Create a test dataset
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "id,name,score")?;
    writeln!(temp_file, "1,Alice,95")?;
    writeln!(temp_file, "2,Bob,87")?;
    writeln!(temp_file, "3,Charlie,92")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    // Step 2: Load dataset and verify properties
    let dataset = Dataset::from_path(&temp_path)?;
    assert_eq!(dataset.row_count, Some(3));
    assert_eq!(dataset.column_count, Some(3));
    assert!(dataset.size > 0);

    // Step 3: Generate proof for the dataset
    let proof_config = ProofConfig {
        proof_type: ProofType::DatasetIntegrity,
        use_groth16: false, // Use legacy proofs for faster testing
        include_merkle_proof: true,
        ..ProofConfig::default()
    };

    let proof = Proof::generate(&dataset, &proof_config)?;
    assert_eq!(proof.dataset_hash, dataset.compute_hash());
    assert!(proof.verify()?);
    assert!(proof.merkle_root.is_some());

    // Step 4: Create ledger and add entry
    let storage = Box::new(MemoryStorage::new());
    let mut ledger = Ledger::new(storage);

    let entry_id = ledger.add_dataset_entry(dataset.clone(), proof.clone())?;
    assert!(!entry_id.is_empty());

    // Step 5: Verify ledger state
    let entry = ledger.get_entry(&entry_id)?.unwrap();
    assert_eq!(entry.dataset_hash, dataset.compute_hash());
    assert_eq!(entry.proof_hash, proof.dataset_hash);

    // Step 6: Test ledger integrity
    assert!(ledger.verify_chain()?);

    // Step 7: Test proof verification through ledger
    assert!(ledger.verify_proof(&entry_id, &proof)?);

    // Step 8: Export and import test
    let ledger_json = ledger.export_json()?;
    let imported_ledger = Ledger::import_json(&ledger_json, Box::new(MemoryStorage::new()))?;
    
    assert!(imported_ledger.verify_chain()?);
    let imported_entry = imported_ledger.get_entry(&entry_id)?.unwrap();
    assert_eq!(imported_entry.dataset_hash, entry.dataset_hash);

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test workflow with multiple datasets and proof types.
#[test]
fn test_multi_dataset_workflow() -> Result<()> {
    let storage = Box::new(MemoryStorage::new());
    let mut ledger = Ledger::new(storage);

    // Create multiple test datasets
    let datasets_data = vec![
        ("users", "id,name\n1,Alice\n2,Bob"),
        ("scores", "user_id,score\n1,95\n2,87"),
        ("metadata", "key,value\nversion,1.0\ndate,2024-01-01"),
    ];

    let mut entry_ids = Vec::new();

    for (name, csv_content) in datasets_data {
        // Create dataset
        let dataset = Dataset::from_csv_string(csv_content)?;
        
        // Choose different proof types for variety
        let proof_type = match name {
            "users" => ProofType::DatasetIntegrity,
            "scores" => ProofType::Statistics,
            "metadata" => ProofType::Schema,
            _ => ProofType::DatasetIntegrity,
        };

        // Generate proof
        let proof_config = ProofConfig {
            proof_type,
            use_groth16: false, // Use legacy proofs for faster testing
            ..ProofConfig::default()
        };

        let proof = Proof::generate(&dataset, &proof_config)?;
        assert!(proof.verify()?);

        // Add to ledger
        let entry_id = ledger.add_dataset_entry(dataset, proof)?;
        entry_ids.push(entry_id);
    }

    // Verify all entries
    assert_eq!(entry_ids.len(), 3);
    for entry_id in &entry_ids {
        let entry = ledger.get_entry(entry_id)?.unwrap();
        assert!(!entry.dataset_hash.is_empty());
    }

    // Verify chain integrity
    assert!(ledger.verify_chain()?);

    // Test ledger statistics
    let stats = ledger.get_statistics()?;
    assert_eq!(stats.total_entries, 3);
    assert!(stats.total_size > 0);

    Ok(())
}

/// Test workflow with different storage backends.
#[test]
fn test_storage_backend_workflow() -> Result<()> {
    // Test with memory storage
    let dataset = Dataset::from_csv_string("a,b\n1,2\n3,4")?;
    
    let proof_config = ProofConfig {
        use_groth16: false, // Use legacy proofs for faster testing
        ..ProofConfig::default()
    };
    let proof = Proof::generate(&dataset, &proof_config)?;

    // Test memory storage
    let memory_storage = Box::new(MemoryStorage::new());
    let mut memory_ledger = Ledger::new(memory_storage);
    
    let entry_id = memory_ledger.add_dataset_entry(dataset.clone(), proof.clone())?;
    assert!(memory_ledger.verify_chain()?);
    
    let entry = memory_ledger.get_entry(&entry_id)?.unwrap();
    assert_eq!(entry.dataset_hash, dataset.compute_hash());

    Ok(())
}

/// Test error handling in workflow.
#[test]
fn test_workflow_error_handling() -> Result<()> {
    let storage = Box::new(MemoryStorage::new());
    let ledger = Ledger::new(storage);

    // Test getting non-existent entry
    let result = ledger.get_entry("nonexistent_id");
    assert!(result.is_ok());
    assert!(result?.is_none());

    // Test verification with invalid proof
    let dataset = Dataset::from_csv_string("x,y\n1,2")?;
    let proof_config = ProofConfig {
        use_groth16: false,
        ..ProofConfig::default()
    };
    let proof = Proof::generate(&dataset, &proof_config)?;

    // Verify with wrong entry ID should return false
    let result = ledger.verify_proof("wrong_id", &proof);
    assert!(result.is_ok());
    assert!(!result?);

    Ok(())
}

/// Test concurrent-like operations (simulated).
#[test]
fn test_concurrent_operations() -> Result<()> {
    let storage = Box::new(MemoryStorage::new());
    let mut ledger = Ledger::new(storage);

    // Simulate multiple operations
    let mut entry_ids = Vec::new();
    
    for i in 0..10 {
        let csv_data = format!("id,value\n{},{}", i, i * 10);
        let dataset = Dataset::from_csv_string(&csv_data)?;
        
        let proof_config = ProofConfig {
            use_groth16: false,
            ..ProofConfig::default()
        };
        let proof = Proof::generate(&dataset, &proof_config)?;
        
        let entry_id = ledger.add_dataset_entry(dataset, proof)?;
        entry_ids.push(entry_id);
    }

    // Verify all entries were added correctly
    assert_eq!(entry_ids.len(), 10);
    assert!(ledger.verify_chain()?);

    // Verify statistics
    let stats = ledger.get_statistics()?;
    assert_eq!(stats.total_entries, 10);

    Ok(())
}

/// Test workflow with batch operations.
#[test]
fn test_batch_workflow() -> Result<()> {
    use zkp_dataset_ledger::proof::BatchProofGenerator;

    // Create multiple datasets
    let csv_data_sets = vec![
        "a,b\n1,2",
        "x,y\n3,4", 
        "p,q\n5,6",
    ];

    let mut datasets = Vec::new();
    for csv_data in csv_data_sets {
        datasets.push(Dataset::from_csv_string(csv_data)?);
    }

    // Generate batch proofs
    let proof_config = ProofConfig {
        use_groth16: false,
        ..ProofConfig::default()
    };
    let mut batch_generator = BatchProofGenerator::new(proof_config);

    for dataset in &datasets {
        batch_generator.add_dataset(dataset.clone());
    }

    let proofs = batch_generator.generate_batch_proof()?;
    assert_eq!(proofs.len(), 3);

    // Verify all proofs
    for proof in &proofs {
        assert!(proof.verify()?);
    }

    // Add all to ledger
    let storage = Box::new(MemoryStorage::new());
    let mut ledger = Ledger::new(storage);

    for (dataset, proof) in datasets.into_iter().zip(proofs.into_iter()) {
        let entry_id = ledger.add_dataset_entry(dataset, proof)?;
        assert!(!entry_id.is_empty());
    }

    assert!(ledger.verify_chain()?);

    Ok(())
}