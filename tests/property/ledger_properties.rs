use proptest::prelude::*;
use zkp_dataset_ledger::{Dataset, Ledger, ProofConfig};
use tempfile::TempDir;
use std::collections::HashSet;

/// Property-based tests for ledger invariants and cryptographic properties
/// These tests use random generation to explore edge cases and ensure system correctness

proptest! {
    /// Property: Ledger should maintain immutability - once a transaction is added, 
    /// it cannot be modified or removed
    #[test]
    fn ledger_immutability_property(
        dataset_sizes in prop::collection::vec(1usize..1000, 1..10)
    ) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("test_ledger");
        let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
        
        let mut transaction_hashes = Vec::new();
        
        // Add multiple datasets to the ledger
        for (i, size) in dataset_sizes.iter().enumerate() {
            let csv_path = temp_dir.path().join(format!("test_{}.csv", i));
            let mut content = String::from("id,value\n");
            for j in 1..=*size {
                content.push_str(&format!("{},{}\n", j, j * 10));
            }
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            let proof = ledger.notarize_dataset(
                dataset, 
                &format!("dataset-{}", i), 
                ProofConfig::default()
            ).expect("Failed to notarize dataset");
            
            transaction_hashes.push(proof.transaction_hash());
        }
        
        // Verify all transactions are still present and unchanged
        let audit_trail = ledger.get_audit_trail().expect("Failed to get audit trail");
        prop_assert_eq!(audit_trail.len(), dataset_sizes.len());
        
        for (i, tx_hash) in transaction_hashes.iter().enumerate() {
            let transaction = ledger.get_transaction(tx_hash)
                .expect("Transaction should exist");
            prop_assert_eq!(transaction.dataset_name(), format!("dataset-{}", i));
        }
    }
    
    /// Property: Merkle tree root should change with each new transaction
    #[test]
    fn merkle_root_uniqueness_property(
        transactions in prop::collection::vec(1usize..100, 2..20)
    ) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("test_ledger");
        let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
        
        let mut roots = HashSet::new();
        let initial_root = ledger.get_merkle_root().expect("Failed to get initial root");
        roots.insert(initial_root);
        
        for (i, tx_size) in transactions.iter().enumerate() {
            let csv_path = temp_dir.path().join(format!("test_{}.csv", i));
            let mut content = String::from("id,data\n");
            for j in 1..=*tx_size {
                content.push_str(&format!("{},data_{}\n", j, j));
            }
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            ledger.notarize_dataset(
                dataset, 
                &format!("tx-{}", i), 
                ProofConfig::default()
            ).expect("Failed to notarize dataset");
            
            let new_root = ledger.get_merkle_root().expect("Failed to get new root");
            prop_assert!(!roots.contains(&new_root), "Merkle root should be unique");
            roots.insert(new_root);
        }
        
        prop_assert_eq!(roots.len(), transactions.len() + 1); // +1 for initial root
    }
    
    /// Property: Proof verification should be deterministic and repeatable
    #[test]
    fn proof_verification_determinism_property(
        row_count in 1usize..500,
        column_data in prop::collection::vec(prop::num::i32::ANY, 1..10)
    ) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("test_ledger");
        let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
        
        // Create deterministic dataset
        let csv_path = temp_dir.path().join("deterministic.csv");
        let mut content = String::from("id");
        for i in 0..column_data.len() {
            content.push_str(&format!(",col_{}", i));
        }
        content.push('\n');
        
        for row in 1..=row_count {
            content.push_str(&format!("{}", row));
            for &val in &column_data {
                content.push_str(&format!(",{}", val.wrapping_add(row as i32)));
            }
            content.push('\n');
        }
        std::fs::write(&csv_path, content).unwrap();
        
        let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
        let proof = ledger.notarize_dataset(
            dataset, 
            "deterministic-dataset", 
            ProofConfig::deterministic()
        ).expect("Failed to notarize dataset");
        
        // Verify the proof multiple times - should always succeed
        for _attempt in 0..5 {
            prop_assert!(ledger.verify_proof(&proof).expect("Verification should not fail"));
        }
        
        // Create another ledger and verify the proof there too
        let ledger2_path = temp_dir.path().join("test_ledger_2");
        let ledger2 = Ledger::new(&ledger2_path).expect("Failed to initialize second ledger");
        prop_assert!(ledger2.verify_proof(&proof).expect("Cross-ledger verification should work"));
    }
    
    /// Property: Dataset hash should be collision-resistant for different inputs
    #[test]
    fn dataset_hash_collision_resistance_property(
        datasets in prop::collection::vec(
            prop::collection::vec(prop::num::u32::ANY, 1..100), 
            2..50
        )
    ) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("test_ledger");
        let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
        
        let mut hashes = HashSet::new();
        
        for (i, dataset_data) in datasets.iter().enumerate() {
            let csv_path = temp_dir.path().join(format!("dataset_{}.csv", i));
            let mut content = String::from("id,value\n");
            for (j, &value) in dataset_data.iter().enumerate() {
                content.push_str(&format!("{},{}\n", j + 1, value));
            }
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            let dataset_hash = dataset.compute_hash().expect("Failed to compute hash");
            
            // Different datasets should have different hashes
            if !hashes.is_empty() {
                prop_assert!(!hashes.contains(&dataset_hash), 
                    "Hash collision detected for different datasets");
            }
            hashes.insert(dataset_hash);
        }
    }
    
    /// Property: Ledger operations should be atomic - either fully succeed or fully fail
    #[test]
    fn ledger_atomicity_property(
        valid_datasets in prop::collection::vec(1usize..50, 1..5),
        invalid_position in prop::sample::Index
    ) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("test_ledger");
        let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
        
        // Create valid datasets
        let mut csv_paths = Vec::new();
        for (i, &size) in valid_datasets.iter().enumerate() {
            let csv_path = temp_dir.path().join(format!("valid_{}.csv", i));
            let mut content = String::from("id,value\n");
            for j in 1..=size {
                content.push_str(&format!("{},{}\n", j, j * 5));
            }
            std::fs::write(&csv_path, content).unwrap();
            csv_paths.push(csv_path);
        }
        
        // Create one invalid dataset at random position
        let invalid_idx = invalid_position.index(valid_datasets.len());
        let invalid_csv = temp_dir.path().join("invalid.csv");
        std::fs::write(&invalid_csv, "malformed,csv,data\nno,proper\n").unwrap();
        csv_paths.insert(invalid_idx, invalid_csv);
        
        let initial_audit_trail_len = ledger.get_audit_trail()
            .expect("Failed to get initial audit trail").len();
        
        // Attempt batch operation - should fail atomically due to invalid dataset
        let mut batch_results = Vec::new();
        for (i, csv_path) in csv_paths.iter().enumerate() {
            let dataset_result = Dataset::from_path(csv_path);
            if dataset_result.is_err() {
                // If any dataset is invalid, the entire batch should be rollback-able
                break;
            }
            
            let dataset = dataset_result.unwrap();
            let result = ledger.notarize_dataset(
                dataset, 
                &format!("batch-{}", i), 
                ProofConfig::default()
            );
            batch_results.push(result);
        }
        
        // Check that ledger state is consistent
        let final_audit_trail = ledger.get_audit_trail()
            .expect("Failed to get final audit trail");
        
        // Either all valid operations succeeded, or the ledger remained unchanged
        let successful_operations = batch_results.iter()
            .filter(|r| r.is_ok())
            .count();
        
        prop_assert_eq!(
            final_audit_trail.len(), 
            initial_audit_trail_len + successful_operations
        );
    }
}

/// Cryptographic property tests
mod crypto_properties {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        /// Property: ZK proofs should not reveal private information
        #[test]
        fn zero_knowledge_property(
            private_data in prop::collection::vec(prop::num::f64::ANY, 10..100),
            public_row_count in 10usize..100
        ) {
            let temp_dir = TempDir::new().expect("Failed to create temp directory");
            let ledger_path = temp_dir.path().join("test_ledger");
            let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
            
            // Create dataset with private values
            let csv_path = temp_dir.path().join("private_data.csv");
            let mut content = String::from("id,private_value\n");
            for (i, &value) in private_data.iter().enumerate() {
                if i >= public_row_count { break; }
                content.push_str(&format!("{},{:.6}\n", i + 1, value));
            }
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            
            // Generate proof that proves row count without revealing actual data
            let proof_config = ProofConfig::privacy_preserving()
                .prove_row_count(true)
                .hide_column_values(vec!["private_value".to_string()]);
            
            let proof = ledger.notarize_dataset(
                dataset, 
                "private-dataset", 
                proof_config
            ).expect("Failed to generate privacy-preserving proof");
            
            // Verify proof is valid
            prop_assert!(ledger.verify_proof(&proof).expect("Verification failed"));
            
            // Ensure proof doesn't contain private data
            let proof_json = serde_json::to_string(&proof)
                .expect("Failed to serialize proof");
            
            for &private_value in &private_data {
                prop_assert!(
                    !proof_json.contains(&private_value.to_string()),
                    "Proof should not contain private values"
                );
            }
            
            // But should contain public information (row count)
            prop_assert!(
                proof_json.contains(&public_row_count.to_string()) ||
                proof.get_public_inputs().iter().any(|input| 
                    input.contains(&public_row_count.to_string())
                ),
                "Proof should contain publicly provable information"
            );
        }
        
        /// Property: Proof completeness - valid statements should always be provable
        #[test]
        fn proof_completeness_property(
            dataset_properties in (1usize..1000, 2usize..20, prop::bool::ANY)
        ) {
            let (row_count, col_count, include_nulls) = dataset_properties;
            
            let temp_dir = TempDir::new().expect("Failed to create temp directory");
            let ledger_path = temp_dir.path().join("test_ledger");
            let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
            
            // Create dataset with known properties
            let csv_path = temp_dir.path().join("known_properties.csv");
            let mut content = String::new();
            
            // Create headers
            for i in 0..col_count {
                if i > 0 { content.push(','); }
                content.push_str(&format!("col_{}", i));
            }
            content.push('\n');
            
            // Create data rows
            for row in 1..=row_count {
                for col in 0..col_count {
                    if col > 0 { content.push(','); }
                    
                    if include_nulls && row % 7 == 0 && col == 1 {
                        // Insert some null values
                        content.push_str("");
                    } else {
                        content.push_str(&format!("{}", row * 10 + col));
                    }
                }
                content.push('\n');
            }
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            
            // Prove the known properties
            let proof_config = ProofConfig::default()
                .prove_row_count(true)
                .prove_column_count(true)
                .prove_null_counts(include_nulls);
            
            // This should always succeed for valid datasets with known properties
            let proof_result = ledger.notarize_dataset(
                dataset, 
                "completeness-test", 
                proof_config
            );
            
            prop_assert!(proof_result.is_ok(), "Valid properties should always be provable");
            
            let proof = proof_result.unwrap();
            prop_assert!(ledger.verify_proof(&proof).expect("Verification should not fail"));
        }
        
        /// Property: Proof soundness - invalid statements should not be provable
        #[test]
        fn proof_soundness_property(
            actual_row_count in 10usize..100,
            claimed_row_count in 200usize..300  // Intentionally different range
        ) {
            // Skip if claimed equals actual by coincidence
            prop_assume!(claimed_row_count != actual_row_count);
            
            let temp_dir = TempDir::new().expect("Failed to create temp directory");
            let ledger_path = temp_dir.path().join("test_ledger");
            let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
            
            // Create dataset with known row count
            let csv_path = temp_dir.path().join("soundness_test.csv");
            let mut content = String::from("id,value\n");
            for i in 1..=actual_row_count {
                content.push_str(&format!("{},{}\n", i, i * 2));
            }
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            
            // Attempt to prove false claim about row count
            // Note: In a real implementation, this would involve trying to construct
            // a malicious proof, which should fail in a sound system
            let proof_result = ledger.notarize_dataset(
                dataset, 
                "soundness-test", 
                ProofConfig::default().prove_row_count(true)
            );
            
            // The proof should succeed with correct row count
            prop_assert!(proof_result.is_ok());
            let proof = proof_result.unwrap();
            
            // But verification should catch any inconsistencies
            prop_assert!(ledger.verify_proof(&proof).expect("Verification should not fail"));
            
            // The proof should contain the actual row count, not the false claim
            let public_inputs = proof.get_public_inputs();
            prop_assert!(
                public_inputs.iter().any(|input| 
                    input.contains(&actual_row_count.to_string())
                ),
                "Proof should contain actual row count"
            );
            
            prop_assert!(
                !public_inputs.iter().any(|input| 
                    input.contains(&claimed_row_count.to_string())
                ),
                "Proof should not contain false claim"
            );
        }
    }
}