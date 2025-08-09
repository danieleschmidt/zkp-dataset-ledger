//! Unit tests for the proof module.

use std::io::Write;
use tempfile::NamedTempFile;
use zkp_dataset_ledger::proof::{Proof, ProofConfig, ProofType};
use zkp_dataset_ledger::{Dataset, Result};

/// Test basic proof generation and verification.
#[test]
fn test_proof_generation_basic() -> Result<()> {
    // Create test dataset
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "id,value")?;
    writeln!(temp_file, "1,100")?;
    writeln!(temp_file, "2,200")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;
    let config = ProofConfig::default();

    // Generate proof
    let proof = Proof::generate(&dataset, &config)?;

    // Verify basic properties
    assert_eq!(proof.dataset_hash, dataset.compute_hash());
    assert!(!proof.proof_data.is_empty());
    assert_eq!(proof.proof_type, ProofType::DatasetIntegrity);
    assert!(!proof.public_inputs.is_empty());

    // Verify the proof
    assert!(proof.verify()?);

    // Clean up
    std::fs::remove_file(temp_path).ok();

    Ok(())
}

/// Test different proof types.
#[test]
fn test_different_proof_types() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "name,age")?;
    writeln!(temp_file, "Alice,25")?;
    writeln!(temp_file, "Bob,30")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;

    let proof_types = vec![
        ProofType::DatasetIntegrity,
        ProofType::RowCount,
        ProofType::Schema,
        ProofType::Statistics,
    ];

    for proof_type in proof_types {
        let config = ProofConfig {
            proof_type: proof_type.clone(),
            use_groth16: false, // Use legacy proofs for speed in tests
            ..ProofConfig::default()
        };

        let proof = Proof::generate(&dataset, &config)?;
        assert_eq!(proof.proof_type, proof_type);
        assert!(proof.verify()?);
    }

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test proof serialization.
#[test]
fn test_proof_serialization() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "data")?;
    writeln!(temp_file, "test")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;
    let config = ProofConfig {
        use_groth16: false, // Use legacy proofs for speed in tests
        ..ProofConfig::default()
    };

    let proof = Proof::generate(&dataset, &config)?;

    // Test JSON serialization
    let json = proof.to_json()?;
    let deserialized_proof = Proof::from_json(&json)?;

    assert_eq!(proof.dataset_hash, deserialized_proof.dataset_hash);
    assert_eq!(proof.proof_type, deserialized_proof.proof_type);
    assert_eq!(proof.proof_data, deserialized_proof.proof_data);

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test proof with Merkle tree.
#[test]
fn test_proof_with_merkle_tree() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "col1,col2")?;
    writeln!(temp_file, "a,1")?;
    writeln!(temp_file, "b,2")?;
    writeln!(temp_file, "c,3")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;
    let config = ProofConfig {
        include_merkle_proof: true,
        use_groth16: false, // Use legacy proofs for speed in tests
        ..ProofConfig::default()
    };

    let proof = Proof::generate(&dataset, &config)?;

    // Verify Merkle proof is included
    assert!(proof.merkle_root.is_some());
    assert!(proof.merkle_proof.is_some());
    assert!(proof.verify()?);

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test batch proof generation.
#[test]
fn test_batch_proof_generation() -> Result<()> {
    use zkp_dataset_ledger::proof::BatchProofGenerator;

    // Create test datasets
    let mut datasets = Vec::new();
    let mut temp_files = Vec::new();

    for i in 0..3 {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "id,value")?;
        writeln!(temp_file, "1,{}", i * 100)?;
        writeln!(temp_file, "2,{}", i * 200)?;

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path)?;

        let dataset = Dataset::from_path(&temp_path)?;
        datasets.push(dataset);
        temp_files.push(temp_path);
    }

    // Generate batch proofs
    let config = ProofConfig {
        use_groth16: false, // Use legacy proofs for speed in tests
        ..ProofConfig::default()
    };
    let mut generator = BatchProofGenerator::new(config);

    for dataset in datasets {
        generator.add_dataset(dataset);
    }

    let proofs = generator.generate_batch_proof()?;
    assert_eq!(proofs.len(), 3);

    // Verify all proofs
    for proof in proofs {
        assert!(proof.verify()?);
    }

    // Clean up
    for temp_file in temp_files {
        std::fs::remove_file(temp_file).ok();
    }

    Ok(())
}

/// Test proof verification with custom inputs.
#[test]
fn test_proof_verification_with_inputs() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "x,y")?;
    writeln!(temp_file, "1,2")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;
    let config = ProofConfig {
        use_groth16: false, // Use legacy proofs for speed in tests
        ..ProofConfig::default()
    };

    let proof = Proof::generate(&dataset, &config)?;

    // Test verification with correct inputs
    assert!(proof.verify_with_inputs(&proof.public_inputs)?);

    // Test verification with incorrect inputs
    let wrong_inputs = vec!["wrong".to_string(), "inputs".to_string()];
    assert!(!proof.verify_with_inputs(&wrong_inputs)?);

    std::fs::remove_file(temp_path).ok();
    Ok(())
}

/// Test proof size calculation.
#[test]
fn test_proof_size() -> Result<()> {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "a")?;
    writeln!(temp_file, "test")?;

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path)?;

    let dataset = Dataset::from_path(&temp_path)?;
    let config = ProofConfig {
        use_groth16: false, // Use legacy proofs for speed in tests
        ..ProofConfig::default()
    };

    let proof = Proof::generate(&dataset, &config)?;

    // Test size calculation
    let size = proof.size_bytes();
    assert!(size > 0);
    assert_eq!(size, proof.proof_data.len());

    // Test proof summary
    let summary = proof.summary();
    assert_eq!(summary.size_bytes, size);
    assert_eq!(summary.proof_type, ProofType::DatasetIntegrity);

    std::fs::remove_file(temp_path).ok();
    Ok(())
}
