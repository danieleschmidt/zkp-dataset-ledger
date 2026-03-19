//! Integration tests for ZKP Dataset Ledger
//!
//! Tests the full workflow: commit → prove → verify → ledger

use zkp_dataset_ledger::{
    commitment::{CommitmentConfig, DatasetCommitment},
    ledger::DatasetLedger,
    proof::{ProofGenerator, ProofType, ProofVerifier},
};

const HR_DATASET: &[u8] = b"employee_id,department,salary,performance\n\
                             1001,engineering,95000,88\n\
                             1002,marketing,72000,92\n\
                             1003,engineering,98000,85\n\
                             1004,hr,68000,90\n\
                             1005,marketing,75000,87\n";

#[test]
fn end_to_end_workflow() {
    // Step 1: Data owner commits to dataset
    let config = CommitmentConfig {
        include_row_count: true,
        include_column_count: true,
        include_schema_hash: true,
        ..Default::default()
    };
    let commitment = DatasetCommitment::commit(HR_DATASET, &config, Some("HR Dataset".to_string()))
        .expect("commitment should succeed");

    // Step 2: Generate proofs
    let proof1 = ProofGenerator::generate(&commitment, HR_DATASET, ProofType::Existence)
        .expect("existence proof should succeed");
    let proof2 = ProofGenerator::generate(&commitment, HR_DATASET, ProofType::ExactRowCount { count: 5 })
        .expect("row count proof should succeed");
    let proof3 = ProofGenerator::generate(
        &commitment,
        HR_DATASET,
        ProofType::StatisticalBounds {
            column: "performance".to_string(),
            mean_min: 80.0,
            mean_max: 95.0,
        },
    ).expect("stats proof should succeed");

    // Step 3: Auditor verifies proofs (no dataset needed)
    assert!(ProofVerifier::verify(&proof1).unwrap(), "existence proof valid");
    assert!(ProofVerifier::verify(&proof2).unwrap(), "row count proof valid");
    assert!(ProofVerifier::verify(&proof3).unwrap(), "stats proof valid");

    // Step 4: Append to ledger
    let mut ledger = DatasetLedger::new("hr-audit");
    ledger.append_with_proofs(commitment, vec![proof1, proof2, proof3])
        .expect("ledger append should succeed");

    assert!(ledger.verify_integrity(), "ledger should be intact");
}

#[test]
fn tampered_proof_is_rejected() {
    let commitment = DatasetCommitment::commit(HR_DATASET, &CommitmentConfig::default(), None).unwrap();
    let mut proof = ProofGenerator::generate(
        &commitment, HR_DATASET, ProofType::ExactRowCount { count: 5 }
    ).unwrap();

    // Tamper: claim different row count
    proof.proof_type = ProofType::ExactRowCount { count: 1000 };

    assert!(!ProofVerifier::verify(&proof).unwrap(), "tampered proof should be rejected");
}

#[test]
fn dishonest_prover_cannot_generate_false_proof() {
    let commitment = DatasetCommitment::commit(HR_DATASET, &CommitmentConfig::default(), None).unwrap();

    // Try to prove the dataset has 9999 rows (it doesn't)
    let result = ProofGenerator::generate(
        &commitment, HR_DATASET, ProofType::ExactRowCount { count: 9999 }
    );

    assert!(result.is_err(), "should not be able to generate dishonest proof");
}

#[test]
fn commitment_binds_to_specific_dataset() {
    let commitment = DatasetCommitment::commit(HR_DATASET, &CommitmentConfig::default(), None).unwrap();

    // Verify against original: should pass
    assert!(commitment.verify(HR_DATASET).unwrap());

    // Verify against modified dataset: should fail
    let modified = b"employee_id,department,salary,performance\n1001,engineering,99999,88\n";
    assert!(!commitment.verify(modified).unwrap());
}

#[test]
fn ledger_chains_entries_correctly() {
    let mut ledger = DatasetLedger::new("test");

    let datasets: Vec<&[u8]> = vec![
        b"id,val\n1,100\n",
        b"id,val\n2,200\n",
        b"id,val\n3,300\n",
    ];

    for ds in &datasets {
        let c = DatasetCommitment::commit(ds, &CommitmentConfig::default(), None).unwrap();
        ledger.append(c).unwrap();
    }

    assert_eq!(ledger.len(), 3);
    assert!(ledger.verify_integrity());

    // Tampering with entry 1 should break the chain
    ledger.entries[1].commitment.commitment_hash = "a".repeat(64);
    assert!(!ledger.verify_integrity());
}

#[test]
fn proof_is_deterministic_for_same_commitment() {
    // Two proofs for the same commitment+data should differ only in generated_at
    let commitment = DatasetCommitment::commit(HR_DATASET, &CommitmentConfig::default(), None).unwrap();

    let proof1 = ProofGenerator::generate(&commitment, HR_DATASET, ProofType::Existence).unwrap();
    let proof2 = ProofGenerator::generate(&commitment, HR_DATASET, ProofType::Existence).unwrap();

    // Same challenge (deterministic Fiat-Shamir)
    assert_eq!(proof1.challenge, proof2.challenge);
    // Same response (deterministic)
    assert_eq!(proof1.response, proof2.response);
}

#[test]
fn range_proof_works() {
    let commitment = DatasetCommitment::commit(HR_DATASET, &CommitmentConfig::default(), None).unwrap();

    let proof = ProofGenerator::generate(
        &commitment, HR_DATASET, ProofType::RowCountRange { min: 3, max: 10 }
    ).unwrap();
    assert!(ProofVerifier::verify(&proof).unwrap());

    // Out-of-range should fail to generate
    let result = ProofGenerator::generate(
        &commitment, HR_DATASET, ProofType::RowCountRange { min: 10, max: 20 }
    );
    assert!(result.is_err());
}
