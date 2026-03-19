//! End-to-end demonstration of ZKP Dataset Ledger
//!
//! Run with: cargo run --example demo

use zkp_dataset_ledger::{
    commitment::{CommitmentConfig, DatasetCommitment},
    ledger::DatasetLedger,
    proof::{ProofGenerator, ProofType, ProofVerifier},
};

fn main() {
    println!("=== ZKP Dataset Ledger Demo ===\n");
    println!("Scenario: Prove ML training data properties to an auditor");
    println!("without revealing the actual dataset.\n");

    let dataset = b"employee_id,dept,salary\n\
                    1001,eng,95000\n\
                    1002,mkt,72000\n\
                    1003,eng,98000\n";

    // Commit
    let config = CommitmentConfig::default();
    let commitment = DatasetCommitment::commit(dataset, &config, Some("Q4 dataset".to_string())).unwrap();
    println!("Commitment: {}", commitment.commitment_hash);

    // Prove
    let proof = ProofGenerator::generate(&commitment, dataset, ProofType::ExactRowCount { count: 3 }).unwrap();
    println!("Proof challenge: {}", &proof.challenge[..16]);

    // Verify
    let valid = ProofVerifier::verify(&proof).unwrap();
    println!("Verification: {}", if valid { "✓ VALID" } else { "✗ INVALID" });

    // Ledger
    let mut ledger = DatasetLedger::new("demo-ledger");
    ledger.append_with_proofs(commitment, vec![proof]).unwrap();
    println!("Ledger integrity: {}", if ledger.verify_integrity() { "✓ OK" } else { "✗ FAILED" });
}
