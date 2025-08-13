//! Simple example demonstrating basic dataset notarization.

use zkp_dataset_ledger::{Dataset, Ledger, ProofConfig, Result};

fn main() -> Result<()> {
    println!("🚀 ZKP Dataset Ledger - Simple Notarization Example");

    // Initialize library
    zkp_dataset_ledger::init()?;

    // Create a new ledger
    let mut ledger = Ledger::new("example-project")?;

    // Create a simple dataset
    let dataset = Dataset {
        name: "training-data-v1".to_string(),
        hash: "abc123def456".to_string(),
        size: 1024,
        row_count: Some(100),
        column_count: Some(5),
        schema: None,
        statistics: None,
        format: zkp_dataset_ledger::DatasetFormat::Csv,
        path: Some("data/train.csv".to_string()),
    };

    println!("📋 Notarizing dataset: {}", dataset.name);

    // Generate proof and add to ledger
    let proof_config = ProofConfig::default();
    let proof = ledger.notarize_dataset(dataset, "training-data-v1", proof_config)?;

    println!("✅ Successfully notarized dataset!");
    println!("   Proof size: {} bytes", proof.size_bytes());
    println!("   Dataset hash: {}", &proof.dataset_hash[..16]);
    println!("   Timestamp: {}", proof.timestamp);

    // Verify the proof
    let is_valid = proof.verify()?;
    if is_valid {
        println!("🔍 Proof verification: PASSED");
    } else {
        println!("❌ Proof verification: FAILED");
    }

    // Show ledger status
    let summary = ledger.get_summary()?;
    println!("\n📊 Ledger Summary:");
    println!("   Total entries: {}", summary.total_entries);
    println!("   Datasets tracked: {}", summary.datasets_tracked);
    println!("   Storage size: {} bytes", summary.storage_size_bytes);

    Ok(())
}
