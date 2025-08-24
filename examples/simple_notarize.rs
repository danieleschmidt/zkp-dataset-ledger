//! Simple example demonstrating basic dataset notarization.

use zkp_dataset_ledger::{Dataset, Ledger, Result};

fn main() -> Result<()> {
    println!("🚀 ZKP Dataset Ledger - Simple Notarization Example");

    // Initialize logging
    env_logger::init();

    // Create a new ledger with storage
    let mut ledger = Ledger::with_storage(
        "example-project".to_string(),
        "./example_ledger/ledger.json".to_string(),
    )?;

    // Create test CSV file
    std::fs::create_dir_all("./tmp")?;
    let test_path = "./tmp/test_data.csv";
    std::fs::write(test_path, "name,age,score\nAlice,25,85\nBob,30,92\n")?;

    // Create dataset from file
    let dataset = Dataset::new("training-data-v1".to_string(), test_path.to_string())?;

    println!("📋 Notarizing dataset: {}", dataset.name);

    // Generate proof and add to ledger
    let proof = ledger.notarize_dataset(dataset, "integrity".to_string())?;

    println!("✅ Successfully notarized dataset!");
    println!("   Dataset hash: {}", &proof.dataset_hash[..16]);
    println!("   Proof type: {}", proof.proof_type);
    println!("   Timestamp: {}", proof.timestamp);

    // Verify the proof
    let is_valid = ledger.verify_proof(&proof);
    if is_valid {
        println!("🔍 Proof verification: PASSED");
    } else {
        println!("❌ Proof verification: FAILED");
    }

    // Show ledger status
    let stats = ledger.get_statistics();
    println!("\n📊 Ledger Summary:");
    println!("   Total entries: {}", stats.total_operations);
    println!("   Datasets tracked: {}", stats.total_datasets);
    if let Some(path) = stats.storage_path {
        println!("   Storage path: {}", path);
    }

    // Cleanup
    std::fs::remove_file(test_path).ok();
    std::fs::remove_dir_all("./tmp").ok();
    std::fs::remove_dir_all("./example_ledger").ok();

    Ok(())
}
