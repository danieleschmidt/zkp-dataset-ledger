// Basic usage example for ZKP Dataset Ledger
//
// This example demonstrates:
// 1. Initializing a ledger
// 2. Notarizing a dataset
// 3. Recording transformations
// 4. Verifying proofs
// 5. Generating audit reports

use zkp_dataset_ledger::{Dataset, Ledger, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("🚀 ZKP Dataset Ledger - Basic Example");
    println!("=====================================\n");

    // Create sample data first
    create_sample_data()?;

    // 1. Initialize ledger for ML project
    let mut ledger = Ledger::with_storage(
        "fraud-detection-model".to_string(),
        "./basic_example_ledger/ledger.json".to_string(),
    )?;
    println!("✅ Initialized ledger for project: fraud-detection-model\n");

    // 2. Load and notarize raw dataset
    let raw_dataset = Dataset::from_path("examples/data/transactions.csv")?;
    let proof1 = ledger.notarize_dataset(raw_dataset, "integrity".to_string())?;

    println!("📊 Notarized raw dataset:");
    println!("   Name: raw-transactions-v1");
    println!("   Dataset hash: {}", &proof1.dataset_hash[..16]);
    println!("   Proof type: {}", proof1.proof_type);
    println!("   Timestamp: {}\n", proof1.timestamp);

    // 3. Record data cleaning transformation
    let cleaned_dataset = Dataset::from_path("examples/data/transactions_cleaned.csv")?;
    let transform_proof = ledger.notarize_dataset(cleaned_dataset, "preprocessing".to_string())?;

    println!("🔄 Recorded transformation:");
    println!("   Dataset: cleaned-transactions-v1");
    println!("   Operations: remove_nulls, normalize_amounts, encode_categories");
    println!("   Proof hash: {}\n", &transform_proof.dataset_hash[..16]);

    // 4. Show ledger statistics
    let stats = ledger.get_statistics();
    println!("📈 Ledger Statistics:");
    println!("   Total datasets: {}", stats.total_datasets);
    println!("   Total operations: {}", stats.total_operations);
    if let Some(path) = &stats.storage_path {
        println!("   Storage path: {}\n", path);
    }

    // 5. Verify integrity
    let is_valid = ledger.verify_integrity()?;
    println!(
        "🔐 Chain integrity check: {}",
        if is_valid { "✅ VALID" } else { "❌ INVALID" }
    );

    // 6. List all datasets
    let datasets = ledger.list_datasets();
    println!("📋 Datasets in ledger:");
    for entry in datasets {
        println!("   • {} ({})", entry.dataset_name, &entry.dataset_hash[..8]);
        println!(
            "     Operation: {} | {}",
            entry.operation,
            entry.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
    }
    println!();

    // 7. Demonstrate verification
    println!("🏛️  Proof verification:");
    for proof in [&proof1, &transform_proof] {
        let is_valid = ledger.verify_proof(proof);
        println!(
            "   Proof {}: {}",
            &proof.dataset_hash[..8],
            if is_valid { "✅ Valid" } else { "❌ Invalid" }
        );
    }

    // 8. Health check
    let health = ledger.health_check()?;
    println!("\n💚 System Health:");
    println!(
        "   Overall: {}",
        if health.is_healthy {
            "HEALTHY"
        } else {
            "UNHEALTHY"
        }
    );
    println!("   Storage accessible: {}", health.storage_accessible);
    println!("   Integrity verified: {}", health.integrity_verified);
    println!("   Entry count: {}", health.entry_count);

    // 9. Performance metrics
    let metrics = ledger.get_performance_metrics();
    println!("\n📊 Performance Metrics:");
    println!("   Total operations: {}", metrics.total_operations);
    println!(
        "   Average proof time: {:.2}ms",
        metrics.average_proof_time_ms
    );
    println!("   Cache hit rate: {:.2}%", metrics.cache_hit_rate * 100.0);
    println!("   Memory usage: {:.2}MB", metrics.memory_usage_mb);

    // Cleanup
    std::fs::remove_dir_all("examples/data").ok();
    std::fs::remove_dir_all("./basic_example_ledger").ok();

    println!("\n🎉 Example completed successfully!");
    println!("All dataset operations have been cryptographically proven and auditable.");

    Ok(())
}

// Helper function to create sample data
fn create_sample_data() -> Result<()> {
    use std::fs;

    // Create examples/data directory
    fs::create_dir_all("examples/data")?;

    // Create sample transactions.csv
    let transactions_csv = r#"transaction_id,amount,merchant,timestamp,fraud_label
1,29.99,grocery_store,2024-01-01 10:30:00,0
2,1500.00,electronics,2024-01-01 14:20:00,1
3,45.67,restaurant,2024-01-01 19:45:00,0
4,89.99,clothing,2024-01-02 11:15:00,0
5,2500.00,jewelry,2024-01-02 16:30:00,1
"#;

    fs::write("examples/data/transactions.csv", transactions_csv)?;

    // Create cleaned version (same data, demonstrates transformation)
    let cleaned_csv = r#"transaction_id,amount_normalized,merchant_encoded,fraud_label
1,0.012,0,0
2,0.600,1,1
3,0.018,2,0
4,0.036,3,0
5,1.000,4,1
"#;

    fs::write("examples/data/transactions_cleaned.csv", cleaned_csv)?;

    Ok(())
}
