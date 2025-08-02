// Basic usage example for ZKP Dataset Ledger
//
// This example demonstrates:
// 1. Initializing a ledger
// 2. Notarizing a dataset
// 3. Recording transformations
// 4. Verifying proofs
// 5. Generating audit reports

use std::path::Path;
use zkp_dataset_ledger::{Dataset, Ledger, ProofConfig, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🚀 ZKP Dataset Ledger - Basic Example");
    println!("=====================================\n");

    // 1. Initialize ledger for ML project
    let mut ledger = Ledger::new("fraud-detection-model")?;
    println!("✅ Initialized ledger for project: fraud-detection-model\n");

    // 2. Load and notarize raw dataset
    let raw_dataset = Dataset::from_csv("examples/data/transactions.csv")?;
    let proof1 =
        ledger.notarize_dataset(raw_dataset, "raw-transactions-v1", ProofConfig::default())?;

    println!("📊 Notarized raw dataset:");
    println!("   Name: raw-transactions-v1");
    println!("   Proof size: {} bytes", proof1.size_bytes());
    println!("   Dataset hash: {}\n", proof1.dataset_hash());

    // 3. Record data cleaning transformation
    let cleaned_dataset = Dataset::from_csv("examples/data/transactions_cleaned.csv")?;
    let transform_proof = ledger.record_transformation(
        "raw-transactions-v1",
        "cleaned-transactions-v1",
        vec!["remove_nulls", "normalize_amounts", "encode_categories"],
        cleaned_dataset,
    )?;

    println!("🔄 Recorded transformation:");
    println!("   From: raw-transactions-v1");
    println!("   To: cleaned-transactions-v1");
    println!("   Operations: remove_nulls, normalize_amounts, encode_categories");
    println!("   Proof size: {} bytes\n", transform_proof.size_bytes());

    // 4. Create train/test split with stratification
    let (train_proof, test_proof) = ledger.create_split(
        "cleaned-transactions-v1",
        0.8,                 // 80% train, 20% test
        Some("fraud_label"), // Stratify by fraud label
        Some(42),            // Random seed for reproducibility
    )?;

    println!("📈 Created train/test split:");
    println!("   Train dataset: {}", train_proof.dataset_name());
    println!("   Test dataset: {}", test_proof.dataset_name());
    println!("   Split ratio: 80/20");
    println!("   Stratified by: fraud_label\n");

    // 5. Verify all proofs in the chain
    let dataset_history = ledger.get_dataset_history("test-set-v1")?;
    println!("🔍 Audit trail for test-set-v1:");
    for (i, event) in dataset_history.iter().enumerate() {
        println!(
            "   {}. {} - {} ({})",
            i + 1,
            event.timestamp.format("%Y-%m-%d %H:%M:%S"),
            event.operation,
            event.dataset_name
        );
    }
    println!();

    // 6. Verify chain integrity
    let chain_valid = ledger.verify_chain_integrity()?;
    println!(
        "🔐 Chain integrity check: {}",
        if chain_valid {
            "✅ VALID"
        } else {
            "❌ INVALID"
        }
    );

    // 7. Generate compliance report
    let audit_report = ledger.generate_audit_report(
        Some("raw-transactions-v1"), // Start from raw data
        None,                        // To latest
        true,                        // Include proofs
    )?;

    audit_report.export_json("fraud_detection_audit.json")?;
    audit_report.export_pdf("fraud_detection_audit.pdf")?;

    println!("📋 Generated audit reports:");
    println!("   JSON: fraud_detection_audit.json");
    println!("   PDF: fraud_detection_audit.pdf\n");

    // 8. Demonstrate verification by external party
    println!("🏛️  External verification example:");
    for proof in [&proof1, &transform_proof, &train_proof, &test_proof] {
        let is_valid = ledger.verify_proof(proof)?;
        println!(
            "   Proof {}: {}",
            proof.id(),
            if is_valid { "✅ Valid" } else { "❌ Invalid" }
        );
    }

    println!("\n🎉 Example completed successfully!");
    println!("All dataset operations have been cryptographically proven and auditable.");

    Ok(())
}

// Helper function to create sample data if files don't exist
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
