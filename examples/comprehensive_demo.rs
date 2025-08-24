//! Comprehensive demonstration of ZKP Dataset Ledger capabilities
//! This example showcases core features in a realistic workflow.

use zkp_dataset_ledger::{Dataset, Ledger, Result};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 ZKP Dataset Ledger - Comprehensive Demo");
    println!("=========================================\n");

    // Initialize logging
    env_logger::init();

    // Demo 1: Basic Dataset Notarization
    println!("📋 Demo 1: Basic Dataset Notarization");
    demo_basic_notarization().await?;

    // Demo 2: Dataset Analysis and Verification
    println!("\n🔍 Demo 2: Dataset Analysis and Verification");
    demo_analysis_verification().await?;

    // Demo 3: Performance Monitoring
    println!("\n📊 Demo 3: Performance Monitoring");
    demo_performance_monitoring().await?;

    // Demo 4: Complete ML Pipeline
    println!("\n🤖 Demo 4: ML Pipeline Auditing");
    demo_ml_pipeline_audit().await?;

    println!("\n✅ All demos completed successfully!");
    println!("🎉 ZKP Dataset Ledger is production-ready!");

    Ok(())
}

async fn demo_basic_notarization() -> Result<()> {
    // Create a ledger
    let mut ledger = Ledger::with_storage(
        "demo-project".to_string(),
        "./demo_ledger/ledger.json".to_string(),
    )?;

    // Create test data files
    std::fs::create_dir_all("./demo_data")?;

    // Training dataset
    std::fs::write(
        "./demo_data/training.csv",
        "name,age,salary,department\nAlice,25,50000,Engineering\nBob,30,60000,Sales\nCharlie,35,70000,Marketing\n"
    )?;

    // Test dataset
    std::fs::write(
        "./demo_data/test.csv",
        "name,age,salary,department\nDave,28,55000,Engineering\nEve,32,65000,Sales\n",
    )?;

    let training_dataset = Dataset::new(
        "training-data-v1".to_string(),
        "./demo_data/training.csv".to_string(),
    )?;

    let test_dataset = Dataset::new(
        "test-data-v1".to_string(),
        "./demo_data/test.csv".to_string(),
    )?;

    println!(
        "  📊 Training dataset: {} rows, {} columns",
        training_dataset.row_count.unwrap_or(0),
        training_dataset.column_count.unwrap_or(0)
    );

    println!(
        "  📊 Test dataset: {} rows, {} columns",
        test_dataset.row_count.unwrap_or(0),
        test_dataset.column_count.unwrap_or(0)
    );

    // Notarize datasets
    let training_proof = ledger.notarize_dataset(training_dataset, "integrity".to_string())?;
    println!(
        "  ✅ Training data notarized: {}",
        &training_proof.dataset_hash[..16]
    );

    let test_proof = ledger.notarize_dataset(test_dataset, "integrity".to_string())?;
    println!(
        "  ✅ Test data notarized: {}",
        &test_proof.dataset_hash[..16]
    );

    println!(
        "  📈 Ledger now contains {} datasets",
        ledger.get_statistics().total_datasets
    );

    Ok(())
}

async fn demo_analysis_verification() -> Result<()> {
    let ledger = Ledger::with_storage(
        "demo-project".to_string(),
        "./demo_ledger/ledger.json".to_string(),
    )?;

    // Verify integrity
    let is_valid = ledger.verify_integrity()?;
    println!(
        "  🔍 Ledger integrity: {}",
        if is_valid { "VALID" } else { "INVALID" }
    );

    // Health check
    let health = ledger.health_check()?;
    println!(
        "  💚 System health: {}",
        if health.is_healthy {
            "HEALTHY"
        } else {
            "UNHEALTHY"
        }
    );
    println!("  📦 Storage accessible: {}", health.storage_accessible);
    println!("  🔐 Integrity verified: {}", health.integrity_verified);
    println!("  📊 Entry count: {}", health.entry_count);

    // List all datasets
    let datasets = ledger.list_datasets();
    println!("  📋 Datasets in ledger:");
    for entry in datasets {
        println!(
            "    • {} ({})",
            entry.dataset_name,
            &entry.dataset_hash[..16]
        );
        println!(
            "      Operation: {} | {}",
            entry.operation,
            entry.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
    }

    Ok(())
}

async fn demo_performance_monitoring() -> Result<()> {
    let ledger = Ledger::with_storage(
        "demo-project".to_string(),
        "./demo_ledger/ledger.json".to_string(),
    )?;

    // Get performance metrics
    let metrics = ledger.get_performance_metrics();
    println!("  📊 Performance Metrics:");
    println!("    Total operations: {}", metrics.total_operations);
    println!(
        "    Average proof time: {:.2}ms",
        metrics.average_proof_time_ms
    );
    println!("    Cache hit rate: {:.2}%", metrics.cache_hit_rate * 100.0);
    println!("    Memory usage: {:.2}MB", metrics.memory_usage_mb);
    println!(
        "    Throughput: {:.2} ops/sec",
        metrics.throughput_ops_per_sec
    );

    Ok(())
}

async fn demo_ml_pipeline_audit() -> Result<()> {
    let mut ml_ledger = Ledger::with_storage(
        "ml-pipeline".to_string(),
        "./ml_pipeline_ledger/ledger.json".to_string(),
    )?;

    println!("  🔄 Simulating ML pipeline workflow...");

    // Step 1: Raw data ingestion
    std::fs::create_dir_all("./ml_data")?;
    std::fs::write(
        "./ml_data/raw_data.csv",
        "user_id,feature1,feature2,label\n1,0.5,0.3,1\n2,0.8,0.1,0\n3,0.2,0.9,1\n4,0.7,0.4,0\n",
    )?;

    let raw_data = Dataset::new(
        "raw-customer-data".to_string(),
        "./ml_data/raw_data.csv".to_string(),
    )?;

    let _raw_proof = ml_ledger.notarize_dataset(raw_data, "data-ingestion".to_string())?;
    println!("    ✅ Raw data ingested and notarized");

    // Step 2: Data preprocessing
    std::fs::write(
        "./ml_data/cleaned_data.csv",
        "user_id,feature1,feature2,label\n1,0.5,0.3,1\n2,0.8,0.1,0\n3,0.2,0.9,1\n4,0.7,0.4,0\n",
    )?;

    let cleaned_data = Dataset::new(
        "cleaned-customer-data".to_string(),
        "./ml_data/cleaned_data.csv".to_string(),
    )?;

    let _clean_proof = ml_ledger.notarize_dataset(cleaned_data, "preprocessing".to_string())?;
    println!("    ✅ Data cleaned and preprocessing recorded");

    // Step 3: Feature engineering
    std::fs::write(
        "./ml_data/features.csv",
        "user_id,feature1,feature2,feature1_squared,feature_interaction,label\n1,0.5,0.3,0.25,0.15,1\n2,0.8,0.1,0.64,0.08,0\n"
    )?;

    let feature_data = Dataset::new(
        "engineered-features".to_string(),
        "./ml_data/features.csv".to_string(),
    )?;

    let _feature_proof =
        ml_ledger.notarize_dataset(feature_data, "feature-engineering".to_string())?;
    println!("    ✅ Feature engineering completed and verified");

    // Step 4: Model training dataset split
    std::fs::write(
        "./ml_data/train_split.csv",
        "user_id,feature1,feature2,feature1_squared,feature_interaction,label\n1,0.5,0.3,0.25,0.15,1\n"
    )?;

    let train_split = Dataset::new(
        "train-split".to_string(),
        "./ml_data/train_split.csv".to_string(),
    )?;

    let _split_proof = ml_ledger.notarize_dataset(train_split, "train-test-split".to_string())?;
    println!("    ✅ Train/test split recorded with cryptographic proof");

    // Generate final audit report
    println!("\n  📋 ML Pipeline Audit Report:");
    let final_stats = ml_ledger.get_statistics();
    println!("    Total pipeline steps: {}", final_stats.total_operations);
    println!("    Datasets tracked: {}", final_stats.total_datasets);

    // Verify all proofs in the pipeline
    let pipeline_datasets = ml_ledger.list_datasets();
    println!("    Proof verification results:");
    for entry in pipeline_datasets {
        let is_valid = ml_ledger.verify_proof(&entry.proof);
        println!(
            "      • {}: {}",
            entry.dataset_name,
            if is_valid { "✅ VALID" } else { "❌ INVALID" }
        );
    }

    // Cleanup
    std::fs::remove_dir_all("./demo_data").ok();
    std::fs::remove_dir_all("./ml_data").ok();
    std::fs::remove_dir_all("./demo_ledger").ok();
    std::fs::remove_dir_all("./ml_pipeline_ledger").ok();

    Ok(())
}
