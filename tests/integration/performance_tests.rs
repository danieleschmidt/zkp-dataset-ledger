use std::time::Instant;
use tempfile::NamedTempFile;
use tokio::test;
use zkp_dataset_ledger::{Dataset, Ledger, Proof, ProofConfig};

#[test]
async fn test_ledger_creation_performance() {
    let start = Instant::now();

    for _ in 0..100 {
        let _ledger = Ledger::new("perf-test-ledger").unwrap();
    }

    let duration = start.elapsed();
    println!("Creating 100 ledgers took: {:?}", duration);
    assert!(duration.as_millis() < 1000); // Should be under 1 second
}

#[test]
async fn test_dataset_loading_performance() {
    // Create a test CSV file with multiple rows
    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, b"id,name,value,category\n").unwrap();

    // Write 1000 rows of test data
    for i in 0..1000 {
        let line = format!("{},name_{},value_{},category_{}\n", i, i, i, i % 10);
        std::io::Write::write_all(&mut temp_file, line.as_bytes()).unwrap();
    }

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let start = Instant::now();

    for _ in 0..50 {
        let _dataset = Dataset::from_path(&temp_path).unwrap();
    }

    let duration = start.elapsed();
    println!("Loading dataset 50 times took: {:?}", duration);
    assert!(duration.as_millis() < 5000); // Should be under 5 seconds

    std::fs::remove_file(temp_path).ok();
}

#[test]
async fn test_proof_generation_performance() {
    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, b"data,label\n").unwrap();

    // Create dataset with 100 rows
    for i in 0..100 {
        let line = format!("{},{}\n", i, i % 2);
        std::io::Write::write_all(&mut temp_file, line.as_bytes()).unwrap();
    }

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let dataset = Dataset::from_path(&temp_path).unwrap();
    let config = ProofConfig::default();

    let start = Instant::now();

    for _ in 0..10 {
        let _proof = Proof::generate(&dataset, &config).unwrap();
    }

    let duration = start.elapsed();
    println!("Generating 10 proofs took: {:?}", duration);
    assert!(duration.as_millis() < 3000); // Should be under 3 seconds

    std::fs::remove_file(temp_path).ok();
}

#[test]
async fn test_proof_verification_performance() {
    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, b"x,y\n1,2\n3,4\n").unwrap();

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let dataset = Dataset::from_path(&temp_path).unwrap();
    let config = ProofConfig::default();
    let proof = Proof::generate(&dataset, &config).unwrap();

    let start = Instant::now();

    for _ in 0..100 {
        assert!(proof.verify().unwrap());
    }

    let duration = start.elapsed();
    println!("Verifying proof 100 times took: {:?}", duration);
    assert!(duration.as_millis() < 1000); // Should be under 1 second

    std::fs::remove_file(temp_path).ok();
}

#[test]
async fn test_ledger_notarization_performance() {
    let mut ledger = Ledger::new("perf-test-ledger").unwrap();

    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, b"data\ntest\n").unwrap();

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let dataset = Dataset::from_path(&temp_path).unwrap();
    let config = ProofConfig::default();

    let start = Instant::now();

    for i in 0..50 {
        let dataset_name = format!("dataset-{}", i);
        let _proof = ledger
            .notarize_dataset(dataset.clone(), &dataset_name, config.clone())
            .unwrap();
    }

    let duration = start.elapsed();
    println!("Notarizing 50 datasets took: {:?}", duration);
    assert!(duration.as_secs() < 10); // Should be under 10 seconds

    std::fs::remove_file(temp_path).ok();
}

#[test]
async fn test_ledger_query_performance() {
    let mut ledger = Ledger::new("perf-test-ledger").unwrap();

    // Add 100 entries to the ledger
    for i in 0..100 {
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut temp_file,
            format!("data{}\nvalue{}\n", i, i).as_bytes(),
        )
        .unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();
        let dataset_name = format!("dataset-{}", i);

        ledger
            .notarize_dataset(dataset, &dataset_name, config)
            .unwrap();
        std::fs::remove_file(temp_path).ok();
    }

    let start = Instant::now();

    // Query for specific datasets
    for i in 0..20 {
        let dataset_name = format!("dataset-{}", i * 5); // Query every 5th dataset
        let _history = ledger.get_dataset_history(&dataset_name).unwrap();
    }

    let duration = start.elapsed();
    println!("Querying 20 dataset histories took: {:?}", duration);
    assert!(duration.as_millis() < 2000); // Should be under 2 seconds
}

#[test]
async fn test_ledger_integrity_verification_performance() {
    let mut ledger = Ledger::new("perf-test-ledger").unwrap();

    // Add 50 entries
    for i in 0..50 {
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(
            &mut temp_file,
            format!("id,val\n{},{}\n", i, i * 2).as_bytes(),
        )
        .unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();
        let dataset_name = format!("integrity-test-{}", i);

        ledger
            .notarize_dataset(dataset, &dataset_name, config)
            .unwrap();
        std::fs::remove_file(temp_path).ok();
    }

    let start = Instant::now();

    // Verify chain integrity
    assert!(ledger.verify_chain_integrity().unwrap());

    let duration = start.elapsed();
    println!(
        "Verifying chain integrity with 50 entries took: {:?}",
        duration
    );
    assert!(duration.as_secs() < 5); // Should be under 5 seconds
}

#[test]
async fn test_ledger_summary_performance() {
    let mut ledger = Ledger::new("perf-test-ledger").unwrap();

    // Add 200 entries
    for i in 0..200 {
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, format!("x,y\n{},{}\n", i, i + 1).as_bytes())
            .unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();
        let dataset_name = format!("summary-test-{}", i);

        ledger
            .notarize_dataset(dataset, &dataset_name, config)
            .unwrap();
        std::fs::remove_file(temp_path).ok();
    }

    let start = Instant::now();

    let _summary = ledger.get_summary().unwrap();

    let duration = start.elapsed();
    println!("Generating summary with 200 entries took: {:?}", duration);
    assert!(duration.as_millis() < 3000); // Should be under 3 seconds
}

#[test]
async fn test_large_dataset_handling() {
    let mut temp_file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, b"id,data,category,value\n").unwrap();

    // Create larger dataset with 5000 rows
    for i in 0..5000 {
        let line = format!("{},data_{},cat_{},{}\n", i, i, i % 100, i * 3.14);
        std::io::Write::write_all(&mut temp_file, line.as_bytes()).unwrap();
    }

    let temp_path = temp_file.path().with_extension("csv");
    std::fs::copy(temp_file.path(), &temp_path).unwrap();

    let start = Instant::now();

    let dataset = Dataset::from_path(&temp_path).unwrap();
    let config = ProofConfig::default();
    let proof = Proof::generate(&dataset, &config).unwrap();

    let duration = start.elapsed();
    println!("Processing large dataset (5000 rows) took: {:?}", duration);
    assert!(duration.as_secs() < 30); // Should be under 30 seconds

    // Verify the proof
    let verify_start = Instant::now();
    assert!(proof.verify().unwrap());
    let verify_duration = verify_start.elapsed();

    println!("Verifying large dataset proof took: {:?}", verify_duration);
    assert!(verify_duration.as_secs() < 5); // Verification should be fast

    std::fs::remove_file(temp_path).ok();
}

#[test]
async fn test_memory_usage_during_operations() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let peak_memory = Arc::new(AtomicUsize::new(0));
    let mut ledger = Ledger::new("memory-test-ledger").unwrap();

    // Simulate memory-intensive operations
    for i in 0..100 {
        let mut temp_file = NamedTempFile::new().unwrap();

        // Create moderately sized dataset
        std::io::Write::write_all(&mut temp_file, b"col1,col2,col3,col4,col5\n").unwrap();
        for j in 0..1000 {
            let line = format!("{},{},{},{},{}\n", i * 1000 + j, j, j * 2, j * 3, j * 4);
            std::io::Write::write_all(&mut temp_file, line.as_bytes()).unwrap();
        }

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();
        let dataset_name = format!("memory-test-{}", i);

        let _proof = ledger
            .notarize_dataset(dataset, &dataset_name, config)
            .unwrap();

        std::fs::remove_file(temp_path).ok();

        // Simple memory usage tracking (in real implementation we'd use proper profiling)
        if i % 10 == 0 {
            println!("Processed {} datasets", i + 1);
        }
    }

    // This test mainly ensures we don't run out of memory or crash
    println!("Successfully processed 100 moderate-sized datasets");
}

#[test]
async fn test_concurrent_proof_generation() {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    // Limit concurrent operations to avoid overwhelming the system
    let semaphore = Arc::new(Semaphore::new(5));
    let mut handles = vec![];

    for i in 0..20 {
        let semaphore_clone = Arc::clone(&semaphore);
        let handle = tokio::spawn(async move {
            let _permit = semaphore_clone.acquire().await.unwrap();

            let mut temp_file = NamedTempFile::new().unwrap();
            std::io::Write::write_all(&mut temp_file, format!("data{}\ntest{}\n", i, i).as_bytes())
                .unwrap();

            let temp_path = temp_file.path().with_extension("csv");
            std::fs::copy(temp_file.path(), &temp_path).unwrap();

            let dataset = Dataset::from_path(&temp_path).unwrap();
            let config = ProofConfig::default();
            let proof = Proof::generate(&dataset, &config).unwrap();

            assert!(proof.verify().unwrap());

            std::fs::remove_file(temp_path).ok();
            i
        });
        handles.push(handle);
    }

    let start = Instant::now();

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    let duration = start.elapsed();
    println!("20 concurrent proof generations took: {:?}", duration);
    assert!(duration.as_secs() < 15); // Should be under 15 seconds with concurrency limit
}

#[test]
async fn test_export_performance() {
    let mut ledger = Ledger::new("export-perf-test").unwrap();

    // Add 100 entries
    for i in 0..100 {
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, format!("export,test,{}\n", i).as_bytes())
            .unwrap();

        let temp_path = temp_file.path().with_extension("csv");
        std::fs::copy(temp_file.path(), &temp_path).unwrap();

        let dataset = Dataset::from_path(&temp_path).unwrap();
        let config = ProofConfig::default();
        let dataset_name = format!("export-dataset-{}", i);

        ledger
            .notarize_dataset(dataset, &dataset_name, config)
            .unwrap();
        std::fs::remove_file(temp_path).ok();
    }

    let start = Instant::now();

    let query = zkp_dataset_ledger::ledger::LedgerQuery::default();
    let _json_export = ledger.export_to_json(&query).unwrap();

    let duration = start.elapsed();
    println!("Exporting 100 entries to JSON took: {:?}", duration);
    assert!(duration.as_millis() < 2000); // Should be under 2 seconds
}
