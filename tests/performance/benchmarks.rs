use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tempfile::TempDir;
use zkp_dataset_ledger::{Dataset, Ledger, ProofConfig};
use std::time::Duration;

/// Performance benchmarks for ZKP Dataset Ledger components
/// These benchmarks measure critical operations to ensure performance targets are met

pub fn dataset_loading_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_loading");
    
    // Test different dataset sizes
    let dataset_sizes = vec![100, 1000, 10000, 100000];
    
    for size in dataset_sizes {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let csv_path = temp_dir.path().join(format!("benchmark_{}.csv", size));
        
        // Generate CSV data
        let mut content = String::from("id,name,value,category,timestamp\n");
        for i in 1..=size {
            content.push_str(&format!(
                "{},item_{},{:.2},category_{},{}\n",
                i, i, i as f64 * 1.5, i % 10, 1640995200 + i * 60
            ));
        }
        std::fs::write(&csv_path, content).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("csv_parsing", size),
            &csv_path,
            |b, path| {
                b.iter(|| {
                    let dataset = Dataset::from_path(black_box(path))
                        .expect("Failed to load dataset");
                    black_box(dataset);
                });
            },
        );
    }
    
    group.finish();
}

pub fn proof_generation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for crypto operations
    
    let dataset_sizes = vec![100, 1000, 10000];
    
    for size in dataset_sizes {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join("benchmark_ledger");
        let csv_path = temp_dir.path().join(format!("benchmark_{}.csv", size));
        
        // Generate test data
        let mut content = String::from("id,value,score,category\n");
        for i in 1..=size {
            content.push_str(&format!("{},{},{:.1},{}\n", i, i * 10, i as f64 / 10.0, i % 5));
        }
        std::fs::write(&csv_path, content).unwrap();
        
        let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
        let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
        
        // Benchmark basic proof generation
        group.bench_with_input(
            BenchmarkId::new("basic_proof", size),
            &size,
            |b, _size| {
                b.iter(|| {
                    let proof = ledger.notarize_dataset(
                        black_box(dataset.clone()),
                        &format!("benchmark-{}", rand::random::<u32>()),
                        black_box(ProofConfig::fast())
                    ).expect("Failed to generate proof");
                    black_box(proof);
                });
            },
        );
        
        // Benchmark statistical proof generation
        group.bench_with_input(
            BenchmarkId::new("statistical_proof", size),
            &size,
            |b, _size| {
                b.iter(|| {
                    let proof = ledger.notarize_dataset(
                        black_box(dataset.clone()),
                        &format!("stats-benchmark-{}", rand::random::<u32>()),
                        black_box(ProofConfig::statistical())
                    ).expect("Failed to generate statistical proof");
                    black_box(proof);
                });
            },
        );
        
        // Benchmark privacy-preserving proof generation
        if size <= 1000 { // Skip large datasets for privacy proofs to keep benchmark time reasonable
            group.bench_with_input(
                BenchmarkId::new("privacy_proof", size),
                &size,
                |b, _size| {
                    b.iter(|| {
                        let proof = ledger.notarize_dataset(
                            black_box(dataset.clone()),
                            &format!("privacy-benchmark-{}", rand::random::<u32>()),
                            black_box(ProofConfig::privacy_preserving())
                        ).expect("Failed to generate privacy proof");
                        black_box(proof);
                    });
                },
            );
        }
    }
    
    group.finish();
}

pub fn proof_verification_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_verification");
    
    // Pre-generate proofs of different types
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("verification_ledger");
    let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
    
    let mut proofs = Vec::new();
    let proof_types = vec![
        ("basic", ProofConfig::fast()),
        ("statistical", ProofConfig::statistical()),
        ("privacy", ProofConfig::privacy_preserving()),
    ];
    
    for (proof_type, config) in proof_types {
        let csv_path = temp_dir.path().join(format!("{}_data.csv", proof_type));
        let content = "id,value,category\n1,100,A\n2,200,B\n3,300,A\n4,400,C\n5,500,B\n";
        std::fs::write(&csv_path, content).unwrap();
        
        let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
        let proof = ledger.notarize_dataset(
            dataset,
            &format!("verification-{}", proof_type),
            config
        ).expect("Failed to generate proof");
        
        proofs.push((proof_type, proof));
    }
    
    // Benchmark verification for each proof type
    for (proof_type, proof) in proofs {
        group.bench_function(
            BenchmarkId::new("verify", proof_type),
            |b| {
                b.iter(|| {
                    let is_valid = ledger.verify_proof(black_box(&proof))
                        .expect("Verification should not fail");
                    black_box(is_valid);
                });
            },
        );
    }
    
    group.finish();
}

pub fn merkle_tree_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree");
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("merkle_ledger");
    let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
    
    // Benchmark Merkle tree operations with increasing number of transactions
    let transaction_counts = vec![10, 100, 1000];
    
    for tx_count in transaction_counts {
        // Pre-populate ledger with transactions
        for i in 1..=tx_count {
            let csv_path = temp_dir.path().join(format!("tx_{}.csv", i));
            let content = format!("id,value\n1,{}\n2,{}\n", i, i * 2);
            std::fs::write(&csv_path, content).unwrap();
            
            let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
            ledger.notarize_dataset(
                dataset,
                &format!("tx-{}", i),
                ProofConfig::fast()
            ).expect("Failed to notarize");
        }
        
        // Benchmark root computation
        group.bench_function(
            BenchmarkId::new("compute_root", tx_count),
            |b| {
                b.iter(|| {
                    let root = ledger.get_merkle_root()
                        .expect("Failed to compute root");
                    black_box(root);
                });
            },
        );
        
        // Benchmark audit trail retrieval
        group.bench_function(
            BenchmarkId::new("audit_trail", tx_count),
            |b| {
                b.iter(|| {
                    let trail = ledger.get_audit_trail()
                        .expect("Failed to get audit trail");
                    black_box(trail);
                });
            },
        );
    }
    
    group.finish();
}

pub fn storage_backend_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_backend");
    
    // Test different storage backends
    let backends = vec!["rocksdb", "memory"];
    
    for backend in backends {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let ledger_path = temp_dir.path().join(format!("{}_ledger", backend));
        
        // Configure ledger with specific backend
        let mut ledger = if backend == "memory" {
            Ledger::with_memory_storage().expect("Failed to create memory ledger")
        } else {
            Ledger::new(&ledger_path).expect("Failed to create RocksDB ledger")
        };
        
        // Create test dataset
        let csv_path = temp_dir.path().join("storage_test.csv");
        let mut content = String::from("id,data,timestamp\n");
        for i in 1..=100 {
            content.push_str(&format!("{},data_{},{}\n", i, i, 1640995200 + i));
        }
        std::fs::write(&csv_path, content).unwrap();
        
        let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
        
        // Benchmark write operations
        group.bench_function(
            BenchmarkId::new("write", backend),
            |b| {
                b.iter(|| {
                    let proof = ledger.notarize_dataset(
                        black_box(dataset.clone()),
                        &format!("storage-test-{}", rand::random::<u32>()),
                        black_box(ProofConfig::fast())
                    ).expect("Failed to write to storage");
                    black_box(proof);
                });
            },
        );
        
        // Add some data for read benchmarks
        for i in 1..=50 {
            ledger.notarize_dataset(
                dataset.clone(),
                &format!("read-test-{}", i),
                ProofConfig::fast()
            ).expect("Failed to setup read test data");
        }
        
        // Benchmark read operations
        group.bench_function(
            BenchmarkId::new("read", backend),
            |b| {
                b.iter(|| {
                    let trail = ledger.get_audit_trail()
                        .expect("Failed to read from storage");
                    black_box(trail);
                });
            },
        );
    }
    
    group.finish();
}

pub fn memory_usage_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Test memory efficiency with different dataset sizes
    let dataset_sizes = vec![1000, 10000, 100000];
    
    for size in dataset_sizes {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let csv_path = temp_dir.path().join(format!("memory_test_{}.csv", size));
        
        // Generate large dataset
        let mut content = String::from("id,data1,data2,data3,data4,data5\n");
        for i in 1..=size {
            content.push_str(&format!("{},{},{},{},{},{}\n", 
                i, i * 2, i * 3, i * 4, i * 5, i * 6));
        }
        std::fs::write(&csv_path, content).unwrap();
        
        group.bench_function(
            BenchmarkId::new("streaming_load", size),
            |b| {
                b.iter(|| {
                    // Test streaming dataset loading (should use constant memory)
                    let dataset = Dataset::from_path_streaming(black_box(&csv_path))
                        .expect("Failed to load dataset with streaming");
                    black_box(dataset);
                });
            },
        );
        
        if size <= 10000 { // Only test full loading for smaller datasets
            group.bench_function(
                BenchmarkId::new("full_load", size),
                |b| {
                    b.iter(|| {
                        let dataset = Dataset::from_path(black_box(&csv_path))
                            .expect("Failed to load full dataset");
                        black_box(dataset);
                    });
                },
            );
        }
    }
    
    group.finish();
}

pub fn concurrent_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("concurrent_ledger");
    let ledger = std::sync::Arc::new(std::sync::Mutex::new(
        Ledger::new(&ledger_path).expect("Failed to initialize ledger")
    ));
    
    // Create test datasets
    let mut datasets = Vec::new();
    for i in 1..=10 {
        let csv_path = temp_dir.path().join(format!("concurrent_{}.csv", i));
        let content = format!("id,value\n1,{}\n2,{}\n3,{}\n", i * 10, i * 20, i * 30);
        std::fs::write(&csv_path, content).unwrap();
        
        let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
        datasets.push(dataset);
    }
    
    // Benchmark concurrent proof generation
    group.bench_function("concurrent_proofs", |b| {
        b.iter(|| {
            use std::thread;
            
            let handles: Vec<_> = datasets.iter().enumerate().map(|(i, dataset)| {
                let ledger = ledger.clone();
                let dataset = dataset.clone();
                
                thread::spawn(move || {
                    let mut lg = ledger.lock().unwrap();
                    let proof = lg.notarize_dataset(
                        dataset,
                        &format!("concurrent-{}-{}", i, rand::random::<u32>()),
                        ProofConfig::fast()
                    ).expect("Failed to generate concurrent proof");
                    black_box(proof);
                })
            }).collect();
            
            for handle in handles {
                handle.join().expect("Thread should complete successfully");
            }
        });
    });
    
    group.finish();
}

// Performance regression tests
pub fn regression_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_tests");
    
    // These benchmarks establish baseline performance expectations
    // They should be updated when intentional performance changes are made
    
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let ledger_path = temp_dir.path().join("regression_ledger");
    let mut ledger = Ledger::new(&ledger_path).expect("Failed to initialize ledger");
    
    // Standard 1000-row dataset
    let csv_path = temp_dir.path().join("standard_1k.csv");
    let mut content = String::from("id,name,value,category,timestamp\n");
    for i in 1..=1000 {
        content.push_str(&format!("{},item_{},{:.2},cat_{},{}\n", 
            i, i, i as f64 * 1.23, i % 10, 1640995200 + i * 60));
    }
    std::fs::write(&csv_path, content).unwrap();
    
    let dataset = Dataset::from_path(&csv_path).expect("Failed to load dataset");
    
    // Performance targets (these should be updated based on actual performance)
    group.bench_function("target_1k_rows_basic_proof", |b| {
        b.iter(|| {
            let proof = ledger.notarize_dataset(
                black_box(dataset.clone()),
                &format!("regression-{}", rand::random::<u32>()),
                black_box(ProofConfig::fast())
            ).expect("Failed to generate proof");
            black_box(proof);
        });
    });
    
    // Target: <5s for 1M rows (tested with smaller dataset for CI)
    // Target: <100ms verification
    // Target: <2GB memory for 1GB dataset
    
    group.finish();
}

criterion_group!(
    benches,
    dataset_loading_benchmark,
    proof_generation_benchmark,
    proof_verification_benchmark,
    merkle_tree_benchmark,
    storage_backend_benchmark,
    memory_usage_benchmark,
    concurrent_operations_benchmark,
    regression_tests
);

criterion_main!(benches);