// Performance benchmark tests using criterion
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use zkp_dataset_ledger::{Ledger, Dataset, LedgerConfig, ProofConfig};
use crate::fixtures::{TestLedger, TestDataGenerator};
use std::time::Duration;

pub fn bench_ledger_initialization(c: &mut Criterion) {
    c.bench_function("ledger_init", |b| {
        b.iter(|| {
            let test_ledger = TestLedger::new();
            let _ledger = Ledger::new(
                black_box(test_ledger.path()),
                black_box(LedgerConfig::default())
            ).expect("Failed to initialize ledger");
        });
    });
}

pub fn bench_dataset_loading(c: &mut Criterion) {
    let generator = TestDataGenerator::new();
    let mut group = c.benchmark_group("dataset_loading");
    
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let dataset_path = generator.create_medium_csv(&format!("bench_{}", size), *size);
        
        group.bench_with_input(BenchmarkId::new("csv_loading", size), size, |b, _| {
            b.iter(|| {
                let _dataset = Dataset::from_path(black_box(&dataset_path))
                    .expect("Failed to load dataset");
            });
        });
    }
    
    group.finish();
}

pub fn bench_proof_generation(c: &mut Criterion) {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    let mut group = c.benchmark_group("proof_generation");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for crypto ops
    
    for size in [100, 1_000, 10_000].iter() {
        let dataset_path = generator.create_medium_csv(&format!("proof_bench_{}", size), *size);
        let dataset = Dataset::from_path(&dataset_path)
            .expect("Failed to load dataset");
        
        group.bench_with_input(BenchmarkId::new("row_count_proof", size), size, |b, i| {
            b.iter(|| {
                let _proof = ledger.notarize_dataset(
                    black_box(dataset.clone()),
                    black_box(&format!("bench-dataset-{}-{}", i, rand::random::<u32>())),
                    black_box(ProofConfig::default())
                ).expect("Failed to generate proof");
            });
        });
    }
    
    group.finish();
}

pub fn bench_proof_verification(c: &mut Criterion) {
    let test_ledger = TestLedger::new();
    let mut ledger = Ledger::new(test_ledger.path(), LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_medium_csv("verify_bench", 1000);
    let dataset = Dataset::from_path(&dataset_path)
        .expect("Failed to load dataset");
    
    let proof = ledger.notarize_dataset(
        dataset,
        "verification-benchmark",
        ProofConfig::default()
    ).expect("Failed to generate proof for verification benchmark");
    
    c.bench_function("proof_verification", |b| {
        b.iter(|| {
            let is_valid = ledger.verify_proof(black_box(&proof))
                .expect("Failed to verify proof");
            assert!(is_valid);
        });
    });
}

pub fn bench_merkle_tree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree");
    
    for leaf_count in [100, 1_000, 10_000, 100_000].iter() {
        let leaves: Vec<Vec<u8>> = (0..*leaf_count)
            .map(|i| format!("leaf_{}", i).into_bytes())
            .collect();
        
        group.bench_with_input(BenchmarkId::new("construction", leaf_count), leaf_count, |b, _| {
            b.iter(|| {
                let _tree = zkp_dataset_ledger::crypto::MerkleTree::new(
                    black_box(leaves.clone()),
                    black_box("sha3-256")
                ).expect("Failed to construct Merkle tree");
            });
        });
        
        // Benchmark proof generation for constructed tree
        let tree = zkp_dataset_ledger::crypto::MerkleTree::new(leaves.clone(), "sha3-256")
            .expect("Failed to construct tree for proof benchmark");
        
        group.bench_with_input(BenchmarkId::new("proof_generation", leaf_count), leaf_count, |b, _| {
            b.iter(|| {
                let index = black_box(rand::random::<usize>() % leaf_count);
                let _proof = tree.generate_proof(index)
                    .expect("Failed to generate Merkle proof");
            });
        });
    }
    
    group.finish();
}

pub fn bench_hash_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_functions");
    
    let data_sizes = [1_000, 10_000, 100_000, 1_000_000]; // Bytes
    let algorithms = ["sha3-256", "blake3"];
    
    for algorithm in algorithms.iter() {
        for size in data_sizes.iter() {
            let data = vec![0xAB; *size];
            let hasher = zkp_dataset_ledger::crypto::Hash::new(algorithm)
                .expect("Failed to create hasher");
            
            group.bench_with_input(
                BenchmarkId::new(format!("{}_hash", algorithm), size), 
                size, 
                |b, _| {
                    b.iter(|| {
                        let _hash = hasher.hash(black_box(&data));
                    });
                }
            );
        }
    }
    
    group.finish();
}

pub fn bench_concurrent_operations(c: &mut Criterion) {
    let test_ledger = TestLedger::new();
    let ledger_path = test_ledger.path().to_path_buf();
    let _ledger = Ledger::new(&ledger_path, LedgerConfig::default())
        .expect("Failed to initialize ledger");
    
    let generator = TestDataGenerator::new();
    
    c.bench_function("concurrent_notarization", |b| {
        b.iter(|| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let mut handles = Vec::new();
                
                for i in 0..4 { // 4 concurrent operations
                    let ledger_path = ledger_path.clone();
                    let dataset_path = generator.create_small_csv(&format!("concurrent_{}", i));
                    
                    let handle = tokio::spawn(async move {
                        let mut ledger = Ledger::open(&ledger_path, LedgerConfig::default())
                            .expect("Failed to open ledger");
                        
                        let dataset = Dataset::from_path(&dataset_path)
                            .expect("Failed to load dataset");
                        
                        ledger.notarize_dataset(
                            dataset,
                            &format!("concurrent-{}-{}", i, rand::random::<u32>()),
                            ProofConfig::default()
                        ).expect("Failed to notarize dataset")
                    });
                    
                    handles.push(handle);
                }
                
                for handle in handles {
                    let _proof = handle.await.expect("Concurrent task failed");
                }
            });
        });
    });
}

pub fn bench_storage_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_backends");
    
    // Benchmark RocksDB
    let rocks_ledger = TestLedger::with_name("rocks_bench");
    let rocks_config = LedgerConfig {
        storage_backend: "rocksdb".to_string(),
        ..Default::default()
    };
    
    group.bench_function("rocksdb_write", |b| {
        b.iter(|| {
            let mut ledger = Ledger::new(rocks_ledger.path(), rocks_config.clone())
                .expect("Failed to create RocksDB ledger");
            
            let generator = TestDataGenerator::new();
            let dataset_path = generator.create_small_csv("rocks_bench");
            let dataset = Dataset::from_path(&dataset_path)
                .expect("Failed to load dataset");
            
            let _proof = ledger.notarize_dataset(
                black_box(dataset),
                black_box(&format!("rocks-{}", rand::random::<u32>())),
                black_box(ProofConfig::default())
            ).expect("Failed to notarize with RocksDB");
        });
    });
    
    // PostgreSQL benchmark (if available)
    if let Ok(db_url) = std::env::var("BENCH_DATABASE_URL") {
        let pg_config = LedgerConfig {
            storage_backend: "postgres".to_string(),
            postgres_connection_string: Some(db_url),
            ..Default::default()
        };
        
        group.bench_function("postgres_write", |b| {
            b.iter(|| {
                let pg_ledger = TestLedger::with_name("pg_bench");
                let mut ledger = Ledger::new(pg_ledger.path(), pg_config.clone());
                
                if let Ok(mut ledger) = ledger {
                    let generator = TestDataGenerator::new();
                    let dataset_path = generator.create_small_csv("pg_bench");
                    let dataset = Dataset::from_path(&dataset_path)
                        .expect("Failed to load dataset");
                    
                    let _proof = ledger.notarize_dataset(
                        black_box(dataset),
                        black_box(&format!("pg-{}", rand::random::<u32>())),
                        black_box(ProofConfig::default())
                    ).expect("Failed to notarize with PostgreSQL");
                }
            });
        });
    }
    
    group.finish();
}

pub fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(BenchmarkId::new("dataset_processing", size), size, |b, s| {
            b.iter(|| {
                let generator = TestDataGenerator::new();
                let dataset_path = generator.create_medium_csv(&format!("memory_bench_{}", s), *s);
                
                let dataset = Dataset::from_path(black_box(&dataset_path))
                    .expect("Failed to load large dataset");
                
                // Process dataset to measure memory usage
                let _stats = dataset.compute_statistics();
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_ledger_initialization,
    bench_dataset_loading,
    bench_proof_generation,
    bench_proof_verification,
    bench_merkle_tree_operations,
    bench_hash_functions,
    bench_concurrent_operations,
    bench_storage_backends,
    bench_memory_usage
);

criterion_main!(benches);