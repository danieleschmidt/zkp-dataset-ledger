use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zkp_dataset_ledger::{Ledger, Dataset, ProofConfig};
use tempfile::TempDir;

fn benchmark_batch_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_verification");
    
    // Generate multiple proofs for batch verification
    let temp_dir = TempDir::new().unwrap();
    let ledger_path = temp_dir.path().join("batch_ledger");
    let mut ledger = Ledger::new(&ledger_path).unwrap();
    
    let mut proofs = Vec::new();
    for i in 0..10 {
        let test_csv = temp_dir.path().join(format!("batch_data_{}.csv", i));
        std::fs::write(&test_csv, format!("id,value\n1,{}\n2,{}\n", i, i * 10)).unwrap();
        
        let dataset = Dataset::from_path(&test_csv).unwrap();
        let proof = ledger.notarize_dataset(
            dataset,
            &format!("batch-{}", i),
            ProofConfig::default()
        ).unwrap();
        proofs.push(proof);
    }
    
    group.bench_function("verify_batch_10", |b| {
        b.iter(|| {
            for proof in &proofs {
                let result = ledger.verify_proof(black_box(proof)).unwrap();
                black_box(result);
            }
        });
    });
    
    group.finish();
}

fn benchmark_chain_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_verification");
    
    let temp_dir = TempDir::new().unwrap();
    let ledger_path = temp_dir.path().join("chain_ledger");
    let mut ledger = Ledger::new(&ledger_path).unwrap();
    
    // Create a chain of dataset transformations
    for i in 0..5 {
        let test_csv = temp_dir.path().join(format!("chain_data_{}.csv", i));
        std::fs::write(&test_csv, format!("id,value\n1,{}\n", i * 100)).unwrap();
        
        let dataset = Dataset::from_path(&test_csv).unwrap();
        ledger.notarize_dataset(
            dataset,
            &format!("chain-v{}", i + 1),
            ProofConfig::default()
        ).unwrap();
    }
    
    group.bench_function("verify_chain_5_steps", |b| {
        b.iter(|| {
            let valid = ledger.verify_chain_integrity().unwrap();
            black_box(valid);
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_batch_verification, benchmark_chain_verification);
criterion_main!(benches);