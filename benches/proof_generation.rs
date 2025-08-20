use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::time::Duration;
use tempfile::TempDir;
use zkp_dataset_ledger::{Dataset, Ledger};

fn generate_test_data(rows: usize) -> String {
    let mut data = String::from("id,value,category\n");
    for i in 0..rows {
        data.push_str(&format!("{},{},{}\n", i, i * 10, i % 5));
    }
    data
}

fn benchmark_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");
    group.measurement_time(Duration::from_secs(30));

    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        group.bench_with_input(BenchmarkId::new("rows", size), &size, |b, &size| {
            b.iter_with_setup(
                || {
                    let temp_dir = TempDir::new().unwrap();
                    let test_csv = temp_dir.path().join("benchmark_data.csv");
                    std::fs::write(&test_csv, generate_test_data(size)).unwrap();

                    let ledger_path = temp_dir.path().join("benchmark_ledger.json");
                    let mut ledger = Ledger::with_storage(
                        "benchmark".to_string(),
                        ledger_path.to_string_lossy().to_string(),
                    )
                    .unwrap();
                    let dataset = Dataset::from_path(&test_csv).unwrap();

                    (ledger, dataset)
                },
                |(mut ledger, dataset)| {
                    let proof = ledger
                        .notarize_dataset(black_box(dataset), "integrity".to_string())
                        .unwrap();

                    black_box(proof);
                },
            );
        });
    }

    group.finish();
}

fn benchmark_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_verification");

    // Pre-generate proofs for verification benchmarking
    let temp_dir = TempDir::new().unwrap();
    let test_csv = temp_dir.path().join("verification_data.csv");
    std::fs::write(&test_csv, generate_test_data(10000)).unwrap();

    let ledger_path = temp_dir.path().join("verification_ledger.json");
    let mut ledger = Ledger::with_storage(
        "verification".to_string(),
        ledger_path.to_string_lossy().to_string(),
    )
    .unwrap();
    let dataset = Dataset::from_path(&test_csv).unwrap();

    let proof = ledger
        .notarize_dataset(dataset, "integrity".to_string())
        .unwrap();

    group.bench_function("verify_10k_rows", |b| {
        b.iter(|| {
            let result = ledger.verify_proof(black_box(&proof));
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_proof_generation,
    benchmark_proof_verification
);
criterion_main!(benches);
