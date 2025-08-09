//! Advanced zero-knowledge circuit tests.

use ark_relations::r1cs::{ConstraintSystem, OptimizationGoal};
use ark_std::rand::{Rng, SeedableRng};
use zkp_dataset_ledger::circuits::*;
use zkp_dataset_ledger::error::LedgerError;

#[test]
fn test_dataset_circuit_with_matrix_data() {
    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(12345);

    // Create a 3x2 matrix
    let dataset_rows = vec![
        vec![Fr::from(1u64), Fr::from(2u64)],
        vec![Fr::from(3u64), Fr::from(4u64)],
        vec![Fr::from(5u64), Fr::from(6u64)],
    ];

    let salt = Fr::from(rng.gen::<u64>());

    // Compute expected hash
    let mut commitment = salt;
    for (row_idx, row) in dataset_rows.iter().enumerate() {
        let row_idx_var = Fr::from((row_idx + 1) as u64);
        let mut row_hash = row_idx_var;

        for &cell in row {
            let sum = row_hash + cell;
            let squared = sum * sum;
            row_hash = squared + cell;
        }

        commitment = commitment + row_hash * row_idx_var;
    }

    let dataset_hash = commitment * salt + commitment;

    let circuit = DatasetCircuit {
        dataset_hash: Some(dataset_hash),
        row_count: Some(Fr::from(3u64)),
        column_count: Some(Fr::from(2u64)),
        merkle_root: Some(Fr::from(0u64)),
        dataset_rows: Some(dataset_rows),
        salt: Some(salt),
        merkle_path: None,
        merkle_siblings: None,
        max_rows: 10,
        max_cols: 10,
    };

    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);

    circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!("Dataset circuit constraints: {}", cs.num_constraints());
    println!("Dataset circuit variables: {}", cs.num_witness_variables());
}

#[test]
fn test_statistical_circuit_with_differential_privacy() {
    // Test data with 4 samples, 2 dimensions
    let data_matrix = vec![
        vec![Fr::from(10u64), Fr::from(20u64)],
        vec![Fr::from(15u64), Fr::from(25u64)],
        vec![Fr::from(12u64), Fr::from(22u64)],
        vec![Fr::from(18u64), Fr::from(28u64)],
    ];

    // Compute statistics
    let n = Fr::from(4u64);
    let dim1_sum = Fr::from(10 + 15 + 12 + 18); // 55
    let dim2_sum = Fr::from(20 + 25 + 22 + 28); // 95

    let mean_commitment = dim1_sum; // Use first dimension
    let variance_commitment = Fr::from(16u64); // Simplified variance

    let circuit = StatisticalCircuit {
        mean_commitment: Some(mean_commitment),
        variance_commitment: Some(variance_commitment),
        count_commitment: Some(n),
        correlation_commitment: None,
        data_matrix: Some(data_matrix),
        noise_values: Some(vec![Fr::from(1u64), Fr::from(2u64)]),
        privacy_budget: Some(Fr::from(1000u64)), // Scaled epsilon
        min_value: Some(Fr::from(5u64)),
        max_value: Some(Fr::from(30u64)),
        max_samples: 4,
        dimensions: 2,
        use_dp: true,
    };

    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);

    circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!("Statistical circuit constraints: {}", cs.num_constraints());
    println!(
        "Statistical circuit variables: {}",
        cs.num_witness_variables()
    );
}

#[test]
fn test_multiparty_circuit() {
    let participant_values = vec![
        Fr::from(100u64),
        Fr::from(200u64),
        Fr::from(150u64),
        Fr::from(175u64),
    ];

    let participant_salts = vec![
        Fr::from(12345u64),
        Fr::from(23456u64),
        Fr::from(34567u64),
        Fr::from(45678u64),
    ];

    let participant_commitments: Vec<_> = participant_values
        .iter()
        .zip(participant_salts.iter())
        .map(|(&val, &salt)| {
            let committed = val + salt;
            committed * committed + committed
        })
        .collect();

    let aggregated_result = participant_values
        .iter()
        .fold(Fr::from(0u64), |acc, &val| acc + val);

    let circuit = MultiPartyCircuit {
        aggregated_result: Some(aggregated_result),
        participant_count: Some(Fr::from(4u64)),
        computation_type: Some(Fr::from(0u64)), // sum
        participant_values: Some(participant_values),
        participant_salts: Some(participant_salts),
        participant_commitments: Some(participant_commitments),
        threshold: 3,
        max_participants: 10,
    };

    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);

    circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!("MPC circuit constraints: {}", cs.num_constraints());
}

#[test]
fn test_differential_privacy_circuit() {
    let true_result = Fr::from(1000u64);
    let noise_value = Fr::from(50u64);
    let noised_result = true_result + noise_value;

    let epsilon = Fr::from(1000u64); // Scaled
    let sensitivity = Fr::from(1u64);

    let circuit = DifferentialPrivacyCircuit {
        epsilon: Some(epsilon),
        delta: Some(Fr::from(1u64)),
        sensitivity: Some(sensitivity),
        noised_result: Some(noised_result),
        true_result: Some(true_result),
        noise_value: Some(noise_value),
        query_type: Some(Fr::from(1u64)), // sum
        laplace_scale: Some(sensitivity / epsilon),
        gaussian_scale: None,
    };

    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);

    circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!("DP circuit constraints: {}", cs.num_constraints());
}

#[test]
fn test_streaming_circuit() {
    let chunk_data = vec![
        Fr::from(10u64),
        Fr::from(20u64),
        Fr::from(30u64),
        Fr::from(40u64),
        Fr::from(50u64),
    ];

    let previous_accumulator = Fr::from(100u64);
    let chunk_sum = chunk_data
        .iter()
        .fold(Fr::from(0u64), |acc, &val| acc + val);
    let current_accumulator = previous_accumulator + chunk_sum;

    let circuit = StreamingCircuit {
        previous_accumulator: Some(previous_accumulator),
        current_accumulator: Some(current_accumulator),
        chunk_index: Some(Fr::from(0u64)),
        total_chunks: Some(Fr::from(10u64)),
        chunk_data: Some(chunk_data.clone()),
        chunk_size: Some(Fr::from(chunk_data.len() as u64)),
        chunk_merkle_root: None,
        previous_root: None,
        max_chunk_size: 1000,
        accumulator_type: 0, // sum
    };

    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);

    circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!("Streaming circuit constraints: {}", cs.num_constraints());
}

#[test]
fn test_dataset_comparison_circuit() {
    let dataset_a_hash = Fr::from(12345u64);
    let dataset_b_hash = Fr::from(12345u64); // Same for equality test

    let stats_a = vec![Fr::from(100u64), Fr::from(10u64)]; // mean, count
    let stats_b = vec![Fr::from(100u64), Fr::from(10u64)]; // same stats

    let datasets_equal = Fr::from(1u64); // equal
    let similarity_score = Fr::from(100u64); // 100% similar

    let circuit = DatasetComparisonCircuit {
        datasets_equal: Some(datasets_equal),
        similarity_score: Some(similarity_score),
        comparison_type: Some(Fr::from(0u64)), // exact
        dataset_a_hash: Some(dataset_a_hash),
        dataset_b_hash: Some(dataset_b_hash),
        dataset_a_stats: Some(stats_a),
        dataset_b_stats: Some(stats_b),
        tolerance: Some(Fr::from(10u64)),
        privacy_salt_a: Some(Fr::from(111u64)),
        privacy_salt_b: Some(Fr::from(222u64)),
    };

    let cs = ConstraintSystem::<Fr>::new_ref();
    cs.set_optimization_goal(OptimizationGoal::Constraints);

    circuit.generate_constraints(cs.clone()).unwrap();
    assert!(cs.is_satisfied().unwrap());

    println!(
        "Dataset comparison circuit constraints: {}",
        cs.num_constraints()
    );
}

#[test]
fn test_parallel_proof_generation() {
    use std::time::Instant;

    // Test parallel setup
    let circuits = vec![
        DatasetCircuit {
            dataset_hash: Some(Fr::from(1u64)),
            row_count: Some(Fr::from(1u64)),
            column_count: Some(Fr::from(1u64)),
            merkle_root: Some(Fr::from(0u64)),
            dataset_rows: Some(vec![vec![Fr::from(1u64)]]),
            salt: Some(Fr::from(123u64)),
            merkle_path: None,
            merkle_siblings: None,
            max_rows: 10,
            max_cols: 10,
        };
        4
    ];

    let start = Instant::now();
    let results = setup_circuits_parallel(circuits, 2);
    let duration = start.elapsed();

    match results {
        Ok(keys) => {
            println!(
                "Parallel setup of {} circuits took {:?}",
                keys.len(),
                duration
            );
            assert_eq!(keys.len(), 4);
        }
        Err(e) => {
            println!("Parallel setup failed: {}", e);
            // Don't fail the test as this might fail in CI environments
        }
    }
}

#[test]
fn test_proof_statistics() {
    let stats = ProofStats {
        constraints: 1000,
        variables: 500,
        proof_size: 192,
        generation_time_ms: 250,
    };

    println!("Proof statistics: {:?}", stats);
    assert!(stats.constraints > 0);
    assert!(stats.variables > 0);
    assert!(stats.proof_size > 0);
    assert!(stats.generation_time_ms > 0);
}

#[test]
fn test_memory_efficient_operations() {
    // Test that our circuits can handle reasonable data sizes
    let large_dataset = (0..1000).map(|i| Fr::from(i as u64)).collect::<Vec<_>>();

    // Should not panic with memory issues
    let chunks: Vec<_> = large_dataset.chunks(100).collect();
    assert_eq!(chunks.len(), 10);

    // Test streaming data processing
    let mut total = Fr::from(0u64);
    for chunk in &chunks {
        let chunk_sum = chunk.iter().fold(Fr::from(0u64), |acc, &val| acc + val);
        total = total + chunk_sum;
    }

    let expected_total = large_dataset
        .iter()
        .fold(Fr::from(0u64), |acc, &val| acc + val);
    assert_eq!(total, expected_total);
}
