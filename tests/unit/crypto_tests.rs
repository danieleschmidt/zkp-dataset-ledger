// Unit tests for cryptographic operations
use crate::fixtures::{constants, PerformanceTester, TestDataGenerator};
use zkp_dataset_ledger::crypto::{Hash, MerkleTree, ProofGenerator, Verifier};

#[test]
fn test_sha3_hash_deterministic() {
    let hasher = Hash::new("sha3-256").expect("Failed to create SHA3 hasher");
    let data = b"test data for hashing";

    let hash1 = hasher.hash(data);
    let hash2 = hasher.hash(data);

    assert_eq!(hash1, hash2, "Hash should be deterministic");
    assert_eq!(hash1.len(), 32, "SHA3-256 should produce 32-byte hash");
}

#[test]
fn test_blake3_hash_deterministic() {
    let hasher = Hash::new("blake3").expect("Failed to create Blake3 hasher");
    let data = b"test data for hashing";

    let hash1 = hasher.hash(data);
    let hash2 = hasher.hash(data);

    assert_eq!(hash1, hash2, "Hash should be deterministic");
    assert_eq!(hash1.len(), 32, "Blake3 should produce 32-byte hash");
}

#[test]
fn test_hash_different_algorithms() {
    let sha3_hasher = Hash::new("sha3-256").expect("Failed to create SHA3 hasher");
    let blake3_hasher = Hash::new("blake3").expect("Failed to create Blake3 hasher");
    let data = b"same input data";

    let sha3_hash = sha3_hasher.hash(data);
    let blake3_hash = blake3_hasher.hash(data);

    assert_ne!(
        sha3_hash, blake3_hash,
        "Different algorithms should produce different hashes"
    );
}

#[test]
fn test_hash_sensitivity() {
    let hasher = Hash::new("sha3-256").expect("Failed to create hasher");

    let hash1 = hasher.hash(b"data");
    let hash2 = hasher.hash(b"data2"); // Single character difference

    assert_ne!(hash1, hash2, "Hash should be sensitive to input changes");
}

#[test]
fn test_hash_empty_input() {
    let hasher = Hash::new("sha3-256").expect("Failed to create hasher");
    let hash = hasher.hash(b"");

    assert_eq!(hash.len(), 32, "Should handle empty input");
    assert_ne!(hash, [0u8; 32], "Empty hash should not be all zeros");
}

#[test]
fn test_hash_large_input() {
    let hasher = Hash::new("sha3-256").expect("Failed to create hasher");
    let large_data = vec![0xAB; 1_000_000]; // 1MB of data

    let timer = PerformanceTester::new();
    let hash = hasher.hash(&large_data);
    timer.assert_under_ms(1000, "Large data hashing"); // Should be fast

    assert_eq!(hash.len(), 32, "Should handle large input");
}

#[test]
fn test_invalid_hash_algorithm() {
    let result = Hash::new("invalid_algorithm");
    assert!(result.is_err(), "Should reject invalid hash algorithm");
}

#[test]
fn test_merkle_tree_construction() {
    let leaves = vec![
        b"leaf1".to_vec(),
        b"leaf2".to_vec(),
        b"leaf3".to_vec(),
        b"leaf4".to_vec(),
    ];

    let tree = MerkleTree::new(leaves, "sha3-256").expect("Failed to construct Merkle tree");

    assert!(!tree.root().is_empty(), "Root should not be empty");
    assert_eq!(tree.leaf_count(), 4, "Should have correct leaf count");
}

#[test]
fn test_merkle_tree_single_leaf() {
    let leaves = vec![b"single_leaf".to_vec()];

    let tree =
        MerkleTree::new(leaves, "sha3-256").expect("Failed to construct single-leaf Merkle tree");

    assert!(!tree.root().is_empty(), "Root should not be empty");
    assert_eq!(tree.leaf_count(), 1, "Should have one leaf");
}

#[test]
fn test_merkle_tree_empty() {
    let leaves: Vec<Vec<u8>> = vec![];

    let result = MerkleTree::new(leaves, "sha3-256");
    assert!(result.is_err(), "Should reject empty leaf set");
}

#[test]
fn test_merkle_tree_proof_generation() {
    let leaves = vec![
        b"leaf1".to_vec(),
        b"leaf2".to_vec(),
        b"leaf3".to_vec(),
        b"leaf4".to_vec(),
    ];

    let tree = MerkleTree::new(leaves, "sha3-256").expect("Failed to construct Merkle tree");

    for i in 0..4 {
        let proof = tree
            .generate_proof(i)
            .expect("Failed to generate Merkle proof");

        assert!(!proof.is_empty(), "Proof should not be empty");

        let is_valid = tree
            .verify_proof(i, &proof)
            .expect("Failed to verify Merkle proof");
        assert!(is_valid, "Merkle proof should be valid");
    }
}

#[test]
fn test_merkle_tree_proof_invalid_index() {
    let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
    let tree = MerkleTree::new(leaves, "sha3-256").expect("Failed to construct Merkle tree");

    let result = tree.generate_proof(5); // Index out of bounds
    assert!(result.is_err(), "Should reject invalid leaf index");
}

#[test]
fn test_merkle_tree_deterministic() {
    let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec(), b"leaf3".to_vec()];

    let tree1 =
        MerkleTree::new(leaves.clone(), "sha3-256").expect("Failed to construct first tree");
    let tree2 = MerkleTree::new(leaves, "sha3-256").expect("Failed to construct second tree");

    assert_eq!(tree1.root(), tree2.root(), "Trees should have same root");
}

#[test]
fn test_merkle_tree_order_sensitivity() {
    let leaves1 = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
    let leaves2 = vec![b"leaf2".to_vec(), b"leaf1".to_vec()]; // Different order

    let tree1 = MerkleTree::new(leaves1, "sha3-256").expect("Failed to construct first tree");
    let tree2 = MerkleTree::new(leaves2, "sha3-256").expect("Failed to construct second tree");

    assert_ne!(tree1.root(), tree2.root(), "Order should affect tree root");
}

#[test]
fn test_proof_generator_basic() {
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_small_csv("test");

    let proof_gen =
        ProofGenerator::new("groth16", "bls12-381").expect("Failed to create proof generator");

    let timer = PerformanceTester::new();
    let proof = proof_gen
        .generate_row_count_proof(&dataset_path)
        .expect("Failed to generate proof");
    timer.assert_under_ms(constants::MAX_PROOF_TIME_MS, "Proof generation");

    assert!(!proof.is_empty(), "Proof should not be empty");
}

#[test]
fn test_proof_generator_invalid_system() {
    let result = ProofGenerator::new("invalid_system", "bls12-381");
    assert!(result.is_err(), "Should reject invalid proof system");
}

#[test]
fn test_proof_generator_invalid_curve() {
    let result = ProofGenerator::new("groth16", "invalid_curve");
    assert!(result.is_err(), "Should reject invalid elliptic curve");
}

#[test]
fn test_verifier_valid_proof() {
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_small_csv("test");

    let proof_gen =
        ProofGenerator::new("groth16", "bls12-381").expect("Failed to create proof generator");
    let verifier = Verifier::new("groth16", "bls12-381").expect("Failed to create verifier");

    let proof = proof_gen
        .generate_row_count_proof(&dataset_path)
        .expect("Failed to generate proof");

    let timer = PerformanceTester::new();
    let is_valid = verifier.verify(&proof).expect("Failed to verify proof");
    timer.assert_under_ms(constants::MAX_VERIFICATION_TIME_MS, "Proof verification");

    assert!(is_valid, "Valid proof should verify successfully");
}

#[test]
fn test_verifier_invalid_proof() {
    let verifier = Verifier::new("groth16", "bls12-381").expect("Failed to create verifier");

    let invalid_proof = vec![0u8; 288]; // Invalid proof data
    let is_valid = verifier
        .verify(&invalid_proof)
        .expect("Verification should not error on invalid proof");

    assert!(!is_valid, "Invalid proof should not verify");
}

#[test]
fn test_proof_serialization() {
    let generator = TestDataGenerator::new();
    let dataset_path = generator.create_small_csv("test");

    let proof_gen =
        ProofGenerator::new("groth16", "bls12-381").expect("Failed to create proof generator");

    let proof = proof_gen
        .generate_row_count_proof(&dataset_path)
        .expect("Failed to generate proof");

    // Test serialization round-trip
    let serialized = serde_json::to_string(&proof).expect("Failed to serialize proof");
    let deserialized: Vec<u8> =
        serde_json::from_str(&serialized).expect("Failed to deserialize proof");

    assert_eq!(proof, deserialized, "Serialization should be lossless");
}

#[test]
fn test_constant_time_operations() {
    // This test ensures cryptographic operations are constant-time
    // to prevent timing attacks

    let hasher = Hash::new("sha3-256").expect("Failed to create hasher");
    let data1 = vec![0u8; 1000];
    let data2 = vec![1u8; 1000];

    let mut times = Vec::new();

    // Measure multiple hash operations
    for _ in 0..100 {
        let timer = PerformanceTester::new();
        let _hash = hasher.hash(&data1);
        times.push(timer.elapsed_ms());

        let timer = PerformanceTester::new();
        let _hash = hasher.hash(&data2);
        times.push(timer.elapsed_ms());
    }

    // Times should be relatively consistent (within 2x variance)
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();

    assert!(
        max_time <= min_time * 2,
        "Hash timing variance too high: min={}ms, max={}ms",
        min_time,
        max_time
    );
}
