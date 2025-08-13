//! Merkle tree implementation for cryptographic commitments.

use crate::crypto::hash::{hash_bytes, hash_combined, sha3_hash, HashAlgorithm};
use crate::error::LedgerError;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// A Merkle tree for creating cryptographic commitments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    pub root: String,
    pub leaves: Vec<String>,
    pub algorithm: HashAlgorithm,
    levels: Vec<Vec<String>>,
}

/// A Merkle proof for verifying inclusion of a leaf in the tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: usize,
    pub leaf_hash: String,
    pub siblings: Vec<String>,
    pub root: String,
}

impl MerkleTree {
    /// Create a new Merkle tree from leaf data.
    pub fn new(leaves: Vec<Vec<u8>>, algorithm: HashAlgorithm) -> Result<Self, LedgerError> {
        if leaves.is_empty() {
            return Err(LedgerError::validation(
                "Cannot create tree with no leaves",
            );
        }

        // Hash all leaf data
        let leaf_hashes: Result<Vec<String>, LedgerError> = leaves
            .iter()
            .map(|leaf| hash_bytes(leaf, algorithm.clone()))
            .collect();
        let leaf_hashes = leaf_hashes?;

        let mut levels = vec![leaf_hashes.clone()];
        let mut current_level = leaf_hashes.clone();

        // Build tree level by level
        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let combined_hash = if chunk.len() == 2 {
                    // Hash the concatenation of two hashes
                    let combined_data = [chunk[0].as_bytes(), chunk[1].as_bytes()];
                    hash_combined(&combined_data, algorithm.clone())?
                } else {
                    // Odd number of nodes, promote the last one
                    chunk[0].clone()
                };
                next_level.push(combined_hash);
            }

            levels.push(next_level.clone());
            current_level = next_level;
        }

        let root = current_level[0].clone();

        Ok(Self {
            root,
            leaves: leaf_hashes,
            algorithm,
            levels,
        })
    }

    /// Create a new Merkle tree from string hashes.
    pub fn from_hashes(
        leaf_hashes: Vec<String>,
        algorithm: HashAlgorithm,
    ) -> Result<Self, LedgerError> {
        if leaf_hashes.is_empty() {
            return Err(LedgerError::validation(
                "Cannot create tree with no leaves",
            );
        }

        let mut levels = vec![leaf_hashes.clone()];
        let mut current_level = leaf_hashes.clone();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let combined_hash = if chunk.len() == 2 {
                    let combined_data = [chunk[0].as_bytes(), chunk[1].as_bytes()];
                    hash_combined(&combined_data, algorithm.clone())?
                } else {
                    chunk[0].clone()
                };
                next_level.push(combined_hash);
            }

            levels.push(next_level.clone());
            current_level = next_level;
        }

        let root = current_level[0].clone();

        Ok(Self {
            root,
            leaves: leaf_hashes,
            algorithm,
            levels,
        })
    }

    /// Generate a Merkle proof for a specific leaf.
    pub fn generate_proof(&self, leaf_index: usize) -> Result<MerkleProof, LedgerError> {
        if leaf_index >= self.leaves.len() {
            return Err(LedgerError::validation(format!(
                "Leaf index {} out of bounds for tree with {} leaves",
                leaf_index,
                self.leaves.len()
            )));
        }

        let mut siblings = Vec::new();
        let mut current_index = leaf_index;

        for level in &self.levels[..self.levels.len() - 1] {
            let sibling_index = if current_index % 2 == 0 {
                current_index + 1
            } else {
                current_index - 1
            };

            if sibling_index < level.len() {
                siblings.push(level[sibling_index].clone());
            }

            current_index /= 2;
        }

        Ok(MerkleProof {
            leaf_index,
            leaf_hash: self.leaves[leaf_index].clone(),
            siblings,
            root: self.root.clone(),
        })
    }

    /// Verify a Merkle proof.
    pub fn verify_proof(
        proof: &MerkleProof,
        algorithm: HashAlgorithm,
    ) -> Result<bool, LedgerError> {
        let mut current_hash = proof.leaf_hash.clone();
        let mut current_index = proof.leaf_index;

        for sibling in &proof.siblings {
            let combined_data = if current_index % 2 == 0 {
                [current_hash.as_bytes(), sibling.as_bytes()]
            } else {
                [sibling.as_bytes(), current_hash.as_bytes()]
            };

            current_hash = hash_combined(&combined_data, algorithm.clone())?;
            current_index /= 2;
        }

        Ok(current_hash == proof.root)
    }

    /// Get the root hash of the tree.
    pub fn root_hash(&self) -> &str {
        &self.root
    }

    /// Get the number of leaves in the tree.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Get the height of the tree.
    pub fn height(&self) -> usize {
        self.levels.len()
    }

    /// Add a new leaf to the tree (creates a new tree).
    pub fn append_leaf(&self, leaf_data: &[u8]) -> Result<Self, LedgerError> {
        let mut new_leaves = self.leaves.clone();
        let new_leaf_hash = hash_bytes(leaf_data, self.algorithm.clone())?;
        new_leaves.push(new_leaf_hash);

        Self::from_hashes(new_leaves, self.algorithm.clone())
    }

    /// Optimized parallel construction for large datasets.
    pub fn new_parallel(
        leaves: Vec<Vec<u8>>,
        algorithm: HashAlgorithm,
        num_threads: Option<usize>,
    ) -> Result<Self, LedgerError> {
        if leaves.is_empty() {
            return Err(LedgerError::validation(
                "Cannot create tree with no leaves",
            );
        }

        let thread_count = num_threads.unwrap_or_else(num_cpus::get);

        // Parallel hashing of leaves
        let leaf_hashes: Result<Vec<String>, LedgerError> =
            if leaves.len() > 1000 && thread_count > 1 {
                // Use parallel processing for large datasets
                let chunk_size = (leaves.len() + thread_count - 1) / thread_count;
                let results = Arc::new(Mutex::new(Vec::new()));

                leaves
                    .par_chunks(chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let mut chunk_hashes = Vec::new();
                        for (i, leaf) in chunk.iter().enumerate() {
                            match hash_bytes(leaf, algorithm.clone()) {
                                Ok(hash) => chunk_hashes.push((chunk_idx * chunk_size + i, hash)),
                                Err(e) => {
                                    println!(
                                        "Hash error for leaf {}: {}",
                                        chunk_idx * chunk_size + i,
                                        e
                                    );
                                    return;
                                }
                            }
                        }
                        results.lock().unwrap().extend(chunk_hashes);
                    });

                let mut all_hashes = results.lock().unwrap();
                all_hashes.sort_by_key(|(i, _)| *i);
                Ok(all_hashes.into_iter().map(|(_, hash)| hash).collect())
            } else {
                // Sequential for smaller datasets
                leaves
                    .iter()
                    .map(|leaf| hash_bytes(leaf, algorithm.clone()))
                    .collect()
            };

        let leaf_hashes = leaf_hashes?;
        Self::build_tree_parallel(leaf_hashes, algorithm, thread_count)
    }

    /// Build tree levels in parallel for optimal performance.
    fn build_tree_parallel(
        leaf_hashes: Vec<String>,
        algorithm: HashAlgorithm,
        num_threads: usize,
    ) -> Result<Self, LedgerError> {
        let mut levels = vec![leaf_hashes.clone()];
        let mut current_level = leaf_hashes.clone();

        while current_level.len() > 1 {
            let next_level = if current_level.len() > 100 && num_threads > 1 {
                // Parallel processing for large levels
                let pairs: Vec<_> = current_level.chunks(2).collect();
                let results: Result<Vec<String>, LedgerError> = pairs
                    .par_iter()
                    .map(|chunk| {
                        if chunk.len() == 2 {
                            let combined_data = [chunk[0].as_bytes(), chunk[1].as_bytes()];
                            hash_combined(&combined_data, algorithm.clone())
                        } else {
                            Ok(chunk[0].clone())
                        }
                    })
                    .collect();
                results?
            } else {
                // Sequential for smaller levels
                let mut next = Vec::new();
                for chunk in current_level.chunks(2) {
                    let combined_hash = if chunk.len() == 2 {
                        let combined_data = [chunk[0].as_bytes(), chunk[1].as_bytes()];
                        hash_combined(&combined_data, algorithm.clone())?
                    } else {
                        chunk[0].clone()
                    };
                    next.push(combined_hash);
                }
                next
            };

            levels.push(next_level.clone());
            current_level = next_level;
        }

        let root = current_level[0].clone();

        Ok(Self {
            root,
            leaves: leaf_hashes,
            algorithm,
            levels,
        })
    }

    /// Batch verification of multiple proofs.
    pub fn verify_proofs_batch(
        proofs: &[MerkleProof],
        algorithm: HashAlgorithm,
    ) -> Result<Vec<bool>, LedgerError> {
        if proofs.is_empty() {
            return Ok(vec![]);
        }

        // Parallel verification for multiple proofs
        let results: Result<Vec<bool>, LedgerError> = proofs
            .par_iter()
            .map(|proof| Self::verify_proof(proof, algorithm.clone()))
            .collect();

        results
    }

    /// Generate proofs for multiple leaves in parallel.
    pub fn generate_proofs_batch(
        &self,
        leaf_indices: &[usize],
    ) -> Result<Vec<MerkleProof>, LedgerError> {
        if leaf_indices.is_empty() {
            return Ok(vec![]);
        }

        // Validate all indices first
        for &idx in leaf_indices {
            if idx >= self.leaves.len() {
                return Err(LedgerError::validation(format!(
                    "Leaf index {} out of bounds for tree with {} leaves",
                    idx,
                    self.leaves.len()
                )));
            }
        }

        // Generate proofs in parallel
        let results: Result<Vec<MerkleProof>, LedgerError> = leaf_indices
            .par_iter()
            .map(|&idx| self.generate_proof(idx))
            .collect();

        results
    }

    /// Incremental update: add multiple leaves efficiently.
    pub fn append_leaves(&self, leaf_data: &[Vec<u8>]) -> Result<Self, LedgerError> {
        if leaf_data.is_empty() {
            return Ok(self.clone());
        }

        let mut new_leaves = self.leaves.clone();

        // Hash new leaves in parallel if there are many
        let new_hashes: Result<Vec<String>, LedgerError> = if leaf_data.len() > 100 {
            leaf_data
                .par_iter()
                .map(|data| hash_bytes(data, self.algorithm.clone()))
                .collect()
        } else {
            leaf_data
                .iter()
                .map(|data| hash_bytes(data, self.algorithm.clone()))
                .collect()
        };

        new_leaves.extend(new_hashes?);
        Self::from_hashes(new_leaves, self.algorithm.clone())
    }

    /// Memory-efficient streaming construction for very large datasets.
    pub fn build_streaming<I>(
        leaf_iter: I,
        algorithm: HashAlgorithm,
        chunk_size: usize,
    ) -> Result<Self, LedgerError>
    where
        I: Iterator<Item = Vec<u8>>,
    {
        let mut all_leaves = Vec::new();
        let mut processed = 0;

        // Process in chunks to manage memory usage
        for chunk in &leaf_iter.collect::<Vec<_>>().chunks(chunk_size) {
            let chunk_hashes: Result<Vec<String>, LedgerError> = chunk
                .iter()
                .map(|leaf| hash_bytes(leaf, algorithm.clone()))
                .collect();

            all_leaves.extend(chunk_hashes?);
            processed += chunk.len();

            if processed % 10000 == 0 {
                println!("Processed {} leaves...", processed);
            }
        }

        if all_leaves.is_empty() {
            return Err(LedgerError::validation(
                "Cannot create tree with no leaves",
            );
        }

        // Use parallel construction for the final tree
        Self::build_tree_parallel(all_leaves, algorithm, num_cpus::get())
    }

    /// Get tree statistics for performance monitoring.
    pub fn get_stats(&self) -> MerkleTreeStats {
        let total_nodes: usize = self.levels.iter().map(|level| level.len()).sum();

        MerkleTreeStats {
            leaf_count: self.leaves.len(),
            height: self.height(),
            total_nodes,
            algorithm: self.algorithm.clone(),
            root_hash: self.root.clone(),
        }
    }

    /// Optimize tree structure for better performance.
    pub fn optimize(&self) -> Result<Self, LedgerError> {
        // For now, just rebuild with parallel construction
        let leaf_data: Vec<Vec<u8>> = self
            .leaves
            .iter()
            .map(|hash| hash.as_bytes().to_vec())
            .collect();

        Self::new_parallel(leaf_data, self.algorithm.clone(), Some(num_cpus::get()))
    }
}

/// Legacy structure for backward compatibility
/// Statistics for Merkle tree performance monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTreeStats {
    pub leaf_count: usize,
    pub height: usize,
    pub total_nodes: usize,
    pub algorithm: HashAlgorithm,
    pub root_hash: String,
}

/// Batch operations result for multiple tree operations.
#[derive(Debug, Clone)]
pub struct BatchResult<T> {
    pub successes: Vec<(usize, T)>,
    pub errors: Vec<(usize, LedgerError)>,
}

impl<T> BatchResult<T> {
    pub fn success_rate(&self) -> f64 {
        let total = self.successes.len() + self.errors.len();
        if total == 0 {
            0.0
        } else {
            self.successes.len() as f64 / total as f64
        }
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

/// Legacy structure for backward compatibility.
pub struct LegacyMerkleTree {
    pub root: Vec<u8>,
    nodes: Vec<Vec<u8>>,
}

impl LegacyMerkleTree {
    pub fn new(leaves: Vec<Vec<u8>>) -> Self {
        if leaves.is_empty() {
            return Self {
                root: vec![],
                nodes: vec![],
            };
        }

        let mut nodes = leaves;

        while nodes.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in nodes.chunks(2) {
                let combined = if chunk.len() == 2 {
                    [&chunk[0][..], &chunk[1][..]].concat()
                } else {
                    chunk[0].clone()
                };
                next_level.push(sha3_hash(&combined));
            }

            nodes = next_level;
        }

        Self {
            root: nodes[0].clone(),
            nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_creation() {
        let leaves = vec![
            b"leaf1".to_vec(),
            b"leaf2".to_vec(),
            b"leaf3".to_vec(),
            b"leaf4".to_vec(),
        ];

        let tree = MerkleTree::new(leaves, HashAlgorithm::Sha3_256).unwrap();
        assert_eq!(tree.leaf_count(), 4);
        assert!(tree.height() >= 3);
        assert!(!tree.root.is_empty());
    }

    #[test]
    fn test_merkle_proof_generation_and_verification() {
        let leaves = vec![
            b"leaf1".to_vec(),
            b"leaf2".to_vec(),
            b"leaf3".to_vec(),
            b"leaf4".to_vec(),
        ];

        let tree = MerkleTree::new(leaves, HashAlgorithm::Sha3_256).unwrap();

        for i in 0..tree.leaf_count() {
            let proof = tree.generate_proof(i).unwrap();
            let is_valid = MerkleTree::verify_proof(&proof, HashAlgorithm::Sha3_256).unwrap();
            assert!(is_valid, "Proof for leaf {} should be valid", i);
        }
    }

    #[test]
    fn test_single_leaf_tree() {
        let leaves = vec![b"single_leaf".to_vec()];
        let tree = MerkleTree::new(leaves, HashAlgorithm::Blake3).unwrap();

        assert_eq!(tree.leaf_count(), 1);
        let proof = tree.generate_proof(0).unwrap();
        let is_valid = MerkleTree::verify_proof(&proof, HashAlgorithm::Blake3).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_empty_tree_fails() {
        let leaves = vec![];
        let result = MerkleTree::new(leaves, HashAlgorithm::Sha3_256);
        assert!(result.is_err());
    }

    #[test]
    fn test_append_leaf() {
        let leaves = vec![b"leaf1".to_vec(), b"leaf2".to_vec()];
        let tree = MerkleTree::new(leaves, HashAlgorithm::Sha3_256).unwrap();

        let new_tree = tree.append_leaf(b"leaf3").unwrap();
        assert_eq!(new_tree.leaf_count(), 3);
        assert_ne!(tree.root, new_tree.root);
    }

    #[test]
    fn test_different_algorithms() {
        let leaves = vec![b"test".to_vec()];

        let tree_sha3 = MerkleTree::new(leaves.clone(), HashAlgorithm::Sha3_256).unwrap();
        let tree_blake3 = MerkleTree::new(leaves, HashAlgorithm::Blake3).unwrap();

        assert_ne!(tree_sha3.root, tree_blake3.root);
    }
}
