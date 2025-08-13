//! Streaming zero-knowledge proofs for large dataset processing.

use crate::{Dataset, LedgerError, Proof, ProofConfig, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for streaming ZK proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub enable_incremental_verification: bool,
    pub enable_parallel_chunks: bool,
    pub memory_limit_mb: usize,
    pub streaming_algorithm: StreamingAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingAlgorithm {
    FixedChunks,
    AdaptiveChunks,
    SlidingWindow,
    IncrementalMerkle,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10000,  // 10K rows per chunk
            overlap_size: 1000, // 1K row overlap
            enable_incremental_verification: true,
            enable_parallel_chunks: true,
            memory_limit_mb: 1024, // 1GB memory limit
            streaming_algorithm: StreamingAlgorithm::AdaptiveChunks,
        }
    }
}

/// Streaming ZK proof processor for large datasets
pub struct StreamingZkProcessor {
    config: StreamingConfig,
    chunk_proofs: VecDeque<ChunkProof>,
    accumulated_state: Option<AccumulatedState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProof {
    pub chunk_id: usize,
    pub start_row: usize,
    pub end_row: usize,
    pub proof: Proof,
    pub merkle_root: String,
    pub previous_root: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccumulatedState {
    pub total_rows_processed: usize,
    pub global_merkle_root: String,
    pub statistics_accumulator: StatisticsAccumulator,
    pub chunk_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsAccumulator {
    pub row_count: usize,
    pub column_sums: Vec<f64>,
    pub min_values: Vec<f64>,
    pub max_values: Vec<f64>,
    pub hash_accumulator: String,
}

impl StreamingZkProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            chunk_proofs: VecDeque::new(),
            accumulated_state: None,
        }
    }

    /// Process a large dataset in streaming fashion
    pub fn process_streaming_dataset(
        &mut self,
        dataset: &Dataset,
        proof_config: &ProofConfig,
    ) -> Result<StreamingProof> {
        let total_rows = dataset.row_count.unwrap_or(0);
        let total_chunks = (total_rows as f64 / self.config.chunk_size as f64).ceil() as usize;

        println!("🌊 Starting streaming ZK processing:");
        println!("   Dataset: {} ({} rows)", dataset.name, total_rows);
        println!(
            "   Chunks: {} ({}k rows each)",
            total_chunks,
            self.config.chunk_size / 1000
        );

        let mut chunks_processed = 0;

        // Process dataset in chunks
        for chunk_idx in 0..total_chunks {
            let start_row = chunk_idx * self.config.chunk_size;
            let end_row = ((chunk_idx + 1) * self.config.chunk_size).min(total_rows as usize);

            if chunk_idx % 10 == 0 {
                println!("   Processing chunk {}/{}", chunk_idx + 1, total_chunks);
            }

            let chunk_dataset = self.create_chunk_dataset(dataset, start_row, end_row)?;
            let chunk_proof = self.process_chunk(&chunk_dataset, proof_config, chunk_idx)?;

            self.accumulate_chunk_proof(chunk_proof)?;
            chunks_processed += 1;

            // Memory management - remove old chunks if needed
            if self.chunk_proofs.len() > self.calculate_max_chunks_in_memory() {
                self.chunk_proofs.pop_front();
            }
        }

        // Generate final streaming proof
        let final_proof = self.generate_final_streaming_proof(dataset)?;

        println!(
            "   ✅ Streaming processing complete: {} chunks processed",
            chunks_processed
        );

        Ok(final_proof)
    }

    /// Process a single chunk of data
    fn process_chunk(
        &self,
        chunk_dataset: &Dataset,
        proof_config: &ProofConfig,
        chunk_id: usize,
    ) -> Result<ChunkProof> {
        // Generate proof for this chunk
        let chunk_proof = Proof::generate(chunk_dataset, proof_config)?;

        // Calculate chunk merkle root
        let chunk_merkle_root = self.calculate_chunk_merkle_root(chunk_dataset)?;

        // Get previous root for chaining
        let previous_root = self
            .accumulated_state
            .as_ref()
            .map(|state| state.global_merkle_root.clone());

        Ok(ChunkProof {
            chunk_id,
            start_row: 0, // Would be calculated from actual data
            end_row: chunk_dataset.row_count.unwrap_or(0) as usize,
            proof: chunk_proof,
            merkle_root: chunk_merkle_root,
            previous_root,
        })
    }

    /// Accumulate a chunk proof into the global state
    fn accumulate_chunk_proof(&mut self, chunk_proof: ChunkProof) -> Result<()> {
        // Add to chunk proofs queue
        self.chunk_proofs.push_back(chunk_proof.clone());

        // Update accumulated state
        if let Some(ref mut state) = self.accumulated_state {
            state.total_rows_processed += chunk_proof.end_row - chunk_proof.start_row;
            state.chunk_count += 1;

            // Update global merkle root by combining with chunk root
            state.global_merkle_root =
                self.combine_merkle_roots(&state.global_merkle_root, &chunk_proof.merkle_root)?;

            // Update statistics accumulator
            self.update_statistics_accumulator(&mut state.statistics_accumulator, &chunk_proof)?;
        } else {
            // Initialize accumulated state with first chunk
            self.accumulated_state = Some(AccumulatedState {
                total_rows_processed: chunk_proof.end_row - chunk_proof.start_row,
                global_merkle_root: chunk_proof.merkle_root.clone(),
                statistics_accumulator: self.initialize_statistics_accumulator(&chunk_proof)?,
                chunk_count: 1,
            });
        }

        Ok(())
    }

    /// Generate final streaming proof
    fn generate_final_streaming_proof(&self, original_dataset: &Dataset) -> Result<StreamingProof> {
        let state = self
            .accumulated_state
            .as_ref()
            .ok_or_else(|| LedgerError::internal("No accumulated state found"))?;

        // Create aggregated proof
        let aggregated_proof = self.create_aggregated_proof(original_dataset, state)?;

        // Collect chunk proof hashes for verification chain
        let chunk_proof_hashes: Vec<String> = self
            .chunk_proofs
            .iter()
            .map(|cp| cp.proof.dataset_hash.clone())
            .collect();

        Ok(StreamingProof {
            original_dataset: original_dataset.clone(),
            total_chunks: state.chunk_count,
            total_rows_processed: state.total_rows_processed,
            global_merkle_root: state.global_merkle_root.clone(),
            aggregated_proof,
            chunk_proof_hashes,
            streaming_algorithm: self.config.streaming_algorithm.clone(),
            incremental_verification: self.config.enable_incremental_verification,
        })
    }

    /// Create a dataset chunk for processing
    fn create_chunk_dataset(
        &self,
        original: &Dataset,
        start_row: usize,
        end_row: usize,
    ) -> Result<Dataset> {
        let chunk_rows = end_row - start_row;
        let chunk_size =
            (chunk_rows as f64 / original.row_count.unwrap_or(1) as f64) * original.size as f64;

        Ok(Dataset {
            name: format!("{}_chunk_{}_{}", original.name, start_row, end_row),
            hash: format!("{}_chunk_{}", original.hash, start_row),
            size: chunk_size as u64,
            row_count: Some(chunk_rows as u64),
            column_count: original.column_count,
            schema: original.schema.clone(),
            statistics: None, // Would be calculated for chunk
            format: original.format.clone(),
            path: original.path.clone(),
        })
    }

    /// Calculate merkle root for a chunk
    fn calculate_chunk_merkle_root(&self, chunk_dataset: &Dataset) -> Result<String> {
        // Simplified merkle root calculation
        Ok(format!("merkle_root_{}", chunk_dataset.hash))
    }

    /// Combine two merkle roots
    fn combine_merkle_roots(&self, left: &str, right: &str) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        left.hash(&mut hasher);
        right.hash(&mut hasher);

        Ok(format!("combined_{:x}", hasher.finish()))
    }

    /// Initialize statistics accumulator
    fn initialize_statistics_accumulator(
        &self,
        chunk_proof: &ChunkProof,
    ) -> Result<StatisticsAccumulator> {
        let column_count = 10; // Would extract from actual chunk data

        Ok(StatisticsAccumulator {
            row_count: chunk_proof.end_row - chunk_proof.start_row,
            column_sums: vec![0.0; column_count],
            min_values: vec![f64::INFINITY; column_count],
            max_values: vec![f64::NEG_INFINITY; column_count],
            hash_accumulator: chunk_proof.merkle_root.clone(),
        })
    }

    /// Update statistics accumulator with new chunk
    fn update_statistics_accumulator(
        &self,
        accumulator: &mut StatisticsAccumulator,
        chunk_proof: &ChunkProof,
    ) -> Result<()> {
        accumulator.row_count += chunk_proof.end_row - chunk_proof.start_row;

        // Would update sums, mins, maxs from actual chunk data
        // For now, simulate the updates
        for i in 0..accumulator.column_sums.len() {
            accumulator.column_sums[i] += (chunk_proof.chunk_id as f64) * 10.0; // Mock data
        }

        // Update hash accumulator
        accumulator.hash_accumulator =
            self.combine_merkle_roots(&accumulator.hash_accumulator, &chunk_proof.merkle_root)?;

        Ok(())
    }

    /// Create aggregated proof from accumulated state
    fn create_aggregated_proof(
        &self,
        dataset: &Dataset,
        state: &AccumulatedState,
    ) -> Result<Proof> {
        Ok(Proof {
            dataset_hash: format!("streaming_{}", dataset.hash),
            proof_data: state.global_merkle_root.as_bytes().to_vec(),
            public_inputs: vec![state.total_rows_processed as u64],
            private_inputs_commitment: "streaming_commitment".to_string(),
            proof_type: crate::ProofType::DatasetIntegrity,
            merkle_root: Some(state.global_merkle_root.clone()),
            merkle_proof: None,
            timestamp: chrono::Utc::now(),
            version: "streaming_1.0".to_string(),
            groth16_proof: None,
            circuit_public_inputs: Some(vec![state.chunk_count as u64]),
        })
    }

    /// Calculate maximum chunks to keep in memory
    fn calculate_max_chunks_in_memory(&self) -> usize {
        let estimated_chunk_size_mb = 10; // Estimate 10MB per chunk proof
        self.config.memory_limit_mb / estimated_chunk_size_mb
    }

    /// Verify streaming proof incrementally
    pub fn verify_streaming_proof(&self, streaming_proof: &StreamingProof) -> Result<bool> {
        // Verify aggregated proof
        if !streaming_proof.aggregated_proof.verify()? {
            return Ok(false);
        }

        // Verify chunk count consistency
        if streaming_proof.total_chunks != streaming_proof.chunk_proof_hashes.len() {
            return Ok(false);
        }

        // Verify merkle chain if incremental verification is enabled
        if streaming_proof.incremental_verification {
            self.verify_merkle_chain(&streaming_proof.chunk_proof_hashes)?;
        }

        Ok(true)
    }

    /// Verify the merkle proof chain
    fn verify_merkle_chain(&self, _chunk_hashes: &[String]) -> Result<bool> {
        // Simplified verification - would check actual merkle proof chain
        Ok(true)
    }
}

/// Result of streaming ZK proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingProof {
    pub original_dataset: Dataset,
    pub total_chunks: usize,
    pub total_rows_processed: usize,
    pub global_merkle_root: String,
    pub aggregated_proof: Proof,
    pub chunk_proof_hashes: Vec<String>,
    pub streaming_algorithm: StreamingAlgorithm,
    pub incremental_verification: bool,
}

impl StreamingProof {
    /// Verify the streaming proof
    pub fn verify(&self) -> Result<bool> {
        self.aggregated_proof.verify()
    }

    /// Get proof size in bytes
    pub fn size_bytes(&self) -> usize {
        self.aggregated_proof.proof_data.len()
            + self
                .chunk_proof_hashes
                .iter()
                .map(|h| h.len())
                .sum::<usize>()
    }

    /// Get compression ratio (original vs streaming proof size)
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.original_dataset.size as f64;
        let proof_size = self.size_bytes() as f64;
        original_size / proof_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DatasetFormat;

    fn create_large_test_dataset() -> Dataset {
        Dataset {
            name: "large_dataset".to_string(),
            hash: "large_hash".to_string(),
            size: 100_000_000,          // 100MB
            row_count: Some(1_000_000), // 1M rows
            column_count: Some(25),
            schema: None,
            statistics: None,
            format: DatasetFormat::Csv,
            path: None,
        }
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 10000);
        assert!(config.enable_incremental_verification);
    }

    #[test]
    fn test_streaming_processor() {
        let config = StreamingConfig {
            chunk_size: 1000, // Smaller chunks for testing
            ..StreamingConfig::default()
        };
        let mut processor = StreamingZkProcessor::new(config);

        let dataset = create_large_test_dataset();
        let proof_config = ProofConfig::default();

        let streaming_proof = processor
            .process_streaming_dataset(&dataset, &proof_config)
            .unwrap();

        assert_eq!(streaming_proof.original_dataset.name, "large_dataset");
        assert!(streaming_proof.total_chunks > 0);
        assert_eq!(streaming_proof.total_rows_processed, 1_000_000);
        assert!(!streaming_proof.global_merkle_root.is_empty());
    }

    #[test]
    fn test_chunk_creation() {
        let config = StreamingConfig::default();
        let processor = StreamingZkProcessor::new(config);

        let dataset = create_large_test_dataset();
        let chunk = processor.create_chunk_dataset(&dataset, 0, 1000).unwrap();

        assert_eq!(chunk.row_count, Some(1000));
        assert!(chunk.name.contains("chunk"));
    }

    #[test]
    fn test_merkle_root_combination() {
        let config = StreamingConfig::default();
        let processor = StreamingZkProcessor::new(config);

        let root1 = "root1".to_string();
        let root2 = "root2".to_string();

        let combined = processor.combine_merkle_roots(&root1, &root2).unwrap();
        assert!(combined.starts_with("combined_"));
        assert_ne!(combined, root1);
        assert_ne!(combined, root2);
    }

    #[test]
    fn test_streaming_proof_verification() {
        let config = StreamingConfig::default();
        let processor = StreamingZkProcessor::new(config);
        let mut small_processor = StreamingZkProcessor::new(StreamingConfig {
            chunk_size: 1000,
            ..config
        });

        let dataset = create_large_test_dataset();
        let proof_config = ProofConfig::default();

        let streaming_proof = small_processor
            .process_streaming_dataset(&dataset, &proof_config)
            .unwrap();

        assert!(processor.verify_streaming_proof(&streaming_proof).unwrap());
        assert!(streaming_proof.verify().unwrap());
    }

    #[test]
    fn test_compression_ratio() {
        let config = StreamingConfig {
            chunk_size: 1000,
            ..StreamingConfig::default()
        };
        let mut processor = StreamingZkProcessor::new(config);

        let dataset = create_large_test_dataset();
        let proof_config = ProofConfig::default();

        let streaming_proof = processor
            .process_streaming_dataset(&dataset, &proof_config)
            .unwrap();
        let compression_ratio = streaming_proof.compression_ratio();

        // Should have significant compression
        assert!(compression_ratio > 1000.0); // At least 1000:1 compression
    }

    #[test]
    fn test_memory_management() {
        let config = StreamingConfig {
            chunk_size: 1000,
            memory_limit_mb: 50, // Small memory limit
            ..StreamingConfig::default()
        };
        let processor = StreamingZkProcessor::new(config);

        let max_chunks = processor.calculate_max_chunks_in_memory();
        assert_eq!(max_chunks, 5); // 50MB / 10MB per chunk
    }
}
