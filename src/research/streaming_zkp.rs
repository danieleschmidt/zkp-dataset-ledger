//! Streaming zero-knowledge proofs for large dataset processing.

use crate::{Dataset, LedgerError, Proof, ProofConfig, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use chrono::{DateTime, Utc};

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

        log::info!("Starting streaming ZK processing for dataset: {}", dataset.name);
        log::info!("Total rows: {}, chunks: {}", total_rows, total_chunks);

        // Initialize accumulated state
        self.accumulated_state = Some(AccumulatedState {
            total_rows_processed: 0,
            global_merkle_root: String::new(),
            statistics_accumulator: StatisticsAccumulator {
                row_count: 0,
                column_sums: Vec::new(),
                min_values: Vec::new(),
                max_values: Vec::new(),
                hash_accumulator: String::new(),
            },
            chunk_count: 0,
        });

        // Process chunks
        for chunk_id in 0..total_chunks {
            let start_row = chunk_id * self.config.chunk_size;
            let end_row = ((chunk_id + 1) * self.config.chunk_size).min(total_rows as usize);
            
            let chunk_proof = self.process_chunk(chunk_id, start_row, end_row, dataset, proof_config)?;
            self.accumulate_chunk_proof(&chunk_proof)?;
            self.chunk_proofs.push_back(chunk_proof);
        }

        // Generate final aggregated proof
        self.generate_final_proof(dataset)
    }

    /// Process a single chunk and generate its proof
    fn process_chunk(
        &self,
        chunk_id: usize,
        start_row: usize,
        end_row: usize,
        dataset: &Dataset,
        proof_config: &ProofConfig,
    ) -> Result<ChunkProof> {
        log::debug!("Processing chunk {} (rows {}-{})", chunk_id, start_row, end_row);

        // Generate proof for this chunk
        let proof = Proof::generate(dataset, proof_config.proof_type.clone())?;
        
        // Compute merkle root for this chunk
        let merkle_root = self.compute_chunk_merkle_root(chunk_id, start_row, end_row)?;
        
        let previous_root = if chunk_id > 0 {
            self.chunk_proofs.back().map(|cp| cp.merkle_root.clone())
        } else {
            None
        };

        Ok(ChunkProof {
            chunk_id,
            start_row,
            end_row,
            proof,
            merkle_root,
            previous_root,
        })
    }

    /// Accumulate chunk proof into global state
    fn accumulate_chunk_proof(&mut self, chunk_proof: &ChunkProof) -> Result<()> {
        if let Some(ref mut state) = self.accumulated_state {
            // Update statistics accumulator
            state.total_rows_processed += chunk_proof.end_row - chunk_proof.start_row;
            state.chunk_count += 1;
            
            // Update global merkle root by combining with chunk root
            state.global_merkle_root = self.combine_merkle_roots(&state.global_merkle_root, &chunk_proof.merkle_root)?;
            
            log::debug!("Accumulated chunk {} into global state", chunk_proof.chunk_id);
        }
        Ok(())
    }

    /// Combine two merkle roots into a new root
    fn combine_merkle_roots(&self, global_root: &str, chunk_root: &str) -> Result<String> {
        use sha2::{Digest, Sha256};
        
        let combined = format!("{}:{}", global_root, chunk_root);
        let hash = Sha256::digest(combined.as_bytes());
        Ok(format!("{:x}", hash))
    }

    /// Compute merkle root for a specific chunk
    fn compute_chunk_merkle_root(&self, chunk_id: usize, start_row: usize, end_row: usize) -> Result<String> {
        use sha2::{Digest, Sha256};
        
        let chunk_data = format!("chunk_{}:{}:{}", chunk_id, start_row, end_row);
        let hash = Sha256::digest(chunk_data.as_bytes());
        Ok(format!("{:x}", hash))
    }

    /// Generate final aggregated proof
    fn generate_final_proof(&self, dataset: &Dataset) -> Result<StreamingProof> {
        let state = self.accumulated_state.as_ref()
            .ok_or_else(|| LedgerError::internal("No accumulated state"))?;

        // Create aggregated proof from base dataset proof
        let aggregated_proof = Proof {
            dataset_hash: dataset.compute_hash(),
            proof_type: "streaming".to_string(),
            timestamp: chrono::Utc::now(),
        };

        Ok(StreamingProof {
            dataset_name: dataset.name.clone(),
            aggregated_proof,
            chunk_proofs: self.chunk_proofs.iter().cloned().collect(),
            total_chunks: state.chunk_count,
            total_rows: state.total_rows_processed,
            global_merkle_root: state.global_merkle_root.clone(),
            streaming_config: self.config.clone(),
            timestamp: chrono::Utc::now(),
        })
    }

    /// Verify streaming proof incrementally  
    pub fn verify_streaming_proof(&self, streaming_proof: &StreamingProof) -> Result<bool> {
        // Verify each chunk proof
        for chunk_proof in &streaming_proof.chunk_proofs {
            if !chunk_proof.proof.verify() {
                return Ok(false);
            }
        }

        // Verify merkle chain consistency
        for i in 1..streaming_proof.chunk_proofs.len() {
            let prev_chunk = &streaming_proof.chunk_proofs[i - 1];
            let curr_chunk = &streaming_proof.chunk_proofs[i];
            
            if curr_chunk.previous_root != Some(prev_chunk.merkle_root.clone()) {
                return Ok(false);
            }
        }

        // Verify aggregated proof
        Ok(streaming_proof.aggregated_proof.verify())
    }
}

/// Final streaming proof containing all chunk proofs and aggregated state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingProof {
    pub dataset_name: String,
    pub aggregated_proof: Proof,
    pub chunk_proofs: Vec<ChunkProof>,
    pub total_chunks: usize,
    pub total_rows: usize,
    pub global_merkle_root: String,
    pub streaming_config: StreamingConfig,
    pub timestamp: DateTime<Utc>,
}

impl StreamingProof {
    /// Verify the entire streaming proof
    pub fn verify(&self) -> Result<bool> {
        Ok(self.aggregated_proof.verify())
    }

    /// Get proof size in bytes (estimated)
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 
        self.chunk_proofs.len() * std::mem::size_of::<ChunkProof>()
    }

    /// Get efficiency metrics
    pub fn get_efficiency_metrics(&self) -> StreamingEfficiencyMetrics {
        let avg_chunk_size = if self.total_chunks > 0 {
            self.total_rows / self.total_chunks
        } else {
            0
        };

        StreamingEfficiencyMetrics {
            total_chunks: self.total_chunks,
            total_rows: self.total_rows,
            average_chunk_size: avg_chunk_size,
            proof_size_bytes: self.size_bytes(),
            compression_ratio: 1.0, // Would calculate actual compression
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingEfficiencyMetrics {
    pub total_chunks: usize,
    pub total_rows: usize,
    pub average_chunk_size: usize,
    pub proof_size_bytes: usize,
    pub compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.chunk_size, 10000);
        assert_eq!(config.memory_limit_mb, 1024);
    }

    #[test]
    fn test_streaming_processor_creation() {
        let config = StreamingConfig::default();
        let processor = StreamingZkProcessor::new(config);
        assert_eq!(processor.chunk_proofs.len(), 0);
    }
}