//! Streaming Zero-Knowledge Proofs for Large Datasets
//!
//! This module enables ZKP generation for datasets that are too large to fit in memory
//! by processing data in chunks and creating incrementally verifiable proofs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs::File as AsyncFile;
use tokio::io::{AsyncReadExt, BufReader as AsyncBufReader};

use crate::circuits::{DatasetProperty, ZKProof};
use crate::{LedgerError, Result};

/// Configuration for streaming ZKP generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub chunk_size_bytes: usize,
    pub max_chunk_size_rows: usize,
    pub overlap_bytes: usize,
    pub parallel_chunks: usize,
    pub memory_limit_mb: usize,
    pub checkpoint_interval: usize,
    pub compression_enabled: bool,
    pub incremental_verification: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size_bytes: 1024 * 1024 * 10, // 10 MB chunks
            max_chunk_size_rows: 100_000,
            overlap_bytes: 1024, // 1 KB overlap for CSV parsing
            parallel_chunks: 4,
            memory_limit_mb: 512,
            checkpoint_interval: 10, // Every 10 chunks
            compression_enabled: true,
            incremental_verification: true,
        }
    }
}

/// A streaming dataset processor for ZKP generation
#[derive(Debug)]
pub struct StreamingZKPProcessor {
    pub dataset_path: String,
    pub config: StreamingConfig,
    pub chunk_proofs: Vec<ChunkProof>,
    pub aggregated_properties: HashMap<String, AggregatedProperty>,
    pub processing_state: ProcessingState,
    pub total_bytes_processed: u64,
    pub total_rows_processed: u64,
    pub start_time: DateTime<Utc>,
}

/// State of streaming processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingState {
    Initialized,
    Processing {
        current_chunk: usize,
        total_chunks: usize,
        progress_percent: f64,
    },
    Aggregating,
    Completed,
    Failed {
        error: String,
    },
    Paused {
        checkpoint: ProcessingCheckpoint,
    },
}

/// Proof for a single chunk of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProof {
    pub chunk_id: usize,
    pub chunk_offset: u64,
    pub chunk_size: usize,
    pub chunk_hash: String,
    pub row_count: u64,
    pub properties: HashMap<String, ChunkProperty>,
    pub proof_data: Vec<u8>,
    pub verification_key: String,
    pub timestamp: DateTime<Utc>,
    pub processing_time_ms: u64,
}

/// Property computed for a single chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkProperty {
    Count(u64),
    Sum(f64),
    SumOfSquares(f64),
    Min(f64),
    Max(f64),
    Histogram {
        bins: Vec<u64>,
        min_val: f64,
        max_val: f64,
    },
    Cardinality(u64),
    Null(u64),
    Schema(Vec<String>),
}

/// Aggregated property across all chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProperty {
    pub property_type: String,
    pub total_count: u64,
    pub aggregated_value: AggregatedValue,
    pub confidence_interval: Option<(f64, f64)>,
    pub chunks_contributing: usize,
}

/// Aggregated values for different property types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregatedValue {
    Count(u64),
    Mean {
        mean: f64,
        variance: f64,
    },
    Range {
        min: f64,
        max: f64,
    },
    Distribution {
        histogram: Vec<u64>,
        total_bins: usize,
    },
    Unique {
        estimated_cardinality: u64,
        confidence: f64,
    },
    Schema {
        columns: Vec<String>,
        consistency: f64,
    },
}

/// Checkpoint for resumable processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingCheckpoint {
    pub last_processed_chunk: usize,
    pub last_byte_offset: u64,
    pub partial_aggregations: HashMap<String, AggregatedProperty>,
    pub checkpoint_timestamp: DateTime<Utc>,
    pub chunk_proofs: Vec<ChunkProof>,
}

/// Streaming result with incremental proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingZKPResult {
    pub final_proof: ZKProof,
    pub chunk_proofs: Vec<ChunkProof>,
    pub aggregated_properties: HashMap<String, AggregatedProperty>,
    pub processing_stats: ProcessingStats,
    pub verification_chain: Vec<String>,
}

/// Statistics about the streaming processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub total_processing_time_ms: u64,
    pub total_bytes_processed: u64,
    pub total_rows_processed: u64,
    pub chunks_processed: usize,
    pub average_chunk_time_ms: f64,
    pub throughput_mbps: f64,
    pub memory_peak_mb: f64,
    pub verification_time_ms: u64,
}

impl StreamingZKPProcessor {
    /// Create a new streaming ZKP processor
    pub fn new<P: AsRef<Path>>(dataset_path: P, config: StreamingConfig) -> Result<Self> {
        let path_str = dataset_path.as_ref().to_string_lossy().to_string();

        log::info!("Creating streaming ZKP processor for dataset: {}", path_str);

        // Validate file exists and is readable
        if !dataset_path.as_ref().exists() {
            return Err(LedgerError::not_found("file", path_str));
        }

        let metadata = std::fs::metadata(&dataset_path)?;
        if metadata.len() == 0 {
            return Err(LedgerError::dataset_error("Empty dataset file", false));
        }

        log::info!(
            "Dataset size: {} bytes, chunk size: {} bytes",
            metadata.len(),
            config.chunk_size_bytes
        );

        Ok(StreamingZKPProcessor {
            dataset_path: path_str,
            config,
            chunk_proofs: Vec::new(),
            aggregated_properties: HashMap::new(),
            processing_state: ProcessingState::Initialized,
            total_bytes_processed: 0,
            total_rows_processed: 0,
            start_time: Utc::now(),
        })
    }

    /// Process the dataset in streaming fashion
    pub async fn process_streaming(
        &mut self,
        target_properties: Vec<DatasetProperty>,
    ) -> Result<StreamingZKPResult> {
        log::info!(
            "Starting streaming ZKP processing for {} properties",
            target_properties.len()
        );

        self.start_time = Utc::now();

        // Calculate total chunks
        let file_size = std::fs::metadata(&self.dataset_path)?.len();
        let estimated_chunks =
            (file_size as usize + self.config.chunk_size_bytes - 1) / self.config.chunk_size_bytes;

        self.processing_state = ProcessingState::Processing {
            current_chunk: 0,
            total_chunks: estimated_chunks,
            progress_percent: 0.0,
        };

        // Process chunks
        let mut chunk_id = 0;
        let mut file_offset = 0u64;
        let file = AsyncFile::open(&self.dataset_path).await?;
        let mut reader = AsyncBufReader::new(file);

        while file_offset < file_size {
            let chunk_start_time = std::time::Instant::now();

            // Read chunk
            let chunk_data = self.read_chunk(&mut reader, file_offset).await?;
            if chunk_data.is_empty() {
                break;
            }

            // Process chunk and generate proof
            let chunk_proof = self
                .process_chunk(chunk_id, file_offset, chunk_data, &target_properties)
                .await?;

            // Update aggregations
            self.update_aggregations(&chunk_proof).await?;

            // Store chunk proof
            self.chunk_proofs.push(chunk_proof.clone());

            // Update progress
            chunk_id += 1;
            file_offset += chunk_proof.chunk_size as u64;
            self.total_bytes_processed += chunk_proof.chunk_size as u64;
            self.total_rows_processed += chunk_proof.row_count;

            let progress_percent = (file_offset as f64 / file_size as f64) * 100.0;
            self.processing_state = ProcessingState::Processing {
                current_chunk: chunk_id,
                total_chunks: estimated_chunks,
                progress_percent,
            };

            let chunk_time = chunk_start_time.elapsed().as_millis() as u64;
            log::info!(
                "Processed chunk {} ({:.1}%) in {}ms: {} bytes, {} rows",
                chunk_id - 1,
                progress_percent,
                chunk_time,
                chunk_proof.chunk_size,
                chunk_proof.row_count
            );

            // Checkpoint if needed
            if chunk_id % self.config.checkpoint_interval == 0 {
                self.create_checkpoint().await?;
            }

            // Check memory usage
            if self.check_memory_limit().await? {
                self.flush_chunks_to_disk().await?;
            }
        }

        // Final aggregation and proof generation
        self.processing_state = ProcessingState::Aggregating;
        let final_result = self.finalize_processing().await?;

        self.processing_state = ProcessingState::Completed;

        let total_time = self
            .start_time
            .signed_duration_since(Utc::now())
            .num_milliseconds()
            .abs() as u64;
        log::info!(
            "Streaming ZKP processing completed in {}ms: {} chunks, {} bytes, {} rows",
            total_time,
            self.chunk_proofs.len(),
            self.total_bytes_processed,
            self.total_rows_processed
        );

        Ok(final_result)
    }

    /// Read a chunk of data from the file
    async fn read_chunk(
        &self,
        reader: &mut AsyncBufReader<AsyncFile>,
        _offset: u64,
    ) -> Result<Vec<u8>> {
        let mut chunk_data = Vec::with_capacity(self.config.chunk_size_bytes);
        let mut bytes_read = 0;

        // Read up to chunk size
        while bytes_read < self.config.chunk_size_bytes {
            let mut buffer = vec![0; (self.config.chunk_size_bytes - bytes_read).min(8192)];
            let read = reader.read(&mut buffer).await?;

            if read == 0 {
                break; // EOF
            }

            chunk_data.extend_from_slice(&buffer[..read]);
            bytes_read += read;
        }

        Ok(chunk_data)
    }

    /// Process a single chunk and generate proof
    async fn process_chunk(
        &self,
        chunk_id: usize,
        offset: u64,
        chunk_data: Vec<u8>,
        target_properties: &[DatasetProperty],
    ) -> Result<ChunkProof> {
        let start_time = std::time::Instant::now();

        // Calculate chunk hash
        let chunk_hash = format!("{:x}", Sha256::digest(&chunk_data));

        // Parse chunk for CSV (simplified parsing)
        let (row_count, properties) = self.analyze_chunk_data(&chunk_data).await?;

        // Generate proof for chunk
        let proof_data = self
            .generate_chunk_proof(&chunk_data, &properties, target_properties)
            .await?;
        let verification_key = self
            .generate_chunk_verification_key(chunk_id, &chunk_hash)
            .await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ChunkProof {
            chunk_id,
            chunk_offset: offset,
            chunk_size: chunk_data.len(),
            chunk_hash,
            row_count,
            properties,
            proof_data,
            verification_key,
            timestamp: Utc::now(),
            processing_time_ms: processing_time,
        })
    }

    /// Analyze chunk data to extract properties
    async fn analyze_chunk_data(
        &self,
        chunk_data: &[u8],
    ) -> Result<(u64, HashMap<String, ChunkProperty>)> {
        let mut properties = HashMap::new();
        let content = String::from_utf8_lossy(chunk_data);
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() {
            return Ok((0, properties));
        }

        let row_count = lines.len() as u64;

        // Basic CSV analysis
        if let Some(header_line) = lines.first() {
            let columns: Vec<&str> = header_line.split(',').collect();
            properties.insert(
                "schema".to_string(),
                ChunkProperty::Schema(columns.iter().map(|s| s.to_string()).collect()),
            );

            // Analyze numeric columns (simplified)
            for (col_idx, column) in columns.iter().enumerate() {
                let mut values = Vec::new();
                let mut null_count = 0u64;

                for line in lines.iter().skip(1) {
                    let fields: Vec<&str> = line.split(',').collect();
                    if col_idx < fields.len() {
                        if let Ok(value) = fields[col_idx].trim().parse::<f64>() {
                            values.push(value);
                        } else {
                            null_count += 1;
                        }
                    }
                }

                if !values.is_empty() {
                    let sum: f64 = values.iter().sum();
                    let sum_of_squares: f64 = values.iter().map(|x| x * x).sum();
                    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                    properties.insert(format!("{}_sum", column), ChunkProperty::Sum(sum));
                    properties.insert(
                        format!("{}_sum_squares", column),
                        ChunkProperty::SumOfSquares(sum_of_squares),
                    );
                    properties.insert(format!("{}_min", column), ChunkProperty::Min(min_val));
                    properties.insert(format!("{}_max", column), ChunkProperty::Max(max_val));
                    properties.insert(
                        format!("{}_count", column),
                        ChunkProperty::Count(values.len() as u64),
                    );
                }

                if null_count > 0 {
                    properties.insert(format!("{}_nulls", column), ChunkProperty::Null(null_count));
                }
            }
        }

        properties.insert("row_count".to_string(), ChunkProperty::Count(row_count));

        Ok((row_count, properties))
    }

    /// Generate ZKP for a chunk
    async fn generate_chunk_proof(
        &self,
        chunk_data: &[u8],
        properties: &HashMap<String, ChunkProperty>,
        target_properties: &[DatasetProperty],
    ) -> Result<Vec<u8>> {
        // Simulate ZKP generation for chunk
        let mut proof_data = Vec::new();

        // Add proof header
        proof_data.extend_from_slice(b"CHUNK_PROOF");

        // Add chunk hash
        let chunk_hash = Sha256::digest(chunk_data);
        proof_data.extend_from_slice(&chunk_hash[..8]);

        // Add property commitments
        for (prop_name, prop_value) in properties {
            let prop_hash = Sha256::digest(format!("{}:{:?}", prop_name, prop_value).as_bytes());
            proof_data.extend_from_slice(&prop_hash[..4]);
        }

        // Add target property proofs
        for target_prop in target_properties {
            let target_hash = Sha256::digest(format!("{:?}", target_prop).as_bytes());
            proof_data.extend_from_slice(&target_hash[..4]);
        }

        // Pad to minimum size
        while proof_data.len() < 64 {
            proof_data.push(0);
        }

        Ok(proof_data)
    }

    /// Generate verification key for chunk
    async fn generate_chunk_verification_key(
        &self,
        chunk_id: usize,
        chunk_hash: &str,
    ) -> Result<String> {
        let mut hasher = Sha256::new();
        hasher.update(format!("chunk_{}", chunk_id).as_bytes());
        hasher.update(chunk_hash.as_bytes());
        hasher.update(b"VERIFICATION_KEY");

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Update running aggregations with new chunk
    async fn update_aggregations(&mut self, chunk_proof: &ChunkProof) -> Result<()> {
        for (prop_name, chunk_prop) in &chunk_proof.properties {
            let aggregated = self
                .aggregated_properties
                .entry(prop_name.clone())
                .or_insert_with(|| AggregatedProperty {
                    property_type: format!("{:?}", chunk_prop),
                    total_count: 0,
                    aggregated_value: AggregatedValue::Count(0),
                    confidence_interval: None,
                    chunks_contributing: 0,
                });

            // Update aggregation based on property type
            match chunk_prop {
                ChunkProperty::Count(count) => {
                    if let AggregatedValue::Count(ref mut total) = aggregated.aggregated_value {
                        *total += count;
                    }
                }
                ChunkProperty::Sum(sum) => {
                    aggregated.total_count += 1;
                    // Update running mean calculation
                    if let AggregatedValue::Mean {
                        ref mut mean,
                        ref mut variance,
                    } = aggregated.aggregated_value
                    {
                        let new_count = aggregated.total_count as f64;
                        let delta = sum - *mean;
                        *mean += delta / new_count;
                        *variance += delta * (sum - *mean);
                    } else {
                        aggregated.aggregated_value = AggregatedValue::Mean {
                            mean: *sum,
                            variance: 0.0,
                        };
                    }
                }
                ChunkProperty::Min(min_val) => {
                    if let AggregatedValue::Range {
                        ref mut min,
                        max: _,
                    } = aggregated.aggregated_value
                    {
                        *min = min.min(*min_val);
                    } else {
                        aggregated.aggregated_value = AggregatedValue::Range {
                            min: *min_val,
                            max: *min_val,
                        };
                    }
                }
                ChunkProperty::Max(max_val) => {
                    if let AggregatedValue::Range {
                        min: _,
                        ref mut max,
                    } = aggregated.aggregated_value
                    {
                        *max = max.max(*max_val);
                    } else {
                        aggregated.aggregated_value = AggregatedValue::Range {
                            min: *max_val,
                            max: *max_val,
                        };
                    }
                }
                _ => {
                    // Handle other property types
                }
            }

            aggregated.chunks_contributing += 1;
        }

        Ok(())
    }

    /// Create processing checkpoint
    async fn create_checkpoint(&self) -> Result<()> {
        let checkpoint = ProcessingCheckpoint {
            last_processed_chunk: self.chunk_proofs.len(),
            last_byte_offset: self.total_bytes_processed,
            partial_aggregations: self.aggregated_properties.clone(),
            checkpoint_timestamp: Utc::now(),
            chunk_proofs: self.chunk_proofs.clone(),
        };

        let checkpoint_path = format!("{}.checkpoint", self.dataset_path);
        let checkpoint_data = serde_json::to_vec(&checkpoint)?;
        tokio::fs::write(checkpoint_path, checkpoint_data).await?;

        log::info!(
            "Created checkpoint at chunk {}",
            checkpoint.last_processed_chunk
        );
        Ok(())
    }

    /// Check memory usage against limit
    async fn check_memory_limit(&self) -> Result<bool> {
        // Simplified memory check (in practice, use system memory monitoring)
        let estimated_memory = self.chunk_proofs.len() * 1024; // Rough estimate
        Ok(estimated_memory > self.config.memory_limit_mb * 1024 * 1024)
    }

    /// Flush chunks to disk to free memory
    async fn flush_chunks_to_disk(&mut self) -> Result<()> {
        if self.chunk_proofs.is_empty() {
            return Ok(());
        }

        let flush_path = format!("{}.chunks.{}", self.dataset_path, Utc::now().timestamp());
        let chunk_data = serde_json::to_vec(&self.chunk_proofs)?;
        tokio::fs::write(flush_path, chunk_data).await?;

        log::info!("Flushed {} chunks to disk", self.chunk_proofs.len());

        // Keep only recent chunks in memory
        let keep_count = self.config.parallel_chunks;
        if self.chunk_proofs.len() > keep_count {
            self.chunk_proofs
                .drain(0..self.chunk_proofs.len() - keep_count);
        }

        Ok(())
    }

    /// Finalize processing and generate final proof
    async fn finalize_processing(&self) -> Result<StreamingZKPResult> {
        log::info!("Finalizing streaming ZKP processing");

        // Generate final aggregated proof
        let final_proof = self.generate_final_proof().await?;

        // Create verification chain
        let verification_chain = self
            .chunk_proofs
            .iter()
            .map(|chunk| chunk.verification_key.clone())
            .collect();

        // Calculate processing statistics
        let total_time = Utc::now()
            .signed_duration_since(self.start_time)
            .num_milliseconds()
            .abs() as u64;
        let processing_stats = ProcessingStats {
            total_processing_time_ms: total_time,
            total_bytes_processed: self.total_bytes_processed,
            total_rows_processed: self.total_rows_processed,
            chunks_processed: self.chunk_proofs.len(),
            average_chunk_time_ms: if !self.chunk_proofs.is_empty() {
                self.chunk_proofs
                    .iter()
                    .map(|c| c.processing_time_ms)
                    .sum::<u64>() as f64
                    / self.chunk_proofs.len() as f64
            } else {
                0.0
            },
            throughput_mbps: if total_time > 0 {
                (self.total_bytes_processed as f64 / (1024.0 * 1024.0))
                    / (total_time as f64 / 1000.0)
            } else {
                0.0
            },
            memory_peak_mb: 0.0,     // Would be tracked in real implementation
            verification_time_ms: 0, // Would measure verification time
        };

        Ok(StreamingZKPResult {
            final_proof,
            chunk_proofs: self.chunk_proofs.clone(),
            aggregated_properties: self.aggregated_properties.clone(),
            processing_stats,
            verification_chain,
        })
    }

    /// Generate final aggregated proof
    async fn generate_final_proof(&self) -> Result<ZKProof> {
        let mut final_proof_data = Vec::new();

        // Add proof header
        final_proof_data.extend_from_slice(b"STREAMING_FINAL_PROOF");

        // Add aggregated property commitments
        for (prop_name, aggregated_prop) in &self.aggregated_properties {
            let prop_commitment =
                Sha256::digest(format!("{}:{:?}", prop_name, aggregated_prop).as_bytes());
            final_proof_data.extend_from_slice(&prop_commitment[..8]);
        }

        // Add chunk proof hashes
        for chunk_proof in &self.chunk_proofs {
            let chunk_commitment = Sha256::digest(&chunk_proof.proof_data);
            final_proof_data.extend_from_slice(&chunk_commitment[..4]);
        }

        // Generate verification key
        let mut hasher = Sha256::new();
        hasher.update(&final_proof_data);
        hasher.update(self.dataset_path.as_bytes());
        let verification_key = format!("{:x}", hasher.finalize());

        // Serialize public inputs
        let public_inputs = serde_json::to_vec(&self.aggregated_properties)?;

        let proof_size = final_proof_data.len();

        Ok(ZKProof {
            proof_id: format!(
                "streaming-{}-{}",
                Sha256::digest(self.dataset_path.as_bytes()).to_vec()[..8]
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<String>(),
                Utc::now().timestamp()
            ),
            circuit_id: format!("streaming-dataset-{}", self.chunk_proofs.len()),
            proof_data: final_proof_data,
            public_inputs,
            verification_key,
            timestamp: Utc::now(),
            proof_size,
            generation_time_ms: Utc::now()
                .signed_duration_since(self.start_time)
                .num_milliseconds()
                .abs() as u64,
        })
    }

    /// Resume processing from checkpoint
    pub async fn resume_from_checkpoint<P: AsRef<Path>>(checkpoint_path: P) -> Result<Self> {
        let checkpoint_data = tokio::fs::read(checkpoint_path).await?;
        let checkpoint: ProcessingCheckpoint = serde_json::from_slice(&checkpoint_data)?;

        log::info!(
            "Resuming processing from checkpoint: chunk {}, offset {}",
            checkpoint.last_processed_chunk,
            checkpoint.last_byte_offset
        );

        // Clone checkpoint before moving parts
        let checkpoint_clone = checkpoint.clone();

        // Reconstruct processor state
        let processor = Self {
            dataset_path: "resumed".to_string(), // Would need to be stored in checkpoint
            config: StreamingConfig::default(),
            chunk_proofs: checkpoint.chunk_proofs,
            aggregated_properties: checkpoint.partial_aggregations,
            processing_state: ProcessingState::Paused {
                checkpoint: checkpoint_clone,
            },
            total_bytes_processed: checkpoint.last_byte_offset,
            total_rows_processed: 0, // Would need to be calculated
            start_time: checkpoint.checkpoint_timestamp,
        };

        Ok(processor)
    }

    /// Get current processing progress
    pub fn get_progress(&self) -> f64 {
        match &self.processing_state {
            ProcessingState::Processing {
                progress_percent, ..
            } => *progress_percent,
            ProcessingState::Completed => 100.0,
            _ => 0.0,
        }
    }
}

/// Factory for creating streaming processors
pub struct StreamingZKPFactory;

impl StreamingZKPFactory {
    /// Create processor optimized for large CSV files
    pub fn create_csv_processor<P: AsRef<Path>>(
        dataset_path: P,
        memory_limit_mb: usize,
    ) -> Result<StreamingZKPProcessor> {
        let config = StreamingConfig {
            chunk_size_bytes: 1024 * 1024 * 50, // 50 MB chunks for CSV
            max_chunk_size_rows: 500_000,
            memory_limit_mb,
            ..Default::default()
        };

        StreamingZKPProcessor::new(dataset_path, config)
    }

    /// Create processor optimized for time series data
    pub fn create_timeseries_processor<P: AsRef<Path>>(
        dataset_path: P,
        _time_window_hours: u64,
    ) -> Result<StreamingZKPProcessor> {
        let config = StreamingConfig {
            chunk_size_bytes: 1024 * 1024 * 20, // 20 MB chunks
            incremental_verification: true,
            checkpoint_interval: 5, // More frequent checkpoints for time series
            ..Default::default()
        };

        StreamingZKPProcessor::new(dataset_path, config)
    }

    /// Create processor with custom configuration
    pub fn create_custom_processor<P: AsRef<Path>>(
        dataset_path: P,
        config: StreamingConfig,
    ) -> Result<StreamingZKPProcessor> {
        StreamingZKPProcessor::new(dataset_path, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_streaming_processor_creation() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "id,value,category").unwrap();
        writeln!(temp_file, "1,100,A").unwrap();
        writeln!(temp_file, "2,200,B").unwrap();

        let processor =
            StreamingZKPProcessor::new(temp_file.path(), StreamingConfig::default()).unwrap();

        assert!(matches!(
            processor.processing_state,
            ProcessingState::Initialized
        ));
    }

    #[tokio::test]
    async fn test_chunk_analysis() {
        let config = StreamingConfig::default();
        let processor = StreamingZKPProcessor {
            dataset_path: "test".to_string(),
            config,
            chunk_proofs: Vec::new(),
            aggregated_properties: HashMap::new(),
            processing_state: ProcessingState::Initialized,
            total_bytes_processed: 0,
            total_rows_processed: 0,
            start_time: Utc::now(),
        };

        let chunk_data = b"id,value\n1,100\n2,200\n3,300";
        let (row_count, properties) = processor.analyze_chunk_data(chunk_data).await.unwrap();

        assert_eq!(row_count, 4); // Including header
        assert!(properties.contains_key("schema"));
        assert!(properties.contains_key("row_count"));
    }
}
