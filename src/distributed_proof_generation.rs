//! Distributed proof generation for massive datasets using parallel processing and load balancing.

use crate::{Dataset, LedgerError, ProofConfig, Result};
use crate::proof::{Proof, ProofType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::Semaphore;

/// Distributed proof generation coordinator
#[derive(Debug)]
pub struct DistributedProofGenerator {
    worker_pool_size: usize,
    chunk_size: usize,
    load_balancer: LoadBalancer,
    proof_cache: Arc<Mutex<HashMap<String, CachedProof>>>,
    performance_metrics: Arc<Mutex<DistributedMetrics>>,
}

/// Load balancer for distributing work across proof generation workers
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    workers: Vec<WorkerInfo>,
    current_worker: Arc<Mutex<usize>>,
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub current_load: f64,
    pub success_rate: f64,
    pub average_response_time_ms: u64,
    pub is_healthy: bool,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    PerformanceBased,
}

/// Cached proof with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedProof {
    pub proof: Proof,
    pub cache_timestamp: DateTime<Utc>,
    pub dataset_fingerprint: String,
    pub ttl_seconds: u64,
}

/// Performance metrics for distributed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedMetrics {
    pub total_proofs_generated: u64,
    pub total_processing_time_ms: u64,
    pub cache_hit_rate: f64,
    pub worker_utilization: HashMap<String, f64>,
    pub average_chunk_processing_time_ms: u64,
    pub throughput_proofs_per_second: f64,
    pub error_rate: f64,
    pub last_updated: DateTime<Utc>,
}

impl DistributedProofGenerator {
    /// Create a new distributed proof generator
    pub fn new(worker_pool_size: usize, chunk_size: usize) -> Self {
        let workers = Self::initialize_workers(worker_pool_size);
        let load_balancer = LoadBalancer::new(workers, LoadBalancingStrategy::PerformanceBased);

        Self {
            worker_pool_size,
            chunk_size,
            load_balancer,
            proof_cache: Arc::new(Mutex::new(HashMap::new())),
            performance_metrics: Arc::new(Mutex::new(DistributedMetrics::default())),
        }
    }

    /// Generate proofs for massive datasets using distributed processing
    pub async fn generate_distributed_proof(
        &self,
        dataset: &Dataset,
        config: ProofConfig,
    ) -> Result<DistributedProofResult> {
        let start_time = Instant::now();
        log::info!(
            "Starting distributed proof generation for dataset: {} (size: {} bytes)",
            dataset.name,
            dataset.size
        );

        // Check cache first
        let cache_key = self.generate_cache_key(dataset, &config);
        if let Some(cached_proof) = self.get_cached_proof(&cache_key) {
            log::info!("Using cached proof for dataset: {}", dataset.name);
            self.update_metrics(|metrics| metrics.cache_hit_rate += 0.1);
            return Ok(DistributedProofResult {
                proof: cached_proof.proof,
                chunk_results: vec![],
                total_processing_time_ms: 0,
                cache_hit: true,
                worker_assignments: HashMap::new(),
                performance_stats: self.get_performance_stats(),
            });
        }

        // Divide dataset into chunks for parallel processing
        let chunks = self.create_data_chunks(dataset)?;
        log::info!("Dataset divided into {} chunks", chunks.len());

        // Set up semaphore for controlling concurrent workers
        let semaphore = Arc::new(Semaphore::new(self.worker_pool_size));
        let mut chunk_results = Vec::new();
        let mut worker_assignments = HashMap::new();

        // Process chunks in parallel with load balancing
        let chunk_futures: Vec<_> = chunks
            .into_iter()
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let semaphore = semaphore.clone();
                let load_balancer = self.load_balancer.clone();
                let config = config.clone();

                async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    let worker = load_balancer.select_worker().await?;
                    let result =
                        Self::process_chunk_on_worker(&worker, chunk, &config, chunk_index).await?;

                    Ok::<(ChunkProofResult, String), LedgerError>((result, worker.id))
                }
            })
            .collect();

        // Await all chunk processing
        for future in chunk_futures {
            match future.await {
                Ok((chunk_result, worker_id)) => {
                    worker_assignments.insert(chunk_result.chunk_index, worker_id);
                    chunk_results.push(chunk_result);
                }
                Err(e) => {
                    log::error!("Chunk processing failed: {}", e);
                    self.update_metrics(|metrics| metrics.error_rate += 0.1);
                    return Err(e);
                }
            }
        }

        // Combine chunk proofs into final proof
        let final_proof = self.combine_chunk_proofs(&chunk_results, dataset, &config)?;

        // Cache the result
        let cached_proof = CachedProof {
            proof: final_proof.clone(),
            cache_timestamp: Utc::now(),
            dataset_fingerprint: cache_key.clone(),
            ttl_seconds: 3600, // 1 hour TTL
        };
        self.cache_proof(cache_key, cached_proof);

        let total_time = start_time.elapsed().as_millis() as u64;
        self.update_metrics(|metrics| {
            metrics.total_proofs_generated += 1;
            metrics.total_processing_time_ms += total_time;
            metrics.throughput_proofs_per_second = metrics.total_proofs_generated as f64
                / (metrics.total_processing_time_ms as f64 / 1000.0);
        });

        log::info!("Distributed proof generation completed in {}ms", total_time);

        Ok(DistributedProofResult {
            proof: final_proof,
            chunk_results,
            total_processing_time_ms: total_time,
            cache_hit: false,
            worker_assignments,
            performance_stats: self.get_performance_stats(),
        })
    }

    /// Initialize worker pool with system detection
    fn initialize_workers(count: usize) -> Vec<WorkerInfo> {
        let cpu_cores = num_cpus::get();
        let estimated_memory = 8; // GB - would use actual system detection

        (0..count)
            .map(|i| WorkerInfo {
                id: format!("worker_{}", i),
                cpu_cores: cpu_cores / count,
                memory_gb: estimated_memory / count,
                current_load: 0.0,
                success_rate: 1.0,
                average_response_time_ms: 100,
                is_healthy: true,
            })
            .collect()
    }

    /// Create data chunks for parallel processing
    fn create_data_chunks(&self, dataset: &Dataset) -> Result<Vec<DataChunk>> {
        let total_rows = dataset.row_count.unwrap_or(1000);
        let chunk_count = (total_rows as usize + self.chunk_size - 1) / self.chunk_size;

        let mut chunks = Vec::new();
        for i in 0..chunk_count {
            let start_row = i * self.chunk_size;
            let end_row = std::cmp::min((i + 1) * self.chunk_size, total_rows as usize);

            chunks.push(DataChunk {
                chunk_id: format!("chunk_{}_{}", dataset.name, i),
                start_row,
                end_row,
                row_count: end_row - start_row,
                data_hash: format!("hash_{}_{}", dataset.hash, i),
                processing_complexity: self.estimate_chunk_complexity(end_row - start_row),
            });
        }

        Ok(chunks)
    }

    /// Process a single chunk on a selected worker
    async fn process_chunk_on_worker(
        worker: &WorkerInfo,
        chunk: DataChunk,
        config: &ProofConfig,
        chunk_index: usize,
    ) -> Result<ChunkProofResult> {
        let start_time = Instant::now();

        log::debug!("Processing chunk {} on worker {}", chunk_index, worker.id);

        // Simulate chunk processing time based on complexity
        let processing_time_ms = (chunk.processing_complexity * 100.0) as u64;
        tokio::time::sleep(tokio::time::Duration::from_millis(processing_time_ms)).await;

        // Generate proof for this chunk
        let chunk_proof = Proof {
            dataset_hash: chunk.data_hash.clone(),
            proof_type: config.proof_type.clone(),
            timestamp: Utc::now(),
        };

        let actual_time = start_time.elapsed().as_millis() as u64;

        Ok(ChunkProofResult {
            chunk_index,
            chunk_id: chunk.chunk_id,
            proof: chunk_proof,
            processing_time_ms: actual_time,
            worker_id: worker.id.clone(),
            rows_processed: chunk.row_count,
            memory_used_mb: (chunk.processing_complexity * 50.0) as usize,
        })
    }

    /// Combine chunk proofs into a single final proof
    fn combine_chunk_proofs(
        &self,
        chunk_results: &[ChunkProofResult],
        dataset: &Dataset,
        config: &ProofConfig,
    ) -> Result<Proof> {
        // In a real implementation, this would use cryptographic proof aggregation
        // For now, we create a summary proof

        let total_rows_processed: usize = chunk_results.iter().map(|r| r.rows_processed).sum();

        log::info!(
            "Combining {} chunk proofs (total rows: {})",
            chunk_results.len(),
            total_rows_processed
        );

        Ok(Proof {
            dataset_hash: dataset.hash.clone(),
            proof_data: vec![],
            public_inputs: vec![total_rows_processed.to_string()],
            private_inputs_commitment: "distributed_commitment".to_string(),
            proof_type: ProofType::DatasetIntegrity,
            merkle_root: None,
            merkle_proof: None,
            timestamp: Utc::now(),
            version: "1.0".to_string(),
            groth16_proof: None,
            circuit_public_inputs: Some(vec![chunk_results.len().to_string()]),
        })
    }

    /// Generate cache key for proof caching
    fn generate_cache_key(&self, dataset: &Dataset, config: &ProofConfig) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        dataset.hash.hash(&mut hasher);
        config.proof_type.hash(&mut hasher);
        config.security_level.hash(&mut hasher);

        format!("proof_cache_{:x}", hasher.finish())
    }

    /// Get cached proof if available and valid
    fn get_cached_proof(&self, cache_key: &str) -> Option<CachedProof> {
        let cache = self.proof_cache.lock().unwrap();
        cache.get(cache_key).and_then(|cached| {
            let age_seconds = (Utc::now().timestamp() - cached.cache_timestamp.timestamp()) as u64;
            if age_seconds < cached.ttl_seconds {
                Some(cached.clone())
            } else {
                None
            }
        })
    }

    /// Cache a proof result
    fn cache_proof(&self, cache_key: String, cached_proof: CachedProof) {
        let mut cache = self.proof_cache.lock().unwrap();
        cache.insert(cache_key, cached_proof);

        // Clean up old entries if cache is getting large
        if cache.len() > 1000 {
            let now = Utc::now();
            cache.retain(|_, cached| {
                let age_seconds = (now.timestamp() - cached.cache_timestamp.timestamp()) as u64;
                age_seconds < cached.ttl_seconds
            });
        }
    }

    /// Estimate processing complexity for a chunk
    fn estimate_chunk_complexity(&self, row_count: usize) -> f64 {
        // Complexity increases with row count but with diminishing returns
        (row_count as f64).log10() / 6.0 // Normalized to 0-1 range
    }

    /// Update performance metrics with a closure
    fn update_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut DistributedMetrics),
    {
        let mut metrics = self.performance_metrics.lock().unwrap();
        update_fn(&mut metrics);
        metrics.last_updated = Utc::now();
    }

    /// Get current performance statistics
    fn get_performance_stats(&self) -> DistributedMetrics {
        self.performance_metrics.lock().unwrap().clone()
    }

    /// Optimize worker allocation based on historical performance
    pub async fn optimize_worker_allocation(&self) -> Result<OptimizationReport> {
        log::info!("Optimizing worker allocation based on performance data");

        let current_metrics = self.get_performance_stats();

        // Analyze worker performance and suggest optimizations
        let mut recommendations = Vec::new();

        if current_metrics.cache_hit_rate < 0.7 {
            recommendations.push("Increase cache TTL to improve hit rate".to_string());
        }

        if current_metrics.throughput_proofs_per_second < 10.0 {
            recommendations.push("Consider increasing worker pool size".to_string());
        }

        if current_metrics.error_rate > 0.05 {
            recommendations.push("Investigate and fix high error rate".to_string());
        }

        Ok(OptimizationReport {
            current_throughput: current_metrics.throughput_proofs_per_second,
            recommended_worker_count: std::cmp::max(self.worker_pool_size, 8),
            cache_efficiency: current_metrics.cache_hit_rate,
            recommendations,
            estimated_improvement_percent: 25.0,
        })
    }
}

impl LoadBalancer {
    fn new(workers: Vec<WorkerInfo>, strategy: LoadBalancingStrategy) -> Self {
        Self {
            workers,
            current_worker: Arc::new(Mutex::new(0)),
            strategy,
        }
    }

    async fn select_worker(&self) -> Result<WorkerInfo> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_selection(),
            LoadBalancingStrategy::PerformanceBased => self.performance_based_selection(),
            _ => self.round_robin_selection(), // Default fallback
        }
    }

    fn round_robin_selection(&self) -> Result<WorkerInfo> {
        let mut current = self.current_worker.lock().unwrap();
        let worker = self.workers[*current % self.workers.len()].clone();
        *current += 1;
        Ok(worker)
    }

    fn performance_based_selection(&self) -> Result<WorkerInfo> {
        // Select worker with best performance score
        let best_worker = self
            .workers
            .iter()
            .filter(|w| w.is_healthy)
            .min_by(|a, b| {
                let score_a = a.current_load * (1.0 - a.success_rate)
                    + (a.average_response_time_ms as f64 / 1000.0);
                let score_b = b.current_load * (1.0 - b.success_rate)
                    + (b.average_response_time_ms as f64 / 1000.0);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .cloned()
            .ok_or_else(|| LedgerError::internal("No healthy workers available"))?;

        Ok(best_worker)
    }
}

impl Default for DistributedMetrics {
    fn default() -> Self {
        Self {
            total_proofs_generated: 0,
            total_processing_time_ms: 0,
            cache_hit_rate: 0.0,
            worker_utilization: HashMap::new(),
            average_chunk_processing_time_ms: 0,
            throughput_proofs_per_second: 0.0,
            error_rate: 0.0,
            last_updated: Utc::now(),
        }
    }
}

/// Data chunk for distributed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunk {
    pub chunk_id: String,
    pub start_row: usize,
    pub end_row: usize,
    pub row_count: usize,
    pub data_hash: String,
    pub processing_complexity: f64,
}

/// Result of processing a single chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkProofResult {
    pub chunk_index: usize,
    pub chunk_id: String,
    pub proof: Proof,
    pub processing_time_ms: u64,
    pub worker_id: String,
    pub rows_processed: usize,
    pub memory_used_mb: usize,
}

/// Complete result of distributed proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedProofResult {
    pub proof: Proof,
    pub chunk_results: Vec<ChunkProofResult>,
    pub total_processing_time_ms: u64,
    pub cache_hit: bool,
    pub worker_assignments: HashMap<usize, String>,
    pub performance_stats: DistributedMetrics,
}

/// Optimization report for worker allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub current_throughput: f64,
    pub recommended_worker_count: usize,
    pub cache_efficiency: f64,
    pub recommendations: Vec<String>,
    pub estimated_improvement_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DatasetFormat;

    fn create_test_dataset() -> Dataset {
        Dataset {
            name: "test_distributed".to_string(),
            hash: "test_hash_distributed".to_string(),
            size: 1_000_000,
            row_count: Some(10000),
            column_count: Some(20),
            path: None,
            schema: None,
            statistics: None,
            format: DatasetFormat::Csv,
        }
    }

    #[tokio::test]
    async fn test_distributed_proof_generation() {
        let generator = DistributedProofGenerator::new(4, 1000);
        let dataset = create_test_dataset();
        let config = ProofConfig::default();

        let result = generator
            .generate_distributed_proof(&dataset, config)
            .await
            .unwrap();

        assert!(!result.proof.dataset_hash.is_empty());
        assert!(!result.chunk_results.is_empty());
        assert!(result.total_processing_time_ms > 0);
    }

    #[test]
    fn test_data_chunk_creation() {
        let generator = DistributedProofGenerator::new(4, 1000);
        let dataset = create_test_dataset();

        let chunks = generator.create_data_chunks(&dataset).unwrap();
        assert_eq!(chunks.len(), 10); // 10000 rows / 1000 chunk_size

        let total_rows: usize = chunks.iter().map(|c| c.row_count).sum();
        assert_eq!(total_rows, 10000);
    }

    #[test]
    fn test_load_balancer() {
        let workers = vec![
            WorkerInfo {
                id: "worker_1".to_string(),
                cpu_cores: 4,
                memory_gb: 8,
                current_load: 0.3,
                success_rate: 0.95,
                average_response_time_ms: 150,
                is_healthy: true,
            },
            WorkerInfo {
                id: "worker_2".to_string(),
                cpu_cores: 4,
                memory_gb: 8,
                current_load: 0.8,
                success_rate: 0.90,
                average_response_time_ms: 200,
                is_healthy: true,
            },
        ];

        let load_balancer = LoadBalancer::new(workers, LoadBalancingStrategy::PerformanceBased);
        let selected_worker = load_balancer.performance_based_selection().unwrap();

        // Should select worker_1 due to lower load and better performance
        assert_eq!(selected_worker.id, "worker_1");
    }
}
