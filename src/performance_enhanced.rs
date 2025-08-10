//! High-performance parallel processing and optimization for ZKP operations

use crate::{LedgerError, Result};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn, error, instrument, span, Level};
use uuid::Uuid;

/// High-performance parallel processor for ZK operations
pub struct ParallelZKProcessor {
    config: ParallelProcessingConfig,
    thread_pool: rayon::ThreadPool,
    async_semaphore: Arc<Semaphore>,
    performance_monitor: PerformanceMonitor,
    cache_manager: CacheManager,
}

impl ParallelZKProcessor {
    pub fn new(config: ParallelProcessingConfig) -> Result<Self> {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.cpu_threads)
            .build()
            .map_err(|e| LedgerError::InitializationError(format!("Failed to create thread pool: {}", e)))?;
        
        let async_semaphore = Arc::new(Semaphore::new(config.max_concurrent_operations));
        let performance_monitor = PerformanceMonitor::new();
        let cache_manager = CacheManager::new(config.cache_config.clone());
        
        Ok(Self {
            config,
            thread_pool,
            async_semaphore,
            performance_monitor,
            cache_manager,
        })
    }
    
    /// Process multiple datasets in parallel with optimal resource allocation
    #[instrument(skip(self, datasets))]
    pub async fn parallel_dataset_processing<T, F>(&self, 
        datasets: Vec<T>, 
        processor: F
    ) -> Result<Vec<ProcessingResult<T>>>
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T) -> Result<T> + Send + Sync + 'static,
    {
        let start_time = Instant::now();
        let processor = Arc::new(processor);
        
        // Adaptive batch sizing based on dataset complexity
        let batch_size = self.calculate_optimal_batch_size(datasets.len());
        info!("Processing {} datasets in batches of {}", datasets.len(), batch_size);
        
        let mut results = Vec::with_capacity(datasets.len());
        
        // Process in parallel batches
        for batch in datasets.chunks(batch_size) {
            let batch_results = self.process_batch_parallel(batch.to_vec(), processor.clone()).await?;
            results.extend(batch_results);
        }
        
        let total_duration = start_time.elapsed();
        self.performance_monitor.record_parallel_operation(
            "dataset_processing",
            datasets.len(),
            total_duration,
            true
        );
        
        info!("Completed parallel processing of {} datasets in {:?}", datasets.len(), total_duration);
        Ok(results)
    }
    
    /// Process batch in parallel with resource management
    async fn process_batch_parallel<T, F>(&self,
        batch: Vec<T>,
        processor: Arc<F>
    ) -> Result<Vec<ProcessingResult<T>>>
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T) -> Result<T> + Send + Sync + 'static,
    {
        let mut handles = Vec::new();
        
        for item in batch {
            let permit = self.async_semaphore.clone().acquire_owned().await
                .map_err(|e| LedgerError::ConcurrencyError(format!("Failed to acquire semaphore: {}", e)))?;
            
            let processor = processor.clone();
            let performance_monitor = self.performance_monitor.clone();
            
            let handle = tokio::task::spawn_blocking(move || {
                let _permit = permit; // Keep permit alive
                let start_time = Instant::now();
                
                let result = match processor(item.clone()) {
                    Ok(processed_item) => {
                        performance_monitor.record_operation("item_processing", start_time.elapsed(), true);
                        ProcessingResult::Success {
                            input: item,
                            output: processed_item,
                            duration: start_time.elapsed(),
                        }
                    },
                    Err(e) => {
                        performance_monitor.record_operation("item_processing", start_time.elapsed(), false);
                        ProcessingResult::Error {
                            input: item,
                            error: e.to_string(),
                            duration: start_time.elapsed(),
                        }
                    }
                };
                
                result
            });
            
            handles.push(handle);
        }
        
        // Await all results
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await
                .map_err(|e| LedgerError::ConcurrencyError(format!("Task join error: {}", e)))?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Calculate optimal batch size based on system resources and data complexity
    fn calculate_optimal_batch_size(&self, total_items: usize) -> usize {
        let cpu_cores = num_cpus::get();
        let memory_factor = self.get_available_memory_factor();
        
        // Base batch size on CPU cores and available memory
        let base_batch_size = (cpu_cores * 2).max(1);
        let memory_adjusted = (base_batch_size as f64 * memory_factor) as usize;
        
        // Ensure batch size is reasonable for the dataset size
        let adaptive_size = if total_items < 100 {
            (total_items / 4).max(1)
        } else if total_items < 1000 {
            memory_adjusted.min(total_items / 8)
        } else {
            memory_adjusted.min(total_items / 16)
        };
        
        adaptive_size.clamp(1, self.config.max_batch_size)
    }
    
    /// Get memory availability factor (0.1 to 2.0)
    fn get_available_memory_factor(&self) -> f64 {
        // Simplified memory check - in production, use proper memory monitoring
        let available_memory_ratio = 0.7; // Assume 70% memory available
        
        if available_memory_ratio > 0.8 {
            2.0  // Plenty of memory, increase batch size
        } else if available_memory_ratio > 0.6 {
            1.0  // Normal memory, standard batch size
        } else if available_memory_ratio > 0.4 {
            0.7  // Low memory, reduce batch size
        } else {
            0.3  // Very low memory, minimal batch size
        }
    }
    
    /// Parallel proof generation with workload balancing
    #[instrument(skip(self, proof_requests))]
    pub async fn parallel_proof_generation(&self, 
        proof_requests: Vec<ProofRequest>
    ) -> Result<Vec<ProofGenerationResult>> {
        let start_time = Instant::now();
        
        // Sort requests by estimated complexity for better load balancing
        let mut sorted_requests = proof_requests;
        sorted_requests.sort_by_key(|req| self.estimate_proof_complexity(req));
        
        // Use work-stealing approach for optimal load balancing
        let results = self.work_stealing_proof_generation(sorted_requests).await?;
        
        let total_duration = start_time.elapsed();
        self.performance_monitor.record_parallel_operation(
            "proof_generation",
            results.len(),
            total_duration,
            results.iter().all(|r| matches!(r, ProofGenerationResult::Success { .. }))
        );
        
        info!("Generated {} proofs in parallel in {:?}", results.len(), total_duration);
        Ok(results)
    }
    
    /// Work-stealing proof generation for optimal load balancing
    async fn work_stealing_proof_generation(&self,
        proof_requests: Vec<ProofRequest>
    ) -> Result<Vec<ProofGenerationResult>> {
        let work_queue = Arc::new(Mutex::new(proof_requests.into_iter().enumerate().collect::<Vec<_>>()));
        let results = Arc::new(Mutex::new(vec![None; work_queue.lock().unwrap().len()]));
        let num_workers = self.config.cpu_threads;
        
        let mut handles = Vec::new();
        
        for worker_id in 0..num_workers {
            let work_queue = work_queue.clone();
            let results = results.clone();
            let cache_manager = self.cache_manager.clone();
            let performance_monitor = self.performance_monitor.clone();
            
            let handle = tokio::task::spawn_blocking(move || {
                loop {
                    // Try to get work from queue
                    let work_item = {
                        let mut queue = work_queue.lock().unwrap();
                        queue.pop()
                    };
                    
                    match work_item {
                        Some((index, proof_request)) => {
                            let start_time = Instant::now();
                            
                            // Check cache first
                            let cache_key = cache_manager.generate_proof_cache_key(&proof_request);
                            if let Some(cached_proof) = cache_manager.get_cached_proof(&cache_key) {
                                let result = ProofGenerationResult::Success {
                                    request: proof_request,
                                    proof: cached_proof,
                                    duration: start_time.elapsed(),
                                    cache_hit: true,
                                };
                                
                                let mut results_lock = results.lock().unwrap();
                                results_lock[index] = Some(result);
                                
                                performance_monitor.record_operation("proof_generation_cached", start_time.elapsed(), true);
                                continue;
                            }
                            
                            // Generate proof
                            let result = match Self::generate_proof_internal(&proof_request) {
                                Ok(proof) => {
                                    // Cache the result
                                    cache_manager.cache_proof(&cache_key, &proof);
                                    
                                    performance_monitor.record_operation("proof_generation", start_time.elapsed(), true);
                                    ProofGenerationResult::Success {
                                        request: proof_request,
                                        proof,
                                        duration: start_time.elapsed(),
                                        cache_hit: false,
                                    }
                                },
                                Err(e) => {
                                    performance_monitor.record_operation("proof_generation", start_time.elapsed(), false);
                                    ProofGenerationResult::Error {
                                        request: proof_request,
                                        error: e.to_string(),
                                        duration: start_time.elapsed(),
                                    }
                                }
                            };
                            
                            // Store result
                            let mut results_lock = results.lock().unwrap();
                            results_lock[index] = Some(result);
                        },
                        None => {
                            // No more work available
                            debug!("Worker {} finished - no more work", worker_id);
                            break;
                        }
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all workers to complete
        for handle in handles {
            handle.await
                .map_err(|e| LedgerError::ConcurrencyError(format!("Worker thread error: {}", e)))?;
        }
        
        // Extract results
        let results_lock = results.lock().unwrap();
        let final_results: Vec<ProofGenerationResult> = results_lock
            .iter()
            .map(|opt| opt.as_ref().unwrap().clone())
            .collect();
        
        Ok(final_results)
    }
    
    /// Estimate proof complexity for load balancing
    fn estimate_proof_complexity(&self, request: &ProofRequest) -> u64 {
        match request {
            ProofRequest::DatasetIntegrity { dataset_size, .. } => *dataset_size / 1000,
            ProofRequest::Statistics { data_points, .. } => *data_points / 100,
            ProofRequest::Transform { input_size, .. } => *input_size / 500,
            ProofRequest::Custom { complexity_hint, .. } => *complexity_hint,
        }
    }
    
    /// Internal proof generation implementation
    fn generate_proof_internal(request: &ProofRequest) -> Result<GeneratedProof> {
        // Simulate proof generation with complexity-based delay
        let complexity = match request {
            ProofRequest::DatasetIntegrity { dataset_size, .. } => *dataset_size / 10000,
            ProofRequest::Statistics { data_points, .. } => *data_points / 5000,
            ProofRequest::Transform { input_size, .. } => *input_size / 8000,
            ProofRequest::Custom { complexity_hint, .. } => *complexity_hint / 1000,
        };
        
        // Simulate work with sleep (in real implementation, this would be actual ZK proof generation)
        std::thread::sleep(Duration::from_millis(complexity.min(100)));
        
        Ok(GeneratedProof {
            id: Uuid::new_v4(),
            proof_type: request.proof_type().to_string(),
            proof_data: vec![0u8; 288], // Standard proof size
            generated_at: Utc::now(),
        })
    }
    
    /// Streaming processing for very large datasets
    #[instrument(skip(self, data_stream))]
    pub async fn streaming_proof_generation<S>(&self, 
        mut data_stream: S,
        chunk_size: usize
    ) -> Result<Vec<StreamingProofResult>>
    where
        S: futures::Stream<Item = Vec<u8>> + Unpin + Send,
    {
        use futures::StreamExt;
        
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut chunk_count = 0;
        let mut total_bytes = 0;
        
        // Process stream in chunks
        while let Some(chunk) = data_stream.next().await {
            chunk_count += 1;
            total_bytes += chunk.len();
            
            let chunk_start = Instant::now();
            
            // Process chunk in parallel subunits if large enough
            let chunk_result = if chunk.len() > chunk_size * 4 {
                self.process_large_chunk_parallel(chunk, chunk_size).await?
            } else {
                self.process_chunk_sequential(chunk).await?
            };
            
            let chunk_duration = chunk_start.elapsed();
            
            results.push(StreamingProofResult {
                chunk_index: chunk_count,
                chunk_size: chunk_result.processed_bytes,
                processing_time: chunk_duration,
                proof: chunk_result.proof,
                success: chunk_result.success,
                error_message: chunk_result.error_message,
            });
            
            // Adaptive throttling based on performance
            if chunk_duration > Duration::from_millis(1000) {
                warn!("Chunk {} took {:?} to process - may need optimization", chunk_count, chunk_duration);
            }
        }
        
        let total_duration = start_time.elapsed();
        self.performance_monitor.record_streaming_operation(
            "streaming_proof_generation",
            total_bytes,
            chunk_count,
            total_duration
        );
        
        info!("Streaming processing completed: {} chunks, {} bytes in {:?}",
              chunk_count, total_bytes, total_duration);
        
        Ok(results)
    }
    
    /// Process large chunk in parallel subunits
    async fn process_large_chunk_parallel(&self, 
        chunk: Vec<u8>, 
        subunit_size: usize
    ) -> Result<ChunkProcessingResult> {
        let subunits: Vec<Vec<u8>> = chunk.chunks(subunit_size)
            .map(|subunit| subunit.to_vec())
            .collect();
        
        // Process subunits in parallel
        let subunit_results: Vec<_> = self.thread_pool.install(|| {
            subunits.into_par_iter()
                .map(|subunit| self.process_subunit(subunit))
                .collect()
        });
        
        // Combine results
        let mut total_bytes = 0;
        let mut combined_proof_data = Vec::new();
        let mut has_error = false;
        let mut error_message = None;
        
        for result in subunit_results {
            match result {
                Ok(subunit_result) => {
                    total_bytes += subunit_result.len();
                    combined_proof_data.extend(subunit_result);
                },
                Err(e) => {
                    has_error = true;
                    error_message = Some(e.to_string());
                    break;
                }
            }
        }
        
        if has_error {
            return Ok(ChunkProcessingResult {
                processed_bytes: total_bytes,
                proof: None,
                success: false,
                error_message,
            });
        }
        
        // Create combined proof
        let combined_proof = GeneratedProof {
            id: Uuid::new_v4(),
            proof_type: "streaming_chunk".to_string(),
            proof_data: combined_proof_data,
            generated_at: Utc::now(),
        };
        
        Ok(ChunkProcessingResult {
            processed_bytes: total_bytes,
            proof: Some(combined_proof),
            success: true,
            error_message: None,
        })
    }
    
    /// Process chunk sequentially
    async fn process_chunk_sequential(&self, chunk: Vec<u8>) -> Result<ChunkProcessingResult> {
        let chunk_size = chunk.len();
        
        match self.process_subunit(chunk) {
            Ok(proof_data) => {
                let proof = GeneratedProof {
                    id: Uuid::new_v4(),
                    proof_type: "sequential_chunk".to_string(),
                    proof_data,
                    generated_at: Utc::now(),
                };
                
                Ok(ChunkProcessingResult {
                    processed_bytes: chunk_size,
                    proof: Some(proof),
                    success: true,
                    error_message: None,
                })
            },
            Err(e) => Ok(ChunkProcessingResult {
                processed_bytes: chunk_size,
                proof: None,
                success: false,
                error_message: Some(e.to_string()),
            })
        }
    }
    
    /// Process individual subunit (basic proof generation)
    fn process_subunit(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        // Simulate processing time based on data size
        let processing_time = Duration::from_micros((data.len() / 100).max(1) as u64);
        std::thread::sleep(processing_time);
        
        // Generate proof data (simplified)
        let mut proof_data = vec![0u8; 32];
        for (i, &byte) in data.iter().enumerate() {
            if i < 32 {
                proof_data[i] = byte.wrapping_add(i as u8);
            }
        }
        
        Ok(proof_data)
    }
}

/// Performance monitoring and optimization
#[derive(Clone)]
pub struct PerformanceMonitor {
    metrics: Arc<RwLock<PerformanceMetrics>>,
    operation_history: Arc<Mutex<Vec<OperationRecord>>>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
            operation_history: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Record single operation performance
    pub fn record_operation(&self, operation_type: &str, duration: Duration, success: bool) {
        let operation_record = OperationRecord {
            operation_type: operation_type.to_string(),
            duration,
            success,
            timestamp: Utc::now(),
        };
        
        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.update_operation(operation_type, duration, success);
        }
        
        // Store in history
        if let Ok(mut history) = self.operation_history.lock() {
            history.push(operation_record);
            
            // Keep only recent operations
            if history.len() > 10000 {
                history.drain(..5000);
            }
        }
    }
    
    /// Record parallel operation performance
    pub fn record_parallel_operation(&self, 
        operation_type: &str, 
        item_count: usize, 
        total_duration: Duration,
        success: bool
    ) {
        let avg_duration_per_item = total_duration / item_count.max(1) as u32;
        
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.update_parallel_operation(operation_type, item_count, total_duration, success);
        }
        
        info!("Parallel operation {} completed: {} items in {:?} (avg {:?} per item)",
              operation_type, item_count, total_duration, avg_duration_per_item);
    }
    
    /// Record streaming operation performance
    pub fn record_streaming_operation(&self,
        operation_type: &str,
        total_bytes: usize,
        chunk_count: usize,
        total_duration: Duration
    ) {
        let throughput_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / total_duration.as_secs_f64();
        
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.update_streaming_operation(operation_type, total_bytes, chunk_count, total_duration);
        }
        
        info!("Streaming operation {} completed: {} bytes in {} chunks, {:?} duration ({:.2} MB/s)",
              operation_type, total_bytes, chunk_count, total_duration, throughput_mbps);
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> Result<PerformanceStats> {
        let metrics = self.metrics.read()
            .map_err(|_| LedgerError::ConcurrencyError("Failed to read performance metrics".to_string()))?;
        
        let history = self.operation_history.lock()
            .map_err(|_| LedgerError::ConcurrencyError("Failed to read operation history".to_string()))?;
        
        // Calculate recent performance trends
        let recent_operations: Vec<_> = history.iter()
            .filter(|op| op.timestamp > Utc::now() - chrono::Duration::minutes(10))
            .collect();
        
        let recent_success_rate = if recent_operations.is_empty() {
            1.0
        } else {
            recent_operations.iter().filter(|op| op.success).count() as f64 / recent_operations.len() as f64
        };
        
        Ok(PerformanceStats {
            total_operations: metrics.total_operations,
            successful_operations: metrics.successful_operations,
            failed_operations: metrics.failed_operations,
            average_duration_ms: metrics.average_duration.as_millis() as f64,
            recent_success_rate,
            throughput_ops_per_second: metrics.calculate_throughput(),
            operation_breakdown: metrics.operation_stats.clone(),
        })
    }
}

/// Smart caching system for proofs and computations
#[derive(Clone)]
pub struct CacheManager {
    config: CacheConfig,
    proof_cache: Arc<RwLock<HashMap<String, CachedProof>>>,
    computation_cache: Arc<RwLock<HashMap<String, CachedComputation>>>,
    cache_stats: Arc<Mutex<CacheStats>>,
}

impl CacheManager {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            proof_cache: Arc::new(RwLock::new(HashMap::new())),
            computation_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(Mutex::new(CacheStats::new())),
        }
    }
    
    /// Generate cache key for proof request
    pub fn generate_proof_cache_key(&self, request: &ProofRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        request.hash(&mut hasher);
        format!("proof_{:x}", hasher.finish())
    }
    
    /// Cache a generated proof
    pub fn cache_proof(&self, key: &str, proof: &GeneratedProof) {
        if !self.config.enable_proof_caching {
            return;
        }
        
        let cached_proof = CachedProof {
            proof: proof.clone(),
            cached_at: Utc::now(),
            hit_count: 0,
            size_bytes: proof.proof_data.len(),
        };
        
        if let Ok(mut cache) = self.proof_cache.write() {
            // Check cache size limits
            if cache.len() >= self.config.max_cached_proofs {
                self.evict_lru_proofs(&mut cache);
            }
            
            cache.insert(key.to_string(), cached_proof);
            
            // Update stats
            if let Ok(mut stats) = self.cache_stats.lock() {
                stats.total_cached_items += 1;
                stats.total_cache_size_bytes += proof.proof_data.len() as u64;
            }
        }
    }
    
    /// Get cached proof
    pub fn get_cached_proof(&self, key: &str) -> Option<GeneratedProof> {
        if !self.config.enable_proof_caching {
            return None;
        }
        
        if let Ok(mut cache) = self.proof_cache.write() {
            if let Some(cached_proof) = cache.get_mut(key) {
                // Check expiration
                let age = Utc::now().signed_duration_since(cached_proof.cached_at);
                if age.num_seconds() > self.config.cache_ttl_seconds as i64 {
                    cache.remove(key);
                    return None;
                }
                
                // Update hit count and stats
                cached_proof.hit_count += 1;
                if let Ok(mut stats) = self.cache_stats.lock() {
                    stats.cache_hits += 1;
                }
                
                return Some(cached_proof.proof.clone());
            }
        }
        
        // Cache miss
        if let Ok(mut stats) = self.cache_stats.lock() {
            stats.cache_misses += 1;
        }
        
        None
    }
    
    /// Evict least recently used proofs
    fn evict_lru_proofs(&self, cache: &mut HashMap<String, CachedProof>) {
        if cache.is_empty() {
            return;
        }
        
        // Find oldest entries (simple LRU based on cached_at timestamp)
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, cached)| cached.cached_at);
        
        // Remove oldest 25% of entries
        let remove_count = (cache.len() / 4).max(1);
        let keys_to_remove: Vec<String> = entries.iter()
            .take(remove_count)
            .map(|(key, _)| (*key).clone())
            .collect();
        
        for key in keys_to_remove {
            if let Some(cached_proof) = cache.remove(&key) {
                if let Ok(mut stats) = self.cache_stats.lock() {
                    stats.total_cached_items -= 1;
                    stats.total_cache_size_bytes -= cached_proof.size_bytes as u64;
                    stats.evicted_items += 1;
                }
            }
        }
        
        debug!("Evicted {} items from proof cache", remove_count);
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache_stats.lock()
            .map(|stats| stats.clone())
            .unwrap_or_else(|_| CacheStats::new())
    }
}

// Configuration and data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelProcessingConfig {
    pub cpu_threads: usize,
    pub max_concurrent_operations: usize,
    pub max_batch_size: usize,
    pub enable_work_stealing: bool,
    pub cache_config: CacheConfig,
    pub memory_limit_mb: usize,
}

impl Default for ParallelProcessingConfig {
    fn default() -> Self {
        Self {
            cpu_threads: num_cpus::get().max(4),
            max_concurrent_operations: num_cpus::get() * 8,
            max_batch_size: 1000,
            enable_work_stealing: true,
            cache_config: CacheConfig::default(),
            memory_limit_mb: 2048, // 2GB default
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enable_proof_caching: bool,
    pub max_cached_proofs: usize,
    pub cache_ttl_seconds: u64,
    pub max_cache_size_mb: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_proof_caching: true,
            max_cached_proofs: 10000,
            cache_ttl_seconds: 3600, // 1 hour
            max_cache_size_mb: 500, // 500MB
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProcessingResult<T> {
    Success {
        input: T,
        output: T,
        duration: Duration,
    },
    Error {
        input: T,
        error: String,
        duration: Duration,
    },
}

#[derive(Debug, Clone, Hash)]
pub enum ProofRequest {
    DatasetIntegrity {
        dataset_id: String,
        dataset_size: u64,
    },
    Statistics {
        dataset_id: String,
        data_points: u64,
        properties: Vec<String>,
    },
    Transform {
        input_id: String,
        output_id: String,
        input_size: u64,
        operation: String,
    },
    Custom {
        operation_type: String,
        complexity_hint: u64,
        parameters: HashMap<String, String>,
    },
}

impl ProofRequest {
    pub fn proof_type(&self) -> &str {
        match self {
            ProofRequest::DatasetIntegrity { .. } => "dataset_integrity",
            ProofRequest::Statistics { .. } => "statistics",
            ProofRequest::Transform { .. } => "transform",
            ProofRequest::Custom { operation_type, .. } => operation_type,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProofGenerationResult {
    Success {
        request: ProofRequest,
        proof: GeneratedProof,
        duration: Duration,
        cache_hit: bool,
    },
    Error {
        request: ProofRequest,
        error: String,
        duration: Duration,
    },
}

#[derive(Debug, Clone)]
pub struct GeneratedProof {
    pub id: Uuid,
    pub proof_type: String,
    pub proof_data: Vec<u8>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct StreamingProofResult {
    pub chunk_index: usize,
    pub chunk_size: usize,
    pub processing_time: Duration,
    pub proof: Option<GeneratedProof>,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChunkProcessingResult {
    pub processed_bytes: usize,
    pub proof: Option<GeneratedProof>,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug)]
struct PerformanceMetrics {
    total_operations: u64,
    successful_operations: u64,
    failed_operations: u64,
    total_duration: Duration,
    average_duration: Duration,
    operation_stats: HashMap<String, OperationStats>,
    start_time: Instant,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration: Duration::default(),
            average_duration: Duration::default(),
            operation_stats: HashMap::new(),
            start_time: Instant::now(),
        }
    }
    
    fn update_operation(&mut self, operation_type: &str, duration: Duration, success: bool) {
        self.total_operations += 1;
        self.total_duration += duration;
        self.average_duration = self.total_duration / self.total_operations.max(1) as u32;
        
        if success {
            self.successful_operations += 1;
        } else {
            self.failed_operations += 1;
        }
        
        // Update operation-specific stats
        let op_stats = self.operation_stats.entry(operation_type.to_string())
            .or_insert_with(OperationStats::new);
        op_stats.update(duration, success);
    }
    
    fn update_parallel_operation(&mut self, 
        operation_type: &str,
        _item_count: usize,
        total_duration: Duration,
        success: bool
    ) {
        // For parallel operations, treat as single operation with total duration
        self.update_operation(operation_type, total_duration, success);
    }
    
    fn update_streaming_operation(&mut self,
        operation_type: &str,
        _total_bytes: usize,
        _chunk_count: usize,
        total_duration: Duration
    ) {
        // For streaming operations, always consider successful if completed
        self.update_operation(operation_type, total_duration, true);
    }
    
    fn calculate_throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed();
        if elapsed.as_secs() == 0 {
            return 0.0;
        }
        self.total_operations as f64 / elapsed.as_secs() as f64
    }
}

#[derive(Debug, Clone)]
pub struct OperationStats {
    pub count: u64,
    pub successful_count: u64,
    pub failed_count: u64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
}

impl OperationStats {
    fn new() -> Self {
        Self {
            count: 0,
            successful_count: 0,
            failed_count: 0,
            total_duration: Duration::default(),
            average_duration: Duration::default(),
            min_duration: Duration::MAX,
            max_duration: Duration::default(),
        }
    }
    
    fn update(&mut self, duration: Duration, success: bool) {
        self.count += 1;
        self.total_duration += duration;
        self.average_duration = self.total_duration / self.count.max(1) as u32;
        
        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }
        
        if success {
            self.successful_count += 1;
        } else {
            self.failed_count += 1;
        }
    }
}

#[derive(Debug, Clone)]
pub struct OperationRecord {
    pub operation_type: String,
    pub duration: Duration,
    pub success: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_duration_ms: f64,
    pub recent_success_rate: f64,
    pub throughput_ops_per_second: f64,
    pub operation_breakdown: HashMap<String, OperationStats>,
}

#[derive(Debug, Clone)]
pub struct CachedProof {
    pub proof: GeneratedProof,
    pub cached_at: DateTime<Utc>,
    pub hit_count: u64,
    pub size_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct CachedComputation {
    pub result: Vec<u8>,
    pub cached_at: DateTime<Utc>,
    pub hit_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_cached_items: u64,
    pub evicted_items: u64,
    pub total_cache_size_bytes: u64,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            total_cached_items: 0,
            evicted_items: 0,
            total_cache_size_bytes: 0,
        }
    }
    
    pub fn hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total_requests as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use futures::stream;
    
    #[tokio::test]
    async fn test_parallel_processor() {
        let config = ParallelProcessingConfig::default();
        let processor = ParallelZKProcessor::new(config).unwrap();
        
        // Test parallel dataset processing
        let test_datasets = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let results = processor.parallel_dataset_processing(
            test_datasets,
            |x| Ok(x * 2)
        ).await.unwrap();
        
        assert_eq!(results.len(), 8);
        for result in results {
            match result {
                ProcessingResult::Success { input, output, .. } => {
                    assert_eq!(output, input * 2);
                },
                ProcessingResult::Error { .. } => panic!("Unexpected error result"),
            }
        }
    }
    
    #[tokio::test]
    async fn test_proof_generation() {
        let config = ParallelProcessingConfig::default();
        let processor = ParallelZKProcessor::new(config).unwrap();
        
        let proof_requests = vec![
            ProofRequest::DatasetIntegrity {
                dataset_id: "test1".to_string(),
                dataset_size: 1000,
            },
            ProofRequest::Statistics {
                dataset_id: "test2".to_string(),
                data_points: 500,
                properties: vec!["mean".to_string(), "variance".to_string()],
            },
        ];
        
        let results = processor.parallel_proof_generation(proof_requests).await.unwrap();
        
        assert_eq!(results.len(), 2);
        for result in results {
            match result {
                ProofGenerationResult::Success { proof, .. } => {
                    assert!(!proof.proof_data.is_empty());
                },
                ProofGenerationResult::Error { .. } => panic!("Unexpected error result"),
            }
        }
    }
    
    #[tokio::test]
    async fn test_streaming_processing() {
        let config = ParallelProcessingConfig::default();
        let processor = ParallelZKProcessor::new(config).unwrap();
        
        // Create test stream
        let data_chunks = vec![
            vec![1u8; 100],
            vec![2u8; 200],
            vec![3u8; 150],
        ];
        let test_stream = stream::iter(data_chunks.into_iter());
        
        let results = processor.streaming_proof_generation(test_stream, 50).await.unwrap();
        
        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.chunk_index, i + 1);
            assert!(result.success);
            assert!(result.proof.is_some());
        }
    }
    
    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        // Record some operations
        monitor.record_operation("test_op", Duration::from_millis(100), true);
        monitor.record_operation("test_op", Duration::from_millis(200), true);
        monitor.record_operation("test_op", Duration::from_millis(150), false);
        
        let stats = monitor.get_performance_stats().unwrap();
        assert_eq!(stats.total_operations, 3);
        assert_eq!(stats.successful_operations, 2);
        assert_eq!(stats.failed_operations, 1);
    }
    
    #[test]
    fn test_cache_manager() {
        let config = CacheConfig::default();
        let cache_manager = CacheManager::new(config);
        
        // Test proof caching
        let proof_request = ProofRequest::DatasetIntegrity {
            dataset_id: "test".to_string(),
            dataset_size: 1000,
        };
        
        let cache_key = cache_manager.generate_proof_cache_key(&proof_request);
        
        // Initially no cached proof
        assert!(cache_manager.get_cached_proof(&cache_key).is_none());
        
        // Cache a proof
        let proof = GeneratedProof {
            id: Uuid::new_v4(),
            proof_type: "test".to_string(),
            proof_data: vec![1, 2, 3, 4],
            generated_at: Utc::now(),
        };
        
        cache_manager.cache_proof(&cache_key, &proof);
        
        // Should now be cached
        let cached_proof = cache_manager.get_cached_proof(&cache_key);
        assert!(cached_proof.is_some());
        assert_eq!(cached_proof.unwrap().proof_data, proof.proof_data);
        
        // Check cache stats
        let stats = cache_manager.get_cache_stats();
        assert_eq!(stats.cache_hits, 1);
    }
}