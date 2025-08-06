//! Performance optimization and scaling features for ZKP Dataset Ledger

use crate::{LedgerError, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use futures::stream::{FuturesUnordered, StreamExt};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Maximum number of concurrent operations
    pub max_concurrent_operations: usize,
    /// Cache size in MB
    pub cache_size_mb: usize,
    /// Connection pool size
    pub connection_pool_size: usize,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Enable operation caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Enable async I/O
    pub enable_async_io: bool,
    /// Enable streaming for large datasets
    pub enable_streaming: bool,
    /// Stream chunk size in bytes
    pub stream_chunk_size: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel_processing: true,
            max_concurrent_operations: num_cpus::get() * 2,
            cache_size_mb: 256,
            connection_pool_size: 10,
            batch_size: 1000,
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            enable_async_io: true,
            enable_streaming: true,
            stream_chunk_size: 1024 * 1024, // 1MB
        }
    }
}

/// Cached item with TTL
#[derive(Debug, Clone)]
struct CacheItem<T> {
    value: T,
    created_at: Instant,
    ttl: Duration,
}

impl<T> CacheItem<T> {
    fn new(value: T, ttl: Duration) -> Self {
        Self {
            value,
            created_at: Instant::now(),
            ttl,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// High-performance cache with TTL and LRU eviction
pub struct PerformanceCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    cache: Arc<RwLock<HashMap<K, CacheItem<V>>>>,
    access_order: Arc<Mutex<VecDeque<K>>>,
    max_size: usize,
    default_ttl: Duration,
}

impl<K, V> PerformanceCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
            default_ttl,
        }
    }

    pub async fn get(&self, key: &K) -> Option<V> {
        // Check if item exists and is not expired
        {
            let cache = self.cache.read().unwrap();
            if let Some(item) = cache.get(key) {
                if !item.is_expired() {
                    // Update access order
                    let mut access_order = self.access_order.lock().unwrap();
                    access_order.retain(|k| k != key);
                    access_order.push_back(key.clone());
                    return Some(item.value.clone());
                }
            }
        }

        // Remove expired item
        self.remove(key).await;
        None
    }

    pub async fn put(&self, key: K, value: V) {
        self.put_with_ttl(key, value, self.default_ttl).await;
    }

    pub async fn put_with_ttl(&self, key: K, value: V, ttl: Duration) {
        let item = CacheItem::new(value, ttl);
        
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key.clone(), item);
        }

        // Update access order
        {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.retain(|k| k != &key);
            access_order.push_back(key.clone());
        }

        // Evict if over capacity
        self.evict_if_needed().await;
    }

    pub async fn remove(&self, key: &K) -> Option<V> {
        let removed = {
            let mut cache = self.cache.write().unwrap();
            cache.remove(key).map(|item| item.value)
        };

        if removed.is_some() {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.retain(|k| k != key);
        }

        removed
    }

    pub async fn clear(&self) {
        {
            let mut cache = self.cache.write().unwrap();
            cache.clear();
        }
        {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.clear();
        }
    }

    pub fn size(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.len()
    }

    async fn evict_if_needed(&self) {
        let current_size = self.size();
        if current_size <= self.max_size {
            return;
        }

        // Evict LRU items
        let to_evict = current_size - self.max_size;
        let keys_to_remove: Vec<K> = {
            let mut access_order = self.access_order.lock().unwrap();
            (0..to_evict).filter_map(|_| access_order.pop_front()).collect()
        };

        {
            let mut cache = self.cache.write().unwrap();
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }
}

/// Parallel processing manager
pub struct ParallelProcessor {
    semaphore: Arc<Semaphore>,
    config: PerformanceConfig,
}

impl ParallelProcessor {
    pub fn new(config: PerformanceConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_operations));
        Self { semaphore, config }
    }

    /// Process tasks in parallel with concurrency control
    pub async fn process_parallel<T, R, F, Fut>(&self, tasks: Vec<T>, processor: F) -> Result<Vec<R>>
    where
        F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
        T: Send + 'static,
        R: Send + 'static,
    {
        if !self.config.enable_parallel_processing {
            // Sequential processing fallback
            let mut results = Vec::new();
            for task in tasks {
                let result = processor(task).await?;
                results.push(result);
            }
            return Ok(results);
        }

        let semaphore = self.semaphore.clone();
        let mut futures = FuturesUnordered::new();

        for task in tasks {
            let processor = processor.clone();
            let semaphore = semaphore.clone();
            
            let future = async move {
                let _permit = semaphore.acquire().await
                    .map_err(|_| LedgerError::InvalidInput("Failed to acquire semaphore".to_string()))?;
                processor(task).await
            };
            
            futures.push(future);
        }

        let mut results = Vec::new();
        while let Some(result) = futures.next().await {
            results.push(result?);
        }

        Ok(results)
    }

    /// Process tasks in batches
    pub async fn process_batched<T, R, F, Fut>(&self, tasks: Vec<T>, processor: F) -> Result<Vec<R>>
    where
        F: Fn(Vec<T>) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Vec<R>>> + Send + 'static,
        T: Send + 'static,
        R: Send + 'static,
    {
        let mut all_results = Vec::new();
        
        for batch in tasks.chunks(self.config.batch_size) {
            let batch_results = processor(batch.to_vec()).await?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }
}

/// Connection pool for managing database/storage connections
pub struct ConnectionPool<T>
where
    T: Send + Sync,
{
    connections: Arc<Mutex<Vec<T>>>,
    max_size: usize,
    current_size: Arc<Mutex<usize>>,
    factory: Arc<dyn Fn() -> Result<T> + Send + Sync>,
}

impl<T> ConnectionPool<T>
where
    T: Send + Sync,
{
    pub fn new<F>(max_size: usize, factory: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        Self {
            connections: Arc::new(Mutex::new(Vec::new())),
            max_size,
            current_size: Arc::new(Mutex::new(0)),
            factory: Arc::new(factory),
        }
    }

    pub async fn get_connection(&self) -> Result<PooledConnection<T>> {
        // Try to get an existing connection
        {
            let mut connections = self.connections.lock().unwrap();
            if let Some(conn) = connections.pop() {
                return Ok(PooledConnection::new(conn, self.connections.clone()));
            }
        }

        // Create new connection if under limit
        {
            let mut current_size = self.current_size.lock().unwrap();
            if *current_size < self.max_size {
                let conn = (self.factory)()?;
                *current_size += 1;
                return Ok(PooledConnection::new(conn, self.connections.clone()));
            }
        }

        // Wait for connection to become available
        // In a real implementation, we'd use a proper waiting mechanism
        tokio::time::sleep(Duration::from_millis(10)).await;
        self.get_connection().await
    }

    pub fn stats(&self) -> ConnectionPoolStats {
        let connections = self.connections.lock().unwrap();
        let current_size = self.current_size.lock().unwrap();
        
        ConnectionPoolStats {
            total_connections: *current_size,
            available_connections: connections.len(),
            max_connections: self.max_size,
        }
    }
}

/// Pooled connection wrapper
pub struct PooledConnection<T>
where
    T: Send + Sync,
{
    connection: Option<T>,
    pool: Arc<Mutex<Vec<T>>>,
}

impl<T> PooledConnection<T>
where
    T: Send + Sync,
{
    fn new(connection: T, pool: Arc<Mutex<Vec<T>>>) -> Self {
        Self {
            connection: Some(connection),
            pool,
        }
    }

    pub fn as_ref(&self) -> Option<&T> {
        self.connection.as_ref()
    }

    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.connection.as_mut()
    }
}

impl<T> Drop for PooledConnection<T>
where
    T: Send + Sync,
{
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            let mut pool = self.pool.lock().unwrap();
            pool.push(conn);
        }
    }
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    pub total_connections: usize,
    pub available_connections: usize,
    pub max_connections: usize,
}

/// Streaming processor for large datasets
pub struct StreamProcessor {
    chunk_size: usize,
}

impl StreamProcessor {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Process data in streaming chunks
    pub async fn process_stream<T, R, F, Fut>(
        &self,
        mut data: Vec<u8>,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(Vec<u8>) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<R>> + Send,
        R: Send,
    {
        let mut results = Vec::new();
        
        while !data.is_empty() {
            let chunk_size = std::cmp::min(self.chunk_size, data.len());
            let chunk = data.drain(..chunk_size).collect();
            
            let result = processor(chunk).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Process file in streaming chunks
    pub async fn process_file_stream<R, F, Fut>(
        &self,
        file_path: &std::path::Path,
        processor: F,
    ) -> Result<Vec<R>>
    where
        F: Fn(Vec<u8>) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<R>> + Send,
        R: Send,
    {
        use tokio::io::{AsyncReadExt, BufReader};
        use tokio::fs::File;

        let file = File::open(file_path).await?;
        let mut reader = BufReader::new(file);
        let mut results = Vec::new();
        let mut buffer = vec![0u8; self.chunk_size];

        loop {
            let bytes_read = reader.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }

            let chunk = buffer[..bytes_read].to_vec();
            let result = processor(chunk).await?;
            results.push(result);
        }

        Ok(results)
    }
}

/// Performance optimizer that applies various optimization techniques
pub struct PerformanceOptimizer {
    config: PerformanceConfig,
    cache: PerformanceCache<String, Vec<u8>>,
    parallel_processor: ParallelProcessor,
    stream_processor: StreamProcessor,
}

impl PerformanceOptimizer {
    pub fn new(config: PerformanceConfig) -> Self {
        let cache_ttl = Duration::from_secs(config.cache_ttl_seconds);
        let cache_size = config.cache_size_mb * 1024 * 1024 / 100; // Rough estimate
        
        Self {
            cache: PerformanceCache::new(cache_size, cache_ttl),
            parallel_processor: ParallelProcessor::new(config.clone()),
            stream_processor: StreamProcessor::new(config.stream_chunk_size),
            config,
        }
    }

    /// Optimized dataset processing with caching and parallelization
    pub async fn process_dataset_optimized(
        &self,
        dataset_path: &str,
        operation: &str,
    ) -> Result<Vec<u8>> {
        let cache_key = format!("{}:{}", dataset_path, operation);

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_result) = self.cache.get(&cache_key).await {
                tracing::debug!(key = %cache_key, "Cache hit for dataset processing");
                return Ok(cached_result);
            }
        }

        // Process with streaming if file is large
        let file_size = std::fs::metadata(dataset_path)?.len();
        let result = if file_size > (self.config.stream_chunk_size * 10) as u64 {
            tracing::info!("Using streaming processing for large dataset");
            self.process_with_streaming(dataset_path, operation).await?
        } else {
            self.process_with_parallel(dataset_path, operation).await?
        };

        // Cache the result
        if self.config.enable_caching {
            self.cache.put(cache_key, result.clone()).await;
        }

        Ok(result)
    }

    async fn process_with_streaming(&self, dataset_path: &str, _operation: &str) -> Result<Vec<u8>> {
        let path = std::path::Path::new(dataset_path);
        
        let results = self.stream_processor
            .process_file_stream(path, |chunk| async move {
                // Simulate processing - in real implementation, this would be actual operation
                tokio::time::sleep(Duration::from_millis(1)).await;
                Ok(chunk.len())
            })
            .await?;

        // Combine results
        let total_bytes: usize = results.iter().sum();
        Ok(total_bytes.to_string().into_bytes())
    }

    async fn process_with_parallel(&self, dataset_path: &str, _operation: &str) -> Result<Vec<u8>> {
        let data = tokio::fs::read(dataset_path).await?;
        
        // Split data into chunks for parallel processing
        let chunks: Vec<Vec<u8>> = data.chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let results = self.parallel_processor
            .process_parallel(chunks, |chunk| async move {
                // Simulate processing
                tokio::time::sleep(Duration::from_millis(1)).await;
                Ok(chunk.len())
            })
            .await?;

        let total_bytes: usize = results.iter().sum();
        Ok(total_bytes.to_string().into_bytes())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.size(),
            max_size: self.config.cache_size_mb * 1024 * 1024 / 100,
            hit_rate: 0.0, // Would need to track hits/misses for real implementation
        }
    }

    /// Clear all caches
    pub async fn clear_caches(&self) {
        self.cache.clear().await;
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hit_rate: f64,
}

/// Load balancer for distributing work across multiple instances
pub struct LoadBalancer<T>
where
    T: Send + Sync + Clone,
{
    instances: Arc<RwLock<Vec<T>>>,
    current_index: Arc<Mutex<usize>>,
    strategy: LoadBalancingStrategy,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    Random,
    LeastConnections,
}

impl<T> LoadBalancer<T>
where
    T: Send + Sync + Clone,
{
    pub fn new(instances: Vec<T>, strategy: LoadBalancingStrategy) -> Self {
        Self {
            instances: Arc::new(RwLock::new(instances)),
            current_index: Arc::new(Mutex::new(0)),
            strategy,
        }
    }

    pub fn get_instance(&self) -> Option<T> {
        let instances = self.instances.read().unwrap();
        if instances.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut index = self.current_index.lock().unwrap();
                let instance = instances[*index].clone();
                *index = (*index + 1) % instances.len();
                Some(instance)
            }
            LoadBalancingStrategy::Random => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let index = rng.gen_range(0..instances.len());
                Some(instances[index].clone())
            }
            LoadBalancingStrategy::LeastConnections => {
                // Simplified - in real implementation, track connection counts
                Some(instances[0].clone())
            }
        }
    }

    pub fn add_instance(&self, instance: T) {
        let mut instances = self.instances.write().unwrap();
        instances.push(instance);
    }

    pub fn remove_instance(&self, index: usize) -> Option<T> {
        let mut instances = self.instances.write().unwrap();
        if index < instances.len() {
            Some(instances.remove(index))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.enable_parallel_processing);
        assert!(config.enable_caching);
        assert_eq!(config.batch_size, 1000);
    }

    #[tokio::test]
    async fn test_performance_cache() {
        let cache = PerformanceCache::new(2, Duration::from_secs(1));
        
        cache.put("key1".to_string(), vec![1, 2, 3]).await;
        let value = cache.get(&"key1".to_string()).await;
        assert_eq!(value, Some(vec![1, 2, 3]));
        
        // Test TTL expiration
        tokio::time::sleep(Duration::from_secs(2)).await;
        let expired_value = cache.get(&"key1".to_string()).await;
        assert_eq!(expired_value, None);
    }

    #[tokio::test]
    async fn test_parallel_processor() {
        let config = PerformanceConfig::default();
        let processor = ParallelProcessor::new(config);
        
        let tasks = vec![1, 2, 3, 4, 5];
        let results = processor.process_parallel(tasks, |x| async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(x * 2)
        }).await.unwrap();
        
        assert_eq!(results.len(), 5);
        assert!(results.contains(&2));
        assert!(results.contains(&10));
    }

    #[tokio::test]
    async fn test_stream_processor() {
        let processor = StreamProcessor::new(1024);
        let data = vec![1u8; 2048]; // 2KB of data
        
        let results = processor.process_stream(data, |chunk| async move {
            Ok(chunk.len())
        }).await.unwrap();
        
        assert_eq!(results.len(), 2); // Two chunks
        assert_eq!(results[0], 1024);
        assert_eq!(results[1], 1024);
    }

    #[test]
    fn test_load_balancer() {
        let instances = vec!["instance1", "instance2", "instance3"];
        let lb = LoadBalancer::new(instances, LoadBalancingStrategy::RoundRobin);
        
        assert_eq!(lb.get_instance(), Some("instance1"));
        assert_eq!(lb.get_instance(), Some("instance2"));
        assert_eq!(lb.get_instance(), Some("instance3"));
        assert_eq!(lb.get_instance(), Some("instance1")); // Round robin
    }

    #[tokio::test]
    async fn test_performance_optimizer() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "test data").unwrap();
        
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);
        
        let result = optimizer.process_dataset_optimized(
            temp_file.path().to_str().unwrap(),
            "test_operation"
        ).await.unwrap();
        
        assert!(!result.is_empty());
    }
}